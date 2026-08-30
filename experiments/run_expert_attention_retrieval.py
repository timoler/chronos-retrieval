#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_expert_attention_retrieval.py

第一版：Expert Text -> Embedding -> Query/Expert Attention -> Retrieval -> Memory

默认先跑 Solar VAL；不访问 TEST。
不使用 hour/doy/level/change 等人工专家特征。
不实现最终 lambda Gate；只输出 Memory，并对固定 lambda 做 VAL 融合扫描。

依赖:
    pip install torch sentence-transformers pandas numpy

示例:
    python run_expert_attention_retrieval.py \
        --dataset solar \
        --root results/multidataset \
        --epochs 20 \
        --topk 5

说明:
- Agent 审核：本脚本读取 expert_knowledge.txt。正式实验时，应把该文件替换为
  Agent 审核通过后的文本；模型本身不依赖固定领域特征。
- 文本编码器默认 all-MiniLM-L6-v2，冻结。
- Query 编码器从原始 512 点时间序列直接学习表示。
- Attention 中，Query token 与 Expert token 联合进入 self-attention；
  取更新后的 Query token 做检索。
- 检索训练只使用 TRAIN 内部 future 监督；VAL 只评估。
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def zscore_window(x, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd


class QueryEncoder(nn.Module):
    """原始时间序列 -> d_model。没有人工时间/状态特征。"""
    def __init__(self, context, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = zscore_window(x).unsqueeze(1)
        return self.norm(self.net(x).squeeze(-1))


class ExpertAttentionRetriever(nn.Module):
    """
    [Query token, Expert token] -> Self-Attention -> conditioned query embedding
    """
    def __init__(self, context, text_dim, d_model=128, heads=4, dropout=0.1):
        super().__init__()
        self.query_encoder = QueryEncoder(context, d_model)
        self.expert_proj = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, expert_embedding):
        q = self.query_encoder(x)                         # [B,D]
        e = self.expert_proj(expert_embedding)            # [B,D]
        tokens = torch.stack([q, e], dim=1)               # [B,2,D]
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)
        q_conditioned = self.out_norm(tokens[:, 0])
        return F.normalize(q_conditioned, dim=-1)


def build_windows(series, starts, context, horizon):
    starts = np.asarray(starts, dtype=np.int64)
    X = np.stack([series[s:s+context] for s in starts]).astype("float32")
    Y = np.stack([series[s+context:s+context+horizon] for s in starts]).astype("float32")
    return X, Y


def batch_embed(model, X, expert_vec, device, batch=512):
    outs = []
    model.eval()
    with torch.no_grad():
        ev = torch.tensor(expert_vec, dtype=torch.float32, device=device)
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
            eb = ev.unsqueeze(0).expand(len(xb), -1)
            outs.append(model(xb, eb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def retrieve_memory(query_z, bank_z, bank_future, topk=5, chunk=256):
    """余弦相似度 Top-K；future 简单均值。"""
    preds = []
    for i in range(0, len(query_z), chunk):
        q = torch.tensor(query_z[i:i+chunk], dtype=torch.float32)
        b = torch.tensor(bank_z, dtype=torch.float32)
        sim = q @ b.T
        idx = torch.topk(sim, k=min(topk, len(bank_z)), dim=1).indices.numpy()
        preds.append(bank_future[idx].mean(axis=1))
    return np.concatenate(preds, axis=0)


def train_retriever(model, X_bank, Y_bank, X_train, Y_train, expert_vec,
                    device, epochs=20, batch_size=64, candidates=256, lr=1e-3):
    """
    Future-supervised retrieval training:
    对每个 TRAIN query，随机采候选历史窗口；
    conditioned embedding 的 softmax 相似度加权候选 future；
    最小化预测 future MSE。
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ev = torch.tensor(expert_vec, dtype=torch.float32, device=device)

    Xb = torch.tensor(X_bank, dtype=torch.float32, device=device)
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)

    n = len(X_train)
    rng = np.random.default_rng(12345)

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []

        # bank embedding 每 epoch 重算，保留梯度会太重；作为当前 epoch 的检索锚点
        with torch.no_grad():
            bz_parts = []
            for j in range(0, len(X_bank), 512):
                xb0 = Xb[j:j+512]
                eb0 = ev.unsqueeze(0).expand(len(xb0), -1)
                bz_parts.append(model(xb0, eb0))
            bank_z = torch.cat(bz_parts, dim=0).detach()

        for p in range(0, n, batch_size):
            ids = order[p:p+batch_size]
            if len(ids) == 0:
                continue

            x = torch.tensor(X_train[ids], dtype=torch.float32, device=device)
            y = torch.tensor(Y_train[ids], dtype=torch.float32, device=device)
            e = ev.unsqueeze(0).expand(len(x), -1)
            qz = model(x, e)

            cand = rng.integers(0, len(X_bank), size=(len(x), candidates))
            cand_t = torch.tensor(cand, dtype=torch.long, device=device)

            cz = bank_z[cand_t]          # [B,C,D]
            cy = Yb[cand_t]              # [B,C,H]
            logits = torch.einsum("bd,bcd->bc", qz, cz) / 0.07
            w = torch.softmax(logits, dim=1)
            pred = torch.einsum("bc,bch->bh", w, cy)

            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        print(f"epoch {ep:02d}/{epochs} loss={np.mean(losses):.6f}")


def load_chronos_optional(path, expected_n):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Chronos prediction 不存在: {p}")
        return None
    z = np.load(p)
    for key in ["chronos_val_pred", "val_chronos_pred", "chronos_pred"]:
        if key in z:
            arr = z[key]
            if len(arr) != expected_n:
                raise ValueError(f"Chronos VAL 数量 {len(arr)} != 当前 VAL {expected_n}")
            return arr
    raise KeyError(f"{p} 找不到 Chronos VAL prediction 字段；现有 {z.files}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--root", type=Path, default=Path("results/multidataset"))
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-bank", type=int, default=25000)
    ap.add_argument("--max-train-query", type=int, default=6000)
    ap.add_argument("--chronos-val", type=str, default=None,
                    help="可选：包含 chronos_val_pred 的 npz")
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = args.root / args.dataset
    series_path = ds / "series.csv"
    index_path = ds / "window_indices.npz"
    expert_path = ds / "expert_knowledge.txt"

    if not (series_path.exists() and index_path.exists() and expert_path.exists()):
        raise FileNotFoundError(f"{ds} 数据准备不完整")

    df = pd.read_csv(series_path)
    series = df["OT"].to_numpy(dtype="float32")
    idx = np.load(index_path)
    train_starts = idx["train"]
    val_starts = idx["val"]

    # TEST 故意不读取/不评估
    print(f"{args.dataset}: train windows={len(train_starts)}, val windows={len(val_starts)}")

    # TRAIN 内再按时间切成 bank 与 retrieval-training query，避免 query future 自检索
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    if len(bank_starts) > args.max_bank:
        take = np.linspace(0, len(bank_starts)-1, args.max_bank).astype(int)
        bank_starts = bank_starts[take]
    if len(fit_starts) > args.max_train_query:
        take = np.linspace(0, len(fit_starts)-1, args.max_train_query).astype(int)
        fit_starts = fit_starts[take]

    X_bank, Y_bank = build_windows(series, bank_starts, args.context, args.horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, args.context, args.horizon)
    X_val, Y_val = build_windows(series, val_starts, args.context, args.horizon)

    expert_text = expert_path.read_text(encoding="utf-8").strip()
    if not expert_text:
        raise ValueError("expert_knowledge.txt 为空")

    print("\nExpert text after review/input:")
    print(expert_text)

    print("\nLoading frozen text encoder:", args.text_model)
    text_encoder = SentenceTransformer(args.text_model, device=str(device))
    expert_vec = text_encoder.encode(
        [expert_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32")
    text_dim = expert_vec.shape[0]
    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = ExpertAttentionRetriever(
        context=args.context,
        text_dim=text_dim,
        d_model=args.d_model,
        heads=args.heads,
    ).to(device)

    print("\nTraining Expert-Attention Retriever...")
    train_retriever(
        model, X_bank, Y_bank, X_fit, Y_fit, expert_vec,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        candidates=args.candidates,
        lr=args.lr,
    )

    print("\nEmbedding bank/VAL...")
    bank_z = batch_embed(model, X_bank, expert_vec, device)
    val_z = batch_embed(model, X_val, expert_vec, device)

    print("Retrieving Top-K...")
    mem_pred = retrieve_memory(val_z, bank_z, Y_bank, topk=args.topk)

    mem_mse = mse(Y_val, mem_pred)
    mem_mae = mae(Y_val, mem_pred)
    print(f"\nExpert-Attention Memory MSE={mem_mse:.6f} MAE={mem_mae:.6f}")

    out = args.root / args.dataset / "expert_attention"
    out.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "expert_text": expert_text,
    }, out / "model.pt")

    np.savez_compressed(
        out / "val_predictions.npz",
        y_true=Y_val,
        expert_attention_memory_pred=mem_pred,
        val_starts=val_starts,
    )

    result = {
        "dataset": args.dataset,
        "expert_text": expert_text,
        "memory_mse": mem_mse,
        "memory_mae": mem_mae,
        "topk": args.topk,
        "seed": args.seed,
        "test_used": False,
    }

    # 如果已有完全同一 VAL 的 Chronos prediction，才做固定 lambda 融合
    chronos = load_chronos_optional(args.chronos_val, len(Y_val))
    if chronos is not None:
        if chronos.shape != Y_val.shape:
            raise ValueError(f"Chronos shape {chronos.shape} != Y_val {Y_val.shape}")

        lambdas = [0.0, .05, .1, .2, .3, .4, .5, .6]
        fusion = []
        for lam in lambdas:
            pred = (1-lam)*chronos + lam*mem_pred
            fusion.append({
                "lambda": lam,
                "mse": mse(Y_val, pred),
                "mae": mae(Y_val, pred),
            })
        result["chronos_mse"] = mse(Y_val, chronos)
        result["fusion"] = fusion
        print("\nFixed-lambda fusion:")
        for r in fusion:
            print(f"lambda={r['lambda']:.2f} MSE={r['mse']:.6f}")

    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nSaved:", out)
    print("TEST was not evaluated.")


if __name__ == "__main__":
    main()
