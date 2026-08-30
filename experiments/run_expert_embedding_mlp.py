#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_expert_embedding_mlp.py

Expert-Embedding + MLP Retrieval（去掉手工 6 维专家特征，保留 V3 的 MLP 检索思想）。

流程：
    expert_knowledge.txt
        -> 真实 LLM Critic 审核 -> 只保留 ACCEPT
        -> SentenceTransformer embedding -> mean-pool -> 单个 Expert Embedding
        -> 与 Query/Candidate 的时间序列表示结合 -> MLP -> expert score
        -> 与 Shape score 相加 -> Top-K=5 -> mean-anchor -> Memory Forecast

关键设计（相对 V3 的唯一改动：专家表示）：
    V3 的 z_q/z_i 是手工 6 维特征（hour_sin/hour_cos/doy_sin/doy_cos/recent_level/recent_change）。
    这里 z_q/z_i 由「可学习的 1D-CNN 时间序列编码 + 冻结的 Expert Embedding」自动得到：

        z = cond( cat([ CNN(x), expert_emb ]) )

    保留 V3 的 pairwise MLP：

        delta(q,i) = MLP([z_q, z_i, |z_q - z_i|])
        score(q,i) = gamma * delta(q,i) + beta * shape(q,i)

不使用 Attention；不重新设计人工统计特征。

数据：results/multidataset/ 的统一划分（Solar stride=1，VAL 9905），不碰 TEST。
shape(q,i) = 线性去趋势残差的 Pearson（与 219891.45 基线同定义）。
评估聚合 = Top-K 均值（与 219891.45 基线同协议，无 softmax、无 mean-anchor）。
训练仍用 V3 的 softmax + mean-anchor + future MSE（可微）。
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 复用 V2 的 Critic + 文本拆分 + 1D-CNN QueryEncoder
from run_expert_attention_retrieval_v2 import (      # noqa: E402
    CriticAgent,
    QueryEncoder,
    _default_model,
    _get_client,
    seed_all,
    split_knowledge_items,
    zscore_window,
)
# 复用 219891 基线的线性去趋势残差
from run_solar_same_val_baselines import residualize  # noqa: E402

SHAPE_BASELINE = 219891.45


def generate_candidate_knowledge(model=None):
    """LLM 生成候选专家知识（只提出候选，不审核）。返回 List[str]。"""
    client = _get_client()
    model = model or _default_model()
    system = (
        "You are a domain expert on solar photovoltaic generation forecasting. "
        "A retrieval model forecasts a univariate solar generation series (OT, 10-minute resolution) "
        "by retrieving similar historical segments from a memory bank and averaging their futures. "
        "Propose candidate pieces of expert knowledge about solar generation dynamics that tell the model "
        "WHICH historical segments are most useful to retrieve. "
        "Each piece must be one concrete, actionable, self-contained statement. "
        "Propose a generous number of candidates (8 to 15) so a separate critic can select the useful ones. "
        "Do NOT output executable retrieval rules or numeric thresholds — only domain mechanisms. "
        "Do NOT invent covariates that are not present in a single univariate generation series. "
        "Output strict JSON only."
    )
    user = (
        'Return exactly: {"knowledge": ["<statement 1>", "<statement 2>", ...]}. '
        "TASK: propose candidate expert knowledge. Output only the JSON object."
    )
    resp = client.messages.create(
        model=model, max_tokens=4000, system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(
        getattr(b, "text", "")
        for b in resp.content
        if getattr(b, "type", "text") == "text"
    )
    if not raw.strip():
        raise RuntimeError("知识生成 LLM 返回空")
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise RuntimeError("知识生成输出无法解析为 JSON：\n" + raw)
    data = json.loads(m.group(0))
    know = data.get("knowledge", data if isinstance(data, list) else [])
    if not isinstance(know, list):
        raise RuntimeError("知识生成输出缺少 knowledge 列表")
    return [str(k).strip() for k in know if str(k).strip()]


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def build_windows(series, starts, context, horizon):
    starts = np.asarray(starts, dtype=np.int64)
    X = np.stack([series[s:s + context] for s in starts]).astype("float32")
    Y = np.stack([series[s + context:s + context + horizon] for s in starts]).astype("float32")
    return X, Y


# --------------------------------------------------------------------------- #
# 模型：CNN 时间序列编码 + Expert Embedding -> z；pairwise MLP -> score
# --------------------------------------------------------------------------- #
class ExpertEmbeddingMLPRetriever(nn.Module):
    def __init__(self, context, expert_dim, d_enc=128, d_z=128, hidden=32, beta_init=0.2, use_expert=True):
        super().__init__()
        self.use_expert = use_expert
        self.ts_encoder = QueryEncoder(context, d_enc)      # 1D CNN，可学习
        if use_expert:
            self.cond = nn.Sequential(
                nn.Linear(d_enc + expert_dim, d_z),
                nn.GELU(),
            )
        else:
            self.cond = None
        # V3 同款 pairwise MLP
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_z, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.logit_gamma = nn.Parameter(torch.tensor(-2.0))
        beta_init = float(np.clip(beta_init, 1e-4, 1 - 1e-4))
        self.logit_beta = nn.Parameter(torch.tensor(np.log(beta_init / (1 - beta_init)), dtype=torch.float32))

    def encode(self, x, expert_emb=None):
        """x: [B, context] -> z: [B, d_z]。CNN-only 时 z = CNN(x)；否则结合 expert embedding。"""
        enc = self.ts_encoder(x)                       # [B, d_enc]
        if not self.use_expert:
            return enc                                  # z = CNN(x)
        e = expert_emb.unsqueeze(0).expand(len(x), -1)  # [B, expert_dim]
        return self.cond(torch.cat([enc, e], dim=-1))   # [B, d_z]

    def delta(self, z_q, z_c):
        """z_q: [B,d_z], z_c: [C,d_z] -> delta: [B,C]"""
        B = z_q.shape[0]
        C = z_c.shape[0]
        qq = z_q[:, None, :].expand(B, C, -1)
        cc = z_c[None, :, :].expand(B, C, -1)
        return self.mlp(torch.cat([qq, cc, torch.abs(qq - cc)], dim=-1)).squeeze(-1)

    def score(self, shape, x_q, x_c, expert_emb):
        z_q = self.encode(x_q, expert_emb)
        z_c = self.encode(x_c, expert_emb)
        delta = self.delta(z_q, z_c)
        return torch.sigmoid(self.logit_gamma) * delta + torch.sigmoid(self.logit_beta) * shape

    def combine(self, delta, shape, mode):
        gamma = torch.sigmoid(self.logit_gamma)
        beta = torch.sigmoid(self.logit_beta)
        if mode == "shape":
            return shape
        if mode == "expert":
            return gamma * delta
        return gamma * delta + beta * shape


# --------------------------------------------------------------------------- #
# 训练（V3 风格：softmax + mean-anchor + future MSE）
# --------------------------------------------------------------------------- #
def train(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res, expert_emb,
          device, epochs=30, batch_size=64, candidates=256, lr=1e-3, temp=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ev = torch.tensor(expert_emb, dtype=torch.float32, device=device)  # [E]
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for p in range(0, n, batch_size):
            ids = order[p:p + batch_size]
            if len(ids) == 0:
                continue
            cidx = rng.choice(len(X_bank), size=min(candidates, len(X_bank)), replace=False)

            x_q = torch.tensor(X_fit[ids], dtype=torch.float32, device=device)
            x_c = torch.tensor(X_bank[cidx], dtype=torch.float32, device=device)
            shape = torch.tensor(
                X_fit_res[ids] @ X_bank_res[cidx].T / L, dtype=torch.float32, device=device
            )  # [B, C]

            s = model.score(shape, x_q, x_c, ev)          # [B, C]
            w = torch.softmax(s / temp, dim=1)

            q_mu = X_fit[ids].mean(axis=1)
            c_mu = X_bank[cidx].mean(axis=1)
            transferred = Y_bank[cidx][None, :, :] + q_mu[:, None, None] - c_mu[None, :, None]
            pred = torch.sum(w.unsqueeze(-1) * torch.tensor(transferred, dtype=torch.float32, device=device), dim=1)
            y = torch.tensor(Y_fit[ids], dtype=torch.float32, device=device)

            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        gamma = float(torch.sigmoid(model.logit_gamma).detach().cpu())
        beta = float(torch.sigmoid(model.logit_beta).detach().cpu())
        print(f"epoch {ep:02d}/{epochs} loss={np.mean(losses):.6f} gamma={gamma:.4f} beta={beta:.4f}")
    return {"gamma": gamma, "beta": beta}


# --------------------------------------------------------------------------- #
# 评估（Top-K 均值，与 219891 基线同协议）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def retrieve_topk_mean(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res, expert_emb,
                       mode, topk, device, q_chunk=128, bank_chunk=4096):
    L = X_val.shape[1]
    ev = torch.tensor(expert_emb, dtype=torch.float32, device=device)
    preds = []
    for qs in range(0, len(X_val), q_chunk):
        qe = min(qs + q_chunk, len(X_val))
        nq = qe - qs
        best_scores = np.full((nq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nq, topk), -1, dtype=np.int64)

        x_q = torch.tensor(X_val[qs:qe], dtype=torch.float32, device=device)
        res_q = torch.tensor(X_val_res[qs:qe], dtype=torch.float32, device=device)
        z_q = model.encode(x_q, ev) if mode != "shape" else None

        for bs in range(0, len(X_bank), bank_chunk):
            be = min(bs + bank_chunk, len(X_bank))
            res_c = torch.tensor(X_bank_res[bs:be], dtype=torch.float32, device=device)
            shape = (res_q @ res_c.T) / L

            if mode == "shape":
                s = shape
            else:
                x_c = torch.tensor(X_bank[bs:be], dtype=torch.float32, device=device)
                z_c = model.encode(x_c, ev)
                delta = model.delta(z_q, z_c)
                s = model.combine(delta, shape, mode)

            s_np = s.cpu().numpy()
            kk = min(topk, be - bs)
            local_part = np.argpartition(s_np, -kk, axis=1)[:, -kk:]
            local_scores = np.take_along_axis(s_np, local_part, axis=1)
            local_idx = local_part + bs
            comb_scores = np.concatenate([best_scores, local_scores], axis=1)
            comb_idx = np.concatenate([best_idx, local_idx], axis=1)
            part = np.argpartition(comb_scores, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(comb_scores, part, axis=1)
            best_idx = np.take_along_axis(comb_idx, part, axis=1)

        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_idx = np.take_along_axis(best_idx, order, axis=1)
        pred = Y_bank[best_idx].mean(axis=1)   # Top-K 均值（无 mean-anchor，对齐 219891）
        preds.append(pred)

    return np.concatenate(preds, axis=0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--root", type=Path, default=Path("results/multidataset"))
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--d-enc", type=int, default=128)
    ap.add_argument("--d-z", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-bank", type=int, default=25000)
    ap.add_argument("--max-train-query", type=int, default=6000)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = args.root / args.dataset
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

    # TEST 不读取
    print(f"{args.dataset}: train windows={len(train_starts)}, val windows={len(val_starts)}")

    # ---- 1. LLM 生成候选知识 + Critic 审核 ---- #
    print("LLM 生成候选知识 ...")
    candidates = generate_candidate_knowledge()
    print(f"LLM 生成 {len(candidates)} 条候选知识：")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c}")

    critic = CriticAgent(max_tokens=16000)
    audit = critic.review(candidates)
    accepted = audit["accepted"]
    rejected = audit["rejected"]
    print(f"\nCritic: accepted={len(accepted)} rejected={len(rejected)}")
    for r in audit["audit"]:
        print(f"  [{r['id']}] {r['verdict']}: {r['reason']}")
    if len(accepted) == 0:
        raise RuntimeError("Critic 拒绝了全部候选知识，无法构造 Expert Embedding。")

    # ---- 2. Expert Embedding（冻结 MiniLM，accepted 条目 mean-pool） ---- #
    print(f"Embedding {len(accepted)} 条 accepted 知识 ...")
    enc = SentenceTransformer(args.text_model, device=str(device))
    emb = enc.encode(accepted, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    expert_emb = emb.mean(axis=0)                     # [E] = mean-pool
    expert_dim = expert_emb.shape[0]
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 3. TRAIN 切分（前 80% bank，后 20% training query） ---- #
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]
    if len(bank_starts) > args.max_bank:
        take = np.linspace(0, len(bank_starts) - 1, args.max_bank).astype(int)
        bank_starts = bank_starts[take]
    if len(fit_starts) > args.max_train_query:
        take = np.linspace(0, len(fit_starts) - 1, args.max_train_query).astype(int)
        fit_starts = fit_starts[take]

    X_bank, Y_bank = build_windows(series, bank_starts, args.context, args.horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, args.context, args.horizon)
    X_val, Y_val = build_windows(series, val_starts, args.context, args.horizon)

    # 残差（线性去趋势，与 219891 基线同定义）
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)

    # ---- 4. 模型：CNN-only 与 CNN+Expert 两个，其余结构完全一致 ---- #
    cnn_only_model = ExpertEmbeddingMLPRetriever(
        context=args.context, expert_dim=expert_dim,
        d_enc=args.d_enc, d_z=args.d_z, hidden=args.hidden, use_expert=False,
    ).to(device)
    cnn_expert_model = ExpertEmbeddingMLPRetriever(
        context=args.context, expert_dim=expert_dim,
        d_enc=args.d_enc, d_z=args.d_z, hidden=args.hidden, use_expert=True,
    ).to(device)

    print("\nTraining CNN-only + MLP ...")
    gates_cnn = train(
        cnn_only_model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res, expert_emb,
        device=device, epochs=args.epochs, batch_size=args.batch_size,
        candidates=args.candidates, lr=args.lr,
    )

    print("\nTraining CNN + Expert-Embedding + MLP ...")
    gates_exp = train(
        cnn_expert_model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res, expert_emb,
        device=device, epochs=args.epochs, batch_size=args.batch_size,
        candidates=args.candidates, lr=args.lr,
    )

    # ---- 5. 三个 arm 评估（Top-K 均值，与 Shape 219891.45 同协议） ---- #
    print("\nEvaluating (Top-K mean) ...")
    shape_pred = retrieve_topk_mean(cnn_expert_model, X_val, X_bank, Y_bank, X_val_res, X_bank_res, expert_emb, "shape", args.topk, device)
    cnn_only_pred = retrieve_topk_mean(cnn_only_model, X_val, X_bank, Y_bank, X_val_res, X_bank_res, expert_emb, "expert", args.topk, device)
    cnn_expert_pred = retrieve_topk_mean(cnn_expert_model, X_val, X_bank, Y_bank, X_val_res, X_bank_res, expert_emb, "expert", args.topk, device)

    shape_mse = mse(Y_val, shape_pred)
    cnn_only_mse = mse(Y_val, cnn_only_pred)
    cnn_expert_mse = mse(Y_val, cnn_expert_pred)

    # 专家独立增益 = (CNN-only - CNN+Expert) / CNN-only
    expert_gain = (cnn_only_mse - cnn_expert_mse) / cnn_only_mse * 100.0

    res = {
        "shape_only": {"mse": shape_mse, "mae": mae(Y_val, shape_pred)},
        "cnn_only_mlp": {"mse": cnn_only_mse, "mae": mae(Y_val, cnn_only_pred)},
        "cnn_expert_mlp": {"mse": cnn_expert_mse, "mae": mae(Y_val, cnn_expert_pred)},
        "expert_independent_gain_pct": expert_gain,
        "cnn_vs_shape_pct": (shape_mse - cnn_only_mse) / shape_mse * 100.0,
        "cnn_expert_vs_shape_pct": (shape_mse - cnn_expert_mse) / shape_mse * 100.0,
        "gamma_cnn_only": gates_cnn["gamma"],
        "gamma_cnn_expert": gates_exp["gamma"],
        "beta_cnn_expert": gates_exp["beta"],
        "num_candidates": len(candidates),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "candidates": candidates,
        "critic_audit": audit["audit"],
        "accepted_knowledge": accepted,
        "rejected_knowledge": rejected,
        "expert_dim": expert_dim,
        "baseline_shape_mse": SHAPE_BASELINE,
        "test_used": False,
    }

    out = ds / "auto_expert_mlp"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cnn_only_state": cnn_only_model.state_dict(),
        "cnn_expert_state": cnn_expert_model.state_dict(),
        "args": vars(args), "accepted_knowledge": accepted, "expert_emb": expert_emb,
    }, out / "models.pt")
    np.savez_compressed(out / "val_predictions.npz", y_true=Y_val, val_starts=val_starts,
                        shape_pred=shape_pred, cnn_only_pred=cnn_only_pred,
                        cnn_expert_pred=cnn_expert_pred)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "critic_audit.json").write_text(
        json.dumps({"candidates": candidates, "accepted": accepted, "rejected": rejected,
                    "audit": audit["audit"]}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 6. 报告 ---- #
    print("\n" + "=" * 72)
    print(f"LLM 候选知识条数        = {len(candidates)}")
    print(f"Critic accepted        = {len(accepted)}")
    print(f"Critic rejected        = {len(rejected)}")
    print("-" * 72)
    print(f"Shape-only             MSE = {shape_mse:.2f}")
    print(f"CNN-only + MLP         MSE = {cnn_only_mse:.2f}")
    print(f"CNN + Expert-Emb + MLP MSE = {cnn_expert_mse:.2f}")
    print("-" * 72)
    print(f"CNN-only vs Shape      = {res['cnn_vs_shape_pct']:+.2f}%")
    print(f"CNN+Expert vs Shape    = {res['cnn_expert_vs_shape_pct']:+.2f}%")
    print(f"Expert independent gain = {expert_gain:+.2f}%")
    print(f"gamma_cnn={gates_cnn['gamma']:.4f}  gamma_exp={gates_exp['gamma']:.4f}  beta_exp={gates_exp['beta']:.4f}")
    print("=" * 72)
    print("Saved to:", out)
    print("TEST was NOT evaluated.")


if __name__ == "__main__":
    main()
