#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_expert_attention_retrieval_v2.py

严格顺序：
    专家自然语言知识 -> Critic Agent 审核 -> 只保留审核通过的知识
    -> 每条通过知识单独 embedding -> 与 Query 一起做 Attention -> 检索历史 -> Memory

与 V1 的区别：
1. 专家文本先按“知识条目”拆分，不再整段文本合成一个 embedding。
2. 增加 Critic Agent 审核阶段：相关 / 可观测 / 是否指导检索 / 是否空话，逐条判断。
   只有 accepted 条目进入 embedding；rejected 完全不进入模型。
   Critic 是真实 LLM 调用，绝不硬编码 accepted=True。
3. 多条通过知识各自 embedding，得到多个 expert token。
4. Attention 输入 [Query, Expert1, ..., ExpertN]，取更新后的 Query token 检索。

严格约束：
- 不使用 hour / doy / level / change 等任何手工数值专家特征。
- Query 只从原始 512 点序列用轻量 1D CNN 编码。
- 训练只使用 TRAIN：前 80% 做 memory bank，后 20% 做 retrieval training query。
- VAL 使用当前统一 9905 个 stride=1 窗口。
- TEST 完全不读取、不评估。
- 第一轮不跑 Chronos、不加 lambda Gate。

输出：
- V2 Memory MSE / MAE
- accepted / rejected 数量与逐条审核结果（critic_audit.json）
- Attention 中每条 expert token 的平均权重

如果当前项目没有可用 LLM Agent 接口，本脚本会在 Critic 阶段明确报错，
提示“需要接入现有 Agent 接口”，绝不硬编码 accepted 冒充审核。
"""

from __future__ import annotations
import argparse
import json
import os
import random
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


# --------------------------------------------------------------------------- #
# 指标 / 随机种子
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Critic Agent（独立模块，真实 LLM 调用）
# --------------------------------------------------------------------------- #
def _default_model():
    return (
        os.environ.get("ANTHROPIC_MODEL")
        or os.environ.get("LLM_MODEL")
        or "deepseek-v4-pro[1m]"
    )


def _get_client():
    import anthropic

    kwargs = {}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if base_url:
        kwargs["base_url"] = base_url
    if auth_token:
        kwargs["auth_token"] = auth_token
    elif api_key:
        kwargs["api_key"] = api_key
    else:
        raise RuntimeError(
            "需要接入现有 Agent 接口：请设置 ANTHROPIC_AUTH_TOKEN 或 ANTHROPIC_API_KEY。"
            "本脚本不会硬编码 accepted 列表冒充 Agent 审核。"
        )
    return anthropic.Anthropic(**kwargs)


def split_knowledge_items(text: str):
    """按句子/换行把专家文本拆成知识条目（不做任何内容改写）。"""
    parts = re.split(r"(?<=[.;。；])\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class CriticAgent:
    """逐条审核专家知识。真实调用 LLM，绝不硬编码 accepted。"""

    CRITERIA = (
        "1. RELEVANCE: is it about solar generation and its forecasting dynamics?\n"
        "2. OBSERVABILITY: can it be observed or supported from a single univariate "
        "generation series (OT), without external covariates?\n"
        "3. RETRIEVAL-GUIDANCE: does it genuinely say HOW historical segments should be "
        "retrieved (not merely restate that retrieval might help)?\n"
        "4. CONCRETENESS: is it specific and executable, not vague / empty / generic filler?"
    )

    def __init__(self, model=None, max_tokens=4000):
        self.model = model or _default_model()
        self.client = _get_client()
        self.max_tokens = max_tokens

    def review(self, items, domain="univariate solar generation (OT)"):
        system = (
            "You are a strict domain-knowledge critic for a time-series retrieval model. "
            "The task is to forecast a univariate solar-generation series by retrieving "
            "similar historical segments. You review a list of expert knowledge statements "
            "and decide, for each, ACCEPT (use it to condition retrieval) or REJECT.\n\n"
            "Judge each statement on these four criteria:\n"
            + self.CRITERIA
            + "\n\nRULES:\n"
            "- Do NOT invent, add, merge, or rewrite any knowledge. Only judge the given items.\n"
            "- Reject statements that are empty, circular, or not observable from a single series.\n"
            "- A statement may be true but still REJECTED if it does not tell the retriever "
            "anything actionable.\n"
            "- Output STRICT JSON only."
        )
        numbered = "\n".join(f"[{i}] {it}" for i, it in enumerate(items))
        schema = (
            'Return exactly: {"reviews": [ {"id": <int>, "verdict": "ACCEPT" or "REJECT", '
            '"reason": "<one sentence>"}, ... ]}. One entry per item, same order, id = the [i] index.'
        )
        user = (
            f"DOMAIN: {domain}\n\n"
            f"KNOWLEDGE ITEMS:\n{numbered}\n\n"
            f"{schema}\n\n"
            f"TASK: review every item. Output only the JSON object."
        )
        raw = self._call(system, user)
        return self._normalize(raw, items)

    def _call(self, system, user):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(
            getattr(b, "text", "")
            for b in resp.content
            if getattr(b, "type", "text") == "text"
        )
        if not raw.strip():
            raise RuntimeError(
                f"Critic LLM 返回空 (stop_reason={getattr(resp, 'stop_reason', None)})"
            )
        return raw

    def _normalize(self, raw, items):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise RuntimeError("Critic 输出无法解析为 JSON：\n" + raw)
        data = json.loads(m.group(0))
        reviews = data.get("reviews", [])
        if not isinstance(reviews, list):
            raise RuntimeError("Critic 输出缺少 reviews 列表")

        by_id = {}
        for r in reviews:
            if isinstance(r, dict) and "id" in r:
                by_id[int(r["id"])] = r

        accepted, rejected, audit = [], [], []
        for i, item in enumerate(items):
            r = by_id.get(i)
            if r is None:
                r = {"id": i, "verdict": "REJECT",
                     "reason": "no review returned for this item (treated as REJECT)"}
            verdict = str(r.get("verdict", "REJECT")).strip().upper()
            reason = str(r.get("reason", "")).strip()
            if verdict == "ACCEPT":
                accepted.append(item)
            else:
                rejected.append({"item": item, "reason": reason})
            audit.append({"id": i, "item": item, "verdict": verdict, "reason": reason})

        return {"accepted": accepted, "rejected": rejected, "audit": audit}


# --------------------------------------------------------------------------- #
# 模型：Query 1D-CNN + [Query, Expert1..N] self-attention
# --------------------------------------------------------------------------- #
class QueryEncoder(nn.Module):
    """原始 512 点时间序列 -> d_model。无任何人工时间/状态特征。"""
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


class ExpertAttentionRetrieverV2(nn.Module):
    """
    [Query token, Expert token 1, ..., Expert token N] -> self-attention
    -> 取更新后的 Query token 作为 expert-conditioned query。
    """
    def __init__(self, context, text_dim, d_model=128, heads=4, dropout=0.1):
        super().__init__()
        self.query_encoder = QueryEncoder(context, d_model)
        self.expert_proj = nn.Linear(text_dim, d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x, expert_embeddings):
        """
        x: [B, context]
        expert_embeddings: [B, N, text_dim]  (N = 通过审核的知识条数，>=1)
        返回 (conditioned_query [B,D], attn_weights [B, 1+N, 1+N])。
        """
        q = self.query_encoder(x)                       # [B, D]
        e = self.expert_proj(expert_embeddings)         # [B, N, D]
        tokens = torch.cat([q.unsqueeze(1), e], dim=1)  # [B, 1+N, D]
        attn_out, attn_weights = self.attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True,
        )
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(tokens)
        q_conditioned = self.out_norm(tokens[:, 0])
        return F.normalize(q_conditioned, dim=-1), attn_weights


# --------------------------------------------------------------------------- #
# 数据 / embedding / 检索
# --------------------------------------------------------------------------- #
def build_windows(series, starts, context, horizon):
    starts = np.asarray(starts, dtype=np.int64)
    X = np.stack([series[s:s + context] for s in starts]).astype("float32")
    Y = np.stack([series[s + context:s + context + horizon] for s in starts]).astype("float32")
    return X, Y


def batch_embed(model, X, expert_emb, device, batch=512, collect_weights=False):
    """返回 (embeddings [N,D], weights [N,1+K,1+K] or None)。"""
    outs, weights = [], []
    model.eval()
    with torch.no_grad():
        ev = torch.tensor(expert_emb, dtype=torch.float32, device=device)  # [K, text_dim]
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            eb = ev.unsqueeze(0).expand(len(xb), -1, -1)
            z, w = model(xb, eb)
            outs.append(z.cpu().numpy())
            if collect_weights:
                weights.append(w.cpu().numpy())
    emb = np.concatenate(outs, axis=0)
    w_all = np.concatenate(weights, axis=0) if collect_weights else None
    return emb, w_all


def retrieve_memory(query_z, bank_z, bank_future, topk=5, chunk=256):
    preds = []
    for i in range(0, len(query_z), chunk):
        q = torch.tensor(query_z[i:i + chunk], dtype=torch.float32)
        b = torch.tensor(bank_z, dtype=torch.float32)
        sim = q @ b.T
        idx = torch.topk(sim, k=min(topk, len(bank_z)), dim=1).indices.numpy()
        preds.append(bank_future[idx].mean(axis=1))
    return np.concatenate(preds, axis=0)


def train_retriever(model, X_bank, Y_bank, X_train, Y_train, expert_emb,
                    device, epochs=20, batch_size=64, candidates=256, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ev = torch.tensor(expert_emb, dtype=torch.float32, device=device)  # [K, text_dim]

    Xb = torch.tensor(X_bank, dtype=torch.float32, device=device)
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)

    n = len(X_train)
    rng = np.random.default_rng(12345)

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []

        with torch.no_grad():
            bz_parts = []
            for j in range(0, len(X_bank), 512):
                xb0 = Xb[j:j + 512]
                eb0 = ev.unsqueeze(0).expand(len(xb0), -1, -1)
                bz_parts.append(model(xb0, eb0)[0])
            bank_z = torch.cat(bz_parts, dim=0).detach()

        for p in range(0, n, batch_size):
            ids = order[p:p + batch_size]
            if len(ids) == 0:
                continue

            x = torch.tensor(X_train[ids], dtype=torch.float32, device=device)
            y = torch.tensor(Y_train[ids], dtype=torch.float32, device=device)
            e = ev.unsqueeze(0).expand(len(x), -1, -1)
            qz, _ = model(x, e)

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


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
SHAPE_BASELINE = 219891.45
V1_BASELINE = 233208.00


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
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--critic-model", default=None,
                    help="Critic Agent 使用的模型；默认读 ANTHROPIC_MODEL")
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
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

    # TEST 故意不读取、不评估
    print(f"{args.dataset}: train windows={len(train_starts)}, val windows={len(val_starts)}")

    expert_text = expert_path.read_text(encoding="utf-8").strip()
    if not expert_text:
        raise ValueError("expert_knowledge.txt 为空")

    # ---- 1. 拆知识条目 ---- #
    items = split_knowledge_items(expert_text)
    print(f"\n拆出 {len(items)} 条知识：")
    for i, it in enumerate(items):
        print(f"  [{i}] {it}")

    # ---- 2. Critic Agent 审核（真实 LLM，先于 embedding） ---- #
    print("\n[Critic] 审核中（调用 LLM）...")
    critic = CriticAgent(model=args.critic_model)
    audit = critic.review(items)
    accepted = audit["accepted"]
    rejected = audit["rejected"]
    print(f"[Critic] accepted={len(accepted)} rejected={len(rejected)}")
    for r in audit["audit"]:
        print(f"  [{r['id']}] {r['verdict']}: {r['reason']}")

    out = ds / "expert_attention_v2"
    out.mkdir(parents=True, exist_ok=True)
    critic_audit = {
        "items": items,
        "accepted_knowledge": accepted,
        "rejected_knowledge": rejected,
        "audit": audit["audit"],
        "critic_model": critic.model,
        "note": "Critic 是真实 LLM 审核；rejected 条目未进入 embedding 与模型。",
    }
    (out / "critic_audit.json").write_text(
        json.dumps(critic_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if len(accepted) == 0:
        print("\n[WARN] Critic 拒绝了全部知识条目，无法进行 expert-conditioned 检索。")
        print("这是合法结果，不硬编码 accepted。")
        # 仍保存一个空的审计结果，然后停止（不训练无专家的模型）
        return

    # ---- 3. 只对 accepted 条目做 embedding（每条一个 token） ---- #
    print(f"\nEmbedding {len(accepted)} 条 accepted 知识（冻结 all-MiniLM-L6-v2）...")
    text_encoder = SentenceTransformer(args.text_model, device=str(device))
    expert_emb = text_encoder.encode(
        accepted, normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")                       # [K, text_dim]
    text_dim = expert_emb.shape[1]
    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 4. TRAIN 切分（与 V1 一致：前 80% bank，后 20% training query） ---- #
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

    model = ExpertAttentionRetrieverV2(
        context=args.context, text_dim=text_dim, d_model=args.d_model, heads=args.heads,
    ).to(device)

    print("\nTraining Expert-Attention V2 ...")
    train_retriever(
        model, X_bank, Y_bank, X_fit, Y_fit, expert_emb,
        device=device, epochs=args.epochs, batch_size=args.batch_size,
        candidates=args.candidates, lr=args.lr,
    )

    print("\nEmbedding bank/VAL ...")
    bank_z, _ = batch_embed(model, X_bank, expert_emb, device)
    val_z, val_weights = batch_embed(model, X_val, expert_emb, device, collect_weights=True)

    print("Retrieving Top-K ...")
    mem_pred = retrieve_memory(val_z, bank_z, Y_bank, topk=args.topk)

    v2_mse = mse(Y_val, mem_pred)
    v2_mae = mae(Y_val, mem_pred)

    # ---- 5. Attention 平均权重：query(位置0) 对每条 expert token(1..K) ---- #
    q2e = val_weights[:, 0, 1:].mean(axis=0)        # [K]
    q2self = float(val_weights[:, 0, 0].mean())
    per_expert_weight = {accepted[j]: float(q2e[j]) for j in range(len(accepted))}

    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "accepted_knowledge": accepted, "expert_embeddings": expert_emb},
               out / "model.pt")
    np.savez_compressed(out / "val_predictions.npz",
                        y_true=Y_val, val_starts=val_starts,
                        expert_attention_v2_memory_pred=mem_pred)

    result = {
        "dataset": args.dataset,
        "v2_memory_mse": v2_mse,
        "v2_memory_mae": v2_mae,
        "critic": critic_audit,
        "attention": {
            "query_self_weight": q2self,
            "per_expert_mean_weight": per_expert_weight,
        },
        "baselines": {"shape_mse": SHAPE_BASELINE, "v1_mse": V1_BASELINE},
        "v2_vs_shape_pct": (SHAPE_BASELINE - v2_mse) / SHAPE_BASELINE * 100.0,
        "v2_vs_v1_pct": (V1_BASELINE - v2_mse) / V1_BASELINE * 100.0,
        "test_used": False,
    }
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print(f"Shape Retrieval      MSE = {SHAPE_BASELINE:.2f}")
    print(f"Expert-Attention V1  MSE = {V1_BASELINE:.2f}")
    print(f"Expert-Attention V2  MSE = {v2_mse:.2f}")
    print(f"V2 vs Shape          = {result['v2_vs_shape_pct']:+.2f}%")
    print(f"V2 vs V1             = {result['v2_vs_v1_pct']:+.2f}%")
    print("-" * 70)
    print(f"accepted knowledge  : {len(accepted)}")
    print(f"rejected knowledge  : {len(rejected)}")
    print("per-expert attention weight:")
    for j, (k, w) in enumerate(per_expert_weight.items()):
        print(f"  expert[{j}] ({w:.4f}): {k}")
    print(f"query self weight   : {q2self:.4f}")
    print("=" * 70)
    print("Saved to:", out)
    print("TEST was NOT evaluated.")


if __name__ == "__main__":
    main()
