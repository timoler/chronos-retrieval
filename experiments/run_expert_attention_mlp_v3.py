#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solar Query-dependent Expert Attention + MLP 消融 (v3)。

统一 Solar VAL 协议（results/multidataset/solar/window_indices.npz，9905 窗，
stride 1；residualize Pearson shape；Top-K=5 均值，无 mean-anchor；无 Chronos）。

四组臂：
  A. Shape-only                          （residualize + Top-K mean，~219891.45）
  B. CNN-only MLP                        （ExpertEmbeddingMLPRetriever use_expert=False）
  C. CNN + mean-pooled fixed Expert      （旧错误版：expert_emb = mean(e1..eN)）
  D. CNN + Query-dependent Expert Attention（新：w_i(q)=Attention(q,e_i)，e(q)=Σ w_i e_i）

C/D 使用同一组 v3 Data-Critic-ACCEPT 的 expert 文本，只差聚合方式。
核心指标 = D vs B 提升（D>B 才证明 query-dependent Expert 有效）。

只读 results/solar_agent_v3_audit.json 的 accepted_expert_texts。
TEST 不读取、不评估。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import run_expert_attention_retrieval_v2 as attn  # QueryEncoder / seed_all / _get_client
from run_expert_embedding_mlp import (  # noqa: E402
    ExpertEmbeddingMLPRetriever,
    build_windows,
    mse,
    mae,
    train,
    retrieve_topk_mean,
)
from run_solar_same_val_baselines import residualize, shape_retrieval  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SHAPE_BASELINE = 219891.45


# --------------------------------------------------------------------------- #
# 新模型：Query-dependent Expert Attention + MLP（臂 D）
# --------------------------------------------------------------------------- #
class QueryExpertAttentionMLPRetriever(nn.Module):
    def __init__(self, context, expert_dim, d_enc=128, d_attn=64, d_z=128,
                 hidden=32, beta_init=0.2):
        super().__init__()
        self.ts_encoder = attn.QueryEncoder(context, d_enc)       # [B,512]->[B,128]
        self.attn_query_proj = nn.Linear(d_enc, d_attn)           # q -> attention space
        self.expert_proj = nn.Linear(expert_dim, d_attn)          # e_i -> attention space
        self.cond = nn.Sequential(
            nn.Linear(d_enc + expert_dim, d_z), nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_z, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.logit_gamma = nn.Parameter(torch.tensor(-2.0))
        beta_init = float(np.clip(beta_init, 1e-4, 1 - 1e-4))
        self.logit_beta = nn.Parameter(
            torch.tensor(np.log(beta_init / (1 - beta_init)), dtype=torch.float32))

    def encode(self, x, expert_emb):
        # x: [B, context]; expert_emb: [N, E]（保留每条专家，不做 mean-pool）
        q = self.ts_encoder(x)                                   # [B, d_enc]
        qa = self.attn_query_proj(q)                             # [B, d_attn]
        ea = self.expert_proj(expert_emb)                        # [N, d_attn]
        scores = qa @ ea.T / math.sqrt(ea.shape[-1])             # [B, N]
        w = torch.softmax(scores, dim=-1)                        # [B, N]
        e_q = w @ expert_emb                                     # [B, E]  query-dependent
        return self.cond(torch.cat([q, e_q], dim=-1))            # [B, d_z]

    def delta(self, z_q, z_c):
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

    def attention_weights(self, x, expert_emb):
        """返回 [B, N] 的 query→expert attention 权重（诊断用）。"""
        q = self.ts_encoder(x)
        qa = self.attn_query_proj(q)
        ea = self.expert_proj(expert_emb)
        scores = qa @ ea.T / math.sqrt(ea.shape[-1])
        return torch.softmax(scores, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/multidataset"))
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--d-enc", type=int, default=128)
    ap.add_argument("--d-attn", type=int, default=64)
    ap.add_argument("--d-z", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-bank", type=int, default=25000)
    ap.add_argument("--max-train-query", type=int, default=6000)
    ap.add_argument("--n-diag", type=int, default=100)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--audit", type=Path, default=Path("results/solar_agent_v3_audit.json"))
    args = ap.parse_args()

    attn.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- 读 v3 审计的 accepted expert texts ----
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    accepted = [str(t) for t in audit.get("accepted_expert_texts", []) if str(t).strip()]
    print(f"accepted expert texts (Data Critic ACCEPT): {len(accepted)}")
    for i, t in enumerate(accepted, 1):
        print(f"  [E{i}] {t}")

    # ---- 数据加载（multidataset 9905-window 协议）----
    ds = args.root / args.dataset
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

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

    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)

    # ---- A. Shape-only ----
    print("\n[Arm A] Shape-only ...")
    shape_pred = shape_retrieval(X_val, X_bank, Y_bank, topk=args.topk)
    shape_mse = mse(Y_val, shape_pred)
    shape_mae = mae(Y_val, shape_pred)
    print(f"  Shape-only MSE={shape_mse:.2f} MAE={shape_mae:.2f}")

    # ---- expert embeddings ----
    if len(accepted) == 0:
        print("\n[WARN] no accepted experts -> skip C/D arms.")
        expert_dim = 0
        emb = np.zeros((0, 0), dtype=np.float32)
    else:
        enc = SentenceTransformer(args.text_model, device=str(device))
        emb = enc.encode(accepted, normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")  # [N, 384]
        expert_dim = emb.shape[1]
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    expert_emb_mean = emb.mean(axis=0) if len(accepted) else np.zeros(1, dtype=np.float32)

    # ---- B. CNN-only MLP ----
    print("\n[Arm B] CNN-only MLP ...")
    cnn_only = ExpertEmbeddingMLPRetriever(
        context=args.context, expert_dim=expert_dim if expert_dim else 1,
        d_enc=args.d_enc, d_z=args.d_z, hidden=args.hidden, use_expert=False,
    ).to(device)
    train(cnn_only, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
          expert_emb_mean, device=device, epochs=args.epochs,
          batch_size=args.batch_size, candidates=args.candidates, lr=args.lr)
    cnn_only_pred = retrieve_topk_mean(
        cnn_only, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
        expert_emb_mean, "expert", args.topk, device)
    cnn_only_mse = mse(Y_val, cnn_only_pred)
    cnn_only_mae = mae(Y_val, cnn_only_pred)
    print(f"  CNN-only MSE={cnn_only_mse:.2f} MAE={cnn_only_mae:.2f}")

    # ---- C / D（需至少 1 条 accepted expert）----
    fixed_mse = fixed_mae = None
    dyn_mse = dyn_mae = None
    attn_diag = None

    if len(accepted) > 0:
        # ---- C. CNN + mean-pool fixed Expert ----
        print("\n[Arm C] CNN + mean-pool fixed Expert ...")
        cnn_fixed = ExpertEmbeddingMLPRetriever(
            context=args.context, expert_dim=expert_dim,
            d_enc=args.d_enc, d_z=args.d_z, hidden=args.hidden, use_expert=True,
        ).to(device)
        train(cnn_fixed, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              expert_emb_mean, device=device, epochs=args.epochs,
              batch_size=args.batch_size, candidates=args.candidates, lr=args.lr)
        fixed_pred = retrieve_topk_mean(
            cnn_fixed, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
            expert_emb_mean, "expert", args.topk, device)
        fixed_mse = mse(Y_val, fixed_pred)
        fixed_mae = mae(Y_val, fixed_pred)
        print(f"  CNN+fixed Expert MSE={fixed_mse:.2f} MAE={fixed_mae:.2f}")

        # ---- D. CNN + Query-dependent Expert Attention ----
        print("\n[Arm D] CNN + Query-dependent Expert Attention ...")
        dynamic = QueryExpertAttentionMLPRetriever(
            context=args.context, expert_dim=expert_dim,
            d_enc=args.d_enc, d_attn=args.d_attn, d_z=args.d_z, hidden=args.hidden,
        ).to(device)
        train(dynamic, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              emb, device=device, epochs=args.epochs,
              batch_size=args.batch_size, candidates=args.candidates, lr=args.lr)
        dyn_pred = retrieve_topk_mean(
            dynamic, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
            emb, "expert", args.topk, device)
        dyn_mse = mse(Y_val, dyn_pred)
        dyn_mae = mae(Y_val, dyn_pred)
        print(f"  CNN+Attention Expert MSE={dyn_mse:.2f} MAE={dyn_mae:.2f}")

        # ---- attention 诊断（均匀抽 n_diag 个 VAL query）----
        print(f"\n[Attention Diagnostic] {args.n_diag} VAL queries ...")
        n_diag = min(args.n_diag, len(X_val))
        diag_idx = np.linspace(0, len(X_val) - 1, n_diag).astype(int)
        ev = torch.tensor(emb, dtype=torch.float32, device=device)
        dynamic.eval()
        with torch.no_grad():
            w_all = []
            for i in diag_idx:
                x = torch.tensor(X_val[i:i + 1], dtype=torch.float32, device=device)
                w = dynamic.attention_weights(x, ev)   # [1, N]
                w_all.append(w[0].cpu().numpy())
        w_all = np.stack(w_all)                        # [n_diag, N]

        per_expert = {}
        for j in range(len(accepted)):
            per_expert[accepted[j]] = {
                "mean": float(w_all[:, j].mean()),
                "std": float(w_all[:, j].std()),
                "max": float(w_all[:, j].max()),
                "min": float(w_all[:, j].min()),
            }
        top1 = w_all.argmax(axis=1)
        top1_counts = {int(j): int((top1 == j).sum()) for j in range(len(accepted))}
        unique_rankings = len({tuple(np.argsort(-w).tolist()) for w in w_all})
        max_weight_std = float(w_all.std(axis=0).max())
        collapse = bool(max_weight_std < 0.02)

        attn_diag = {
            "n_queries": int(n_diag),
            "per_expert": per_expert,
            "top1_counts": top1_counts,
            "unique_rankings": int(unique_rankings),
            "max_weight_std_across_queries": max_weight_std,
            "expert_attention_collapse": collapse,
        }
        print(f"  per-expert mean weights: "
              f"{ {accepted[j][:24]: round(per_expert[accepted[j]]['mean'], 3) for j in range(len(accepted))} }")
        print(f"  unique rankings among {n_diag} queries: {unique_rankings}")
        print(f"  max weight std across queries: {max_weight_std:.4f}")
        print(f"  EXPERT ATTENTION COLLAPSE = {collapse}")

    # ---- 结果 ----
    dyn_vs_b = None
    if dyn_mse is not None:
        dyn_vs_b = float((cnn_only_mse - dyn_mse) / cnn_only_mse * 100.0)

    res = {
        "dataset": args.dataset,
        "protocol": "multidataset 9905-window VAL, residualize Pearson, Top-K=5 mean",
        "accepted_experts": accepted,
        "n_accepted_experts": len(accepted),
        "shape_only": {"mse": shape_mse, "mae": shape_mae},
        "cnn_only_mlp": {"mse": cnn_only_mse, "mae": cnn_only_mae},
        "fixed_expert": ({"mse": fixed_mse, "mae": fixed_mae} if fixed_mse is not None else None),
        "query_attention_expert": ({"mse": dyn_mse, "mae": dyn_mae} if dyn_mse is not None else None),
        "D_vs_B_improvement_pct": dyn_vs_b,
        "attention_diagnostic": attn_diag,
        "baseline_shape_mse": SHAPE_BASELINE,
        "test_used": False,
    }

    out = ds / "query_expert_attention_v3"
    out.mkdir(parents=True, exist_ok=True)

    save_models = {}
    save_preds = {"y_true": Y_val, "val_starts": val_starts, "shape_pred": shape_pred,
                  "cnn_only_pred": cnn_only_pred}
    if fixed_mse is not None:
        save_models["cnn_fixed_state"] = cnn_fixed.state_dict()
        save_preds["fixed_expert_pred"] = fixed_pred
    if dyn_mse is not None:
        save_models["dynamic_state"] = dynamic.state_dict()
        save_preds["query_attention_pred"] = dyn_pred
    save_models["cnn_only_state"] = cnn_only.state_dict()

    torch.save(save_models, out / "models.pt")
    np.savez_compressed(out / "val_predictions.npz", **save_preds)
    (out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "attention_diagnostic.json").write_text(
        json.dumps(attn_diag if attn_diag else {}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n" + "=" * 72)
    print("SOLAR VAL ABLATION (9905 windows, Top-K=5 mean)")
    print("=" * 72)
    print(f"A. Shape-only                  MSE = {shape_mse:.2f}  MAE = {shape_mae:.2f}")
    print(f"B. CNN-only MLP                MSE = {cnn_only_mse:.2f}  MAE = {cnn_only_mae:.2f}")
    if fixed_mse is not None:
        print(f"C. CNN + mean-pool Expert      MSE = {fixed_mse:.2f}  MAE = {fixed_mae:.2f}")
    if dyn_mse is not None:
        print(f"D. CNN + Attention Expert      MSE = {dyn_mse:.2f}  MAE = {dyn_mae:.2f}")
        print(f"   D vs B improvement          = {dyn_vs_b:+.2f}%")
    print("=" * 72)
    print("Saved to:", out)
    print("TEST was NOT evaluated.")


if __name__ == "__main__":
    main()
