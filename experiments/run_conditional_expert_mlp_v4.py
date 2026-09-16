#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solar Conditional Expert MLP 消融 (v4)。

统一 Solar VAL 协议（multidataset 9905 窗，stride 1；residualize Pearson shape；
Top-K=5 均值，无 mean-anchor；无 Chronos；无 Attention）。

三组臂：
  A. CNN-only MLP                       （ExpertEmbeddingMLPRetriever use_expert=False）
  B. Fixed Expert MLP（昨晚旧版）         （所有通过专家 mean-pool 成固定向量）
  C. Conditional Expert MLP（新版）       （ExpertVector(q) = concat(a_i(q)·e_i)，无 attention/mean-pool）

C 的激活 a_i(q) 来自 Critic v4 编译的 activation_feature，确定性从 query context + 时间戳计算。
关键比较 = C vs A（Conditional Expert 是否提供 CNN 没学到的增量信息）。

只读 results/solar_agent_v4_audit.json 的 accepted_experts。
TEST 不读取、不评估。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import run_expert_attention_retrieval_v2 as attn
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
# 确定性 activation 工具
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _linear_slope_vec(X):
    """每行 [L] 的线性斜率（向量化）。"""
    L = X.shape[1]
    t = np.linspace(-1.0, 1.0, L, dtype=np.float32)
    tc = t - t.mean()
    denom = np.sum(tc * tc)
    xm = X.mean(axis=1, keepdims=True)
    return np.sum((X - xm) * tc[None, :], axis=1) / denom


def _lag1_autocorr_vec(X):
    x1 = X[:, 1:]
    x0 = X[:, :-1]
    m1 = x1.mean(axis=1, keepdims=True)
    m0 = x0.mean(axis=1, keepdims=True)
    num = ((x1 - m1) * (x0 - m0)).sum(axis=1)
    den = np.sqrt(((x1 - m1) ** 2).sum(axis=1) * ((x0 - m0) ** 2).sum(axis=1))
    return num / (den + 1e-9)


def build_activation_executor(X_train_all, hourly_mean_OT, monthly_mean_OT):
    """返回 activation(feat, ctx, ts) 可调用对象。stats 用 TRAIN 全部窗口计算。"""
    ctx_means = X_train_all.mean(axis=1)
    ctx_slopes = _linear_slope_vec(X_train_all)
    ctx_std = X_train_all.std(axis=1)
    ctx_lastdiff = X_train_all[:, -1] - X_train_all[:, -2]
    ctx_autocorr = _lag1_autocorr_vec(X_train_all)

    stats = {
        "level_mean": float(ctx_means.mean()), "level_std": float(ctx_means.std()),
        "slope_abs_mean": float(np.abs(ctx_slopes).mean()), "slope_abs_std": float(np.abs(ctx_slopes).std()),
        "diff_abs_mean": float(np.abs(ctx_lastdiff).mean()), "diff_abs_std": float(np.abs(ctx_lastdiff).std()),
        "vol_mean": float(ctx_std.mean()), "vol_std": float(ctx_std.std()),
        "autocorr_mean": float(ctx_autocorr.mean()), "autocorr_std": float(ctx_autocorr.std()),
    }

    def activation(feat, ctx, ts):
        ctx = np.asarray(ctx, dtype=np.float64)
        if feat == "daylight_level":
            h = int(ts.hour)
            peak = float(np.max(hourly_mean_OT)) if len(hourly_mean_OT) else 0.0
            return float(hourly_mean_OT[h] / peak) if peak > 0 else 0.0
        if feat == "seasonal_level":
            m = int(ts.month)
            mx = float(max(monthly_mean_OT.values())) if monthly_mean_OT else 0.0
            return float(monthly_mean_OT.get(m, 0.0) / mx) if mx > 0 else 0.0
        if feat == "recent_level":
            z = (float(ctx.mean()) - stats["level_mean"]) / (stats["level_std"] + 1e-6)
            return float(_sigmoid(z))
        if feat == "recent_slope_magnitude":
            s = abs(float(_linear_slope_vec(ctx[None, :])[0]))
            z = (s - stats["slope_abs_mean"]) / (stats["slope_abs_std"] + 1e-6)
            return float(_sigmoid(z))
        if feat == "recent_diff_magnitude":
            d = abs(float(ctx[-1] - ctx[-2]))
            z = (d - stats["diff_abs_mean"]) / (stats["diff_abs_std"] + 1e-6)
            return float(_sigmoid(z))
        if feat == "recent_volatility":
            v = float(ctx.std())
            z = (v - stats["vol_mean"]) / (stats["vol_std"] + 1e-6)
            return float(_sigmoid(z))
        if feat == "recent_autocorr":
            a = float(_lag1_autocorr_vec(ctx[None, :])[0])
            return float((a + 1.0) / 2.0)
        raise ValueError(f"unknown activation feature: {feat}")

    return activation, stats


# --------------------------------------------------------------------------- #
# 新模型：Conditional Expert MLP（臂 C）
# --------------------------------------------------------------------------- #
class ConditionalExpertMLPRetriever(nn.Module):
    """z = cond([CNN(x), ExpertVector(x)])；ExpertVector(x) = concat(a_i(x)·e_i)。无 attention。"""

    def __init__(self, context, expert_dim, n_experts, d_enc=128, d_exp=128, d_z=128,
                 hidden=32, beta_init=0.2):
        super().__init__()
        self.ts_encoder = attn.QueryEncoder(context, d_enc)          # [B,512]->[B,128]
        self.n_experts = n_experts
        self.expert_proj = nn.Linear(n_experts * expert_dim, d_exp)  # concat 后降维
        self.cond = nn.Sequential(nn.Linear(d_enc + d_exp, d_z), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_z, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.logit_gamma = nn.Parameter(torch.tensor(-2.0))
        beta_init = float(np.clip(beta_init, 1e-4, 1 - 1e-4))
        self.logit_beta = nn.Parameter(
            torch.tensor(np.log(beta_init / (1 - beta_init)), dtype=torch.float32))
        self.register_buffer("expert_emb_matrix", torch.zeros(n_experts, expert_dim))

    def set_expert_embeddings(self, emb_matrix):
        """emb_matrix: [N, E] 冻结的专家文本 embedding。"""
        self.expert_emb_matrix.copy_(torch.from_numpy(emb_matrix.astype(np.float32)))

    def expert_vector(self, x, act):
        """act: [B, N]。返回 ExpertVector [B, N*E]（投影前，供诊断）。"""
        weighted = act.unsqueeze(-1) * self.expert_emb_matrix.unsqueeze(0)  # [B, N, E]
        return weighted.reshape(len(x), -1)                                 # [B, N*E]

    def encode(self, x, act):
        q = self.ts_encoder(x)                                            # [B, d_enc]
        ev = self.expert_vector(x, act)                                   # [B, N*E]
        ev = self.expert_proj(ev)                                         # [B, d_exp]
        return self.cond(torch.cat([q, ev], dim=-1))                      # [B, d_z]

    def delta(self, z_q, z_c):
        B = z_q.shape[0]
        C = z_c.shape[0]
        qq = z_q[:, None, :].expand(B, C, -1)
        cc = z_c[None, :, :].expand(B, C, -1)
        return self.mlp(torch.cat([qq, cc, torch.abs(qq - cc)], dim=-1)).squeeze(-1)

    def score(self, shape, x_q, x_c, act_q, act_c):
        z_q = self.encode(x_q, act_q)
        z_c = self.encode(x_c, act_c)
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
# 训练 / 评估（conditional 专用）
# --------------------------------------------------------------------------- #
def train_conditional(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      bank_act, fit_act, device, epochs=30, batch_size=64,
                      candidates=256, lr=1e-3, temp=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
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
            act_q = torch.tensor(fit_act[ids], dtype=torch.float32, device=device)
            act_c = bank_act_t[cidx]
            shape = torch.tensor(X_fit_res[ids] @ X_bank_res[cidx].T / L,
                                 dtype=torch.float32, device=device)

            s = model.score(shape, x_q, x_c, act_q, act_c)
            w = torch.softmax(s / temp, dim=1)
            q_mu = X_fit[ids].mean(axis=1)
            c_mu = X_bank[cidx].mean(axis=1)
            transferred = Y_bank[cidx][None, :, :] + q_mu[:, None, None] - c_mu[None, :, None]
            pred = torch.sum(w.unsqueeze(-1) * torch.tensor(transferred, dtype=torch.float32, device=device), dim=1)
            y = torch.tensor(Y_fit[ids], dtype=torch.float32, device=device)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        gamma = float(torch.sigmoid(model.logit_gamma).detach().cpu())
        beta = float(torch.sigmoid(model.logit_beta).detach().cpu())
        print(f"epoch {ep:02d}/{epochs} loss={np.mean(losses):.6f} gamma={gamma:.4f} beta={beta:.4f}")
    return {"gamma": gamma, "beta": beta}


@torch.no_grad()
def retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                         val_act, bank_act, mode, topk, device, q_chunk=128, bank_chunk=4096):
    L = X_val.shape[1]
    model.eval()
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    # 预计算 bank 张量 + 编码（逐样本、eval 模式，位一致）。
    # 原实现在每个 query 块内都重新编码整个 bank（O(test*bank) 冗余），
    # 这里只在 query 循环外编码一次，按相同的 bank_chunk 分块保证逐位一致。
    X_bank_gpu = torch.tensor(X_bank, dtype=torch.float32, device=device)
    X_bank_res_gpu = torch.tensor(X_bank_res, dtype=torch.float32, device=device)
    z_bank = None
    if mode != "shape":
        z_chunks = []
        for bs in range(0, len(X_bank), bank_chunk):
            be = min(bs + bank_chunk, len(X_bank))
            z_chunks.append(model.encode(X_bank_gpu[bs:be], bank_act_t[bs:be]))
        z_bank = torch.cat(z_chunks, dim=0)

    preds = []
    for qs in range(0, len(X_val), q_chunk):
        qe = min(qs + q_chunk, len(X_val))
        nq = qe - qs
        best_scores = np.full((nq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nq, topk), -1, dtype=np.int64)

        x_q = torch.tensor(X_val[qs:qe], dtype=torch.float32, device=device)
        act_q = torch.tensor(val_act[qs:qe], dtype=torch.float32, device=device)
        res_q = torch.tensor(X_val_res[qs:qe], dtype=torch.float32, device=device)
        z_q = model.encode(x_q, act_q) if mode != "shape" else None

        for bs in range(0, len(X_bank), bank_chunk):
            be = min(bs + bank_chunk, len(X_bank))
            res_c = X_bank_res_gpu[bs:be]
            shape = (res_q @ res_c.T) / L
            if mode == "shape":
                s = shape
            else:
                z_c = z_bank[bs:be]
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
        pred = Y_bank[best_idx].mean(axis=1)
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/multidataset"))
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--d-enc", type=int, default=128)
    ap.add_argument("--d-exp", type=int, default=128)
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
    ap.add_argument("--audit", type=Path, default=Path("results/solar_agent_v4_audit.json"))
    args = ap.parse_args()

    attn.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- 读 v4 审计的 accepted experts ----
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    accepted = audit.get("accepted_experts", [])
    print(f"accepted experts: {len(accepted)}")
    for i, e in enumerate(accepted, 1):
        print(f"  [E{i}] {e['name']}  activation={e['activation_feature']}")

    # ---- 数据加载 ----
    ds = args.root / args.dataset
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
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
        print("\n[WARN] no accepted experts -> only A/B arms.")
    else:
        enc = SentenceTransformer(args.text_model, device=str(device))
        knowledge_texts = [e["knowledge"] for e in accepted]
        emb = enc.encode(knowledge_texts, normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")  # [N, 384]
        expert_dim = emb.shape[1]
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    expert_emb_mean = emb.mean(axis=0) if len(accepted) else np.zeros(1, dtype=np.float32)

    # ---- B. CNN-only MLP ----
    print("\n[Arm B] CNN-only MLP ...")
    cnn_only = ExpertEmbeddingMLPRetriever(
        context=args.context, expert_dim=expert_dim if len(accepted) else 1,
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

    fixed_mse = fixed_mae = cond_mse = cond_mae = None
    act_diag = None

    if len(accepted) > 0:
        # ---- C-前置：activation executor ----
        hourly_mean_OT = np.array(audit["evidence_executor"]["diurnal_seasonal"]["hourly_mean_OT"])
        monthly_mean_OT = {int(k): float(v) for k, v in
                           audit["evidence_executor"]["diurnal_seasonal"]["monthly_mean_OT"].items()}
        X_train_all = np.concatenate([X_bank, X_fit], axis=0)
        activation_fn, act_stats = build_activation_executor(X_train_all, hourly_mean_OT, monthly_mean_OT)

        bank_ts = [dates[s + args.context] for s in bank_starts]
        fit_ts = [dates[s + args.context] for s in fit_starts]
        val_ts = [dates[s + args.context] for s in val_starts]

        feat_names = [e["activation_feature"] for e in accepted]

        def compute_act(X, tss):
            M = np.zeros((len(X), len(feat_names)), dtype=np.float32)
            for i, feat in enumerate(feat_names):
                for r in range(len(X)):
                    M[r, i] = activation_fn(feat, X[r], tss[r])
            return M

        print("\nComputing activation vectors ...")
        bank_act = compute_act(X_bank, bank_ts)
        fit_act = compute_act(X_fit, fit_ts)
        val_act = compute_act(X_val, val_ts)

        # ---- B-fixed（旧版 mean-pool）----
        print("\n[Arm B-fixed] CNN + mean-pool fixed Expert ...")
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

        # ---- C. Conditional Expert MLP ----
        print("\n[Arm C] Conditional Expert MLP ...")
        conditional = ConditionalExpertMLPRetriever(
            context=args.context, expert_dim=expert_dim, n_experts=len(accepted),
            d_enc=args.d_enc, d_exp=args.d_exp, d_z=args.d_z, hidden=args.hidden,
        ).to(device)
        conditional.set_expert_embeddings(emb)
        train_conditional(
            conditional, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
            bank_act, fit_act, device=device, epochs=args.epochs,
            batch_size=args.batch_size, candidates=args.candidates, lr=args.lr)
        cond_pred = retrieve_conditional(
            conditional, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
            val_act, bank_act, "expert", args.topk, device)
        cond_mse = mse(Y_val, cond_pred)
        cond_mae = mae(Y_val, cond_pred)
        print(f"  CNN+Conditional Expert MSE={cond_mse:.2f} MAE={cond_mae:.2f}")

        # ---- activation 诊断（100 个均匀 VAL query）----
        print(f"\n[Activation Diagnostic] {args.n_diag} VAL queries ...")
        n_diag = min(args.n_diag, len(X_val))
        diag_idx = np.linspace(0, len(X_val) - 1, n_diag).astype(int)
        diag_act = val_act[diag_idx]   # [n_diag, N]

        per_activation = {}
        for j in range(len(accepted)):
            per_activation[accepted[j]["name"]] = {
                "activation_feature": feat_names[j],
                "mean": float(diag_act[:, j].mean()),
                "std": float(diag_act[:, j].std()),
                "min": float(diag_act[:, j].min()),
                "max": float(diag_act[:, j].max()),
            }

        # ExpertVector（投影前）pairwise 变化
        with torch.no_grad():
            x_diag = torch.tensor(X_val[diag_idx], dtype=torch.float32, device=device)
            act_diag_t = torch.tensor(diag_act, dtype=torch.float32, device=device)
            ev = conditional.expert_vector(x_diag, act_diag_t).cpu().numpy()  # [n_diag, N*E]
        ev_std = float(ev.std(axis=0).mean())          # ExpertVector 各维平均 std
        ev_pairwise_mean = float(np.mean([np.linalg.norm(ev[i] - ev[j])
                                          for i in range(0, n_diag, 5) for j in range(i + 1, min(i + 5, n_diag))]))
        dynamic = bool(ev_std > 1e-4)

        act_diag = {
            "n_queries": int(n_diag),
            "per_activation": per_activation,
            "expert_vector_mean_std_across_queries": ev_std,
            "expert_vector_pairwise_euclidean_mean": ev_pairwise_mean,
            "expert_vector_dynamic": dynamic,
        }
        print(f"  per-activation stats:")
        for name, s in per_activation.items():
            print(f"    {name} [{s['activation_feature']}]: mean={s['mean']:.4f} "
                  f"std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}")
        print(f"  ExpertVector mean per-dim std: {ev_std:.6f}")
        print(f"  ExpertVector pairwise euclidean mean: {ev_pairwise_mean:.6f}")
        print(f"  ExpertVector dynamic: {dynamic}")

    # ---- 结果 ----
    cond_vs_a = float((cnn_only_mse - cond_mse) / cnn_only_mse * 100.0) if cond_mse is not None else None
    cond_vs_fixed = float((fixed_mse - cond_mse) / fixed_mse * 100.0) if (fixed_mse is not None and cond_mse is not None) else None

    res = {
        "dataset": args.dataset,
        "protocol": "multidataset 9905-window VAL, residualize Pearson, Top-K=5 mean",
        "accepted_experts": accepted,
        "n_accepted_experts": len(accepted),
        "shape_only": {"mse": shape_mse, "mae": shape_mae},
        "cnn_only_mlp": {"mse": cnn_only_mse, "mae": cnn_only_mae},
        "fixed_expert_mlp": ({"mse": fixed_mse, "mae": fixed_mae} if fixed_mse is not None else None),
        "conditional_expert_mlp": ({"mse": cond_mse, "mae": cond_mae} if cond_mse is not None else None),
        "C_vs_A_improvement_pct": cond_vs_a,
        "C_vs_fixed_improvement_pct": cond_vs_fixed,
        "activation_diagnostic": act_diag,
        "baseline_shape_mse": SHAPE_BASELINE,
        "test_used": False,
    }

    out = ds / "conditional_expert_v4"
    out.mkdir(parents=True, exist_ok=True)

    save_models = {"cnn_only_state": cnn_only.state_dict()}
    save_preds = {"y_true": Y_val, "val_starts": val_starts, "shape_pred": shape_pred,
                  "cnn_only_pred": cnn_only_pred}
    if fixed_mse is not None:
        save_models["fixed_expert_state"] = cnn_fixed.state_dict()
        save_preds["fixed_expert_pred"] = fixed_pred
    if cond_mse is not None:
        save_models["conditional_state"] = conditional.state_dict()
        save_preds["conditional_pred"] = cond_pred

    torch.save(save_models, out / "models.pt")
    np.savez_compressed(out / "val_predictions.npz", **save_preds)
    (out / "result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "activation_diagnostic.json").write_text(
        json.dumps(act_diag if act_diag else {}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n" + "=" * 72)
    print("SOLAR VAL ABLATION (9905 windows, Top-K=5 mean)")
    print("=" * 72)
    print(f"A. Shape-only                 MSE = {shape_mse:.2f}  MAE = {shape_mae:.2f}")
    print(f"B. CNN-only MLP               MSE = {cnn_only_mse:.2f}  MAE = {cnn_only_mae:.2f}")
    if fixed_mse is not None:
        print(f"C-fixed. mean-pool Expert     MSE = {fixed_mse:.2f}  MAE = {fixed_mae:.2f}")
    if cond_mse is not None:
        print(f"C. Conditional Expert         MSE = {cond_mse:.2f}  MAE = {cond_mae:.2f}")
        print(f"   C vs A (CNN-only)          = {cond_vs_a:+.2f}%")
        print(f"   C vs fixed (mean-pool)     = {cond_vs_fixed:+.2f}%")
    print("=" * 72)
    print("Saved to:", out)
    print("TEST was NOT evaluated.")


if __name__ == "__main__":
    main()
