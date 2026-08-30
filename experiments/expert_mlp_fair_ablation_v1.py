#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fair Ablation V1: Original Shape Retriever + Expert-MLP Score Correction
======================================================================

目标
----
做一个“公平版本”：
    原来的 Shape Retriever 不动，
    只在它已经算好的 shape similarity score 上，加一个很小的 Expert-MLP 修正。

    score_new(q, i) = score_shape(q, i) + gamma * delta_score(q, i)

其中：
    delta_score(q, i) = MLP([e_q, e_i, |e_q-e_i|])

这样能保证：
1) 原始 Shape score 完全保留；
2) Top-K、softmax、future transfer、Chronos fusion 均可按原逻辑比较；
3) Expert 只负责“修正谁更值得被检索”，不会重新定义整个 Shape Retriever；
4) TRAIN_A 做 memory，TRAIN_B 训练 MLP，VAL 评估，TEST 完全不碰。

最推荐的输入方式
----------------
让你现有的“原始冻结 Retriever”额外导出两块 score matrix：

    train_b_shape_scores : [Nb, Na]
        TRAIN_B query 对 TRAIN_A memory 的原始 shape similarity score

    val_shape_scores : [Nv, Na+Nb]
        VAL query 对 TRAIN_A+TRAIN_B memory 的原始 shape similarity score

这样本脚本不重新实现 residual / Pearson，
而是直接使用你原管线真正算出来的 score —— 这是最公平的。

bundle 必需字段
--------------
train_a_context       [Na, L] 或 [Na,L,C]
train_a_future        [Na, H]
train_a_expert        [Na, M]

train_b_context       [Nb, L] 或 [Nb,L,C]
train_b_future        [Nb, H]
train_b_expert        [Nb, M]

val_context           [Nv, L] 或 [Nv,L,C]
val_future            [Nv, H]
val_expert            [Nv, M]

train_b_shape_scores  [Nb, Na]
val_shape_scores      [Nv, Na+Nb]

可选：
chronos_val_pred      [Nv,H]

专家特征建议仍使用你 V0 的 6 维：
hour_sin, hour_cos, doy_sin, doy_cos, recent_level, recent_change

运行示例
--------
python experiments/expert_mlp_fair_ablation_v1.py \
  --bundle results/expert_mlp_retrieval_v1/solar_fair_bundle.npz \
  --out-dir results/expert_mlp_retrieval_v1 \
  --epochs 30 \
  --topk 5 \
  --softmax-temperature 0.10

说明
----
这个脚本故意“不自己重算 Shape similarity”。
原因：只有直接复用原冻结管线输出的 score，才能保证 baseline 真正一致。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def improve(old, new):
    if old == 0:
        return float("nan")
    return float((old - new) / old * 100.0)


# ---------------------------------------------------------------------
# Expert scaler
# ---------------------------------------------------------------------
def fit_scaler(*xs):
    x = np.concatenate(xs, axis=0).astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def apply_scaler(x, mu, sd):
    return ((x.astype(np.float32) - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------------
# Original transfer: mean-anchor
# ---------------------------------------------------------------------
def target_series(ctx):
    ctx = np.asarray(ctx, dtype=np.float32)
    if ctx.ndim == 2:
        return ctx
    if ctx.ndim == 3:
        # 默认最后一个通道是预测目标
        return ctx[..., -1]
    raise ValueError(f"Unsupported context shape: {ctx.shape}")


def context_mean(ctx):
    return target_series(ctx).mean(axis=1)


# ---------------------------------------------------------------------
# Expert MLP: candidate-wise score correction
# ---------------------------------------------------------------------
class ExpertScoreMLP(nn.Module):
    """
    输入每个 query-candidate 对的专家状态：
        [e_q, e_i, abs(e_q-e_i)]

    输出一个 scalar delta_score。
    """
    def __init__(self, expert_dim: int, hidden: int = 32):
        super().__init__()
        in_dim = expert_dim * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # 初始尽量接近原始 Shape Retriever
        self.logit_gamma = nn.Parameter(torch.tensor(-2.0))

    def pair_delta(self, q_e, c_e):
        """
        q_e: [B,M]
        c_e: [K,M]

        return: [B,K]
        """
        B, M = q_e.shape
        K = c_e.shape[0]

        q = q_e[:, None, :].expand(B, K, M)
        c = c_e[None, :, :].expand(B, K, M)
        x = torch.cat([q, c, torch.abs(q - c)], dim=-1)

        delta = self.net(x).squeeze(-1)
        return delta

    def corrected_score(self, shape_score, q_e, c_e):
        delta = self.pair_delta(q_e, c_e)
        gamma = torch.sigmoid(self.logit_gamma)
        return shape_score + gamma * delta, gamma


# ---------------------------------------------------------------------
# Differentiable training on TRAIN_B
# ---------------------------------------------------------------------
def train_model(
    model,
    train_b_shape_scores,
    train_a_expert,
    train_b_expert,
    train_a_context,
    train_a_future,
    train_b_context,
    train_b_future,
    epochs,
    batch_size,
    candidate_size,
    lr,
    softmax_temperature,
    device,
    seed,
):
    """
    TRAIN_B query -> TRAIN_A memory

    为了可微训练，不在训练阶段硬 Top-K。
    每个 batch 从 TRAIN_A 抽一组候选，使用 softmax(score) 加权 future，
    用最终 forecast MSE 来训练 Expert-MLP。

    VAL 推理时恢复硬 Top-K + softmax，和正式 Retriever 逻辑一致。
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_q = train_b_shape_scores.shape[0]
    n_mem = train_b_shape_scores.shape[1]

    train_a_e_t = torch.from_numpy(train_a_expert).to(device)
    train_b_e_t = torch.from_numpy(train_b_expert).to(device)

    hist = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_q)
        losses = []

        for st in range(0, n_q, batch_size):
            q_idx = order[st:st + batch_size]
            if len(q_idx) == 0:
                continue

            k = min(candidate_size, n_mem)
            c_idx = rng.choice(n_mem, size=k, replace=False)

            base = torch.from_numpy(
                train_b_shape_scores[np.ix_(q_idx, c_idx)].astype(np.float32)
            ).to(device)

            q_e = train_b_e_t[q_idx]
            c_e = train_a_e_t[c_idx]

            score, gamma = model.corrected_score(base, q_e, c_e)
            weights = torch.softmax(score / softmax_temperature, dim=1)

            # Mean-anchor transfer
            q_mu = context_mean(train_b_context[q_idx]).astype(np.float32)
            c_mu = context_mean(train_a_context[c_idx]).astype(np.float32)

            # [B,K,H]
            transferred = (
                train_a_future[c_idx][None, :, :].astype(np.float32)
                + q_mu[:, None, None]
                - c_mu[None, :, None]
            )

            future_t = torch.from_numpy(transferred).to(device)
            pred = torch.sum(weights.unsqueeze(-1) * future_t, dim=1)

            y = torch.from_numpy(train_b_future[q_idx].astype(np.float32)).to(device)

            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "gamma": float(torch.sigmoid(model.logit_gamma).detach().cpu()),
        }
        hist.append(row)

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={row['loss']:.6f} "
                f"gamma={row['gamma']:.4f}"
            )

    return hist


# ---------------------------------------------------------------------
# Exact evaluation: Top-K + softmax
# ---------------------------------------------------------------------
def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def retrieve_from_scores(
    score_matrix,
    memory_context,
    memory_future,
    query_context,
    topk,
    softmax_temperature,
    batch_size,
):
    """
    使用已经给定的 score matrix 做：
        Top-K -> softmax -> mean-anchor future aggregation
    """
    N, Kall = score_matrix.shape
    H = memory_future.shape[1]
    out = np.empty((N, H), dtype=np.float32)

    t0 = time.perf_counter()

    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        s = score_matrix[st:ed]
        k = min(topk, Kall)

        idx = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]

        # 按 score 重新排序，便于审计
        row = np.arange(ed - st)[:, None]
        selected_scores = s[row, idx]
        order = np.argsort(-selected_scores, axis=1)
        idx = idx[row, order]
        selected_scores = selected_scores[row, order]

        w = softmax_np(selected_scores / softmax_temperature, axis=1)

        q_mu = context_mean(query_context[st:ed])

        for r in range(ed - st):
            ids = idx[r]
            c_mu = context_mean(memory_context[ids])
            transferred = (
                memory_future[ids].astype(np.float32)
                + q_mu[r]
                - c_mu[:, None]
            )
            out[st + r] = np.sum(w[r, :, None] * transferred, axis=0)

    ms_per_query = (time.perf_counter() - t0) / N * 1000.0
    return out, ms_per_query


@torch.no_grad()
def corrected_score_matrix(
    model,
    base_scores,
    query_expert,
    memory_expert,
    query_batch,
    candidate_batch,
    device,
):
    """
    分块计算完整 VAL corrected score matrix，避免一次性构造 [N,K,M] 爆显存。
    """
    model.eval()

    N, K = base_scores.shape
    out = np.empty((N, K), dtype=np.float32)

    q_e_all = torch.from_numpy(query_expert.astype(np.float32))
    c_e_all = torch.from_numpy(memory_expert.astype(np.float32))

    t0 = time.perf_counter()

    for qs in range(0, N, query_batch):
        qe = min(qs + query_batch, N)
        q_e = q_e_all[qs:qe].to(device)

        for cs in range(0, K, candidate_batch):
            ce = min(cs + candidate_batch, K)

            c_e = c_e_all[cs:ce].to(device)
            base = torch.from_numpy(
                base_scores[qs:qe, cs:ce].astype(np.float32)
            ).to(device)

            s, gamma = model.corrected_score(base, q_e, c_e)
            out[qs:qe, cs:ce] = s.cpu().numpy()

    elapsed = time.perf_counter() - t0
    gamma = float(torch.sigmoid(model.logit_gamma).cpu())
    return out, gamma, elapsed / N * 1000.0


# ---------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------
def best_lambda(y, chronos, memory, lambdas):
    rows = []
    for lam in lambdas:
        pred = (1.0 - lam) * chronos + lam * memory
        rows.append({
            "lambda": float(lam),
            "mse": mse(y, pred),
            "mae": mae(y, pred),
        })
    return rows, min(rows, key=lambda x: (x["mse"], x["lambda"]))


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def check_bundle(a):
    required = [
        "train_a_context",
        "train_a_future",
        "train_a_expert",
        "train_b_context",
        "train_b_future",
        "train_b_expert",
        "val_context",
        "val_future",
        "val_expert",
        "train_b_shape_scores",
        "val_shape_scores",
    ]

    missing = [k for k in required if k not in a]
    if missing:
        raise KeyError(
            "bundle 缺少字段:\n  - " + "\n  - ".join(missing)
        )

    Na = len(a["train_a_context"])
    Nb = len(a["train_b_context"])
    Nv = len(a["val_context"])

    if a["train_b_shape_scores"].shape != (Nb, Na):
        raise ValueError(
            "train_b_shape_scores 应为 [Nb, Na] = "
            f"[{Nb},{Na}]，实际 {a['train_b_shape_scores'].shape}"
        )

    if a["val_shape_scores"].shape != (Nv, Na + Nb):
        raise ValueError(
            "val_shape_scores 应为 [Nv, Na+Nb] = "
            f"[{Nv},{Na+Nb}]，实际 {a['val_shape_scores'].shape}"
        )

    Ma = a["train_a_expert"].shape[1]
    Mb = a["train_b_expert"].shape[1]
    Mv = a["val_expert"].shape[1]

    if not (Ma == Mb == Mv):
        raise ValueError(
            f"专家维度不一致: TRAIN_A={Ma}, TRAIN_B={Mb}, VAL={Mv}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/expert_mlp_retrieval_v1"),
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--candidate-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--softmax-temperature", type=float, default=0.10)

    p.add_argument("--score-query-batch", type=int, default=32)
    p.add_argument("--score-candidate-batch", type=int, default=2048)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    # 扩到 0.5，只作 VAL 诊断
    p.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    )

    args = p.parse_args()

    if "test" in str(args.bundle).lower():
        raise RuntimeError(
            "拒绝运行：当前实验严禁 TEST，bundle 文件名不得包含 test。"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("device:", device)

    with np.load(args.bundle, allow_pickle=False) as f:
        A = {k: f[k] for k in f.files}

    check_bundle(A)

    # -----------------------------
    # Split
    # -----------------------------
    a_ctx = A["train_a_context"]
    a_fut = A["train_a_future"].astype(np.float32)
    a_exp = A["train_a_expert"]

    b_ctx = A["train_b_context"]
    b_fut = A["train_b_future"].astype(np.float32)
    b_exp = A["train_b_expert"]

    v_ctx = A["val_context"]
    v_fut = A["val_future"].astype(np.float32)
    v_exp = A["val_expert"]

    b_shape_score = A["train_b_shape_scores"].astype(np.float32)
    v_shape_score = A["val_shape_scores"].astype(np.float32)

    # -----------------------------
    # Expert scaler：只 TRAIN
    # -----------------------------
    e_mu, e_sd = fit_scaler(a_exp, b_exp)
    a_exp_z = apply_scaler(a_exp, e_mu, e_sd)
    b_exp_z = apply_scaler(b_exp, e_mu, e_sd)
    v_exp_z = apply_scaler(v_exp, e_mu, e_sd)

    # -----------------------------
    # Model
    # -----------------------------
    model = ExpertScoreMLP(
        expert_dim=a_exp_z.shape[1],
        hidden=args.hidden,
    ).to(device)

    print("=" * 78)
    print("FAIR ABLATION V1")
    print("Original Shape score + Expert-MLP correction")
    print("=" * 78)

    history = train_model(
        model=model,
        train_b_shape_scores=b_shape_score,
        train_a_expert=a_exp_z,
        train_b_expert=b_exp_z,
        train_a_context=a_ctx,
        train_a_future=a_fut,
        train_b_context=b_ctx,
        train_b_future=b_fut,
        epochs=args.epochs,
        batch_size=args.batch_size,
        candidate_size=args.candidate_size,
        lr=args.lr,
        softmax_temperature=args.softmax_temperature,
        device=device,
        seed=args.seed,
    )

    # VAL memory = TRAIN_A + TRAIN_B
    mem_ctx = np.concatenate([a_ctx, b_ctx], axis=0)
    mem_fut = np.concatenate([a_fut, b_fut], axis=0)
    mem_exp_z = np.concatenate([a_exp_z, b_exp_z], axis=0)

    # -----------------------------
    # A) 原始 Shape Retriever
    # -----------------------------
    original_pred, original_ms = retrieve_from_scores(
        score_matrix=v_shape_score,
        memory_context=mem_ctx,
        memory_future=mem_fut,
        query_context=v_ctx,
        topk=args.topk,
        softmax_temperature=args.softmax_temperature,
        batch_size=args.batch_size,
    )

    # -----------------------------
    # B) 原始 Shape + Expert-MLP
    # -----------------------------
    corrected_scores, gamma, correction_ms = corrected_score_matrix(
        model=model,
        base_scores=v_shape_score,
        query_expert=v_exp_z,
        memory_expert=mem_exp_z,
        query_batch=args.score_query_batch,
        candidate_batch=args.score_candidate_batch,
        device=device,
    )

    expert_pred, expert_retrieval_ms = retrieve_from_scores(
        score_matrix=corrected_scores,
        memory_context=mem_ctx,
        memory_future=mem_fut,
        query_context=v_ctx,
        topk=args.topk,
        softmax_temperature=args.softmax_temperature,
        batch_size=args.batch_size,
    )

    original_mse = mse(v_fut, original_pred)
    original_mae = mae(v_fut, original_pred)

    expert_mse = mse(v_fut, expert_pred)
    expert_mae = mae(v_fut, expert_pred)

    # -----------------------------
    # Top-K overlap diagnostic
    # -----------------------------
    k = min(args.topk, v_shape_score.shape[1])

    orig_idx = np.argpartition(
        -v_shape_score, kth=k - 1, axis=1
    )[:, :k]

    new_idx = np.argpartition(
        -corrected_scores, kth=k - 1, axis=1
    )[:, :k]

    overlaps = []
    for i in range(len(orig_idx)):
        a = set(orig_idx[i].tolist())
        b = set(new_idx[i].tolist())
        overlaps.append(len(a & b) / k)

    mean_topk_overlap = float(np.mean(overlaps))

    # -----------------------------
    # Report
    # -----------------------------
    report = {
        "experiment": "expert_mlp_fair_ablation_v1",
        "test_accessed": False,
        "policy": {
            "shape_score": "loaded directly from frozen original retriever",
            "train_queries": "TRAIN_B",
            "train_memory": "TRAIN_A",
            "val_queries": "VAL",
            "val_memory": "TRAIN_A+TRAIN_B",
            "expert_scaler_fit": "TRAIN_A+TRAIN_B only",
        },
        "model": {
            "formula": "score_new = score_shape + gamma * MLP([e_q,e_i,abs(e_q-e_i)])",
            "expert_dim": int(a_exp_z.shape[1]),
            "hidden": args.hidden,
            "learned_gamma": gamma,
        },
        "memory_only": {
            "original_shape": {
                "mse": original_mse,
                "mae": original_mae,
                "retrieval_ms_per_query": original_ms,
            },
            "shape_plus_expert_mlp": {
                "mse": expert_mse,
                "mae": expert_mae,
                "score_correction_ms_per_query": correction_ms,
                "retrieval_ms_per_query": expert_retrieval_ms,
                "total_ms_per_query": correction_ms + expert_retrieval_ms,
            },
            "improvement_vs_original_shape_pct": improve(
                original_mse, expert_mse
            ),
        },
        "retrieval_diagnostics": {
            "mean_topk_overlap_original_vs_expert": mean_topk_overlap,
            "mean_topk_changed_fraction": 1.0 - mean_topk_overlap,
        },
        "training_history": history,
    }

    # -----------------------------
    # Chronos fusion
    # -----------------------------
    if "chronos_val_pred" in A:
        chronos = A["chronos_val_pred"].astype(np.float32)

        if chronos.shape != v_fut.shape:
            raise ValueError(
                f"chronos_val_pred {chronos.shape} != val_future {v_fut.shape}"
            )

        c_mse = mse(v_fut, chronos)
        c_mae = mae(v_fut, chronos)

        original_grid, original_best = best_lambda(
            v_fut, chronos, original_pred, args.lambdas
        )

        expert_grid, expert_best = best_lambda(
            v_fut, chronos, expert_pred, args.lambdas
        )

        same_lambda = []
        for lam in args.lambdas:
            p0 = (1 - lam) * chronos + lam * original_pred
            p1 = (1 - lam) * chronos + lam * expert_pred

            m0 = mse(v_fut, p0)
            m1 = mse(v_fut, p1)

            same_lambda.append({
                "lambda": float(lam),
                "original_shape_mse": m0,
                "expert_mlp_mse": m1,
                "expert_vs_shape_improvement_pct": improve(m0, m1),
            })

        report["chronos_fusion"] = {
            "chronos": {
                "mse": c_mse,
                "mae": c_mae,
            },
            "original_shape": {
                "grid": original_grid,
                "best": original_best,
                "improvement_vs_chronos_pct": improve(
                    c_mse, original_best["mse"]
                ),
            },
            "shape_plus_expert_mlp": {
                "grid": expert_grid,
                "best": expert_best,
                "improvement_vs_chronos_pct": improve(
                    c_mse, expert_best["mse"]
                ),
                "improvement_vs_best_original_shape_pct": improve(
                    original_best["mse"], expert_best["mse"]
                ),
            },
            "same_lambda_ablation": same_lambda,
        }

    # -----------------------------
    # Save
    # -----------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "expert_mu": e_mu,
        "expert_sd": e_sd,
        "expert_dim": a_exp_z.shape[1],
        "hidden": args.hidden,
    }, args.out_dir / "model.pt")

    np.savez_compressed(
        args.out_dir / "val_predictions.npz",
        y_true=v_fut,
        original_shape_memory_pred=original_pred,
        expert_mlp_memory_pred=expert_pred,
    )

    # score matrix 可能很大，不默认保存 corrected_scores
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = []
    md.append("# Fair Ablation V1 — Original Shape + Expert-MLP")
    md.append("")
    md.append("## Memory-only")
    md.append("")
    md.append("| Method | MSE | MAE |")
    md.append("|---|---:|---:|")
    md.append(
        f"| Original Shape | {original_mse:.6f} | {original_mae:.6f} |"
    )
    md.append(
        f"| Original Shape + Expert-MLP | {expert_mse:.6f} | {expert_mae:.6f} |"
    )
    md.append("")
    md.append(
        f"Expert-MLP vs Original Shape: "
        f"**{improve(original_mse, expert_mse):+.4f}%**"
    )
    md.append("")
    md.append(f"Learned gamma: **{gamma:.4f}**")
    md.append(
        f"Mean Top-K overlap: **{mean_topk_overlap*100:.2f}%**"
    )
    md.append(
        f"Expert-MLP total retrieval overhead: "
        f"**{correction_ms + expert_retrieval_ms:.3f} ms/query**"
    )

    if "chronos_fusion" in report:
        cf = report["chronos_fusion"]

        md.append("")
        md.append("## Chronos Fusion")
        md.append("")
        md.append(
            f"Chronos MSE: **{cf['chronos']['mse']:.6f}**"
        )
        md.append(
            f"Best Chronos + Original Shape: "
            f"**{cf['original_shape']['best']['mse']:.6f}**, "
            f"lambda={cf['original_shape']['best']['lambda']}"
        )
        md.append(
            f"Best Chronos + Expert-MLP: "
            f"**{cf['shape_plus_expert_mlp']['best']['mse']:.6f}**, "
            f"lambda={cf['shape_plus_expert_mlp']['best']['lambda']}"
        )
        md.append(
            f"Expert-MLP vs best Original Shape fusion: "
            f"**{cf['shape_plus_expert_mlp']['improvement_vs_best_original_shape_pct']:+.4f}%**"
        )

        md.append("")
        md.append("## Same-lambda Ablation")
        md.append("")
        md.append(
            "| lambda | Original Shape | Expert-MLP | Expert vs Shape |"
        )
        md.append("|---:|---:|---:|---:|")

        for r in cf["same_lambda_ablation"]:
            md.append(
                f"| {r['lambda']:.2f} | "
                f"{r['original_shape_mse']:.6f} | "
                f"{r['expert_mlp_mse']:.6f} | "
                f"{r['expert_vs_shape_improvement_pct']:+.4f}% |"
            )

    (args.out_dir / "report.md").write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("VAL RESULTS")
    print("=" * 78)
    print(
        f"Original Shape Memory : "
        f"MSE={original_mse:.6f} MAE={original_mae:.6f}"
    )
    print(
        f"+ Expert-MLP          : "
        f"MSE={expert_mse:.6f} MAE={expert_mae:.6f}"
    )
    print(
        f"Memory improvement    : "
        f"{improve(original_mse, expert_mse):+.4f}%"
    )
    print(f"Learned gamma         : {gamma:.4f}")
    print(f"Top-K overlap         : {mean_topk_overlap*100:.2f}%")
    print(
        f"Expert retrieval time : "
        f"{correction_ms + expert_retrieval_ms:.3f} ms/query"
    )

    if "chronos_fusion" in report:
        cf = report["chronos_fusion"]
        print("-" * 78)
        print(
            f"Chronos               : "
            f"{cf['chronos']['mse']:.6f}"
        )
        print(
            f"Best + Original Shape : "
            f"{cf['original_shape']['best']['mse']:.6f} "
            f"(lambda={cf['original_shape']['best']['lambda']})"
        )
        print(
            f"Best + Expert-MLP     : "
            f"{cf['shape_plus_expert_mlp']['best']['mse']:.6f} "
            f"(lambda={cf['shape_plus_expert_mlp']['best']['lambda']})"
        )
        print(
            f"Expert vs Shape fused : "
            f"{cf['shape_plus_expert_mlp']['improvement_vs_best_original_shape_pct']:+.4f}%"
        )

    print("-" * 78)
    print("outputs:", args.out_dir)
    print("=" * 78)


if __name__ == "__main__":
    main()
