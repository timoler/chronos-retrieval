#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-Conditioned MLP Retriever — VAL-only 独立实验

目的
----
只测试我们刚讨论的新检索器：
    q' = normalize(q + gamma * MLP([q ; e]))

其中：
- q：当前时间序列的 shape 表示（这里直接用标准化后的 context 向量）
- e：多条专家意见对应的结构化特征向量
- MLP：学习“专家信息应该怎样修正 shape query”

重要：
1. 不修改原有 Shape / Expert / Gate 脚本。
2. 所有输出写到 results/expert_mlp_retrieval_v0/。
3. TEST 完全禁止读取；只允许 TRAIN_A / TRAIN_B / VAL。
4. TRAIN_A 作为 memory bank；TRAIN_B 用来训练 MLP；VAL 只做模型选择和性能评估。
5. Chronos 若已经有 VAL 预测，可一起比较最终融合；没有也能先比较 Memory-only。

输入 NPZ 至少包含：
    train_a_context   [Na, L] 或 [Na, L, C]
    train_a_future    [Na, H]
    train_a_expert    [Na, M]

    train_b_context   [Nb, L] 或 [Nb, L, C]
    train_b_future    [Nb, H]
    train_b_expert    [Nb, M]

    val_context       [Nv, L] 或 [Nv, L, C]
    val_future        [Nv, H]
    val_expert        [Nv, M]

可选：
    chronos_val_pred  [Nv, H]

这里 M = 专家意见/机制特征数量。
例如：
    第0列 = 工作日/周末
    第1列 = 季节状态
    第2列 = 高/低负荷状态
    第3列 = 同时刻周期状态

注意：这版先做“性能验证 V0”。它要求训练和 VAL 使用相同的 M 个专家特征。
以后若要做到“随时新增专家意见而不改维度”，再把 e 改成可变长 expert-token attention。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 基础指标
# -----------------------------
def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def improve(old, new):
    return float((old - new) / old * 100.0) if old != 0 else float("nan")


# -----------------------------
# 数据预处理
# -----------------------------
def flatten_context(x: np.ndarray) -> np.ndarray:
    """[N,L] 或 [N,L,C] -> [N,D]"""
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def shape_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    每个窗口独立去均值、除标准差。
    对单变量窗口时，点积/余弦与 Pearson shape similarity 很接近。
    """
    x = flatten_context(x)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    z = (x - mu) / (sd + eps)
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / (norm + eps)


def fit_expert_scaler(*xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(xs, axis=0).astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def scale_expert(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mu) / sd).astype(np.float32)


# -----------------------------
# Mean-anchor future transfer
# -----------------------------
def context_target_mean(ctx: np.ndarray) -> np.ndarray:
    """
    如果 context 是 [N,L,C]，默认最后一个通道是目标变量；
    如果 [N,L]，直接用该序列。
    """
    ctx = np.asarray(ctx, dtype=np.float32)
    if ctx.ndim == 3:
        target = ctx[..., -1]
    elif ctx.ndim == 2:
        target = ctx
    else:
        raise ValueError(f"context shape 不支持: {ctx.shape}")
    return target.mean(axis=1)


def transfer_future_mean_anchor(
    query_ctx: np.ndarray,
    cand_ctx: np.ndarray,
    cand_future: np.ndarray,
) -> np.ndarray:
    """批量 mean-anchor：candidate future 平移到 query 当前 level。"""
    q_mean = context_target_mean(query_ctx)[:, None]
    c_mean = context_target_mean(cand_ctx)[:, None]
    return cand_future + (q_mean - c_mean)


# -----------------------------
# 模型：q' = q + gamma * MLP([q;e])
# -----------------------------
class ExpertFusionRetriever(nn.Module):
    def __init__(self, shape_dim: int, expert_dim: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(shape_dim + expert_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, shape_dim),
        )
        # 初始 gamma 较小：先接近纯 Shape，再逐渐学习专家修正
        self.logit_gamma = nn.Parameter(torch.tensor(-2.0))

    def forward(self, shape_vec: torch.Tensor, expert_vec: torch.Tensor):
        delta = self.mlp(torch.cat([shape_vec, expert_vec], dim=-1))
        gamma = torch.sigmoid(self.logit_gamma)   # 0~1
        out = shape_vec + gamma * delta
        out = F.normalize(out, dim=-1)
        return out, gamma


# -----------------------------
# 训练
# -----------------------------
def train_model(
    model: nn.Module,
    a_shape: torch.Tensor,
    a_expert: torch.Tensor,
    a_ctx_np: np.ndarray,
    a_future_np: np.ndarray,
    b_shape: torch.Tensor,
    b_expert: torch.Tensor,
    b_ctx_np: np.ndarray,
    b_future_np: np.ndarray,
    epochs: int,
    batch_size: int,
    candidate_size: int,
    lr: float,
    temperature: float,
    device: torch.device,
    seed: int,
):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_b = len(b_shape)
    n_a = len(a_shape)
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_b)
        losses = []

        for st in range(0, n_b, batch_size):
            q_idx = order[st: st + batch_size]
            if len(q_idx) == 0:
                continue

            # 每个 batch 共用一组候选，降低计算量
            cnum = min(candidate_size, n_a)
            c_idx = rng.choice(n_a, size=cnum, replace=False)

            q_shape = b_shape[q_idx].to(device)
            q_exp = b_expert[q_idx].to(device)
            c_shape = a_shape[c_idx].to(device)
            c_exp = a_expert[c_idx].to(device)

            q_emb, gamma = model(q_shape, q_exp)       # [B,D]
            c_emb, _ = model(c_shape, c_exp)           # [K,D]

            scores = (q_emb @ c_emb.T) / temperature   # [B,K]
            weights = torch.softmax(scores, dim=-1)

            # candidate future 做 mean-anchor transfer
            q_ctx = b_ctx_np[q_idx]
            q_mean = context_target_mean(q_ctx)[:, None, None]       # [B,1,1]
            c_mean = context_target_mean(a_ctx_np[c_idx])[None, :, None]  # [1,K,1]
            c_future = a_future_np[c_idx][None, :, :]                 # [1,K,H]
            transferred = c_future + (q_mean - c_mean)                # [B,K,H]

            tfut = torch.from_numpy(transferred.astype(np.float32)).to(device)
            pred = torch.sum(weights.unsqueeze(-1) * tfut, dim=1)    # [B,H]
            true = torch.from_numpy(b_future_np[q_idx].astype(np.float32)).to(device)

            loss = F.mse_loss(pred, true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss, "gamma": float(torch.sigmoid(model.logit_gamma).detach().cpu())})
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} loss={mean_loss:.6f} gamma={history[-1]['gamma']:.4f}")

    return history


# -----------------------------
# 检索预测
# -----------------------------
@torch.no_grad()
def retrieve_predict(
    model: nn.Module,
    bank_shape: torch.Tensor,
    bank_expert: torch.Tensor,
    bank_ctx_np: np.ndarray,
    bank_future_np: np.ndarray,
    query_shape: torch.Tensor,
    query_expert: torch.Tensor,
    query_ctx_np: np.ndarray,
    topk: int,
    batch_size: int,
    device: torch.device,
):
    model.eval()

    bank_emb, gamma = model(bank_shape.to(device), bank_expert.to(device))
    preds = []
    top_scores_all = []

    t0 = time.perf_counter()
    for st in range(0, len(query_shape), batch_size):
        ed = min(st + batch_size, len(query_shape))
        q_emb, _ = model(query_shape[st:ed].to(device), query_expert[st:ed].to(device))
        scores = q_emb @ bank_emb.T
        vals, idx = torch.topk(scores, k=min(topk, bank_emb.shape[0]), dim=1)

        idx_np = idx.cpu().numpy()
        qctx = query_ctx_np[st:ed]
        batch_pred = []
        for r in range(len(idx_np)):
            ids = idx_np[r]
            q_mean = context_target_mean(qctx[r:r+1])[0]
            c_mean = context_target_mean(bank_ctx_np[ids])
            fut = bank_future_np[ids] + (q_mean - c_mean)[:, None]
            batch_pred.append(fut.mean(axis=0))
        preds.append(np.stack(batch_pred, axis=0))
        top_scores_all.append(vals.cpu().numpy())

    elapsed = time.perf_counter() - t0
    pred = np.concatenate(preds, axis=0)
    top_scores = np.concatenate(top_scores_all, axis=0)
    ms_per_query = elapsed / len(query_shape) * 1000.0
    return pred, top_scores, float(gamma.cpu()), ms_per_query


def pure_shape_predict(
    bank_shape_np: np.ndarray,
    bank_ctx_np: np.ndarray,
    bank_future_np: np.ndarray,
    query_shape_np: np.ndarray,
    query_ctx_np: np.ndarray,
    topk: int,
    batch_size: int,
):
    preds = []
    t0 = time.perf_counter()
    for st in range(0, len(query_shape_np), batch_size):
        ed = min(st + batch_size, len(query_shape_np))
        scores = query_shape_np[st:ed] @ bank_shape_np.T
        k = min(topk, bank_shape_np.shape[0])
        idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]

        batch_pred = []
        for r in range(len(idx)):
            ids = idx[r]
            q_mean = context_target_mean(query_ctx_np[st+r:st+r+1])[0]
            c_mean = context_target_mean(bank_ctx_np[ids])
            fut = bank_future_np[ids] + (q_mean - c_mean)[:, None]
            batch_pred.append(fut.mean(axis=0))
        preds.append(np.stack(batch_pred, axis=0))
    elapsed = time.perf_counter() - t0
    return np.concatenate(preds, axis=0), elapsed / len(query_shape_np) * 1000.0


def best_lambda(y, chronos, memory, lambdas: List[float]):
    rows = []
    for lam in lambdas:
        pred = (1-lam)*chronos + lam*memory
        rows.append({"lambda": float(lam), "mse": mse(y, pred), "mae": mae(y, pred)})
    return rows, min(rows, key=lambda r: (r["mse"], r["lambda"]))


def require(arrays: Dict[str, np.ndarray], names: List[str]):
    miss = [x for x in names if x not in arrays]
    if miss:
        raise KeyError(f"NPZ 缺少字段: {miss}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/expert_mlp_retrieval_v0"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0,0.025,0.05,0.075,0.10,0.15,0.20,0.30])
    args = parser.parse_args()

    # TEST 防误读
    if "test" in str(args.bundle).lower():
        raise RuntimeError("拒绝运行：当前实验只允许 TRAIN_A / TRAIN_B / VAL，bundle 文件名不得包含 test。")

    with np.load(args.bundle, allow_pickle=False) as f:
        A = {k: f[k] for k in f.files}

    req = [
        "train_a_context","train_a_future","train_a_expert",
        "train_b_context","train_b_future","train_b_expert",
        "val_context","val_future","val_expert",
    ]
    require(A, req)

    a_ctx, a_fut, a_exp = A["train_a_context"], A["train_a_future"], A["train_a_expert"]
    b_ctx, b_fut, b_exp = A["train_b_context"], A["train_b_future"], A["train_b_expert"]
    v_ctx, v_fut, v_exp = A["val_context"], A["val_future"], A["val_expert"]

    if a_fut.ndim != 2 or b_fut.ndim != 2 or v_fut.ndim != 2:
        raise ValueError("当前 V0 要求 future 为 [N,H]。")
    if a_exp.ndim != 2 or b_exp.ndim != 2 or v_exp.ndim != 2:
        raise ValueError("expert 必须为 [N,M]，每一列对应一条专家机制特征。")
    if not (a_exp.shape[1] == b_exp.shape[1] == v_exp.shape[1]):
        raise ValueError("TRAIN_A / TRAIN_B / VAL 的专家特征数 M 必须一致。")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)

    # Shape 表示
    a_shape_np = shape_normalize(a_ctx)
    b_shape_np = shape_normalize(b_ctx)
    v_shape_np = shape_normalize(v_ctx)

    # 专家特征只用 TRAIN_A + TRAIN_B 拟合标准化，VAL 不参与
    e_mu, e_sd = fit_expert_scaler(a_exp, b_exp)
    a_exp_np = scale_expert(a_exp, e_mu, e_sd)
    b_exp_np = scale_expert(b_exp, e_mu, e_sd)
    v_exp_np = scale_expert(v_exp, e_mu, e_sd)

    a_shape = torch.from_numpy(a_shape_np)
    b_shape = torch.from_numpy(b_shape_np)
    v_shape = torch.from_numpy(v_shape_np)
    a_exp_t = torch.from_numpy(a_exp_np)
    b_exp_t = torch.from_numpy(b_exp_np)
    v_exp_t = torch.from_numpy(v_exp_np)

    model = ExpertFusionRetriever(
        shape_dim=a_shape_np.shape[1],
        expert_dim=a_exp_np.shape[1],
        hidden=args.hidden,
    ).to(device)

    print("="*72)
    print("Train Expert-Conditioned MLP Retriever")
    print(f"shape_dim={a_shape_np.shape[1]}, expert_dim={a_exp_np.shape[1]}, hidden={args.hidden}")
    print("="*72)

    history = train_model(
        model,
        a_shape,a_exp_t,a_ctx,a_fut,
        b_shape,b_exp_t,b_ctx,b_fut,
        epochs=args.epochs,
        batch_size=args.batch_size,
        candidate_size=args.candidate_size,
        lr=args.lr,
        temperature=args.temperature,
        device=device,
        seed=args.seed,
    )

    # VAL memory bank = TRAIN_A + TRAIN_B（都发生在 VAL 之前，安全）
    bank_ctx = np.concatenate([a_ctx,b_ctx], axis=0)
    bank_fut = np.concatenate([a_fut,b_fut], axis=0)
    bank_shape_np = np.concatenate([a_shape_np,b_shape_np], axis=0)
    bank_exp_np = np.concatenate([a_exp_np,b_exp_np], axis=0)
    bank_shape = torch.from_numpy(bank_shape_np)
    bank_exp = torch.from_numpy(bank_exp_np)

    # 纯 Shape baseline
    shape_pred, shape_ms = pure_shape_predict(
        bank_shape_np,bank_ctx,bank_fut,v_shape_np,v_ctx,
        topk=args.topk,batch_size=args.batch_size,
    )

    # 新 MLP retriever
    expert_pred, top_scores, gamma, expert_ms = retrieve_predict(
        model,bank_shape,bank_exp,bank_ctx,bank_fut,
        v_shape,v_exp_t,v_ctx,
        topk=args.topk,batch_size=args.batch_size,device=device,
    )

    shape_mse, shape_mae = mse(v_fut,shape_pred), mae(v_fut,shape_pred)
    expert_mse, expert_mae = mse(v_fut,expert_pred), mae(v_fut,expert_pred)

    report = {
        "experiment":"expert_mlp_retrieval_v0_val_only",
        "test_accessed":False,
        "split":{"train_model_queries":"TRAIN_B","training_memory":"TRAIN_A","val_memory":"TRAIN_A+TRAIN_B","evaluation":"VAL"},
        "config":vars(args).copy(),
        "dimensions":{"shape_dim":int(a_shape_np.shape[1]),"expert_dim":int(a_exp_np.shape[1])},
        "learned_gamma":gamma,
        "memory_only":{
            "shape":{"mse":shape_mse,"mae":shape_mae,"ms_per_query":shape_ms},
            "expert_mlp":{"mse":expert_mse,"mae":expert_mae,"ms_per_query":expert_ms},
            "expert_mlp_vs_shape_improvement_pct":improve(shape_mse,expert_mse),
        },
        "training_history":history,
    }
    report["config"]["bundle"] = str(args.bundle)
    report["config"]["out_dir"] = str(args.out_dir)

    # 如果已有 Chronos VAL prediction，就同时测试最终融合
    if "chronos_val_pred" in A:
        chronos = A["chronos_val_pred"].astype(np.float32)
        if chronos.shape != v_fut.shape:
            raise ValueError(f"chronos_val_pred shape={chronos.shape} 与 val_future={v_fut.shape} 不一致")
        c_mse, c_mae = mse(v_fut,chronos), mae(v_fut,chronos)
        sg, sb = best_lambda(v_fut,chronos,shape_pred,args.lambdas)
        eg, eb = best_lambda(v_fut,chronos,expert_pred,args.lambdas)
        report["chronos_fusion"] = {
            "chronos":{"mse":c_mse,"mae":c_mae},
            "shape":{"grid":sg,"best":sb,"improvement_vs_chronos_pct":improve(c_mse,sb["mse"])},
            "expert_mlp":{"grid":eg,"best":eb,"improvement_vs_chronos_pct":improve(c_mse,eb["mse"]),"improvement_vs_best_shape_pct":improve(sb["mse"],eb["mse"])},
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":model.state_dict(),
        "shape_dim":a_shape_np.shape[1],
        "expert_dim":a_exp_np.shape[1],
        "hidden":args.hidden,
        "expert_mu":e_mu,
        "expert_sd":e_sd,
    }, args.out_dir / "model.pt")

    np.savez_compressed(
        args.out_dir / "val_predictions.npz",
        y_true=v_fut,
        shape_memory_pred=shape_pred,
        expert_mlp_memory_pred=expert_pred,
        topk_scores=top_scores,
    )

    (args.out_dir / "report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    lines = []
    lines.append("# Expert-Conditioned MLP Retriever — VAL only")
    lines.append("")
    lines.append("| Method | MSE | MAE | ms/query |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Pure Shape Memory | {shape_mse:.6f} | {shape_mae:.6f} | {shape_ms:.3f} |")
    lines.append(f"| Expert-MLP Memory | {expert_mse:.6f} | {expert_mae:.6f} | {expert_ms:.3f} |")
    lines.append("")
    lines.append(f"Expert-MLP vs Shape: **{improve(shape_mse,expert_mse):+.4f}%**")
    lines.append(f"Learned gamma: **{gamma:.4f}**")
    if "chronos_fusion" in report:
        cf = report["chronos_fusion"]
        lines.append("")
        lines.append("## Chronos fusion")
        lines.append(f"Chronos MSE: **{cf['chronos']['mse']:.6f}**")
        lines.append(f"Best Chronos+Shape: **{cf['shape']['best']['mse']:.6f}**, lambda={cf['shape']['best']['lambda']}")
        lines.append(f"Best Chronos+Expert-MLP: **{cf['expert_mlp']['best']['mse']:.6f}**, lambda={cf['expert_mlp']['best']['lambda']}")
        lines.append(f"Expert-MLP vs Chronos: **{cf['expert_mlp']['improvement_vs_chronos_pct']:+.4f}%**")
        lines.append(f"Expert-MLP vs best Shape fusion: **{cf['expert_mlp']['improvement_vs_best_shape_pct']:+.4f}%**")
    (args.out_dir / "report.md").write_text("\n".join(lines),encoding="utf-8")

    print("\n" + "="*72)
    print("VAL RESULTS")
    print("="*72)
    print(f"Pure Shape Memory : MSE={shape_mse:.6f} MAE={shape_mae:.6f} {shape_ms:.3f} ms/query")
    print(f"Expert-MLP Memory : MSE={expert_mse:.6f} MAE={expert_mae:.6f} {expert_ms:.3f} ms/query")
    print(f"Improvement       : {improve(shape_mse,expert_mse):+.4f}%")
    print(f"Learned gamma     : {gamma:.4f}")
    if "chronos_fusion" in report:
        cf=report["chronos_fusion"]
        print(f"Chronos            : MSE={cf['chronos']['mse']:.6f}")
        print(f"Best + Shape       : MSE={cf['shape']['best']['mse']:.6f}, lambda={cf['shape']['best']['lambda']}")
        print(f"Best + Expert-MLP  : MSE={cf['expert_mlp']['best']['mse']:.6f}, lambda={cf['expert_mlp']['best']['lambda']}")
    print("outputs:", args.out_dir)
    print("="*72)


if __name__ == "__main__":
    main()
