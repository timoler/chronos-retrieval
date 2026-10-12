#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""etth2 预测聚合对照（独立实验，不覆盖正式结果）。

检验假设：PAER 检索器训练用「均值对齐」聚合（y_i + x̄ - x̄_i），推理却用原始平均（y_i）。
本脚本复现 seed42 的训练 + 检索，同时输出两种记忆预测（原始 / 对齐，同一 Top-5 索引），
在正式 VAL 上比较融合 MSE。只改聚合公式，不改专家池/权重/评分/Top-5/λ 网格。

输出到独立目录 results/etth2_agg_alignment/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # experiments/
sys.path.insert(0, str(HERE.parent.parent))   # repo root

import run_ours_fusion_v2 as v2  # noqa: E402  复用 load pools / _init_state / train 等
from run_conditional_expert_mlp_v4 import ConditionalExpertMLPRetriever  # noqa: E402

REPO = v2.REPO_ROOT
OUT = REPO / "results/etth2_agg_alignment"
OUT.mkdir(parents=True, exist_ok=True)

DS = "etth2"
SEED = 42
LAMBDA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]


@torch.no_grad()
def retrieve_dual(model, X_q, X_bank, Y_bank, X_q_res, X_bank_res, act_q, bank_act,
                  topk, device, q_chunk=128, bank_chunk=4096):
    """与 retrieve_conditional 完全同构，但额外返回 Top-5 索引与两种记忆预测。"""
    L = X_q.shape[1]
    model.eval()
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    X_bank_gpu = torch.tensor(X_bank, dtype=torch.float32, device=device)
    X_bank_res_gpu = torch.tensor(X_bank_res, dtype=torch.float32, device=device)
    z_bank = None
    z_chunks = []
    for bs in range(0, len(X_bank), bank_chunk):
        be = min(bs + bank_chunk, len(X_bank))
        z_chunks.append(model.encode(X_bank_gpu[bs:be], bank_act_t[bs:be]))
    z_bank = torch.cat(z_chunks, dim=0)

    pred_raw, pred_align, all_idx = [], [], []
    for qs in range(0, len(X_q), q_chunk):
        qe = min(qs + q_chunk, len(X_q))
        nq = qe - qs
        best_scores = np.full((nq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nq, topk), -1, dtype=np.int64)
        x_q = torch.tensor(X_q[qs:qe], dtype=torch.float32, device=device)
        actq = torch.tensor(act_q[qs:qe], dtype=torch.float32, device=device)
        res_q = torch.tensor(X_q_res[qs:qe], dtype=torch.float32, device=device)
        z_q = model.encode(x_q, actq)
        for bs in range(0, len(X_bank), bank_chunk):
            be = min(bs + bank_chunk, len(X_bank))
            res_c = X_bank_res_gpu[bs:be]
            shape = (res_q @ res_c.T) / L
            delta = model.delta(z_q, z_bank[bs:be])
            s = model.combine(delta, shape, "full")
            s_np = s.cpu().numpy()
            kk = min(topk, be - bs)
            lp = np.argpartition(s_np, -kk, axis=1)[:, -kk:]
            ls = np.take_along_axis(s_np, lp, axis=1)
            li = lp + bs
            cs = np.concatenate([best_scores, ls], axis=1)
            ci = np.concatenate([best_idx, li], axis=1)
            part = np.argpartition(cs, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(cs, part, axis=1)
            best_idx = np.take_along_axis(ci, part, axis=1)
        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_idx = np.take_along_axis(best_idx, order, axis=1)

        q_mu = X_q[qs:qe].mean(axis=1)[:, None, None]     # [nq,1,1] 当前窗口均值 x̄
        bank_sel = X_bank[best_idx]                        # [nq,topk,context]
        c_mu = bank_sel.mean(axis=2)[:, :, None]           # [nq,topk,1] x̄_i
        y_sel = Y_bank[best_idx]                           # [nq,topk,horizon]

        raw = y_sel.mean(axis=1)                           # 原始平均
        aligned = (y_sel + (q_mu - c_mu)).mean(axis=1)     # 对齐平均 = y_i + x̄ - x̄_i

        pred_raw.append(raw)
        pred_align.append(aligned)
        all_idx.append(best_idx)
    return (np.concatenate(pred_raw, 0), np.concatenate(pred_align, 0),
            np.concatenate(all_idx, 0))


def mse(y, p):
    return float(((y.astype(np.float64) - p.astype(np.float64)) ** 2).mean())


def main():
    ds_dir = REPO / "results/multidataset" / DS
    cfg = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg["context"]); horizon = int(cfg["horizon"])
    v2.v1.CONTEXT = context; v2.v1.HORIZON = horizon

    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train_starts = np.asarray(idx["train"], np.int64)
    val_starts = np.asarray(idx["val"], np.int64)
    test_starts = np.asarray(idx["test"], np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]; fit_starts = train_starts[cut:]

    X_bank, Y_bank = v2.build_windows(series, bank_starts, context, horizon)
    X_fit, Y_fit = v2.build_windows(series, fit_starts, context, horizon)
    X_val, Y_val = v2.build_windows(series, val_starts, context, horizon)
    X_test, Y_test = v2.build_windows(series, test_starts, context, horizon)
    X_bank_res = v2.residualize(X_bank); X_fit_res = v2.residualize(X_fit)
    X_val_res = v2.residualize(X_val); X_test_res = v2.residualize(X_test)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    val_ts = [dates[s + context] for s in val_starts]
    test_ts = [dates[s + context] for s in test_starts]
    stats = v2.compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    raw_pool = v2.load_raw_pool(DS)
    v1_pool = v2.load_v1_pool(DS)
    v2_pool = v2.load_v2_pool(DS)
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emb_v2_orig = v2.v1.embed_claims([e["claim"] for e in v2_pool], device)
    expert_dim = emb_v2_orig.shape[1]
    act_val = v2.v1.act_mats(v2_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    act_test = v2.v1.act_mats(v2_pool, X_bank, X_fit, X_test, bank_ts, fit_ts, test_ts, stats)
    v2_act_val = {"bank": act_val["bank"], "fit": act_val["fit"], "val": act_val["val"]}
    v2_act_test = {"bank": act_test["bank"], "fit": act_test["fit"], "test": act_test["val"]}
    emb_v2, v2_act_val, _ = v2.v1.pad_to_slots(emb_v2_orig, v2_act_val, n_slots)
    _, v2_act_test, _ = v2.v1.pad_to_slots(emb_v2_orig, v2_act_test, n_slots)

    # 训练（与正式 run_v2_memory_pred 完全一致：seed_everything + train_arm）
    init_state = v2._init_state(SEED, context, expert_dim, n_slots, device)
    mdl = ConditionalExpertMLPRetriever(context=context, expert_dim=expert_dim, n_experts=n_slots,
                                        d_enc=v2.D_ENC, d_exp=v2.D_EXP, d_z=v2.D_Z, hidden=v2.HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb_v2)
    v2.v2p.train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                     v2_act_val["bank"], v2_act_val["fit"], True, device, SEED)

    # 双记忆检索（VAL + TEST，同 Top-5 索引）
    raw_val, align_val, idx_val = retrieve_dual(mdl, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                                v2_act_val["val"], v2_act_val["bank"], v2.TOPK, device)
    raw_test, align_test, idx_test = retrieve_dual(mdl, X_test, X_bank, Y_bank, X_test_res, X_bank_res,
                                                   v2_act_test["test"], v2_act_test["bank"], v2.TOPK, device)

    # 复现检查：raw_val 的 TEST MSE 应与冻结 seed42 一致
    frozen_seed42 = {"etth2": 85.114947}
    raw_mse_val = mse(Y_val, raw_val)
    raw_mse_test = mse(Y_test, raw_test)
    align_mse_val = mse(Y_val, align_val)
    align_mse_test = mse(Y_test, align_test)
    print(f"[repro] raw TEST MSE = {raw_mse_test:.6f}  (冻结 seed42={frozen_seed42['etth2']:.6f})", flush=True)
    print(f"[mem MSE] raw: VAL={raw_mse_val:.4f} TEST={raw_mse_test:.4f} | "
          f"aligned: VAL={align_mse_val:.4f} TEST={align_mse_test:.4f}", flush=True)

    # Chronos-2（复用缓存 preds/etth2.npz）
    preds = np.load(REPO / "results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011/preds/etth2.npz")
    yc_val = preds["yc_val"]; yc_test = preds["yc_test"]
    chronos_val = mse(Y_val, yc_val); chronos_test = mse(Y_test, yc_test)
    print(f"[chronos] VAL={chronos_val:.4f} TEST={chronos_test:.4f}", flush=True)

    # 融合 λ 网格（VAL 选，同一 λ 用于两种记忆）
    rows = []
    for name, mem_val, mem_test in [("raw", raw_val, raw_test), ("aligned", align_val, align_test)]:
        best_lam, best_mse = None, None
        for lam in LAMBDA_GRID:
            fused = (1 - lam) * yc_val + lam * mem_val
            m = mse(Y_val, fused)
            if best_mse is None or m < best_mse:
                best_mse, best_lam = m, lam
        fused_test = (1 - best_lam) * yc_test + best_lam * mem_test
        fused_test_mse = mse(Y_test, fused_test)
        rows.append({"method": name, "mem_val_mse": mse(Y_val, mem_val), "mem_test_mse": mse(Y_test, mem_test),
                     "best_lam": best_lam, "fusion_val_mse": best_mse, "fusion_test_mse": fused_test_mse})
        print(f"[{name}] best_lam={best_lam} fusion VAL={best_mse:.4f} TEST={fused_test_mse:.4f}", flush=True)

    # 四段 VAL MSE（沿用完整 VAL 选的 λ）
    n_val = len(Y_val)
    quart = n_val // 4
    for name, mem_val, best_lam in [(r["method"], raw_val if r["method"] == "raw" else align_val, r["best_lam"]) for r in rows]:
        fused_val = (1 - best_lam) * yc_val + best_lam * mem_val
        segs = []
        for k in range(4):
            a, b = k * quart, (k + 1) * quart if k < 3 else n_val
            segs.append(mse(Y_val[a:b], fused_val[a:b]))
        print(f"[{name}] 四段 VAL MSE: " + "  ".join(f"{s:.4f}" for s in segs), flush=True)

    # 落盘
    np.savez_compressed(OUT / "etth2_seed42_dual.npz",
                        Y_val=Y_val, Y_test=Y_test, yc_val=yc_val, yc_test=yc_test,
                        raw_val=raw_val, raw_test=raw_test, align_val=align_val, align_test=align_test,
                        idx_val=idx_val, idx_test=idx_test,
                        val_starts=val_starts, test_starts=test_starts)
    import csv
    with open(OUT / "result_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "mem_val_mse", "mem_test_mse", "best_lam", "fusion_val_mse", "fusion_test_mse"])
        w.writeheader(); w.writerows(rows)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
