#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Electricity Pool-Context 最后一层机制诊断。

回答一个问题（只用 TRAIN / frozen VAL）：
  为什么 v2 coverage 从 1.00 降到 0.69 后变差，但简单把 E2/E8 加回 v2 pool 又不能
  改善整体 VAL？Expert 价值是否依赖 time-series regime 与 Expert Pool context？

性质：failure analysis / limitation / future work。不修改 frozen v2 / reward / prompt /
pool；不做 v3 / dynamic selector / 新 reward；不访问 TEST；不调 LLM；不做 exhaustive
subset search。

固定五个 Pool（+ 两个 raw-minus 用于 pool-context 边际贡献）：
  P0 = RAW_POOL
  P1 = V2_POOL
  P2 = V2_POOL + E2
  P3 = V2_POOL + E8
  P4 = V2_POOL + E2 + E8
  另有 raw∖E2 / raw∖E8（仅用于 Section 6 的 "E2/E8 in RAW_POOL" 边际贡献）。

注：本脚本修正了 diagnose_v2_failure_steps6_7_9.py 里 electricity synergy 的一个 bug——
后者在 synergy 阶段复用了 alignment 的 ctx cache，但 v2p 模块全局仍指向 traffic 数据，
导致 electricity 的 V2_POOL VAL MSE 被算成 traffic 量级（~7e-4）。本脚本只跑 electricity，
全局只设一次，并断言 shape，杜绝跨数据集污染。

输出：results/reward_feedback_critic_v2_failure_diagnosis/pool_context_dependence/
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from run_expert_embedding_mlp import build_windows, mse, mae
from run_solar_same_val_baselines import residualize
from run_conditional_expert_mlp_v4 import ConditionalExpertMLPRetriever, retrieve_conditional
from run_etth1_prediction_validation import compute_training_stats, compute_window_feats
import run_etth1_reward_feedback_critic_v1 as v1
import run_reward_feedback_critic_v2_poolaware_pilot as v2p

REPO_ROOT = HERE.parent
FROZEN_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_full_cross_dataset"
OUT_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_failure_diagnosis/pool_context_dependence"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

NAME = "electricity"
SEEDS = [42, 123, 2026]
TOPK = v2p.TOPK
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

ACT_THRESH = 0.5

# 冻结 VAL MSE 参照（pilot_val_ablation.csv），用于回验 P0/P1 逐位一致
FROZEN_VAL_REF = {
    ("raw", 42): 298266464.0, ("raw", 123): 254079008.0, ("raw", 2026): 253278784.0,
    ("v2", 42): 278161440.0, ("v2", 123): 437748672.0, ("v2", 2026): 267246560.0,
}

# 预定义 regime 轴（quartile 阈值由 TRAIN 计算，固定应用到 VAL）
REGIME_AXES = ["VOL", "LEVEL", "TREND", "CHANGE"]
REGIME_NAMES = {
    "VOL": "context_volatility",
    "LEVEL": "target_level",
    "TREND": "trend_strength",
    "CHANGE": "recent_change",
}


# --------------------------------------------------------------------------- #
# 数据上下文（bank/fit/val，TEST 不加载）
# --------------------------------------------------------------------------- #
def build_val_context(device):
    ds_dir = REPO_ROOT / "results/multidataset" / NAME
    cfg = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg["context"])
    horizon = int(cfg["horizon"])
    v1.CONTEXT = context
    v1.HORIZON = horizon

    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    X_bank, Y_bank = build_windows(series, bank_starts, context, horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, context, horizon)
    X_val, Y_val = build_windows(series, val_starts, context, horizon)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    val_ts = [dates[s + context] for s in val_starts]
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)

    # 写入 v2p 模块全局（唯一一次），供 v2p.run_val_arm / train_arm 复用
    v2p.X_bank, v2p.Y_bank = X_bank, Y_bank
    v2p.X_fit, v2p.Y_fit = X_fit, Y_fit
    v2p.X_val, v2p.Y_val = X_val, Y_val
    v2p.X_bank_res, v2p.X_fit_res, v2p.X_val_res = X_bank_res, X_fit_res, X_val_res

    return {
        "context": context, "horizon": horizon,
        "X_bank": X_bank, "Y_bank": Y_bank, "X_fit": X_fit, "Y_fit": Y_fit,
        "X_val": X_val, "Y_val": Y_val,
        "bank_ts": bank_ts, "fit_ts": fit_ts, "val_ts": val_ts, "stats": stats,
        "n_val": len(val_starts),
    }


def make_init_state(seed, context, expert_dim, n_slots, device):
    v2p.seed_everything(seed)
    m = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def eval_pool(init_state, emb, act, expert_dim, n_slots, device, seed, context):
    """训练 + VAL 预测，返回 (逐窗口 squared error [n_val], 聚合 MSE)。"""
    emb_p, act_p, _ = v1.pad_to_slots(emb, act, n_slots)
    v2p.seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb_p)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v2p.train_arm(mdl, v2p.X_bank, v2p.Y_bank, v2p.X_fit, v2p.Y_fit,
                      v2p.X_bank_res, v2p.X_fit_res,
                      act_p["bank"], act_p["fit"], True, device, seed)
    pred = retrieve_conditional(mdl, v2p.X_val, v2p.X_bank, v2p.Y_bank,
                                v2p.X_val_res, v2p.X_bank_res,
                                act_p["val"], act_p["bank"], "full", TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    per_win = np.mean((v2p.Y_val - pred) ** 2, axis=1)
    return per_win, float(per_win.mean())


# --------------------------------------------------------------------------- #
# Pool 读取 + 核对
# --------------------------------------------------------------------------- #
def load_pools():
    raw = json.loads((FROZEN_ROOT / NAME / "raw_experts.json").read_text(encoding="utf-8"))
    v2 = json.loads((FROZEN_ROOT / NAME / "final_expert_pool.json").read_text(encoding="utf-8"))
    raw_ids = [e["id"] for e in raw]
    v2_ids = [e["id"] for e in v2]
    removed = [i for i in raw_ids if i not in v2_ids]
    # 核对：E2 与 E8 必须都是被删除的（否则停止报错，不自行替换）
    if "E2" not in removed or "E8" not in removed:
        raise RuntimeError(
            f"预期被删除 Expert 应包含 E2 与 E8，但实际 removed={removed}。停止，不自行替换。")
    return raw, v2, raw_ids, v2_ids, removed


# --------------------------------------------------------------------------- #
# Regime：TRAIN 计算 quartile 阈值，固定应用到 VAL
# --------------------------------------------------------------------------- #
def compute_regimes(ctx):
    X_train_all = np.concatenate([ctx["X_bank"], ctx["X_fit"]], axis=0)
    Y_train_all = np.concatenate([ctx["Y_bank"], ctx["Y_fit"]], axis=0)
    train_ts = ctx["bank_ts"] + ctx["fit_ts"]
    train_feats = compute_window_feats(X_train_all, train_ts, ctx["stats"])
    val_feats = compute_window_feats(ctx["X_val"], ctx["val_ts"], ctx["stats"])

    train_level = Y_train_all.mean(axis=1)
    val_level = ctx["Y_val"].mean(axis=1)

    axes = {
        "VOL": (train_feats["recent_volatility"], val_feats["recent_volatility"]),
        "LEVEL": (train_level, val_level),
        "TREND": (train_feats["recent_slope_magnitude"], val_feats["recent_slope_magnitude"]),
        "CHANGE": (train_feats["recent_diff_magnitude"], val_feats["recent_diff_magnitude"]),
    }

    thresholds = {}
    labels = {}
    for ax, (tr, va) in axes.items():
        q = np.quantile(tr, [0.25, 0.5, 0.75])
        thresholds[ax] = q
        labels[ax] = np.digitize(va, q)  # 0..3 -> Q1..Q4
    return labels, thresholds, val_feats["hour_of_day"]


# --------------------------------------------------------------------------- #
# 输出工具
# --------------------------------------------------------------------------- #
def _write_csv(path, rows, cols):
    with (OUT_ROOT / path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def regime_mask(labels, ax, q):
    return labels[ax] == q


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="subset of seeds (default: all 3)")
    args = ap.parse_args()
    seeds = args.seeds if args.seeds else SEEDS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print(f"[POOL-CONTEXT DIAGNOSIS] dataset={NAME} seeds={seeds}")
    print("boundary: TRAIN/VAL only, no TEST, no LLM, no v3, no pool reselection")

    # 1. 数据上下文（设置 v2p 全局，唯一一次）
    ctx = build_val_context(device)
    assert len(v2p.X_val) == ctx["n_val"], \
        f"v2p.X_val shape 与 electricity VAL 不一致：{len(v2p.X_val)} vs {ctx['n_val']}"

    # 2. pool 读取 + 核对
    raw_pool, v2_pool, raw_ids, v2_ids, removed = load_pools()
    print(f"  raw={raw_ids}  v2={v2_ids}  removed={removed}")

    # 3. embedding + activation（seed 无关，各算一次）
    emb_raw = v1.embed_claims([e["claim"] for e in raw_pool], device)
    emb_v2 = v1.embed_claims([e["claim"] for e in v2_pool], device)
    expert_dim = emb_raw.shape[1]
    n_slots = max(len(raw_pool), len(v2_pool))
    act_raw = v1.act_mats(raw_pool, ctx["X_bank"], ctx["X_fit"], ctx["X_val"],
                          ctx["bank_ts"], ctx["fit_ts"], ctx["val_ts"], ctx["stats"])
    act_v2 = v1.act_mats(v2_pool, ctx["X_bank"], ctx["X_fit"], ctx["X_val"],
                         ctx["bank_ts"], ctx["fit_ts"], ctx["val_ts"], ctx["stats"])

    raw_by_id = {e["id"]: e for e in raw_pool}

    def build_pool(spec):
        """spec: list of (source, expert_id)。source ∈ {'raw','v2'}。返回 (emb, act)。"""
        emb_parts = []
        act_parts = {k: [] for k in ("bank", "fit", "val")}
        for src, eid in spec:
            if src == "raw":
                j = raw_ids.index(eid)
                emb_parts.append(emb_raw[j:j + 1])
                for k in act_parts:
                    act_parts[k].append(act_raw[k][:, j:j + 1])
            else:
                j = v2_ids.index(eid)
                emb_parts.append(emb_v2[j:j + 1])
                for k in act_parts:
                    act_parts[k].append(act_v2[k][:, j:j + 1])
        emb = np.concatenate(emb_parts, axis=0)
        act = {k: np.concatenate(act_parts[k], axis=1) for k in act_parts}
        return emb, act

    pool_specs = {
        "P0_raw": [("raw", e) for e in raw_ids],
        "P1_v2": [("v2", e) for e in v2_ids],
        "P2_v2E2": [("v2", e) for e in v2_ids] + [("raw", "E2")],
        "P3_v2E8": [("v2", e) for e in v2_ids] + [("raw", "E8")],
        "P4_v2E2E8": [("v2", e) for e in v2_ids] + [("raw", "E2"), ("raw", "E8")],
        "raw_minus_E2": [("raw", e) for e in raw_ids if e != "E2"],
        "raw_minus_E8": [("raw", e) for e in raw_ids if e != "E8"],
    }
    pool_label = {
        "P0_raw": "RAW_POOL", "P1_v2": "V2_POOL", "P2_v2E2": "V2_POOL+E2",
        "P3_v2E8": "V2_POOL+E8", "P4_v2E2E8": "V2_POOL+E2+E8",
        "raw_minus_E2": "RAW_POOL-E2", "raw_minus_E8": "RAW_POOL-E8",
    }
    pool_experts = {
        "P0_raw": raw_ids, "P1_v2": v2_ids, "P2_v2E2": v2_ids + ["E2"],
        "P3_v2E8": v2_ids + ["E8"], "P4_v2E2E8": v2_ids + ["E2", "E8"],
        "raw_minus_E2": [e for e in raw_ids if e != "E2"],
        "raw_minus_E8": [e for e in raw_ids if e != "E8"],
    }

    pool_emb = {}
    pool_act = {}
    for k, spec in pool_specs.items():
        pool_emb[k], pool_act[k] = build_pool(spec)

    # 4. coverage（seed 无关）
    def coverage_mask(act_val):
        return (act_val > ACT_THRESH).any(axis=1)

    cov = {k: coverage_mask(pool_act[k]["val"]) for k in pool_specs}

    # 5. regime（TRAIN 阈值，固定应用到 VAL）
    regime_labels, regime_thresholds, hour_of_day = compute_regimes(ctx)

    # 6. 逐 seed 逐 pool 评估
    print("\n=== EVAL ===")
    per_seed = {}          # seed -> pool -> (per_win_err, agg)
    for seed in seeds:
        init = make_init_state(seed, ctx["context"], expert_dim, n_slots, device)
        per_seed[seed] = {}
        for k in pool_specs:
            pw, agg = eval_pool(init, pool_emb[k], pool_act[k], expert_dim, n_slots,
                                device, seed, ctx["context"])
            per_seed[seed][k] = (pw, agg)
            # 回验 P0/P1 vs 冻结
            if k == "P0_raw" and ("raw", seed) in FROZEN_VAL_REF:
                ref = FROZEN_VAL_REF[("raw", seed)]
                ok = abs(agg - ref) <= 1e-3 * max(1.0, abs(ref))
                print(f"  seed={seed} P0_raw agg={agg:.6g} frozen={ref:.6g} [{'OK' if ok else 'MISMATCH'}]")
            if k == "P1_v2" and ("v2", seed) in FROZEN_VAL_REF:
                ref = FROZEN_VAL_REF[("v2", seed)]
                ok = abs(agg - ref) <= 1e-3 * max(1.0, abs(ref))
                print(f"  seed={seed} P1_v2  agg={agg:.6g} frozen={ref:.6g} [{'OK' if ok else 'MISMATCH'}]")
        print(f"  seed={seed}: " + "  ".join(
            f"{k}={per_seed[seed][k][1]:.4e}" for k in pool_specs))

    # ------------------------------------------------------------------ #
    # 输出 1：overall_pool_comparison.csv
    # ------------------------------------------------------------------ #
    rows = []
    for seed in seeds:
        v2_agg = per_seed[seed]["P1_v2"][1]
        for k in pool_specs:
            agg = per_seed[seed][k][1]
            rows.append({
                "seed": seed, "pool": pool_label[k],
                "experts": ";".join(pool_experts[k]),
                "coverage": float(cov[k].mean()),
                "val_mse": agg,
                "relative_vs_v2": (v2_agg - agg) / v2_agg if v2_agg > 0 else 0.0,
            })
    _write_csv("overall_pool_comparison.csv", rows,
               ["seed", "pool", "experts", "coverage", "val_mse", "relative_vs_v2"])

    summary_rows = []
    for k in pool_specs:
        ms = [per_seed[s][k][1] for s in seeds]
        wins = sum(1 for s in seeds if per_seed[s][k][1] < per_seed[s]["P1_v2"][1])
        summary_rows.append({
            "pool": pool_label[k], "mean_mse": float(np.mean(ms)), "std_mse": float(np.std(ms)),
            "mean_coverage": float(cov[k].mean()), "wins_vs_v2": wins,
        })
    _write_csv("overall_pool_comparison_summary.csv", summary_rows,
               ["pool", "mean_mse", "std_mse", "mean_coverage", "wins_vs_v2"])

    # ------------------------------------------------------------------ #
    # 输出 2：pool_regime_performance.csv
    # ------------------------------------------------------------------ #
    reg_rows = []
    for seed in seeds:
        for k in pool_specs:
            pw = per_seed[seed][k][0]
            for ax in REGIME_AXES:
                for q in range(4):
                    m = regime_mask(regime_labels, ax, q)
                    if m.sum() == 0:
                        continue
                    pool_mse = float(pw[m].mean())
                    v2_mse = float(per_seed[seed]["P1_v2"][0][m].mean())
                    reg_rows.append({
                        "dataset": NAME, "seed": seed, "pool": pool_label[k],
                        "regime_type": REGIME_NAMES[ax], "regime": f"Q{q+1}",
                        "n_windows": int(m.sum()), "coverage": float(cov[k][m].mean()),
                        "mse": pool_mse,
                        "relative_vs_v2_same_regime": (v2_mse - pool_mse) / v2_mse if v2_mse > 0 else 0.0,
                    })
    _write_csv("pool_regime_performance.csv", reg_rows,
               ["dataset", "seed", "pool", "regime_type", "regime", "n_windows",
                "coverage", "mse", "relative_vs_v2_same_regime"])

    # ------------------------------------------------------------------ #
    # 输出 3：expert_regime_gain.csv（E2 / E8，V2 context，逐 regime）
    # ------------------------------------------------------------------ #
    def gain_in_regimes(base_pool, plus_pool, seed):
        base_pw = per_seed[seed][base_pool][0]
        plus_pw = per_seed[seed][plus_pool][0]
        out = {"overall": (base_pw.mean() - plus_pw.mean()) / base_pw.mean() if base_pw.mean() > 0 else 0.0}
        for ax in REGIME_AXES:
            for q in range(4):
                m = regime_mask(regime_labels, ax, q)
                b = base_pw[m].mean()
                p = plus_pw[m].mean()
                out[f"{ax}_Q{q+1}"] = (b - p) / b if b > 0 else 0.0
        return out

    gain_rows = []
    gain_store = {}  # (expert, seed) -> dict(regime -> gain)
    for expert, plus_pool in (("E2", "P2_v2E2"), ("E8", "P3_v2E8")):
        for seed in seeds:
            g = gain_in_regimes("P1_v2", plus_pool, seed)
            gain_store[(expert, seed)] = g
            for reg, val in g.items():
                gain_rows.append({
                    "expert": expert, "seed": seed, "regime_type": reg, "regime": "",
                    "gain": val,
                })
    _write_csv("expert_regime_gain.csv", gain_rows,
               ["expert", "seed", "regime_type", "regime", "gain"])

    # 输出 5 相关：REGIME_DEPENDENT_VALUE
    regime_dep = {}
    for expert in ("E2", "E8"):
        for seed in seeds:
            g = gain_store[(expert, seed)]
            overall = g["overall"]
            reg_vals = [g[f"{ax}_Q{q+1}"] for ax in REGIME_AXES for q in range(4)]
            best = max(reg_vals)
            worst = min(reg_vals)
            pos = sum(1 for v in reg_vals if v > 0)
            neg = sum(1 for v in reg_vals if v <= 0)
            # 标记：整体有害，但存在某个 regime 明显有益（>1%）
            flag = int(overall <= 0 and best > 0.01)
            regime_dep[(expert, seed)] = {
                "overall": overall, "best_regime": best, "worst_regime": worst,
                "positive_regime_count": pos, "negative_regime_count": neg,
                "flag": flag,
            }
            print(f"  [{expert}] seed={seed}: overall_gain={overall:+.4f} "
                  f"best_regime={best:+.4f} worst_regime={worst:+.4f} "
                  f"pos_regime={pos}/16 neg_regime={neg}/16 flag={flag}")

    # ------------------------------------------------------------------ #
    # 输出 4：expert_context_contribution.csv（E2/E8 × RAW_POOL/V2_POOL）
    # ------------------------------------------------------------------ #
    def contribution_in_context(base_pool, plus_pool, seed):
        """plus_pool 相对 base_pool 的边际贡献（base 不含 expert，plus 含 expert）。
        正 = expert 有帮助。base 为 '不含 expert' 的 pool。"""
        base_pw = per_seed[seed][base_pool][0]
        plus_pw = per_seed[seed][plus_pool][0]
        def _g(m):
            b = base_pw[m].mean()
            p = plus_pw[m].mean()
            return (b - p) / b if b > 0 else 0.0
        out = {"overall_gain": _g(np.ones(len(base_pw), dtype=bool))}
        for ax in REGIME_AXES:
            for q in range(4):
                out[f"{ax}_Q{q+1}_gain"] = _g(regime_mask(regime_labels, ax, q))
        return out

    # E2 in RAW_POOL: base=raw_minus_E2, plus=P0_raw（raw 含 E2）
    # E2 in V2_POOL: base=P1_v2, plus=P2_v2E2
    # E8 in RAW_POOL: base=raw_minus_E8, plus=P0_raw
    # E8 in V2_POOL: base=P1_v2, plus=P3_v2E8
    ctx_pairs = {
        ("E2", "RAW_POOL"): ("raw_minus_E2", "P0_raw"),
        ("E2", "V2_POOL"): ("P1_v2", "P2_v2E2"),
        ("E8", "RAW_POOL"): ("raw_minus_E8", "P0_raw"),
        ("E8", "V2_POOL"): ("P1_v2", "P3_v2E8"),
    }
    ctx_cols = ["expert", "seed", "context", "overall_gain"] + \
               [f"{ax}_Q{q+1}_gain" for ax in REGIME_AXES for q in range(4)]
    ctx_rows = []
    ctx_store = {}
    for (expert, context), (base, plus) in ctx_pairs.items():
        for seed in seeds:
            c = contribution_in_context(base, plus, seed)
            ctx_store[(expert, context, seed)] = c
            ctx_rows.append({"expert": expert, "seed": seed, "context": context, **c})
    _write_csv("expert_context_contribution.csv", ctx_rows, ctx_cols)

    # POOL_CONTEXT_DEPENDENT：同一 expert 在 RAW_POOL >0 而 V2_POOL <0（或反向）
    pool_context_dep = {}
    for expert in ("E2", "E8"):
        for seed in seeds:
            g_raw = ctx_store[(expert, "RAW_POOL", seed)]["overall_gain"]
            g_v2 = ctx_store[(expert, "V2_POOL", seed)]["overall_gain"]
            flip = int((g_raw > 0) != (g_v2 > 0))
            # 逐 regime sign flip 计数
            n_regime_flip = 0
            for ax in REGIME_AXES:
                for q in range(4):
                    k = f"{ax}_Q{q+1}_gain"
                    if (ctx_store[(expert, "RAW_POOL", seed)][k] > 0) != \
                       (ctx_store[(expert, "V2_POOL", seed)][k] > 0):
                        n_regime_flip += 1
            pool_context_dep[(expert, seed)] = {
                "raw_gain": g_raw, "v2_gain": g_v2, "overall_flip": flip,
                "n_regime_flip": n_regime_flip,
            }
            print(f"  [{expert}] seed={seed}: RAW_POOL gain={g_raw:+.4f} "
                  f"V2_POOL gain={g_v2:+.4f} overall_flip={flip} regime_flip={n_regime_flip}/16")

    # ------------------------------------------------------------------ #
    # 输出 5：coverage_performance_link.csv
    # ------------------------------------------------------------------ #
    v2_cov_mask = cov["P1_v2"]
    covered = v2_cov_mask
    not_covered = ~v2_cov_mask

    cov_rows = []
    for seed in seeds:
        raw_pw = per_seed[seed]["P0_raw"][0]
        v2_pw = per_seed[seed]["P1_v2"][0]
        for gname, m in (("covered_by_v2", covered), ("not_covered_by_v2", not_covered)):
            if m.sum() == 0:
                continue
            r = float(raw_pw[m].mean())
            v = float(v2_pw[m].mean())
            cov_rows.append({
                "kind": "group", "seed": seed, "group": gname,
                "n_windows": int(m.sum()), "raw_mse": r, "v2_mse": v,
                "v2_vs_raw_pct": (v - r) / r * 100.0 if r > 0 else 0.0,
                "restore_pool": "", "n_restored": "", "v2_mse_restored": "",
                "restore_mse_restored": "", "improved_ratio": "",
            })
        # 恢复 coverage 的窗口：P2/P3/P4 覆盖但 v2 未覆盖
        for restore_pool, rk in (("v2+E2", "P2_v2E2"), ("v2+E8", "P3_v2E8"),
                                 ("v2+E2+E8", "P4_v2E2E8")):
            restored = not_covered & cov[rk]
            if restored.sum() == 0:
                continue
            v2_rest = float(v2_pw[restored].mean())
            plus_rest = float(per_seed[seed][rk][0][restored].mean())
            improved = float((per_seed[seed][rk][0][restored] < v2_pw[restored]).mean())
            cov_rows.append({
                "kind": "restore", "seed": seed, "group": "",
                "n_windows": int(restored.sum()), "raw_mse": "", "v2_mse": "",
                "v2_vs_raw_pct": "",
                "restore_pool": restore_pool, "n_restored": int(restored.sum()),
                "v2_mse_restored": v2_rest, "restore_mse_restored": plus_rest,
                "improved_ratio": improved,
            })
    _write_csv("coverage_performance_link.csv", cov_rows,
               ["kind", "seed", "group", "n_windows", "raw_mse", "v2_mse",
                "v2_vs_raw_pct", "restore_pool", "n_restored", "v2_mse_restored",
                "restore_mse_restored", "improved_ratio"])

    # COVERAGE_LINK 判定
    cov_link = {}
    for seed in seeds:
        raw_pw = per_seed[seed]["P0_raw"][0]
        v2_pw = per_seed[seed]["P1_v2"][0]
        # 未覆盖窗口是否明显更差（v2 vs raw 在该子集上的退化 > covered 子集）
        deg_not = (v2_pw[not_covered].mean() - raw_pw[not_covered].mean()) / raw_pw[not_covered].mean()
        deg_cov = (v2_pw[covered].mean() - raw_pw[covered].mean()) / raw_pw[covered].mean()
        uncovered_worse = int(deg_not > deg_cov)
        # 恢复 coverage 是否改善：取 P4（+E2+E8）恢复窗口
        restored = not_covered & cov["P4_v2E2E8"]
        improved_ratio = float((per_seed[seed]["P4_v2E2E8"][0][restored] < v2_pw[restored]).mean()) \
            if restored.sum() > 0 else float("nan")
        cov_link[seed] = {
            "deg_not_covered": deg_not, "deg_covered": deg_cov,
            "uncovered_worse": uncovered_worse,
            "restored_improved_ratio": improved_ratio,
        }
        print(f"  [coverage] seed={seed}: deg_not={deg_not:+.4f} deg_cov={deg_cov:+.4f} "
              f"uncovered_worse={uncovered_worse} restored_improved_ratio={improved_ratio:.3f}")

    # ------------------------------------------------------------------ #
    # 输出 6：pool_context_mechanism_summary.csv（3-seed 证据强度）
    # ------------------------------------------------------------------ #
    def support_count(fn):
        return sum(1 for s in seeds if fn(s))

    summary_labels = []
    # REGIME_DEPENDENT_VALUE
    for expert in ("E2", "E8"):
        sup = support_count(lambda s: regime_dep[(expert, s)]["flag"] == 1)
        summary_labels.append({
            "label": f"REGIME_DEPENDENT_VALUE_{expert}",
            "seed42": regime_dep[(expert, 42)]["flag"] if 42 in seeds else "",
            "seed123": regime_dep[(expert, 123)]["flag"] if 123 in seeds else "",
            "seed2026": regime_dep[(expert, 2026)]["flag"] if 2026 in seeds else "",
            "support_count": sup,
            "interpretation": ("overall<=0 但某些 regime gain>0（3/3 strong, 2/3 moderate, 1/3 weak）"
                               if sup >= 3 else
                               ("moderate" if sup == 2 else "weak/inconclusive")),
        })
    # POOL_CONTEXT_DEPENDENT
    for expert in ("E2", "E8"):
        sup = support_count(lambda s: pool_context_dep[(expert, s)]["overall_flip"] == 1)
        summary_labels.append({
            "label": f"POOL_CONTEXT_DEPENDENT_{expert}",
            "seed42": pool_context_dep[(expert, 42)]["overall_flip"] if 42 in seeds else "",
            "seed123": pool_context_dep[(expert, 123)]["overall_flip"] if 123 in seeds else "",
            "seed2026": pool_context_dep[(expert, 2026)]["overall_flip"] if 2026 in seeds else "",
            "support_count": sup,
            "interpretation": ("RAW/V2 context 边际贡献符号翻转（3/3 strong, 2/3 moderate, 1/3 weak）"
                               if sup >= 3 else ("moderate" if sup == 2 else "weak/inconclusive")),
        })
    # COVERAGE_LINK
    sup = support_count(lambda s: cov_link[s]["uncovered_worse"] == 1)
    restored_improve = [cov_link[s]["restored_improved_ratio"] for s in seeds
                        if not np.isnan(cov_link[s]["restored_improved_ratio"])]
    improved_avg = float(np.mean(restored_improve)) if restored_improve else float("nan")
    cover_label = ("COVERAGE_LINK_SUPPORTED" if (sup >= 2 and improved_avg > 0.5)
                   else "COVERAGE_ALONE_INSUFFICIENT")
    summary_labels.append({
        "label": cover_label,
        "seed42": cov_link[42]["uncovered_worse"] if 42 in seeds else "",
        "seed123": cov_link[123]["uncovered_worse"] if 123 in seeds else "",
        "seed2026": cov_link[2026]["uncovered_worse"] if 2026 in seeds else "",
        "support_count": sup,
        "interpretation": (f"未覆盖窗口更差且恢复 coverage 后改善（restored_improved_ratio={improved_avg:.3f}）"
                           if cover_label == "COVERAGE_LINK_SUPPORTED" else
                           f"coverage 与性能关联不足（restored_improved_ratio={improved_avg:.3f}）"),
    })
    _write_csv("pool_context_mechanism_summary.csv", summary_labels,
               ["label", "seed42", "seed123", "seed2026", "support_count", "interpretation"])

    # ------------------------------------------------------------------ #
    # report
    # ------------------------------------------------------------------ #
    report = render_report(pool_specs, pool_label, per_seed, cov, seeds, regime_dep,
                           pool_context_dep, cov_link, summary_labels, ctx_store,
                           ctx_pairs, restored_improve)
    (OUT_ROOT / "pool_context_mechanism_report.md").write_text(report, encoding="utf-8")
    print(f"\nSaved to {OUT_ROOT}")


# --------------------------------------------------------------------------- #
# report（先结果，后结论）
# --------------------------------------------------------------------------- #
def render_report(pool_specs, pool_label, per_seed, cov, seeds, regime_dep,
                  pool_context_dep, cov_link, summary_labels, ctx_store, ctx_pairs,
                  restored_improve):
    L = []
    L.append("# Electricity Pool-Context 机制诊断")
    L.append("")
    L.append("只诊断，不修改 v2。TRAIN/VAL-only，不访问 TEST，不调 LLM，不重选 pool。")
    L.append("")
    L.append("> 注：本脚本修正了 steps6/7/9 里 electricity synergy 的 bug（v2p 全局被 traffic "
             "污染，导致 electricity V2_POOL VAL MSE 被算成 ~7e-4）。本脚本 electricity-only，"
             "全局只设一次。P0/P1 已逐 seed 回验 frozen pilot 值。")
    L.append("")

    # 结果 1：整体
    L.append("## 结果")
    L.append("")
    L.append("### 1. 整体 VAL MSE（加回是否救回）")
    L.append("")
    L.append("| pool | " + " | ".join(f"seed{s}" for s in seeds) + " | coverage | mean_mse | wins_vs_v2 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in pool_specs:
        ms = [per_seed[s][k][1] for s in seeds]
        wins = sum(1 for s in seeds if per_seed[s][k][1] < per_seed[s]["P1_v2"][1])
        L.append(f"| {pool_label[k]} | " + " | ".join(f"{per_seed[s][k][1]:.4e}" for s in seeds) +
                 f" | {cov[k].mean():.3f} | {np.mean(ms):.4e} | {wins}/3 |")
    L.append("")
    L.append("relative_vs_v2（正 = 比 v2 好）逐 seed 见 overall_pool_comparison.csv。")
    L.append("")

    # 结果 2：regime dependent value
    L.append("### 2. E2/E8 是否“整体有害、某些 regime 有益”")
    L.append("")
    for expert in ("E2", "E8"):
        L.append(f"**{expert}**（V2 context，gain = (MSE_v2 − MSE_v2+{expert})/MSE_v2，正 = 加回有帮助）：")
        L.append("")
        for seed in seeds:
            d = regime_dep[(expert, seed)]
            L.append(f"- seed{seed}: overall={d['overall']:+.4f} best_regime={d['best_regime']:+.4f} "
                     f"worst_regime={d['worst_regime']:+.4f} pos={d['positive_regime_count']}/16 "
                     f"neg={d['negative_regime_count']}/16")
        L.append("")
    L.append("")

    # 结果 3：pool-context dependence
    L.append("### 3. E2/E8 在 RAW_POOL vs V2_POOL 的边际贡献符号")
    L.append("")
    for expert in ("E2", "E8"):
        L.append(f"**{expert}**：")
        L.append("")
        for seed in seeds:
            d = pool_context_dep[(expert, seed)]
            L.append(f"- seed{seed}: RAW_POOL gain={d['raw_gain']:+.4f}  V2_POOL gain={d['v2_gain']:+.4f}  "
                     f"overall_flip={d['overall_flip']}  regime_flip={d['n_regime_flip']}/16")
        L.append("")
    L.append("")

    # 结果 4：coverage-performance link
    L.append("### 4. v2 未覆盖窗口是否更差；恢复 coverage 后是否改善")
    L.append("")
    for seed in seeds:
        d = cov_link[seed]
        L.append(f"- seed{seed}: 未覆盖窗口 v2 退化 {d['deg_not_covered']*100:+.2f}% vs 已覆盖 "
                 f"{d['deg_covered']*100:+.2f}%（uncovered_worse={d['uncovered_worse']}）；"
                 f"P4 恢复窗口改善比例 {d['restored_improved_ratio']:.3f}")
    L.append("")
    if restored_improve:
        L.append(f"- 恢复 coverage 窗口的平均改善比例（跨 seed）= {np.mean(restored_improve):.3f}")
    L.append("")

    # 结论
    L.append("## 结论")
    L.append("")
    # 判定 A/B/C
    regime_sup = max([r["support_count"] for r in summary_labels
                      if r["label"].startswith("REGIME_DEPENDENT_VALUE")])
    context_sup = max([r["support_count"] for r in summary_labels
                       if r["label"].startswith("POOL_CONTEXT_DEPENDENT")])
    if regime_sup >= 2 and context_sup >= 2:
        concl = "A"
    elif regime_sup >= 2:
        concl = "B"
    else:
        concl = "C"
    L.append(f"**结论选择：{concl}**")
    L.append("")
    if concl == "A":
        L.append("Electricity 的失败不只是 coverage 下降；Expert 价值还表现出对时间序列状态和当前 "
                 "Expert Pool 的依赖。全局静态筛选可能无法同时适配不同 regime。")
    elif concl == "B":
        L.append("证据主要支持 Expert value 的 regime dependence，但不足以支持明显的 pool-context "
                 "dependence。")
    else:
        L.append("当前实验不足以支持 context-dependent Expert value。Electricity 保留为 "
                 "coverage-related failure case，不继续扩展机制解释。")
    L.append("")
    L.append("（regime 证据强度：REGIME_DEPENDENT_VALUE support="
             f"{regime_sup}/3；pool-context support={context_sup}/3。）")
    L.append("")
    L.append("## 输出文件")
    L.append("")
    for p in sorted(f.name for f in OUT_ROOT.glob("*")):
        L.append(f"- `{p}`")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
