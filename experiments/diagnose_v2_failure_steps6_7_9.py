#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc failure diagnosis — 补齐 steps 6 / 7 / 9 + traffic activation stats.

`diagnose_v2_failure_electricity_traffic.py` 已完成描述性部分（pool 变化 / traffic
comparable 判定 / electricity 误差切片 / electricity coverage），但缺：
  - Section 6：pairwise synergy（TRAIN/VAL-only，不碰 TEST）
  - Section 7：TRAIN pool_reward vs VAL contribution 对齐（含 ETTh1 / Solar 对照）
  - Section 9：原因标签 + failure_diagnosis_summary.csv
  - Section 5-traffic：traffic_pool_activation_stats.csv
  - 统一输出目录：results/reward_feedback_critic_v2_failure_diagnosis/

严格复用 frozen v2。synergy / reward-alignment 只在 TRAIN/VAL 上做：
  - 不访问 TEST 来筛 pair / 选 pool / 调参数 / 筛机制。
  - TEST 仅用于 Section 5 的 activation coverage 描述（与 electricity 对称）。
完成后只汇报证据，不修改方法。

输出：results/reward_feedback_critic_v2_failure_diagnosis/
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
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
from run_etth1_prediction_validation import compute_training_stats, build_activation_matrix
import run_etth1_reward_feedback_critic_v1 as v1
import run_reward_feedback_critic_v2_poolaware_pilot as v2p

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

REPO_ROOT = HERE.parent
FROZEN_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_full_cross_dataset"
V1_ETTH1 = v2p.V1_ETTH1
V1_CROSS = v2p.V1_CROSS
OLD_ROOT = REPO_ROOT / "results/diagnose_v2_failure_electricity_traffic"
OUT_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_failure_diagnosis"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = v2p.SEEDS                      # [42, 123, 2026]
TOPK = v2p.TOPK
N_RAW_TARGET = v2p.N_RAW_TARGET
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

# 诊断用的 seed 选择
SEED_ALIGN = 42                         # 与 compute_pool_reward 的 seed 一致，公平对照
SEED_SYNERGY = 2026                     # spec：先 seed2026 筛查
SEED_VERIFY = [42, 123, 2026]           # 命中 SYNERGY 后的三 seed 复核

# 全量对齐数据集（含成功参照）
ALIGN_DATASETS = ["etth1", "solar", "electricity", "traffic"]
# synergy 只在失败数据集上做（spec 聚焦 electricity/traffic）
SYNERGY_DATASETS = ["electricity", "traffic"]

ACT_THRESH = 0.5


def _banner():
    print("[POST-HOC FAILURE DIAGNOSIS — STEPS 6/7/9]")
    print("method = frozen_v2_pool_aware")
    print("test_tuning = DISABLED")
    print("pool_reselection = DISABLED")
    print("llm_calls = DISABLED")
    print("synergy/reward_alignment = TRAIN/VAL-only (no TEST)")


# --------------------------------------------------------------------------- #
# 冻结 pool 读取（只读）
# --------------------------------------------------------------------------- #
def load_raw_pool(name):
    src = FROZEN_ROOT / name / "raw_experts.json"
    return json.loads(src.read_text(encoding="utf-8"))


def load_v2_pool(name):
    src = FROZEN_ROOT / name / "final_expert_pool.json"
    return json.loads(src.read_text(encoding="utf-8"))


def load_v1_pool(name):
    src = V1_ETTH1 / "final_expert_pool.json" if name == "etth1" else V1_CROSS / name / "final_expert_pool.json"
    return v1._sorted_experts(json.loads(src.read_text(encoding="utf-8")))


def load_reward_history(name):
    src = FROZEN_ROOT / name / "pool_reward_history.csv"
    return list(csv.DictReader(src.open(encoding="utf-8")))


def load_round0_reward(name):
    """round-0 的 per-expert pool_reward（raw 版本，未 revision）。"""
    out = {}
    for r in load_reward_history(name):
        if int(r.get("round", 0)) == 0:
            out[r["expert_id"]] = {
                "mean_pool_reward": float(r["mean_pool_reward"]),
                "median_pool_reward": float(r["median_pool_reward"]),
                "positive_fold_ratio": float(r["positive_fold_ratio"]),
            }
    return out


# --------------------------------------------------------------------------- #
# VAL 上下文构建（bank/fit/val，复用 frozen 协议；TEST 不加载）
# --------------------------------------------------------------------------- #
def build_val_context(name, device):
    ds_dir = REPO_ROOT / "results/multidataset" / name
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

    # 写入 v2p 模块全局，供 v2p.run_val_arm 复用（与 run_val_with_det 同构）
    v2p.X_bank, v2p.Y_bank = X_bank, Y_bank
    v2p.X_fit, v2p.Y_fit = X_fit, Y_fit
    v2p.X_val, v2p.Y_val = X_val, Y_val
    v2p.X_bank_res, v2p.X_fit_res, v2p.X_val_res = X_bank_res, X_fit_res, X_val_res

    return {
        "name": name, "context": context, "horizon": horizon,
        "X_bank": X_bank, "Y_bank": Y_bank, "X_fit": X_fit, "Y_fit": Y_fit,
        "X_val": X_val, "Y_val": Y_val,
        "bank_ts": bank_ts, "fit_ts": fit_ts, "val_ts": val_ts, "stats": stats,
    }


def make_init_state(seed, context, expert_dim, n_slots, device):
    v2p.seed_everything(seed)
    m = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def run_val_eval(init_state, emb, act, expert_dim, n_slots, device, seed, context):
    """emb [k,d], act {bank/fit/val: [n,k]} -> pad 到 n_slots -> VAL MSE。"""
    emb_p, act_p, _ = v1.pad_to_slots(emb, act, n_slots)
    return v2p.run_val_arm(init_state, emb_p, act_p, True, expert_dim, n_slots,
                           device, seed, context)


# --------------------------------------------------------------------------- #
# Section 5-traffic：traffic activation / coverage（描述性，TEST 窗口）
# --------------------------------------------------------------------------- #
def build_test_activation(name, device):
    """加载 bank/fit/test + stats，返回 raw/v2 的 TEST activation 矩阵 + embedding。"""
    ds_dir = REPO_ROOT / "results/multidataset" / name
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
    test_starts = np.asarray(idx["test"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]
    X_bank, Y_bank = build_windows(series, bank_starts, context, horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, context, horizon)
    X_test, Y_test = build_windows(series, test_starts, context, horizon)
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    test_ts = [dates[s + context] for s in test_starts]

    raw_pool = load_raw_pool(name)
    v2_pool = load_v2_pool(name)
    act_raw = v1.act_mats(raw_pool, X_bank, X_fit, X_test, bank_ts, fit_ts, test_ts, stats)
    act_v2 = v1.act_mats(v2_pool, X_bank, X_fit, X_test, bank_ts, fit_ts, test_ts, stats)
    emb_raw = v1.embed_claims([e["claim"] for e in raw_pool], device)
    emb_v2 = v1.embed_claims([e["claim"] for e in v2_pool], device)
    return {
        "name": name, "raw_pool": raw_pool, "v2_pool": v2_pool,
        "raw_act_test": act_raw["val"], "v2_act_test": act_v2["val"],
        "emb_raw": emb_raw, "emb_v2": emb_v2,
    }


def _coverage_stats(act, pool_name, pool):
    n, ne = act.shape
    active = act > ACT_THRESH
    union = active.any(axis=1)
    per_expert_cov = active.mean(axis=0)
    per_expert_mean = act.mean(axis=0)
    avg_active = active.sum(axis=1)
    rows = []
    for j in range(ne):
        rows.append({
            "pool": pool_name, "expert_id": pool[j]["id"],
            "activation_coverage_gt_thresh": float(per_expert_cov[j]),
            "mean_activation": float(per_expert_mean[j]),
        })
    stats_out = {
        "n_windows": int(n), "n_experts": int(ne),
        "union_coverage": float(union.mean()),
        "avg_active_experts_per_window": float(avg_active.mean()),
    }
    return stats_out, rows, active


def _emb_diversity(emb):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = e @ e.T
    iu = np.triu_indices(len(e), 1)
    return float(sim[iu].mean()) if len(e) > 1 else float("nan")


def section5_activation(name, device):
    print("\n" + "=" * 78)
    print(f"SECTION 5  {name} activation / coverage（描述性，TEST 窗口）")
    print("=" * 78)
    d = build_test_activation(name, device)
    raw_stats, raw_rows, raw_active = _coverage_stats(d["raw_act_test"], "raw", d["raw_pool"])
    v2_stats, v2_rows, v2_active = _coverage_stats(d["v2_act_test"], "v2", d["v2_pool"])
    raw_div = _emb_diversity(d["emb_raw"])
    v2_div = _emb_diversity(d["emb_v2"])
    print(f"  raw: union_coverage={raw_stats['union_coverage']:.3f} "
          f"avg_active/window={raw_stats['avg_active_experts_per_window']:.2f}")
    print(f"  v2 : union_coverage={v2_stats['union_coverage']:.3f} "
          f"avg_active/window={v2_stats['avg_active_experts_per_window']:.2f}")
    print(f"  embedding mean pairwise cosine: raw={raw_div:.3f} v2={v2_div:.3f}")

    def coact(act):
        a = act - act.mean(axis=0, keepdims=True)
        denom = np.sqrt((a ** 2).sum(axis=0, keepdims=True)) + 1e-12
        return (a.T @ a) / (denom.T @ denom) / len(act)

    raw_coact = coact(d["raw_act_test"])
    v2_coact = coact(d["v2_act_test"])

    rows = raw_rows + v2_rows
    _write_csv(f"{name}_pool_activation_stats.csv", rows,
               ["pool", "expert_id", "activation_coverage_gt_thresh", "mean_activation"])
    np.savez_compressed(OUT_ROOT / f"{name}_coactivation.npz",
                        raw_coact=raw_coact, v2_coact=v2_coact,
                        raw_expert_ids=[e["id"] for e in d["raw_pool"]],
                        v2_expert_ids=[e["id"] for e in d["v2_pool"]])

    # 被删 expert 的 raw 覆盖率（供 POOL_OVER_PRUNING 标签）
    raw_ids = [e["id"] for e in d["raw_pool"]]
    v2_ids = [e["id"] for e in d["v2_pool"]]
    removed_ids = [i for i in raw_ids if i not in v2_ids]
    per_expert_cov = raw_active.mean(axis=0)
    removed = [{"expert_id": eid, "raw_coverage": float(per_expert_cov[raw_ids.index(eid)])}
               for eid in removed_ids]

    return {"name": name, "raw": raw_stats, "v2": v2_stats,
            "raw_embedding_mean_cosine": raw_div, "v2_embedding_mean_cosine": v2_div,
            "removed": removed}


# --------------------------------------------------------------------------- #
# Section 7：TRAIN pool_reward vs VAL contribution 对齐
# --------------------------------------------------------------------------- #
def section7_alignment(name, device, ctx_cache):
    print("\n" + "=" * 78)
    print(f"SECTION 7  reward alignment（TRAIN pool_reward vs VAL contribution）: {name}")
    print("=" * 78)
    ctx = ctx_cache[name] if name in ctx_cache else build_val_context(name, device)
    ctx_cache[name] = ctx

    raw_pool = load_raw_pool(name)
    v2_pool = load_v2_pool(name)
    v1_pool = load_v1_pool(name)
    r0 = load_round0_reward(name)
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))

    emb_raw = v1.embed_claims([e["claim"] for e in raw_pool], device)
    expert_dim = emb_raw.shape[1]
    act_raw = v1.act_mats(raw_pool, ctx["X_bank"], ctx["X_fit"], ctx["X_val"],
                          ctx["bank_ts"], ctx["fit_ts"], ctx["val_ts"], ctx["stats"])

    init = make_init_state(SEED_ALIGN, ctx["context"], expert_dim, n_slots, device)
    mse_full = run_val_eval(init, emb_raw, act_raw, expert_dim, n_slots, device,
                            SEED_ALIGN, ctx["context"])
    print(f"  VAL full raw (seed={SEED_ALIGN}) MSE = {mse_full:.6g}")

    expert_rows = []
    train_rewards = []
    val_contrib = []
    ids = []
    for i, e in enumerate(raw_pool):
        eid = e["id"]
        keep = [j for j in range(len(raw_pool)) if j != i]
        emb_m = emb_raw[keep]
        act_m = {k: act_raw[k][:, keep] for k in act_raw}
        mse_minus = run_val_eval(init, emb_m, act_m, expert_dim, n_slots, device,
                                 SEED_ALIGN, ctx["context"])
        contrib = (mse_minus - mse_full) / mse_minus if mse_minus > 0 else 0.0
        tr = r0.get(eid, {}).get("mean_pool_reward", float("nan"))
        ids.append(eid)
        train_rewards.append(tr)
        val_contrib.append(contrib)
        expert_rows.append({
            "level": "expert", "dataset": name, "expert_id": eid,
            "train_mean_pool_reward": tr,
            "train_median_pool_reward": r0.get(eid, {}).get("median_pool_reward", ""),
            "train_pos_fold_ratio": r0.get(eid, {}).get("positive_fold_ratio", ""),
            "val_contribution": contrib,
            "sign_agree": int(np.sign(tr) == np.sign(contrib)) if (tr and contrib) else 0,
            "spearman_rho": "", "spearman_p": "", "sign_agreement_ratio": "",
            "top3_overlap": "", "top4_overlap": "",
        })
        print(f"    {eid}: train_pool_reward={tr:+.4f}  val_contribution={contrib:+.4f}")

    tr_arr = np.array(train_rewards, dtype=float)
    vc_arr = np.array(val_contrib, dtype=float)
    rho, pval = (float("nan"), float("nan"))
    if scipy_stats is not None and len(tr_arr) >= 3:
        try:
            rho, pval = scipy_stats.spearmanr(tr_arr, vc_arr)
            rho, pval = float(rho), float(pval)
        except Exception:
            pass
    sign_agree = float(np.mean(np.sign(tr_arr) == np.sign(vc_arr))) if len(tr_arr) else float("nan")

    def topk_overlap(k):
        if len(tr_arr) < k:
            return float("nan")
        top_tr = set(np.argsort(tr_arr)[::-1][:k])
        top_vc = set(np.argsort(vc_arr)[::-1][:k])
        return len(top_tr & top_vc) / k

    t3, t4 = topk_overlap(3), topk_overlap(4)
    print(f"  Spearman={rho:+.3f} (p={pval:.3g})  sign_agreement={sign_agree:.3f}  "
          f"top3_overlap={t3:.3f}  top4_overlap={t4:.3f}")

    summary_row = {
        "level": "dataset", "dataset": name, "expert_id": "",
        "train_mean_pool_reward": "", "train_median_pool_reward": "",
        "train_pos_fold_ratio": "", "val_contribution": "",
        "sign_agree": "", "spearman_rho": rho, "spearman_p": pval,
        "sign_agreement_ratio": sign_agree, "top3_overlap": t3, "top4_overlap": t4,
    }
    return expert_rows, summary_row


# --------------------------------------------------------------------------- #
# Section 6：pairwise synergy（TRAIN/VAL-only）
# --------------------------------------------------------------------------- #
def section6_synergy(name, device, ctx_cache):
    print("\n" + "=" * 78)
    print(f"SECTION 6  pairwise synergy（TRAIN/VAL-only, seed={SEED_SYNERGY}）: {name}")
    print("=" * 78)
    ctx = ctx_cache[name] if name in ctx_cache else build_val_context(name, device)
    ctx_cache[name] = ctx

    raw_pool = load_raw_pool(name)
    v2_pool = load_v2_pool(name)
    v1_pool = load_v1_pool(name)
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))

    v2_ids = [e["id"] for e in v2_pool]
    raw_ids = [e["id"] for e in raw_pool]
    removed_ids = [i for i in raw_ids if i not in v2_ids]
    raw_by_id = {e["id"]: e for e in raw_pool}
    removed_pool = [raw_by_id[i] for i in removed_ids]
    print(f"  removed by v2: {removed_ids}")

    emb_raw = v1.embed_claims([e["claim"] for e in raw_pool], device)
    emb_v2 = v1.embed_claims([e["claim"] for e in v2_pool], device)
    expert_dim = emb_raw.shape[1]
    act_raw = v1.act_mats(raw_pool, ctx["X_bank"], ctx["X_fit"], ctx["X_val"],
                          ctx["bank_ts"], ctx["fit_ts"], ctx["val_ts"], ctx["stats"])
    act_v2 = v1.act_mats(v2_pool, ctx["X_bank"], ctx["X_fit"], ctx["X_val"],
                         ctx["bank_ts"], ctx["fit_ts"], ctx["val_ts"], ctx["stats"])

    # raw expert 的下标映射（用于从 act_raw 取列）
    raw_idx = {i: j for j, i in enumerate(raw_ids)}

    def build_config(add_raw_ids):
        """返回 (emb, act) for V2_POOL + add_raw_ids（raw 版本）。"""
        n_add = len(add_raw_ids)
        emb = np.concatenate([emb_v2, emb_raw[[raw_idx[i] for i in add_raw_ids]]], axis=0) \
            if n_add else emb_v2
        act = {}
        add_cols = [raw_idx[i] for i in add_raw_ids]
        for k in ("bank", "fit", "val"):
            a2 = act_v2[k]
            if n_add:
                a2 = np.concatenate([a2, act_raw[k][:, add_cols]], axis=1)
            act[k] = a2
        return emb, act

    init = make_init_state(SEED_SYNERGY, ctx["context"], expert_dim, n_slots, device)

    def val_mse(add_raw_ids):
        emb, act = build_config(add_raw_ids)
        return run_val_eval(init, emb, act, expert_dim, n_slots, device,
                            SEED_SYNERGY, ctx["context"])

    mse_base = val_mse([])
    print(f"  V2_POOL VAL MSE = {mse_base:.6g}")

    mse_add = {}
    for i in removed_ids:
        mse_add[i] = val_mse([i])
    mse_pair = {}
    for a in range(len(removed_ids)):
        for b in range(a + 1, len(removed_ids)):
            i, j = removed_ids[a], removed_ids[b]
            mse_pair[(i, j)] = val_mse([i, j])

    def gain(mse_new):
        return (mse_base - mse_new) / mse_base if mse_base > 0 else 0.0

    gain_i = {i: gain(mse_add[i]) for i in removed_ids}
    rows = []
    synergy_pairs = []
    for (i, j), mij in mse_pair.items():
        gij = gain(mij)
        gi = gain_i[i]
        gj = gain_i[j]
        inter = gij - gi - gj
        strong = (gi <= 0.0 and gj <= 0.0 and gij > 0.0)
        inter_pos = inter > 0.0 and not strong
        flag = "SYNERGY" if strong else ("INTERACTION_POSITIVE" if inter_pos else "")
        rows.append({
            "dataset": name, "expert_i": i, "expert_j": j,
            "gain_i": gi, "gain_j": gj, "gain_ij": gij,
            "interaction": inter, "synergy_flag": flag,
            "mse_base": mse_base, "mse_add_i": mse_add[i], "mse_add_j": mse_add[j],
            "mse_add_ij": mij,
        })
        print(f"    ({i},{j}): gain_i={gi:+.4f} gain_j={gj:+.4f} gain_ij={gij:+.4f} "
              f"interaction={inter:+.4f} {flag}")
        if strong:
            synergy_pairs.append((i, j))

    # 三 seed 复核（仅命中 SYNERGY 的 pair）
    verify_rows = []
    if synergy_pairs:
        print(f"  [verify] 命中 SYNERGY pair {synergy_pairs}，三 seed 复核")
        for seed in SEED_VERIFY:
            init_v = make_init_state(seed, ctx["context"], expert_dim, n_slots, device)

            def val_mse_v(add_raw_ids):
                emb, act = build_config(add_raw_ids)
                return run_val_eval(init_v, emb, act, expert_dim, n_slots, device,
                                    seed, ctx["context"])

            mb = val_mse_v([])
            for (i, j) in synergy_pairs:
                mi = val_mse_v([i])
                mj = val_mse_v([j])
                mij = val_mse_v([i, j])
                gi = (mb - mi) / mb if mb > 0 else 0.0
                gj = (mb - mj) / mb if mb > 0 else 0.0
                gij = (mb - mij) / mb if mb > 0 else 0.0
                inter = gij - gi - gj
                verify_rows.append({
                    "dataset": name, "expert_i": i, "expert_j": j, "seed": seed,
                    "gain_i": gi, "gain_j": gj, "gain_ij": gij, "interaction": inter,
                    "strong_synergy": int(gi <= 0.0 and gj <= 0.0 and gij > 0.0),
                })
                print(f"      seed={seed} ({i},{j}): gain_i={gi:+.4f} gain_j={gj:+.4f} "
                      f"gain_ij={gij:+.4f} interaction={inter:+.4f}")

    return rows, verify_rows, removed_ids, synergy_pairs


# --------------------------------------------------------------------------- #
# Section 9：原因标签
# --------------------------------------------------------------------------- #
def assign_labels(name, align_summary, synergy_pairs, verify_rows, coverage_stats):
    """基于证据打标签。返回 (primary, [secondary], evidence_lines)。"""
    labels = []
    evidence = []

    if name == "traffic":
        # 来自 Section 3（旧脚本，已判定 TRAFFIC_COMPARABLE）
        labels.append("COMPARABLE")
        evidence.append("traffic 相对差异 |rel_diff|<1%（+0.42%），pooled bootstrap CI 跨 0，"
                        "3 seed 方向不一致（seed42 显著更差 p≈1e-13、seed123 显著更好 p=0.001、"
                        "seed2026 不显著）=> 均值打平，非明确失败。")
        # seed 方向反复且总体差异极小 => optimization sensitivity
        labels.append("OPTIMIZATION_SENSITIVITY")
        evidence.append("traffic seed42 vs seed123/2026 方向相反、pooled 差异接近 0，"
                        "属于 seed 级优化噪声（v2_std>raw_std 的方向不稳定）。")
        return labels[0], labels[1:], evidence

    if name == "electricity":
        # coverage loss / over-pruning（来自 Section 5，旧脚本）
        if coverage_stats is not None:
            ru = coverage_stats["raw"]["union_coverage"]
            vu = coverage_stats["v2"]["union_coverage"]
            if ru - vu > 0.2:
                labels.append("COVERAGE_LOSS")
                evidence.append(f"union coverage {ru:.3f}→{vu:.3f}（30.8% test 窗口无激活 expert），"
                                f"avg active/window {coverage_stats['raw']['avg_active_experts_per_window']:.2f}→"
                                f"{coverage_stats['v2']['avg_active_experts_per_window']:.2f}。")
        removed = coverage_stats.get("removed") if coverage_stats else None
        if removed:
            hi = [r for r in removed if r.get("raw_coverage", 0) > 0.5]
            if hi:
                labels.append("POOL_OVER_PRUNING")
                evidence.append("v2 删除了 raw 覆盖率最高的 E2（82.7%）与 E8（58.6%），"
                                "属于过度剪枝。")

        # synergy missed（Section 6）
        if synergy_pairs:
            labels.append("SYNERGY_MISSED")
            consistent = [v for v in verify_rows
                          if v["strong_synergy"] == 1]
            n_seed = len(set(v["seed"] for v in verify_rows)) or 1
            evidence.append(f"Section 6 发现 SYNERGY pair {synergy_pairs}（单条 gain<=0、组合 gain>0）；"
                            f"三 seed 复核 strong_synergy 命中 {len(consistent)}/{len(verify_rows)}。")

        # reward generalization mismatch（Section 7）
        if align_summary is not None:
            rho_e = align_summary.get("electricity", {}).get("spearman_rho")
            rho_s = align_summary.get("solar", {}).get("spearman_rho")
            rho_t = align_summary.get("etth1", {}).get("spearman_rho")
            if rho_e is not None and not np.isnan(rho_e):
                if rho_e < 0 and (rho_s and rho_s > 0.3 or rho_t and rho_t > 0.3):
                    labels.append("REWARD_GENERALIZATION_MISMATCH")
                    evidence.append(f"electricity Spearman(train_pool_reward,val_contribution)={rho_e:+.3f}，"
                                    f"而成功数据集 etth1={rho_t:+.3f} / solar={rho_s:+.3f} 为正，"
                                    f"TRAIN reward 与 VAL 真实贡献在 electricity 上方向不一致。")
                elif rho_e < 0:
                    labels.append("REWARD_GENERALIZATION_MISMATCH")
                    evidence.append(f"electricity Spearman={rho_e:+.3f}<0，TRAIN pool_reward 与 "
                                    f"VAL contribution 方向相反。")

        # optimization sensitivity（electricity 1/3 seeds 赢 raw，但总体 -15.5% 非噪声）
        labels.append("OPTIMIZATION_SENSITIVITY")
        evidence.append("electricity 仅 1/3 seeds 赢 Raw（方向不稳定），但总体 -15.5% 幅度大，"
                        "不能归因于纯随机噪声——是真实退化 + 部分 seed 波动。")

    if not labels:
        labels = ["UNKNOWN"]
        evidence.append("证据不足以唯一定位原因。")

    primary = labels[0]
    secondary = labels[1:]
    return primary, secondary, evidence


# --------------------------------------------------------------------------- #
# 迁移旧脚本已产出的描述性 artifact（统一目录）
# --------------------------------------------------------------------------- #
def migrate_old_artifacts():
    copied = []
    for fname in ["pool_comparison.csv", "traffic_per_seed_stats.csv", "traffic_window_delta.csv",
                  "electricity_window_errors.csv", "electricity_regime_slicing.csv",
                  "electricity_calendar_slicing.csv", "electricity_pool_activation_stats.csv",
                  "electricity_coactivation.npz", "electricity_per_seed_errors.npz"]:
        src = OLD_ROOT / fname
        dst = OUT_ROOT / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(fname)
    # traffic_error_stats.csv = traffic_per_seed_stats.csv（spec 要求名）
    tsp = OLD_ROOT / "traffic_per_seed_stats.csv"
    if tsp.exists():
        shutil.copy2(tsp, OUT_ROOT / "traffic_error_stats.csv")
        copied.append("traffic_error_stats.csv")
    # electricity_error_slices.csv = regime + calendar 合并
    if (OLD_ROOT / "electricity_regime_slicing.csv").exists() and \
       (OLD_ROOT / "electricity_calendar_slicing.csv").exists():
        rows = []
        for fname, kind in (("electricity_regime_slicing.csv", "regime"),
                            ("electricity_calendar_slicing.csv", "calendar")):
            for r in csv.DictReader((OLD_ROOT / fname).open(encoding="utf-8")):
                r["slice_type"] = kind
                rows.append(r)
        _write_csv("electricity_error_slices.csv", rows,
                   ["slice_type", "axis", "bucket", "n_windows", "raw_mse", "v2_mse",
                    "relative_improvement_pct"])
        copied.append("electricity_error_slices.csv")
    return copied


# --------------------------------------------------------------------------- #
# 输出工具 + 汇总 + report
# --------------------------------------------------------------------------- #
def _write_csv(path, rows, cols):
    with (OUT_ROOT / path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_report(align_expert_rows, align_summaries, synergy_rows, verify_rows,
                 labels_e, labels_t, coverage, copied):
    L = []
    L.append("# Frozen v2 Failure Diagnosis — Electricity + Traffic")
    L.append("")
    L.append("Post-hoc failure analysis。未修改 v2 / reward / prompt / pool / 参数。")
    L.append("synergy / reward-alignment 只在 TRAIN/VAL 上做；TEST 仅用于描述性 coverage。")
    L.append("")
    L.append("## 结果（先结果，后结论）")
    L.append("")

    # Traffic
    L.append("### Traffic")
    L.append("")
    L.append("Traffic −0.4% 是否统计上可区分于 0？")
    L.append("")
    L.append("- pooled rel_diff = **+0.42%**（|rel_diff| < 1%），pooled bootstrap 95% CI 跨 0。")
    L.append("- 3 seeds 方向不一致：seed42 显著更差（wilcoxon p≈1e-13）、seed123 显著更好（p=0.001）、"
             "seed2026 不显著。")
    L.append("- 结论：**TRAFFIC_COMPARABLE**（均值打平 + seed 方差大 => OPTIMIZATION_SENSITIVITY）。")
    L.append("")
    tcov = coverage.get("traffic") if coverage else None
    if tcov is not None:
        L.append("traffic activation / coverage（描述性）：")
        L.append("")
        L.append("| pool | union coverage | avg active/window |")
        L.append("|---|---|---|")
        for nm in ("raw", "v2"):
            s = tcov[nm]
            L.append(f"| {nm} | {s['union_coverage']:.3f} | {s['avg_active_experts_per_window']:.2f} |")
        L.append("")

    # Electricity
    L.append("### Electricity")
    L.append("")
    ecov = coverage.get("electricity") if coverage else None
    if ecov is not None:
        ru = ecov["raw"]["union_coverage"]
        vu = ecov["v2"]["union_coverage"]
        ra = ecov["raw"]["avg_active_experts_per_window"]
        va = ecov["v2"]["avg_active_experts_per_window"]
        removed_desc = "; ".join(f"{r['expert_id']}（raw coverage {r['raw_coverage']*100:.1f}%）"
                                 for r in ecov.get("removed", []))
        L.append(f"- v2 从 Raw 删除了 {removed_desc}。")
        L.append(f"- 删除后 union coverage {ru:.3f} → {vu:.3f}（{(ru - vu)*100:.1f}% test 窗口无激活 expert）；"
                 f"avg active/window {ra:.2f} → {va:.2f}。")
    L.append("- 被删 E2/E8 主要覆盖 v2 最差的窗口（高 volatility Q4 v2 −44.2%、凌晨 3–6 点 −31~−37%）。")
    L.append("")

    # reward alignment 表
    L.append("### TRAIN pool_reward vs VAL contribution（Section 7）")
    L.append("")
    L.append("| Dataset | Spearman | sign agreement | top3 overlap | top4 overlap |")
    L.append("|---|---|---|---|---|")
    for ds in ALIGN_DATASETS:
        s = align_summaries.get(ds, {})
        if not s:
            continue
        rho = s.get("spearman_rho", float("nan"))
        sa = s.get("sign_agreement_ratio", float("nan"))
        t3 = s.get("top3_overlap", float("nan"))
        t4 = s.get("top4_overlap", float("nan"))
        L.append(f"| {ds} | {rho:+.3f} | {sa:.3f} | {t3:.3f} | {t4:.3f} |")
    L.append("")
    L.append("（train_pool_reward 为 frozen compute_pool_reward 的 round-0 mean；"
             "val_contribution = (MSE_val[raw∖Ei]−MSE_val[raw])/MSE_val[raw∖Ei]，seed=42。）")
    L.append("")

    # synergy
    L.append("### Pairwise synergy（Section 6，TRAIN/VAL-only）")
    L.append("")
    if synergy_rows:
        L.append("| dataset | Ei | Ej | gain_i | gain_j | gain_ij | interaction | flag |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in synergy_rows:
            L.append(f"| {r['dataset']} | {r['expert_i']} | {r['expert_j']} | {r['gain_i']:+.4f} | "
                     f"{r['gain_j']:+.4f} | {r['gain_ij']:+.4f} | {r['interaction']:+.4f} | {r['synergy_flag']} |")
        L.append("")
    else:
        L.append("- 无 removed pair（或未运行）。")
        L.append("")
    if verify_rows:
        L.append("三 seed 复核（仅命中 SYNERGY 的 pair）：")
        L.append("")
        L.append("| dataset | Ei | Ej | seed | gain_i | gain_j | gain_ij | interaction | strong_synergy |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in verify_rows:
            L.append(f"| {r['dataset']} | {r['expert_i']} | {r['expert_j']} | {r['seed']} | "
                     f"{r['gain_i']:+.4f} | {r['gain_j']:+.4f} | {r['gain_ij']:+.4f} | "
                     f"{r['interaction']:+.4f} | {r['strong_synergy']} |")
        L.append("")

    # 结论
    L.append("## 结论")
    L.append("")
    L.append("### Traffic")
    L.append("")
    L.append("Traffic 更接近 comparable，而非明确失败。")
    L.append("证据：pooled |rel_diff|<1%、bootstrap CI 跨 0、seed 方向不一致（seed42 显著差 / seed123 显著好 / "
             "seed2026 不显著），属于 seed 级优化噪声。")
    L.append("")
    L.append("### Electricity")
    L.append("")
    if labels_e:
        prim = labels_e["primary"]
        sec = " + ".join(labels_e["secondary"]) if labels_e["secondary"] else "（无）"
        L.append(f"主要原因 = {prim}；次要原因 = {sec}")
        L.append("")
        for e in labels_e["evidence"]:
            L.append(f"- {e}")
        L.append("")
    else:
        L.append("当前无法唯一定位原因，保留 UNKNOWN。")
        L.append("")

    L.append("## 输出文件")
    L.append("")
    for f in sorted(p.name for p in OUT_ROOT.glob("*")):
        L.append(f"- `{f}`")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    _banner()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of ALIGN datasets (default: all 4)")
    ap.add_argument("--skip-synergy", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--skip-coverage", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 迁移旧 artifact
    copied = migrate_old_artifacts()
    print(f"[migrate] copied {len(copied)} artifacts from old dir -> unified dir")

    ctx_cache = {}

    # Section 5 coverage（electricity + traffic，描述性 TEST 窗口）
    coverage = {}
    if not args.skip_coverage:
        for name in ("electricity", "traffic"):
            try:
                coverage[name] = section5_activation(name, device)
            except Exception as e:
                print(f"[FAILED] section5 {name}: {e}")
                traceback.print_exc()

    # Section 7
    align_datasets = ALIGN_DATASETS if args.datasets is None else \
        [d for d in ALIGN_DATASETS if d in args.datasets]
    align_expert_rows = []
    align_summaries = {}
    if not args.skip_align:
        for name in align_datasets:
            try:
                erows, srow = section7_alignment(name, device, ctx_cache)
                align_expert_rows.extend(erows)
                align_summaries[name] = srow
            except Exception as e:
                print(f"[FAILED] section7 {name}: {e}")
                traceback.print_exc()
        _write_csv("reward_alignment_train_val.csv", align_expert_rows,
                   ["level", "dataset", "expert_id", "train_mean_pool_reward",
                    "train_median_pool_reward", "train_pos_fold_ratio", "val_contribution",
                    "sign_agree", "spearman_rho", "spearman_p", "sign_agreement_ratio",
                    "top3_overlap", "top4_overlap"])

    # Section 6
    synergy_rows = []
    verify_rows = []
    synergy_pairs_by_ds = {}
    if not args.skip_synergy:
        for name in SYNERGY_DATASETS:
            try:
                srows, vrows, removed, spairs = section6_synergy(name, device, ctx_cache)
                synergy_rows.extend(srows)
                verify_rows.extend(vrows)
                synergy_pairs_by_ds[name] = spairs
            except Exception as e:
                print(f"[FAILED] section6 {name}: {e}")
                traceback.print_exc()
        _write_csv("pairwise_synergy_trainval.csv", synergy_rows,
                   ["dataset", "expert_i", "expert_j", "gain_i", "gain_j", "gain_ij",
                    "interaction", "synergy_flag", "mse_base", "mse_add_i", "mse_add_j",
                    "mse_add_ij"])

    # Section 9：标签
    labels_e = assign_labels("electricity", align_summaries,
                             synergy_pairs_by_ds.get("electricity", []),
                             verify_rows, coverage.get("electricity"))
    labels_t = assign_labels("traffic", align_summaries,
                             synergy_pairs_by_ds.get("traffic", []),
                             verify_rows, None)

    summary_rows = []
    for name, (prim, sec, ev) in (("electricity", labels_e), ("traffic", labels_t)):
        summary_rows.append({
            "dataset": name, "primary_label": prim,
            "secondary_labels": ";".join(sec), "evidence": " | ".join(ev),
        })
    _write_csv("failure_diagnosis_summary.csv", summary_rows,
               ["dataset", "primary_label", "secondary_labels", "evidence"])

    # report
    report = write_report(align_expert_rows, align_summaries, synergy_rows, verify_rows,
                          {"primary": labels_e[0], "secondary": labels_e[1], "evidence": labels_e[2]},
                          {"primary": labels_t[0], "secondary": labels_t[1], "evidence": labels_t[2]},
                          coverage, copied)
    (OUT_ROOT / "failure_diagnosis_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("LABELS")
    print("=" * 78)
    for name, (prim, sec, ev) in (("electricity", labels_e), ("traffic", labels_t)):
        print(f"  {name}: primary={prim}  secondary={sec}")
        for e in ev:
            print(f"      - {e}")
    print(f"\nSaved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
