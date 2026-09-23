#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc failure diagnosis for frozen v2 on electricity + traffic.

只诊断：为什么 frozen v2（Pool-Aware Reward）在 electricity / traffic 上没有超过 Raw。

这是 post-hoc failure analysis，不是继续开发方法。禁令：
  - 不修改 v2 / reward / Critic / Expert prompt。
  - 不根据 TEST 重新选 pool / 调参数 / 做 subset search 后重报主结果。
  - 不做 v3。不用诊断结果覆盖 frozen TEST 主表。
  - TEST 只做描述性分析；需要解释 Expert 选择机制的检查只在 TRAIN/VAL 上做（本脚本不碰）。

原则：
  - 直接 import 已冻结实现（v2p / v1 / v4）与已有 artifact，不调 LLM。
  - 只重跑 raw + v2 两个 arm（换 TEST split、完全相同冻结模型），逐窗口捕获预测，
    逐窗口平方误差回验聚合 MSE == 冻结 result.json（保证重跑没跑偏）。
  - 读取：
      results/reward_feedback_critic_v2_full_cross_dataset/   （冻结 pool 来源）
      results/reward_feedback_critic_v2_final_test/           （冻结 TEST 主表，只读对照）
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
from run_etth1_prediction_validation import (
    compute_training_stats,
    compute_window_feats,
    build_activation_matrix,
)
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
TEST_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_final_test"
OUT_ROOT = REPO_ROOT / "results/diagnose_v2_failure_electricity_traffic"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = v2p.SEEDS
TOPK = v2p.TOPK
N_RAW_TARGET = v2p.N_RAW_TARGET
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

DATASETS = [
    {"name": "electricity", "domain": "aggregate_electricity_load", "freq": "15-minute"},
    {"name": "traffic",     "domain": "road_traffic_occupancy",   "freq": "hourly"},
]

ACT_THRESH = 0.5          # activation 视为“激活”的阈值（activation ∈ [0,1]，sigmoid）
BOOT_SEED = 20260916
BOOT_N = 10000


class ArtifactMissingError(RuntimeError):
    pass


def _banner():
    print("[POST-HOC FAILURE DIAGNOSIS]")
    print("method = frozen_v2_pool_aware")
    print("datasets = [electricity, traffic]")
    print("test_tuning = DISABLED")
    print("pool_reselection = DISABLED")
    print("llm_calls = DISABLED")


# --------------------------------------------------------------------------- #
# 冻结 pool 读取（只读，与 final test 完全一致）
# --------------------------------------------------------------------------- #
def load_raw_pool(name):
    src = FROZEN_ROOT / name / "raw_experts.json"
    if src.exists():
        return json.loads(src.read_text(encoding="utf-8"))
    raw_src = V1_ETTH1 / "raw_experts.json" if name == "etth1" else V1_CROSS / name / "raw_experts.json"
    experts = v1._sorted_experts(json.loads(raw_src.read_text(encoding="utf-8")))[:N_RAW_TARGET]
    for e in experts:
        e["claim"] = e.get("mechanism", e.get("reason", ""))
        e["observable_features"] = list(e.get("observables", []))
        e["replaced"] = False
    return experts


def load_v1_pool(name):
    src = V1_ETTH1 / "final_expert_pool.json" if name == "etth1" else V1_CROSS / name / "final_expert_pool.json"
    return v1._sorted_experts(json.loads(src.read_text(encoding="utf-8")))


def load_v2_pool(name):
    src = FROZEN_ROOT / name / "final_expert_pool.json"
    if not src.exists():
        raise ArtifactMissingError(f"missing frozen v2 final pool: {src}")
    return json.loads(src.read_text(encoding="utf-8"))


def load_reward_history(name):
    src = FROZEN_ROOT / name / "pool_reward_history.csv"
    if not src.exists():
        return []
    return list(csv.DictReader(src.open(encoding="utf-8")))


def load_frozen_test(name):
    """冻结 TEST result.json（只读对照，用于回验聚合 MSE）。"""
    rp = TEST_ROOT / name / "result.json"
    if not rp.exists():
        return None
    return json.loads(rp.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Section 2：pool 变化
# --------------------------------------------------------------------------- #
def _short(s, n=120):
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def section2_pool_comparison(name, raw_pool, v1_pool, v2_pool, hist):
    raw_ids = [e["id"] for e in raw_pool]
    v1_ids = [e["id"] for e in v1_pool]
    v2_ids = [e["id"] for e in v2_pool]
    removed = [i for i in raw_ids if i not in v2_ids]
    kept = [i for i in raw_ids if i in v2_ids]

    by_id = {}
    for e in raw_pool:
        by_id[e["id"]] = {"raw": e, "v1": None, "v2": None}
    for e in v1_pool:
        by_id.setdefault(e["id"], {})["v1"] = e
    for e in v2_pool:
        by_id.setdefault(e["id"], {})["v2"] = e

    # 每个 expert 最新的 reward 记录（round 最大）
    hist_by_id = {}
    for r in hist:
        eid = r["expert_id"]
        rnd = int(r.get("round", 0))
        if eid not in hist_by_id or rnd >= int(hist_by_id[eid].get("round", 0)):
            hist_by_id[eid] = r

    print(f"\n--- [{name}] Raw pool: {raw_ids}")
    print(f"--- [{name}] v1 final pool: {v1_ids}")
    print(f"--- [{name}] v2 final pool: {v2_ids}")
    print(f"--- [{name}] Removed by v2: {removed}")
    print(f"--- [{name}] Kept by v2: {kept}")

    rows = []
    for eid in raw_ids + [i for i in v2_ids if i not in raw_ids]:
        rec = by_id.get(eid, {})
        raw = rec.get("raw")
        v2e = rec.get("v2")
        h = hist_by_id.get(eid, {})
        # mechanism/activation/retrieval_action 优先取 v2 final 版本（replaced 后的），否则 raw
        expert = v2e or raw or {}
        rows.append({
            "dataset": name,
            "expert_id": eid,
            "in_raw": eid in raw_ids,
            "in_v1": eid in v1_ids,
            "in_v2": eid in v2_ids,
            "final_verdict": (h.get("verdict") if h else ("KEEP" if eid in v2_ids else "")),
            "round": h.get("round", ""),
            "replaced": bool(expert.get("replaced", False)),
            "mean_pool_reward": h.get("mean_pool_reward", ""),
            "median_pool_reward": h.get("median_pool_reward", ""),
            "positive_fold_ratio": h.get("positive_fold_ratio", ""),
            "worst_pool_reward": h.get("worst_pool_reward", ""),
            "coverage": h.get("coverage", ""),
            "mechanism": _short(expert.get("mechanism", "")),
            "activation_condition": _short(expert.get("activation_condition", "")),
            "retrieval_action": _short(expert.get("retrieval_action", "")),
            "scope": expert.get("scope", ""),
        })
    return rows, removed, kept


# --------------------------------------------------------------------------- #
# 冻结 eval（只 raw + v2 arm，逐窗口捕获预测）
# --------------------------------------------------------------------------- #
# module globals，供 run_arm 使用（与 final test 同构）
G = {}


def _init_state(seed, context, expert_dim, n_slots, device):
    v2p.seed_everything(seed)
    m = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def run_arm(init_state, emb, act_dict, use_shape, expert_dim, n_slots, device, seed, context):
    """与 final test run_test_arm 同构；返回逐窗口预测 pred [n_test, horizon]。"""
    v2p.seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v2p.train_arm(mdl, G["X_bank"], G["Y_bank"], G["X_fit"], G["Y_fit"],
                      G["X_bank_res"], G["X_fit_res"],
                      act_dict["bank"], act_dict["fit"], use_shape, device, seed)
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(mdl, G["X_test"], G["X_bank"], G["Y_bank"],
                                G["X_test_res"], G["X_bank_res"],
                                act_dict["test"], act_dict["bank"], mode, TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


def _per_window_err(pred):
    """逐窗口平方误差（mean over horizon）。"""
    return np.mean((G["Y_test"] - pred) ** 2, axis=1)


def _verify_aggregate(name, arm, seed, pw_err, frozen):
    """回验：逐窗口误差的均值应 == 冻结 result.json 的聚合 TEST MSE。"""
    agg = float(pw_err.mean())
    if frozen is None:
        return agg, None
    ref = None
    try:
        ref = float(frozen["summary"][arm]["seeds_mse"][str(seed)])
    except Exception:
        ref = None
    ok = ref is not None and abs(agg - ref) <= 1e-3 * max(1.0, abs(ref))
    return agg, (ok, agg, ref)


def build_dataset(name, device):
    """加载数据 + 冻结 pool + 逐窗口预测/误差 + 激活。返回诊断用结构。"""
    ds_dir = REPO_ROOT / "results/multidataset" / name
    cfg_json = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg_json["context"])
    horizon = int(cfg_json["horizon"])
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

    G["X_bank"], G["Y_bank"] = build_windows(series, bank_starts, context, horizon)
    G["X_fit"], G["Y_fit"] = build_windows(series, fit_starts, context, horizon)
    G["X_test"], G["Y_test"] = build_windows(series, test_starts, context, horizon)
    G["X_bank_res"] = residualize(G["X_bank"])
    G["X_fit_res"] = residualize(G["X_fit"])
    G["X_test_res"] = residualize(G["X_test"])
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    test_ts = [dates[s + context] for s in test_starts]
    X_train_all = np.concatenate([G["X_bank"], G["X_fit"]], axis=0)
    stats = compute_training_stats(X_train_all)
    print(f"  data: bank={len(bank_starts)} fit={len(fit_starts)} test={len(test_starts)} "
          f"context={context} horizon={horizon}")

    raw_pool = load_raw_pool(name)
    v1_pool = load_v1_pool(name)
    v2_pool = load_v2_pool(name)
    print(f"  pools: raw={len(raw_pool)} v1={len(v1_pool)} v2={len(v2_pool)}")

    emb_raw = v1.embed_claims([e["claim"] for e in raw_pool], device)
    emb_v2 = v1.embed_claims([e["claim"] for e in v2_pool], device)
    expert_dim = emb_raw.shape[1]
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))

    def _act(pool):
        a = v1.act_mats(pool, G["X_bank"], G["X_fit"], G["X_test"],
                        bank_ts, fit_ts, test_ts, stats)
        return {"bank": a["bank"], "fit": a["fit"], "test": a["val"]}

    raw_act_full = _act(raw_pool)      # 未 pad，[n, len(raw_pool)] 供 coverage 分析
    v2_act_full = _act(v2_pool)        # 未 pad，[n, len(v2_pool)]

    # pad 到 n_slots，与 final test 逐位一致
    emb_raw_p, raw_act, _ = v1.pad_to_slots(emb_raw, raw_act_full, n_slots)
    emb_v2_p, v2_act, _ = v1.pad_to_slots(emb_v2, v2_act_full, n_slots)

    frozen = load_frozen_test(name)

    out = {"name": name, "context": context, "horizon": horizon,
           "test_ts": test_ts, "stats": stats,
           "raw_pool": raw_pool, "v1_pool": v1_pool, "v2_pool": v2_pool,
           "emb_raw": emb_raw, "emb_v2": emb_v2, "expert_dim": expert_dim, "n_slots": n_slots,
           "raw_act_full": raw_act_full, "v2_act_full": v2_act_full,
           "frozen": frozen,
           "per_seed": {}}

    for arm, emb, act in (("raw", emb_raw_p, raw_act), ("v2", emb_v2_p, v2_act)):
        for seed in SEEDS:
            init = _init_state(seed, context, expert_dim, n_slots, device)
            pred = run_arm(init, emb, act, True, expert_dim, n_slots, device, seed, context)
            pw_err = _per_window_err(pred)
            agg, check = _verify_aggregate(name, arm, seed, pw_err, frozen)
            out["per_seed"][(arm, seed)] = {"pred": pred, "err": pw_err, "agg": agg}
            if check is not None:
                ok, a, r = check
                tag = "OK" if ok else "MISMATCH"
                print(f"    {arm} seed={seed} agg={a:.6e} frozen={r:.6e} [{tag}]")
            else:
                print(f"    {arm} seed={seed} agg={agg:.6e} frozen=(n/a)")
    return out


# --------------------------------------------------------------------------- #
# Section 3：traffic 是否真失败
# --------------------------------------------------------------------------- #
def section3_traffic(d):
    rows = []
    for seed in SEEDS:
        raw_err = d["per_seed"][("raw", seed)]["err"]
        v2_err = d["per_seed"][("v2", seed)]["err"]
        delta = raw_err - v2_err                      # >0 => v2 更好
        mean_d = float(delta.mean())
        med_d = float(np.median(delta))
        better_ratio = float((delta > 0).mean())

        rng = np.random.default_rng(BOOT_SEED + seed)
        n = len(delta)
        boot = delta[rng.integers(0, n, (BOOT_N, n))].mean(axis=1)
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        p_val = None
        if scipy_stats is not None:
            try:
                _, p_val = scipy_stats.wilcoxon(raw_err, v2_err)
                p_val = float(p_val)
            except Exception:
                p_val = None

        raw_mean = float(raw_err.mean())
        v2_mean = float(v2_err.mean())
        rel_diff = (v2_mean - raw_mean) / raw_mean * 100.0

        # top 10% worst windows（以 raw_err 降序取）
        k = max(1, int(len(raw_err) * 0.10))
        worst = np.argsort(raw_err)[::-1][:k]
        worst_raw = float(raw_err[worst].mean())
        worst_v2 = float(v2_err[worst].mean())

        rows.append({
            "seed": seed, "raw_mean_mse": raw_mean, "v2_mean_mse": v2_mean,
            "relative_diff_pct": rel_diff, "mean_delta": mean_d, "median_delta": med_d,
            "v2_better_ratio": better_ratio, "bootstrap_ci_lo": ci_lo, "bootstrap_ci_hi": ci_hi,
            "wilcoxon_p": ("" if p_val is None else p_val),
            "top10_worst_raw_mse": worst_raw, "top10_worst_v2_mse": worst_v2,
        })

    seed_signs = [1 if r["mean_delta"] > 0 else (-1 if r["mean_delta"] < 0 else 0) for r in rows]
    direction_consistent = len(set(seed_signs)) == 1

    # pooled（3 seed 合并）：spec 判定条件作用在整体，而非逐 seed 硬套
    pooled_raw = np.concatenate([d["per_seed"][("raw", s)]["err"] for s in SEEDS])
    pooled_v2 = np.concatenate([d["per_seed"][("v2", s)]["err"] for s in SEEDS])
    pooled_delta = pooled_raw - pooled_v2
    pooled_rel_diff = (pooled_v2.mean() - pooled_raw.mean()) / pooled_raw.mean() * 100.0
    rng = np.random.default_rng(BOOT_SEED)
    n_p = len(pooled_delta)
    boot_p = pooled_delta[rng.integers(0, n_p, (BOOT_N, n_p))].mean(axis=1)
    pooled_ci = np.percentile(boot_p, [2.5, 97.5])
    cond_lt1 = abs(pooled_rel_diff) < 1.0
    cond_ci = float(pooled_ci[0]) <= 0 <= float(pooled_ci[1])
    cond_inconsist = not direction_consistent
    comparable = cond_lt1 and cond_ci and cond_inconsist

    print("\n" + "=" * 78)
    print(f"SECTION 3  traffic 真失败判定")
    print("=" * 78)
    for r in rows:
        print(f"  seed={r['seed']}: rel_diff={r['relative_diff_pct']:+.3f}%  "
              f"mean_delta={r['mean_delta']:+.6e}  median_delta={r['median_delta']:+.6e}  "
              f"v2_better={r['v2_better_ratio']:.3f}  CI=[{r['bootstrap_ci_lo']:+.6e},{r['bootstrap_ci_hi']:+.6e}]  "
              f"wilcoxon_p={r['wilcoxon_p']}")
    print(f"  pooled: rel_diff={pooled_rel_diff:+.3f}%  mean_delta={pooled_delta.mean():+.6e}  "
          f"CI=[{pooled_ci[0]:+.6e},{pooled_ci[1]:+.6e}]  |rel_diff|<1%={cond_lt1}  "
          f"CI跨0={cond_ci}  seed方向不一致={cond_inconsist}")
    if comparable:
        print("  ==> TRAFFIC_COMPARABLE（相对差 <1% 且 CI 跨 0 且 seed 方向不一致）")
    else:
        print("  ==> 不满足 TRAFFIC_COMPARABLE 全部条件（逐条见上，描述性）")
    return rows, comparable


# --------------------------------------------------------------------------- #
# Section 4：electricity 误差切片
# --------------------------------------------------------------------------- #
def _quantile_bin(vals, nbin=4):
    q = np.quantile(vals, np.linspace(0, 1, nbin + 1))
    q[0], q[-1] = -np.inf, np.inf
    return np.digitize(vals, q[1:-1])  # 0..nbin-1


def section4_electricity(d):
    name = d["name"]
    test_ts = d["test_ts"]
    stats = d["stats"]
    feats = compute_window_feats(G["X_test"], test_ts, stats)
    target_level = G["Y_test"].mean(axis=1)

    raw_mean = np.mean([d["per_seed"][("raw", s)]["err"] for s in SEEDS], axis=0)
    v2_mean = np.mean([d["per_seed"][("v2", s)]["err"] for s in SEEDS], axis=0)
    delta = raw_mean - v2_mean

    n = len(delta)
    better_ratio = float((delta > 0).mean())
    med = float(np.median(delta))
    pct = {p: float(np.percentile(delta, p)) for p in (10, 25, 50, 75, 90)}
    k = max(1, int(n * 0.10))
    worst = np.argsort(raw_mean)[::-1][:k]
    top10_raw = float(raw_mean[worst].mean())
    top10_v2 = float(v2_mean[worst].mean())
    top20_worse = np.argsort(delta)[:20]       # v2 最差（delta 最小）
    top20_better = np.argsort(delta)[::-1][:20]  # v2 最好（delta 最大）

    print("\n" + "=" * 78)
    print(f"SECTION 4  {name} 误差切片（seed-mean）")
    print("=" * 78)
    print(f"  v2 better window ratio = {better_ratio:.3f}")
    print(f"  median delta = {med:+.6e}")
    print(f"  delta percentiles 10/25/50/75/90 = "
          f"{pct[10]:+.6e}/{pct[25]:+.6e}/{pct[50]:+.6e}/{pct[75]:+.6e}/{pct[90]:+.6e}")
    print(f"  top-10%-worst windows: raw={top10_raw:.4e} v2={top10_v2:.4e}")

    # per-window CSV
    win_rows = []
    for i in range(n):
        win_rows.append({
            "window_idx": i,
            "timestamp": str(test_ts[i]),
            "raw_err_mean": float(raw_mean[i]),
            "v2_err_mean": float(v2_mean[i]),
            "delta": float(delta[i]),
            "hour_of_day": int(feats["hour_of_day"][i]),
            "day_of_week": int(feats["day_of_week"][i]),
            "working_day": int(feats["working_day"][i]),
            "target_level": float(target_level[i]),
            "recent_level": float(feats["recent_level"][i]),
            "recent_volatility": float(feats["recent_volatility"][i]),
            "recent_slope_magnitude": float(feats["recent_slope_magnitude"][i]),
            "recent_diff_magnitude": float(feats["recent_diff_magnitude"][i]),
        })

    # 通用 regime slicing
    regimes = {
        "target_level": target_level,
        "recent_volatility": feats["recent_volatility"],
        "recent_slope_magnitude": feats["recent_slope_magnitude"],
        "recent_diff_magnitude": feats["recent_diff_magnitude"],
    }
    regime_rows = []
    for axis, vals in regimes.items():
        b = _quantile_bin(vals)
        for qb in range(4):
            m = b == qb
            nq = int(m.sum())
            if nq == 0:
                continue
            rm = float(raw_mean[m].mean())
            vm = float(v2_mean[m].mean())
            regime_rows.append({
                "axis": axis, "quantile": f"Q{qb + 1}", "n_windows": nq,
                "raw_mse": rm, "v2_mse": vm,
                "relative_improvement_pct": (rm - vm) / rm * 100.0 if rm > 0 else 0.0,
            })
        print(f"  [{axis}] " + "  ".join(
            f"{r['quantile']}: n={r['n_windows']} raw={r['raw_mse']:.4e} v2={r['v2_mse']:.4e} "
            f"RI={r['relative_improvement_pct']:+.1f}%"
            for r in regime_rows if r["axis"] == axis))

    # 日历切片
    cal_rows = []
    for axis, key in (("hour_of_day", "hour_of_day"), ("day_of_week", "day_of_week")):
        uniq = np.unique(feats[key])
        for u in uniq:
            m = feats[key] == u
            rm = float(raw_mean[m].mean())
            vm = float(v2_mean[m].mean())
            cal_rows.append({
                "axis": axis, "bucket": int(u), "n_windows": int(m.sum()),
                "raw_mse": rm, "v2_mse": vm,
                "relative_improvement_pct": (rm - vm) / rm * 100.0 if rm > 0 else 0.0,
            })
    for axis, m in (("weekday", feats["working_day"] < 5), ("weekend", feats["working_day"] >= 5)):
        rm = float(raw_mean[m].mean())
        vm = float(v2_mean[m].mean())
        cal_rows.append({
            "axis": axis, "bucket": 1, "n_windows": int(m.sum()),
            "raw_mse": rm, "v2_mse": vm,
            "relative_improvement_pct": (rm - vm) / rm * 100.0 if rm > 0 else 0.0,
        })

    summary = {
        "better_ratio": better_ratio, "median_delta": med,
        "pct10": pct[10], "pct25": pct[25], "pct50": pct[50], "pct75": pct[75], "pct90": pct[90],
        "top10_worst_raw_mse": top10_raw, "top10_worst_v2_mse": top10_v2,
        "top20_worse_idx": [int(i) for i in top20_worse],
        "top20_better_idx": [int(i) for i in top20_better],
    }
    print(f"  top-20 v2-worse window idx: {summary['top20_worse_idx']}")
    print(f"  top-20 v2-better window idx: {summary['top20_better_idx']}")
    return summary, win_rows, regime_rows, cal_rows


# --------------------------------------------------------------------------- #
# Section 5：activation / coverage 信息损失
# --------------------------------------------------------------------------- #
def _coverage_stats(act, name, pool):
    # act: [n_windows, n_experts]（未 pad）
    n, ne = act.shape
    active = act > ACT_THRESH
    union = active.any(axis=1)
    per_expert_cov = active.mean(axis=0)
    per_expert_mean = act.mean(axis=0)
    avg_active_per_win = active.sum(axis=1)
    rows = []
    for j in range(ne):
        rows.append({
            "pool": name, "expert_id": pool[j]["id"],
            "activation_coverage_gt_thresh": float(per_expert_cov[j]),
            "mean_activation": float(per_expert_mean[j]),
        })
    stats_out = {
        "n_windows": int(n), "n_experts": int(ne),
        "union_coverage": float(union.mean()),
        "avg_active_experts_per_window": float(avg_active_per_win.mean()),
        "activation_count_dist": {str(c): int((avg_active_per_win == c).sum())
                                  for c in np.unique(avg_active_per_win)},
    }
    return stats_out, rows, active


def section5_activation(d):
    raw_act = d["raw_act_full"]["test"]
    v2_act = d["v2_act_full"]["test"]
    raw_stats, raw_rows, raw_active = _coverage_stats(raw_act, "raw", d["raw_pool"])
    v2_stats, v2_rows, v2_active = _coverage_stats(v2_act, "v2", d["v2_pool"])

    print("\n" + "=" * 78)
    print(f"SECTION 5  {d['name']} activation / coverage")
    print("=" * 78)
    print(f"  raw: union_coverage={raw_stats['union_coverage']:.3f} "
          f"avg_active/window={raw_stats['avg_active_experts_per_window']:.2f}")
    print(f"  v2 : union_coverage={v2_stats['union_coverage']:.3f} "
          f"avg_active/window={v2_stats['avg_active_experts_per_window']:.2f}")

    # co-activation（Pearson）
    def coact(act):
        a = act - act.mean(axis=0, keepdims=True)
        denom = np.sqrt((a ** 2).sum(axis=0, keepdims=True)) + 1e-12
        return (a.T @ a) / (denom.T @ denom) / len(act)

    raw_coact = coact(raw_act)
    v2_coact = coact(v2_act)

    # embedding cosine diversity
    def emb_diversity(emb):
        e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sim = e @ e.T
        iu = np.triu_indices(len(e), 1)
        return float(sim[iu].mean())

    raw_div = emb_diversity(d["emb_raw"])
    v2_div = emb_diversity(d["emb_v2"])
    print(f"  embedding mean pairwise cosine: raw={raw_div:.3f} v2={v2_div:.3f}")

    return {
        "raw": raw_stats, "v2": v2_stats,
        "raw_embedding_mean_cosine": raw_div, "v2_embedding_mean_cosine": v2_div,
    }, raw_rows + v2_rows, raw_coact, v2_coact


# --------------------------------------------------------------------------- #
# 汇总 + 输出
# --------------------------------------------------------------------------- #
def write_report(rows_pool, traffic_rows, comparable, elec_summary, elec_regime, elec_cal,
                 act_stats):
    L = []
    L.append("# Post-hoc Failure Diagnosis：frozen v2 on electricity + traffic")
    L.append("")
    L.append("只诊断，不修改 v2 / reward / prompt / pool / 参数。TEST 只做描述性分析。")
    L.append("")
    L.append("## 0. 禁令")
    L.append("")
    L.append("- 未修改 v2；未调 reward/Critic/Expert prompt；未按 TEST 重选 pool/调参/subset search；未做 v3；未覆盖 frozen TEST 主表。")
    L.append("")
    L.append("## 2. Pool 变化（raw → v2）")
    L.append("")
    for name in ("electricity", "traffic"):
        rp = [r for r in rows_pool if r["dataset"] == name]
        if not rp:
            continue
        L.append(f"### {name}")
        L.append("")
        L.append("| expert | raw | v1 | v2 | verdict | round | replaced | mean_rew | median_rew | pos_fold | worst_rew | coverage | mechanism |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rp:
            L.append(f"| {r['expert_id']} | {'Y' if r['in_raw'] else ''} | {'Y' if r['in_v1'] else ''} | "
                     f"{'Y' if r['in_v2'] else ''} | {r['final_verdict']} | {r['round']} | {r['replaced']} | "
                     f"{r['mean_pool_reward']} | {r['median_pool_reward']} | {r['positive_fold_ratio']} | "
                     f"{r['worst_pool_reward']} | {r['coverage']} | {_short(r['mechanism'], 60)} |")
        L.append("")
        L.append("> 注：per-fold pool_reward 原始值未落盘（`compute_pool_reward` 仅打印到 stdout），"
                 "上表为 4-fold 的 aggregate（mean/median/positive_fold_ratio/worst）。")
        L.append("")
    L.append("## 3. Traffic：是否真失败")
    L.append("")
    for r in traffic_rows:
        L.append(f"- seed={r['seed']}: rel_diff={r['relative_diff_pct']:+.3f}%  mean_delta={r['mean_delta']:+.6e}  "
                 f"median_delta={r['median_delta']:+.6e}  v2_better={r['v2_better_ratio']:.3f}  "
                 f"CI=[{r['bootstrap_ci_lo']:+.6e},{r['bootstrap_ci_hi']:+.6e}]  wilcoxon_p={r['wilcoxon_p']}")
    L.append("")
    L.append(f"判定：{'**TRAFFIC_COMPARABLE**' if comparable else '不满足 TRAFFIC_COMPARABLE 全部条件（见上，描述性）'}")
    L.append("")
    L.append("## 4. Electricity 误差切片")
    L.append("")
    L.append(f"- v2 better window ratio = {elec_summary['better_ratio']:.3f}")
    L.append(f"- median delta = {elec_summary['median_delta']:+.6e}")
    L.append(f"- delta p10/p25/p50/p75/p90 = {elec_summary['pct10']:+.6e}/{elec_summary['pct25']:+.6e}/"
             f"{elec_summary['pct50']:+.6e}/{elec_summary['pct75']:+.6e}/{elec_summary['pct90']:+.6e}")
    L.append(f"- top-10%-worst: raw={elec_summary['top10_worst_raw_mse']:.4e} v2={elec_summary['top10_worst_v2_mse']:.4e}")
    L.append("")
    L.append("### regime slicing（seed-mean）")
    L.append("")
    L.append("| axis | quantile | n | raw_mse | v2_mse | RI% |")
    L.append("|---|---|---|---|---|---|")
    for r in elec_regime:
        L.append(f"| {r['axis']} | {r['quantile']} | {r['n_windows']} | {r['raw_mse']:.4e} | "
                 f"{r['v2_mse']:.4e} | {r['relative_improvement_pct']:+.1f}% |")
    L.append("")
    L.append("### calendar slicing")
    L.append("")
    L.append("| axis | bucket | n | raw_mse | v2_mse | RI% |")
    L.append("|---|---|---|---|---|---|")
    for r in elec_cal:
        L.append(f"| {r['axis']} | {r['bucket']} | {r['n_windows']} | {r['raw_mse']:.4e} | "
                 f"{r['v2_mse']:.4e} | {r['relative_improvement_pct']:+.1f}% |")
    L.append("")
    L.append("## 5. Activation / coverage")
    L.append("")
    for nm in ("raw", "v2"):
        s = act_stats[nm]
        L.append(f"- {nm}: union_coverage={s['union_coverage']:.3f}  "
                 f"avg_active/window={s['avg_active_experts_per_window']:.2f}  "
                 f"n_experts={s['n_experts']}")
    L.append(f"- embedding mean pairwise cosine: raw={act_stats['raw_embedding_mean_cosine']:.3f} "
             f"v2={act_stats['v2_embedding_mean_cosine']:.3f}")
    L.append("")
    L.append("## 输出文件")
    L.append("")
    L.append("- `pool_comparison.csv` / `traffic_per_seed_stats.csv` / `traffic_window_delta.csv`")
    L.append("- `electricity_window_errors.csv` / `electricity_regime_slicing.csv` / `electricity_calendar_slicing.csv`")
    L.append("- `electricity_pool_activation_stats.csv`（section 5 明确要求）")
    return "\n".join(L)


def _write_csv(path, rows, cols):
    with (OUT_ROOT / path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    _banner()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset (default: electricity traffic)")
    ap.add_argument("--only-pool", action="store_true",
                    help="只做 section2 pool 对比（artifact 只读，不跑模型）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    targets = [d for d in DATASETS if (args.datasets is None or d["name"] in args.datasets)]

    all_pool_rows = []
    ran = set()
    for cfg in targets:
        name = cfg["name"]
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        raw_pool = load_raw_pool(name)
        v1_pool = load_v1_pool(name)
        v2_pool = load_v2_pool(name)
        hist = load_reward_history(name)
        rows, removed, kept = section2_pool_comparison(name, raw_pool, v1_pool, v2_pool, hist)
        all_pool_rows.extend(rows)
        if args.only_pool:
            continue

        d = build_dataset(name, device)
        ran.add(name)

        if name == "traffic":
            traffic_rows, comparable = section3_traffic(d)
            win_rows = []
            for seed in SEEDS:
                re = d["per_seed"][("raw", seed)]["err"]
                ve = d["per_seed"][("v2", seed)]["err"]
                for i in range(len(re)):
                    win_rows.append({"seed": seed, "window_idx": i,
                                     "raw_err": float(re[i]), "v2_err": float(ve[i]),
                                     "delta": float(re[i] - ve[i])})
            _write_csv("traffic_per_seed_stats.csv", traffic_rows,
                       ["seed", "raw_mean_mse", "v2_mean_mse", "relative_diff_pct",
                        "mean_delta", "median_delta", "v2_better_ratio",
                        "bootstrap_ci_lo", "bootstrap_ci_hi", "wilcoxon_p",
                        "top10_worst_raw_mse", "top10_worst_v2_mse"])
            _write_csv("traffic_window_delta.csv", win_rows,
                       ["seed", "window_idx", "raw_err", "v2_err", "delta"])

        elif name == "electricity":
            elec_summary, win_rows, regime_rows, cal_rows = section4_electricity(d)
            act_stats, act_rows, raw_coact, v2_coact = section5_activation(d)
            _write_csv("electricity_window_errors.csv", win_rows,
                       ["window_idx", "timestamp", "raw_err_mean", "v2_err_mean", "delta",
                        "hour_of_day", "day_of_week", "working_day", "target_level",
                        "recent_level", "recent_volatility", "recent_slope_magnitude",
                        "recent_diff_magnitude"])
            _write_csv("electricity_regime_slicing.csv", regime_rows,
                       ["axis", "quantile", "n_windows", "raw_mse", "v2_mse",
                        "relative_improvement_pct"])
            _write_csv("electricity_calendar_slicing.csv", cal_rows,
                       ["axis", "bucket", "n_windows", "raw_mse", "v2_mse",
                        "relative_improvement_pct"])
            _write_csv("electricity_pool_activation_stats.csv", act_rows,
                       ["pool", "expert_id", "activation_coverage_gt_thresh", "mean_activation"])
            # co-activation 矩阵落盘
            np.savez_compressed(OUT_ROOT / "electricity_coactivation.npz",
                                raw_coact=raw_coact, v2_coact=v2_coact,
                                raw_expert_ids=[e["id"] for e in d["raw_pool"]],
                                v2_expert_ids=[e["id"] for e in d["v2_pool"]])
            # 逐 seed 逐窗口误差落盘（供复现 / 二次分析）
            np.savez_compressed(OUT_ROOT / "electricity_per_seed_errors.npz",
                                **{f"{arm}_s{seed}": d["per_seed"][(arm, seed)]["err"]
                                   for arm in ("raw", "v2") for seed in SEEDS})

    _write_csv("pool_comparison.csv", all_pool_rows,
               ["dataset", "expert_id", "in_raw", "in_v1", "in_v2", "final_verdict",
                "round", "replaced", "mean_pool_reward", "median_pool_reward",
                "positive_fold_ratio", "worst_pool_reward", "coverage",
                "mechanism", "activation_condition", "retrieval_action", "scope"])

    if not args.only_pool and "electricity" in ran and "traffic" in ran:
        report = write_report(all_pool_rows, traffic_rows, comparable,
                              elec_summary, regime_rows, cal_rows, act_stats)
        (OUT_ROOT / "report.md").write_text(report, encoding="utf-8")
    elif not args.only_pool:
        print("[report] 只跑了部分数据集，跳过完整 report.md（各 CSV 已写）")
    print(f"\nSaved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
