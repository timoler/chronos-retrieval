#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen v2 Final TEST Evaluation。

在完全冻结 v2（Pool-Aware Reward）的前提下，对 7 个数据集做最终 TEST 泛化验证。

原则：
  - TEST 只用于最终评估，禁止根据 TEST 改 v2 / 挑 seed / 增删 Expert / 做 v2.1·v3。
  - 不做 Agent 闭环：不调用 LLM Expert、不调用 Critic、不 revision、不重新筛 pool。
  - 直接读取 VAL 阶段已冻结的 Raw / v1 / v2 pool，把评估 split 从 VAL 换成 TEST。
  - 训练协议与 VAL 完全一致（复用 v2p.train_arm / retrieve_conditional），只换 eval split。

冻结来源（只读，不覆盖）：
  - v2 final pool : results/reward_feedback_critic_v2_full_cross_dataset/<name>/final_expert_pool.json
  - raw pool      : 同上 raw_expert_pool 优先，回退 v1 原始 raw_experts.json（同处理）
  - v1 final pool : etth1 -> results/reward_feedback_critic_v1_etth1/，其余 -> ..._v1_cross_dataset/<name>/

输出：results/reward_feedback_critic_v2_final_test/
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
from run_etth1_prediction_validation import compute_training_stats

import run_etth1_reward_feedback_critic_v1 as v1
import run_reward_feedback_critic_v2_poolaware_pilot as v2p

REPO_ROOT = HERE.parent
FROZEN_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_full_cross_dataset"   # 冻结 pool 来源
V1_ETTH1 = v2p.V1_ETTH1
V1_CROSS = v2p.V1_CROSS
VAL_SUMMARY = FROZEN_ROOT / "full_cross_dataset_summary.csv"                         # VAL 结果（仅报告对照）
TEST_OUT = REPO_ROOT / "results/reward_feedback_critic_v2_final_test"
TEST_OUT.mkdir(parents=True, exist_ok=True)

SEEDS = v2p.SEEDS
TOPK = v2p.TOPK
N_RAW_TARGET = v2p.N_RAW_TARGET
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

DATASETS = [
    {"name": "etth1",         "domain": "transformer_temperature",  "freq": "hourly"},
    {"name": "etth2",         "domain": "transformer_temperature",  "freq": "hourly"},
    {"name": "weather",       "domain": "weather_meteorological",    "freq": "10-minute"},
    {"name": "exchange_rate", "domain": "daily_exchange_rate",       "freq": "daily"},
    {"name": "solar",         "domain": "aggregate_solar_generation", "freq": "10-minute"},
    {"name": "electricity",   "domain": "aggregate_electricity_load", "freq": "15-minute"},
    {"name": "traffic",       "domain": "road_traffic_occupancy",    "freq": "hourly"},
]


class ArtifactMissingError(RuntimeError):
    pass


class DeterminismFailure(RuntimeError):
    pass


def _banner():
    print("[FROZEN FINAL TEST]")
    print("method = v2_pool_aware")
    print("datasets = 7")
    print("seeds = [42,123,2026]")
    print("expert_critic_calls = DISABLED")
    print("pool_revision = DISABLED")
    print("test_tuning = DISABLED")
    print("eval_split = TEST (train bank/fit unchanged)")


# --------------------------------------------------------------------------- #
# 冻结 pool 读取（只读）
# --------------------------------------------------------------------------- #
def load_raw_pool(name):
    src = FROZEN_ROOT / name / "raw_experts.json"
    if src.exists():
        return json.loads(src.read_text(encoding="utf-8"))
    # 回退：v1 原始 raw，做与 VAL 相同的处理
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
        raise ArtifactMissingError(
            f"missing frozen v2 final pool: {src}（VAL 阶段未落盘，请先补齐）")
    return json.loads(src.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# TEST 评估（复用冻结训练，只换 eval split）
# --------------------------------------------------------------------------- #
# module globals，供 run_test_arm 使用（与 v2p.run_val_arm 同构）
g_X_bank = g_Y_bank = g_X_fit = g_Y_fit = None
g_X_test = g_Y_test = g_X_bank_res = g_X_fit_res = g_X_test_res = None


def run_test_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed, context):
    v2p.seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v2p.train_arm(mdl, g_X_bank, g_Y_bank, g_X_fit, g_Y_fit,
                      g_X_bank_res, g_X_fit_res,
                      act["bank"], act["fit"], use_shape, device, seed)
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(mdl, g_X_test, g_X_bank, g_Y_bank,
                                g_X_test_res, g_X_bank_res,
                                act["test"], act["bank"], mode, TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(mse(g_Y_test, pred)), float(mae(g_Y_test, pred))


def run_test(name, raw_pool, v1_pool, v2_pool, device):
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

    global g_X_bank, g_Y_bank, g_X_fit, g_Y_fit, g_X_test, g_Y_test
    global g_X_bank_res, g_X_fit_res, g_X_test_res
    g_X_bank, g_Y_bank = build_windows(series, bank_starts, context, horizon)
    g_X_fit, g_Y_fit = build_windows(series, fit_starts, context, horizon)
    g_X_test, g_Y_test = build_windows(series, test_starts, context, horizon)
    g_X_bank_res = residualize(g_X_bank)
    g_X_fit_res = residualize(g_X_fit)
    g_X_test_res = residualize(g_X_test)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    test_ts = [dates[s + context] for s in test_starts]
    X_train_all = np.concatenate([g_X_bank, g_X_fit], axis=0)
    stats = compute_training_stats(X_train_all)
    print(f"  data: bank={len(bank_starts)} fit={len(fit_starts)} test={len(test_starts)} (TEST used)")

    emb_raw = v1.embed_claims([e["claim"] for e in raw_pool], device)
    emb_v1 = v1.embed_claims([e["claim"] for e in v1_pool], device) if v1_pool else \
        np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    emb_v2 = v1.embed_claims([e["claim"] for e in v2_pool], device) if v2_pool else \
        np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    expert_dim = emb_raw.shape[1]
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))

    def _empty_act():
        return {"bank": np.zeros((len(g_X_bank), 0), dtype=np.float32),
                "fit": np.zeros((len(g_X_fit), 0), dtype=np.float32),
                "test": np.zeros((len(g_X_test), 0), dtype=np.float32)}

    def _act_test(pool):
        a = v1.act_mats(pool, g_X_bank, g_X_fit, g_X_test, bank_ts, fit_ts, test_ts, stats)
        return {"bank": a["bank"], "fit": a["fit"], "test": a["val"]}

    raw_act = _act_test(raw_pool)
    v1_act = _act_test(v1_pool) if v1_pool else _empty_act()
    v2_act = _act_test(v2_pool) if v2_pool else _empty_act()
    emb_raw, raw_act, _ = v1.pad_to_slots(emb_raw, raw_act, n_slots)
    emb_v1, v1_act, _ = v1.pad_to_slots(emb_v1, v1_act, n_slots)
    emb_v2, v2_act, _ = v1.pad_to_slots(emb_v2, v2_act, n_slots)
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_act = {k: np.zeros_like(raw_act[k]) for k in raw_act}

    configs = {
        "base": (emb_zero, zero_act, False),
        "shape": (emb_zero, zero_act, True),
        "raw": (emb_raw, raw_act, True),
        "v1": (emb_v1, v1_act, True),
        "v2": (emb_v2, v2_act, True),
    }

    def _init_state(seed):
        v2p.seed_everything(seed)
        m = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    # determinism 检查：v2 seed2026 x2
    emb_d, act_d, use_d = configs["v2"]
    init_d = _init_state(2026)
    det = [run_test_arm(init_d, emb_d, act_d, use_d, expert_dim, n_slots, device, 2026, context)[0]
           for _ in range(2)]
    det_ok = abs(det[0] - det[1]) <= 1e-4
    print(f"  determinism check: v2 seed2026 x2 = {det} -> {'PASS' if det_ok else 'FAIL'}")
    if not det_ok:
        raise DeterminismFailure(f"determinism check FAILED for {name}: {det}")

    per_seed = {nm: {} for nm in configs}
    per_seed_mae = {nm: {} for nm in configs}
    for seed in SEEDS:
        init_state = _init_state(seed)
        for nm, (emb, act, use_shape) in configs.items():
            mm, aa = run_test_arm(init_state, emb, act, use_shape,
                                  expert_dim, n_slots, device, seed, context)
            per_seed[nm][seed] = mm
            per_seed_mae[nm][seed] = aa
            print(f"    seed={seed} [{nm}] TEST_MSE={mm:.4f} TEST_MAE={aa:.4f}")

    summary = {}
    for nm in configs:
        summary[nm] = {
            "mean_mse": float(np.mean([per_seed[nm][s] for s in SEEDS])),
            "std_mse": float(np.std([per_seed[nm][s] for s in SEEDS])),
            "mean_mae": float(np.mean([per_seed_mae[nm][s] for s in SEEDS])),
            "seeds_mse": {s: per_seed[nm][s] for s in SEEDS},
            "seeds_mae": {s: per_seed_mae[nm][s] for s in SEEDS},
        }
    return summary, per_seed, per_seed_mae, det_ok


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def make_test_row(name, summary, det_ok):
    raw_m = summary["raw"]["mean_mse"]
    v1_m = summary["v1"]["mean_mse"]
    v2_m = summary["v2"]["mean_mse"]
    wins_raw = sum(1 for s in SEEDS if summary["v2"]["seeds_mse"][s] < summary["raw"]["seeds_mse"][s])
    wins_v1 = sum(1 for s in SEEDS if summary["v2"]["seeds_mse"][s] < summary["v1"]["seeds_mse"][s])
    return {
        "dataset": name,
        "base_mean_mse": summary["base"]["mean_mse"],
        "shape_mean_mse": summary["shape"]["mean_mse"],
        "raw_mean_mse": raw_m,
        "v1_mean_mse": v1_m,
        "v2_mean_mse": v2_m,
        "v2_std": summary["v2"]["std_mse"],
        "raw_mean_mae": summary["raw"]["mean_mae"],
        "v1_mean_mae": summary["v1"]["mean_mae"],
        "v2_mean_mae": summary["v2"]["mean_mae"],
        "v2_vs_raw_pct": (raw_m - v2_m) / raw_m * 100.0,
        "v2_vs_v1_pct": (v1_m - v2_m) / v1_m * 100.0,
        "wins_vs_raw_seeds": wins_raw,
        "wins_vs_v1_seeds": wins_v1,
        "raw_std": summary["raw"]["std_mse"],
        "determinism_pass": det_ok,
    }


SUMMARY_FIELDS = ["dataset", "base_mean_mse", "shape_mean_mse", "raw_mean_mse",
                  "v1_mean_mse", "v2_mean_mse", "v2_std",
                  "raw_mean_mae", "v1_mean_mae", "v2_mean_mae",
                  "v2_vs_raw_pct", "v2_vs_v1_pct",
                  "wins_vs_raw_seeds", "wins_vs_v1_seeds", "determinism_pass"]


def load_val_summary():
    rows = {}
    if not VAL_SUMMARY.exists():
        return rows
    for r in csv.DictReader(VAL_SUMMARY.open(encoding="utf-8")):
        rows[r["dataset"]] = float(r["v2_vs_raw_pct"])
    return rows


def compute_stats(rows):
    n = len(rows)
    ri_raw = [(r["raw_mean_mse"] - r["v2_mean_mse"]) / r["raw_mean_mse"] for r in rows]
    ri_v1 = [(r["v1_mean_mse"] - r["v2_mean_mse"]) / r["v1_mean_mse"] for r in rows]
    return {
        "n": n,
        "mean_RI_vs_raw": float(np.mean(ri_raw)) if ri_raw else 0.0,
        "median_RI_vs_raw": float(np.median(ri_raw)) if ri_raw else 0.0,
        "mean_RI_vs_v1": float(np.mean(ri_v1)) if ri_v1 else 0.0,
        "median_RI_vs_v1": float(np.median(ri_v1)) if ri_v1 else 0.0,
        "better_raw": sum(1 for r in rows if r["v2_mean_mse"] < r["raw_mean_mse"]),
        "better_v1": sum(1 for r in rows if r["v2_mean_mse"] < r["v1_mean_mse"]),
        "full_wins_raw": sum(1 for r in rows if r["wins_vs_raw_seeds"] == 3),
        "wins_23_raw": sum(1 for r in rows if r["wins_vs_raw_seeds"] >= 2),
        "std_lower": sum(1 for r in rows if r["v2_std"] < r["raw_std"]),
    }


def render_report(rows, stats, val_summary, missing):
    def pct(x):
        return f"{x * 100:+.2f}%"

    # VAL -> TEST 方向保持
    preserved = []
    for r in rows:
        val = val_summary.get(r["dataset"])
        if val is None:
            preserved.append(None)
            continue
        test = r["v2_vs_raw_pct"]
        preserved.append((val > 0) == (test > 0))
    n_preserved = sum(1 for p in preserved if p is True)
    n_preservable = sum(1 for p in preserved if p is not None)

    L = []
    L.append("# Frozen v2 Final TEST Evaluation（Pool-Aware Reward）")
    L.append("")
    L.append("完全冻结 v2，不调用 LLM / Critic / revision / 重新筛 pool，直接读 VAL 冻结 pool 换 TEST split。")
    L.append("TEST 只评估一次，禁止回头改 v2。")
    L.append("")

    L.append("## 结果")
    L.append("")
    L.append(f"- TEST 上 v2 比 Raw 好：{stats['better_raw']}/{stats['n']}")
    L.append(f"- TEST 上 v2 比 v1 好：{stats['better_v1']}/{stats['n']}")
    L.append(f"- mean RI vs Raw = {pct(stats['mean_RI_vs_raw'])}")
    L.append(f"- median RI vs Raw = {pct(stats['median_RI_vs_raw'])}")
    L.append(f"- mean RI vs v1 = {pct(stats['mean_RI_vs_v1'])}")
    L.append(f"- median RI vs v1 = {pct(stats['median_RI_vs_v1'])}")
    L.append(f"- 3/3 seeds 赢 Raw：{stats['full_wins_raw']}/{stats['n']}")
    L.append(f"- 2/3 或 3/3 seeds 赢 Raw：{stats['wins_23_raw']}/{stats['n']}")
    L.append(f"- v2 std < Raw std：{stats['std_lower']}/{stats['n']}")
    L.append(f"- VAL→TEST 方向保持：{n_preserved}/{n_preservable}")
    fail_ds = [r["dataset"] for r in rows if r["v2_mean_mse"] >= r["raw_mean_mse"]]
    if fail_ds:
        L.append(f"- TEST 上 v2 比 Raw 差的数据集：{fail_ds}")
    if missing:
        L.append(f"- artifact 缺失（未纳入 TEST）：{missing}")
    L.append("")

    L.append("主表：")
    L.append("")
    L.append("| Dataset | Raw | v1 | v2 | v2 vs Raw | v2 vs v1 | wins/raw |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['dataset']} | {r['raw_mean_mse']:.6g} | {r['v1_mean_mse']:.6g} "
                 f"| {r['v2_mean_mse']:.6g} | {r['v2_vs_raw_pct']:+.1f}% | {r['v2_vs_v1_pct']:+.1f}% "
                 f"| {r['wins_vs_raw_seeds']}/3 |")
    L.append("")

    L.append("VAL → TEST 对照：")
    L.append("")
    L.append("| Dataset | VAL v2 vs Raw | TEST v2 vs Raw | Direction preserved |")
    L.append("|---|---|---|---|")
    for r, p in zip(rows, preserved):
        val = val_summary.get(r["dataset"])
        val_s = f"{val:+.1f}%" if val is not None else "—"
        p_s = "yes" if p is True else ("no" if p is False else "—")
        L.append(f"| {r['dataset']} | {val_s} | {r['v2_vs_raw_pct']:+.1f}% | {p_s} |")
    L.append("")

    for r in rows:
        if r["v2_mean_mse"] >= r["raw_mean_mse"]:
            L.append(f"### FAIL_DATASET: {r['dataset']}")
            L.append(f"- TEST v2 vs Raw = {r['v2_vs_raw_pct']:+.1f}%")
            L.append("")
    for m in missing:
        L.append(f"### MISSING_ARTIFACT: {m}")
        L.append("- v2 final pool 未落盘，停止该数据集，未擅自重新生成 Expert。")
        L.append("")

    L.append("## 结论")
    L.append("")
    n = stats["n"]
    if n == 0:
        concl = "FAIL TO GENERALIZE"
    elif (stats["better_raw"] > n / 2 and stats["mean_RI_vs_raw"] > 0
          and stats["median_RI_vs_raw"] > 0):
        concl = "STRONG GENERALIZATION"
    elif stats["mean_RI_vs_raw"] > 0:
        concl = "PARTIAL GENERALIZATION"
    else:
        concl = "FAIL TO GENERALIZE"
    L.append(f"- **{concl}**")
    if concl == "STRONG GENERALIZATION":
        L.append("  （多数 TEST 数据集优于 Raw，mean/median RI vs Raw > 0。）")
    elif concl == "PARTIAL GENERALIZATION":
        L.append("  （整体 RI 仍为正，但胜率下降或部分强正例明显衰减。）")
    else:
        L.append("  （TEST 多数数据集不如 Raw，或 median RI <= 0，或 VAL 优势大面积反转。）")
    L.append("")
    L.append("```")
    L.append("test_used = true")
    L.append("test_tuning = false")
    L.append("expert_critic_calls = false")
    L.append("pool_revision = false")
    L.append("method_frozen = v2_pool_aware")
    L.append("```")
    return "\n".join(L)


def main():
    _banner()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of dataset names (default: all 7)")
    ap.add_argument("--resume", action="store_true",
                    help="skip datasets whose TEST result.json already exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    targets = [d for d in DATASETS if (args.datasets is None or d["name"] in args.datasets)]
    if args.resume:
        targets = [d for d in targets if not (TEST_OUT / d["name"] / "result.json").exists()]

    val_summary = load_val_summary()
    results = []
    missing = []
    failed = []
    for cfg in targets:
        name = cfg["name"]
        out_dir = TEST_OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 78}\nDATASET {name} (TEST)\n{'=' * 78}")
        try:
            raw_pool = load_raw_pool(name)
            v1_pool = load_v1_pool(name)
            v2_pool = load_v2_pool(name)
            print(f"  pools: raw={len(raw_pool)} v1={len(v1_pool)} v2={len(v2_pool)}")
            summary, per_seed, per_seed_mae, det_ok = run_test(
                name, raw_pool, v1_pool, v2_pool, device)
            row = make_test_row(name, summary, det_ok)
            results.append(row)

            (out_dir / "result.json").write_text(json.dumps(
                {"dataset": name, "det_ok": det_ok, "summary": summary,
                 "raw_pool_size": len(raw_pool), "v1_pool_size": len(v1_pool),
                 "v2_pool_size": len(v2_pool)},
                ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            with (out_dir / "test_ablation.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["arm", "seed", "test_mse", "test_mae"])
                for nm in ("base", "shape", "raw", "v1", "v2"):
                    for s in SEEDS:
                        w.writerow([nm, s, f"{per_seed[nm][s]:.6f}", f"{per_seed_mae[nm][s]:.6f}"])
        except ArtifactMissingError as e:
            print(f"  [MISSING ARTIFACT] {e}")
            missing.append(name)
        except DeterminismFailure as e:
            print(f"\n[FATAL] {e}")
            print("STOPPING the whole run: determinism check failed.")
            sys.exit(2)
        except Exception as e:
            print(f"\n[DATASET FAILED] {name}: {e}")
            traceback.print_exc()
            failed.append(name)

    # 断点续跑：本次没跑但之前已落 result.json 的也并入
    new_names = {r["dataset"] for r in results}
    for d in DATASETS:
        name = d["name"]
        if name in new_names or name in missing:
            continue
        rp = TEST_OUT / name / "result.json"
        if rp.exists():
            r = json.loads(rp.read_text(encoding="utf-8"))
            summary = {nm: {"mean_mse": s["mean_mse"], "std_mse": s["std_mse"],
                            "mean_mae": s["mean_mae"],
                            "seeds_mse": {int(k): v for k, v in s["seeds_mse"].items()},
                            "seeds_mae": {int(k): v for k, v in s["seeds_mae"].items()}}
                       for nm, s in r["summary"].items()}
            results.append(make_test_row(name, summary, r.get("det_ok", True)))

    order = {d["name"]: i for i, d in enumerate(DATASETS)}
    rows = sorted(results, key=lambda r: order.get(r["dataset"], 99))

    # ablation csv（7 数据集 per-arm-seed）
    ablation_rows = []
    for d in DATASETS:
        rp = TEST_OUT / d["name"] / "test_ablation.csv"
        if not rp.exists():
            continue
        for r in csv.DictReader(rp.open(encoding="utf-8")):
            ablation_rows.append({"dataset": d["name"], "arm": r["arm"], "seed": int(r["seed"]),
                                  "test_mse": float(r["test_mse"]), "test_mae": float(r["test_mae"])})

    def _csv(fname, rlist, cols):
        with (TEST_OUT / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rlist:
                w.writerow(r)

    _csv("final_test_summary.csv", rows, SUMMARY_FIELDS)
    _csv("final_test_ablation.csv", ablation_rows,
         ["dataset", "arm", "seed", "test_mse", "test_mae"])

    gen_rows = []
    for r in rows:
        val = val_summary.get(r["dataset"])
        gen_rows.append({
            "dataset": r["dataset"],
            "val_v2_vs_raw_pct": val if val is not None else "",
            "test_v2_vs_raw_pct": r["v2_vs_raw_pct"],
            "direction_preserved": ("" if val is None else bool((val > 0) == (r["v2_vs_raw_pct"] > 0))),
        })
    _csv("val_test_generalization.csv", gen_rows,
         ["dataset", "val_v2_vs_raw_pct", "test_v2_vs_raw_pct", "direction_preserved"])

    stats = compute_stats(rows)
    (TEST_OUT / "final_test_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(rows, stats, val_summary, missing)
    (TEST_OUT / "final_test_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("FROZEN FINAL TEST SUMMARY")
    print("=" * 78)
    for r in rows:
        print(f"{r['dataset']:<14} raw={r['raw_mean_mse']:.4e} v1={r['v1_mean_mse']:.4e} "
              f"v2={r['v2_mean_mse']:.4e} v2_vs_raw={r['v2_vs_raw_pct']:+.1f}% "
              f"v2_vs_v1={r['v2_vs_v1_pct']:+.1f}% wins_raw={r['wins_vs_raw_seeds']}/3")
    print(f"\nTEST v2 比 Raw 好：{stats['better_raw']}/{stats['n']}；v2 比 v1 好：{stats['better_v1']}/{stats['n']}")
    print(f"mean RI vs Raw={stats['mean_RI_vs_raw']*100:+.2f}%  median={stats['median_RI_vs_raw']*100:+.2f}%")
    if missing:
        print(f"MISSING ARTIFACT: {missing}")
    if failed:
        print(f"FAILED datasets: {failed}")
    print(f"\nSaved to {TEST_OUT}")


if __name__ == "__main__":
    main()
