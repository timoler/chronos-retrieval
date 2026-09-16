#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward-Feedback Critic v2 全量跨数据集验证（冻结 Pool-Aware Reward）。

冻结当前最优 v2（Pool-Aware Reward），不再改方法，跑剩余 4 个数据集：
ETTh2 / Solar / ExchangeRate / Traffic，并与已有 ETTh1 / Electricity / Weather 合并成
7 数据集总表。

唯一被允许的 per-dataset 输入：数据集名称 + domain/freq metadata + context/horizon
（来自各数据集 config.json，是数据准备属性而非方法超参）。

冻结项（全部保持 v2，不修改）：
  Expert prompt / Critic prompt / Pool-aware reward 与阈值 / Compiler /
  retrieval_action / activation / text embedding / learned retriever / Top-K /
  optimizer / epochs / batch / revision rounds / deterministic protocol / seeds。

禁止：针对单数据集改规则、改 reward/Critic/Expert prompt、挑 seed、访问 TEST、
用 v2.1/v3、根据 VAL 回头改方法。

v2 实现完全复用 `run_reward_feedback_critic_v2_poolaware_pilot`（模块级复用，只
重定向输出目录），保证与已冻结的 v2 代码逐字节一致。

输出：results/reward_feedback_critic_v2_full_cross_dataset/
"""

from __future__ import annotations

import argparse
import csv
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
import run_agent_v2_electricity as av2
import run_reward_feedback_critic_v2_poolaware_pilot as v2p

REPO_ROOT = HERE.parent
FULL_OUT_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_full_cross_dataset"
FULL_OUT_ROOT.mkdir(parents=True, exist_ok=True)
FULL_LLM_CACHE = REPO_ROOT / "cache/llm_priors_reward_feedback_critic_v2_full_cross_dataset"

# 复用冻结 v2 的模块级输出目录（把 pool_reward_history 等写到本次目录，逻辑不变）
v2p.OUT_ROOT = FULL_OUT_ROOT
v2p.LLM_CACHE_ROOT = FULL_LLM_CACHE

# 已有 v2 正式结果（3 数据集，只读不覆盖）
V2_PILOT = REPO_ROOT / "results/reward_feedback_critic_v2_poolaware_pilot"
# v1 冻结 pool 来源（raw_experts.json / final_expert_pool.json）
V1_CROSS = v2p.V1_CROSS  # results/reward_feedback_critic_v1_cross_dataset

SEEDS = v2p.SEEDS  # [42, 123, 2026]
TOPK = v2p.TOPK
N_RAW_TARGET = v2p.N_RAW_TARGET
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

# 本次要跑 v2 的剩余 4 个数据集（metadata 来自各 config.json，非专家模板）
REMAINING_DATASETS = [
    {"name": "etth2",         "domain": "transformer_temperature",  "freq": "hourly"},
    {"name": "solar",         "domain": "aggregate_solar_generation", "freq": "10-minute"},
    {"name": "exchange_rate", "domain": "daily_exchange_rate",       "freq": "daily"},
    {"name": "traffic",       "domain": "road_traffic_occupancy",    "freq": "hourly"},
]

# 总表数据集顺序（3 个已有 + 4 个新）
ALL_DATASET_ORDER = ["etth1", "electricity", "weather",
                     "etth2", "solar", "exchange_rate", "traffic"]


class DeterminismFailure(RuntimeError):
    """determinism 检查失败：必须停止，不可继续（协议本身不可复现）。"""


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# --------------------------------------------------------------------------- #
# VAL ablation A/B/C/D(v1)/E(v2)，含 determinism 检查
# --------------------------------------------------------------------------- #
def run_val_with_det(name, raw_experts, v1_pool, v2_pool, device):
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

    # 写入 v2p 模块全局，供 v2p.run_val_arm 复用（避免重写训练逻辑）
    v2p.X_bank, v2p.Y_bank = X_bank, Y_bank
    v2p.X_fit, v2p.Y_fit = X_fit, Y_fit
    v2p.X_val, v2p.Y_val = X_val, Y_val
    v2p.X_bank_res, v2p.X_fit_res, v2p.X_val_res = X_bank_res, X_fit_res, X_val_res

    emb_raw = v1.embed_claims([e["claim"] for e in raw_experts], device)
    emb_v1 = v1.embed_claims([e["claim"] for e in v1_pool], device) if v1_pool else \
        np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    emb_v2 = v1.embed_claims([e["claim"] for e in v2_pool], device) if v2_pool else \
        np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    expert_dim = emb_raw.shape[1]
    n_slots = max(len(raw_experts), len(v1_pool), len(v2_pool))

    def _empty_act():
        return {"bank": np.zeros((len(X_bank), 0), dtype=np.float32),
                "fit": np.zeros((len(X_fit), 0), dtype=np.float32),
                "val": np.zeros((len(X_val), 0), dtype=np.float32)}

    raw_act = v1.act_mats(raw_experts, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    v1_act = v1.act_mats(v1_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats) if v1_pool else _empty_act()
    v2_act = v1.act_mats(v2_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats) if v2_pool else _empty_act()
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
        seed_everything(seed)
        m = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    # determinism 检查：v2 seed2026 重复 2 次（同一 init_state）
    emb_d, act_d, use_d = configs["v2"]
    init_state_d = _init_state(2026)
    det_mse = []
    for _ in range(2):
        mm = v2p.run_val_arm(init_state_d, emb_d, act_d, use_d,
                             expert_dim, n_slots, device, 2026, context)
        det_mse.append(mm)
    det_ok = abs(det_mse[0] - det_mse[1]) <= 1e-4
    print(f"  determinism check: v2 seed2026 x2 = {det_mse} -> {'PASS' if det_ok else 'FAIL'}")
    if not det_ok:
        raise DeterminismFailure(f"determinism check FAILED for {name}: {det_mse}")

    per_seed = {nm: {} for nm in configs}
    for seed in SEEDS:
        init_state = _init_state(seed)
        for nm, (emb, act, use_shape) in configs.items():
            mm = v2p.run_val_arm(init_state, emb, act, use_shape,
                                 expert_dim, n_slots, device, seed, context)
            per_seed[nm][seed] = mm
            print(f"    seed={seed} [{nm}] MSE={mm:.4f}")

    summary = {}
    for nm in configs:
        ms = [per_seed[nm][s] for s in SEEDS]
        summary[nm] = {"mean": float(np.mean(ms)), "std": float(np.std(ms)),
                       "seeds": {s: per_seed[nm][s] for s in SEEDS}}
    return summary, per_seed, det_ok


# --------------------------------------------------------------------------- #
# 单数据集
# --------------------------------------------------------------------------- #
def run_one_dataset(name, cfg, device):
    out_dir = FULL_OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 78}\nDATASET {name}  domain={cfg['domain']} freq={cfg['freq']}\n{'=' * 78}")

    # 1. v2 闭环（复用冻结 v2 的 run_v2_loop，只重定向输出目录）
    raw_experts, v2_pool = v2p.run_v2_loop(name, cfg, device)
    print(f"  v2 final pool: {len(v2_pool)} -> {[e['id'] for e in v2_pool]}")

    # 2. v1 冻结 pool（只读，不复跑 v1 方法）
    v1_pool = v1._sorted_experts(json.loads(
        (V1_CROSS / name / "final_expert_pool.json").read_text(encoding="utf-8")))
    print(f"  v1 final pool (frozen): {len(v1_pool)} -> {[e['id'] for e in v1_pool]}")

    # 3. VAL ablation（5 arms x 3 seeds，含 determinism 检查）
    summary, per_seed, det_ok = run_val_with_det(name, raw_experts, v1_pool, v2_pool, device)

    # 4. 写 per-dataset 输出
    (out_dir / "raw_experts.json").write_text(
        json.dumps(raw_experts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "final_expert_pool.json").write_text(
        json.dumps(v2_pool, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    with (out_dir / "val_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "mse"])
        for nm in ("base", "shape", "raw", "v1", "v2"):
            for s in SEEDS:
                w.writerow([nm, s, f"{per_seed[nm][s]:.6f}"])
        w.writerow([])
        w.writerow(["arm", "mse_mean", "mse_std"])
        for nm in ("base", "shape", "raw", "v1", "v2"):
            w.writerow([nm, f"{summary[nm]['mean']:.6f}", f"{summary[nm]['std']:.6f}"])

    result = {
        "dataset": name, "det_ok": det_ok,
        "raw_pool_size": len(raw_experts), "v1_pool_size": len(v1_pool), "v2_pool_size": len(v2_pool),
        "v2_pool_ids": [e["id"] for e in v2_pool],
        "summary": summary,
        "per_seed": {nm: {str(s): per_seed[nm][s] for s in SEEDS} for nm in per_seed},
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  -> saved to {out_dir}")
    return result


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def make_row(dataset, summary, raw_pool_size, v1_pool_size, v2_pool_size):
    raw_m = summary["raw"]["mean"]
    v1_m = summary["v1"]["mean"]
    v2_m = summary["v2"]["mean"]
    wins_raw = sum(1 for s in SEEDS if summary["v2"]["seeds"][s] < summary["raw"]["seeds"][s])
    wins_v1 = sum(1 for s in SEEDS if summary["v2"]["seeds"][s] < summary["v1"]["seeds"][s])
    return {
        "dataset": dataset,
        "base_mean_mse": summary["base"]["mean"],
        "shape_mean_mse": summary["shape"]["mean"],
        "raw_mean_mse": raw_m,
        "v1_mean_mse": v1_m,
        "v2_mean_mse": v2_m,
        "raw_std": summary["raw"]["std"],
        "v2_std": summary["v2"]["std"],
        "v2_vs_raw_pct": (raw_m - v2_m) / raw_m * 100.0,
        "v2_vs_v1_pct": (v1_m - v2_m) / v1_m * 100.0,
        "wins_vs_raw_seeds": wins_raw,
        "wins_vs_v1_seeds": wins_v1,
        "raw_pool_size": raw_pool_size,
        "v1_pool_size": v1_pool_size,
        "v2_pool_size": v2_pool_size,
    }


def load_saved_result(name):
    """读取本目录下已跑完的数据集 result.json（断点续跑合并用），并把 JSON
    stringify 后的 seed 键转回 int。"""
    p = FULL_OUT_ROOT / name / "result.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    r["summary"] = {
        nm: {"mean": float(s["mean"]), "std": float(s["std"]),
             "seeds": {int(k): float(v) for k, v in s["seeds"].items()}}
        for nm, s in r["summary"].items()}
    return r


def load_pilot_rows():
    """读取已冻结的 3 个数据集正式结果（只读，不重跑）。"""
    srows = {r["dataset"]: r for r in csv.DictReader(
        (V2_PILOT / "pilot_summary.csv").open(encoding="utf-8"))}
    abl = {}
    for r in csv.DictReader((V2_PILOT / "pilot_val_ablation.csv").open(encoding="utf-8")):
        abl[(r["dataset"], r["arm"], int(r["seed"]))] = float(r["mse"])

    def arm_mean(ds, arm):
        return float(np.mean([abl[(ds, arm, s)] for s in SEEDS]))

    def arm_std(ds, arm):
        return float(np.std([abl[(ds, arm, s)] for s in SEEDS]))

    rows = []
    for ds in ("etth1", "electricity", "weather"):
        sr = srows[ds]
        raw_m = float(sr["raw_mean_mse"])
        v1_m = float(sr["v1_mean_mse"])
        v2_m = float(sr["v2_mean_mse"])
        rows.append({
            "dataset": ds,
            "base_mean_mse": arm_mean(ds, "base"),
            "shape_mean_mse": arm_mean(ds, "shape"),
            "raw_mean_mse": raw_m,
            "v1_mean_mse": v1_m,
            "v2_mean_mse": v2_m,
            "raw_std": arm_std(ds, "raw"),
            "v2_std": arm_std(ds, "v2"),
            "v2_vs_raw_pct": (raw_m - v2_m) / raw_m * 100.0,
            "v2_vs_v1_pct": (v1_m - v2_m) / v1_m * 100.0,
            "wins_vs_raw_seeds": int(sr["v2_wins_vs_raw_seeds"]),
            "wins_vs_v1_seeds": int(sr["v2_wins_vs_v1_seeds"]),
            "raw_pool_size": N_RAW_TARGET,
            "v1_pool_size": int(sr["v1_pool_size"]),
            "v2_pool_size": int(sr["v2_pool_size"]),
        })
    return rows


def load_pilot_ablation_rows():
    """3 个已有数据集的 per-arm-seed，用于合并 full ablation csv。"""
    rows = []
    for r in csv.DictReader((V2_PILOT / "pilot_val_ablation.csv").open(encoding="utf-8")):
        rows.append({"dataset": r["dataset"], "arm": r["arm"], "seed": int(r["seed"]),
                     "mse": float(r["mse"])})
    return rows


def compute_cross_stats(rows):
    n = len(rows)
    ri_raw = [(r["raw_mean_mse"] - r["v2_mean_mse"]) / r["raw_mean_mse"] for r in rows]
    ri_v1 = [(r["v1_mean_mse"] - r["v2_mean_mse"]) / r["v1_mean_mse"] for r in rows]
    return {
        "n": n,
        "mean_RI_vs_raw": float(np.mean(ri_raw)) if ri_raw else 0.0,
        "median_RI_vs_raw": float(np.median(ri_raw)) if ri_raw else 0.0,
        "positive_dataset_ratio_vs_raw": float(np.mean([x > 0 for x in ri_raw])) if ri_raw else 0.0,
        "mean_RI_vs_v1": float(np.mean(ri_v1)) if ri_v1 else 0.0,
        "median_RI_vs_v1": float(np.median(ri_v1)) if ri_v1 else 0.0,
        "positive_dataset_ratio_vs_v1": float(np.mean([x > 0 for x in ri_v1])) if ri_v1 else 0.0,
        "better_raw": sum(1 for r in rows if r["v2_mean_mse"] < r["raw_mean_mse"]),
        "better_v1": sum(1 for r in rows if r["v2_mean_mse"] < r["v1_mean_mse"]),
        "full_wins_raw": sum(1 for r in rows if r["wins_vs_raw_seeds"] == 3),
        "wins_23_raw": sum(1 for r in rows if r["wins_vs_raw_seeds"] >= 2),
        "std_lower": sum(1 for r in rows if r["v2_std"] < r["raw_std"]),
    }


SUMMARY_FIELDS = ["dataset", "base_mean_mse", "shape_mean_mse", "raw_mean_mse",
                  "v1_mean_mse", "v2_mean_mse", "v2_std", "v2_vs_raw_pct",
                  "v2_vs_v1_pct", "wins_vs_raw_seeds", "wins_vs_v1_seeds",
                  "raw_pool_size", "v2_pool_size", "raw_std", "v1_pool_size"]


def render_report(rows, stats, failed):
    def pct(x):
        return f"{x * 100:+.2f}%"

    L = []
    L.append("# Reward-Feedback Critic v2 全量跨数据集验证（Pool-Aware Reward）")
    L.append("")
    L.append("冻结 v2（Pool-Aware Reward），跑剩余 4 个数据集，合并已有 3 个，共 7 数据集。")
    L.append("不访问 TEST、不改方法、不挑 seed、失败数据集保留并标记。")
    L.append("")

    L.append("## 结果")
    L.append("")
    L.append(f"- v2 比 Raw 好：{stats['better_raw']}/{stats['n']}")
    L.append(f"- v2 比 v1 好：{stats['better_v1']}/{stats['n']}")
    L.append(f"- mean RI vs Raw = {pct(stats['mean_RI_vs_raw'])}")
    L.append(f"- median RI vs Raw = {pct(stats['median_RI_vs_raw'])}")
    L.append(f"- mean RI vs v1 = {pct(stats['mean_RI_vs_v1'])}")
    L.append(f"- median RI vs v1 = {pct(stats['median_RI_vs_v1'])}")
    L.append(f"- 3/3 seeds 赢 Raw：{stats['full_wins_raw']}/{stats['n']}")
    L.append(f"- 2/3 或 3/3 seeds 赢 Raw：{stats['wins_23_raw']}/{stats['n']}")
    L.append(f"- v2 std < Raw std：{stats['std_lower']}/{stats['n']}")
    fail_ds = [r["dataset"] for r in rows if r["v2_mean_mse"] >= r["raw_mean_mse"]]
    if fail_ds:
        L.append(f"- v2 比 Raw 差的数据集：{fail_ds}")
    if failed:
        L.append(f"- 运行失败/未纳入：{failed}")
    L.append("")

    L.append("主表：")
    L.append("")
    L.append("| Dataset | Raw | v1 | v2 | v2 vs Raw | v2 vs v1 | wins/raw | Pool v1→v2 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['dataset']} | {r['raw_mean_mse']:.6g} | {r['v1_mean_mse']:.6g} "
                 f"| {r['v2_mean_mse']:.6g} | {r['v2_vs_raw_pct']:+.1f}% | {r['v2_vs_v1_pct']:+.1f}% "
                 f"| {r['wins_vs_raw_seeds']}/3 | {r['v1_pool_size']}→{r['v2_pool_size']} |")
    L.append("")

    L.append("Pool size 是否仍有明显 collapse：")
    L.append("")
    L.append("| Dataset | Raw pool | v1 pool | v2 pool |")
    L.append("|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['dataset']} | {r['raw_pool_size']} | {r['v1_pool_size']} | {r['v2_pool_size']} |")
    L.append("")

    for r in rows:
        if r["v2_mean_mse"] >= r["raw_mean_mse"]:
            L.append(f"### FAIL_DATASET: {r['dataset']}")
            L.append(f"- v2 vs Raw = {r['v2_vs_raw_pct']:+.1f}%（v2={r['v2_mean_mse']:.6g}，Raw={r['raw_mean_mse']:.6g}）")
            L.append("- possible causes: marginal reward mismatch / pool over-pruning / "
                     "weak expert signal / optimization sensitivity / combination effect")
            L.append("")
    for d in failed:
        L.append(f"### FAIL_DATASET: {d}")
        L.append("- 运行失败（错误或 determinism 检查未通过），未纳入汇总。")
        L.append("")

    L.append("## 结论")
    L.append("")
    n = stats["n"]
    if n == 0:
        concl = "FAIL"
    elif (stats["better_raw"] > n / 2 and stats["mean_RI_vs_raw"] > 0
          and stats["median_RI_vs_raw"] > 0 and stats["better_v1"] > n / 2):
        concl = "STRONG SUCCESS"
    elif stats["better_v1"] > n / 2 and stats["mean_RI_vs_v1"] > 0:
        concl = "PARTIAL SUCCESS"
    else:
        concl = "FAIL"
    L.append(f"- **{concl}**")
    if concl == "STRONG SUCCESS":
        L.append("  （v2 多数数据集优于 Raw，mean_RI_vs_raw>0，median_RI_vs_raw>0，且多数优于 v1。）")
    elif concl == "PARTIAL SUCCESS":
        L.append("  （v2 明显优于 v1，但 vs Raw 仍有若干失败。）")
    else:
        L.append("  （v2 对 Raw 整体不占优，或多数数据集明显退化。）")
    L.append("")
    L.append("```")
    L.append("test_used = false")
    L.append("deterministic_protocol = true")
    L.append("method_frozen = v2_pool_aware_reward")
    L.append("```")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of remaining dataset names (default: all 4)")
    ap.add_argument("--resume", action="store_true",
                    help="skip remaining datasets whose result.json already exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    targets = [d for d in REMAINING_DATASETS
               if (args.datasets is None or d["name"] in args.datasets)]
    if args.resume:
        targets = [d for d in targets
                   if not (FULL_OUT_ROOT / d["name"] / "result.json").exists()]
    if args.datasets is None and not args.resume:
        print("running all 4 remaining datasets (use --resume to skip completed)")

    new_results = []
    failed = []
    for cfg in targets:
        name = cfg["name"]
        try:
            r = run_one_dataset(name, dict(cfg), device)
            new_results.append(r)
        except DeterminismFailure as e:
            print(f"\n[FATAL] {e}")
            print("STOPPING the whole run: determinism check failed (protocol not reproducible).")
            sys.exit(2)
        except Exception as e:
            print(f"\n[DATASET FAILED] {name}: {e}")
            traceback.print_exc()
            failed.append(name)

    # 断点续跑：本次没跑、但之前已跑完并落了 result.json 的剩余数据集也并入
    new_names = {r["dataset"] for r in new_results}
    for d in REMAINING_DATASETS:
        name = d["name"]
        if name in new_names:
            continue
        saved = load_saved_result(name)
        if saved is not None:
            new_results.append(saved)

    # 汇总：3 个已有（只读 pilot）+ 本次新跑（含续跑读回）
    rows = load_pilot_rows()
    for r in new_results:
        rows.append(make_row(
            r["dataset"], r["summary"],
            r["raw_pool_size"], r["v1_pool_size"], r["v2_pool_size"]))

    # 按固定顺序排
    order = {d: i for i, d in enumerate(ALL_DATASET_ORDER)}
    rows = sorted(rows, key=lambda r: order.get(r["dataset"], 99))

    # 全量 ablation csv（7 数据集 per-arm-seed）
    ablation_rows = load_pilot_ablation_rows()
    for r in new_results:
        for nm in ("base", "shape", "raw", "v1", "v2"):
            for s in SEEDS:
                ablation_rows.append({"dataset": r["dataset"], "arm": nm, "seed": s,
                                      "mse": r["per_seed"][nm][str(s)]})

    def _csv(fname, rlist, cols):
        with (FULL_OUT_ROOT / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rlist:
                w.writerow(r)

    _csv("full_cross_dataset_summary.csv", rows, SUMMARY_FIELDS)
    _csv("full_cross_dataset_ablation.csv", ablation_rows, ["dataset", "arm", "seed", "mse"])

    pool_rows = [{"dataset": r["dataset"], "raw_pool_size": r["raw_pool_size"],
                  "v1_pool_size": r["v1_pool_size"], "v2_pool_size": r["v2_pool_size"]}
                 for r in rows]
    _csv("full_cross_dataset_pool_stats.csv", pool_rows,
         ["dataset", "raw_pool_size", "v1_pool_size", "v2_pool_size"])

    stats = compute_cross_stats(rows)
    report = render_report(rows, stats, failed)
    (FULL_OUT_ROOT / "full_cross_dataset_report.md").write_text(report, encoding="utf-8")
    (FULL_OUT_ROOT / "full_cross_dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("FULL CROSS-DATASET SUMMARY (v2)")
    print("=" * 78)
    for r in rows:
        print(f"{r['dataset']:<14} raw={r['raw_mean_mse']:.4e} v1={r['v1_mean_mse']:.4e} "
              f"v2={r['v2_mean_mse']:.4e} v2_vs_raw={r['v2_vs_raw_pct']:+.1f}% "
              f"v2_vs_v1={r['v2_vs_v1_pct']:+.1f}% pool={r['v1_pool_size']}->{r['v2_pool_size']} "
              f"wins_raw={r['wins_vs_raw_seeds']}/3")
    print(f"\nv2 比 Raw 好：{stats['better_raw']}/{stats['n']}；v2 比 v1 好：{stats['better_v1']}/{stats['n']}")
    print(f"mean RI vs Raw={stats['mean_RI_vs_raw']*100:+.2f}%  median={stats['median_RI_vs_raw']*100:+.2f}%")
    if failed:
        print(f"FAILED datasets: {failed}")
    print(f"\nSaved to {FULL_OUT_ROOT}")


if __name__ == "__main__":
    main()
