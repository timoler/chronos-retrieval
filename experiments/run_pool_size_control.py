#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 等 Pool Size 控制（Pool-Aware v2 是否只是「保留更多 experts」）。

关键审稿问题 #2/#3：
  v2 比 v1 好是不是仅仅因为保留了更多 experts？
  Pool-Aware Reward 是否提供了独立于 pool size 的价值？

设计（TRAIN / frozen VAL only，绝不访问 TEST）：
  每个 dataset 读 v2 冻结 pool size K_v2，比较：
    1 V2_POOL_AWARE   —— v2 冻结 pool（K_v2 个 expert，pool-aware reward 选出）
    2 RANDOM_K        —— 从 Raw 8 个 expert 随机选 K_v2（control seeds=[11,22,33,44,55]）
    3 V1_SCORE_TOPK   —— 按 v1 逐条 individual reward 选 K_v2
    4 RAW_ALL         —— Raw 全 8 个 expert

  同一 retriever、初始化、训练预算、split，唯一区别 expert subset。

优化：raw pool 的 embedding 与 activation matrix 只算一次；RANDOM_K / V1_SCORE_TOPK
是 raw 的列/行子集（build_activation_matrix 逐 expert 独立，故列选择 == 重新计算）。

输出：results/paper_completion/paper_pool_size_control.csv + pool_size_control_report.md
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import time
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

from run_expert_embedding_mlp import build_windows, mse
from run_solar_same_val_baselines import residualize
from run_conditional_expert_mlp_v4 import ConditionalExpertMLPRetriever
from run_etth1_prediction_validation import compute_training_stats, build_activation_matrix

import run_etth1_reward_feedback_critic_v1 as v1
import run_reward_feedback_critic_v2_poolaware_pilot as v2p

REPO = HERE.parent
FROZEN_ROOT = REPO / "results/reward_feedback_critic_v2_full_cross_dataset"
V1_CROSS = v2p.V1_CROSS
V1_ETTH1 = v2p.V1_ETTH1
OUT = REPO / "results/paper_completion"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = v2p.SEEDS            # [42,123,2026]
N_RAW_TARGET = v2p.N_RAW_TARGET
CONTROL_SEEDS = [11, 22, 33, 44, 55]

DATASETS = [
    {"name": "etth1", "domain": "transformer_temperature", "freq": "hourly"},
    {"name": "etth2", "domain": "transformer_temperature", "freq": "hourly"},
    {"name": "weather", "domain": "weather_meteorological", "freq": "10-minute"},
    {"name": "exchange_rate", "domain": "daily_exchange_rate", "freq": "daily"},
    {"name": "solar", "domain": "aggregate_solar_generation", "freq": "10-minute"},
    {"name": "electricity", "domain": "aggregate_electricity_load", "freq": "15-minute"},
    {"name": "traffic", "domain": "road_traffic_occupancy", "freq": "hourly"},
]


def load_raw_pool(name):
    src = FROZEN_ROOT / name / "raw_experts.json"
    if src.exists():
        return v1._sorted_experts(json.loads(src.read_text(encoding="utf-8")))
    raw_src = V1_ETTH1 / "raw_experts.json" if name == "etth1" else V1_CROSS / name / "raw_experts.json"
    experts = v1._sorted_experts(json.loads(raw_src.read_text(encoding="utf-8")))[:N_RAW_TARGET]
    for e in experts:
        e["claim"] = e.get("mechanism", e.get("reason", ""))
        e["observable_features"] = list(e.get("observables", []))
        e["replaced"] = False
    return experts


def load_v2_pool(name):
    return json.loads((FROZEN_ROOT / name / "final_expert_pool.json").read_text(encoding="utf-8"))


def v1_individual_scores(name):
    """v1 逐条 individual reward（final-round reward + action_effect_size 作 tiebreak）。"""
    src = V1_ETTH1 / "reward_history.csv" if name == "etth1" else V1_CROSS / name / "reward_history.csv"
    best = {}
    for r in csv.DictReader(src.open(encoding="utf-8")):
        lid = r["lineage_id"]
        if lid not in best or int(r["round"]) >= int(best[lid]["round"]):
            best[lid] = r

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")
    return {lid: (int(r["reward"]), _f(r.get("action_effect_size")), _f(r.get("median_gain")))
            for lid, r in best.items()}


def _init_state(seed, context, expert_dim, n_slots, device):
    v2p.seed_everything(seed)
    base = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=v2p.D_ENC, d_exp=v2p.D_EXP, d_z=v2p.D_Z, hidden=v2p.HIDDEN).to(device)
    st = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    return st


def run_arm(emb, act, context, expert_dim, n_slots, device, seed):
    return v2p.run_val_arm(_init_state(seed, context, expert_dim, n_slots, device),
                           emb, act, True, expert_dim, n_slots, device, seed, context)


def run_dataset(name, device):
    ds_dir = REPO / "results/multidataset" / name
    cfg_json = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg_json["context"])
    horizon = int(cfg_json["horizon"])
    v1.CONTEXT = context
    v1.HORIZON = horizon

    t0 = time.perf_counter()
    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    v2p.X_bank, v2p.Y_bank = build_windows(series, bank_starts, context, horizon)
    v2p.X_fit, v2p.Y_fit = build_windows(series, fit_starts, context, horizon)
    v2p.X_val, v2p.Y_val = build_windows(series, val_starts, context, horizon)
    v2p.X_bank_res = residualize(v2p.X_bank)
    v2p.X_fit_res = residualize(v2p.X_fit)
    v2p.X_val_res = residualize(v2p.X_val)
    v2p.bank_ts = [dates[s + context] for s in bank_starts]
    v2p.fit_ts = [dates[s + context] for s in fit_starts]
    v2p.val_ts = [dates[s + context] for s in val_starts]
    v2p.stats = compute_training_stats(np.concatenate([v2p.X_bank, v2p.X_fit], axis=0))
    t_data = time.perf_counter() - t0

    raw_pool = load_raw_pool(name)
    v2_pool = load_v2_pool(name)
    K = len(v2_pool)
    print(f"  raw={len(raw_pool)} K_v2={K} v2_ids={[e['id'] for e in v2_pool]} "
          f"data_sec={t_data:.1f}")

    # raw emb/act 只算一次（8 experts）
    t0 = time.perf_counter()
    raw_emb = v1.embed_claims([e["claim"] for e in raw_pool], device)
    raw_act = v1.act_mats(raw_pool, v2p.X_bank, v2p.X_fit, v2p.X_val,
                          v2p.bank_ts, v2p.fit_ts, v2p.val_ts, v2p.stats)
    t_emb_act = time.perf_counter() - t0
    expert_dim = raw_emb.shape[1]
    n_slots = N_RAW_TARGET

    def eval_subset(indices, label):
        idxs = np.array(sorted(indices), dtype=int)
        sub_emb = raw_emb[idxs]
        sub_act = {k: raw_act[k][:, idxs] for k in raw_act}
        sub_emb, sub_act, _ = v1.pad_to_slots(sub_emb, sub_act, n_slots)
        per = {s: run_arm(sub_emb, sub_act, context, expert_dim, n_slots, device, s) for s in SEEDS}
        return per

    def eval_fresh(pool, label):
        emb = v1.embed_claims([e["claim"] for e in pool], device)
        act = v1.act_mats(pool, v2p.X_bank, v2p.X_fit, v2p.X_val,
                          v2p.bank_ts, v2p.fit_ts, v2p.val_ts, v2p.stats)
        emb, act, _ = v1.pad_to_slots(emb, act, n_slots)
        per = {s: run_arm(emb, act, context, expert_dim, n_slots, device, s) for s in SEEDS}
        return per

    # 4 RAW_ALL（raw 全 8）
    raw_emb8, raw_act8, _ = v1.pad_to_slots(raw_emb, raw_act, n_slots)
    raw_seeds = {s: run_arm(raw_emb8, raw_act8, context, expert_dim, n_slots, device, s)
                 for s in SEEDS}
    # 1 V2_POOL_AWARE（v2 是修订版，需 fresh）
    v2_seeds = eval_fresh(v2_pool, "v2")
    print(f"    emb+act_sec={t_emb_act:.1f}  V2={np.mean(list(v2_seeds.values())):.6g} "
          f"RAW={np.mean(list(raw_seeds.values())):.6g}")

    # 2 RANDOM_K
    rand_draws, rand_ids = {}, {}
    for c in CONTROL_SEEDS:
        rng = np.random.default_rng(c)
        idxs = sorted(rng.choice(len(raw_pool), size=K, replace=False).tolist())
        rand_ids[c] = [raw_pool[i]["id"] for i in idxs]
        rand_draws[c] = eval_subset(idxs, f"rand{c}")
        print(f"    RANDOM_K seed={c} subset={rand_ids[c]} "
              f"mse={np.mean(list(rand_draws[c].values())):.6g}")

    # 3 V1_SCORE_TOPK
    scores = v1_individual_scores(name)
    ranked_idx = sorted(range(len(raw_pool)),
                        key=lambda i: scores.get(raw_pool[i]["id"], (-9, float("nan"), float("nan"))),
                        reverse=True)
    topk_idx = ranked_idx[:K]
    v1score_seeds = eval_subset(topk_idx, "v1topk")
    v1score_ids = [raw_pool[i]["id"] for i in topk_idx]
    print(f"    V1_SCORE_TOPK subset={v1score_ids} mse={np.mean(list(v1score_seeds.values())):.6g}")

    return {
        "dataset": name, "K_v2": K, "v2": v2_seeds, "raw": raw_seeds,
        "v1_score_topk": v1score_seeds, "v1_score_topk_ids": v1score_ids,
        "random_k": {str(c): rand_draws[c] for c in CONTROL_SEEDS},
        "random_k_ids": {str(c): rand_ids[c] for c in CONTROL_SEEDS},
        "timing": {"data_sec": t_data, "emb_act_sec": t_emb_act,
                   "n_arm_trainings": (len(SEEDS) * (2 + len(CONTROL_SEEDS) + 1))},
    }


def _result_to_jsonable(r):
    return {
        "dataset": r["dataset"], "K_v2": r["K_v2"],
        "v2": {str(k): v for k, v in r["v2"].items()},
        "raw": {str(k): v for k, v in r["raw"].items()},
        "v1_score_topk": {str(k): v for k, v in r["v1_score_topk"].items()},
        "v1_score_topk_ids": r["v1_score_topk_ids"],
        "random_k": {str(k): {str(s): v for s, v in d.items()} for k, d in r["random_k"].items()},
        "random_k_ids": r["random_k_ids"],
        "timing": r["timing"],
    }


def merge_parts():
    """读 _p2_parts/*.json 合并成 rows，并写 CSV + report。"""
    part_dir = OUT / "_p2_parts"
    results = []
    if part_dir.exists():
        for p in sorted(part_dir.glob("*.json")):
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                results.append({
                    "dataset": j["dataset"], "K_v2": j["K_v2"],
                    "v2": {int(k): v for k, v in j["v2"].items()},
                    "raw": {int(k): v for k, v in j["raw"].items()},
                    "v1_score_topk": {int(k): v for k, v in j["v1_score_topk"].items()},
                    "v1_score_topk_ids": j["v1_score_topk_ids"],
                    "random_k": {int(k): {int(s): v for s, v in d.items()} for k, d in j["random_k"].items()},
                    "random_k_ids": j["random_k_ids"],
                    "timing": j.get("timing", {}),
                })
            except Exception as e:
                print(f"  [merge skip] {p}: {e}")
    order = {d["name"]: i for i, d in enumerate(DATASETS)}
    results = sorted(results, key=lambda r: order.get(r["dataset"], 99))
    _write_merged(results)


def _write_merged(results):
    rows = []
    for r in results:
        name = r["dataset"]; K = r["K_v2"]
        v2m = float(np.mean([r["v2"][s] for s in SEEDS]))
        rawm = float(np.mean([r["raw"][s] for s in SEEDS]))
        v1topm = float(np.mean([r["v1_score_topk"][s] for s in SEEDS]))
        rand_means = [float(np.mean([r["random_k"][c][s] for s in SEEDS])) for c in CONTROL_SEEDS]
        rand_m = float(np.mean(rand_means)); rand_std = float(np.std(rand_means))
        rows.append({
            "dataset": name, "K_v2": K,
            "v2_mean_mse": v2m, "raw_all_mean_mse": rawm,
            "v1_score_topk_mean_mse": v1topm,
            "random_k_mean_mse": rand_m, "random_k_std_mse": rand_std,
            "v2_vs_raw_pct": (rawm - v2m) / rawm * 100.0,
            "v2_vs_random_pct": (rand_m - v2m) / rand_m * 100.0,
            "v2_vs_v1score_pct": (v1topm - v2m) / v1topm * 100.0,
            "v2_beats_random": int(v2m < rand_m),
            "v2_beats_v1score": int(v2m < v1topm),
            "v2_pool_ids": ",".join(e["id"] for e in load_v2_pool(name)),
        })

    if rows:
        with open(OUT / "paper_pool_size_control.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

        n_v2_rand = sum(1 for r in rows if r["v2_beats_random"])
        n_v2_v1s = sum(1 for r in rows if r["v2_beats_v1score"])
        L = ["# P2 等 Pool Size 控制（Pool-Aware v2 是否只是「保留更多 experts」）", ""]
        L.append("TRAIN / frozen VAL only，未访问 TEST。同一 retriever/初始化/训练预算/split，唯一区别 expert subset。")
        L.append("RANDOM_K：raw 8 选 K_v2，control seeds=[11,22,33,44,55]；V1_SCORE_TOPK：按 v1 individual reward 选 K_v2。")
        L.append("")
        L.append("| Dataset | K_v2 | v2 | RAW_ALL | RANDOM_K | V1_SCORE_TOPK | v2 vs RAW | v2 vs RANDOM | v2 vs V1score |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['dataset']} | {r['K_v2']} | {r['v2_mean_mse']:.6g} | {r['raw_all_mean_mse']:.6g} "
                     f"| {r['random_k_mean_mse']:.6g}±{r['random_k_std_mse']:.6g} | {r['v1_score_topk_mean_mse']:.6g} "
                     f"| {r['v2_vs_raw_pct']:+.1f}% | {r['v2_vs_random_pct']:+.1f}% | {r['v2_vs_v1score_pct']:+.1f}% |")
        L.append("")
        L.append("## 结果")
        L.append(f"- v2 优于 RANDOM_K（同 K）：{n_v2_rand}/{len(rows)} 数据集")
        L.append(f"- v2 优于 V1_SCORE_TOPK（同 K）：{n_v2_v1s}/{len(rows)} 数据集")
        L.append("")
        L.append("## 结论")
        if n_v2_rand > len(rows) / 2 and n_v2_v1s > len(rows) / 2:
            L.append("- Pool-Aware Reward 提供了独立于 pool size 的价值（同 K 下 v2 优于随机与 v1-score 基线）。")
        elif n_v2_rand > len(rows) / 2:
            L.append("- v2 优于随机同 K，但相对 v1-score 未见稳定优势 → 部分支持「独立于 size」。")
        else:
            L.append("- v2 相对随机同 K 未普遍占优 → 提升可能主要由 pool size 驱动（如实报告）。")
        with open(OUT / "pool_size_control_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(L))

        print("\n" + "=" * 70)
        for r in rows:
            print(f"{r['dataset']:<14} K={r['K_v2']} v2={r['v2_mean_mse']:.4e} raw={r['raw_all_mean_mse']:.4e} "
                  f"randK={r['random_k_mean_mse']:.4e} v1topK={r['v1_score_topk_mean_mse']:.4e} "
                  f"v2_vs_rand={r['v2_vs_random_pct']:+.1f}% v2_vs_v1s={r['v2_vs_v1score_pct']:+.1f}%")
        print(f"\nv2>RANDOM_K: {n_v2_rand}/{len(rows)}   v2>V1_SCORE_TOPK: {n_v2_v1s}/{len(rows)}")
    print(f"Saved to {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--merge-only", action="store_true",
                    help="只合并 _p2_parts 里的已有结果，不跑新数据集")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    part_dir = OUT / "_p2_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    targets = [d for d in DATASETS if (args.datasets is None or d["name"] in args.datasets)]
    for cfg in targets:
        name = cfg["name"]
        part = part_dir / f"{name}.json"
        if part.exists():
            print(f"  [skip] {name} already has part json", flush=True)
            continue
        print(f"\n{'=' * 70}\nDATASET {name}\n{'=' * 70}", flush=True)
        try:
            r = run_dataset(name, device)
            part.write_text(json.dumps(_result_to_jsonable(r), ensure_ascii=False, indent=2),
                            encoding="utf-8")
            print(f"  [part saved] {part}", flush=True)
        except Exception as e:
            print(f"  [FAILED] {name}: {e}", flush=True)
            traceback.print_exc()

    merge_parts()


if __name__ == "__main__":
    main()
