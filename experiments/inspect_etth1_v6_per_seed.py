#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect ETTh1 v6 module-ablation per-seed results (read-only, no retrain).

Reads the ALREADY-COMPUTED v6 module-ablation results and lays out the per-seed
MSE / MAE for the four arms A/B/C/D, then isolates whether the seed-42 anomaly is
specific to the Critic arm.

This script does NOT re-train, does NOT modify the Critic / network / hyperparams,
and does NOT access TEST.  It only re-organises and compares numbers already on
disk.

Inputs:
  results/etth1_module_ablation_v6/per_seed_results.json
  results/etth1_module_ablation_v6/summary.json

Arms:
  A = BASE            (base)
  B = +SHAPE          (shape)
  C = +RAW EXPERT     (raw_expert)
  D = +CRITIC EXPERT  (critic_expert, v6 pool)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Emit UTF-8 so the Chinese section headers print correctly on any console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PER_SEED = Path("results/etth1_module_ablation_v6/per_seed_results.json")
SUMMARY = Path("results/etth1_module_ablation_v6/summary.json")

SEEDS = [42, 123, 2026]

# arm label -> key in the result JSONs
ARMS = [
    ("A", "base", "BASE"),
    ("B", "shape", "+SHAPE"),
    ("C", "raw_expert", "+RAW EXPERT"),
    ("D", "critic_expert", "+CRITIC EXPERT v6"),
]


def _fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    per_seed = json.loads(PER_SEED.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    arms_data = per_seed["arms"]
    seeds_used = [int(s) for s in per_seed["seeds"]]

    # per-arm, per-seed dicts: mse[arm_key][seed], mae[arm_key][seed]
    mse = {}
    mae = {}
    for _, key, _ in ARMS:
        rows = arms_data[key]
        mse[key] = {int(r["seed"]): float(r["mse"]) for r in rows}
        mae[key] = {int(r["seed"]): float(r["mae"]) for r in rows}

    def mean(vals):
        return sum(vals) / len(vals)

    def std(vals):
        m = mean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    print("=" * 100)
    print("ETTH1 v6 MODULE ABLATION — PER-SEED INSPECTION (read-only)")
    print("=" * 100)
    print(f"seeds = {seeds_used}")
    print()

    # ---- 1. A/B/C/D per-seed MSE / MAE tables ----
    print("【结果】")
    print()
    print("## 1. A/B/C/D per-seed MSE")
    print()
    hdr = "| arm | " + " | ".join(f"seed{s} MSE" for s in SEEDS) + " | mean MSE | std MSE |"
    sep = "|---|" + "|".join(["---"] * (len(SEEDS) + 2)) + "|"
    print(hdr)
    print(sep)
    for label, key, name in ARMS:
        vals = [mse[key][s] for s in SEEDS]
        cells = " | ".join(_fmt(v) for v in vals)
        print(f"| {label} {name} | {cells} | {_fmt(mean(vals))} | {_fmt(std(vals))} |")
    print()

    print("## 2. A/B/C/D per-seed MAE")
    print()
    print(hdr.replace("MSE", "MAE"))
    print(sep)
    for label, key, name in ARMS:
        vals = [mae[key][s] for s in SEEDS]
        cells = " | ".join(_fmt(v, 4) for v in vals)
        print(f"| {label} {name} | {cells} | {_fmt(mean(vals), 4)} | {_fmt(std(vals), 4)} |")
    print()

    # ---- 2. C vs D within each seed ----
    print("## 3. Raw Expert (C) vs Critic (D) within each seed")
    print()
    print("| seed | Raw Expert C MSE | Critic D MSE | D-C | Critic relative to Raw (%) | verdict |")
    print("|---|---|---|---|---|---|")
    c_key, d_key = "raw_expert", "critic_expert"
    for s in SEEDS:
        c = mse[c_key][s]
        d = mse[d_key][s]
        delta = d - c
        rel = (c - d) / c * 100.0 if c != 0 else 0.0
        verdict = "WORSE" if delta > 0 else ("BETTER" if delta < 0 else "SAME")
        print(f"| {s} | {_fmt(c)} | {_fmt(d)} | {delta:+.2f} | {rel:+.2f}% | {verdict} |")
    print()

    # ---- 3. seed42 relative to the other two seeds ----
    print("## 4. seed42 relative to mean(seed123, seed2026)")
    print()
    print("| arm | seed42 MSE | mean(seed123,seed2026) | ratio |")
    print("|---|---|---|---|")
    ratios = {}
    for label, key, name in ARMS:
        s42 = mse[key][42]
        other = mean([mse[key][123], mse[key][2026]])
        ratio = s42 / other if other != 0 else float("nan")
        ratios[key] = ratio
        print(f"| {label} {name} | {_fmt(s42)} | {_fmt(other)} | {ratio:.3f} |")
    print()

    # ---- 4. z-score / deviation of seed42 from the arm's 3-seed distribution ----
    print("## 5. seed42 deviation from arm mean (|seed42 - mean| / std)")
    print()
    print("| arm | seed42 MSE | arm mean | arm std | |dev|/std |")
    print("|---|---|---|---|---|")
    devs = {}
    for label, key, name in ARMS:
        vals = [mse[key][s] for s in SEEDS]
        m = mean(vals)
        sd = std(vals)
        dev = abs(mse[key][42] - m) / sd if sd != 0 else float("nan")
        devs[key] = dev
        print(f"| {label} {name} | {_fmt(mse[key][42])} | {_fmt(m)} | {_fmt(sd)} | {dev:.3f} |")
    print()

    # ---- 5. is seed42 critic-specific? ----
    # heuristic (NOT a statistical test):
    #   A/B/C ratios "close to 1" = within +-25% of 1
    #   D ratio "clearly > 1" = above 1.5
    NEAR_ONE = 0.25
    D_CLEAR = 1.5
    other_keys = [k for _, k, _ in ARMS if k != d_key]
    near_one = [abs(ratios[k] - 1.0) <= NEAR_ONE for k in other_keys]
    d_outlier = ratios[d_key] > D_CLEAR
    critic_specific = bool(all(near_one) and d_outlier)

    print("## 6. seed42 outlier — specific to Critic?")
    print()
    print(f"  rule: A/B/C ratio within ±{NEAR_ONE:.2f} of 1, D ratio > {D_CLEAR}")
    print(f"  A ratio={ratios['base']:.3f}  B ratio={ratios['shape']:.3f}  "
          f"C ratio={ratios['raw_expert']:.3f}  D ratio={ratios['critic_expert']:.3f}")
    print(f"  seed42_outlier_specific_to_critic = {critic_specific}")
    print()

    # ---- conclusion ----
    print("【结论】")
    print()
    s42_outlier_other = any(ratios[k] > 1.25 for k in other_keys)
    print(f"A. seed42 是否只有 Critic v6 异常？ {'是' if critic_specific else '否'}"
          f"  (D ratio {ratios[d_key]:.2f} vs A/B/C = "
          f"{ratios['base']:.2f}/{ratios['shape']:.2f}/{ratios['raw_expert']:.2f})")
    print(f"B. Raw Expert 在 seed42 是否也异常？ {'是' if abs(ratios['raw_expert'] - 1.0) > NEAR_ONE else '否'}"
          f"  (C ratio {ratios['raw_expert']:.3f}, seed42 C MSE={mse['raw_expert'][42]:.2f} vs "
          f"{mse['raw_expert'][123]:.2f}/{mse['raw_expert'][2026]:.2f})")
    print(f"C. Base / Shape 在 seed42 是否也异常？ "
          f"{'是' if (abs(ratios['base'] - 1.0) > NEAR_ONE or abs(ratios['shape'] - 1.0) > NEAR_ONE) else '否'}"
          f"  (A ratio {ratios['base']:.3f}, B ratio {ratios['shape']:.3f})")
    print(f"D. 下一步应该查 Critic 分支，还是整个训练过程？ "
          f"{'优先查 Critic 分支' if critic_specific else '需要看整个训练过程'}"
          f"  (D 的 seed42 |dev|/std={devs[d_key]:.2f} 为四组最大；"
          f"A/B/C 为 {devs['base']:.2f}/{devs['shape']:.2f}/{devs['raw_expert']:.2f})")
    print()
    print("retrained = false")
    print("critic_modified = false")
    print("network_modified = false")
    print("test_used = false")


if __name__ == "__main__":
    main()
