# -*- coding: utf-8 -*-
"""
Aggregate external-baseline results into the main table.

Reads results/external_baselines/external_baseline_results_full.csv (one row per
(model, dataset, seed), status=OK) and the frozen Ours v2 summary, and writes a
markdown main table with per-dataset mean±std raw-value MSE.

Safe to re-run at any time; incomplete (model,dataset) cells are marked "…".
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "results", "external_baselines")
CSV = os.path.join(RESULTS, "external_baseline_results_full.csv")
OURS_SUMMARY = os.path.join(RESULTS, "..", "reward_feedback_critic_v2_final_test",
                            "final_test_summary.csv")
OUT = os.path.join(RESULTS, "EXTERNAL_BASELINE_MAIN_TABLE.md")

DATASETS = ["etth1", "etth2", "exchange_rate", "traffic", "weather", "solar", "electricity"]
BASELINES = ["PatchTST", "iTransformer", "RAFT", "RAF"]
SEEDS = [42, 123, 2026]


def load_ext():
    rows = defaultdict(lambda: defaultdict(list))  # (model, dataset) -> [mse...]
    if os.path.exists(CSV):
        with open(CSV, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] != "OK" or not r["mse"]:
                    continue
                rows[r["model"]][r["dataset"]].append(float(r["mse"]))
    return rows


def load_ours():
    ours = {}
    if os.path.exists(OURS_SUMMARY):
        with open(OURS_SUMMARY, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ours[r["dataset"]] = float(r["v2_mean_mse"])
    return ours


def fmt(values, full_seeds=len(SEEDS)):
    if not values:
        return "…"
    a = np.array(values)
    mean, std = a.mean(), a.std(ddof=0)
    n = len(values)
    tag = "" if n == full_seeds else f" ({n}/{full_seeds})"
    return f"{mean:.4g}±{std:.3g}{tag}"


def main():
    ext = load_ext()
    ours = load_ours()

    lines = []
    lines.append("# 外部 Baseline 主表（原始值 MSE，mean±std over seeds）\n")
    lines.append("> 状态：**进行中**。Ours v2 为冻结结果（只读，不改动）。"
                 "RAF 采用 num_samples=5（官方默认 20，见 RAF_REPRODUCTION_AUDIT §6）。\n")
    lines.append("| dataset | PatchTST | iTransformer | RAFT | RAF | Ours v2 (frozen) |")
    lines.append("|---|---|---|---|---|---|")
    for ds in DATASETS:
        cells = [ds]
        for b in BASELINES:
            cells.append(fmt(ext[b][ds]))
        cells.append(f"{ours[ds]:.4g}" if ds in ours else "…")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 完成度")
    for b in BASELINES:
        done = sum(1 for ds in DATASETS if len(ext[b][ds]) == len(SEEDS))
        partial = sum(1 for ds in DATASETS if 0 < len(ext[b][ds]) < len(SEEDS))
        lines.append(f"- **{b}**: {done}/{len(DATASETS)} 数据集全 seed 完成"
                     + (f"，{partial} 个部分完成" if partial else ""))

    text = "\n".join(lines) + "\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
