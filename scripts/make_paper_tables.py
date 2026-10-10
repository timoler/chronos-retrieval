#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从实际结果 CSV 生成论文对比表 PNG（表2 端到端 + 外部 baseline 主表）。

只读结果 CSV，不重新计算实验；数字与 results/ 下已落盘的结果一致。
输出：figures/table_*.png（仓库根目录 figures/，纳入版本控制）。
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

FINAL_TEST = REPO / "results/reward_feedback_critic_v2_final_test/final_test_summary.csv"
EXT_CSV = REPO / "results/external_baselines/external_baseline_results_full.csv"
FUSION_CSV = REPO / "results/paper_completion/phase1_external_baselines/ours_fusion_v2_results.csv"

DATASETS = ["etth1", "etth2", "exchange_rate", "traffic", "weather", "solar", "electricity"]


def fmt(x, nd=4):
    """数值格式：大数/小数用科学计数法，其余 4 位有效数字。"""
    if x is None:
        return "—"
    x = float(x)
    ax = abs(x)
    if ax != 0 and (ax >= 1e6 or ax < 1e-3):
        return f"{x:.{nd-1}g}"
    return f"{x:.{nd}g}"


def fmt_msd(mean, std, nd=4):
    if mean is None:
        return "—"
    return f"{fmt(mean, nd)}±{fmt(std, nd)}"


def _render_table(title, header, rows, out_path, col_width=None, fontsize=11):
    """用 matplotlib 渲染一张表格 PNG。"""
    n = len(rows)
    ncols = len(header)
    fig, ax = plt.subplots(figsize=(max(6, ncols * 1.6), max(2, n * 0.55 + 1.4)))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12, weight="bold")

    table = ax.table(
        cellText=rows, colLabels=header, loc="center", cellLoc="center",
        colWidths=col_width,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#e8eef7")
            cell.set_text_props(weight="bold")

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# 表 A：冻结 v2 最终 TEST（端到端，7 数据集）
# --------------------------------------------------------------------------- #
def table_frozen_v2():
    header = ["Dataset", "Raw", "v1", "v2", "v2 vs Raw", "v2 vs v1", "wins Raw"]
    rows = []
    order = {d: i for i, d in enumerate(DATASETS)}
    rec = {}
    with FINAL_TEST.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rec[r["dataset"]] = r
    for ds in DATASETS:
        r = rec[ds]
        raw, v1, v2 = float(r["raw_mean_mse"]), float(r["v1_mean_mse"]), float(r["v2_mean_mse"])
        pr, pv = float(r["v2_vs_raw_pct"]), float(r["v2_vs_v1_pct"])
        wins = f"{int(float(r['wins_vs_raw_seeds']))}/3"
        rows.append([ds, fmt(raw), fmt(v1), fmt(v2),
                     f"{pr:+.1f}%", f"{pv:+.1f}%", wins])
    _render_table("Table 2 (end-to-end): Frozen Pool-Aware v2 final TEST — MSE (3-seed mean)",
                  header, rows, OUT / "table_frozen_v2_final_test.png")
    return rows


# --------------------------------------------------------------------------- #
# 表 B：外部 baseline 主表（7 数据集 × 6 模型）
# --------------------------------------------------------------------------- #
def load_ext():
    rows = defaultdict(lambda: defaultdict(list))
    with EXT_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "OK" and r["mse"]:
                rows[r["model"]][r["dataset"]].append(float(r["mse"]))
    return rows


def load_fusion():
    chronos, ours = {}, {}
    if FUSION_CSV.exists():
        with FUSION_CSV.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ds = r["dataset"]
                chronos[ds] = float(r["chronos_raw_test_mse"])
                ours[ds] = (float(r["fusion_test_mse_mean"]), float(r["fusion_test_mse_std"]))
    return chronos, ours


def table_external():
    ext = load_ext()
    chronos, ours = load_fusion()
    header = ["Dataset", "Chronos-2", "PatchTST", "iTransformer", "RAFT", "RAF", "Ours"]
    rows = []

    def multi(model, ds, need=3):
        vals = ext[model][ds]
        if not vals:
            return "—"
        a = np.array(vals)
        if len(vals) == 1:
            return f"{fmt(a[0])} (1s)"
        tag = "" if len(vals) == need else f" ({len(vals)}/{need}s)"
        return f"{fmt(a.mean())}±{fmt(a.std(ddof=0))}{tag}"

    for ds in DATASETS:
        c = fmt(chronos[ds]) if ds in chronos else "—"
        o = fmt_msd(ours[ds][0], ours[ds][1]) if ds in ours else "—"
        rows.append([ds, c, multi("PatchTST", ds), multi("iTransformer", ds),
                     multi("RAFT", ds), multi("RAF", ds, need=1), o])
    _render_table("External baseline main table — MSE (lower is better)",
                  header, rows, OUT / "table_external_baselines.png")
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r1 = table_frozen_v2()
    print("\n=== Table A: frozen v2 final TEST ===")
    print("\t".join(["dataset", "raw", "v1", "v2", "vs_raw", "vs_v1", "wins"]))
    for row in r1:
        print("\t".join(row))
    r2 = table_external()
    print("\n=== Table B: external baselines ===")
    print("\t".join(["dataset", "chronos2", "patchtst", "itransformer", "raft", "raf", "ours"]))
    for row in r2:
        print("\t".join(row))
