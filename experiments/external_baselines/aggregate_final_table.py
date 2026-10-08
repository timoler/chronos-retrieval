# -*- coding: utf-8 -*-
"""生成论文最终外部模型主表（EXTERNAL_BASELINE_FINAL_TABLE.md）。

输入：
  - results/external_baselines/external_baseline_results_full.csv  (PatchTST/iTransformer/RAFT/RAF)
  - results/paper_completion/phase1_external_baselines/ours_fusion_v2_results.csv  (Chronos-2 + Ours Fusion)

输出：results/paper_completion/phase1_external_baselines/EXTERNAL_BASELINE_FINAL_TABLE.md
     + ours_fusion_summary.csv（Ours vs Chronos-2 逐集 RI + 汇总）

口径：原始值 MSE，越低越好。3-seed 报 mean±std；RAF(1 seed)/Chronos-2(确定性) 报单值并注明。
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
EXT_CSV = REPO / "results/external_baselines/external_baseline_results_full.csv"
FUSION_CSV = REPO / "results/paper_completion/phase1_external_baselines/ours_fusion_v2_results.csv"
OUT_MD = REPO / "results/paper_completion/phase1_external_baselines/EXTERNAL_BASELINE_FINAL_TABLE.md"
OUT_SUMMARY = REPO / "results/paper_completion/phase1_external_baselines/ours_fusion_summary.csv"

DATASETS = ["etth1", "etth2", "exchange_rate", "traffic", "weather", "solar", "electricity"]
BASELINES = ["PatchTST", "iTransformer", "RAFT", "RAF"]
N_SEEDS = 3


def load_ext():
    rows = defaultdict(lambda: defaultdict(list))
    if EXT_CSV.exists():
        with EXT_CSV.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] == "OK" and r["mse"]:
                    rows[r["model"]][r["dataset"]].append(float(r["mse"]))
    return rows


def load_fusion():
    chronos, ours_mean, ours_std, ours_seeds = {}, {}, {}, {}
    if FUSION_CSV.exists():
        with FUSION_CSV.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                ds = r["dataset"]
                chronos[ds] = float(r["chronos_raw_test_mse"])
                ours_mean[ds] = float(r["fusion_test_mse_mean"])
                ours_std[ds] = float(r["fusion_test_mse_std"])
                ours_seeds[ds] = {s: float(r[f"fusion_test_mse_seed{s}"]) for s in (42, 123, 2026)}
    return chronos, ours_mean, ours_std, ours_seeds


def fmt_multi(vals, need=N_SEEDS):
    if not vals:
        return "N/A"
    a = np.array(vals)
    if len(vals) == 1:
        return f"{a[0]:.4g} (1 seed)"
    tag = "" if len(vals) == need else f" ({len(vals)}/{need})"
    return f"{a.mean():.4g}±{a.std(ddof=0):.3g}{tag}"


def main():
    ext = load_ext()
    chronos, ours_mean, ours_std, ours_seeds = load_fusion()

    # 逐数据集 Ours vs Chronos-2 RI
    ri_rows = []
    for ds in DATASETS:
        if ds in chronos and ds in ours_mean:
            c = chronos[ds]
            o = ours_mean[ds]
            ri = (c - o) / c * 100.0
            ri_rows.append({"dataset": ds, "chronos2_mse": c, "ours_mse": o,
                            "ours_std": ours_std[ds], "ri_pct": ri})
    mean_ri = float(np.mean([r["ri_pct"] for r in ri_rows])) if ri_rows else None
    med_ri = float(np.median([r["ri_pct"] for r in ri_rows])) if ri_rows else None
    n_win = sum(1 for r in ri_rows if r["ri_pct"] > 0)

    L = []
    L.append("# 外部模型最终主表（原始值 MSE，越低越好）\n")
    L.append("> 生成：2026-10-09。Ours = Chronos-2 + Frozen Pool-Aware v2 Expert Memory/Fusion。")
    L.append("> 口径：统一 split/context/horizon/target/原始值 MSE。")
    L.append("> 种子说明：PatchTST/iTransformer/RAFT/Ours 为 3-seed mean±std；")
    L.append("> RAF 无训练参数（seed 只影响 Chronos 采样，num_samples=5，1 seed）；")
    L.append("> Chronos-2 为零样本确定性（seed 无关，单值）。\n")

    L.append("| Dataset | Chronos-2 | PatchTST | iTransformer | RAFT | RAF | Ours |")
    L.append("|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        c = f"{chronos[ds]:.4g}" if ds in chronos else "N/A"
        cells = [ds, c]
        for b in BASELINES:
            cells.append(fmt_multi(ext[b][ds]))
        o = f"{ours_mean[ds]:.4g}±{ours_std[ds]:.3g}" if ds in ours_mean else "N/A"
        cells.append(o)
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # 完成度
    L.append("## 完成度\n")
    for b in BASELINES:
        done = sum(1 for ds in DATASETS if len(ext[b][ds]) >= (1 if b == "RAF" else N_SEEDS))
        L.append(f"- **{b}**: {done}/{len(DATASETS)} 数据集完成")
    done_chronos = sum(1 for ds in DATASETS if ds in chronos)
    done_ours = sum(1 for ds in DATASETS if ds in ours_mean)
    L.append(f"- **Chronos-2**: {done_chronos}/{len(DATASETS)}")
    L.append(f"- **Ours (Chronos-2 + Pool-Aware v2)**: {done_ours}/{len(DATASETS)}\n")

    # A. Ours vs Chronos-2
    L.append("## A. Ours vs Chronos-2（RI = (MSE_Chronos2 − MSE_Ours) / MSE_Chronos2）\n")
    L.append("| Dataset | Chronos-2 | Ours | RI |")
    L.append("|---|---|---|---|")
    for r in ri_rows:
        L.append(f"| {r['dataset']} | {r['chronos2_mse']:.4g} | {r['ours_mse']:.4g} | {r['ri_pct']:+.1f}% |")
    L.append("")
    if ri_rows:
        L.append(f"- 胜/负：**{n_win}/{len(ri_rows)}** 数据集 RI > 0")
        L.append(f"- mean RI = **{mean_ri:+.2f}%**；median RI = **{med_ri:+.2f}%**\n")

    # B. Ours vs External baselines
    L.append("## B. Ours vs 外部 baseline（胜/负，Ours MSE < baseline MSE 记胜）\n")
    for b in BASELINES:
        wins = 0
        rows = 0
        for ds in DATASETS:
            if ds in ours_mean and ext[b][ds]:
                rows += 1
                if ours_mean[ds] < np.mean(ext[b][ds]):
                    wins += 1
        L.append(f"- **vs {b}**: {wins}/{rows} 胜")
    L.append("")

    # C. 结果解释
    L.append("## C. 结果解释\n")
    L.append("- **机制是否有效**：由冻结 CNN-only v2 内部消融回答（v2 vs Raw 5/7、mean RI +13.4%，见 `results/reward_feedback_critic_v2_final_test/`）。")
    L.append("- **最终预测系统是否有竞争力**：由本表 Chronos-2 + Ours Fusion 与外部 baseline 的统一实验回答。")
    L.append("- 两者是不同结论，不混用。\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    # summary csv
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "chronos2_mse", "ours_mse", "ours_std", "ri_pct"])
        w.writeheader()
        for r in ri_rows:
            w.writerow(r)

    print("\n".join(L))
    print(f"\nSaved -> {OUT_MD}")
    print(f"Saved -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
