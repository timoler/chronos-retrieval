#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合结果独立重算与最终表生成。

从 preds/<ds>.npz（逐窗口预测缓存）**独立重算**：
  - Chronos-2 零样本 TEST MSE/MAE
  - PAER 专家记忆逐 seed TEST MSE/MAE，并逐 seed 对冻结 result.json 复现核对（相对容差）
  - 逐 seed 在 VAL 网格选 λ（新规则），固定到各自 TEST 融合
  - seed42 共用 λ（旧规则参考）
  - 相对 Chronos-2 改善率 RI = (chronos - fusion)/chronos × 100%

不读 v3 的 fusion_per_seed_results.csv，全部从预测缓存重算，供 Codex 独立核对。
输出：repro_check.csv / val_lambda_search.csv / fusion_summary.csv / 最终 MSE+MAE 表（CSV+MD）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
FROZEN_ROOT = REPO / "results/reward_feedback_critic_v2_final_test"
DEFAULT_PRED_DIR = REPO / "results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011/preds"
OUT_DIR = REPO / "results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011"

SEEDS = [42, 123, 2026]
LAMBDA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
REPRO_TOL_REL = 1e-5  # 相对容差：float32 复现误差 ~1e-7，1e-5 留裕度
DS_ORDER = ["etth1", "etth2", "exchange_rate", "traffic", "weather", "solar", "electricity"]


def mse(y, p):
    return float(((y.astype(np.float64) - p.astype(np.float64)) ** 2).mean())


def mae(y, p):
    return float(np.abs(y.astype(np.float64) - p.astype(np.float64)).mean())


def select_lambda(yc_val, mem_val, Y_val):
    best_lam, best_mse = None, None
    for lam in LAMBDA_GRID:
        fused = (1 - lam) * yc_val + lam * mem_val
        m = mse(Y_val, fused)
        if best_mse is None or m < best_mse:
            best_mse, best_lam = m, lam
    return best_lam, best_mse


def load_frozen_seeds(ds):
    rj = FROZEN_ROOT / ds / "result.json"
    if not rj.exists():
        return None
    fr = json.loads(rj.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in fr["summary"]["v2"]["seeds_mse"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    pred_dir = Path(args.preds)
    out_dir = Path(args.out)

    repro_rows, lambda_rows, summary_rows = [], [], []

    for ds in DS_ORDER:
        npz = pred_dir / f"{ds}.npz"
        if not npz.exists():
            print(f"[{ds}] SKIP: no preds cache", flush=True)
            continue
        data = np.load(npz)
        yc_val = data["yc_val"]; yc_test = data["yc_test"]
        Y_val = data["Y_val"]; Y_test = data["Y_test"]
        frozen = load_frozen_seeds(ds)

        chronos_mse = mse(Y_test, yc_test)
        chronos_mae = mae(Y_test, yc_test)

        # PAER 记忆逐 seed + 复现核对
        mem_test = {}
        for s in SEEDS:
            mem_test[s] = data[f"mem_test_{s}"]
            mem_val_s = data[f"mem_val_{s}"]
            m_test = mse(Y_test, mem_test[s])
            m_val = mse(Y_val, mem_val_s)
            # 复现核对
            if frozen and str(s) in frozen:
                delta = abs(m_test - frozen[str(s)])
                status = "PASS" if delta < REPRO_TOL_REL * max(1.0, abs(frozen[str(s)])) else "FAIL"
            else:
                delta, status = "", "NO_FROZEN"
            repro_rows.append({"dataset": ds, "seed": s,
                               "frozen_mse": f"{frozen[str(s)]:.6f}" if frozen and str(s) in frozen else "",
                               "repro_mse": f"{m_test:.6f}",
                               "delta_abs": f"{delta:.3e}" if isinstance(delta, float) else "",
                               "delta_rel": f"{delta / max(1.0, abs(frozen[str(s)])):.3e}" if isinstance(delta, float) else "",
                               "status": status})
            # λ 网格搜索表（VAL）
            for lam in LAMBDA_GRID:
                fused_val = (1 - lam) * yc_val + lam * mem_val_s
                lambda_rows.append({"dataset": ds, "seed": s, "lambda": lam,
                                    "val_mse": f"{mse(Y_val, fused_val):.6g}"})

        # 逐 seed λ（新规则）
        lam_per_seed, fused_test = {}, {}
        for s in SEEDS:
            lam_per_seed[s], _ = select_lambda(yc_val, data[f"mem_val_{s}"], Y_val)
            fused_test[s] = mse(Y_test, (1 - lam_per_seed[s]) * yc_test + lam_per_seed[s] * mem_test[s])

        # seed42 共用 λ（旧规则参考）
        lam_seed42, _ = select_lambda(yc_val, data["mem_val_42"], Y_val)
        fused_seed42 = [mse(Y_test, (1 - lam_seed42) * yc_test + lam_seed42 * mem_test[s]) for s in SEEDS]

        fusion_mean = float(np.mean([fused_test[s] for s in SEEDS]))
        fusion_std = float(np.std([fused_test[s] for s in SEEDS]))
        fusion_seed42_mean = float(np.mean(fused_seed42))

        # PAER standalone（记忆单独，3 seed 均值 = 冻结 v2 均值）
        paer_standalone_mean = float(np.mean([mse(Y_test, mem_test[s]) for s in SEEDS]))

        ri_new = (chronos_mse - fusion_mean) / chronos_mse * 100.0
        ri_seed42 = (chronos_mse - fusion_seed42_mean) / chronos_mse * 100.0

        summary_rows.append({
            "dataset": ds,
            "chronos2_mse": f"{chronos_mse:.6f}",
            "chronos2_mae": f"{chronos_mae:.6f}",
            "paer_standalone_mse": f"{paer_standalone_mean:.6f}",
            "fusion_new_mse_mean": f"{fusion_mean:.6f}",
            "fusion_new_mse_std": f"{fusion_std:.6f}",
            "fusion_new_mse_42": f"{fused_test[42]:.6f}",
            "fusion_new_mse_123": f"{fused_test[123]:.6f}",
            "fusion_new_mse_2026": f"{fused_test[2026]:.6f}",
            "lam_42": f"{lam_per_seed[42]:.3f}", "lam_123": f"{lam_per_seed[123]:.3f}",
            "lam_2026": f"{lam_per_seed[2026]:.3f}",
            "fusion_seed42_mse_mean": f"{fusion_seed42_mean:.6f}",
            "seed42_lam": f"{lam_seed42:.3f}",
            "ri_new_pct": f"{ri_new:.2f}",
            "ri_seed42_pct": f"{ri_seed42:.2f}",
        })
        print(f"[{ds}] chronos={chronos_mse:.4g} fusion_new={fusion_mean:.4g}±{fusion_std:.3g} "
              f"ri_new={ri_new:+.2f}% λ={[lam_per_seed[s] for s in SEEDS]}", flush=True)

    # 写 repro_check.csv
    with open(out_dir / "repro_check.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "frozen_mse", "repro_mse", "delta_abs", "delta_rel", "status"])
        w.writeheader(); w.writerows(repro_rows)

    # 写 val_lambda_search.csv
    with open(out_dir / "val_lambda_search.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "lambda", "val_mse"])
        w.writeheader(); w.writerows(lambda_rows)

    # 写 fusion_summary.csv
    fields = ["dataset", "chronos2_mse", "chronos2_mae", "paer_standalone_mse",
              "fusion_new_mse_mean", "fusion_new_mse_std",
              "fusion_new_mse_42", "fusion_new_mse_123", "fusion_new_mse_2026",
              "lam_42", "lam_123", "lam_2026", "fusion_seed42_mse_mean", "seed42_lam",
              "ri_new_pct", "ri_seed42_pct"]
    with open(out_dir / "fusion_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(summary_rows)

    # 最终 MSE 表（Markdown）
    md = ["| Dataset | Chronos-2 | PAER standalone | 融合模型(新规则) | RI vs Chronos-2 | λ(42/123/2026) | 复现 |",
          "|---|---|---|---|---|---|---|"]
    for r in summary_rows:
        repro = {x["seed"]: x["status"] for x in repro_rows if x["dataset"] == r["dataset"]}
        repro_s = "/".join(repro.get(str(s), "?") for s in SEEDS)
        md.append(f"| {r['dataset']} | {r['chronos2_mse']} | {r['paer_standalone_mse']} | "
                  f"{r['fusion_new_mse_mean']}±{r['fusion_new_mse_std']} | {r['ri_new_pct']}% | "
                  f"{r['lam_42']}/{r['lam_123']}/{r['lam_2026']} | {repro_s} |")
    (out_dir / "fusion_final_table_md.txt").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDone. outputs -> {out_dir}")

    fails = [r for r in repro_rows if r["status"] == "FAIL"]
    n_new_win = sum(1 for r in summary_rows if float(r["ri_new_pct"]) > 0)
    print(f"repro: total={len(repro_rows)} FAIL={len(fails)}")
    print(f"fusion(new rule) 正改善数据集数: {n_new_win}/{len(summary_rows)}")
    for r in summary_rows:
        print(f"  {r['dataset']}: ri_new={r['ri_new_pct']}%  ri_seed42={r['ri_seed42_pct']}%")


if __name__ == "__main__":
    main()
