#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐 seed 复现检查：fusion v3 落盘的 v2 记忆预测 vs 冻结 final_test result.json 的 seeds_mse。

严格同 dataset、同 seed、同窗口、同指标比较。输出 repro_check.csv。
用法：
  python check_frozen_paer_reproduction.py [--preds <dir>] [--out <csv>]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_TEST_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_final_test"
DEFAULT_PREDS = REPO_ROOT / "results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011/preds"
REPRO_TOL_REL = 1e-5  # 相对容差：float32 预测下 MSE 复现误差 ~1e-7，1e-5 留足裕度且不吞真实错配


def mse(y_true, y_pred):
    return float(((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(DEFAULT_PREDS))
    ap.add_argument("--out", default=str(Path(DEFAULT_PREDS).parent / "repro_check.csv"))
    args = ap.parse_args()

    pred_dir = Path(args.preds)
    rows = []
    for npz in sorted(pred_dir.glob("*.npz")):
        ds = npz.stem
        rj = FROZEN_TEST_ROOT / ds / "result.json"
        if not rj.exists():
            rows.append({"dataset": ds, "seed": "", "frozen_mse": "", "repro_mse": "",
                         "delta": "", "status": "NO_FROZEN_JSON"})
            continue
        frozen = json.loads(rj.read_text(encoding="utf-8"))["summary"]["v2"]["seeds_mse"]
        data = np.load(npz)
        Y_test = data["Y_test"]
        for seed in ("42", "123", "2026"):
            mem = data[f"mem_test_{seed}"]
            repro = mse(Y_test, mem)
            fr = frozen.get(seed)
            if fr is None:
                status, delta = "NO_FROZEN_SEED", ""
            else:
                delta = abs(repro - float(fr))
                status = "PASS" if delta < REPRO_TOL_REL * max(1.0, abs(float(fr))) else "FAIL"
            rows.append({
                "dataset": ds, "seed": seed,
                "frozen_mse": f"{fr:.6f}" if fr is not None else "",
                "repro_mse": f"{repro:.6f}",
                "delta": f"{delta:.3e}" if isinstance(delta, float) else "",
                "status": status,
            })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "frozen_mse", "repro_mse", "delta", "status"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved -> {args.out}")
    fails = [r for r in rows if r["status"] == "FAIL"]
    print(f"total={len(rows)} FAIL={len(fails)}")
    for r in fails:
        print("  FAIL:", r)


if __name__ == "__main__":
    main()
