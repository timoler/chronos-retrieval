#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chronos-2 + Frozen Pool-Aware v2 融合 —— per-seed λ 协议修订版（v3）。

相对 run_ours_fusion_v2.py 的三处修正（协议修订，不是旧实现"算错"）：
  1. **逐 seed 在 VAL 独立选 λ**：每个 seed s 用各自的 VAL 记忆 mem_val[s] 网格选 λ_s，
     再固定到该 seed 的 TEST。旧版用 seed42 单 λ 套全部 seed。
  2. **逐 seed 复现检查**：mem_test[s] 对冻结 result.json 的 seeds_mse[s]（同 seed 对同 seed），
     旧版误拿 seed42 单值比 3 种子均值。
  3. **落盘逐窗口预测缓存**（preds/<ds>.npz），供后续复用与复现审计，不再只写汇总 CSV。

同时**保留 seed42 共用 λ（旧协议）**作为参考列 fusion_seed42lam_*，不用于选择、也不据 TEST 挑协议。

不改动：Chronos-2 零样本、冻结 v2 专家池（final_expert_pool.json）、train_arm/retrieve_conditional、
λ 网格、正式 split / window_indices.npz。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # experiments/
sys.path.insert(0, str(HERE.parent.parent))   # repo root

import run_ours_fusion_v2 as v2  # noqa: E402  复用 load pools / run_chronos / run_v2_memory_pred / _init_state

REPO_ROOT = v2.REPO_ROOT
FROZEN_TEST_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_final_test"
OUT_DIR = REPO_ROOT / "results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011"
PRED_DIR = OUT_DIR / "preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = v2.SEEDS                     # [42, 123, 2026]
LAMBDA_GRID = v2.LAMBDA_GRID
REPRO_TOL_REL = 1e-5               # 相对容差（float32 预测下 MSE 复现误差 ~1e-7，1e-5 留裕度不吞真实错配）


def load_frozen_per_seed(ds_name):
    """冻结 final_test 的 v2 逐 seed MSE：dict[str seed -> float]；缺失返回 None。"""
    rj = FROZEN_TEST_ROOT / ds_name / "result.json"
    if not rj.exists():
        return None
    fr = json.loads(rj.read_text(encoding="utf-8"))
    v2_summary = fr.get("summary", {}).get("v2", {})
    seeds = v2_summary.get("seeds_mse")
    if not seeds:
        return None
    return {k: float(v) for k, v in seeds.items()}


def select_lambda(yc_val, mem_val, Y_val):
    """在 VAL 上网格选 λ（fused=(1-λ)*chronos + λ*mem）。返回 (best_lam, best_val_mse)。"""
    best_lam, best_mse = None, None
    for lam in LAMBDA_GRID:
        fused = (1 - lam) * yc_val + lam * mem_val
        m = float(((fused - Y_val) ** 2).mean())
        if best_mse is None or m < best_mse:
            best_mse, best_lam = m, lam
    return best_lam, best_mse


def compute(ds_name, pipeline, device):
    ds_dir = REPO_ROOT / "results/multidataset" / ds_name
    cfg = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg["context"])
    horizon = int(cfg["horizon"])
    v2.v1.CONTEXT = context
    v2.v1.HORIZON = horizon

    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    test_starts = np.asarray(idx["test"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    X_bank, Y_bank = v2.build_windows(series, bank_starts, context, horizon)
    X_fit, Y_fit = v2.build_windows(series, fit_starts, context, horizon)
    X_val, Y_val = v2.build_windows(series, val_starts, context, horizon)
    X_test, Y_test = v2.build_windows(series, test_starts, context, horizon)
    X_bank_res = v2.residualize(X_bank)
    X_fit_res = v2.residualize(X_fit)
    X_val_res = v2.residualize(X_val)
    X_test_res = v2.residualize(X_test)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    val_ts = [dates[s + context] for s in val_starts]
    test_ts = [dates[s + context] for s in test_starts]
    stats = v2.compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    # 未来信息泄漏断言
    max_mem_end = int(max(bank_starts.max(), fit_starts.max()) + context + horizon)
    min_val_start = int(val_starts.min())
    min_test_start = int(test_starts.min())
    assert max_mem_end <= min_val_start, f"[{ds_name}] LEAK: memory bank future into val"
    assert int(val_starts.max()) + context + horizon <= min_test_start, f"[{ds_name}] LEAK: val into test"

    raw_pool = v2.load_raw_pool(ds_name)
    v1_pool = v2.load_v1_pool(ds_name)
    v2_pool = v2.load_v2_pool(ds_name)
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))
    emb_v2_orig = v2.v1.embed_claims([e["claim"] for e in v2_pool], device)
    expert_dim = emb_v2_orig.shape[1]
    act_val = v2.v1.act_mats(v2_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    act_test = v2.v1.act_mats(v2_pool, X_bank, X_fit, X_test, bank_ts, fit_ts, test_ts, stats)
    v2_act_val = {"bank": act_val["bank"], "fit": act_val["fit"], "val": act_val["val"]}
    v2_act_test = {"bank": act_test["bank"], "fit": act_test["fit"], "test": act_test["val"]}
    emb_v2, v2_act_val, _ = v2.v1.pad_to_slots(emb_v2_orig, v2_act_val, n_slots)
    _, v2_act_test, _ = v2.v1.pad_to_slots(emb_v2_orig, v2_act_test, n_slots)

    # Chronos-2 零样本（确定性，seed 无关）
    print(f"  [Chronos-2] val={len(val_starts)} test={len(test_starts)} ...", flush=True)
    yc_val = v2.run_chronos(pipeline, list(X_val), horizon)
    yc_test = v2.run_chronos(pipeline, list(X_test), horizon)

    # v2 记忆预测（逐 seed）
    mem_val, mem_test = {}, {}
    for seed in SEEDS:
        init_state = v2._init_state(seed, context, expert_dim, n_slots, device)
        pv = v2.run_v2_memory_pred(init_state, emb_v2, v2_act_val, True, expert_dim, n_slots,
                                   device, seed, context, X_bank, Y_bank, X_fit, Y_fit,
                                   X_bank_res, X_fit_res, X_val, X_val_res, v2_act_val["val"])
        pt = v2.run_v2_memory_pred(init_state, emb_v2, v2_act_test, True, expert_dim, n_slots,
                                   device, seed, context, X_bank, Y_bank, X_fit, Y_fit,
                                   X_bank_res, X_fit_res, X_test, X_test_res, v2_act_test["test"])
        mem_val[seed] = pv.astype(np.float64)
        mem_test[seed] = pt.astype(np.float64)
        print(f"    seed={seed} v2_mem TEST_MSE={v2.mse(Y_test, pt):.6g} VAL_MSE={v2.mse(Y_val, pv):.6g}", flush=True)

    # 落盘逐窗口预测缓存
    np.savez_compressed(
        PRED_DIR / f"{ds_name}.npz",
        yc_val=yc_val, yc_test=yc_test,
        **{f"mem_val_{s}": mem_val[s] for s in SEEDS},
        **{f"mem_test_{s}": mem_test[s] for s in SEEDS},
        Y_val=Y_val, Y_test=Y_test,
        val_starts=val_starts, test_starts=test_starts,
        context=np.array(context), horizon=np.array(horizon),
    )

    # ---- per-seed λ（协议修订）----
    lam_per_seed, val_mse_per_seed = {}, {}
    for seed in SEEDS:
        lam, vm = select_lambda(yc_val, mem_val[seed], Y_val)
        lam_per_seed[seed] = lam
        val_mse_per_seed[seed] = vm
        print(f"    seed={seed} per-seed λ={lam} (VAL MSE={vm:.6g})", flush=True)

    # ---- seed42 共用 λ（旧协议参考，不用于选择）----
    lam_seed42, val_mse_seed42 = select_lambda(yc_val, mem_val[42], Y_val)

    chronos_raw_mse = float(((yc_test - Y_test) ** 2).mean())

    # per-seed 融合 TEST
    fused_test = {}
    for seed in SEEDS:
        f = (1 - lam_per_seed[seed]) * yc_test + lam_per_seed[seed] * mem_test[seed]
        fused_test[seed] = float(((f - Y_test) ** 2).mean())
    fusion_mean = float(np.mean([fused_test[s] for s in SEEDS]))
    fusion_std = float(np.std([fused_test[s] for s in SEEDS]))

    # seed42 共用 λ 融合（旧参考）
    fused_seed42 = {}
    for seed in SEEDS:
        f = (1 - lam_seed42) * yc_test + lam_seed42 * mem_test[seed]
        fused_seed42[seed] = float(((f - Y_test) ** 2).mean())
    fusion_mean_seed42 = float(np.mean([fused_seed42[s] for s in SEEDS]))

    # 逐 seed 复现检查（vs 冻结 seeds_mse）
    frozen_seeds = load_frozen_per_seed(ds_name)
    repro = {}
    for seed in SEEDS:
        m = float(v2.mse(Y_test, mem_test[seed]))
        if frozen_seeds and str(seed) in frozen_seeds:
            delta = abs(m - frozen_seeds[str(seed)])
            repro[seed] = ("OK" if delta < REPRO_TOL_REL * max(1.0, abs(frozen_seeds[str(seed)])) else "FAIL",
                           delta, frozen_seeds[str(seed)])
        else:
            repro[seed] = ("NO_FROZEN", None, None)

    row = {
        "dataset": ds_name,
        "n_val": len(val_starts), "n_test": len(test_starts),
        "chronos_raw_test_mse": f"{chronos_raw_mse:.6f}",
        # per-seed λ（新协议）
        "lam_42": f"{lam_per_seed[42]:.3f}",
        "lam_123": f"{lam_per_seed[123]:.3f}",
        "lam_2026": f"{lam_per_seed[2026]:.3f}",
        # per-seed 融合 TEST
        "fusion_test_mse_mean": f"{fusion_mean:.6f}",
        "fusion_test_mse_std": f"{fusion_std:.6f}",
        "fusion_test_mse_seed42": f"{fused_test[42]:.6f}",
        "fusion_test_mse_seed123": f"{fused_test[123]:.6f}",
        "fusion_test_mse_seed2026": f"{fused_test[2026]:.6f}",
        # seed42 共用 λ（旧协议参考）
        "seed42_lam": f"{lam_seed42:.3f}",
        "fusion_seed42lam_mean": f"{fusion_mean_seed42:.6f}",
        # v2 记忆逐 seed TEST MSE（复现用）
        "v2_mem_test_mse_42": f"{v2.mse(Y_test, mem_test[42]):.6f}",
        "v2_mem_test_mse_123": f"{v2.mse(Y_test, mem_test[123]):.6f}",
        "v2_mem_test_mse_2026": f"{v2.mse(Y_test, mem_test[2026]):.6f}",
        # 复现状态
        "repro_42": repro[42][0], "repro_123": repro[123][0], "repro_2026": repro[2026][0],
    }
    return row


FIELDS = ["dataset", "n_val", "n_test", "chronos_raw_test_mse",
          "lam_42", "lam_123", "lam_2026",
          "fusion_test_mse_mean", "fusion_test_mse_std",
          "fusion_test_mse_seed42", "fusion_test_mse_seed123", "fusion_test_mse_seed2026",
          "seed42_lam", "fusion_seed42lam_mean",
          "v2_mem_test_mse_42", "v2_mem_test_mse_123", "v2_mem_test_mse_2026",
          "repro_42", "repro_123", "repro_2026"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[d["name"] for d in v2.DATASETS])
    ap.add_argument("--out", default=str(OUT_DIR / "fusion_per_seed_results.csv"))
    ap.add_argument("--resume", action="store_true",
                    help="skip datasets whose preds npz already exists (reuse prediction cache)")
    args = ap.parse_args()

    names = [d["name"] for d in v2.DATASETS if d["name"] in args.datasets]
    if args.resume:
        names = [n for n in names if not (PRED_DIR / f"{n}.npz").exists()]

    v2.set_seed(42)
    from chronos import Chronos2Pipeline
    print(f"Loading {v2.MODEL} ...", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained(v2.MODEL, device_map="cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exists = (args.out and Path(args.out).exists())
    for name in names:
        print(f"\n{'=' * 78}\nDATASET {name}\n{'=' * 78}", flush=True)
        try:
            row = compute(name, pipeline, device)
            with open(args.out, "a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if not exists:
                    w.writeheader()
                    exists = True
                w.writerow(row)
            print(f"[{name}] chronos={row['chronos_raw_test_mse']} "
                  f"per-seed-fusion={row['fusion_test_mse_mean']}±{row['fusion_test_mse_std']} "
                  f"λ=({row['lam_42']},{row['lam_123']},{row['lam_2026']}) "
                  f"repro=({row['repro_42']},{row['repro_123']},{row['repro_2026']})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {e!r}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nDone. Results ->", args.out)


if __name__ == "__main__":
    main()
