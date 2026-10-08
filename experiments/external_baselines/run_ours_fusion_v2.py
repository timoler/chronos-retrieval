#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chronos-2 + Frozen Pool-Aware v2 Expert Memory/Fusion（正式协议）。

最终「Ours Fusion」= Chronos-2 预测 ⊕ 冻结 Pool-Aware v2 专家记忆（
ConditionalExpertMLPRetriever 在 v2 专家池下的条件检索输出），按仓库既定
凸融合公式 fused = (1-λ)·chronos + λ·v2_memory，λ 在 VAL 网格选（TEST 只评一次）。

忠实复现冻结 v2（与 run_reward_feedback_critic_v2_final_test.py 同构）：
  - 读冻结 v2 pool（final_expert_pool.json），n_slots = max(raw,v1,v2)=8；
  - use_shape=True（mode="full"），seed_everything(seed)，同一 train_arm/retrieve_conditional；
  - 只新增：VAL 窗口/激活 + 捕获 v2 记忆预测数组 + Chronos-2 预测 + λ 融合。

不改动 v2 算法 / 专家选择 / Reward 定义。Chronos-2 是零样本 backbone（确定性，seed 无关）。

输出：results/paper_completion/phase1_external_baselines/ours_fusion_v2_*.{csv,json}
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # experiments/
sys.path.insert(0, str(HERE.parent.parent))   # repo root

from run_expert_embedding_mlp import build_windows, mse, mae  # noqa: E402
from run_solar_same_val_baselines import residualize            # noqa: E402
from run_conditional_expert_mlp_v4 import (                     # noqa: E402
    ConditionalExpertMLPRetriever, retrieve_conditional)
from run_etth1_prediction_validation import compute_training_stats  # noqa: E402

import run_etth1_reward_feedback_critic_v1 as v1              # noqa: E402
import run_reward_feedback_critic_v2_poolaware_pilot as v2p    # noqa: E402

REPO_ROOT = HERE.parent.parent
FROZEN_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_full_cross_dataset"
V1_ETTH1 = v2p.V1_ETTH1
V1_CROSS = v2p.V1_CROSS
FROZEN_TEST_SUMMARY = REPO_ROOT / "results/reward_feedback_critic_v2_final_test/final_test_summary.csv"
OUT_DIR = REPO_ROOT / "results/paper_completion/phase1_external_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = v2p.SEEDS                                   # [42, 123, 2026]
TOPK = v2p.TOPK
N_RAW_TARGET = v2p.N_RAW_TARGET
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

MODEL = "amazon/chronos-2"
BATCH = 256
# 凸融合 λ 网格：继承 run_ours_fusion.py 的既定值并扩展到 1.0（v2 记忆可能足够强需大 λ）。
# 仅在 VAL 上选，TEST 只用一次。
LAMBDA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

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


# --------------------------------------------------------------------------- #
# 冻结 pool 读取（与 final_test 完全一致，只读）
# --------------------------------------------------------------------------- #
def load_raw_pool(name):
    src = FROZEN_ROOT / name / "raw_experts.json"
    if src.exists():
        return json.loads(src.read_text(encoding="utf-8"))
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
        raise ArtifactMissingError(f"missing frozen v2 final pool: {src}")
    return json.loads(src.read_text(encoding="utf-8"))


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_chronos(pipeline, contexts, horizon):
    """Chronos-2 零样本 mean 点预测（确定性，seed 无关）。"""
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=horizon, batch_size=BATCH)
    return np.stack([m.squeeze(0).numpy() for m in means]).astype(np.float32)


# --------------------------------------------------------------------------- #
# v2 记忆预测（train_arm + retrieve_conditional，返回 pred 数组而非 MSE）
# --------------------------------------------------------------------------- #
def run_v2_memory_pred(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed, context,
                       X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                       X_q, X_q_res, act_q):
    v2p.seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v2p.train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      act["bank"], act["fit"], use_shape, device, seed)
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(mdl, X_q, X_bank, Y_bank, X_q_res, X_bank_res,
                                act_q, act["bank"], mode, TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred


def _init_state(seed, context, expert_dim, n_slots, device):
    v2p.seed_everything(seed)
    m = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


# --------------------------------------------------------------------------- #
# 单数据集计算
# --------------------------------------------------------------------------- #
def compute(ds_name, pipeline, device):
    ds_dir = REPO_ROOT / "results/multidataset" / ds_name
    cfg = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg["context"])
    horizon = int(cfg["horizon"])
    v1.CONTEXT = context
    v1.HORIZON = horizon

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

    X_bank, Y_bank = build_windows(series, bank_starts, context, horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, context, horizon)
    X_val, Y_val = build_windows(series, val_starts, context, horizon)
    X_test, Y_test = build_windows(series, test_starts, context, horizon)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)
    X_test_res = residualize(X_test)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    val_ts = [dates[s + context] for s in val_starts]
    test_ts = [dates[s + context] for s in test_starts]
    stats = compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    # ---- 未来信息泄漏断言（显式，禁止 memory bank 的未来进入 val/test 预测区）----
    max_mem_end = int(max(bank_starts.max(), fit_starts.max()) + context + horizon)
    min_val_start = int(val_starts.min())
    min_test_start = int(test_starts.min())
    assert max_mem_end <= min_val_start, \
        f"[{ds_name}] memory bank future end {max_mem_end} > val start {min_val_start} (LEAK)"
    assert int(val_starts.max()) + context + horizon <= min_test_start, \
        f"[{ds_name}] val future end leaks into test (LEAK)"

    raw_pool = load_raw_pool(ds_name)
    v1_pool = load_v1_pool(ds_name)
    v2_pool = load_v2_pool(ds_name)
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))

    emb_v2_orig = v1.embed_claims([e["claim"] for e in v2_pool], device)
    expert_dim = emb_v2_orig.shape[1]

    # v2 激活：bank/fit 相同，query 分 val / test 两组
    act_val = v1.act_mats(v2_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    act_test = v1.act_mats(v2_pool, X_bank, X_fit, X_test, bank_ts, fit_ts, test_ts, stats)
    v2_act_val = {"bank": act_val["bank"], "fit": act_val["fit"], "val": act_val["val"]}
    v2_act_test = {"bank": act_test["bank"], "fit": act_test["fit"], "test": act_test["val"]}
    # 都从原始（未 pad）embedding 出发做 padding，得到 n_slots 宽度
    emb_v2, v2_act_val, _ = v1.pad_to_slots(emb_v2_orig, v2_act_val, n_slots)
    _, v2_act_test, _ = v1.pad_to_slots(emb_v2_orig, v2_act_test, n_slots)

    # Chronos-2 零样本（确定性）
    print(f"  [Chronos-2] val={len(val_starts)} test={len(test_starts)} ...", flush=True)
    yc_val = run_chronos(pipeline, list(X_val), horizon)
    yc_test = run_chronos(pipeline, list(X_test), horizon)

    # v2 记忆预测（每 seed：训练 + 检索 VAL/TEST）
    mem_val = {}
    mem_test = {}
    for seed in SEEDS:
        init_state = _init_state(seed, context, expert_dim, n_slots, device)
        pv = run_v2_memory_pred(init_state, emb_v2, v2_act_val, True, expert_dim, n_slots,
                                device, seed, context, X_bank, Y_bank, X_fit, Y_fit,
                                X_bank_res, X_fit_res, X_val, X_val_res, v2_act_val["val"])
        pt = run_v2_memory_pred(init_state, emb_v2, v2_act_test, True, expert_dim, n_slots,
                                device, seed, context, X_bank, Y_bank, X_fit, Y_fit,
                                X_bank_res, X_fit_res, X_test, X_test_res, v2_act_test["test"])
        mem_val[seed] = pv.astype(np.float64)
        mem_test[seed] = pt.astype(np.float64)
        print(f"    seed={seed} v2_mem TEST_MSE={mse(Y_test, pt):.6g} "
              f"VAL_MSE={mse(Y_val, pv):.6g}", flush=True)

    # λ 在 VAL 上选（seed 42 的代表性 VAL 记忆，冻结配置）。此循环只用 VAL（mem_val/yc_val/Y_val），
    # 绝不引用 TEST（Y_test/mem_test/yc_test），保证 λ 不因 TEST 泄漏。
    best_lam = None
    best_val_mse = None
    for lam in LAMBDA_GRID:
        fused_val = (1 - lam) * yc_val + lam * mem_val[42]
        vm = float(((fused_val - Y_val) ** 2).mean())
        if best_val_mse is None or vm < best_val_mse:
            best_val_mse, best_lam = vm, lam
    print(f"  selected lambda={best_lam} (VAL MSE={best_val_mse:.6g})", flush=True)

    # TEST 上融合（3 seed，同一 λ）
    per_seed_mse = {}
    for seed in SEEDS:
        fused_test = (1 - best_lam) * yc_test + best_lam * mem_test[seed]
        per_seed_mse[seed] = float(((fused_test - Y_test) ** 2).mean())

    chronos_raw_mse = float(((yc_test - Y_test) ** 2).mean())
    fusion_mean = float(np.mean([per_seed_mse[s] for s in SEEDS]))
    fusion_std = float(np.std([per_seed_mse[s] for s in SEEDS]))

    # 验证：复现冻结 v2 TEST MSE（本脚本 v2 记忆 arm 应与冻结一致）
    frozen_v2_mse = None
    if FROZEN_TEST_SUMMARY.exists():
        for r in csv.DictReader(FROZEN_TEST_SUMMARY.open(encoding="utf-8")):
            if r["dataset"] == ds_name:
                frozen_v2_mse = float(r["v2_mean_mse"])
                break

    row = {
        "dataset": ds_name,
        "n_val": len(val_starts), "n_test": len(test_starts),
        "v2_pool_size": len(v2_pool), "n_slots": n_slots,
        "context": context, "horizon": horizon,
        "chronos_raw_test_mse": f"{chronos_raw_mse:.6f}",
        "selected_lambda": f"{best_lam:.3f}",
        "fusion_test_mse_mean": f"{fusion_mean:.6f}",
        "fusion_test_mse_std": f"{fusion_std:.6f}",
        "fusion_test_mse_seed42": f"{per_seed_mse[42]:.6f}",
        "fusion_test_mse_seed123": f"{per_seed_mse[123]:.6f}",
        "fusion_test_mse_seed2026": f"{per_seed_mse[2026]:.6f}",
        "v2_mem_test_mse_seed42": f"{mse(Y_test, mem_test[42]):.6f}",
        "frozen_v2_test_mse": f"{frozen_v2_mse:.6f}" if frozen_v2_mse is not None else "",
        "v2_repro_check": ("" if frozen_v2_mse is None else
                           "OK" if abs(mse(Y_test, mem_test[42]) - frozen_v2_mse) < 1e-3 else "MISMATCH"),
    }
    return row


FIELDS = ["dataset", "n_val", "n_test", "v2_pool_size", "n_slots", "context", "horizon",
          "chronos_raw_test_mse", "selected_lambda", "fusion_test_mse_mean", "fusion_test_mse_std",
          "fusion_test_mse_seed42", "fusion_test_mse_seed123", "fusion_test_mse_seed2026",
          "v2_mem_test_mse_seed42", "frozen_v2_test_mse", "v2_repro_check"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[d["name"] for d in DATASETS])
    ap.add_argument("--resume", action="store_true", help="skip datasets whose result.json exists")
    ap.add_argument("--out", default=str(OUT_DIR / "ours_fusion_v2_results.csv"))
    args = ap.parse_args()

    names = [d["name"] for d in DATASETS if d["name"] in args.datasets]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8", newline="") as f:
            done = {r["dataset"] for r in csv.DictReader(f)}
    if args.resume:
        names = [n for n in names if n not in done]

    set_seed(42)
    from chronos import Chronos2Pipeline
    print(f"Loading {MODEL} ...", flush=True)
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exists = os.path.exists(args.out)
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
            print(f"[{name}] chronos_raw={row['chronos_raw_test_mse']} "
                  f"fusion={row['fusion_test_mse_mean']}±{row['fusion_test_mse_std']} "
                  f"lambda={row['selected_lambda']} v2_repro={row['v2_repro_check']}", flush=True)
        except ArtifactMissingError as e:
            print(f"[{name}] MISSING ARTIFACT: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {e!r}", flush=True)
            import traceback
            traceback.print_exc()

    print("\nDone. Results ->", args.out)


if __name__ == "__main__":
    main()
