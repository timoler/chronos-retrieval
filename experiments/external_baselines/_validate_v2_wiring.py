#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量验证：确认 run_ours_fusion_v2.py 的 v2 记忆预测接线位一致（CPU，不加载 Chronos-2）。

只对 exchange_rate（最小）跑 v2 记忆预测（seed42，TEST），比对冻结 v2 TEST MSE。
用于在长任务前抓运行时 bug。不改动任何冻结产物。
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

from run_expert_embedding_mlp import build_windows, mse  # noqa: E402
from run_solar_same_val_baselines import residualize        # noqa: E402
from run_conditional_expert_mlp_v4 import (                 # noqa: E402
    ConditionalExpertMLPRetriever, retrieve_conditional)
from run_etth1_prediction_validation import compute_training_stats  # noqa: E402
import run_etth1_reward_feedback_critic_v1 as v1          # noqa: E402
import run_reward_feedback_critic_v2_poolaware_pilot as v2p  # noqa: E402

REPO_ROOT = HERE.parent.parent
FROZEN_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_full_cross_dataset"
V1_ETTH1 = v2p.V1_ETTH1
V1_CROSS = v2p.V1_CROSS
SEEDS = v2p.SEEDS
TOPK = v2p.TOPK
N_RAW_TARGET = v2p.N_RAW_TARGET
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

NAME = "exchange_rate"
FROZEN_V2_MSE = 0.03647978603839874  # 来自 final_test_summary.csv


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
    return json.loads((FROZEN_ROOT / name / "final_expert_pool.json").read_text(encoding="utf-8"))


def main():
    device = torch.device("cpu")
    ds_dir = REPO_ROOT / "results/multidataset" / NAME
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
    test_starts = np.asarray(idx["test"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    X_bank, Y_bank = build_windows(series, bank_starts, context, horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, context, horizon)
    X_test, Y_test = build_windows(series, test_starts, context, horizon)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_test_res = residualize(X_test)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    test_ts = [dates[s + context] for s in test_starts]
    stats = compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    raw_pool = load_raw_pool(NAME)
    v1_pool = load_v1_pool(NAME)
    v2_pool = load_v2_pool(NAME)
    n_slots = max(len(raw_pool), len(v1_pool), len(v2_pool))
    print(f"pools raw={len(raw_pool)} v1={len(v1_pool)} v2={len(v2_pool)} n_slots={n_slots}")

    emb_v2_orig = v1.embed_claims([e["claim"] for e in v2_pool], device)
    expert_dim = emb_v2_orig.shape[1]
    act_test = v1.act_mats(v2_pool, X_bank, X_fit, X_test, bank_ts, fit_ts, test_ts, stats)
    v2_act = {"bank": act_test["bank"], "fit": act_test["fit"], "test": act_test["val"]}
    emb_v2, v2_act, _ = v1.pad_to_slots(emb_v2_orig, v2_act, n_slots)

    seed = 42
    v2p.seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    init_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
    mdl.set_expert_embeddings(emb_v2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v2p.train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      v2_act["bank"], v2_act["fit"], True, device, seed)
    pred = retrieve_conditional(mdl, X_test, X_bank, Y_bank, X_test_res, X_bank_res,
                                v2_act["test"], v2_act["bank"], "full", TOPK, device)
    got = mse(Y_test, pred)
    print(f"v2 memory pred TEST_MSE (seed42, CPU) = {got:.6f}")
    print(f"frozen v2 TEST_MSE (seed-mean)     = {FROZEN_V2_MSE:.6f}")
    print(f"repro delta = {abs(got - FROZEN_V2_MSE):.6f}")
    # 注：CPU vs CUDA + seed42 vs seed-mean 会有小差异；这里只校验量级/接线对不对
    print("RESULT:", "OK (wiring plausible)" if abs(got - FROZEN_V2_MSE) < 0.01 else "NEED CHECK")


if __name__ == "__main__":
    main()
