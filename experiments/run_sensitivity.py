#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P6 最小参数敏感性：retrieval Top-K ∈ {1,3,5,10}（正式 Top-K=5）。

TRAIN / frozen VAL only。只动正式方法真正存在的核心参数 Top-K，证明不是极窄参数点有效。
不改变 frozen v2 pool / reward / 阈值 / seeds / split。

输出：results/paper_completion/paper_sensitivity.csv + sensitivity_report.md
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
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
from run_conditional_expert_mlp_v4 import ConditionalExpertMLPRetriever, retrieve_conditional
from run_etth1_prediction_validation import compute_training_stats

import run_etth1_reward_feedback_critic_v1 as v1
import run_reward_feedback_critic_v2_poolaware_pilot as v2p

REPO = HERE.parent
FROZEN_ROOT = REPO / "results/reward_feedback_critic_v2_full_cross_dataset"
OUT = REPO / "results/paper_completion"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = v2p.SEEDS
TOPK_LIST = [1, 3, 5, 10]

DATASETS = ["etth1", "etth2", "weather", "exchange_rate", "solar", "electricity", "traffic"]


def load_v2_pool(name):
    return json.loads((FROZEN_ROOT / name / "final_expert_pool.json").read_text(encoding="utf-8"))


def eval_topk(pool, topk, device, context):
    emb = v1.embed_claims([e["claim"] for e in pool], device)
    act = v1.act_mats(pool, v2p.X_bank, v2p.X_fit, v2p.X_val,
                      v2p.bank_ts, v2p.fit_ts, v2p.val_ts, v2p.stats)
    expert_dim = emb.shape[1]
    n_slots = max(v2p.N_RAW_TARGET, len(pool))  # 与冻结 v2 臂一致：pad 到 8 slots
    emb, act, _ = v1.pad_to_slots(emb, act, n_slots)
    per_seed = {}
    for seed in SEEDS:
        v2p.seed_everything(seed)
        base = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=v2p.D_ENC, d_exp=v2p.D_EXP, d_z=v2p.D_Z, hidden=v2p.HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        del base
        mdl = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=v2p.D_ENC, d_exp=v2p.D_EXP, d_z=v2p.D_Z, hidden=v2p.HIDDEN).to(device)
        mdl.load_state_dict(init_state)
        mdl.set_expert_embeddings(emb)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            v2p.train_arm(mdl, v2p.X_bank, v2p.Y_bank, v2p.X_fit, v2p.Y_fit,
                          v2p.X_bank_res, v2p.X_fit_res, act["bank"], act["fit"],
                          True, device, seed)
        pred = retrieve_conditional(mdl, v2p.X_val, v2p.X_bank, v2p.Y_bank,
                                    v2p.X_val_res, v2p.X_bank_res,
                                    act["val"], act["bank"], "full", topk, device)
        del mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        per_seed[seed] = float(mse(v2p.Y_val, pred))
    return per_seed


def run_dataset(name, device):
    ds_dir = REPO / "results/multidataset" / name
    cfg_json = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg_json["context"])
    horizon = int(cfg_json["horizon"])
    v1.CONTEXT = context
    v1.HORIZON = horizon

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

    pool = load_v2_pool(name)
    out = {"dataset": name}
    for k in TOPK_LIST:
        per = eval_topk(pool, k, device, context)
        out[f"mse_k{k}"] = float(np.mean([per[s] for s in SEEDS]))
        out[f"std_k{k}"] = float(np.std([per[s] for s in SEEDS]))
        print(f"    TopK={k} mean_mse={out[f'mse_k{k}']:.6g}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    targets = [d for d in DATASETS if (args.datasets is None or d in args.datasets)]
    rows = []
    for name in targets:
        print(f"\n{'=' * 60}\nDATASET {name}\n{'=' * 60}")
        try:
            rows.append(run_dataset(name, device))
        except Exception as e:
            print(f"  [FAILED] {name}: {e}")
            traceback.print_exc()

    cols = ["dataset"] + [f"mse_k{k}" for k in TOPK_LIST] + [f"std_k{k}" for k in TOPK_LIST]
    with open(OUT / "paper_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    L = ["# P6 最小参数敏感性（Top-K）", ""]
    L.append("正式 retrieval Top-K=5。仅 TRAIN/frozen VAL，测试 K ∈ {1,3,5,10}。不改 frozen v2。")
    L.append("")
    L.append("| Dataset | K=1 | K=3 | K=5（正式） | K=10 |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['dataset']} | {r['mse_k1']:.6g} | {r['mse_k3']:.6g} "
                 f"| {r['mse_k5']:.6g} | {r['mse_k10']:.6g} |")
    L.append("")
    L.append("## 结果 / 结论")
    L.append("- 若 K=5 非孤立最优点（邻近 K=3/10 也稳定），说明结果对 Top-K 稳健。")
    L.append("- 目的仅证明不是极窄参数点有效；不据此改 K。")
    with open(OUT / "sensitivity_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
