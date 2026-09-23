#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P5 效率开销（RAW / v1 / v2）。

日志无 wall-clock，按协议在 TRAIN/VAL 重新测时间（固定机器环境，每项 3 次取 median）。
禁止重新跑 TEST。

测量（etth1 为代表，固定 RTX 5070 Ti / 16GB）：
  - retriever_training_time：final arm 的 30-epoch retriever 训练时间
  - inference_time_per_window：retrieve_conditional 单窗推理时间
  - peak GPU memory

解析（无需重跑）：
  - number_of_expert_evaluations：v1 逐条 vs v2 pool-aware 的 reward 打分次数
  - final_pool_size

输出：results/paper_completion/paper_efficiency.csv + efficiency_report.md
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import time
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
V1_CROSS = v2p.V1_CROSS
V1_ETTH1 = v2p.V1_ETTH1
OUT = REPO / "results/paper_completion"

SEEDS = v2p.SEEDS
N_FOLD_BLOCKS = v2p.N_FOLD_BLOCKS   # 5 -> folds 1..4 = 4 folds
N_FOLDS = N_FOLD_BLOCKS - 1


def median3(fn):
    vals = [fn() for _ in range(3)]
    return float(np.median(vals))


def measure(name, device):
    ds_dir = REPO / "results/multidataset" / name
    cfg_json = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg_json["context"]); horizon = int(cfg_json["horizon"])
    v1.CONTEXT = context; v1.HORIZON = horizon
    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]; fit_starts = train_starts[cut:]
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

    pool = json.loads((FROZEN_ROOT / name / "final_expert_pool.json").read_text(encoding="utf-8"))
    emb = v1.embed_claims([e["claim"] for e in pool], device)
    act = v1.act_mats(pool, v2p.X_bank, v2p.X_fit, v2p.X_val,
                      v2p.bank_ts, v2p.fit_ts, v2p.val_ts, v2p.stats)
    expert_dim = emb.shape[1]; n_slots = max(v2p.N_RAW_TARGET, len(pool))  # 与冻结一致：8 slots
    emb, act, _ = v1.pad_to_slots(emb, act, n_slots)

    def one_train():
        seed = 42
        v2p.seed_everything(seed)
        mdl = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=v2p.D_ENC, d_exp=v2p.D_EXP, d_z=v2p.D_Z, hidden=v2p.HIDDEN).to(device)
        mdl.set_expert_embeddings(emb)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            v2p.train_arm(mdl, v2p.X_bank, v2p.Y_bank, v2p.X_fit, v2p.Y_fit,
                          v2p.X_bank_res, v2p.X_fit_res, act["bank"], act["fit"],
                          True, device, seed)
        return mdl

    train_time = median3(lambda: _time_train(one_train, device))

    # inference per window
    def one_infer():
        seed = 42
        v2p.seed_everything(seed)
        mdl = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=v2p.D_ENC, d_exp=v2p.D_EXP, d_z=v2p.D_Z, hidden=v2p.HIDDEN).to(device)
        mdl.set_expert_embeddings(emb)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            v2p.train_arm(mdl, v2p.X_bank, v2p.Y_bank, v2p.X_fit, v2p.Y_fit,
                          v2p.X_bank_res, v2p.X_fit_res, act["bank"], act["fit"],
                          True, device, seed)
        t = time.perf_counter()
        retrieve_conditional(mdl, v2p.X_val, v2p.X_bank, v2p.Y_bank,
                             v2p.X_val_res, v2p.X_bank_res,
                             act["val"], act["bank"], "full", v2p.TOPK, device)
        return time.perf_counter() - t

    infer_total = median3(one_infer)
    infer_per_window = infer_total / len(v2p.X_val)

    # peak memory
    peak = 0.0
    try:
        torch.cuda.reset_peak_memory_stats()
        _time_train(one_train, device)
        peak = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    except Exception:
        peak = float("nan")

    return {"train_time": train_time, "infer_per_window": infer_per_window,
            "n_val_windows": len(v2p.X_val), "peak_mem_mb": peak}


def _time_train(make, device):
    t = time.perf_counter()
    mdl = make()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dt


def analytic_counts():
    """v1 / v2 reward 打分次数（解析）。"""
    out = {}
    for name in ["etth1", "etth2", "weather", "exchange_rate", "solar", "electricity", "traffic"]:
        # v1 reward history rows = expert x round scorings
        v1src = (V1_ETTH1 / "reward_history.csv") if name == "etth1" else (V1_CROSS / name / "reward_history.csv")
        v1_rows = sum(1 for _ in csv.DictReader(v1src.open(encoding="utf-8"))) if v1src.exists() else 0
        # v2 pool reward rounds
        v2src = FROZEN_ROOT / name / "pool_reward_history.csv"
        rounds = set()
        for r in csv.DictReader(v2src.open(encoding="utf-8")):
            rounds.add(int(r["round"]))
        n_rounds = max(rounds) + 1 if rounds else 0
        n_experts = 8
        v2_trainings = n_rounds * N_FOLDS * (n_experts + 1)  # 每轮 N_folds x (full + 8 minus)
        out[name] = {"v1_expert_evaluations": v1_rows, "v2_rounds": n_rounds,
                     "v2_retriever_trainings_for_reward": v2_trainings}
    return out


def main():
    ap = None
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="etth1")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    m = measure(args.dataset, device)
    print(f"train_time={m['train_time']:.3f}s infer_per_window={m['infer_per_window']*1000:.3f}ms "
          f"n_val={m['n_val_windows']} peak_mem={m['peak_mem_mb']:.1f}MB")

    counts = analytic_counts()
    pool_stats = {r["dataset"]: r for r in csv.DictReader(
        (REPO / "results/reward_feedback_critic_v2_full_cross_dataset/full_cross_dataset_pool_stats.csv").open(encoding="utf-8"))}

    # 效率表（以 etth1 实测时间代表；v1/v2 reward 时间 = 次数 x 单次 retriever 训练时间）
    rows = []
    for name, c in counts.items():
        p = pool_stats[name]
        # final retriever training: 每 arm 30 epoch，RAW/v1/v2 相同（同一训练预算）
        # expert_validation_time 估算：reward 打分 = retriever train+eval 次数 x train_time
        v1_val_time = c["v1_expert_evaluations"] * m["train_time"] / 8  # v1 逐条用 fold-mean-gain，非全 30ep retriever；保守按 per-expert 折算
        v2_val_time = c["v2_retriever_trainings_for_reward"] * m["train_time"]
        rows.append({
            "dataset": name,
            "raw_final_pool_size": p["raw_pool_size"], "v1_final_pool_size": p["v1_pool_size"],
            "v2_final_pool_size": p["v2_pool_size"],
            "raw_expert_evaluations": 0,
            "v1_expert_evaluations": c["v1_expert_evaluations"],
            "v2_retriever_trainings_for_reward": c["v2_retriever_trainings_for_reward"],
            "retriever_training_time_sec": round(m["train_time"], 3),
            "inference_time_per_window_ms": round(m["infer_per_window"] * 1000, 4),
            "v1_expert_validation_time_sec_est": round(v1_val_time, 2),
            "v2_expert_validation_time_sec_est": round(v2_val_time, 2),
            "peak_mem_mb": round(m["peak_mem_mb"], 1),
        })

    with open(OUT / "paper_efficiency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    L = ["# P5 效率开销（RAW / REWARD_V1 / POOL_AWARE_V2）", ""]
    L.append(f"实测环境：RTX 5070 Ti / 16GB，代表性数据集 {args.dataset}，每项 3 次取 median。")
    L.append(f"单次 final retriever 训练（30 epoch）：{m['train_time']:.3f}s；"
             f"单窗推理：{m['infer_per_window']*1000:.3f}ms；峰值显存：{m['peak_mem_mb']:.1f}MB。")
    L.append("")
    L.append("| Dataset | raw pool | v1 pool | v2 pool | v1 eval 次数 | v2 reward 训练次数 | v1 val 时间(估) | v2 val 时间(估) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['dataset']} | {r['raw_final_pool_size']} | {r['v1_final_pool_size']} | {r['v2_final_pool_size']} "
                 f"| {r['v1_expert_evaluations']} | {r['v2_retriever_trainings_for_reward']} "
                 f"| {r['v1_expert_validation_time_sec_est']}s | {r['v2_expert_validation_time_sec_est']}s |")
    L.append("")
    L.append("## 结果")
    L.append("- 最终推理开销三者相同（同一 retriever，同一 30-epoch 训练，同一单窗推理）。")
    L.append("- v2 pool-aware reward 比 v1 逐条 reward 贵：每轮要 N_folds × (N_experts+1) 次 retriever 训练"
             "（v1 是逐条 fold-mean-gain，不做 full-minus 重训）。")
    L.append("- 但 reward 只发生在 TRAIN 阶段，不进入推理；最终 TEST/推理的 wall-clock 三者无差异。")
    L.append("")
    L.append("## 结论")
    L.append("- v2 的额外成本 = TRAIN-only 的 pool-aware reward 计算（一次性），不增加推理延迟。")
    L.append("- 时间估算基于解析次数 × 实测单次训练时间；v1 val 时间为保守折算（未重跑 v1 全流程）。")
    with open(OUT / "efficiency_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
