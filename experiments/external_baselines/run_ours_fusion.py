# -*- coding: utf-8 -*-
"""
Ours = Chronos-2 + Expert/Memory Retrieval Fusion, re-run under the FORMAL
external-baseline protocol (identical split / context / horizon / target /
raw-MSE as PatchTST / iTransformer / RAFT / RAF).

Fusion algorithm is UNCHANGED from the validated continuous temporal-weight
fusion (archive/old_experiments/run_continuous_temporal_weights_all_ett.py +
run_temporal_ablation_all_ett.py). Only the outer data protocol is switched
from the calendar-date ETT split to the formal config.json + window_indices.npz.

Guarantees (see audit):
  - memory bank built from TRAIN only (never val/test)
  - lambda / alpha / temporal-weights / period / trend_window selected on VAL only
  - TEST evaluated exactly once with the frozen config
  - raw-value MSE, no scaler, no inverse-scaler (backbone output and future
    both live in raw OT space, so no re-scaling exists to duplicate)
  - Chronos-2 raw and Ours fusion use the SAME test windows (window_indices.npz)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import torch

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

MULTI = os.path.join(REPO, "results", "multidataset")
OUT_CSV = os.path.join(REPO, "results", "external_baselines", "ours_fusion_results.csv")

MODEL = "amazon/chronos-2"
K = 5
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
STEP = 0.1  # continuous temporal-weight grid step
BATCH = 256
SEED = 42

# canonical dataset name -> actual multidataset directory
DATASET_DIRS = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "exchange_rate": "exchange_rate",
    "electricity": "electricity",
    "weather": "weather",
    "traffic": "traffic",
    "solar": "solar",
}

# per-dataset: stride (points per hour) + period candidates (period, trend_window).
# exchange_rate keeps BOTH config A (period=1) and B (period=7); the winner is
# chosen by VAL MSE only (never TEST).
DATASET_CONFIGS = {
    "etth1":         dict(stride=1, period_candidates=[(24, 48)]),
    "etth2":         dict(stride=1, period_candidates=[(24, 48)]),
    "traffic":       dict(stride=1, period_candidates=[(24, 48)]),
    "electricity":   dict(stride=4, period_candidates=[(96, 192)]),
    "weather":       dict(stride=6, period_candidates=[(144, 288)]),
    "solar":         dict(stride=6, period_candidates=[(144, 288)]),
    "exchange_rate": dict(stride=1, period_candidates=[(1, 1), (7, 14)]),  # A, B
}


# --------------------------------------------------------------------------- #
# Algorithm helpers — VERBATIM from run_continuous_temporal_weights_all_ett.py
# --------------------------------------------------------------------------- #
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def circular_lut(n):
    lut = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            lut[i, j] = 1.0 - d / (n // 2)
    return lut


def weight_grid(step=STEP):
    n = int(round(1.0 / step))
    grid = []
    for h in range(n + 1):
        for w in range(n + 1 - h):
            m = n - h - w
            grid.append((h * step, w * step, m * step))
    return grid


def standardize(B, S_p):
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)                                   # (n, K)
    fut = bank.futures[idx]                               # (n, K, H)
    ctx = bank.contexts[idx]                              # (n, K, L)
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def shape_tiebreak_order(shape_sim):
    return np.argsort(-shape_sim, axis=-1, kind="stable")


def topk_deterministic(scores, shape_order, k):
    """Lexicographic top-K: final_score desc, shape_sim desc, memory_index asc."""
    score_tb = np.take_along_axis(scores, shape_order, axis=-1)
    o = np.argsort(-score_tb, axis=-1, kind="stable")
    return np.take_along_axis(shape_order, o, axis=-1)[..., :k]


def topk_fast(scores, k):
    """argpartition top-K; valid only when scores have no exact ties (alpha < 1.0)."""
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


# --------------------------------------------------------------------------- #
# Data protocol (FORMAL) — the ONLY changed part vs the old calendar-date split
# --------------------------------------------------------------------------- #
def load_formal(ds):
    """Return dates, ot, context, horizon, train_end, val_starts, test_starts."""
    d = os.path.join(MULTI, DATASET_DIRS[ds])
    df = pd.read_csv(os.path.join(d, "series.csv"))
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    cfg = json.loads(open(os.path.join(d, "config.json"), encoding="utf-8").read())
    context = int(cfg["context"])
    horizon = int(cfg["horizon"])
    train_end = int(cfg["split"]["train"][1])
    idx = np.load(os.path.join(d, "window_indices.npz"))
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    test_starts = np.asarray(idx["test"], dtype=np.int64)
    return dates, ot, context, horizon, train_end, val_starts, test_starts


def build_memory_bank(train, context, horizon, stride):
    window = context + horizon
    starts = np.arange(0, len(train) - window + 1, stride)
    contexts = np.stack([train[s:s + context] for s in starts]).astype(np.float32)
    futures = np.stack([train[s + context:s + window] for s in starts]).astype(np.float32)
    return MemoryBank(contexts=contexts, futures=futures), starts


def residual_pearson(contexts, bank, period, trend_window, horizon):
    _, _, R_q, _, _ = seasonal_decompose(contexts, horizon, period, trend_window)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, horizon, period, trend_window)
    return pearson_similarity_matrix(R_q, R_m).astype(np.float32)


def build_alignments(q_dates, mem_dates, luts):
    month_lut, dow_lut, hour_lut = luts
    month = month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
    dow = dow_lut[q_dates.dt.dayofweek.to_numpy()][:, mem_dates.dt.dayofweek.to_numpy()]
    hour = hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]
    return {"hour": hour, "dow": dow, "month": month}


def windows(ot, starts, context, horizon):
    # formal window convention (matches build_windows / frozen Ours v2):
    # s is the CONTEXT start; context=[s, s+context), future=[s+context, s+context+horizon)
    contexts = np.stack([ot[s:s + context] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s + context:s + context + horizon] for s in starts]).astype(np.float32)
    return contexts, futures


def run_chronos(pipeline, contexts, horizon):
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=horizon, batch_size=BATCH)
    return np.stack([m.squeeze(0).numpy() for m in means])


# --------------------------------------------------------------------------- #
# Per-dataset compute: grid-search on VAL, evaluate once on TEST
# --------------------------------------------------------------------------- #
def compute(ds, pipeline, out_csv):
    stride = DATASET_CONFIGS[ds]["stride"]
    period_candidates = DATASET_CONFIGS[ds]["period_candidates"]

    dates, ot, context, horizon, train_end, val_starts, test_starts = load_formal(ds)
    train = ot[:train_end]
    train_dates = dates[:train_end]

    bank, mem_starts = build_memory_bank(train, context, horizon, stride)
    mem_dates = train_dates.iloc[mem_starts + context]

    val_contexts, val_futures = windows(ot, val_starts, context, horizon)
    test_contexts, test_futures = windows(ot, test_starts, context, horizon)
    # temporal-LUT "query date" = the FUTURE-start date (s + context), same as mem_dates
    val_dates = dates.iloc[val_starts + context]
    test_dates = dates.iloc[test_starts + context]

    # Chronos-2 raw (frozen backbone, deterministic zero-shot) on the SAME test windows
    y_chronos_test = run_chronos(pipeline, test_contexts, horizon)
    y_chronos_val = run_chronos(pipeline, val_contexts, horizon)

    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    aligns_val = build_alignments(val_dates, mem_dates, luts)
    aligns_test = build_alignments(test_dates, mem_dates, luts)

    grid = weight_grid()

    best = None  # (val_mse, val_mae, wh, ww, wm, alpha, lam, period, trend_window)
    for period, trend_window in period_candidates:
        S_p_val = residual_pearson(val_contexts, bank, period, trend_window, horizon)
        shape_order_val = shape_tiebreak_order(S_p_val)
        for wh, ww, wm in grid:
            TR_val = wh * aligns_val["hour"] + ww * aligns_val["dow"] + wm * aligns_val["month"]
            Tstd_val = standardize(TR_val, S_p_val)
            for alpha in ALPHAS:
                S = (1 - alpha) * S_p_val + alpha * Tstd_val
                if alpha == 1.0:
                    idx = topk_deterministic(S, shape_order_val, K)
                else:
                    idx = topk_fast(S, K)
                sc = np.take_along_axis(S, idx, axis=-1)
                y_mem = mean_anchor(bank, val_contexts, idx, sc)
                for lam in LAMS:
                    fused = (1 - lam) * y_chronos_val + lam * y_mem
                    mse = float(((fused - val_futures) ** 2).mean())
                    mae = float(np.abs(fused - val_futures).mean())
                    if best is None or (mse, mae) < (best[0], best[1]):
                        best = (mse, mae, wh, ww, wm, alpha, lam, period, trend_window)

    _, _, wh, ww, wm, alpha, lam, period, trend_window = best

    # freeze + evaluate once on TEST (no further search on TEST)
    S_p_test = residual_pearson(test_contexts, bank, period, trend_window, horizon)
    shape_order_test = shape_tiebreak_order(S_p_test)
    TR_test = wh * aligns_test["hour"] + ww * aligns_test["dow"] + wm * aligns_test["month"]
    Tstd_test = standardize(TR_test, S_p_test)
    S_test = (1 - alpha) * S_p_test + alpha * Tstd_test
    if alpha == 1.0:
        idx = topk_deterministic(S_test, shape_order_test, K)
    else:
        idx = topk_fast(S_test, K)
    sc = np.take_along_axis(S_test, idx, axis=-1)
    y_mem = mean_anchor(bank, test_contexts, idx, sc)
    fused = (1 - lam) * y_chronos_test + lam * y_mem

    chronos_raw_mse = float(((y_chronos_test - test_futures) ** 2).mean())
    ours_fusion_mse = float(((fused - test_futures) ** 2).mean())

    row = {
        "dataset": ds, "seed": str(SEED),
        "chronos_raw_mse": f"{chronos_raw_mse:.6f}",
        "ours_fusion_mse": f"{ours_fusion_mse:.6f}",
        "selected_lambda": f"{lam:.2f}",
        "selected_alpha": f"{alpha:.2f}",
        "selected_temporal_weights": f"({wh:.1f},{ww:.1f},{wm:.1f})",
        "selected_period": str(period),
        "selected_trend_window": str(trend_window),
        "selected_stride": str(stride),
    }

    if out_csv:
        exists = os.path.exists(out_csv)
        with open(out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS))
    ap.add_argument("--dry-run", action="store_true",
                    help="print per-dataset config + window counts, then exit (no Chronos, no GPU)")
    ap.add_argument("--out-csv", default=OUT_CSV)
    args = ap.parse_args()

    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            raise SystemExit(f"unknown dataset {ds!r}")

    if args.dry_run:
        for ds in args.datasets:
            dates, ot, context, horizon, train_end, val_starts, test_starts = load_formal(ds)
            stride = DATASET_CONFIGS[ds]["stride"]
            pc = DATASET_CONFIGS[ds]["period_candidates"]
            n_mem = (train_end - (context + horizon)) // stride + 1
            print(f"{ds:14s} ctx={context} hor={horizon} train_end={train_end} "
                  f"stride={stride} n_val={len(val_starts)} n_test={len(test_starts)} "
                  f"n_mem={n_mem} period_candidates={pc}")
        return

    set_seed(SEED)
    from chronos import Chronos2Pipeline
    print(f"Loading {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")

    for ds in args.datasets:
        r = compute(ds, pipeline, args.out_csv)
        print(f"[{ds}] chronos_raw_mse={r['chronos_raw_mse']} "
              f"ours_fusion_mse={r['ours_fusion_mse']} "
              f"lambda={r['selected_lambda']} alpha={r['selected_alpha']} "
              f"w={r['selected_temporal_weights']} period={r['selected_period']} "
              f"trend={r['selected_trend_window']} stride={r['selected_stride']}")


if __name__ == "__main__":
    main()
