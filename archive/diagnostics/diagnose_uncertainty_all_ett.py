"""Prediction-uncertainty diagnostics for the temporal-alignment fusion.

Chronos-2 is a quantile-regression model; its ``predict_quantiles`` returns the
native training quantiles [0.01, ..., 0.99].  For each test window we compute a
single scalar uncertainty spread WITHOUT touching y_true:

    spread_i = mean_h( q0.9[i, h] - q0.1[i, h] )

Then we (a) bucket windows by uncertainty quintiles and report per-bucket
n / mean spread / Chronos MSE / Memory MSE / Fusion MSE / fusion improvement,
and (b) report Spearman + Pearson correlation between the per-window spread and
the per-window Chronos MSE.

Fusion reuses the BEST temporal combination (frozen alpha/lambda from
run_temporal_ablation_all_ett.py's val selection) — nothing re-selected, no
gate fitted, retrieval unchanged.  Quantile forecasts are cached per dataset
under results/<name>_quantiles.npz.
"""

import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
MODEL = "amazon/chronos-2"
BATCH = 256

DATASETS = {
    "ETTh1": dict(period=24, trend_window=48, stride=1),
    "ETTh2": dict(period=24, trend_window=48, stride=1),
    "ETTm1": dict(period=96, trend_window=192, stride=4),
    "ETTm2": dict(period=96, trend_window=192, stride=4),
}

# BEST temporal combination per dataset (frozen from the ablation run's val
# selection): (calendar keys, alpha, lambda)
BEST = {
    "ETTh1": (["hour", "month"], 0.75, 0.20),
    "ETTh2": (["hour", "dow", "month"], 1.00, 0.30),
    "ETTm1": (["hour", "dow"], 0.75, 0.10),
    "ETTm2": (["hour"], 0.75, 0.20),
}


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    return dates, ot, train_mask, val_mask, test_mask


def circular_lut(n):
    lut = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            lut[i, j] = 1.0 - d / (n // 2)
    return lut


def topk(scores, k):
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def windows(ot, start, end, stride):
    s = np.arange(start, end - PRED + 1, stride)
    c = np.stack([ot[i - CONTEXT : i] for i in s]).astype(np.float32)
    f = np.stack([ot[i : i + PRED] for i in s]).astype(np.float32)
    return s, c, f


def build_memory_bank(train, stride):
    window = CONTEXT + PRED
    starts = np.arange(0, len(train) - window + 1, stride)
    contexts = np.stack([train[s : s + CONTEXT] for s in starts]).astype(np.float32)
    futures = np.stack([train[s + CONTEXT : s + window] for s in starts]).astype(np.float32)
    return MemoryBank(contexts=contexts, futures=futures), starts


def residual_pearson(contexts, bank, period, trend_window):
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, period, trend_window)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, period, trend_window)
    return pearson_similarity_matrix(R_q, R_m).astype(np.float32)


def build_alignments(q_dates, mem_dates, luts):
    month_lut, dow_lut, hour_lut = luts
    month = month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
    dow = dow_lut[q_dates.dt.dayofweek.to_numpy()][:, mem_dates.dt.dayofweek.to_numpy()]
    hour = hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]
    return {"hour": hour, "dow": dow, "month": month}


def combined_B(aligns, keys):
    B = aligns[keys[0]].copy()
    for k in keys[1:]:
        B += aligns[k]
    B /= len(keys)
    return B


def standardize(B, S_p):
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def run_quantiles(pipeline, contexts):
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    quantiles, _ = pipeline.predict_quantiles(
        inputs, prediction_length=PRED, quantile_levels=[0.1, 0.5, 0.9], batch_size=BATCH
    )
    q = np.stack([x.squeeze(0).numpy() for x in quantiles])  # (n, H, 3) = [q10, q50, q90]
    return q[..., 0], q[..., 1], q[..., 2]


def load_pipeline():
    from chronos import Chronos2Pipeline

    print(f"Loading {MODEL} ...")
    return Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")


def ensure_quantiles(pipeline, name, test_contexts, cache_dir, force):
    path = os.path.join(cache_dir, f"{name}_quantiles.npz")
    if (not force) and os.path.exists(path):
        d = np.load(path)
        q10, q90, median = d["q10"], d["q90"], d["median"]
        assert q10.shape[0] == len(test_contexts), f"{name}: cached quantile count mismatch"
        return q10, q90, median
    assert pipeline is not None, f"{name}: no quantile cache and no pipeline"
    q10, median, q90 = run_quantiles(pipeline, test_contexts)
    np.savez(path, q10=q10, q90=q90, median=median)
    return q10, q90, median


def analyze(name, pipeline, cache_dir, force):
    cfg = DATASETS[name]
    keys, alpha, lam = BEST[name]
    period, trend_window, stride = cfg["period"], cfg["trend_window"], cfg["stride"]

    dates, ot, train_mask, val_mask, test_mask = load(f"data/{name}.csv")
    train = ot[train_mask]
    train_dates = dates[train_mask]

    bank, mem_starts = build_memory_bank(train, stride)
    mem_dates = train_dates.iloc[mem_starts + CONTEXT]

    test_start = int(np.flatnonzero(test_mask)[0])
    test_end = int(np.flatnonzero(test_mask)[-1]) + 1
    test_starts, test_contexts, test_futures = windows(ot, test_start, test_end, stride)
    test_dates = dates.iloc[test_starts]

    y_chronos = np.load(f"results/{name}_chronos.npz")["y_chronos_test"]
    q10, q90, median = ensure_quantiles(pipeline, name, test_contexts, cache_dir, force)

    uncertainty = (q90 - q10).mean(axis=1)  # per-window scalar spread, no y_true

    # retrieval fusion (frozen alpha/lambda)
    S_p = residual_pearson(test_contexts, bank, period, trend_window)
    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    aligns = build_alignments(test_dates, mem_dates, luts)
    B = standardize(combined_B(aligns, keys), S_p)
    S = (1 - alpha) * S_p + alpha * B
    idx = topk(S, K)
    sc = np.take_along_axis(S, idx, axis=-1)
    y_mem = mean_anchor(bank, test_contexts, idx, sc)
    fused = (1 - lam) * y_chronos + lam * y_mem

    # per-window MSE (mean over the 96-step horizon)
    chronos_mse_w = ((y_chronos - test_futures) ** 2).mean(axis=1)
    mem_mse_w = ((y_mem - test_futures) ** 2).mean(axis=1)
    fus_mse_w = ((fused - test_futures) ** 2).mean(axis=1)

    # consistency: regenerated median should match the cached point forecast
    median_diff = float(np.abs(median - y_chronos).max())

    from scipy.stats import pearsonr, spearmanr

    spearman = float(spearmanr(uncertainty, chronos_mse_w).statistic)
    pearson = float(pearsonr(uncertainty, chronos_mse_w).statistic)

    combo = "+".join({"hour": "Hour", "dow": "Weekday", "month": "Month"}[k] for k in keys)
    print(f"\n=== {name}  (best={combo}, alpha={alpha}, lambda={lam}) ===")
    print(f"n_windows = {len(test_starts)}")
    print(f"uncertainty (mean_h q0.9-q0.1)  mean/std = {uncertainty.mean():.4f} / {uncertainty.std():.4f}")
    print(f"spearman(unc, per-window Chronos MSE) = {spearman:+.4f}")
    print(f"pearson (unc, per-window Chronos MSE) = {pearson:+.4f}")
    print(f"median-vs-cached-point max|diff|       = {median_diff:.6f}")

    edges = np.quantile(uncertainty, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins = np.digitize(uncertainty, edges) - 1

    print(f"\n{'unc bucket':>11} {'n':>7} {'mean_unc':>9} {'ChronosMSE':>11} "
          f"{'MemoryMSE':>10} {'FusionMSE':>10} {'imp%':>8}")
    for b in range(5):
        m = bins == b
        chronos_mse = float(chronos_mse_w[m].mean())
        mem_mse = float(mem_mse_w[m].mean())
        fus_mse = float(fus_mse_w[m].mean())
        imp = (chronos_mse - fus_mse) / chronos_mse * 100
        print(f"{'Q%d' % (b + 1):>11} {int(m.sum()):>7} {uncertainty[m].mean():9.4f} "
              f"{chronos_mse:11.4f} {mem_mse:10.4f} {fus_mse:10.4f} {imp:+7.2f}%")


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    names = [n.strip() for n in args.datasets.split(",")] if args.datasets else list(DATASETS)
    for n in names:
        if n not in DATASETS:
            raise SystemExit(f"unknown dataset {n!r}")

    need_pipeline = any(
        args.force
        or not os.path.exists(os.path.join(args.cache_dir, f"{n}_quantiles.npz"))
        for n in names
    )
    pipeline = load_pipeline() if need_pipeline else None

    for name in names:
        analyze(name, pipeline, args.cache_dir, args.force)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None)
    p.add_argument("--force", action="store_true", help="regenerate quantile forecasts (ignore cache)")
    p.add_argument("--cache-dir", type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    main()
