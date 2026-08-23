"""Cross-dataset: Chronos-2 baseline vs + Hour/Month temporal-aligned retrieval.

Per dataset: OT target, context=512, horizon=96, Top-K=5, Mean-anchor.
Residual decomposition period: 24 (ETTh, hourly) / 96 (ETTm, 15-min).
Temporal alignment: Hour (hour-of-day, timestamp) + Month, alpha=0.75.
Lambda selected on VAL (grid), applied once to TEST.  Stride: 1 (ETTh) / 4 (ETTm).
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
ALPHA = 0.75
MODEL = "amazon/chronos-2"
BATCH = 256
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

DATASETS = {
    "ETTh1": dict(period=24, trend_window=48, stride=1),
    "ETTh2": dict(period=24, trend_window=48, stride=1),
    "ETTm1": dict(period=96, trend_window=192, stride=4),
    "ETTm2": dict(period=96, trend_window=192, stride=4),
}


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


def build_memory_bank(train, context, pred, stride):
    window = context + pred
    starts = np.arange(0, len(train) - window + 1, stride)
    contexts = np.stack([train[s : s + context] for s in starts]).astype(np.float32)
    futures = np.stack([train[s + context : s + window] for s in starts]).astype(np.float32)
    return MemoryBank(contexts=contexts, futures=futures), starts


def windows(ot, start, end, context, pred, stride):
    s = np.arange(start, end - pred + 1, stride)
    c = np.stack([ot[i - context : i] for i in s]).astype(np.float32)
    f = np.stack([ot[i : i + pred] for i in s]).astype(np.float32)
    return s, c, f


def residual_pearson(contexts, bank, period, trend_window):
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, period, trend_window)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, period, trend_window)
    return pearson_similarity_matrix(R_q, R_m).astype(np.float32)


def temporal_B(q_dates, mem_dates, S_p):
    month_lut, hour_lut = circular_lut(12), circular_lut(24)
    month = month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
    hour = hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]
    B = (month + hour) / 2.0
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def run_chronos(pipeline, contexts):
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    return np.stack([m.squeeze(0).numpy() for m in means])


def process(name, cfg, dates, ot, pipeline):
    period, trend_window, stride = cfg["period"], cfg["trend_window"], cfg["stride"]
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    train_dates = dates[train_mask]

    bank, mem_starts = build_memory_bank(train, CONTEXT, PRED, stride)
    mem_dates = train_dates.iloc[mem_starts + CONTEXT]

    val_idx = np.flatnonzero(val_mask)
    val_start, val_end = int(val_idx[0]), int(val_idx[-1]) + 1
    test_idx = np.flatnonzero(test_mask)
    test_start, test_end = int(test_idx[0]), int(test_idx[-1]) + 1

    val_starts, val_contexts, val_futures = windows(ot, val_start, val_end, CONTEXT, PRED, stride)
    test_starts, test_contexts, test_futures = windows(ot, test_start, test_end, CONTEXT, PRED, stride)
    val_dates = dates.iloc[val_starts]
    test_dates = dates.iloc[test_starts]

    S_p_val = residual_pearson(val_contexts, bank, period, trend_window)
    S_p_test = residual_pearson(test_contexts, bank, period, trend_window)
    B_val = temporal_B(val_dates, mem_dates, S_p_val)
    B_test = temporal_B(test_dates, mem_dates, S_p_test)

    y_chronos_val = run_chronos(pipeline, val_contexts)
    y_chronos_test = run_chronos(pipeline, test_contexts)

    S_val = (1 - ALPHA) * S_p_val + ALPHA * B_val
    idx_val = topk(S_val, K)
    y_mem_val = mean_anchor(bank, val_contexts, idx_val, np.take_along_axis(S_val, idx_val, axis=-1))

    best_lam, best_mse = 0.0, None
    for lam in LAMS:
        fused = (1 - lam) * y_chronos_val + lam * y_mem_val
        mse = float(((fused - val_futures) ** 2).mean())
        if best_mse is None or mse < best_mse:
            best_mse, best_lam = mse, lam

    S_test = (1 - ALPHA) * S_p_test + ALPHA * B_test
    idx_test = topk(S_test, K)
    y_mem_test = mean_anchor(bank, test_contexts, idx_test, np.take_along_axis(S_test, idx_test, axis=-1))
    fused_test = (1 - best_lam) * y_chronos_test + best_lam * y_mem_test

    chronos_mse = float(((y_chronos_test - test_futures) ** 2).mean())
    chronos_mae = float(np.abs(y_chronos_test - test_futures).mean())
    fused_mse = float(((fused_test - test_futures) ** 2).mean())
    fused_mae = float(np.abs(fused_test - test_futures).mean())
    return dict(name=name, lam=best_lam, chronos_mse=chronos_mse, chronos_mae=chronos_mae,
                fused_mse=fused_mse, fused_mae=fused_mae, n_test=len(test_starts), n_mem=bank.n_memories)


def main():
    print(f"Loading {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")

    rows = []
    for name, cfg in DATASETS.items():
        print(f"\n=== {name} ===")
        dates, ot = None, None
        df = pd.read_csv(f"data/{name}.csv")
        dates = pd.to_datetime(df["date"])
        ot = df["OT"].to_numpy(dtype=np.float32)
        r = process(name, cfg, dates, ot, pipeline)
        rows.append(r)
        print(f"  val-best lambda={r['lam']}  test_windows={r['n_test']}  memories={r['n_mem']}")
        print(f"  Chronos: MSE={r['chronos_mse']:.4f} MAE={r['chronos_mae']:.4f}")
        print(f"  Ours   : MSE={r['fused_mse']:.4f} MAE={r['fused_mae']:.4f}")

    print("\n=== SUMMARY ===")
    print(f"{'Dataset':>8} {'lambda':>7} {'Chronos MSE':>12} {'Ours MSE':>10} {'Chronos MAE':>12} {'Ours MAE':>10} {'improve%':>9}")
    for r in rows:
        imp = (r["chronos_mse"] - r["fused_mse"]) / r["chronos_mse"] * 100
        print(f"{r['name']:>8} {r['lam']:7.2f} {r['chronos_mse']:12.4f} {r['fused_mse']:10.4f} "
              f"{r['chronos_mae']:12.4f} {r['fused_mae']:10.4f} {imp:+9.2f}%")


if __name__ == "__main__":
    main()
