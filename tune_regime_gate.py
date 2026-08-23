"""Regime-aware two-tier gate: tune (lambda_up, lambda_down) on VAL, apply to TEST.

Fixed: alpha_time=0.5 time-aligned retrieval + Mean-anchor memory.
Gate: trend>0 -> lambda_up, trend<=0 -> lambda_down.
Grid on VAL only; best combo applied to TEST (no test tuning).
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
PERIOD, TREND_WINDOW = 24, 48
ALPHA_TIME = 0.5
MODEL = "amazon/chronos-2"
BATCH = 256
LAM_UP_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
LAM_DOWN_GRID = [0.0, 0.05, 0.1, 0.15, 0.2]


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, train, train_mask, val_mask, int(test_idx[0]), int(test_idx[-1]) + 1


def slope(x):
    W = x.shape[1]
    t = np.arange(W, dtype=np.float64)
    A = np.vstack([t, np.ones(W)]).T
    return np.linalg.lstsq(A, x.T, rcond=None)[0][0]


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


def time_memory(contexts, q_dates, bank, mem_dates, luts):
    month_lut, dow_lut, hour_lut = luts
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    S_p = pearson_similarity_matrix(R_q, R_m).astype(np.float32)

    align = (month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
             + dow_lut[q_dates.dt.dayofweek.to_numpy()][:, mem_dates.dt.dayofweek.to_numpy()]
             + hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]) / 3.0
    B_raw = 2.0 * align - 1.0
    B_time = (B_raw - B_raw.mean()) / B_raw.std() * S_p.std() + S_p.mean()

    S = (1 - ALPHA_TIME) * S_p + ALPHA_TIME * B_time
    idx = topk(S, K)
    sc = np.take_along_axis(S, idx, axis=-1)
    w = softmax(sc)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def windows(ot, start, end):
    s = np.arange(start, end - PRED + 1)
    c = np.stack([ot[i - CONTEXT : i] for i in s]).astype(np.float32)
    f = np.stack([ot[i : i + PRED] for i in s]).astype(np.float32)
    return s, c, f


def main():
    dates, ot, train, train_mask, val_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]
    bank = MemoryBank.from_series(train, CONTEXT, PRED)
    mem_dates = train_dates.iloc[np.arange(bank.n_memories) + CONTEXT]
    luts = (circular_lut(12), circular_lut(7), circular_lut(24))

    val_idx = np.flatnonzero(val_mask)
    val_start, val_end = int(val_idx[0]), int(val_idx[-1]) + 1

    # --- TEST (chronos from saved npz) ---
    test_starts, test_contexts, test_futures = windows(ot, test_start, test_end)
    test_dates = dates.iloc[test_starts]
    y_chronos_test = np.load("results/forecasts.npz")["y_chronos"]
    y_mem_test = time_memory(test_contexts, test_dates, bank, mem_dates, luts)
    test_trend = slope(test_contexts)

    # --- VAL (need Chronos on val) ---
    val_starts, val_contexts, val_futures = windows(ot, val_start, val_end)
    val_dates = dates.iloc[val_starts]
    print(f"Loading {MODEL} for val ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    inputs_val = [torch.tensor(c, dtype=torch.float32) for c in val_contexts]
    _, means_val = pipeline.predict_quantiles(inputs_val, prediction_length=PRED, batch_size=BATCH)
    y_chronos_val = np.stack([m.squeeze(0).numpy() for m in means_val])
    y_mem_val = time_memory(val_contexts, val_dates, bank, mem_dates, luts)
    val_trend = slope(val_contexts)
    val_up = val_trend > 0

    val_chronos_mse = float(((y_chronos_val - val_futures) ** 2).mean())

    # grid search on val
    best = None
    for lu in LAM_UP_GRID:
        for ld in LAM_DOWN_GRID:
            lam = np.where(val_up, lu, ld)
            fused = (1 - lam[:, None]) * y_chronos_val + lam[:, None] * y_mem_val
            mse = float(((fused - val_futures) ** 2).mean())
            if best is None or mse < best[2]:
                best = (lu, ld, mse)
    lu_best, ld_best, val_best_mse = best

    print(f"\nVAL: chronos_MSE={val_chronos_mse:.4f}")
    print(f"VAL best gate: lambda_up={lu_best}, lambda_down={ld_best}  -> val MSE={val_best_mse:.4f}")

    # apply to test
    test_up = test_trend > 0
    lam_test = np.where(test_up, lu_best, ld_best)
    fused_test = (1 - lam_test[:, None]) * y_chronos_test + lam_test[:, None] * y_mem_test
    chronos_mse = ((y_chronos_test - test_futures) ** 2).mean(axis=1)
    fused_mse = ((fused_test - test_futures) ** 2).mean(axis=1)

    print("\n=== TEST (best val gate applied) ===")
    print(f"overall: Chronos MSE={chronos_mse.mean():.4f}  Fusion MSE={fused_mse.mean():.4f}")
    print(f"overall MAE: Chronos={np.abs(y_chronos_test - test_futures).mean():.4f}  "
          f"Fusion={np.abs(fused_test - test_futures).mean():.4f}")
    print(f"\n{'Regime':>6} {'n':>6} {'Chronos MSE':>12} {'Fusion MSE':>11} {'improve%':>9}")
    for name, mask in [("Up", test_up), ("Down", ~test_up)]:
        n = int(mask.sum())
        c = chronos_mse[mask].mean()
        f = fused_mse[mask].mean()
        print(f"{name:>6} {n:>6} {c:12.4f} {f:11.4f} {(c - f) / c * 100:+9.2f}%")

    np.savez("results/val_forecasts.npz", y_chronos_val=y_chronos_val, y_mem_val=y_mem_val,
             val_futures=val_futures, val_trend=val_trend)
    print("\nsaved results/val_forecasts.npz")


if __name__ == "__main__":
    main()
