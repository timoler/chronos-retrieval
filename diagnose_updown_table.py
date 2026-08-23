"""The key combined table: Up/Down regime x Chronos / Time-Memory / Fusion.

Fixed: alpha_time=0.5, lambda=0.1.  Time-Memory = time-aligned retrieval
(residual Pearson + calendar B_time, alpha=0.5) + Mean-anchor transfer.
Split windows by past-512 trend sign (Up = slope > 0, Down = slope <= 0).
"""

import numpy as np
import pandas as pd

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
PERIOD, TREND_WINDOW = 24, 48
ALPHA_TIME = 0.5
LAM = 0.1


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, train, train_mask, int(test_idx[0]), int(test_idx[-1]) + 1


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


def main():
    dates, ot, train, train_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    S_pearson = pearson_similarity_matrix(R_q, R_m).astype(np.float32)

    month_lut, dow_lut, hour_lut = circular_lut(12), circular_lut(7), circular_lut(24)
    q_dates = dates.iloc[starts]
    mem_dates = train_dates.iloc[np.arange(bank.n_memories) + CONTEXT]
    align = (month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
             + dow_lut[q_dates.dt.dayofweek.to_numpy()][:, mem_dates.dt.dayofweek.to_numpy()]
             + hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]) / 3.0
    B_raw = 2.0 * align - 1.0
    B_time = (B_raw - B_raw.mean()) / B_raw.std() * S_pearson.std() + S_pearson.mean()

    S = (1 - ALPHA_TIME) * S_pearson + ALPHA_TIME * B_time
    idx = topk(S, K)
    sc = np.take_along_axis(S, idx, axis=-1)
    w = softmax(sc)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    y_mem = contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)
    fused = (1 - LAM) * y_chronos + LAM * y_mem

    chronos_mse = ((y_chronos - futures) ** 2).mean(axis=1)
    mem_mse = ((y_mem - futures) ** 2).mean(axis=1)
    fused_mse = ((fused - futures) ** 2).mean(axis=1)

    trend = slope(contexts)
    up = trend > 0
    down = ~up

    print(f"(alpha_time={ALPHA_TIME}, lambda={LAM})")
    print(f"{'Regime':>6} {'n':>6} {'Chronos MSE':>12} {'TimeMem MSE':>12} "
          f"{'Fusion MSE':>11} {'improve%':>9}")
    for name, mask in [("Up", up), ("Down", down)]:
        n = int(mask.sum())
        c = chronos_mse[mask].mean()
        m = mem_mse[mask].mean()
        f = fused_mse[mask].mean()
        imp = (c - f) / c * 100
        print(f"{name:>6} {n:>6} {c:12.4f} {m:12.4f} {f:11.4f} {imp:+9.2f}%")


if __name__ == "__main__":
    main()
