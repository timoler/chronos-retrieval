"""Confounder check: is the regime MSE separation just month/season?

Recomputes the same KMeans regimes, then reports:
  1. regime -> month count distribution
  2. month -> regime mean Chronos MSE (with counts)
so we can see whether the regime effect on MSE survives within each month.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CONTEXT, PRED = 512, 96
K = 4
MONTHS = [11, 12, 1, 2, 3, 4, 5, 6]


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    test_mask = (dates >= "2017-11-01").to_numpy()
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, int(test_idx[0]), int(test_idx[-1]) + 1


def slope(x):
    W = x.shape[1]
    t = np.arange(W, dtype=np.float64)
    A = np.vstack([t, np.ones(W)]).T
    return np.linalg.lstsq(A, x.T, rcond=None)[0][0]


def lag24_autocorr(x):
    a = x[:, :-24]
    b = x[:, 24:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return num / np.where(den == 0, 1.0, den)


def main():
    dates, ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    F = np.stack([contexts.mean(axis=1), slope(contexts),
                  contexts.std(axis=1), lag24_autocorr(contexts)], axis=1)
    Z = (F - F.mean(axis=0)) / F.std(axis=0)
    labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z).labels_

    chronos_mse = ((y_chronos - futures) ** 2).mean(axis=1)
    months = dates.iloc[starts].dt.month.to_numpy()

    print("=== regime -> month counts ===")
    print(f"{'regime':>6} " + " ".join(f"{f'm{m:02d}':>6}" for m in MONTHS) + f" {'total':>7}")
    for k in range(K):
        m = labels == k
        counts = [int((months[m] == mm).sum()) for mm in MONTHS]
        print(f"{k:>6} " + " ".join(f"{c:>6}" for c in counts) + f" {int(m.sum()):>7}")

    print("\n=== month -> regime: mean Chronos MSE (n) ===")
    print(f"{'month':>6} " + " ".join(f"{f'r{k}':>14}" for k in range(K)))
    for mm in MONTHS:
        m = months == mm
        cells = []
        for k in range(K):
            sel = m & (labels == k)
            if sel.sum() == 0:
                cells.append("            --")
            else:
                cells.append(f"{chronos_mse[sel].mean():8.3f}({int(sel.sum()):<4})")
        print(f"{f'm{mm:02d}':>6} " + " ".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
