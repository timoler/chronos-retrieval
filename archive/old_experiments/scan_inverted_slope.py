"""Inverted (mean-reversion) level-shift estimator: delta_hat = -alpha * slope_168 * H.

y_corrected = y_chronos + (delta_hat - delta_chronos).  Scan alpha.
alpha=0 is a reference (delta_hat=0 -> re-center Chronos to recent level).
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96
H = 60
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    test_mask = (dates >= "2017-11-01").to_numpy()
    test_idx = np.flatnonzero(test_mask)
    return ot, int(test_idx[0]), int(test_idx[-1]) + 1


def slope(x):
    W = x.shape[1]
    t = np.arange(W, dtype=np.float64)
    A = np.vstack([t, np.ones(W)]).T
    coef = np.linalg.lstsq(A, x.T, rcond=None)[0]
    return coef[0]


def mse(y, f):
    return float(np.mean((y - f) ** 2))


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def main():
    ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)

    d = np.load("results/forecasts.npz")
    y_chronos = d["y_chronos"]

    recent_level = contexts[:, -24:].mean(axis=1)
    delta_true = futures.mean(axis=1) - recent_level
    delta_chronos = y_chronos.mean(axis=1) - recent_level

    s168 = slope(contexts[:, -168:])  # per-hour slope

    print(f"{'alpha':>7} {'MSE':>9} {'MAE':>8} {'corr(dh,dt)':>12}")
    print(f"{'raw':>7} {mse(y_chronos, futures):9.4f} {mae(y_chronos, futures):8.4f} {'-':>12}")
    for a in ALPHAS:
        dh = -a * s168 * H
        y = y_chronos + (dh - delta_chronos)[:, None]
        c = corr(dh, delta_true)
        cs = "-" if np.isnan(c) else f"{c:12.4f}"
        print(f"{a:7.2f} {mse(y, futures):9.4f} {mae(y, futures):8.4f} {cs:>12}")
    y_oracle = y_chronos + (delta_true - delta_chronos)[:, None]
    print(f"{'oracle':>7} {mse(y_oracle, futures):9.4f} {mae(y_oracle, futures):8.4f} {corr(delta_true, delta_true):12.4f}")


if __name__ == "__main__":
    main()
