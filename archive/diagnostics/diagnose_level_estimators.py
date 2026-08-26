"""Simple context-only level-shift estimators (no training).

For each estimator delta_hat, apply
    y_final = y_chronos + (delta_hat - delta_chronos)
and report MSE / MAE / corr(delta_hat, delta_true).

Estimators (delta_hat = predicted future level shift):
  day_diff     : mean(x[-24:]) - mean(x[-48:-24])
  2day_diff    : mean(x[-48:]) - mean(x[-96:-48])
  week_diff    : mean(x[-168:]) - mean(x[-336:-168])
  slope_168    : slope(last 168h) * H
  multi_slope  : mean(slope_24, slope_48, slope_168) * H

Note: "same-hour yesterday" == day_diff (mean over hours of pointwise diff
equals mean difference), so it is not listed separately.

H = 60: last-24h midpoint (-12h) to future midpoint (+48h).
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96
H = 60


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    test_mask = (dates >= "2017-11-01").to_numpy()
    test_idx = np.flatnonzero(test_mask)
    return ot, int(test_idx[0]), int(test_idx[-1]) + 1


def slope(x):
    """Per-row least-squares slope over the last axis."""
    W = x.shape[1]
    t = np.arange(W, dtype=np.float64)
    A = np.vstack([t, np.ones(W)]).T
    coef = np.linalg.lstsq(A, x.T, rcond=None)[0]  # (2, n)
    return coef[0]  # per-hour slope


def mse(y, f):
    return float(np.mean((y - f) ** 2))


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


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

    estimators = {
        "day_diff (24v24)": contexts[:, -24:].mean(axis=1) - contexts[:, -48:-24].mean(axis=1),
        "2day_diff (48v48)": contexts[:, -48:].mean(axis=1) - contexts[:, -96:-48].mean(axis=1),
        "week_diff (168v168)": contexts[:, -168:].mean(axis=1) - contexts[:, -336:-168].mean(axis=1),
        "slope_168 * H": slope(contexts[:, -168:]) * H,
        "multi_slope * H": (slope(contexts[:, -24:]) + slope(contexts[:, -48:]) + slope(contexts[:, -168:])) / 3 * H,
    }

    def apply(delta_hat):
        y_final = y_chronos + (delta_hat - delta_chronos)[:, None]
        return y_final

    print(f"{'method':>22s} {'MSE':>9} {'MAE':>8} {'corr(dh,dt)':>12}")
    print("-" * 55)
    print(f"{'Chronos raw':>22s} {mse(y_chronos, futures):9.4f} {mae(y_chronos, futures):8.4f} {'-':>12}")
    y_oracle = apply(delta_true)
    print(f"{'Oracle (upper)':>22s} {mse(y_oracle, futures):9.4f} {mae(y_oracle, futures):8.4f} {corr(delta_true, delta_true):12.4f}")
    for name, dh in estimators.items():
        y_final = apply(dh)
        print(f"{name:>22s} {mse(y_final, futures):9.4f} {mae(y_final, futures):8.4f} {corr(dh, delta_true):12.4f}")


if __name__ == "__main__":
    main()
