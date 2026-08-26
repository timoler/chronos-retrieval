"""Trend-sign x Month diagnostic.

Split windows by the past-512 trend sign (up = slope > 0, down = slope <= 0),
then report Chronos MSE overall and per month.  Tests whether trend direction
alone identifies Chronos's hard/easy states (R0 hard vs R2 easy), or whether
the full multi-feature regime is needed.
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96
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


def main():
    dates, ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    trend = slope(contexts)
    chronos_mse = ((y_chronos - futures) ** 2).mean(axis=1)
    months = dates.iloc[starts].dt.month.to_numpy()

    up = trend > 0
    down = ~up

    print("=== overall ===")
    print(f"  Up   (trend > 0) : n={int(up.sum()):>5}  mean_trend={trend[up].mean():+.5f}  MSE={chronos_mse[up].mean():.4f}")
    print(f"  Down (trend <= 0): n={int(down.sum()):>5}  mean_trend={trend[down].mean():+.5f}  MSE={chronos_mse[down].mean():.4f}")

    print("\n=== per month ===")
    print(f"{'month':>6} {'Up n':>6} {'Up MSE':>9} {'Down n':>7} {'Down MSE':>9}")
    for mm in MONTHS:
        m = months == mm
        up_n = int((m & up).sum())
        down_n = int((m & down).sum())
        up_mse = chronos_mse[m & up].mean() if up_n > 0 else float("nan")
        down_mse = chronos_mse[m & down].mean() if down_n > 0 else float("nan")
        up_s = f"{up_mse:9.4f}" if up_n > 0 else f"{'--':>9}"
        down_s = f"{down_mse:9.4f}" if down_n > 0 else f"{'--':>9}"
        print(f"{f'm{mm:02d}':>6} {up_n:>6} {up_s} {down_n:>7} {down_s}")


if __name__ == "__main__":
    main()
