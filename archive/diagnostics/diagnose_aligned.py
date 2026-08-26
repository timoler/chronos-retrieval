"""Test the hypothesis: Chronos is harder on trend-aligned-with-season windows.

seasonalTrend[month] = mean 512-slope of train windows ending in that month
(train only, no test leakage).  For a test query, product = trend_q * seasonal,
  Aligned = product > 0,  Opposed = product < 0.
Report Chronos MSE overall and per month.  No correction, no module.
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96
MONTHS = [11, 12, 1, 2, 3, 4, 5, 6]


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


def main():
    dates, ot, train, train_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]

    # seasonal trend from train: mean 512-slope by month of window end
    ws = np.arange(CONTEXT, len(train))
    train_ctx = np.stack([train[s - CONTEXT : s] for s in ws]).astype(np.float64)
    train_trend = slope(train_ctx)
    train_months = train_dates.iloc[ws].dt.month.to_numpy()
    seasonal_trend = {m: float(train_trend[train_months == m].mean()) for m in range(1, 13)}

    print("seasonal trend (mean 512-slope) by month [from train]:")
    for m in range(1, 13):
        print(f"  m{m:02d}: {seasonal_trend[m]:+.5f}")

    # test queries
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    trend_q = slope(contexts)
    chronos_mse = ((y_chronos - futures) ** 2).mean(axis=1)
    months = dates.iloc[starts].dt.month.to_numpy()

    seasonal = np.array([seasonal_trend[m] for m in months])
    product = trend_q * seasonal
    aligned = product > 0
    opposed = product < 0

    print("\n=== overall ===")
    print(f"  Aligned : n={int(aligned.sum()):>5}  MSE={chronos_mse[aligned].mean():.4f}")
    print(f"  Opposed : n={int(opposed.sum()):>5}  MSE={chronos_mse[opposed].mean():.4f}")

    print("\n=== per month ===")
    print(f"{'month':>6} {'Aligned n':>9} {'Aligned MSE':>11} {'Opposed n':>9} {'Opposed MSE':>11}")
    for mm in MONTHS:
        m = months == mm
        an = int((m & aligned).sum())
        on = int((m & opposed).sum())
        amse = chronos_mse[m & aligned].mean() if an > 0 else float("nan")
        omse = chronos_mse[m & opposed].mean() if on > 0 else float("nan")
        as_ = f"{amse:11.4f}" if an > 0 else f"{'--':>11}"
        os_ = f"{omse:11.4f}" if on > 0 else f"{'--':>11}"
        print(f"{f'm{mm:02d}':>6} {an:>9} {as_} {on:>9} {os_}")


if __name__ == "__main__":
    main()
