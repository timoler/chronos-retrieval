"""Compare level-shift estimators directly on delta_true (no forecast change).

  delta_chronos : Chronos's own level-shift estimate
  delta_slope   : reversed slope, -alpha * slope_168 * H  (alpha=0.75)

Report corr / MSE / MAE of each vs delta_true, plus per-|delta_true|-bin
delta MAE.  No new estimator, no fusion.
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96
H = 60
ALPHA = 0.75


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


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def bins_by(signal, n_bins=5):
    edges = np.quantile(signal, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(signal, edges) - 1


def main():
    ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)

    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    recent_level = contexts[:, -24:].mean(axis=1)
    delta_true = futures.mean(axis=1) - recent_level
    delta_chronos = y_chronos.mean(axis=1) - recent_level
    delta_slope = -ALPHA * slope(contexts[:, -168:]) * H

    print("=== overall: who predicts delta_true better? ===")
    print(f"  zero (predict no shift): MSE={mse(np.zeros_like(delta_true), delta_true):.4f}  MAE={mae(np.zeros_like(delta_true), delta_true):.4f}  (reference)")
    for name, dh in [("delta_chronos", delta_chronos), ("reversed_slope(0.75)", delta_slope)]:
        print(f"  {name:20s}: corr={corr(dh, delta_true):+.4f}  MSE={mse(dh, delta_true):.4f}  MAE={mae(dh, delta_true):.4f}")

    b = bins_by(np.abs(delta_true))
    print("\n=== per |delta_true| bin: delta MAE ===")
    print(f"{'bin':>4} {'n':>6} {'chronos_MAE':>12} {'slope_MAE':>11}")
    for i in range(5):
        m = b == i
        print(f"{i:>4} {int(m.sum()):>6} {mae(delta_chronos[m], delta_true[m]):12.4f} "
              f"{mae(delta_slope[m], delta_true[m]):11.4f}")


if __name__ == "__main__":
    main()
