"""Diagnose Chronos's error vs future characteristics (no model change).

Two per-window signals:
  D (future variation) = mean |first-difference of future|, anchored to the
      last context point (96 differences: last-context -> future_0, and the
      95 internal differences).
  L (future level shift) = |mean(future 96) - mean(last 24 context points)|.

Bin each by quintiles and report n, signal mean, Chronos MSE, Chronos MAE.
Saves D, L, per-window chronos MSE/MAE for later analysis.
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    test_mask = (dates >= "2017-11-01").to_numpy()
    test_idx = np.flatnonzero(test_mask)
    return ot, int(test_idx[0]), int(test_idx[-1]) + 1


def bins_by(signal, n_bins=5):
    edges = np.quantile(signal, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(signal, edges) - 1


def report(title, signal, chronos_mse, chronos_mae):
    b = bins_by(signal)
    print(f"\n=== bins by {title} (quintiles) ===")
    print(f"{'bin':>4} {'n':>6} {'signal_mean':>12} {'chronos_MSE':>12} {'chronos_MAE':>12}")
    for i in range(5):
        m = b == i
        print(f"{i:>4} {int(m.sum()):>6} {signal[m].mean():12.4f} "
              f"{chronos_mse[m].mean():12.4f} {chronos_mae[m].mean():12.4f}")


def main():
    ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    d = np.load("results/forecasts.npz")
    y_chronos = d["y_chronos"]

    chronos_mse = ((y_chronos - futures) ** 2).mean(axis=1)
    chronos_mae = np.abs(y_chronos - futures).mean(axis=1)

    # D: mean abs first-difference, anchored to the last context point
    full_seq = np.concatenate([ot[starts - 1][:, None], futures], axis=1)  # (n, 97)
    D = np.abs(np.diff(full_seq, axis=1)).mean(axis=1)                      # (n,)

    # L: |mean(future) - mean(last 24 context)|
    L = np.abs(futures.mean(axis=1) - contexts[:, -24:].mean(axis=1))

    report("D (future variation)", D, chronos_mse, chronos_mae)
    report("L (future level shift)", L, chronos_mse, chronos_mae)

    np.savez(
        "results/future_stats.npz",
        D=D, L=L, chronos_mse=chronos_mse, chronos_mae=chronos_mae,
    )
    print("\nsaved results/future_stats.npz (D, L, chronos_mse, chronos_mae)")


if __name__ == "__main__":
    main()
