"""Regime feasibility diagnostic: KMeans on past-512 context features.

Features (from the past 512 context points only, no future):
  level       = mean(x)                                  [512-point mean]
  trend       = slope of least-squares line over x      [per-hour]
  volatility  = std(x)                                  [population std, ddof=0]
  seasonality = lag-24 autocorrelation = Pearson(x[0:488], x[24:512])

Features are z-score standardized, then KMeans (K=4, fixed seed).  Report per
regime: n, mean of the 4 raw features, Chronos MSE, Chronos MAE.  No gate /
correction, no future info in clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CONTEXT, PRED = 512, 96
K_CLUSTERS = 4


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


def lag24_autocorr(x):
    """Pearson(x[:, :-24], x[:, 24:]) per row."""
    a = x[:, :-24]
    b = x[:, 24:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return num / np.where(den == 0, 1.0, den)


def main():
    ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    level = contexts.mean(axis=1)
    trend = slope(contexts)
    volatility = contexts.std(axis=1)
    seasonality = lag24_autocorr(contexts)

    features = np.stack([level, trend, volatility, seasonality], axis=1)
    feat_names = ["level", "trend", "volatility", "seasonality"]

    mu = features.mean(axis=0)
    sd = features.std(axis=0)
    Z = (features - mu) / sd

    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10).fit(Z)
    labels = km.labels_

    chronos_mse = ((y_chronos - futures) ** 2).mean(axis=1)
    chronos_mae = np.abs(y_chronos - futures).mean(axis=1)

    print(f"overall: Chronos MSE = {chronos_mse.mean():.4f}   MAE = {chronos_mae.mean():.4f}")
    print(f"\n{'regime':>6} {'n':>6} " + " ".join(f"{f:>12}" for f in feat_names)
          + f" {'chronos_MSE':>12} {'chronos_MAE':>12}")
    for k in range(K_CLUSTERS):
        m = labels == k
        row = f"{k:>6} {int(m.sum()):>6} "
        row += " ".join(f"{features[m, i].mean():12.4f}" for i in range(4))
        row += f" {chronos_mse[m].mean():12.4f} {chronos_mae[m].mean():12.4f}"
        print(row)


if __name__ == "__main__":
    main()
