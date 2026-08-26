"""Regime generalization check: train-learned KMeans applied to test.

Fit KMeans (K=4) on TRAIN window features (level/trend/volatility/seasonality,
standardized with TRAIN stats), then freeze and assign TEST windows.  Report
test Chronos MSE/MAE per regime.  If the frozen regime still separates Chronos
difficulty on test, the regime is a real signal; if all regimes collapse to
~5-7, the earlier separation was an artifact of test-side clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from retrieval.memory_bank import MemoryBank

CONTEXT, PRED = 512, 96
K = 4


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


def lag24_autocorr(x):
    a = x[:, :-24]
    b = x[:, 24:]
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return num / np.where(den == 0, 1.0, den)


def features(contexts):
    return np.stack([
        contexts.mean(axis=1),
        slope(contexts),
        contexts.std(axis=1),
        lag24_autocorr(contexts),
    ], axis=1)


def main():
    dates, ot, train, train_mask, test_start, test_end = load("data/ETTh1.csv")
    bank = MemoryBank.from_series(train, CONTEXT, PRED)  # train windows = bank contexts

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts_test = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures_test = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    F_train = features(bank.contexts.astype(np.float64))   # (8153, 4)
    F_test = features(contexts_test)                        # (5613, 4)

    mu = F_train.mean(axis=0)
    sd = F_train.std(axis=0)
    Z_train = (F_train - mu) / sd
    Z_test = (F_test - mu) / sd                              # TRAIN stats for test

    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Z_train)
    labels_test = km.predict(Z_test)

    chronos_mse = ((y_chronos - futures_test) ** 2).mean(axis=1)
    chronos_mae = np.abs(y_chronos - futures_test).mean(axis=1)

    feat_names = ["level", "trend", "volatility", "seasonality"]
    print(f"overall test Chronos MSE = {chronos_mse.mean():.4f}  (expect 6.1968)")
    print(f"\n{'regime':>6} {'n':>6} " + " ".join(f"{f:>12}" for f in feat_names)
          + f" {'chronos_MSE':>12} {'chronos_MAE':>12}")
    for k in range(K):
        m = labels_test == k
        row = f"{k:>6} {int(m.sum()):>6} "
        row += " ".join(f"{F_test[m, i].mean():12.4f}" for i in range(4))
        row += f" {chronos_mse[m].mean():12.4f} {chronos_mae[m].mean():12.4f}"
        print(row)

    # also report train-side regime MSE (for reference)
    print("\n(train-side regime MSE, reference):")
    # need train chronos -> not saved; skip, just show cluster sizes
    for k in range(K):
        print(f"  regime {k}: train n = {int((km.labels_ == k).sum())}")


if __name__ == "__main__":
    main()
