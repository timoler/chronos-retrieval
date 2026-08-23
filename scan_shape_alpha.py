"""Scan the shape-term scale alpha in the Mean-anchor transfer.

  y_mem(alpha) = mean(x_q) + alpha * sum_m w_m * (x_fut,m - mean(x_m))

alpha=0 must reproduce pure-mean MSE ~7.5075; alpha=1 must reproduce the
current memory MSE ~18.0965.  No Chronos fusion, Retriever unchanged.
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    test_mask = (dates >= "2017-11-01").to_numpy()
    test_idx = np.flatnonzero(test_mask)
    return ot, int(test_idx[0]), int(test_idx[-1]) + 1


def main():
    ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    d = np.load("results/forecasts.npz")
    y_mem = d["y_mem"]  # alpha = 1 (current memory forecast)

    mean_q = contexts.mean(axis=1).astype(np.float64)[:, None]  # (n, 1)
    shape = y_mem - mean_q                                    # (n, 96), alpha=1 shape

    print(f"{'alpha':>6} {'Memory MSE':>12} {'Memory MAE':>12}")
    for a in ALPHAS:
        y = mean_q + a * shape
        mse = float(np.mean((y - futures) ** 2))
        mae = float(np.mean(np.abs(y - futures)))
        print(f"{a:6.2f} {mse:12.4f} {mae:12.4f}")


if __name__ == "__main__":
    main()
