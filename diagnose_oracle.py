"""Level-Shift Oracle diagnostic (no model change, no training).

Given the under-estimation finding, ask: how much of Chronos's error is
level (mean) bias vs shape error?  The oracle perfectly re-centers each
window's forecast to the true future mean while keeping Chronos's shape:

  y_oracle = y_chronos + (delta_true - delta_chronos)

where delta_true / delta_chronos are the signed level shifts of the true
future / Chronos forecast relative to the last-24h context mean.  This uses
the true future mean, so it is an upper bound (cheating), isolating level
error from shape error.
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


def mse(y, f):
    return float(np.mean((y - f) ** 2))


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def main():
    ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    d = np.load("results/forecasts.npz")
    y_chronos = d["y_chronos"]

    recent_level = contexts[:, -24:].mean(axis=1)
    delta_true = futures.mean(axis=1) - recent_level
    delta_chronos = y_chronos.mean(axis=1) - recent_level
    y_oracle = y_chronos + (delta_true - delta_chronos)[:, None]

    chronos_mse = mse(y_chronos, futures)
    chronos_mae = mae(y_chronos, futures)
    oracle_mse = mse(y_oracle, futures)
    oracle_mae = mae(y_oracle, futures)

    print("=== overall ===")
    print(f"  Chronos: MSE={chronos_mse:.4f}  MAE={chronos_mae:.4f}")
    print(f"  Oracle : MSE={oracle_mse:.4f}  MAE={oracle_mae:.4f}")
    print(f"  MSE improvement: {(chronos_mse - oracle_mse) / chronos_mse * 100:.2f}%")

    b = bins_by(np.abs(delta_true))
    print("\n=== bins by |delta_true| (quintiles) ===")
    print(f"{'bin':>4} {'n':>6} {'raw_MSE':>10} {'oracle_MSE':>11} {'improve%':>9}")
    for i in range(5):
        m = b == i
        raw = mse(y_chronos[m], futures[m])
        ora = mse(y_oracle[m], futures[m])
        imp = (raw - ora) / raw * 100
        print(f"{i:>4} {int(m.sum()):>6} {raw:10.4f} {ora:11.4f} {imp:+9.2f}%")


if __name__ == "__main__":
    main()
