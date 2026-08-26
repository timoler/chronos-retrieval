"""ACF of train OT: explain why Hour works, Weekday doesn't, Month does.

Computes autocorrelation at lag 24 / 168 / 720 (8760 not computable: train is
8760 long), and scans lag 7..2000 for the top-20 |ACF| lags.  No model change.
"""

import numpy as np
import pandas as pd

CONTEXT = 512


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    return ot[train_mask]


def acf(x, lag):
    a = x[: len(x) - lag].astype(np.float64)
    b = x[lag:].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def main():
    x = load("data/ETTh1.csv")
    N = len(x)
    print(f"train length = {N}")

    print("\n=== specified lags ===")
    for lag in [1, 24, 168, 720]:
        print(f"  ACF lag={lag:>5}: {acf(x, lag):+.4f}")
    print(f"  ACF lag=8760: not computable (train length {N}, would need 0 points)")

    # scan lag 7..2000
    lags = np.arange(7, 2001)
    vals = np.array([acf(x, l) for l in lags])

    top_idx = np.argsort(-np.abs(vals))[:20]
    print("\n=== Top-20 lags by |ACF| (lag >= 7) ===")
    for i in top_idx:
        print(f"  lag={lags[i]:>5}: {vals[i]:+.4f}")

    # local peak check for 24 / 168 / 720 (vs +/-2 neighbors)
    print("\n=== local peak check (vs +/-2 neighbors) ===")
    for target in [24, 168, 720]:
        v = acf(x, target)
        neighbors = [acf(x, target + d) for d in [-2, -1, 1, 2]]
        is_peak = all(v > n for n in neighbors)
        print(f"  lag={target}: ACF={v:+.4f}  neighbors={[f'{n:+.3f}' for n in neighbors]}  peak={is_peak}")


if __name__ == "__main__":
    main()
