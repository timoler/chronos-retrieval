"""Diagnose how Chronos responds to future level shifts (signed deltas).

Per window:
  recent_level   = mean(last 24 context points)
  delta_true     = mean(y_true_future) - recent_level
  delta_chronos  = mean(y_chronos_future) - recent_level

Then: bin by |delta_true| (quintiles) and report signed + absolute delta means;
measure sign agreement; and the amplitude ratio r = |delta_chronos| / |delta_true|
in the high-drift bin.  Saves a scatter plot (delta_true vs delta_chronos) with
y=x and y=0 reference lines.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONTEXT, PRED = 512, 96
EPS = 1e-6


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

    abs_dt = np.abs(delta_true)
    abs_dc = np.abs(delta_chronos)
    b = bins_by(abs_dt)

    # 1. per-bin signed + absolute means
    print("=== bins by |delta_true| (quintiles) ===")
    print(f"{'bin':>4} {'n':>6} {'mean(dt)':>10} {'mean(dc)':>10} "
          f"{'mean|dt|':>10} {'mean|dc|':>10}")
    for i in range(5):
        m = b == i
        print(f"{i:>4} {int(m.sum()):>6} {delta_true[m].mean():10.4f} "
              f"{delta_chronos[m].mean():10.4f} {abs_dt[m].mean():10.4f} "
              f"{abs_dc[m].mean():10.4f}")

    # 2. sign agreement
    same = float((np.sign(delta_true) == np.sign(delta_chronos)).mean())
    high = b == 4
    same_high = float((np.sign(delta_true[high]) == np.sign(delta_chronos[high])).mean())
    print(f"\nsign(delta_true) == sign(delta_chronos):")
    print(f"  overall            = {same * 100:.2f}%")
    print(f"  high-drift bin     = {same_high * 100:.2f}%")

    # 3. amplitude ratio r, per bin (median robust) and high-drift detail
    r = abs_dc / (abs_dt + EPS)
    print(f"\namplitude ratio r = |delta_chronos| / |delta_true|:")
    for i in range(5):
        m = b == i
        print(f"  bin {i}: mean={r[m].mean():6.3f}  median={np.median(r[m]):6.3f}")
    print(f"  high-drift bin: mean={r[high].mean():.3f}  median={np.median(r[high]):.3f}")

    # 4. scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(delta_true, delta_chronos, s=4, alpha=0.3, color="tab:blue")
    lim = max(abs_dt.max(), abs_dc.max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y=x")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("delta_true  (mean future - recent level)")
    ax.set_ylabel("delta_chronos")
    ax.set_title("delta_true vs delta_chronos")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig("results/delta_scatter.png", dpi=110)
    print("\nsaved results/delta_scatter.png")


if __name__ == "__main__":
    main()
