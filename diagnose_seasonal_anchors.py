"""Long-period seasonal level anchors (no training, no test leakage).

Anchors are built from TRAIN only (2016-07..2017-06, the single full year):
  same-month       : mean level of the query's calendar month in train
  same-day-of-year : train daily mean by (month, day), smoothed +-7 days
  same-hour        : train mean by hour-of-day (diurnal)

delta_hat = anchor_level - recent_level, evaluated against delta_true
(corr / MSE / MAE), and per-|delta_true|-bin delta MAE vs Chronos's own
delta_chronos.  No forecast change, no fusion.
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, train, train_mask, int(test_idx[0]), int(test_idx[-1]) + 1


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float("nan") if denom == 0 else float((a * b).sum() / denom)


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
    dates, ot, train, train_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]
    train_vals = ot[train_mask].astype(np.float64)

    # --- build anchors from train only ---
    tdf = pd.DataFrame({
        "val": train_vals,
        "month": train_dates.dt.month,
        "day": train_dates.dt.day,
        "hour": train_dates.dt.hour,
    })
    monthly = tdf.groupby("month")["val"].mean().to_dict()            # month -> mean
    daily = tdf.groupby(["month", "day"])["val"].mean()               # (m,d) -> daily mean
    smoothed = daily.rolling(window=15, center=True, min_periods=1).mean()
    daily_smoothed = smoothed.to_dict()                               # {(m,d): smoothed}
    hourly = tdf.groupby("hour")["val"].mean().to_dict()              # hour -> mean

    # --- test queries ---
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float64)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float64)
    q_dates = dates.iloc[starts]

    recent_level = contexts[:, -24:].mean(axis=1)
    delta_true = futures.mean(axis=1) - recent_level

    y_chronos = np.load("results/forecasts.npz")["y_chronos"]
    delta_chronos = y_chronos.mean(axis=1) - recent_level

    q_month = q_dates.dt.month.to_numpy()
    q_day = q_dates.dt.day.to_numpy()
    q_hour = q_dates.dt.hour.to_numpy()

    anchors = {
        "same-month": np.array([monthly[m] for m in q_month]),
        "same-day-of-year": np.array([daily_smoothed[(m, d)] for m, d in zip(q_month, q_day)]),
        "same-hour": np.array([hourly[h] for h in q_hour]),
    }

    print("=== overall: who predicts delta_true? ===")
    print(f"  {'zero':20s} MSE={mse(np.zeros_like(delta_true), delta_true):8.4f}  MAE={mae(np.zeros_like(delta_true), delta_true):8.4f}")
    print(f"  {'delta_chronos':20s} corr={corr(delta_chronos, delta_true):+.4f}  MSE={mse(delta_chronos, delta_true):8.4f}  MAE={mae(delta_chronos, delta_true):8.4f}")
    for name, anchor in anchors.items():
        dh = anchor - recent_level
        print(f"  {name:20s} corr={corr(dh, delta_true):+.4f}  MSE={mse(dh, delta_true):8.4f}  MAE={mae(dh, delta_true):8.4f}")

    # --- per-bin delta MAE ---
    b = bins_by(np.abs(delta_true))
    print("\n=== per |delta_true| bin: delta MAE ===")
    header = f"{'bin':>4} {'n':>6} {'chronos':>10}"
    for name in anchors:
        header += f" {name:>16}"
    print(header)
    for i in range(5):
        m = b == i
        row = f"{i:>4} {int(m.sum()):>6} {mae(delta_chronos[m], delta_true[m]):10.4f}"
        for name, anchor in anchors.items():
            dh = anchor - recent_level
            row += f" {mae(dh[m], delta_true[m]):16.4f}"
        print(row)


if __name__ == "__main__":
    main()
