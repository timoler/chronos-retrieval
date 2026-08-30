"""Preprocess NREL Solar-Energy -> data/solar.csv (OT aggregate + 6 covariates).

Lineage (documented for the dataset registry):
    NREL Solar-Energy (137 PV plants, 10-min, year 2006)
    -> laiguokun/multivariate-time-series-data  (Autoformer mirror)
       solar-energy/solar_AL.txt.gz  (raw.githubusercontent via jsDelivr CDN)
    -> data/_raw/solar_AL.txt        (137 numeric columns, no header, no date)
    -> data/solar.csv                (date, OT, P001..P006)

Structure mirrors the electricity negative case exactly:
    electricity:  OT = sum(314 consumers),   X1..X6 = first 6 consumers
    solar:        OT = sum(137 plants),      X1..X6 = first 6 plants (P001..P006)

No representativeness selection is performed -- P001..P006 are literally the
first six plant columns in file order, matching electricity's `keep[:6]`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RAW = "data/_raw/solar_AL.txt"
OUT = "data/solar.csv"


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    raw = np.loadtxt(RAW, delimiter=",")  # (52560, 137)
    n, n_plants = raw.shape
    print(f"raw shape = {raw.shape}  (n_timesteps={n}, n_plants={n_plants})")

    start = datetime(2006, 1, 1, 0, 0, 0)
    dates = [start + timedelta(minutes=10 * i) for i in range(n)]

    ot = raw.sum(axis=1)                      # aggregate total solar output
    six = [raw[:, i].astype(np.float32) for i in range(6)]

    df = pd.DataFrame({
        "date": dates,
        "OT": ot,
        "P001": six[0], "P002": six[1], "P003": six[2],
        "P004": six[3], "P005": six[4], "P006": six[5],
    })
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  shape={df.shape}")

    # ---- TRAIN-only evidence (first 60%), to write honest metadata ----
    train_len = int(0.6 * n)
    y = ot[:train_len].astype(np.float64)
    s6 = sum(series[:train_len].astype(np.float64) for series in six)
    print(f"\nTRAIN-only evidence (n_train={train_len}):")
    print(f"  corr(S6, Y)          = {_corr(s6, y):.4f}")
    print(f"  corr(diff S6, diff Y)= {_corr(np.diff(s6), np.diff(y)):.4f}")
    print(f"  S6/Y ratio mean      = {float(np.mean(s6 / y)):.4f}")
    print(f"  coverage             = 6/{n_plants} = {6 / n_plants:.4f}")
    # same-clock prev-day (10-min: 144 samples/day) on Y
    print(f"  Y lag-1 autocorr     = {_corr(y[1:], y[:-1]):.4f}")
    print(f"  Y same-clock prev-day= {_corr(y[144:], y[:-144]):.4f}")


if __name__ == "__main__":
    main()
