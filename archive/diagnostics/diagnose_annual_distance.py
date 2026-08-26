"""Annual-phase distance vs memory level-shift error.

For every (query, memory) pair, compute the circular day-of-year distance and
the level-shift mismatch |delta_m - delta_q|, where delta = mean(future) -
mean(context).  The mean-anchor forecast's level bias equals delta_m - delta_q,
so |delta_m - delta_q| is the level component of the memory error.  Bin by
annual distance and see whether error grows monotonically.
"""

import numpy as np
import pandas as pd

from retrieval.memory_bank import MemoryBank

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


def main():
    dates, ot, train, train_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)
    q_dates = dates.iloc[starts]

    q_doy = q_dates.dt.dayofyear.to_numpy()
    delta_q = futures.mean(axis=1) - contexts.mean(axis=1)          # (5613,)

    mem_dates = train_dates.iloc[np.arange(bank.n_memories) + CONTEXT]
    m_doy = mem_dates.dt.dayofyear.to_numpy()
    delta_m = bank.futures.mean(axis=1) - bank.contexts.mean(axis=1)  # (8153,)

    d = np.abs(q_doy[:, None] - m_doy[None, :]).astype(np.int16)
    dist = np.minimum(d, 365 - d)                                    # circular day distance
    lerr = np.abs(delta_q[:, None] - delta_m[None, :]).astype(np.float32)

    print("=== annual distance vs |delta_m - delta_q| (level-shift error) ===")
    bins = [(0, 7), (7, 15), (15, 30), (30, 45), (45, 60), (60, 90),
            (90, 120), (120, 150), (150, 182)]
    print(f"{'DOY dist (days)':>16} {'n pairs':>12} {'mean level err':>15}")
    for lo, hi in bins:
        mask = (dist >= lo) & (dist < hi)
        print(f"{f'{lo}-{hi}':>16} {int(mask.sum()):>12} {lerr[mask].mean():>15.4f}")


if __name__ == "__main__":
    main()
