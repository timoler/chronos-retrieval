"""Diagnose Top-5 redundancy under Hour+Month time-aligned retrieval.

For each test query's top-5 memories, measure: time gaps, pairwise context
Pearson similarity, adjacent-stride overlap, unique dates.  Aggregates over
queries to decide whether a Diversity module is warranted.
"""

import numpy as np
import pandas as pd

from retrieval.decompose import seasonal_decompose
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
PERIOD, TREND_WINDOW = 24, 48
ALPHA = 0.75  # val-best alpha for Hour+Month


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, train, train_mask, int(test_idx[0]), int(test_idx[-1]) + 1


def circular_lut(n):
    lut = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            lut[i, j] = 1.0 - d / (n // 2)
    return lut


def topk(scores, k):
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def main():
    dates, ot, train, train_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]
    bank = MemoryBank.from_series(train, CONTEXT, PRED)
    mem_dates = train_dates.iloc[np.arange(bank.n_memories) + CONTEXT]

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    q_dates = dates.iloc[starts]

    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    S_p = pearson_similarity_matrix(R_q, R_m).astype(np.float32)

    month_lut, hour_lut = circular_lut(12), circular_lut(24)
    month = month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
    hour = hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]
    B = (month + hour) / 2.0
    B = (B - B.mean()) / B.std() * S_p.std() + S_p.mean()

    S = (1 - ALPHA) * S_p + ALPHA * B
    idx = topk(S, K)  # (n, 5), sorted descending by score

    # ---- 1. time gaps (index diff == hours, since train is hourly) ----
    idx_sorted = np.sort(idx, axis=1)
    gaps = idx_sorted[:, 1:] - idx_sorted[:, :-1]          # (n, 4) adjacent gaps (hours)
    min_gap = gaps.min(axis=1)                              # (n,)
    span = idx_sorted[:, -1] - idx_sorted[:, 0]             # (n,) hours

    # ---- 2. pairwise context Pearson ----
    ctx5 = bank.contexts[idx].astype(np.float64)            # (n, 5, 512)
    c = ctx5 - ctx5.mean(axis=2, keepdims=True)
    sd = np.sqrt((c ** 2).sum(axis=2, keepdims=True))
    c = c / np.where(sd == 0, 1.0, sd)
    R = np.einsum("nij,nkj->nik", c, c)                     # (n, 5, 5) correlation
    eye = np.eye(K, dtype=bool)
    off = ~eye
    # mean off-diagonal correlation per query
    off_sum = R.sum(axis=(1, 2)) - R[:, eye].sum(axis=1)
    pair_corr = off_sum / (K * (K - 1))                     # (n,)

    # ---- 3. adjacent-stride overlap (pairwise index diff <= threshold) ----
    pd_ = idx_sorted[:, :, None] - idx_sorted[:, None, :]   # (n,5,5)
    pd_ = np.abs(pd_)
    mask = np.broadcast_to(~eye, pd_.shape)
    def frac_within(thr):
        return ((pd_ <= thr) & mask).sum(axis=(1, 2)) / (K * (K - 1))

    frac_1h = frac_within(1)     # literally adjacent stride-1
    frac_24h = frac_within(24)   # within 1 day
    frac_168h = frac_within(168) # within 1 week

    # ---- 4. unique days ----
    days = np.sort(idx // 24, axis=1)
    unique_days = 1 + (days[:, 1:] != days[:, :-1]).sum(axis=1)  # (n,)

    print(f"n queries = {len(idx)}\n")
    print("=== time gaps (hours) ===")
    print(f"  min adjacent gap: mean={min_gap.mean():.1f}  median={np.median(min_gap):.0f}")
    print(f"  span (max-min)   : mean={span.mean():.1f}  median={np.median(span):.0f}")

    print("\n=== pairwise context Pearson (top-5) ===")
    print(f"  mean={pair_corr.mean():.4f}  median={np.median(pair_corr):.4f}")

    print("\n=== adjacent-stride overlap (fraction of 10 pairs) ===")
    print(f"  within 1h (stride-1 adjacent): {frac_1h.mean() * 100:.1f}%")
    print(f"  within 24h (same day)        : {frac_24h.mean() * 100:.1f}%")
    print(f"  within 168h (same week)      : {frac_168h.mean() * 100:.1f}%")

    print("\n=== unique dates among top-5 ===")
    print(f"  mean unique days = {unique_days.mean():.2f}  (max 5)")
    print(f"  queries with all 5 in one day = {(unique_days == 1).mean() * 100:.1f}%")
    print(f"  queries with 5 distinct days  = {(unique_days == 5).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
