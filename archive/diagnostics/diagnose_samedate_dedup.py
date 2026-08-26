"""Same-date dedup Top-5 vs naive Top-5 (Hour+Month, alpha=0.75, lambda=0.20).

Greedily force the top-5 to come from 5 distinct calendar days.  Compare test
fusion MSE against the non-dedup baseline 5.6986.
"""

import numpy as np
import pandas as pd

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
PERIOD, TREND_WINDOW = 24, 48
ALPHA = 0.75
LAM = 0.20


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


def same_date_topk(S, mem_day, k=5, M=50):
    n, n_mem = S.shape
    M = min(M, n_mem)
    cand = np.argpartition(S, -M, axis=1)[:, -M:]
    cs = np.take_along_axis(S, cand, axis=1)
    order = np.argsort(-cs, axis=1)
    cand = np.take_along_axis(cand, order, axis=1)
    cd = mem_day[cand]  # (n, M)
    idx = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        seen = set()
        sel = 0
        for m in range(M):
            d = int(cd[i, m])
            if d not in seen:
                seen.add(d)
                idx[i, sel] = cand[i, m]
                sel += 1
                if sel == k:
                    break
        if sel < k:  # safety fallback (should not trigger)
            for m in range(M):
                if cand[i, m] not in idx[i, :sel]:
                    idx[i, sel] = cand[i, m]
                    sel += 1
                    if sel == k:
                        break
    return idx


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def main():
    dates, ot, train, train_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]
    bank = MemoryBank.from_series(train, CONTEXT, PRED)
    mem_dates = train_dates.iloc[np.arange(bank.n_memories) + CONTEXT]
    mem_day = (np.arange(bank.n_memories) + CONTEXT) // 24

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)
    q_dates = dates.iloc[starts]
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    S_p = pearson_similarity_matrix(R_q, R_m).astype(np.float32)

    month_lut, hour_lut = circular_lut(12), circular_lut(24)
    month = month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
    hour = hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]
    B = (month + hour) / 2.0
    B = (B - B.mean()) / B.std() * S_p.std() + S_p.mean()
    S = (1 - ALPHA) * S_p + ALPHA * B

    # non-dedup baseline
    idx_nd = topk(S, K)
    y_nd = mean_anchor(bank, contexts, idx_nd, np.take_along_axis(S, idx_nd, axis=-1))
    fused_nd = (1 - LAM) * y_chronos + LAM * y_nd
    mse_nd = float(((fused_nd - futures) ** 2).mean())

    # same-date dedup
    idx_dd = same_date_topk(S, mem_day, K)
    y_dd = mean_anchor(bank, contexts, idx_dd, np.take_along_axis(S, idx_dd, axis=-1))
    fused_dd = (1 - LAM) * y_chronos + LAM * y_dd
    mse_dd = float(((fused_dd - futures) ** 2).mean())
    mae_dd = float(np.abs(fused_dd - futures).mean())
    mem_mse_dd = float(((y_dd - futures) ** 2).mean())

    print(f"(alpha={ALPHA}, lambda={LAM})")
    print(f"non-dedup (baseline): fusion MSE = {mse_nd:.4f}  (expect ~5.6986)")
    print(f"same-date dedup     : memory MSE = {mem_mse_dd:.4f}  fusion MSE = {mse_dd:.4f}  "
          f"MAE = {mae_dd:.4f}")


if __name__ == "__main__":
    main()
