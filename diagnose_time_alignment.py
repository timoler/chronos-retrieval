"""SARAF-style time-aligned retrieval score (calendar alignment).

Only the retrieval score changes: S = (1-alpha)*S_pearson + alpha*B_time.
Pipeline otherwise unchanged: Residual Pearson + Mean-anchor, Top-K=5,
Chronos-2 forecasts fixed.

B_time (calendar alignment, hour-of-day / day-of-week / month-of-year):
  month : 1 - min(|dm|, 12-|dm|)/6
  dow   : 1 - min(|dd|, 7-|dd|)/3
  hour  : 1 - min(|dh|, 24-|dh|)/12
  align = (month + dow + hour)/3  in [0,1]
  B_raw = 2*align - 1  in [-1,1]
  B_time = standardized to match S_pearson's global mean/std.
"""

import numpy as np
import pandas as pd

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
PERIOD, TREND_WINDOW = 24, 48
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAM = 0.1


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

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)
    y_chronos = np.load("results/forecasts.npz")["y_chronos"]

    # residual Pearson score (unchanged)
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    S_pearson = pearson_similarity_matrix(R_q, R_m).astype(np.float32)

    # calendar alignment
    month_lut = circular_lut(12)
    dow_lut = circular_lut(7)
    hour_lut = circular_lut(24)

    q_dates = dates.iloc[starts]
    month_q = q_dates.dt.month.to_numpy() - 1
    dow_q = q_dates.dt.dayofweek.to_numpy()
    hour_q = q_dates.dt.hour.to_numpy()

    mem_dates = train_dates.iloc[np.arange(bank.n_memories) + CONTEXT]
    month_m = mem_dates.dt.month.to_numpy() - 1
    dow_m = mem_dates.dt.dayofweek.to_numpy()
    hour_m = mem_dates.dt.hour.to_numpy()

    align = (month_lut[month_q][:, month_m] + dow_lut[dow_q][:, dow_m]
             + hour_lut[hour_q][:, hour_m]) / 3.0
    B_raw = 2.0 * align - 1.0
    B_time = (B_raw - B_raw.mean()) / B_raw.std() * S_pearson.std() + S_pearson.mean()
    del align, B_raw

    def mem_forecast(idx, scores):
        w = softmax(scores)
        fut = bank.futures[idx]
        ctx = bank.contexts[idx]
        shape = fut - ctx.mean(axis=2)[:, :, None]
        return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)

    def eval_alpha(a):
        S = (1 - a) * S_pearson + a * B_time
        idx = topk(S, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        y_mem = mem_forecast(idx, sc)
        mem_mse = float(((y_mem - futures) ** 2).mean())
        mem_mae = float(np.abs(y_mem - futures).mean())
        fused = (1 - LAM) * y_chronos + LAM * y_mem
        f_mse = float(((fused - futures) ** 2).mean())
        f_mae = float(np.abs(fused - futures).mean())
        return mem_mse, mem_mae, f_mse, f_mae, idx

    base_mem, base_mae, base_f, base_fmae, _ = eval_alpha(0.0)
    print(f"Chronos-2 baseline: MSE=6.1968  MAE=1.8649")
    print(f"{'alpha':>6} {'mem_MSE':>9} {'mem_MAE':>8} {'fused_MSE':>10} {'fused_MAE':>9} "
          f"{'dMemMSE':>9} {'dFusedMSE':>10}")
    for a in ALPHAS:
        m_mse, m_mae, f_mse, f_mae, _ = eval_alpha(a)
        print(f"{a:6.2f} {m_mse:9.4f} {m_mae:8.4f} {f_mse:10.4f} {f_mae:9.4f} "
              f"{m_mse - base_mem:+9.4f} {f_mse - 6.1968:+10.4f}")

    # spot-check: 3 queries (winter / spring / summer), top-5 dates at a=0,0.5,1
    q_month = q_dates.dt.month.to_numpy()
    print("\n=== spot-check: top-5 memory dates (forecast-start) ===")
    for target in [1, 4, 6]:
        idxs = np.flatnonzero(q_month == target)
        qi = int(idxs[len(idxs) // 2])
        qd = q_dates.iloc[qi]
        print(f"\nquery {qi}: {qd}")
        for a in [0.0, 0.5, 1.0]:
            S = (1 - a) * S_pearson + a * B_time
            tk = topk(S[qi][None, :], K)[0]
            ds = [str(mem_dates.iloc[j])[:10] for j in tk]
            print(f"  alpha={a}: {ds}")


if __name__ == "__main__":
    main()
