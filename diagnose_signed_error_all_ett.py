"""Per-point signed-error diagnostics for the temporal-alignment fusion.

For each dataset, using the BEST temporal combination (frozen alpha/lambda
from run_temporal_ablation_all_ett.py's val selection — nothing re-selected
here), compute the per-point signed errors:

    e_C = y_chronos - y
    e_M = y_mem     - y

and report:
    corr(e_C, e_M)
    same_sign_ratio     = mean(sign(e_C) == sign(e_M))
    opposite_sign_ratio = mean(sign(e_C) * sign(e_M) < 0)
    Chronos / Memory error mean & std
    quintile buckets by |e_C| -> Chronos MSE / Memory MSE / Fusion MSE / imp%

No gate, no alpha/lambda re-selection, no retrieval change.  y_mem is
recomputed exactly as in the ablation script (Residual Pearson + temporal
alignment at the frozen alpha, Top-K=5, Mean-anchor).
"""

import numpy as np
import pandas as pd

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5

# sampling-frequency config (matches run_temporal_ablation_all_ett.py)
DATASETS = {
    "ETTh1": dict(period=24, trend_window=48, stride=1),
    "ETTh2": dict(period=24, trend_window=48, stride=1),
    "ETTm1": dict(period=96, trend_window=192, stride=4),
    "ETTm2": dict(period=96, trend_window=192, stride=4),
}

# BEST temporal combination per dataset (frozen from the ablation run's val
# selection): (calendar keys, alpha, lambda)
BEST = {
    "ETTh1": (["hour", "month"], 0.75, 0.20),
    "ETTh2": (["hour", "dow", "month"], 1.00, 0.30),
    "ETTm1": (["hour", "dow"], 0.75, 0.10),
    "ETTm2": (["hour"], 0.75, 0.20),
}


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    return dates, ot, train_mask, val_mask, test_mask


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


def windows(ot, start, end, stride):
    s = np.arange(start, end - PRED + 1, stride)
    c = np.stack([ot[i - CONTEXT : i] for i in s]).astype(np.float32)
    f = np.stack([ot[i : i + PRED] for i in s]).astype(np.float32)
    return s, c, f


def build_memory_bank(train, stride):
    window = CONTEXT + PRED
    starts = np.arange(0, len(train) - window + 1, stride)
    contexts = np.stack([train[s : s + CONTEXT] for s in starts]).astype(np.float32)
    futures = np.stack([train[s + CONTEXT : s + window] for s in starts]).astype(np.float32)
    return MemoryBank(contexts=contexts, futures=futures), starts


def residual_pearson(contexts, bank, period, trend_window):
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, period, trend_window)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, period, trend_window)
    return pearson_similarity_matrix(R_q, R_m).astype(np.float32)


def build_alignments(q_dates, mem_dates, luts):
    month_lut, dow_lut, hour_lut = luts
    month = month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
    dow = dow_lut[q_dates.dt.dayofweek.to_numpy()][:, mem_dates.dt.dayofweek.to_numpy()]
    hour = hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]
    return {"hour": hour, "dow": dow, "month": month}


def combined_B(aligns, keys):
    B = aligns[keys[0]].copy()
    for k in keys[1:]:
        B += aligns[k]
    B /= len(keys)
    return B


def standardize(B, S_p):
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def analyze(name):
    cfg = DATASETS[name]
    keys, alpha, lam = BEST[name]
    period, trend_window, stride = cfg["period"], cfg["trend_window"], cfg["stride"]

    dates, ot, train_mask, val_mask, test_mask = load(f"data/{name}.csv")
    train = ot[train_mask]
    train_dates = dates[train_mask]

    bank, mem_starts = build_memory_bank(train, stride)
    mem_dates = train_dates.iloc[mem_starts + CONTEXT]

    test_start = int(np.flatnonzero(test_mask)[0])
    test_end = int(np.flatnonzero(test_mask)[-1]) + 1
    test_starts, test_contexts, test_futures = windows(ot, test_start, test_end, stride)
    test_dates = dates.iloc[test_starts]

    y_chronos = np.load(f"results/{name}_chronos.npz")["y_chronos_test"]

    S_p = residual_pearson(test_contexts, bank, period, trend_window)
    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    aligns = build_alignments(test_dates, mem_dates, luts)
    B = standardize(combined_B(aligns, keys), S_p)
    S = (1 - alpha) * S_p + alpha * B
    idx = topk(S, K)
    sc = np.take_along_axis(S, idx, axis=-1)
    y_mem = mean_anchor(bank, test_contexts, idx, sc)
    fused = (1 - lam) * y_chronos + lam * y_mem

    e_C = (y_chronos - test_futures).ravel()
    e_M = (y_mem - test_futures).ravel()
    e_F = (fused - test_futures).ravel()
    n = e_C.size

    combo = "+".join({"hour": "Hour", "dow": "Weekday", "month": "Month"}[k] for k in keys)
    print(f"\n=== {name}  (best={combo}, alpha={alpha}, lambda={lam}) ===")
    print(f"points = {n}  (test windows x 96)")

    corr = float(np.corrcoef(e_C, e_M)[0, 1])
    same = float((np.sign(e_C) == np.sign(e_M)).mean())
    opp = float((np.sign(e_C) * np.sign(e_M) < 0).mean())
    print(f"corr(e_C, e_M)          = {corr:+.4f}")
    print(f"same_sign_ratio         = {same:.4f}")
    print(f"opposite_sign_ratio     = {opp:.4f}")
    print(f"Chronos error  mean/std = {e_C.mean():+.4f} / {e_C.std():.4f}")
    print(f"Memory error  mean/std  = {e_M.mean():+.4f} / {e_M.std():.4f}")

    # quintile buckets by |e_C|
    q = np.abs(e_C)
    edges = np.quantile(q, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins = np.digitize(q, edges) - 1

    print(f"\n{'|e_C| bucket':>14} {'n':>8} {'mean|e_C|':>9} {'ChronosMSE':>11} "
          f"{'MemoryMSE':>10} {'FusionMSE':>10} {'imp%':>8}")
    for b in range(5):
        m = bins == b
        chronos_mse = float((e_C[m] ** 2).mean())
        mem_mse = float((e_M[m] ** 2).mean())
        fus_mse = float((e_F[m] ** 2).mean())
        imp = (chronos_mse - fus_mse) / chronos_mse * 100
        print(f"{b + 1:>6} (Q{b + 1}) {int(m.sum()):>8} {q[m].mean():9.4f} "
              f"{chronos_mse:11.4f} {mem_mse:10.4f} {fus_mse:10.4f} {imp:+7.2f}%")


def main():
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        analyze(name)


if __name__ == "__main__":
    main()
