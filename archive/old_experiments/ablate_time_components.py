"""Ablate Time-Alignment calendar components (val selects, test evaluates once).

8 methods: None / Hour / Weekday / Month / Hour+Weekday / Hour+Month /
Weekday+Month / Full.  For each, independently select (alpha_time, lambda)
on VAL by val MSE, then freeze and evaluate once on TEST.
No regime gate, no diversity, no other modules.
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
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
CHRONOS_TEST_MSE = 6.1968

COMBOS = [
    ("None", []),
    ("Hour", ["hour"]),
    ("Weekday", ["dow"]),
    ("Month", ["month"]),
    ("Hour+Weekday", ["hour", "dow"]),
    ("Hour+Month", ["hour", "month"]),
    ("Weekday+Month", ["dow", "month"]),
    ("Full", ["hour", "dow", "month"]),
]


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, train, train_mask, val_mask, int(test_idx[0]), int(test_idx[-1]) + 1


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


def windows(ot, start, end):
    s = np.arange(start, end - PRED + 1)
    c = np.stack([ot[i - CONTEXT : i] for i in s]).astype(np.float32)
    f = np.stack([ot[i : i + PRED] for i in s]).astype(np.float32)
    return s, c, f


def residual_pearson(contexts, bank):
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    return pearson_similarity_matrix(R_q, R_m).astype(np.float32)


def build_alignments(q_dates, mem_dates, luts):
    month_lut, dow_lut, hour_lut = luts
    month = month_lut[q_dates.dt.month.to_numpy() - 1][:, mem_dates.dt.month.to_numpy() - 1]
    dow = dow_lut[q_dates.dt.dayofweek.to_numpy()][:, mem_dates.dt.dayofweek.to_numpy()]
    hour = hour_lut[q_dates.dt.hour.to_numpy()][:, mem_dates.dt.hour.to_numpy()]
    return {"hour": hour, "dow": dow, "month": month}


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def combined_B(aligns, keys):
    B = aligns[keys[0]].copy()
    for k in keys[1:]:
        B += aligns[k]
    B /= len(keys)
    return B  # in [0,1]


def standardize(B, S_p):
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def eval_combo(S_p, B, chronos, futures, bank, contexts):
    """Grid search (alpha, lambda) -> best (alpha, lam, mse). B=None => Pearson only."""
    best = None
    if B is None:
        S = S_p
        idx = topk(S, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        y_mem = mean_anchor(bank, contexts, idx, sc)
        for lam in LAMS:
            fused = (1 - lam) * chronos + lam * y_mem
            mse = float(((fused - futures) ** 2).mean())
            if best is None or mse < best[2]:
                best = (0.0, lam, mse)
        return best
    for alpha in ALPHAS:
        S = (1 - alpha) * S_p + alpha * B
        idx = topk(S, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        y_mem = mean_anchor(bank, contexts, idx, sc)
        for lam in LAMS:
            fused = (1 - lam) * chronos + lam * y_mem
            mse = float(((fused - futures) ** 2).mean())
            if best is None or mse < best[2]:
                best = (alpha, lam, mse)
    return best


def apply(S_p, B, alpha, lam, chronos, futures, bank, contexts):
    if B is None:
        S = S_p
    else:
        S = (1 - alpha) * S_p + alpha * B
    idx = topk(S, K)
    sc = np.take_along_axis(S, idx, axis=-1)
    y_mem = mean_anchor(bank, contexts, idx, sc)
    fused = (1 - lam) * chronos + lam * y_mem
    mem_mse = float(((y_mem - futures) ** 2).mean())
    fus_mse = float(((fused - futures) ** 2).mean())
    fus_mae = float(np.abs(fused - futures).mean())
    return mem_mse, fus_mse, fus_mae


def main():
    dates, ot, train, train_mask, val_mask, test_start, test_end = load("data/ETTh1.csv")
    train_dates = dates[train_mask]
    bank = MemoryBank.from_series(train, CONTEXT, PRED)
    mem_dates = train_dates.iloc[np.arange(bank.n_memories) + CONTEXT]
    luts = (circular_lut(12), circular_lut(7), circular_lut(24))

    val_idx = np.flatnonzero(val_mask)
    val_start, val_end = int(val_idx[0]), int(val_idx[-1]) + 1

    # val
    val_starts, val_contexts, val_futures = windows(ot, val_start, val_end)
    val_dates = dates.iloc[val_starts]
    y_chronos_val = np.load("results/val_forecasts.npz")["y_chronos_val"]
    S_p_val = residual_pearson(val_contexts, bank)
    aligns_val = build_alignments(val_dates, mem_dates, luts)

    # test
    test_starts, test_contexts, test_futures = windows(ot, test_start, test_end)
    test_dates = dates.iloc[test_starts]
    y_chronos_test = np.load("results/forecasts.npz")["y_chronos"]
    S_p_test = residual_pearson(test_contexts, bank)
    aligns_test = build_alignments(test_dates, mem_dates, luts)

    print(f"val chronos MSE = {float(((y_chronos_val - val_futures) ** 2).mean()):.4f}")
    print(f"test chronos MSE = {float(((y_chronos_test - test_futures) ** 2).mean()):.4f} (expect 6.1968)\n")

    rows = []
    for name, keys in COMBOS:
        if not keys:
            B_val = None
            B_test = None
        else:
            B_val = standardize(combined_B(aligns_val, keys), S_p_val)
            B_test = standardize(combined_B(aligns_test, keys), S_p_test)
        alpha, lam, val_mse = eval_combo(S_p_val, B_val, y_chronos_val, val_futures, bank, val_contexts)
        mem_mse, fus_mse, fus_mae = apply(S_p_test, B_test, alpha, lam, y_chronos_test, test_futures, bank, test_contexts)
        vs = (CHRONOS_TEST_MSE - fus_mse) / CHRONOS_TEST_MSE * 100
        rows.append((name, alpha, lam, val_mse, mem_mse, fus_mse, fus_mae, vs))

    print(f"{'components':>16} {'val_a':>6} {'val_l':>6} {'val_MSE':>8} "
          f"{'test_memMSE':>12} {'test_fusMSE':>12} {'test_fusMAE':>12} {'vsChronos%':>10}")
    for name, alpha, lam, val_mse, mem_mse, fus_mse, fus_mae, vs in rows:
        print(f"{name:>16} {alpha:6.2f} {lam:6.2f} {val_mse:8.4f} "
              f"{mem_mse:12.4f} {fus_mse:12.4f} {fus_mae:12.4f} {vs:+9.2f}%")


if __name__ == "__main__":
    main()
