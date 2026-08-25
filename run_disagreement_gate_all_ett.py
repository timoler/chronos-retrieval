"""Disagreement-aware fusion: does a 3-bin disagreement gate beat fixed lambda?

Freezes the best Temporal Retrieval config per dataset (Chronos / retrieval /
mean-anchor / Top-K / temporal combo / alpha_time are all kept unchanged):

    ETTh1: Hour+Month   alpha=0.75  fixed lambda=0.20
    ETTh2: Full         alpha=1.00  fixed lambda=0.30
    ETTm1: Hour+Weekday alpha=0.75  fixed lambda=0.10
    ETTm2: Hour         alpha=0.75  fixed lambda=0.20

Disagreement (prediction-time scalar, no y_true):

    D_i = mean_h | y_chronos_i - y_memory_i |

On VAL only, compute q33 / q67 percentiles of D and grid-search three per-bin
lambdas (lambda_low / lambda_mid / lambda_high) over a 7^3 = 343 grid, selected
by val MSE (val MAE tie-break).  Then freeze q33 / q67 / the three lambdas and
evaluate ONCE on TEST.  Test windows are bucketed with the val thresholds; test
quantiles are NEVER recomputed.

Chronos-2 forecasts are read from the existing cache results/<name>_chronos.npz
(val + test).  Run run_temporal_ablation_all_ett.py first if that cache is
missing.
"""

import argparse
import os

import numpy as np
import pandas as pd

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5

DATASETS = {
    "ETTh1": dict(period=24, trend_window=48, stride=1),
    "ETTh2": dict(period=24, trend_window=48, stride=1),
    "ETTm1": dict(period=96, trend_window=192, stride=4),
    "ETTm2": dict(period=96, trend_window=192, stride=4),
}

# frozen best Temporal Retrieval config: (combo name, calendar keys, alpha_time, fixed lambda)
BEST = {
    "ETTh1": ("Hour+Month", ["hour", "month"], 0.75, 0.20),
    "ETTh2": ("Full", ["hour", "dow", "month"], 1.00, 0.30),
    "ETTm1": ("Hour+Weekday", ["hour", "dow"], 0.75, 0.10),
    "ETTm2": ("Hour", ["hour"], 0.75, 0.20),
}

LAMBDA_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    return dates, ot, train_mask, val_mask, test_mask


def circular_lut(n):
    """Circular-distance similarity LUT on n values, entries in [0, 1]."""
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
    return B  # in [0, 1]


def standardize(B, S_p):
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def assign_lambda(D, q33, q67, lam_low, lam_mid, lam_high):
    """Bucket each window by disagreement D against val thresholds q33/q67."""
    return np.where(D <= q33, lam_low, np.where(D <= q67, lam_mid, lam_high))


def mse(fused, futures):
    return float(((fused - futures) ** 2).mean())


def mae(fused, futures):
    return float(np.abs(fused - futures).mean())


def compute(name, cache_dir):
    cfg = DATASETS[name]
    combo_name, keys, alpha, lam_fixed = BEST[name]
    period, trend_window, stride = cfg["period"], cfg["trend_window"], cfg["stride"]

    dates, ot, train_mask, val_mask, test_mask = load(f"data/{name}.csv")
    train = ot[train_mask]
    train_dates = dates[train_mask]

    bank, mem_starts = build_memory_bank(train, stride)
    mem_dates = train_dates.iloc[mem_starts + CONTEXT]

    val_start = int(np.flatnonzero(val_mask)[0])
    val_end = int(np.flatnonzero(val_mask)[-1]) + 1
    test_start = int(np.flatnonzero(test_mask)[0])
    test_end = int(np.flatnonzero(test_mask)[-1]) + 1

    val_starts, val_contexts, val_futures = windows(ot, val_start, val_end, stride)
    test_starts, test_contexts, test_futures = windows(ot, test_start, test_end, stride)
    val_dates = dates.iloc[val_starts]
    test_dates = dates.iloc[test_starts]

    cache = np.load(os.path.join(cache_dir, f"{name}_chronos.npz"))
    y_chronos_val = cache["y_chronos_val"]
    y_chronos_test = cache["y_chronos_test"]
    assert y_chronos_val.shape[0] == len(val_starts), f"{name}: cached val forecast count mismatch"
    assert y_chronos_test.shape[0] == len(test_starts), f"{name}: cached test forecast count mismatch"

    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    S_p_val = residual_pearson(val_contexts, bank, period, trend_window)
    S_p_test = residual_pearson(test_contexts, bank, period, trend_window)
    aligns_val = build_alignments(val_dates, mem_dates, luts)
    aligns_test = build_alignments(test_dates, mem_dates, luts)
    B_val = standardize(combined_B(aligns_val, keys), S_p_val)
    B_test = standardize(combined_B(aligns_test, keys), S_p_test)
    S_val = (1 - alpha) * S_p_val + alpha * B_val
    S_test = (1 - alpha) * S_p_test + alpha * B_test

    idx_val = topk(S_val, K)
    idx_test = topk(S_test, K)
    sc_val = np.take_along_axis(S_val, idx_val, axis=-1)
    sc_test = np.take_along_axis(S_test, idx_test, axis=-1)
    y_mem_val = mean_anchor(bank, val_contexts, idx_val, sc_val)
    y_mem_test = mean_anchor(bank, test_contexts, idx_test, sc_test)

    # ---- disagreement (prediction-time, no y_true) ----
    D_val = np.abs(y_chronos_val - y_mem_val).mean(axis=1)
    D_test = np.abs(y_chronos_test - y_mem_test).mean(axis=1)

    # ---- val-only thresholds ----
    q33, q67 = float(np.quantile(D_val, 0.33)), float(np.quantile(D_val, 0.67))

    # ---- val metrics: chronos / fixed-lambda ----
    chronos_val_mse = mse(y_chronos_val, val_futures)
    chronos_val_mae = mae(y_chronos_val, val_futures)
    fused_val_fixed = (1 - lam_fixed) * y_chronos_val + lam_fixed * y_mem_val
    fixed_val_mse = mse(fused_val_fixed, val_futures)
    fixed_val_mae = mae(fused_val_fixed, val_futures)

    # ---- val grid search over 3-bin lambdas (7^3 = 343) ----
    best = None
    for lam_low in LAMBDA_GRID:
        for lam_mid in LAMBDA_GRID:
            for lam_high in LAMBDA_GRID:
                lam_i = assign_lambda(D_val, q33, q67, lam_low, lam_mid, lam_high)
                fused = (1 - lam_i[:, None]) * y_chronos_val + lam_i[:, None] * y_mem_val
                m = mse(fused, val_futures)
                a = mae(fused, val_futures)
                if best is None or (m, a) < (best[3], best[4]):
                    best = (lam_low, lam_mid, lam_high, m, a)
    lam_low, lam_mid, lam_high, gate_val_mse, gate_val_mae = best

    # ---- freeze val thresholds + lambdas, apply to TEST ----
    lam_test = assign_lambda(D_test, q33, q67, lam_low, lam_mid, lam_high)
    fused_gate_test = (1 - lam_test[:, None]) * y_chronos_test + lam_test[:, None] * y_mem_test
    fused_fixed_test = (1 - lam_fixed) * y_chronos_test + lam_fixed * y_mem_test

    chronos_test_mse = mse(y_chronos_test, test_futures)
    chronos_test_mae = mae(y_chronos_test, test_futures)
    fixed_test_mse = mse(fused_fixed_test, test_futures)
    fixed_test_mae = mae(fused_fixed_test, test_futures)
    gate_test_mse = mse(fused_gate_test, test_futures)
    gate_test_mae = mae(fused_gate_test, test_futures)

    vs_chronos = (chronos_test_mse - gate_test_mse) / chronos_test_mse * 100
    vs_fixed = (fixed_test_mse - gate_test_mse) / fixed_test_mse * 100

    # ---- per-bucket breakdown on TEST (val thresholds, no re-bucketing) ----
    buckets_test = np.digitize(D_test, [q33, q67])  # 0=Low, 1=Mid, 2=High
    bucket_rows = []
    for b, bname, lam_b in [(0, "Low", lam_low), (1, "Mid", lam_mid), (2, "High", lam_high)]:
        m = buckets_test == b
        n = int(m.sum())
        if n == 0:
            bucket_rows.append(dict(bucket=bname, n=0, mean_d=float("nan"), lam=lam_b,
                                    chronos_mse=float("nan"), mem_mse=float("nan"),
                                    fixed_mse=float("nan"), gate_mse=float("nan")))
            continue
        bucket_rows.append(dict(
            bucket=bname, n=n, mean_d=float(D_test[m].mean()), lam=lam_b,
            chronos_mse=mse(y_chronos_test[m], test_futures[m]),
            mem_mse=mse(y_mem_test[m], test_futures[m]),
            fixed_mse=mse(fused_fixed_test[m], test_futures[m]),
            gate_mse=mse(fused_gate_test[m], test_futures[m]),
        ))

    return dict(
        name=name, combo=combo_name,
        alpha=alpha, lam_fixed=lam_fixed,
        n_val=len(val_starts), n_test=len(test_starts),
        q33=q33, q67=q67,
        lam_low=lam_low, lam_mid=lam_mid, lam_high=lam_high,
        monotone=bool(lam_low <= lam_mid <= lam_high),
        chronos_val_mse=chronos_val_mse, chronos_val_mae=chronos_val_mae,
        fixed_val_mse=fixed_val_mse, fixed_val_mae=fixed_val_mae,
        gate_val_mse=gate_val_mse, gate_val_mae=gate_val_mae,
        chronos_test_mse=chronos_test_mse, chronos_test_mae=chronos_test_mae,
        fixed_test_mse=fixed_test_mse, fixed_test_mae=fixed_test_mae,
        gate_test_mse=gate_test_mse, gate_test_mae=gate_test_mae,
        vs_chronos=vs_chronos, vs_fixed=vs_fixed,
        buckets=bucket_rows,
    )


def print_val(r):
    print(f"\n=== {r['name']}  (combo={r['combo']}, alpha={r['alpha']}, "
          f"fixed lambda={r['lam_fixed']}) ===")
    print(f"val windows={r['n_val']}  test windows={r['n_test']}")
    print(f"q33 = {r['q33']:.4f}")
    print(f"q67 = {r['q67']:.4f}")
    print(f"best lambda_low  = {r['lam_low']:.2f}")
    print(f"best lambda_mid  = {r['lam_mid']:.2f}")
    print(f"best lambda_high = {r['lam_high']:.2f}")
    print(f"val Chronos MSE / MAE       = {r['chronos_val_mse']:.4f} / {r['chronos_val_mae']:.4f}")
    print(f"val fixed-lambda MSE / MAE  = {r['fixed_val_mse']:.4f} / {r['fixed_val_mae']:.4f}")
    print(f"val disagreement-gate MSE / MAE = {r['gate_val_mse']:.4f} / {r['gate_val_mae']:.4f}")
    print(f"monotone (low<=mid<=high)   = {r['monotone']}")


def print_buckets(r):
    print(f"\n--- {r['name']}: TEST per-bucket (val thresholds frozen) ---")
    print(f"{'Bucket':>7} {'n':>7} {'meanD':>9} {'lambda':>7} {'ChronosMSE':>11} "
          f"{'MemoryMSE':>10} {'FixedMSE':>10} {'GateMSE':>9}")
    for b in r["buckets"]:
        print(f"{b['bucket']:>7} {b['n']:>7} {b['mean_d']:9.4f} {b['lam']:7.2f} "
              f"{b['chronos_mse']:11.4f} {b['mem_mse']:10.4f} {b['fixed_mse']:10.4f} "
              f"{b['gate_mse']:9.4f}")


def main():
    args = parse_args()
    names = [n.strip() for n in args.datasets.split(",")] if args.datasets else list(DATASETS)
    for n in names:
        if n not in DATASETS:
            raise SystemExit(f"unknown dataset {n!r} (choices: {sorted(DATASETS)})")
        path = os.path.join(args.cache_dir, f"{n}_chronos.npz")
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {path}; run run_temporal_ablation_all_ett.py first to "
                f"cache Chronos forecasts"
            )

    results = {}
    for name in names:
        r = compute(name, args.cache_dir)
        results[name] = r
        print_val(r)
        print_buckets(r)

    # ---- final summary table ----
    print("\n\n=== SUMMARY: Disagreement-aware 3-bin gate vs fixed lambda ===")
    print(f"{'Dataset':>8} {'val_q33':>9} {'val_q67':>9} {'l_low':>6} {'l_mid':>6} "
          f"{'l_high':>6} {'ChronosTestMSE':>15} {'FixedFusMSE':>12} {'GateFusMSE':>11} "
          f"{'GateMAE':>9} {'vsChronos%':>11} {'vsFixed%':>10}")
    for name in names:
        r = results[name]
        print(f"{name:>8} {r['q33']:9.4f} {r['q67']:9.4f} {r['lam_low']:6.2f} "
              f"{r['lam_mid']:6.2f} {r['lam_high']:6.2f} {r['chronos_test_mse']:15.4f} "
              f"{r['fixed_test_mse']:12.4f} {r['gate_test_mse']:11.4f} "
              f"{r['gate_test_mae']:9.4f} {r['vs_chronos']:+10.2f}% {r['vs_fixed']:+9.2f}%")

    # ---- persist ----
    summary_rows = [
        dict(dataset=name, val_q33=r["q33"], val_q67=r["q67"],
             lambda_low=r["lam_low"], lambda_mid=r["lam_mid"], lambda_high=r["lam_high"],
             chronos_test_mse=r["chronos_test_mse"], fixed_fusion_test_mse=r["fixed_test_mse"],
             gate_fusion_test_mse=r["gate_test_mse"], gate_test_mae=r["gate_test_mae"],
             vs_chronos_pct=r["vs_chronos"], vs_fixed_pct=r["vs_fixed"])
        for name, r in results.items()
    ]
    bucket_rows = [
        dict(dataset=name, **b)
        for name, r in results.items() for b in r["buckets"]
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.cache_dir, "disagreement_gate_summary.csv"), index=False)
    pd.DataFrame(bucket_rows).to_csv(os.path.join(args.cache_dir, "disagreement_gate_buckets.csv"), index=False)
    print(f"\nSaved summary -> {args.cache_dir}/disagreement_gate_summary.csv")
    print(f"Saved buckets -> {args.cache_dir}/disagreement_gate_buckets.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated subset, e.g. ETTh1,ETTm1 (default: all four)")
    p.add_argument("--cache-dir", type=str, default="results",
                   help="directory holding the cached Chronos forecasts")
    return p.parse_args()


if __name__ == "__main__":
    main()
