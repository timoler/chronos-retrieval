"""Continuous temporal weight search (dataset-level w_H / w_W / w_M).

Replaces the 8 discrete calendar combinations with a continuous, non-negative
weight simplex:

    TimeSim = w_H * HourSim + w_W * WeekdaySim + w_M * MonthSim
    w_H, w_W, w_M >= 0,  w_H + w_W + w_M = 1

searched on a 0.1-step grid (66 combinations).  The retrieval score keeps its
two-layer structure:

    FinalScore = (1 - alpha) * ShapeSim + alpha * TimeSim

so (w_H, w_W, w_M) decide *what the temporal component looks like* while alpha
decides *shape vs time*.  alpha / lambda reuse the exact same grids as before.

Top-K is deterministic: at alpha == 1.0 (pure discrete temporal -> exact score
ties) it uses the lexicographic tie-break (final_score desc, shape_sim desc,
memory_index asc); at alpha < 1.0 the score is continuous (verified: zero
boundary ties) so the equivalent fast argpartition path is used.

Frozen pipeline: Chronos-2 (cached), residual-Pearson shape sim, mean-anchor
transfer, Top-K=5, softmax, Chronos-Memory fusion, and the standard splits.
HourSim / WeekdaySim / MonthSim / ShapeSim are cached once per dataset so no
decomposition / Pearson / Chronos work is repeated across weight combos.

Everything is selected on VAL only; the frozen (w, alpha, lambda) is evaluated
once on TEST.
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
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
STEP = 0.1

DATASETS = {
    "ETTh1": dict(period=24, trend_window=48, stride=1),
    "ETTh2": dict(period=24, trend_window=48, stride=1),
    "ETTm1": dict(period=96, trend_window=192, stride=4),
    "ETTm2": dict(period=96, trend_window=192, stride=4),
}

# frozen discrete best baseline: (combo name, discrete fusion test MSE)
DISCRETE = {
    "ETTh1": ("Hour+Month", 5.6986),
    "ETTh2": ("Full", 25.3748),      # deterministic tie-break result
    "ETTm1": ("Hour+Weekday", 2.9089),
    "ETTm2": ("Hour", 14.6477),
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


def standardize(B, S_p):
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def shape_tiebreak_order(shape_sim):
    return np.argsort(-shape_sim, axis=-1, kind="stable")


def topk_deterministic(scores, shape_order, k):
    """Lexicographic top-K: final_score desc, shape_sim desc, memory_index asc."""
    score_tb = np.take_along_axis(scores, shape_order, axis=-1)
    o = np.argsort(-score_tb, axis=-1, kind="stable")
    return np.take_along_axis(shape_order, o, axis=-1)[..., :k]


def topk_fast(scores, k):
    """argpartition top-K; valid only when scores have no exact ties (alpha < 1.0)."""
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def weight_grid(step=STEP):
    n = int(round(1.0 / step))
    grid = []
    for h in range(n + 1):
        for w in range(n + 1 - h):
            m = n - h - w
            grid.append((h * step, w * step, m * step))
    return grid


def prepare(name):
    cfg = DATASETS[name]
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

    cache = np.load(f"results/{name}_chronos.npz")
    y_chronos_val = cache["y_chronos_val"]
    y_chronos_test = cache["y_chronos_test"]

    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    S_p_val = residual_pearson(val_contexts, bank, period, trend_window)
    S_p_test = residual_pearson(test_contexts, bank, period, trend_window)
    aligns_val = build_alignments(val_dates, mem_dates, luts)
    aligns_test = build_alignments(test_dates, mem_dates, luts)

    return dict(
        name=name, bank=bank,
        val_contexts=val_contexts, val_futures=val_futures,
        test_contexts=test_contexts, test_futures=test_futures,
        y_chronos_val=y_chronos_val, y_chronos_test=y_chronos_test,
        S_p_val=S_p_val, S_p_test=S_p_test,
        H_val=aligns_val["hour"], W_val=aligns_val["dow"], M_val=aligns_val["month"],
        H_test=aligns_test["hour"], W_test=aligns_test["dow"], M_test=aligns_test["month"],
        shape_order_val=shape_tiebreak_order(S_p_val),
        shape_order_test=shape_tiebreak_order(S_p_test),
    )


def compute(name):
    P = prepare(name)
    grid = weight_grid()

    best = None
    for wh, ww, wm in grid:
        TR_val = wh * P["H_val"] + ww * P["W_val"] + wm * P["M_val"]
        Tstd_val = standardize(TR_val, P["S_p_val"])
        for alpha in ALPHAS:
            S = (1 - alpha) * P["S_p_val"] + alpha * Tstd_val
            if alpha == 1.0:
                idx = topk_deterministic(S, P["shape_order_val"], K)
            else:
                idx = topk_fast(S, K)
            sc = np.take_along_axis(S, idx, axis=-1)
            y_mem = mean_anchor(P["bank"], P["val_contexts"], idx, sc)
            for lam in LAMS:
                fused = (1 - lam) * P["y_chronos_val"] + lam * y_mem
                mse = float(((fused - P["val_futures"]) ** 2).mean())
                mae = float(np.abs(fused - P["val_futures"]).mean())
                if best is None or (mse, mae) < (best[5], best[6]):
                    best = (wh, ww, wm, alpha, lam, mse, mae)
    wh, ww, wm, alpha, lam, val_mse, val_mae = best

    # freeze and evaluate on TEST
    TR_test = wh * P["H_test"] + ww * P["W_test"] + wm * P["M_test"]
    Tstd_test = standardize(TR_test, P["S_p_test"])
    S_test = (1 - alpha) * P["S_p_test"] + alpha * Tstd_test
    if alpha == 1.0:
        idx = topk_deterministic(S_test, P["shape_order_test"], K)
    else:
        idx = topk_fast(S_test, K)
    sc = np.take_along_axis(S_test, idx, axis=-1)
    y_mem = mean_anchor(P["bank"], P["test_contexts"], idx, sc)
    fused = (1 - lam) * P["y_chronos_test"] + lam * y_mem

    chronos_mse = float(((P["y_chronos_test"] - P["test_futures"]) ** 2).mean())
    chronos_mae = float(np.abs(P["y_chronos_test"] - P["test_futures"]).mean())
    mem_mse = float(((y_mem - P["test_futures"]) ** 2).mean())
    mem_mae = float(np.abs(y_mem - P["test_futures"]).mean())
    fus_mse = float(((fused - P["test_futures"]) ** 2).mean())
    fus_mae = float(np.abs(fused - P["test_futures"]).mean())
    improvement = (chronos_mse - fus_mse) / chronos_mse * 100

    return dict(
        name=name, w_H=wh, w_W=ww, w_M=wm, alpha=alpha, lam=lam,
        val_mse=val_mse, val_mae=val_mae,
        chronos_mse=chronos_mse, chronos_mae=chronos_mae,
        mem_mse=mem_mse, mem_mae=mem_mae, fus_mse=fus_mse, fus_mae=fus_mae,
        improvement=improvement,
    )


def main():
    args = parse_args()
    names = [n.strip() for n in args.datasets.split(",")] if args.datasets else list(DATASETS)
    for n in names:
        if n not in DATASETS:
            raise SystemExit(f"unknown dataset {n!r}")
        if not os.path.exists(os.path.join(args.cache_dir, f"{n}_chronos.npz")):
            raise SystemExit(f"missing results/{n}_chronos.npz; run the ablation first")

    results = {}
    for name in names:
        r = compute(name)
        results[name] = r
        print(f"\n=== {name} ===")
        print(f"best w = (H={r['w_H']:.1f}, W={r['w_W']:.1f}, M={r['w_M']:.1f})")
        print(f"alpha = {r['alpha']:.2f}   lambda = {r['lam']:.2f}")
        print(f"val_MSE = {r['val_mse']:.4f}   val_MAE = {r['val_mae']:.4f}")
        print(f"Chronos test MSE = {r['chronos_mse']:.4f}   MAE = {r['chronos_mae']:.4f}")
        print(f"Memory test MSE = {r['mem_mse']:.4f}   MAE = {r['mem_mae']:.4f}")
        print(f"Fusion  test MSE = {r['fus_mse']:.4f}   MAE = {r['fus_mae']:.4f}")
        print(f"improvement vs Chronos = {r['improvement']:+.2f}%")

    # ---- per-dataset best config table ----
    print("\n=== Per-dataset continuous best (VAL-selected) ===")
    print(f"{'Dataset':>8} {'w_H':>5} {'w_W':>5} {'w_M':>5} {'alpha':>6} {'lambda':>7} "
          f"{'val_MSE':>9} {'mem_MSE':>9} {'mem_MAE':>8} {'fus_MSE':>9} {'fus_MAE':>8} {'imp%':>7}")
    for name in names:
        r = results[name]
        print(f"{name:>8} {r['w_H']:5.1f} {r['w_W']:5.1f} {r['w_M']:5.1f} {r['alpha']:6.2f} "
              f"{r['lam']:7.2f} {r['val_mse']:9.4f} {r['mem_mse']:9.4f} {r['mem_mae']:8.4f} "
              f"{r['fus_mse']:9.4f} {r['fus_mae']:8.4f} {r['improvement']:+6.2f}%")

    # ---- continuous vs discrete summary ----
    print("\n=== Continuous vs discrete baseline ===")
    print(f"{'Dataset':>8} {'DiscreteBest':>14} {'DiscMSE':>10} {'ContWeights':>18} "
          f"{'ContAlpha':>9} {'ContLambda':>10} {'ContMSE':>10} {'vsDisc%':>9}")
    summary_rows = []
    for name in names:
        r = results[name]
        dname, disc_mse = DISCRETE[name]
        vs_disc = (disc_mse - r["fus_mse"]) / disc_mse * 100
        wstr = f"({r['w_H']:.1f},{r['w_W']:.1f},{r['w_M']:.1f})"
        print(f"{name:>8} {dname:>14} {disc_mse:10.4f} {wstr:>18} "
              f"{r['alpha']:9.2f} {r['lam']:10.2f} {r['fus_mse']:10.4f} {vs_disc:+8.2f}%")
        summary_rows.append(dict(
            dataset=name, discrete_best=dname, discrete_mse=disc_mse,
            cont_w_H=r["w_H"], cont_w_W=r["w_W"], cont_w_M=r["w_M"],
            cont_alpha=r["alpha"], cont_lambda=r["lam"],
            cont_fusion_mse=r["fus_mse"], vs_discrete_pct=vs_disc,
        ))

    # ---- persist ----
    os.makedirs(args.cache_dir, exist_ok=True)
    long_rows = [
        dict(dataset=r["name"], w_H=r["w_H"], w_W=r["w_W"], w_M=r["w_M"],
             alpha=r["alpha"], lambda_=r["lam"], val_mse=r["val_mse"], val_mae=r["val_mae"],
             chronos_mse=r["chronos_mse"], chronos_mae=r["chronos_mae"],
             mem_mse=r["mem_mse"], mem_mae=r["mem_mae"], fus_mse=r["fus_mse"], fus_mae=r["fus_mae"],
             improvement_pct=r["improvement"])
        for r in results.values()
    ]
    pd.DataFrame(long_rows).to_csv(
        os.path.join(args.cache_dir, "continuous_temporal_weights_all_ett.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.cache_dir, "continuous_vs_discrete_summary.csv"), index=False)
    print(f"\nSaved -> {args.cache_dir}/continuous_temporal_weights_all_ett.csv")
    print(f"Saved -> {args.cache_dir}/continuous_vs_discrete_summary.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated subset, e.g. ETTh1,ETTm1 (default: all four)")
    p.add_argument("--cache-dir", type=str, default="results",
                   help="directory holding cached forecasts and CSV output")
    return p.parse_args()


if __name__ == "__main__":
    main()
