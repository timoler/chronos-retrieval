"""Data-driven temporal state vs hand-crafted Calendar alignment.

Question: can Hour / Weekday / Month (manual calendar alignment) be replaced by
a temporal-state representation discovered automatically from the TRAIN series
itself?

For each dataset we (1) linearly detrend the TRAIN OT, (2) run a periodogram,
(3) find the top-3 local peaks in a physical-period window (ETTh 6h..2000h,
ETTm 1.5h..2000h), and (4) encode each discovered period Tk as a continuous
phase feature z_k(t) = [cos(2*pi*t/Tk), sin(2*pi*t/Tk)].

The data-driven temporal similarity between query time tq and memory time tm is
the mean over the 3 periods of the cosine similarity of the phase vectors:

    DataDrivenTimeSim = mean_k cos(2*pi*(tq - tm)/Tk)

which is continuous (so no tie-break ambiguity) and uses NO manual calendar
labels.  It is standardized to the shape-similarity scale (same `standardize`
as the calendar path) and blended as FinalScore = (1-alpha)*ShapeSim +
alpha*TimeSim; alpha/lambda are selected on VAL only and frozen on TEST.

Three methods are compared per dataset:
    A. Shape only            (alpha = 0, lambda selected on val)
    B. Calendar temporal     (frozen best Hour/Weekday/Month combo, known alpha/lambda)
    C. Data-driven periodic  (alpha x lambda selected on val)

Pipeline is otherwise frozen: Chronos-2 (cached), residual-Pearson shape sim,
Top-K=5, mean-anchor, softmax fusion, deterministic tie-break, train-only bank.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import periodogram, find_peaks

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
EPOCH = pd.Timestamp("2016-01-01")

DATASETS = {
    "ETTh1": dict(dt=1.0, period=24, trend_window=48, stride=1, period_range=(6.0, 2000.0)),
    "ETTh2": dict(dt=1.0, period=24, trend_window=48, stride=1, period_range=(6.0, 2000.0)),
    "ETTm1": dict(dt=0.25, period=96, trend_window=192, stride=4, period_range=(1.5, 2000.0)),
    "ETTm2": dict(dt=0.25, period=96, trend_window=192, stride=4, period_range=(1.5, 2000.0)),
}

# frozen calendar best: (calendar keys, alpha, lambda, known fusion test MSE)
CALENDAR = {
    "ETTh1": (["hour", "month"], 0.75, 0.20, 5.6986),
    "ETTh2": (["hour", "dow", "month"], 1.00, 0.30, 25.3748),
    "ETTm1": (["hour", "dow"], 0.75, 0.10, 2.9089),
    "ETTm2": (["hour"], 0.75, 0.20, 14.6477),
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


def shape_tiebreak_order(shape_sim):
    return np.argsort(-shape_sim, axis=-1, kind="stable")


def topk_deterministic(scores, shape_order, k):
    score_tb = np.take_along_axis(scores, shape_order, axis=-1)
    o = np.argsort(-score_tb, axis=-1, kind="stable")
    return np.take_along_axis(shape_order, o, axis=-1)[..., :k]


def topk_fast(scores, k):
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def select_topk(S, shape_order, alpha):
    if alpha == 1.0:
        return topk_deterministic(S, shape_order, K)
    return topk_fast(S, K)


# ---- data-driven temporal-state helpers ----

def linear_detrend(x):
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x), dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - (A @ coef)


def discover_periods(name, cfg, train_ot):
    dt, (p_lo, p_hi) = cfg["dt"], cfg["period_range"]
    fs = 1.0 / dt
    x = linear_detrend(train_ot)
    f, P = periodogram(x, fs=fs, window="hann")
    mask = (f > 1.0 / p_hi) & (f <= 1.0 / p_lo)
    f_sel, P_sel = f[mask], P[mask]
    df = fs / len(x)
    distance = max(1, int(round(0.01 / df)))
    prom = float(P_sel.max()) * 1e-4
    peaks, _ = find_peaks(P_sel, distance=distance, prominence=prom)
    peak_f, peak_P = f_sel[peaks], P_sel[peaks]
    order = np.argsort(-peak_P)[:3]
    periods = 1.0 / peak_f[order]  # hours
    powers = peak_P[order]
    return periods.astype(np.float64), powers.astype(np.float64)


def to_hours(dates):
    return (dates - EPOCH).dt.total_seconds().to_numpy(dtype=np.float64) / 3600.0


def phase_features(t_hours, T):
    phase = 2.0 * np.pi * t_hours / T
    return np.stack([np.cos(phase), np.sin(phase)], axis=1).astype(np.float32)


def data_driven_timesim(t_q, t_mem, periods):
    """mean over periods of cos(2*pi*(tq - tm)/Tk), in [-1, 1]."""
    acc = np.zeros((len(t_q), len(t_mem)), dtype=np.float32)
    z_mem = None
    for T in periods:
        zq = phase_features(t_q, T)
        zm = phase_features(t_mem, T)
        acc += zq @ zm.T
    return acc / len(periods)


def annotate_period(T):
    labels = []
    for ref, nm in [(24.0, "24h"), (168.0, "168h"), (720.0, "720h")]:
        if abs(T - ref) / ref <= 0.15:
            labels.append("~" + nm)
    return ",".join(labels) if labels else "-"


# ---- val search / test apply ----

def search_val(S_p, B, shape_order, alpha_grid, chronos, futures, bank, contexts):
    best = None
    for alpha in alpha_grid:
        S = S_p if alpha == 0.0 else ((1 - alpha) * S_p + alpha * B)
        idx = select_topk(S, shape_order, alpha)
        sc = np.take_along_axis(S, idx, axis=-1)
        y_mem = mean_anchor(bank, contexts, idx, sc)
        for lam in LAMS:
            fused = (1 - lam) * chronos + lam * y_mem
            mse = float(((fused - futures) ** 2).mean())
            mae = float(np.abs(fused - futures).mean())
            if best is None or (mse, mae) < (best[2], best[3]):
                best = (alpha, lam, mse, mae)
    return best


def apply_test(S_p, B, shape_order, alpha, lam, chronos, futures, bank, contexts):
    S = S_p if alpha == 0.0 else ((1 - alpha) * S_p + alpha * B)
    idx = select_topk(S, shape_order, alpha)
    sc = np.take_along_axis(S, idx, axis=-1)
    y_mem = mean_anchor(bank, contexts, idx, sc)
    fused = (1 - lam) * chronos + lam * y_mem
    return dict(
        mem_mse=float(((y_mem - futures) ** 2).mean()),
        mem_mae=float(np.abs(y_mem - futures).mean()),
        fus_mse=float(((fused - futures) ** 2).mean()),
        fus_mae=float(np.abs(fused - futures).mean()),
    )


def analyze(name, cfg):
    dt = cfg["dt"]
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

    # ---- discover periods from TRAIN only ----
    periods, powers = discover_periods(name, cfg, train)

    # ---- shape similarity + calendar alignments ----
    S_p_val = residual_pearson(val_contexts, bank, period, trend_window)
    S_p_test = residual_pearson(test_contexts, bank, period, trend_window)
    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    aligns_val = build_alignments(val_dates, mem_dates, luts)
    aligns_test = build_alignments(test_dates, mem_dates, luts)
    o_shape_val = shape_tiebreak_order(S_p_val)
    o_shape_test = shape_tiebreak_order(S_p_test)

    chronos_test_mse = float(((y_chronos_test - test_futures) ** 2).mean())

    # ---- data-driven temporal similarity (continuous cosine) ----
    t_mem = to_hours(mem_dates)
    t_val = to_hours(val_dates)
    t_test = to_hours(test_dates)
    DDT_val = data_driven_timesim(t_val, t_mem, periods)
    DDT_test = data_driven_timesim(t_test, t_mem, periods)
    DDT_val = standardize(DDT_val, S_p_val)
    DDT_test = standardize(DDT_test, S_p_test)

    # ---- A. shape only ----
    _, lam_a, val_mse_a, _ = search_val(
        S_p_val, None, o_shape_val, [0.0], y_chronos_val, val_futures, bank, val_contexts)
    a_test = apply_test(S_p_test, None, o_shape_test, 0.0, lam_a,
                        y_chronos_test, test_futures, bank, test_contexts)

    # ---- B. calendar (frozen) ----
    keys, alpha_b, lam_b, cal_mse = CALENDAR[name]
    B_cal_val = standardize(combined_B(aligns_val, keys), S_p_val)
    B_cal_test = standardize(combined_B(aligns_test, keys), S_p_test)
    b_test = apply_test(S_p_test, B_cal_test, o_shape_test, alpha_b, lam_b,
                        y_chronos_test, test_futures, bank, test_contexts)

    # ---- C. data-driven (alpha x lambda on val) ----
    alpha_c, lam_c, val_mse_c, _ = search_val(
        S_p_val, DDT_val, o_shape_val, ALPHAS, y_chronos_val, val_futures, bank, val_contexts)
    c_test = apply_test(S_p_test, DDT_test, o_shape_test, alpha_c, lam_c,
                        y_chronos_test, test_futures, bank, test_contexts)

    def row(method, alpha, lam, val_mse, tst):
        return dict(
            dataset=name, method=method, alpha=alpha, lam=lam, val_mse=val_mse,
            mem_mse=tst["mem_mse"], mem_mae=tst["mem_mae"],
            fus_mse=tst["fus_mse"], fus_mae=tst["fus_mae"],
            vs_chronos=(chronos_test_mse - tst["fus_mse"]) / chronos_test_mse * 100,
            vs_calendar=(cal_mse - tst["fus_mse"]) / cal_mse * 100,
        )

    return dict(
        name=name, dt=dt, periods=periods, powers=powers,
        chronos_test_mse=chronos_test_mse, cal_mse=cal_mse,
        cal_combo="+".join({"hour": "Hour", "dow": "Weekday", "month": "Month"}[k] for k in keys),
        rows=[
            row("A.Shape-only", 0.0, lam_a, val_mse_a, a_test),
            row("B.Calendar", alpha_b, lam_b, float("nan"), b_test),
            row("C.Data-driven", alpha_c, lam_c, val_mse_c, c_test),
        ],
    )


def main():
    args = parse_args()
    names = [n.strip() for n in args.datasets.split(",")] if args.datasets else list(DATASETS)
    for n in names:
        if n not in DATASETS:
            raise SystemExit(f"unknown dataset {n!r}")
        if not os.path.exists(os.path.join(args.cache_dir, f"{n}_chronos.npz")):
            raise SystemExit(f"missing results/{n}_chronos.npz")

    results = {}
    for name in names:
        results[name] = analyze(name, DATASETS[name])

    # ---- discovered periods ----
    print("\n=== Auto-discovered periods (TRAIN only) ===")
    print(f"{'Dataset':>8} {'T1_h':>9} {'T1_pow':>10} {'T2_h':>9} {'T2_pow':>10} "
          f"{'T3_h':>9} {'T3_pow':>10} {'near':>12}")
    for name in names:
        r = results[name]
        p, pw = r["periods"], r["powers"]
        near = " | ".join(f"T{i+1}={annotate_period(p[i])}" for i in range(3))
        print(f"{name:>8} {p[0]:>9.1f} {pw[0]:>10.3e} {p[1]:>9.1f} {pw[1]:>10.3e} "
              f"{p[2]:>9.1f} {pw[2]:>10.3e} {near:>12}")

    # ---- cross-sampling consistency ----
    if "ETTh1" in results and "ETTm1" in results:
        print("\n=== Cross-sampling-rate consistency (ETTh vs ETTm, same physical series) ===")
        for h, m in [("ETTh1", "ETTm1"), ("ETTh2", "ETTm2")]:
            if h in results and m in results:
                ph, pm = results[h]["periods"], results[m]["periods"]
                print(f"{h} -> {m}: {[f'{x:.1f}' for x in ph]} vs {[f'{x:.1f}' for x in pm]}")

    # ---- per-method results ----
    print("\n=== Per-method results ===")
    hdr = (f"{'Dataset':>8} {'Method':>14} {'alpha':>6} {'lam':>5} {'val_MSE':>8} "
           f"{'mem_MSE':>9} {'mem_MAE':>8} {'fus_MSE':>9} {'fus_MAE':>8} "
           f"{'vsChronos%':>10} {'vsCal%':>8}")
    print(hdr)
    for name in names:
        for row in results[name]["rows"]:
            print(f"{row['dataset']:>8} {row['method']:>14} {row['alpha']:6.2f} {row['lam']:5.2f} "
                  f"{row['val_mse']:8.4f} {row['mem_mse']:9.4f} {row['mem_mae']:8.4f} "
                  f"{row['fus_mse']:9.4f} {row['fus_mae']:8.4f} {row['vs_chronos']:+9.2f}% "
                  f"{row['vs_calendar']:+7.2f}%")

    # ---- data-driven vs calendar ----
    print("\n=== Data-driven vs Calendar best ===")
    print(f"{'Dataset':>8} {'CalendarBest':>22} {'CalMSE':>9} {'DD_MSE':>9} {'vsCal%':>8}")
    for name in names:
        r = results[name]
        cal = next(x for x in r["rows"] if x["method"] == "B.Calendar")
        dd = next(x for x in r["rows"] if x["method"] == "C.Data-driven")
        print(f"{name:>8} {r['cal_combo']:>22} {r['cal_mse']:>9.4f} {dd['fus_mse']:>9.4f} "
              f"{dd['vs_calendar']:+7.2f}%")

    # ---- persist ----
    os.makedirs(args.cache_dir, exist_ok=True)
    period_rows = [
        dict(dataset=name, T1_hours=results[name]["periods"][0], T1_power=results[name]["powers"][0],
             T2_hours=results[name]["periods"][1], T2_power=results[name]["powers"][1],
             T3_hours=results[name]["periods"][2], T3_power=results[name]["powers"][2])
        for name in names
    ]
    pd.DataFrame(period_rows).to_csv(
        os.path.join(args.cache_dir, "data_driven_periods.csv"), index=False)
    long_rows = [row for name in names for row in results[name]["rows"]]
    pd.DataFrame(long_rows).to_csv(
        os.path.join(args.cache_dir, "data_driven_temporal_alignment.csv"), index=False)
    vs_rows = []
    for name in names:
        r = results[name]
        dd = next(x for x in r["rows"] if x["method"] == "C.Data-driven")
        vs_rows.append(dict(dataset=name, calendar_best=r["cal_combo"], calendar_mse=r["cal_mse"],
                            data_driven_mse=dd["fus_mse"], vs_calendar_pct=dd["vs_calendar"]))
    pd.DataFrame(vs_rows).to_csv(
        os.path.join(args.cache_dir, "data_driven_vs_calendar.csv"), index=False)
    print(f"\nSaved -> {args.cache_dir}/data_driven_periods.csv, "
          f"data_driven_temporal_alignment.csv, data_driven_vs_calendar.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated subset, e.g. ETTh1,ETTm1 (default: all four)")
    p.add_argument("--cache-dir", type=str, default="results")
    return p.parse_args()


if __name__ == "__main__":
    main()
