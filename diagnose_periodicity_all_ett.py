"""Train-only periodicity diagnosis of OT across all four ETT datasets.

Uses ONLY the train split (dates 2016-07-01 .. 2017-07-01, identical to every
existing experiment) to check whether the temporal components Hour / Weekday /
Month are explainable by the series' own periodic structure.

Read-only diagnostic: no retrieval / Chronos / mean-anchor / alpha / lambda is
touched, no model trained, no selector designed.

Per dataset it reports
  * ACF (raw + linearly-detrended) at the daily / weekly / ~monthly lags, with a
    local prominence score to separate a real period peak from slow decay;
  * periodogram (raw + detrended) with a Top-20 local-peak list;
  * a check of the three scales of interest (24h / 168h / 720h);
  * calendar-month seasonality indicators (between-month vs within-month).
and saves detrended ACF / periodogram plots plus one 4-dataset summary figure.

Sampling: ETTh1/ETTh2 are hourly (dt=1h); ETTm1/ETTm2 are 15-min (dt=0.25h).
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import periodogram, find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- per-dataset config (dt in hours; period_range in hours) ----
DATASETS = {
    "ETTh1": dict(dt=1.0, k_prom=6, period_range=(6.0, 4000.0)),
    "ETTh2": dict(dt=1.0, k_prom=6, period_range=(6.0, 4000.0)),
    "ETTm1": dict(dt=0.25, k_prom=24, period_range=(1.5, 4000.0)),
    "ETTm2": dict(dt=0.25, k_prom=24, period_range=(1.5, 4000.0)),
}

# read-only reference only (never feeds the computation)
BEST_COMBO = {
    "ETTh1": "Hour+Month",
    "ETTh2": "Full (Hour+Weekday+Month)",
    "ETTm1": "Hour+Weekday",
    "ETTm2": "Hour",
}

# three temporal scales (target period in hours)
SCALES = [("Daily", 24.0), ("Weekly", 168.0), ("Monthly", 720.0)]


def load_train(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    return dates[train_mask], ot[train_mask].astype(np.float64)


def linear_detrend(x):
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x), dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - (A @ coef)


def compute_acf(x, nlags):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    out = np.empty(nlags + 1, dtype=np.float64)
    out[0] = 1.0
    for lag in range(1, nlags + 1):
        out[lag] = float(np.dot(x[:-lag], x[lag:])) / denom
    return out


def acf_prominence(acf, p, k):
    """ACF[p] minus the mean of its +/-k neighbors (local prominence)."""
    lo = acf[max(0, p - k) : p]
    hi = acf[p + 1 : p + 1 + k]
    nb = np.concatenate([lo, hi])
    return float(acf[p] - nb.mean())


def fmt_period_hours(h):
    if h >= 24:
        return f"{h:.0f}h ({h/24:.1f}d)"
    return f"{h:.1f}h"


def analyze(name, cfg):
    dt = cfg["dt"]
    fs = 1.0 / dt
    k_prom = cfg["k_prom"]
    p_lo, p_hi = cfg["period_range"]

    train_dates, x_raw = load_train(f"data/{name}.csv")
    n = len(x_raw)
    start_ts = train_dates.iloc[0]
    end_ts = train_dates.iloc[-1]

    # daily / weekly / ~monthly lags in samples
    daily_lag = int(round(24 / dt))
    weekly_lag = int(round(168 / dt))
    monthly_lag = int(round(720 / dt))
    key_lags = [daily_lag, 2 * daily_lag, 3 * daily_lag, weekly_lag, 2 * weekly_lag, monthly_lag]

    x_det = linear_detrend(x_raw)

    # ---- ACF (raw + detrended) ----
    max_lag = monthly_lag + k_prom
    acf_raw = compute_acf(x_raw, max_lag)
    acf_det = compute_acf(x_det, max_lag)

    # ---- periodogram (raw + detrended) ----
    f_raw, P_raw = periodogram(x_raw, fs=fs, window="hann")
    f_det, P_det = periodogram(x_det, fs=fs, window="hann")
    total_nonzero = float(P_det[f_det > 0].sum())  # normalization over detrended spectrum

    # restrict to period range (ignore DC)
    f_lo, f_hi = 1.0 / p_hi, 1.0 / p_lo
    mask = (f_det > f_lo) & (f_det <= f_hi)
    f_sel, P_sel = f_det[mask], P_det[mask]
    df = fs / n
    distance_bins = max(1, int(round(0.01 / df)))
    prom_thresh = float(P_sel.max()) * 1e-4
    peaks, props = find_peaks(P_sel, distance=distance_bins, prominence=prom_thresh)
    peak_f = f_sel[peaks]
    peak_P = P_sel[peaks]
    peak_prom = props["prominences"]
    peak_period = 1.0 / peak_f  # hours
    peak_norm = peak_P / total_nonzero

    order = np.argsort(-peak_P)  # descending power
    top20 = order[:20]

    # ---- scale analysis (nearest local peak to each target period) ----
    scale_rows = []
    for scale_name, target in SCALES:
        if len(peaks) == 0:
            scale_rows.append(dict(scale=scale_name, target=target, nearest=np.nan,
                                   dist=np.nan, norm=np.nan, prom=np.nan, top20=False))
            continue
        j = int(np.argmin(np.abs(peak_period - target)))
        scale_rows.append(dict(
            scale=scale_name, target=target, nearest=float(peak_period[j]),
            dist=float(peak_period[j] - target), norm=float(peak_norm[j]),
            prom=float(peak_prom[j]), top20=bool(j in top20),
        ))

    # ---- monthly seasonality (calendar month, train only) ----
    months = train_dates.dt.month.to_numpy()
    month_means = np.array([x_raw[months == m].mean() for m in range(1, 13)])
    month_vars = np.array([x_raw[months == m].var() for m in range(1, 13)])
    monthly_mean_variance = float(np.var(month_means))
    month_mean_range = float(month_means.max() - month_means.min())
    between = float(np.var(month_means))
    within = float(np.mean(month_vars))
    ratio = between / within if within > 0 else float("nan")

    return dict(
        name=name, dt=dt, fs=fs, n=n, start_ts=start_ts, end_ts=end_ts,
        daily_lag=daily_lag, weekly_lag=weekly_lag, monthly_lag=monthly_lag,
        key_lags=key_lags, acf_raw=acf_raw, acf_det=acf_det, k_prom=k_prom,
        f_det=f_det, P_det=P_det, f_sel=f_sel, P_sel=P_sel,
        peak_f=peak_f, peak_P=peak_P, peak_prom=peak_prom, peak_period=peak_period,
        peak_norm=peak_norm, order=order, top20=top20,
        scale_rows=scale_rows,
        monthly_mean_variance=monthly_mean_variance, month_mean_range=month_mean_range,
        between_within_ratio=ratio,
        month_means=month_means,
    )


def print_dataset(r):
    print(f"\n{'='*72}\n=== {r['name']} ===")
    print(f"sampling dt = {r['dt']:.2f} h   (fs = {r['fs']:.2f} samples/hour)")
    print(f"train: {r['start_ts']} .. {r['end_ts']}   length = {r['n']}")

    print("\n--- ACF at key lags (raw / detrended / prominence) ---")
    print(f"{'lag':>6} {'period':>14} {'ACF_raw':>10} {'ACF_det':>10} {'prom_det':>10}")
    for lag in r["key_lags"]:
        ph = lag * r["dt"]
        prom = acf_prominence(r["acf_det"], lag, r["k_prom"])
        print(f"{lag:>6} {fmt_period_hours(ph):>14} {r['acf_raw'][lag]:>+10.4f} "
              f"{r['acf_det'][lag]:>+10.4f} {prom:>+10.4f}")

    print("\n--- Top-20 local peaks (detrended periodogram) ---")
    print(f"{'rank':>4} {'freq(1/h)':>10} {'period':>14} {'period_d':>9} {'power':>12} {'norm_pow':>10}")
    for rank, j in enumerate(r["top20"]):
        per = r["peak_period"][j]
        print(f"{rank+1:>4} {r['peak_f'][j]:>10.6f} {fmt_period_hours(per):>14} "
              f"{per/24:>9.2f} {r['peak_P'][j]:>12.3e} {r['peak_norm'][j]:>10.4f}")

    print("\n--- Scale check (nearest local peak) ---")
    print(f"{'scale':>9} {'target':>7} {'nearest':>8} {'dist_h':>9} {'norm_pow':>9} {'prom':>10} {'top20':>6}")
    for s in r["scale_rows"]:
        print(f"{s['scale']:>9} {s['target']:>7.0f} {s['nearest']:>8.1f} {s['dist']:>+9.1f} "
              f"{s['norm']:>9.4f} {s['prom']:>10.3e} {str(s['top20']):>6}")

    print("\n--- Monthly seasonality ---")
    print(f"month_mean_range        = {r['month_mean_range']:.4f}")
    print(f"monthly_mean_variance   = {r['monthly_mean_variance']:.4f}")
    print(f"between/within ratio    = {r['between_within_ratio']:.4f}")
    print(f"12 month means: " + " ".join(f"{v:.2f}" for v in r["month_means"]))


def print_summary_tables(results):
    names = list(results)

    print("\n\n" + "=" * 72 + "\n=== TABLE 1: ACF (detrended) ===")
    print(f"{'Dataset':>8} {'ACF_daily':>10} {'ACF_weekly':>11} {'ACF_month':>10} "
          f"{'prom_daily':>11} {'prom_weekly':>12} {'prom_month':>11}")
    for name in names:
        r = results[name]
        dl, wl, ml = r["daily_lag"], r["weekly_lag"], r["monthly_lag"]
        pd_ = acf_prominence(r["acf_det"], dl, r["k_prom"])
        pw_ = acf_prominence(r["acf_det"], wl, r["k_prom"])
        pm_ = acf_prominence(r["acf_det"], ml, r["k_prom"])
        print(f"{name:>8} {r['acf_det'][dl]:>+10.4f} {r['acf_det'][wl]:>+11.4f} "
              f"{r['acf_det'][ml]:>+10.4f} {pd_:>+11.4f} {pw_:>+12.4f} {pm_:>+11.4f}")

    print("\n=== TABLE 2: Periodogram (detrended) ===")
    print(f"{'Dataset':>8} {'DailyPeak_h':>12} {'DailyPow':>11} {'WeeklyPeak_h':>13} "
          f"{'WeeklyPow':>11} {'MonthPeak_h':>12} {'MonthPow':>11}")
    for name in names:
        r = results[name]
        def nearest(target):
            j = int(np.argmin(np.abs(r["peak_period"] - target)))
            return r["peak_period"][j], r["peak_norm"][j]
        dp, dn = nearest(24.0)
        wp, wn = nearest(168.0)
        mp, mn = nearest(720.0)
        print(f"{name:>8} {dp:>12.1f} {dn:>11.4f} {wp:>13.1f} {wn:>11.4f} "
              f"{mp:>12.1f} {mn:>11.4f}")

    print("\n=== TABLE 3: Seasonal-state (month) ===")
    print(f"{'Dataset':>8} {'MonthMeanRange':>15} {'MonthMeanVar':>13} {'Between/Within':>15}")
    for name in names:
        r = results[name]
        print(f"{name:>8} {r['month_mean_range']:>15.4f} {r['monthly_mean_variance']:>13.4f} "
              f"{r['between_within_ratio']:>15.4f}")


def nearest_peak(r, target_hours):
    j = int(np.argmin(np.abs(r["peak_period"] - target_hours)))
    return r["peak_period"][j], r["peak_norm"][j]


def evidence_label(score, strong, weak):
    if score >= strong:
        return "Strong"
    if score >= weak:
        return "Medium"
    return "Weak"


def print_comparison(results):
    """Tentative Strong/Medium/Weak from TRAIN-only metrics.

    Daily   = normalized power of the 24h periodogram peak (a sharp peak exists
              at 24h for every dataset; magnitude differs).
    Weekly  = distance of the nearest periodogram peak to 168h (none is close).
    Monthly = between-month / within-month variance ratio (annual seasonality).

    Thresholds are heuristic and documented inline; the raw tables above are the
    authoritative result, not these labels.
    """
    print("\n\n" + "=" * 72 + "\n=== READ-ONLY comparison vs observed best combo ===")
    print("(labels are tentative & threshold-based; raw numbers follow each label)")
    print(f"{'Dataset':>8} {'ObservedCombo':>26} {'Daily':>16} {'Weekly':>16} {'Monthly':>16}")
    for name, r in results.items():
        dperiod, dnorm = nearest_peak(r, 24.0)
        wperiod, _ = nearest_peak(r, 168.0)
        wdist = abs(wperiod - 168.0)
        bwr = r["between_within_ratio"]

        daily = evidence_label(dnorm, 0.02, 0.005)
        weekly = "Strong" if wdist <= 12.0 else ("Medium" if wdist <= 48.0 else "Weak")
        monthly = evidence_label(bwr, 2.0, 1.0)

        print(f"{name:>8} {BEST_COMBO[name]:>26} "
              f"{daily:>7} ({dnorm:.4f}) {weekly:>7} ({wdist:.0f}h) {monthly:>8} ({bwr:.2f})")


def plot_dataset(r, outdir):
    name = r["name"]
    dl, wl, ml = r["daily_lag"], r["weekly_lag"], r["monthly_lag"]

    # ---- detrended ACF ----
    fig, ax = plt.subplots(figsize=(8, 3.5))
    lags = np.arange(len(r["acf_det"]))
    ax.plot(lags, r["acf_det"], lw=1.0, color="#1f77b4")
    for lag, label, color in [(dl, "Daily", "#d62728"), (wl, "Weekly", "#2ca02c"),
                              (ml, "~Monthly", "#9467bd")]:
        ax.axvline(lag, color=color, ls="--", lw=1.0, alpha=0.8)
        ax.text(lag, ax.get_ylim()[1] * 0.9, label, rotation=90, color=color,
                fontsize=8, va="top", ha="right")
    ax.set_title(f"{name}: detrended ACF (train OT)")
    ax.set_xlabel("lag (samples)")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_acf.png"), dpi=130)
    plt.close(fig)

    # ---- detrended periodogram ----
    fig, ax = plt.subplots(figsize=(8, 3.5))
    period_h = 1.0 / np.where(r["f_det"] > 0, r["f_det"], np.nan)
    ok = ~np.isnan(period_h)
    ax.plot(period_h[ok], r["P_det"][ok], lw=1.0, color="#1f77b4")
    ax.set_xscale("log")
    for period, label, color in [(24.0, "Daily", "#d62728"), (168.0, "Weekly", "#2ca02c"),
                                 (720.0, "~Monthly", "#9467bd")]:
        ax.axvline(period, color=color, ls="--", lw=1.0, alpha=0.8)
        ax.text(period, ax.get_ylim()[1] * 0.9, label, rotation=90, color=color,
                fontsize=8, va="top", ha="right")
    ax.set_title(f"{name}: detrended periodogram (train OT)")
    ax.set_xlabel("period (hours, log)")
    ax.set_ylabel("power")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_periodogram.png"), dpi=130)
    plt.close(fig)


def plot_summary(results, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, name in zip(axes.ravel(), results):
        r = results[name]
        period_h = 1.0 / np.where(r["f_det"] > 0, r["f_det"], np.nan)
        ok = ~np.isnan(period_h)
        ax.plot(period_h[ok], r["P_det"][ok], lw=1.0, color="#1f77b4")
        ax.set_xscale("log")
        for period, color in [(24.0, "#d62728"), (168.0, "#2ca02c"), (720.0, "#9467bd")]:
            ax.axvline(period, color=color, ls="--", lw=1.0, alpha=0.7)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Detrended periodogram (train OT) — red=Daily, green=Weekly, purple=~Monthly")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, "periodicity_summary_all_ett.png"), dpi=130)
    plt.close(fig)


def main():
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    results = {}
    for name in DATASETS:
        results[name] = analyze(name, DATASETS[name])
        print_dataset(results[name])
        plot_dataset(results[name], outdir)

    print_summary_tables(results)
    print_comparison(results)
    plot_summary(results, outdir)
    print(f"\nSaved plots -> {outdir}/<dataset>_acf.png, <dataset>_periodogram.png, "
          f"periodicity_summary_all_ett.png")


if __name__ == "__main__":
    main()
