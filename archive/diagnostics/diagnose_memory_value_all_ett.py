"""Diagnose which prediction-time signals predict Memory's incremental value.

Oracle labels (diagnostic only, use y_true, NEVER fed to a model):

    C_i = MSE(y_chronos_i, y_true_i)
    M_i = MSE(y_memory_i, y_true_i)
    B_i = C_i - M_i            (>0: memory better than chronos)
    F_i = MSE(y_fusion_i, y_true_i)
    G_i = C_i - F_i            (>0: frozen fusion helps)

Candidate signals are all computed from existing cached forecasts / retrieval
results WITHOUT y_true:
    uncertainty          U_i  = mean_h(q0.9 - q0.1)          (quantile cache)
    disagreement         D_i  = mean_h|y_chronos - y_memory|
    D_level              abs(mean_h(y_chronos) - mean_h(y_memory))
    shape sim            residual-Pearson S_p (top1 / softmax-topK)
    temporal sim         raw calendar alignment B (top1 / softmax-topK)
    retrieval score      final S = (1-a)S_p + a*B_std (top1 / softmax-topK)
    concentration        max softmax weight / entropy / top1-top2 weight

Frozen best config per dataset (nothing re-selected, no gate fitted, no model
trained, retrieval unchanged):

    ETTh1: Hour+Month     alpha=0.75 lambda=0.20
    ETTh2: Full           alpha=1.00 lambda=0.30
    ETTm1: Hour+Weekday   alpha=0.75 lambda=0.10
    ETTm2: Hour           alpha=0.75 lambda=0.20
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

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

BEST = {
    "ETTh1": (["hour", "month"], 0.75, 0.20),
    "ETTh2": (["hour", "dow", "month"], 1.00, 0.30),
    "ETTm1": (["hour", "dow"], 0.75, 0.10),
    "ETTm2": (["hour"], 0.75, 0.20),
}

SIGNAL_ORDER = [
    "uncertainty", "disagreement", "D_level",
    "top1_shape_sim", "topK_shape_sim",
    "top1_temporal_sim", "topK_temporal_sim",
    "top1_score", "topK_score",
    "max_weight", "entropy", "weight_gap",
]

BUCKET_SIGNALS = [
    "uncertainty", "disagreement", "D_level",
    "top1_shape_sim", "top1_temporal_sim", "top1_score",
    "max_weight", "entropy",
]


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
    return B  # in [0,1]


def standardize(B, S_p):
    return (B - B.mean()) / B.std() * S_p.std() + S_p.mean()


def mean_anchor(bank, contexts, idx, scores):
    w = softmax(scores)
    fut = bank.futures[idx]
    ctx = bank.contexts[idx]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def corr(x, y, fn):
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(fn(x, y).statistic)


def compute(name):
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
    qn = np.load(f"results/{name}_quantiles.npz")
    U = (qn["q90"] - qn["q10"]).mean(axis=1)

    S_p = residual_pearson(test_contexts, bank, period, trend_window)
    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    aligns = build_alignments(test_dates, mem_dates, luts)
    B_time = combined_B(aligns, keys)                # raw calendar alignment in [0,1]
    B_std = standardize(B_time, S_p)
    S = (1 - alpha) * S_p + alpha * B_std

    idx = topk(S, K)
    sc = np.take_along_axis(S, idx, axis=-1)          # (n, K) sorted desc
    w = softmax(sc)                                   # (n, K)
    y_mem = mean_anchor(bank, test_contexts, idx, sc)
    fused = (1 - lam) * y_chronos + lam * y_mem

    # oracle labels (per-window MSE over the 96-step horizon)
    C = ((y_chronos - test_futures) ** 2).mean(axis=1)
    M = ((y_mem - test_futures) ** 2).mean(axis=1)
    F = ((fused - test_futures) ** 2).mean(axis=1)
    B_val = C - M
    G_val = C - F

    # prediction-time signals
    shape_topk = np.take_along_axis(S_p, idx, axis=-1)        # (n, K)
    temporal_topk = np.take_along_axis(B_time, idx, axis=-1)  # (n, K)
    entropy = -(w * np.log(w)).sum(axis=1)

    signals = {
        "uncertainty": U,
        "disagreement": np.abs(y_chronos - y_mem).mean(axis=1),
        "D_level": np.abs(y_chronos.mean(axis=1) - y_mem.mean(axis=1)),
        "top1_shape_sim": shape_topk[:, 0],
        "topK_shape_sim": (w * shape_topk).sum(axis=1),
        "top1_temporal_sim": temporal_topk[:, 0],
        "topK_temporal_sim": (w * temporal_topk).sum(axis=1),
        "top1_score": sc[:, 0],
        "topK_score": (w * sc).sum(axis=1),
        "max_weight": w[:, 0],
        "entropy": entropy,
        "weight_gap": w[:, 0] - w[:, 1],
    }

    return dict(name=name, C=C, M=M, F=F, B_val=B_val, G_val=G_val, signals=signals)


def print_basic(d):
    print(f"=== {d['name']}  (best={', '.join(BEST[d['name']][0])}, "
          f"alpha={BEST[d['name']][1]}, lambda={BEST[d['name']][2]}) ===")
    print(f"n_windows = {len(d['C'])}")
    print(f"mean(C)={d['C'].mean():.4f}  mean(M)={d['M'].mean():.4f}  mean(F)={d['F'].mean():.4f}")
    print(f"P(B>0)={float((d['B_val'] > 0).mean()):.4f}  P(G>0)={float((d['G_val'] > 0).mean()):.4f}")
    print(f"mean(B)={d['B_val'].mean():+.4f}  median(B)={np.median(d['B_val']):+.4f}")
    print(f"mean(G)={d['G_val'].mean():+.4f}  median(G)={np.median(d['G_val']):+.4f}")


def print_correlations(d):
    print(f"\n--- {d['name']}: signal correlations ---")
    print(f"{'signal':>18} {'Pearson_B':>10} {'Spearman_B':>11} {'Pearson_G':>10} {'Spearman_G':>11}")
    for sig in SIGNAL_ORDER:
        X = d["signals"][sig]
        pB = corr(X, d["B_val"], pearsonr)
        sB = corr(X, d["B_val"], spearmanr)
        pG = corr(X, d["G_val"], pearsonr)
        sG = corr(X, d["G_val"], spearmanr)
        print(f"{sig:>18} {pB:+10.4f} {sB:+11.4f} {pG:+10.4f} {sG:+11.4f}")


def bucket(d, sig):
    X = d["signals"][sig]
    if len(np.unique(X)) < 5:
        print(f"\n[{sig}]  (near-constant: {len(np.unique(X))} distinct value(s), bucketing skipped)")
        return
    e = np.quantile(X, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    e[0] = -np.inf
    e[-1] = np.inf
    bins = np.digitize(X, e) - 1
    print(f"\n[{sig}]")
    print(f"{'bucket':>6} {'n':>7} {'mean_sig':>9} {'ChronosMSE':>11} {'MemoryMSE':>10} "
          f"{'FusionMSE':>10} {'B':>9} {'G':>9} {'imp%':>8}")
    for b in range(5):
        m = bins == b
        if int(m.sum()) == 0:
            print(f"{'Q%d' % (b + 1):>6} {0:>7} {'-':>9}")
            continue
        cm = float(d["C"][m].mean())
        mm = float(d["M"][m].mean())
        fm = float(d["F"][m].mean())
        Bb = float(d["B_val"][m].mean())
        Gb = float(d["G_val"][m].mean())
        imp = (cm - fm) / cm * 100
        print(f"{'Q%d' % (b + 1):>6} {int(m.sum()):>7} {X[m].mean():9.4f} "
              f"{cm:11.4f} {mm:10.4f} {fm:10.4f} {Bb:+9.4f} {Gb:+9.4f} {imp:+7.2f}%")


def main():
    all_res = {}
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        d = compute(name)
        all_res[name] = d
        print_basic(d)
        print_correlations(d)
        for sig in BUCKET_SIGNALS:
            bucket(d, sig)

    # ---- section 7: best signal per dataset ----
    print("\n\n=== SUMMARY: best signal per dataset (by |Spearman|) ===")
    print(f"{'Dataset':>8} {'best_for_B':>18} {'Spearman_B':>11} {'best_for_G':>18} {'Spearman_G':>11}")
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        d = all_res[name]
        def best_by(target):
            best_sig, best_v = None, -1.0
            for sig in SIGNAL_ORDER:
                v = abs(corr(d["signals"][sig], target, spearmanr))
                if not np.isnan(v) and v > best_v:
                    best_sig, best_v = sig, v
            return best_sig, corr(d["signals"][best_sig], target, spearmanr)
        sb, sbv = best_by(d["B_val"])
        sg, sgv = best_by(d["G_val"])
        print(f"{name:>8} {sb:>18} {sbv:+11.4f} {sg:>18} {sgv:+11.4f}")

    # ---- cross-dataset Spearman(G) table ----
    print("\n=== SUMMARY: cross-dataset Spearman(G) ===")
    header = f"{'signal':>18} " + " ".join(f"{name:>11}" for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    print(header)
    for sig in SIGNAL_ORDER:
        row = f"{sig:>18} "
        for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
            v = corr(all_res[name]["signals"][sig], all_res[name]["G_val"], spearmanr)
            row += f"{v:+11.4f} "
        print(row)


if __name__ == "__main__":
    main()
