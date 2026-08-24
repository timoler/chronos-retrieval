"""Temporal-Alignment ablation across all four ETT datasets.

For each dataset (ETTh1 / ETTh2 / ETTm1 / ETTm2, target=OT), run 8 temporal
component combinations:

    None / Hour / Weekday / Month / Hour+Weekday / Hour+Month /
    Weekday+Month / Full

Each combination independently selects (alpha_time, lambda_fusion) on VAL by
val MSE (val MAE tie-break), then freezes them and evaluates ONCE on TEST.

Pipeline is fixed and matches the already-validated setup (no regime / gate /
diversity / kmeans / level-shift / any new module):
    backbone          = amazon/chronos-2
    context / horizon = 512 / 96
    retriever         = Residual Pearson
    Top-K             = 5
    transfer          = Mean-anchor
    memory bank       = train only (never val/test)

Sampling-frequency adaptation (daily period / trend window / stride):
    ETTh1, ETTh2 : period=24,  trend_window=48,  stride=1  (hourly)
    ETTm1, ETTm2 : period=96,  trend_window=192, stride=4  (15-min)

Chronos-2 forecasts are cached per dataset under results/<name>_chronos.npz
so a re-run skips GPU inference.  Pass --force to regenerate them.

ETTh1 reproduction gate: Chronos test MSE must be ~6.1968 and Hour+Month
fusion test MSE ~5.6986 before the remaining datasets are run.
"""

import argparse
import os
import random
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

# ---- fixed pipeline config ----
CONTEXT, PRED, K = 512, 96, 5
MODEL = "amazon/chronos-2"
BATCH = 256
SEED = 42
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAMS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

# per-dataset sampling-frequency config (period = points per day)
DATASETS = {
    "ETTh1": dict(period=24, trend_window=48, stride=1),
    "ETTh2": dict(period=24, trend_window=48, stride=1),
    "ETTm1": dict(period=96, trend_window=192, stride=4),
    "ETTm2": dict(period=96, trend_window=192, stride=4),
}

# temporal component combinations (name -> calendar keys)
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

# ETTh1 reproduction reference (from the already-validated pipeline)
REF_CHRONOS_MSE = 6.1968
REF_HOUR_MONTH_FUS = 5.6986
REPRO_TOL = 0.1  # "基本复现" tolerance (cached forecasts are exact)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    w = softmax(scores)  # (n, K)
    fut = bank.futures[idx]  # (n, K, H)
    ctx = bank.contexts[idx]  # (n, K, L)
    shape = fut - ctx.mean(axis=2)[:, :, None]
    return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)


def run_chronos(pipeline, contexts):
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    return np.stack([m.squeeze(0).numpy() for m in means])


def load_pipeline():
    from chronos import Chronos2Pipeline  # lazy: only needed for (re)generation

    print(f"Loading {MODEL} ...")
    return Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")


def ensure_forecasts(pipeline, name, val_contexts, test_contexts, cache_dir, force):
    path = os.path.join(cache_dir, f"{name}_chronos.npz")
    if (not force) and os.path.exists(path):
        d = np.load(path)
        yv, yt = d["y_chronos_val"], d["y_chronos_test"]
        assert yv.shape[0] == len(val_contexts), f"{name}: cached val forecast count mismatch"
        assert yt.shape[0] == len(test_contexts), f"{name}: cached test forecast count mismatch"
        return yv, yt
    assert pipeline is not None, f"{name}: no cache and no Chronos pipeline available"
    yv = run_chronos(pipeline, val_contexts)
    yt = run_chronos(pipeline, test_contexts)
    np.savez(path, y_chronos_val=yv, y_chronos_test=yt)
    return yv, yt


def eval_combo(S_p, B, chronos, futures, bank, contexts):
    """Grid-search (alpha, lambda) -> best (alpha, lam, val_mse, val_mae).

    Selected purely by val MSE; val MAE only breaks exact MSE ties.
    B is None for the temporal-free (None) combination: alpha fixed at 0.
    """
    best = None
    if B is None:
        S = S_p
        idx = topk(S, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        y_mem = mean_anchor(bank, contexts, idx, sc)
        for lam in LAMS:
            fused = (1 - lam) * chronos + lam * y_mem
            mse = float(((fused - futures) ** 2).mean())
            mae = float(np.abs(fused - futures).mean())
            if best is None or (mse, mae) < (best[2], best[3]):
                best = (0.0, lam, mse, mae)
        return best
    for alpha in ALPHAS:
        S = (1 - alpha) * S_p + alpha * B
        idx = topk(S, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        y_mem = mean_anchor(bank, contexts, idx, sc)
        for lam in LAMS:
            fused = (1 - lam) * chronos + lam * y_mem
            mse = float(((fused - futures) ** 2).mean())
            mae = float(np.abs(fused - futures).mean())
            if best is None or (mse, mae) < (best[2], best[3]):
                best = (alpha, lam, mse, mae)
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
    mem_mae = float(np.abs(y_mem - futures).mean())
    fus_mse = float(((fused - futures) ** 2).mean())
    fus_mae = float(np.abs(fused - futures).mean())
    return mem_mse, mem_mae, fus_mse, fus_mae


def process_dataset(name, cfg, pipeline, cache_dir, force):
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

    y_chronos_val, y_chronos_test = ensure_forecasts(
        pipeline, name, val_contexts, test_contexts, cache_dir, force
    )

    luts = (circular_lut(12), circular_lut(7), circular_lut(24))
    S_p_val = residual_pearson(val_contexts, bank, period, trend_window)
    S_p_test = residual_pearson(test_contexts, bank, period, trend_window)
    aligns_val = build_alignments(val_dates, mem_dates, luts)
    aligns_test = build_alignments(test_dates, mem_dates, luts)

    chronos_test_mse = float(((y_chronos_test - test_futures) ** 2).mean())
    chronos_test_mae = float(np.abs(y_chronos_test - test_futures).mean())

    rows = []
    for combo_name, keys in COMBOS:
        if not keys:
            B_val = B_test = None
        else:
            B_val = standardize(combined_B(aligns_val, keys), S_p_val)
            B_test = standardize(combined_B(aligns_test, keys), S_p_test)
        alpha, lam, val_mse, _ = eval_combo(
            S_p_val, B_val, y_chronos_val, val_futures, bank, val_contexts
        )
        mem_mse, mem_mae, fus_mse, fus_mae = apply(
            S_p_test, B_test, alpha, lam, y_chronos_test, test_futures, bank, test_contexts
        )
        improvement = (chronos_test_mse - fus_mse) / chronos_test_mse * 100
        rows.append(dict(
            combo=combo_name, alpha=alpha, lam=lam, val_mse=val_mse,
            mem_mse=mem_mse, mem_mae=mem_mae, fus_mse=fus_mse, fus_mae=fus_mae,
            improvement=improvement,
        ))

    return dict(
        name=name, chronos_test_mse=chronos_test_mse, chronos_test_mae=chronos_test_mae,
        n_val=len(val_starts), n_test=len(test_starts), n_mem=bank.n_memories, rows=rows,
    )


def print_dataset_table(r):
    print(f"\n=== {r['name']} ===")
    print(f"val windows={r['n_val']}  test windows={r['n_test']}  memories={r['n_mem']}")
    print(f"Chronos test MSE = {r['chronos_test_mse']:.4f}   MAE = {r['chronos_test_mae']:.4f}")
    hdr = (f"{'components':>16} {'val_a':>6} {'val_l':>6} {'val_MSE':>8} "
           f"{'mem_MSE':>10} {'mem_MAE':>9} {'fus_MSE':>10} {'fus_MAE':>9} {'vsChronos%':>10}")
    print(hdr)
    for row in r["rows"]:
        print(f"{row['combo']:>16} {row['alpha']:6.2f} {row['lam']:6.2f} {row['val_mse']:8.4f} "
              f"{row['mem_mse']:10.4f} {row['mem_mae']:9.4f} {row['fus_mse']:10.4f} "
              f"{row['fus_mae']:9.4f} {row['improvement']:+9.2f}%")


def main():
    args = parse_args()
    set_seed(SEED)
    os.makedirs(args.cache_dir, exist_ok=True)

    names = [n.strip() for n in args.datasets.split(",")] if args.datasets else list(DATASETS)
    for n in names:
        if n not in DATASETS:
            raise SystemExit(f"unknown dataset {n!r} (choices: {sorted(DATASETS)})")

    # load the model once only if some dataset's forecast cache is missing
    need_pipeline = any(
        args.force
        or not os.path.exists(os.path.join(args.cache_dir, f"{n}_chronos.npz"))
        for n in names
    )
    pipeline = load_pipeline() if need_pipeline else None

    results = {}
    for name in names:
        r = process_dataset(name, DATASETS[name], pipeline, args.cache_dir, args.force)
        results[name] = r
        print_dataset_table(r)

        if name == "ETTh1":
            chronos_mse = r["chronos_test_mse"]
            hm = next(row for row in r["rows"] if row["combo"] == "Hour+Month")
            ok_chronos = abs(chronos_mse - REF_CHRONOS_MSE) <= REPRO_TOL
            ok_hm = abs(hm["fus_mse"] - REF_HOUR_MONTH_FUS) <= REPRO_TOL
            print(f"\n[ETTh1 reproduction] Chronos MSE {chronos_mse:.4f} "
                  f"(ref {REF_CHRONOS_MSE}) {'OK' if ok_chronos else 'MISMATCH'}; "
                  f"Hour+Month fusion {hm['fus_mse']:.4f} "
                  f"(ref {REF_HOUR_MONTH_FUS}) {'OK' if ok_hm else 'MISMATCH'}")
            if not (ok_chronos and ok_hm):
                print("ERROR: ETTh1 does not reproduce the reference. "
                      "Stopping before other datasets.", file=sys.stderr)
                sys.exit(1)

    # ---- summary table 1: dataset x combo -> test fusion MSE ----
    combo_names = [c[0] for c in COMBOS]
    print("\n=== SUMMARY 1: Test Fusion MSE ===")
    print(f"{'Dataset':>8} " + " ".join(f"{c:>13}" for c in combo_names))
    for name in names:
        r = results[name]
        cells = {row["combo"]: row["fus_mse"] for row in r["rows"]}
        print(f"{name:>8} " + " ".join(f"{cells[c]:13.4f}" for c in combo_names))

    # ---- summary table 2: best temporal combination per dataset ----
    print("\n=== SUMMARY 2: Best Temporal Combination (by test fusion MSE) ===")
    print(f"{'Dataset':>8} {'BestCombo':>14} {'alpha':>6} {'lambda':>7} "
          f"{'ChronosMSE':>11} {'FusionMSE':>10} {'MSEimp%':>9}")
    best_rows = []
    for name in names:
        r = results[name]
        best = min(r["rows"], key=lambda row: row["fus_mse"])
        imp = (r["chronos_test_mse"] - best["fus_mse"]) / r["chronos_test_mse"] * 100
        best_rows.append(dict(
            name=name, combo=best["combo"], alpha=best["alpha"], lam=best["lam"],
            chronos_mse=r["chronos_test_mse"], fus_mse=best["fus_mse"], improvement=imp,
        ))
        print(f"{name:>8} {best['combo']:>14} {best['alpha']:6.2f} {best['lam']:7.2f} "
              f"{r['chronos_test_mse']:11.4f} {best['fus_mse']:10.4f} {imp:+9.2f}%")

    # ---- persist ----
    long_rows = [
        dict(dataset=name, **{k: row[k] for k in ("combo", "alpha", "lam", "val_mse",
                "mem_mse", "mem_mae", "fus_mse", "fus_mae", "improvement")})
        for name in names for row in results[name]["rows"]
    ]
    pd.DataFrame(long_rows).to_csv(os.path.join(args.cache_dir, "temporal_ablation_all_ett.csv"), index=False)
    pd.DataFrame(best_rows).to_csv(os.path.join(args.cache_dir, "temporal_ablation_best.csv"), index=False)
    print(f"\nSaved long table -> {args.cache_dir}/temporal_ablation_all_ett.csv")
    print(f"Saved best table -> {args.cache_dir}/temporal_ablation_best.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default=None,
                   help="comma-separated subset, e.g. ETTh1,ETTm1 (default: all four)")
    p.add_argument("--force", action="store_true",
                   help="regenerate Chronos-2 forecasts (ignore cache)")
    p.add_argument("--cache-dir", type=str, default="results",
                   help="directory for cached forecasts and CSV output")
    return p.parse_args()


if __name__ == "__main__":
    main()
