"""Deterministic Top-K tie-break: rerun the ETTh2 temporal ablation.

Root cause of the ETTh2 reproducibility gap: with ``Full + alpha=1.00`` the
retrieval score reduces to pure calendar alignment
(``score = temporal_sim``), which is highly discrete, so many memory
candidates share an *exact* final score.  ``np.argpartition`` resolves those
ties arbitrarily (unstable across numpy/BLAS builds), producing 25.5044 vs
25.5249.

This script keeps the retrieval score definition UNCHANGED but replaces the
Top-K selection with a deterministic lexicographic ranking:

    Priority 1: final_score      descending
    Priority 2: shape_similarity descending   (tie-break ONLY, not added back)
    Priority 3: memory_index     ascending

``shape_similarity`` (residual-Pearson ``S_p``) is used strictly as a
tie-breaker; at ``alpha=1.0`` the final score is still exactly
``temporal_similarity``.

Nothing else changes: alpha grid, lambda grid, temporal formula, Hour/Weekday/
Month, residual-Pearson, mean-anchor, Top-K count, Chronos, splits, memory bank
are all untouched.  Only ETTh2 is run; output goes to a new CSV so the old
results files are preserved.

Modes:
    (default)      full 8-combo temporal ablation -> ETTh2_temporal_ablation_deterministic.csv
    --repro N      N in-process repeats of Full/alpha=1.00/lambda=0.30 (reproducibility check)
    --tiecheck     tie-break statistics + 10 tie cases
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

DATASETS = {"ETTh2": dict(period=24, trend_window=48, stride=1)}

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

# Chronos baseline must stay unchanged; abort if it drifts.
REF_CHRONOS_MSE = 26.9971
REF_CHRONOS_MAE = 3.8534
REPRO_TOL = 1e-3

FULL_KEYS = ["hour", "dow", "month"]
FULL_ALPHA, FULL_LAM = 1.0, 0.30


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
    """Stable argsort of -shape_sim: shape_sim desc, ties -> ascending index."""
    return np.argsort(-shape_sim, axis=-1, kind="stable")


def topk_deterministic(scores, shape_order, k):
    """Deterministic Top-K: final_score desc, then shape_sim desc, then index asc.

    ``shape_order`` is the precomputed tie-break ordering (from
    ``shape_tiebreak_order(S_p)``).  Sorting the scores stably within that
    ordering yields exactly the lexicographic ranking; ``scores`` itself is
    never modified.
    """
    score_tb = np.take_along_axis(scores, shape_order, axis=-1)
    o = np.argsort(-score_tb, axis=-1, kind="stable")
    return np.take_along_axis(shape_order, o, axis=-1)[..., :k]


def prepare(name):
    """Shared setup: data, memory bank, windows, chronos, S_p, alignments."""
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
        name=name, cfg=cfg, bank=bank, mem_dates=mem_dates,
        val_starts=val_starts, val_contexts=val_contexts, val_futures=val_futures,
        val_dates=val_dates,
        test_starts=test_starts, test_contexts=test_contexts, test_futures=test_futures,
        test_dates=test_dates,
        y_chronos_val=y_chronos_val, y_chronos_test=y_chronos_test,
        S_p_val=S_p_val, S_p_test=S_p_test,
        aligns_val=aligns_val, aligns_test=aligns_test,
    )


def fused_scores(S_p, B, alpha):
    if B is None:
        return S_p
    return (1 - alpha) * S_p + alpha * B


def eval_combo(P, o_shape_val, alpha_grid, chronos, futures, bank, contexts):
    """VAL grid search (alpha x lambda); deterministic top-k; val-MSE / val-MAE."""
    S_p, B = P["S_p"], P["B"]
    best = None
    for alpha in alpha_grid:
        S = fused_scores(S_p, B, alpha)
        idx = topk_deterministic(S, o_shape_val, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        y_mem = mean_anchor(bank, contexts, idx, sc)
        for lam in LAMS:
            fused = (1 - lam) * chronos + lam * y_mem
            mse = float(((fused - futures) ** 2).mean())
            mae = float(np.abs(fused - futures).mean())
            if best is None or (mse, mae) < (best[2], best[3]):
                best = (alpha, lam, mse, mae)
    return best


def apply(P, o_shape_test, alpha, lam, chronos, futures, bank, contexts):
    S_p, B = P["S_p"], P["B"]
    S = fused_scores(S_p, B, alpha)
    idx = topk_deterministic(S, o_shape_test, K)
    sc = np.take_along_axis(S, idx, axis=-1)
    y_mem = mean_anchor(bank, contexts, idx, sc)
    fused = (1 - lam) * chronos + lam * y_mem
    mem_mse = float(((y_mem - futures) ** 2).mean())
    mem_mae = float(np.abs(y_mem - futures).mean())
    fus_mse = float(((fused - futures) ** 2).mean())
    fus_mae = float(np.abs(fused - futures).mean())
    return idx, y_mem, mem_mse, mem_mae, fus_mse, fus_mae


def topk_checksum(idx):
    w = np.array([1, 8191, 8191 ** 2, 8191 ** 3, 8191 ** 4], dtype=np.int64)
    return int(np.sum(idx.astype(np.int64) * w))


# --------------------------------------------------------------------------- #
# section 4: reproducibility check (Full / alpha=1.00 / lambda=0.30)
# --------------------------------------------------------------------------- #
def compute_full_test(P, o_shape_test):
    S = fused_scores(P["S_p_test"], P["B_test"], FULL_ALPHA)
    idx = topk_deterministic(S, o_shape_test, K)
    sc = np.take_along_axis(S, idx, axis=-1)
    y_mem = mean_anchor(P["bank"], P["test_contexts"], idx, sc)
    fused = (1 - FULL_LAM) * P["y_chronos_test"] + FULL_LAM * y_mem
    fut = P["test_futures"]
    return dict(
        idx=idx, y_mem=y_mem,
        mem_mse=float(((y_mem - fut) ** 2).mean()),
        fus_mse=float(((fused - fut) ** 2).mean()),
        fus_mae=float(np.abs(fused - fut).mean()),
        sum_mem=float(y_mem.sum()),
        topk_cs=topk_checksum(idx),
    )


def run_repro(n):
    P = prepare("ETTh2")
    keys = FULL_KEYS
    P["B_test"] = standardize(combined_B(P["aligns_test"], keys), P["S_p_test"])
    o_shape_test = shape_tiebreak_order(P["S_p_test"])

    results = [compute_full_test(P, o_shape_test) for _ in range(n)]
    for i, r in enumerate(results):
        print(f"run {i}: mem_mse={r['mem_mse']!r} fus_mse={r['fus_mse']!r} "
              f"fus_mae={r['fus_mae']!r} sum_mem={r['sum_mem']!r} topk_cs={r['topk_cs']}")

    print(f"\nmax |diff| across {n} runs:")
    for k in ("mem_mse", "fus_mse", "fus_mae", "sum_mem"):
        vals = [r[k] for r in results]
        maxdiff = max(abs(a - b) for a, b in zip(vals, vals[1:]))
        ok = "OK" if maxdiff < 1e-10 else "FAIL"
        print(f"  {k:>8}: {maxdiff:.3e}  {ok}")
    cs = [r["topk_cs"] for r in results]
    print(f"  topk_cs : {'OK' if len(set(cs)) == 1 else 'FAIL'} (all identical)")


# --------------------------------------------------------------------------- #
# section 5: tie-break verification
# --------------------------------------------------------------------------- #
def run_tiecheck():
    P = prepare("ETTh2")
    keys = FULL_KEYS
    B = standardize(combined_B(P["aligns_test"], keys), P["S_p_test"])
    S = fused_scores(P["S_p_test"], B, FULL_ALPHA)  # alpha=1.0 -> S == B
    S_p = P["S_p_test"]
    o_shape = shape_tiebreak_order(S_p)
    idx = topk_deterministic(S, o_shape, K)

    srt = np.sort(S, axis=-1)[:, ::-1]
    r5, r6 = srt[:, K - 1], srt[:, K]
    tied_mask = r5 == r6
    print(f"A. windows with rank5_score == rank6_score: {int(tied_mask.sum())} / {len(tied_mask)}")

    # B: verify tie-break on every tied window
    tied_idx = np.flatnonzero(tied_mask)
    bad = 0
    for q in tied_idx:
        b = r5[q]
        group = np.flatnonzero(S[q] == b)  # exact boundary-score candidates
        n_above = int((S[q] > b).sum())
        need = K - n_above  # slots the boundary group must fill in the top-5
        # expected: top-`need` of group by (shape desc, index asc)
        gshape = S_p[q, group]
        gorder = np.lexsort((group, -gshape))  # shape desc, index asc
        expected = set(group[gorder[:need]].tolist())
        actual = set(idx[q].tolist()) & set(group.tolist())
        if expected != actual:
            bad += 1
    print(f"B. tie-break verification over {len(tied_idx)} tied windows: "
          f"{'PASS' if bad == 0 else 'FAIL (' + str(bad) + ' mismatches)'}")

    # print 10 evenly spaced tie cases
    n_show = min(10, len(tied_idx))
    pick = tied_idx[np.linspace(0, len(tied_idx) - 1, n_show).astype(int)]
    mem_ts = P["mem_dates"].dt.strftime("%Y-%m-%d %H").to_numpy()
    print(f"\n--- {n_show} tie cases (query window index, boundary-score candidates) ---")
    for q in pick:
        b = r5[q]
        group = np.flatnonzero(S[q] == b)
        gshape = S_p[q, group]
        gorder = np.lexsort((group, -gshape))  # shape desc, index asc
        selected = set(idx[q].tolist())
        print(f"\nquery window={int(q)} (test ot index {int(P['test_starts'][q])})  "
              f"boundary_score={float(b):.8f}  group_size={len(group)}")
        print(f"{'mem_index':>9} {'timestamp':>16} {'final_score':>12} {'shape_sim':>10} {'status':>8}")
        for gi in gorder:
            c = int(group[gi])
            st = "SELECTED" if c in selected else "rejected"
            print(f"{c:>9} {mem_ts[c]:>16} {float(S[q, c]):12.8f} {float(S_p[q, c]):10.6f} {st:>8}")


# --------------------------------------------------------------------------- #
# sections 6-8: full ETTh2 temporal ablation
# --------------------------------------------------------------------------- #
def run_ablation(cache_dir):
    P = prepare("ETTh2")

    chronos_test_mse = float(((P["y_chronos_test"] - P["test_futures"]) ** 2).mean())
    chronos_test_mae = float(np.abs(P["y_chronos_test"] - P["test_futures"]).mean())
    print(f"Chronos test MSE = {chronos_test_mse:.4f}  MAE = {chronos_test_mae:.4f}")
    if abs(chronos_test_mse - REF_CHRONOS_MSE) > REPRO_TOL or abs(chronos_test_mae - REF_CHRONOS_MAE) > REPRO_TOL:
        raise SystemExit(
            f"ERROR: Chronos baseline drifted (got {chronos_test_mse:.4f}/{chronos_test_mae:.4f}, "
            f"ref {REF_CHRONOS_MSE}/{REF_CHRONOS_MAE}). Stopping."
        )

    o_shape_val = shape_tiebreak_order(P["S_p_val"])
    o_shape_test = shape_tiebreak_order(P["S_p_test"])

    rows = []
    for combo_name, keys in COMBOS:
        if not keys:
            B_val = B_test = None
        else:
            B_val = standardize(combined_B(P["aligns_val"], keys), P["S_p_val"])
            B_test = standardize(combined_B(P["aligns_test"], keys), P["S_p_test"])
        Pv = {"S_p": P["S_p_val"], "B": B_val}
        Pt = {"S_p": P["S_p_test"], "B": B_test}

        alpha_grid = [0.0] if not keys else ALPHAS
        alpha, lam, val_mse, val_mae = eval_combo(
            Pv, o_shape_val, alpha_grid, P["y_chronos_val"], P["val_futures"],
            P["bank"], P["val_contexts"],
        )
        _, _, mem_mse, mem_mae, fus_mse, fus_mae = apply(
            Pt, o_shape_test, alpha, lam, P["y_chronos_test"], P["test_futures"],
            P["bank"], P["test_contexts"],
        )
        improvement = (chronos_test_mse - fus_mse) / chronos_test_mse * 100
        rows.append(dict(
            combo=combo_name, alpha=alpha, lam=lam, val_mse=val_mse, val_mae=val_mae,
            mem_mse=mem_mse, mem_mae=mem_mae, fus_mse=fus_mse, fus_mae=fus_mae,
            improvement=improvement,
        ))

    # full table
    print(f"\n=== ETTh2 temporal ablation (deterministic tie-break) ===")
    print(f"Chronos test MSE = {chronos_test_mse:.4f}  MAE = {chronos_test_mae:.4f}")
    hdr = (f"{'components':>16} {'val_a':>6} {'val_l':>6} {'val_MSE':>8} "
           f"{'mem_MSE':>10} {'mem_MAE':>9} {'fus_MSE':>10} {'fus_MAE':>9} {'vsChronos%':>10}")
    print(hdr)
    for row in rows:
        print(f"{row['combo']:>16} {row['alpha']:6.2f} {row['lam']:6.2f} {row['val_mse']:8.4f} "
              f"{row['mem_mse']:10.4f} {row['mem_mae']:9.4f} {row['fus_mse']:10.4f} "
              f"{row['fus_mae']:9.4f} {row['improvement']:+9.2f}%")

    # best combo selected by VAL MSE (val MAE tie-break)
    best = min(rows, key=lambda r: (r["val_mse"], r["val_mae"]))
    print(f"\n=== ETTh2 VAL-selected configuration ===")
    print(f"Best components = {best['combo']}")
    print(f"Best alpha      = {best['alpha']:.2f}")
    print(f"Best lambda     = {best['lam']:.2f}")
    print(f"val MSE         = {best['val_mse']:.4f}")
    print(f"\n--- TEST (frozen val selection) ---")
    print(f"Chronos MSE       = {chronos_test_mse:.4f}")
    print(f"Memory MSE        = {best['mem_mse']:.4f}")
    print(f"Fusion MSE        = {best['fus_mse']:.4f}")
    print(f"Fusion MAE        = {best['fus_mae']:.4f}")
    print(f"Improvement vs Chronos = {best['improvement']:+.2f}%")

    out = os.path.join(cache_dir, "ETTh2_temporal_ablation_deterministic.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


def main():
    args = parse_args()
    if args.repro:
        run_repro(args.repro)
    elif args.tiecheck:
        run_tiecheck()
    else:
        run_ablation(args.cache_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repro", type=int, default=0, metavar="N",
                   help="run N in-process reproducibility checks (Full/alpha=1.0/lambda=0.30)")
    p.add_argument("--tiecheck", action="store_true",
                   help="verify the deterministic tie-break and print tie cases")
    p.add_argument("--cache-dir", type=str, default="results",
                   help="directory holding cached forecasts and CSV output")
    return p.parse_args()


if __name__ == "__main__":
    main()
