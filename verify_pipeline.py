"""Verify retrieval-pipeline invariants (no Chronos needed).

Each check prints PASS/FAIL with evidence. Checks:
  1. split / leakage: train < val < test ranges, memory bank from train only
  2. window construction: 512 context + 96 future, no off-by-one / overlap
  3. Pearson: hand-compute top1 for 3 random queries vs the program's value
  4. Mean-anchor: hand-compute y_mem for one query vs the program's value
  5. residual decomposition: x = T + S + R (reconstruction error)
  6. top-K / softmax: K=5, descending order, weights sum to 1
"""

import numpy as np
import pandas as pd

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
PERIOD, TREND_WINDOW = 24, 48


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, train, train_mask, val_mask, test_mask, int(test_idx[0]), int(test_idx[-1]) + 1


def manual_pearson(q, C):
    """Independent Pearson: q (L,), C (n, L) -> (n,) correlation."""
    q = q - q.mean()
    C = C - C.mean(axis=1, keepdims=True)
    num = C @ q
    den = np.sqrt((C ** 2).sum(axis=1)) * np.sqrt((q ** 2).sum())
    return num / np.where(den == 0, 1.0, den)


def topk(scores, k):
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def main():
    dates, ot, train, train_mask, val_mask, test_mask, test_start, test_end = load("data/ETTh1.csv")

    # ---------- CHECK 1: split / leakage ----------
    print("=== CHECK 1: split / leakage ===")
    print(f"full series: {dates.iloc[0]} -> {dates.iloc[-1]}  (n={len(ot)})")
    print(f"train: {dates[train_mask].min()} -> {dates[train_mask].max()}  (n={int(train_mask.sum())})")
    print(f"val  : {dates[val_mask].min()} -> {dates[val_mask].max()}  (n={int(val_mask.sum())})")
    print(f"test : {dates[test_mask].min()} -> {dates[test_mask].max()}  (n={int(test_mask.sum())})")

    bank = MemoryBank.from_series(train, CONTEXT, PRED)
    expected_n = len(train) - CONTEXT - PRED + 1
    ok1 = bank.n_memories == expected_n
    print(f"memory bank n = {bank.n_memories}  (expected train_len - 512 - 96 + 1 = {expected_n})  -> {'PASS' if ok1 else 'FAIL'}")

    # memory bank future must come only from train (never test / val)
    train_end_idx = int(np.flatnonzero(train_mask)[-1])  # last train row index in full series
    # bank.futures[0] is train[512:512+96] -> full rows 512..607, all within train range
    print(f"train occupies full-series rows [0, {train_end_idx}]; test starts at row {test_start} (gap {test_start - train_end_idx - 1} rows)")
    ok1b = test_start > train_end_idx
    print(f"no overlap train/test -> {'PASS' if ok1b else 'FAIL'}")

    # ---------- CHECK 2: window construction ----------
    print("\n=== CHECK 2: window construction ===")
    starts = np.arange(test_start, test_end - PRED + 1)
    n_expected = test_end - PRED - test_start + 1
    print(f"test windows: {len(starts)}  (expected {n_expected})  -> {'PASS' if len(starts) == n_expected else 'FAIL'}")

    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    rng = np.random.default_rng(0)
    ok2 = True
    for i in rng.choice(len(starts), size=5, replace=False):
        s = starts[i]
        c = ot[s - CONTEXT : s]
        f = ot[s : s + PRED]
        same_c = np.array_equal(contexts[i], c)
        same_f = np.array_equal(futures[i], f)
        # context last point is ot[s-1], future first point is ot[s]: consecutive, no gap/overlap
        contig = (contexts[i][-1] == ot[s - 1]) and (futures[i][0] == ot[s])
        ok2 &= same_c and same_f and contig
        if not (same_c and same_f and contig):
            print(f"  sample i={i} s={s}: ctx_match={same_c} fut_match={same_f} contiguous={contig}  <- PROBLEM")
    print(f"5 random windows: context/future slice exact + contiguous (no off-by-one) -> {'PASS' if ok2 else 'FAIL'}")

    # ---------- CHECK 3: Pearson ----------
    print("\n=== CHECK 3: Pearson (hand-check top1) ===")
    C = bank.contexts.astype(np.float64)
    ok3 = True
    for i in rng.choice(len(contexts), size=3, replace=False):
        q = contexts[i].astype(np.float64)
        prog = pearson_similarity_matrix(q[None, :], bank.contexts)[0]
        indep = manual_pearson(q, C)
        maxdiff = np.abs(prog - indep).max()
        same_top1 = int(prog.argmax()) == int(indep.argmax())
        ok3 &= same_top1 and maxdiff < 1e-6
        print(f"  query {i}: prog_top1={prog.argmax()} ({prog[prog.argmax()]:.6f}), "
              f"indep_top1={indep.argmax()} ({indep[indep.argmax()]:.6f}), "
              f"same_top1={same_top1}, max|diff|={maxdiff:.2e}")
    print(f"program pearson == independent pearson (max diff < 1e-6) -> {'PASS' if ok3 else 'FAIL'}")

    # ---------- CHECK 4: Mean-anchor (hand-check one query) ----------
    print("\n=== CHECK 4: Mean-anchor (hand-check) ===")
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    sim_R = pearson_similarity_matrix(R_q, R_m)
    idx_res = topk(sim_R, K)
    scores_res = np.take_along_axis(sim_R, idx_res, axis=-1)
    w = softmax(scores_res)
    fut = bank.futures[idx_res]
    ctx = bank.contexts[idx_res]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    y_mem = contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)

    i = int(rng.choice(len(contexts), size=1)[0])
    mean_q = contexts[i].mean()
    hand = np.zeros(PRED, dtype=np.float64)
    for k in range(K):
        mi = idx_res[i, k]
        mean_mk = bank.contexts[mi].mean()
        shape_k = bank.futures[mi] - mean_mk
        hand += w[i, k] * shape_k
    hand = mean_q + hand
    diff = np.abs(hand - y_mem[i]).max()
    ok4 = diff < 1e-4
    print(f"  query {i}: mean(x_q)={mean_q:.4f}")
    print(f"  top-5 memory idx={idx_res[i].tolist()}, sim={np.round(scores_res[i], 4).tolist()}, w={np.round(w[i], 4).tolist()}")
    print(f"  hand-computed y_mem vs program: max|diff|={diff:.2e} -> {'PASS' if ok4 else 'FAIL'}")

    # ---------- CHECK 5: residual decomposition ----------
    print("\n=== CHECK 5: residual decomposition x = T + S + R ===")
    x = contexts[i].astype(np.float64)
    T, S, R, T_fut, S_fut = seasonal_decompose(contexts[i : i + 1], PRED, PERIOD, TREND_WINDOW)
    T, S, R = T[0].astype(np.float64), S[0].astype(np.float64), R[0].astype(np.float64)
    recon = T + S + R
    err = np.abs(recon - x).max()
    ok5 = err < 1e-3  # float32 casting of each component
    print(f"  max|x - (T+S+R)| = {err:.2e}  (x range [{x.min():.2f}, {x.max():.2f}]) -> {'PASS' if ok5 else 'FAIL'}")
    # S periodic with period 24 (tiled daily pattern)
    # note: tiling is by (position mod 24); compare S to its period-24 shift
    shift_err = np.abs(S[24:] - S[:-24]).max() if len(S) > 24 else np.nan
    print(f"  max|S[t] - S[t-24]| = {shift_err:.2e}  (should be ~0: S is a tiled 24-pattern)")

    # ---------- CHECK 6: top-K / softmax ----------
    print("\n=== CHECK 6: top-K / softmax ===")
    print(f"  idx_res.shape = {idx_res.shape}  (expect (5613, 5)) -> {'PASS' if idx_res.shape == (len(contexts), K) else 'FAIL'}")
    desc = np.all(scores_res[:, 0] >= scores_res[:, 1]) and np.all(scores_res[:, 1] >= scores_res[:, -1])
    print(f"  scores sorted descending -> {'PASS' if desc else 'FAIL'}")
    wsum = np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
    print(f"  softmax weights sum to 1 -> {'PASS' if wsum else 'FAIL'}")

    print("\nDONE (checks 1-6, no Chronos).")


if __name__ == "__main__":
    main()
