"""TRAIN/VAL oracle diagnostic for adaptive fusion (diagnostic only).

Uses the already-selected retriever per ETT dataset (unchanged). For every VAL
query computes D = mean absolute disagreement between Chronos and Memory, splits
into 5 quintiles, and evaluates a fixed lambda grid per quintile. Reports the
oracle bucket-wise lambda MSE as an upper bound. No TEST, no Chronos rerun, no
model training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_memory_reliability_diagnostic as rmd
from priors.executable import _prepare, compute_feature, feature_similarity
from priors.validator import PERIOD, PRED, TREND_WINDOW, _topk
from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.similarity import pearson_similarity_matrix

K = 5
CHUNK = 256
LAMBDAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

SELECTED_PRIOR = rmd.SELECTED_PRIOR
METHOD = rmd.METHOD


def forecasts(dataset):
    """Return per-query (chronos, memory, y_true) for the selected retriever."""
    prep = _prepare(dataset)
    yc = np.load(f"results/{dataset}_chronos.npz")["y_chronos_val"]
    assert yc.shape[0] == len(prep["val_ctx"]), "cached val forecast mismatch"

    method, prior_name = METHOD[dataset]
    if method == "shape":
        _, _, Rm, _, _ = seasonal_decompose(prep["mem_ctx"], PRED, PERIOD, TREND_WINDOW)
        prior, mem_feat, Rm_keep = None, None, Rm
    else:
        prior = SELECTED_PRIOR[prior_name]
        mem_feat, val_feat, _k = compute_feature(
            prior, prep["series"], prep["mem_starts"], prep["val_starts"], prep["eq_coef"]
        )
        Rm_keep = None

    mem_fut = prep["mem_fut"]
    mem_ctx_mean = prep["mem_ctx"].mean(axis=1)
    val_ctx_mean = prep["val_ctx"].mean(axis=1)
    Q = len(prep["val_ctx"])
    chronos = np.empty((Q, PRED))
    memory = np.empty((Q, PRED))

    for q0 in range(0, Q, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, Q))
        if method == "shape":
            _, _, Rv, _, _ = seasonal_decompose(prep["val_ctx"][sl], PRED, PERIOD, TREND_WINDOW)
            S = pearson_similarity_matrix(Rv, Rm_keep)
        else:
            S = feature_similarity(prior, val_feat[sl], mem_feat)
        idx = _topk(S, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        fut = mem_fut[idx]
        cm = mem_ctx_mean[idx]
        w = softmax(sc)
        mem = val_ctx_mean[sl][:, None] + np.einsum("nk,nkh->nh", w, fut - cm[:, :, None])
        chronos[sl] = yc[sl]
        memory[sl] = mem

    return chronos, memory, prep["val_fut"]


def fused_mse_quad(lmb, mA, mB, mC):
    return (1 - lmb) ** 2 * mA + lmb ** 2 * mB + 2 * lmb * (1 - lmb) * mC


def main():
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        chronos, memory, y_true = forecasts(dataset)

        ec = chronos - y_true
        em = memory - y_true
        A = (ec ** 2).mean(axis=1)                      # per-query chronos MSE
        B = (em ** 2).mean(axis=1)                      # per-query memory MSE
        C = (ec * em).mean(axis=1)                      # per-query cross term
        D = np.abs(chronos - memory).mean(axis=1)       # mean absolute disagreement

        order = np.argsort(D)
        n = len(D)
        nq = 5
        buckets = []
        for i in range(nq):
            lo, hi = i * n // nq, (i + 1) * n // nq
            idx = order[lo:hi]
            buckets.append((i + 1, A[idx].mean(), B[idx].mean(), C[idx].mean()))

        # per-bucket best lambda
        print("\n" + "=" * 96)
        print(f"{dataset}  |  method: {rmd._method_label(*METHOD[dataset])}")
        print("=" * 96)
        print(f"{'q':>3}{'best λ':>8}{'MSE@best':>11}{'MSE@λ=0':>11}{'MSE@λ=0.1':>11}")
        print("-" * 96)
        bucket_best = []
        for q, mA, mB, mC in buckets:
            scores = [fused_mse_quad(l, mA, mB, mC) for l in LAMBDAS]
            best_i = int(np.argmin(scores))
            best_l = LAMBDAS[best_i]
            bucket_best.append((q, best_l, scores[best_i]))
            print(f"{q:>3}{best_l:>8.2f}{scores[best_i]:>11.4f}"
                  f"{scores[0]:>11.4f}{scores[2]:>11.4f}")  # index 0->lambda0, 2->lambda0.1
        print("-" * 96)

        # overall
        mA_all, mB_all, mC_all = A.mean(), B.mean(), C.mean()
        chronos_mse = mA_all                                   # lambda = 0
        fixed_010 = fused_mse_quad(0.10, mA_all, mB_all, mC_all)
        # oracle bucket-wise: weighted mean of bucket-best MSE
        oracle = 0.0
        total = n
        for i in range(nq):
            lo, hi = i * n // nq, (i + 1) * n // nq
            cnt = hi - lo
            oracle += cnt * bucket_best[i][2]
        oracle /= total

        print(f"Chronos MSE (lambda=0):        {chronos_mse:.4f}")
        print(f"fixed lambda=0.1 MSE:          {fixed_010:.4f}")
        print(f"oracle bucket-wise lambda MSE: {oracle:.4f}  "
              f"(gain vs λ=0.1: {(fixed_010 - oracle) / fixed_010 * 100:+.2f}%)")


if __name__ == "__main__":
    main()
