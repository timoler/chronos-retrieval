"""TRAIN/VAL memory reliability diagnostic on 4 ETT datasets.

For each dataset, use its selected retrieval method (the frozen multi-agent
selected prior, or the shape fallback for ETTm2). For every VAL query compute

    memory_advantage = MSE(Chronos, y_true) - MSE(Memory, y_true)

and six pre-quantile reliability signals. Signal 7 (Chronos quantile
uncertainty) is skipped: the cached *_quantiles.npz are TEST-aligned only
(5613/5685 rows), not VAL.

No gate training, no lambda fitting, no signal combination, no TEST, no Chronos
rerun. Retriever and priors are unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.executable import (
    ExecutablePrior,
    _prepare,
    compute_feature,
    feature_similarity,
)
from priors.validator import PERIOD, PRED, TREND_WINDOW, _spearman, _topk
from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.similarity import pearson_similarity_matrix

K = 5
LAMBDA = 0.1
CHUNK = 256

# frozen multi-agent selected priors (unchanged; specs from the frozen protocol)
SELECTED_PRIOR = {
    "Nonlinear Heating Regime": ExecutablePrior(
        name="Nonlinear Heating Regime", source="multi_agent",
        input_variables=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        transform="square", temporal_aggregation="window", similarity="correlation",
        expected_mechanism="", confidence=0.6,
    ),
    "Accumulated Heat Persistence": ExecutablePrior(
        name="Accumulated Heat Persistence", source="multi_agent",
        input_variables=["OT"],
        transform="ewma", temporal_aggregation="last", similarity="abs_diff",
        expected_mechanism="", confidence=0.7,
    ),
}

# selected retrieval method per dataset (from the cross-dataset VAL selection)
METHOD = {
    "ETTh1": ("prior", "Nonlinear Heating Regime"),
    "ETTh2": ("prior", "Accumulated Heat Persistence"),
    "ETTm1": ("prior", "Nonlinear Heating Regime"),
    "ETTm2": ("shape", None),
}

SIGNAL_KEYS = [
    "top1_sim", "mean_topk_sim", "gap_top1_mean", "std_topk_sim",
    "topk_disagreement", "chronos_mem_disagreement",
]
SIGNAL_LABELS = {
    "top1_sim": "top1 similarity",
    "mean_topk_sim": "mean top-k similarity",
    "gap_top1_mean": "top1 - mean(top-k) gap",
    "std_topk_sim": "std top-k similarity",
    "topk_disagreement": "top-k forecast disagreement",
    "chronos_mem_disagreement": "Chronos-Memory disagreement",
}


def _method_label(method, prior_name):
    if method == "shape":
        return "shape fallback (residual Pearson on OT)"
    p = SELECTED_PRIOR[prior_name]
    return f"prior {prior_name} ({p.transform}/{p.temporal_aggregation}/{p.similarity})"


def run_diagnostic(dataset):
    prep = _prepare(dataset)
    yc = np.load(f"results/{dataset}_chronos.npz")["y_chronos_val"]
    assert yc.shape[0] == len(prep["val_ctx"]), "cached val forecast mismatch"

    method, prior_name = METHOD[dataset]

    # memory-side key (computed once)
    if method == "shape":
        _, _, Rm, _, _ = seasonal_decompose(prep["mem_ctx"], PRED, PERIOD, TREND_WINDOW)
        prior = None
        mem_feat = val_feat = None
    else:
        prior = SELECTED_PRIOR[prior_name]
        mem_feat, val_feat, _kind = compute_feature(
            prior, prep["series"], prep["mem_starts"], prep["val_starts"], prep["eq_coef"]
        )
        Rm = None

    mem_fut = prep["mem_fut"]
    mem_ctx_mean = prep["mem_ctx"].mean(axis=1)
    val_ctx_mean = prep["val_ctx"].mean(axis=1)
    val_fut = prep["val_fut"]

    Q = len(prep["val_ctx"])
    chronos_mse = np.empty(Q)
    memory_mse = np.empty(Q)
    fused_mse = np.empty(Q)
    advantage = np.empty(Q)
    sig = {k: np.empty(Q) for k in SIGNAL_KEYS}

    for q0 in range(0, Q, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, Q))
        if method == "shape":
            _, _, Rv, _, _ = seasonal_decompose(prep["val_ctx"][sl], PRED, PERIOD, TREND_WINDOW)
            S = pearson_similarity_matrix(Rv, Rm)
        else:
            S = feature_similarity(prior, val_feat[sl], mem_feat)

        idx = _topk(S, K)                       # (n, K)
        sc = np.take_along_axis(S, idx, axis=-1)  # (n, K)
        fut = mem_fut[idx]                      # (n, K, H)
        cm = mem_ctx_mean[idx]                  # (n, K)
        w = softmax(sc)                         # (n, K)
        mem = val_ctx_mean[sl][:, None] + np.einsum(
            "nk,nkh->nh", w, fut - cm[:, :, None]
        )

        yc_sl = yc[sl]
        y_true = val_fut[sl]
        fused = (1 - LAMBDA) * yc_sl + LAMBDA * mem

        sig["top1_sim"][sl] = sc.max(axis=1)
        sig["mean_topk_sim"][sl] = sc.mean(axis=1)
        sig["gap_top1_mean"][sl] = sc.max(axis=1) - sc.mean(axis=1)
        sig["std_topk_sim"][sl] = sc.std(axis=1)
        sig["topk_disagreement"][sl] = fut.std(axis=1).mean(axis=1)
        sig["chronos_mem_disagreement"][sl] = ((yc_sl - mem) ** 2).mean(axis=1)

        chronos_mse[sl] = ((yc_sl - y_true) ** 2).mean(axis=1)
        memory_mse[sl] = ((mem - y_true) ** 2).mean(axis=1)
        fused_mse[sl] = ((fused - y_true) ** 2).mean(axis=1)
        advantage[sl] = chronos_mse[sl] - memory_mse[sl]

    return dict(sig=sig, chronos_mse=chronos_mse, memory_mse=memory_mse,
                fused_mse=fused_mse, advantage=advantage)


def _quintiles(signal, chronos_mse, memory_mse, fused_mse, advantage, nq=5):
    order = np.argsort(signal)
    n = len(signal)
    rows = []
    for i in range(nq):
        lo, hi = i * n // nq, (i + 1) * n // nq
        idx = order[lo:hi]
        rows.append((i + 1, chronos_mse[idx].mean(), memory_mse[idx].mean(),
                     fused_mse[idx].mean(), advantage[idx].mean()))
    return rows


def main():
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        method, prior_name = METHOD[dataset]
        d = run_diagnostic(dataset)
        frac_mem_better = float((d["memory_mse"] < d["chronos_mse"]).mean())

        print("\n" + "=" * 110)
        print(f"{dataset}  |  method: {_method_label(method, prior_name)}")
        print("=" * 110)
        print(f"fraction of VAL queries where Memory MSE < Chronos MSE: "
              f"{frac_mem_better:.4f}  (n={len(d['advantage'])})")

        # Spearman summary
        print("\nSPEARMAN(signal, memory_advantage):")
        for k in SIGNAL_KEYS:
            rho = _spearman(d["sig"][k], d["advantage"])
            print(f"  {SIGNAL_LABELS[k]:<30}: {rho:+.3f}")
        print(f"  {'Chronos quantile uncertainty':<30}: unavailable "
              f"(cached *_quantiles.npz are TEST-aligned, not VAL)")

        # quintile buckets
        print("\nQUINTILE BUCKETS  (signal quintile | ChronosMSE | MemoryMSE | "
              "fused@0.1 | memory_advantage)")
        for k in SIGNAL_KEYS:
            print(f"\n  [{SIGNAL_LABELS[k]}]")
            print(f"    {'q':>3}{'ChronosMSE':>12}{'MemoryMSE':>12}{'fused@0.1':>12}"
                  f"{'advantage':>12}")
            for q, cm, mm, fm, adv in _quintiles(
                    d["sig"][k], d["chronos_mse"], d["memory_mse"],
                    d["fused_mse"], d["advantage"]):
                print(f"    {q:>3}{cm:>12.4f}{mm:>12.4f}{fm:>12.4f}{adv:>+12.4f}")


if __name__ == "__main__":
    main()
