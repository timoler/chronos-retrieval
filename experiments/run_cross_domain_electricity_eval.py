"""Cross-domain electricity: TRAIN validation + VAL prior selection + Adaptive Gate V1.

Reuses the frozen retrieval primitives, Chronos backbone, and gate (imported from
run_adaptive_fusion_gate). Electricity data is 15-min; VAL stride = 4 (like ETTm).
Chronological split: TRAIN 60% / VAL 20% / TEST 20% (TEST untouched).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from priors.executable import (
    CONTEXT,
    PRED,
    ExecutablePrior,
    compute_feature,
    feature_similarity,
)
from priors.validator import PERIOD, TREND_WINDOW, _mem_forecast, _topk
from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.similarity import pearson_similarity_matrix

import run_adaptive_fusion_gate as gate_mod
from run_adaptive_fusion_gate import Gate, load_pipeline, run_chronos, train_gate

K = 5
CHUNK = 256
VAL_STRIDE = 4
TRAIN_B_FRAC = 0.20
LAMBDA = 0.1
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LOAD_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def load_electricity():
    df = pd.read_csv("data/electricity.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    val_start, val_end = train_len, train_len + val_len
    series = {"OT": df["OT"].to_numpy(np.float32)}
    for v in LOAD_VARS:
        series[v] = df[v].to_numpy(np.float32)
    series["total_load"] = np.sum([series[v] for v in LOAD_VARS], axis=0)
    return dates, series, N, train_len, val_start, val_end


def build(series, mem_starts, q_starts):
    ot = series["OT"]
    mem_ctx = np.stack([ot[m:m + CONTEXT] for m in mem_starts]).astype(np.float32)
    mem_fut = np.stack([ot[m + CONTEXT:m + CONTEXT + PRED] for m in mem_starts]).astype(np.float32)
    q_ctx = np.stack([ot[s - CONTEXT:s] for s in q_starts]).astype(np.float32)
    q_fut = np.stack([ot[s:s + PRED] for s in q_starts]).astype(np.float32)
    return mem_ctx, mem_fut, q_ctx, q_fut


def prior_feats(prior, series, mem_starts, q_starts):
    return compute_feature(prior, series, mem_starts, q_starts, np.zeros(2))


def leave_one_in(priors, series, mem_starts, mem_ctx, mem_fut, q_starts, q_ctx, q_fut, yc):
    """Sweep alpha per prior (VAL); return rows + shape baselines. No TEST."""
    _, _, Rm, _, _ = seasonal_decompose(mem_ctx, PRED, PERIOD, TREND_WINDOW)
    mem_ctx_mean = mem_ctx.mean(axis=1)
    q_ctx_mean = q_ctx.mean(axis=1)
    n_val = len(q_ctx)
    n_pts = n_val * PRED
    nonzero = [a for a in ALPHAS if a != 0.0]

    # per-prior features
    pf = {}
    for p in priors:
        mf, qf, _ = prior_feats(p, series, mem_starts, q_starts)
        pf[p.name] = (p, mf, qf)

    chronos_sq = 0.0
    shape_mem_sq = 0.0
    shape_fus_sq = 0.0
    prior_mem_sq = {p.name: {a: 0.0 for a in nonzero} for p in priors}
    prior_fus_sq = {p.name: {a: 0.0 for a in nonzero} for p in priors}

    for q0 in range(0, n_val, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, n_val))
        _, _, Rv, _, _ = seasonal_decompose(q_ctx[sl], PRED, PERIOD, TREND_WINDOW)
        S_shape = pearson_similarity_matrix(Rv, Rm)
        y_mem_shape = _mem_forecast(S_shape, mem_fut, mem_ctx_mean, q_ctx_mean[sl])
        shape_mem_sq += float(((y_mem_shape - q_fut[sl]) ** 2).sum())
        yc_sl = yc[sl]
        fused_shape = (1 - LAMBDA) * yc_sl + LAMBDA * y_mem_shape
        shape_fus_sq += float(((fused_shape - q_fut[sl]) ** 2).sum())
        chronos_sq += float(((yc_sl - q_fut[sl]) ** 2).sum())

        for name, (p, mf, qf) in pf.items():
            S_prior = feature_similarity(p, qf[sl], mf)
            for a in nonzero:
                S = (1 - a) * S_shape + a * S_prior
                y_mem = _mem_forecast(S, mem_fut, mem_ctx_mean, q_ctx_mean[sl])
                prior_mem_sq[name][a] += float(((y_mem - q_fut[sl]) ** 2).sum())
                fused = (1 - LAMBDA) * yc_sl + LAMBDA * y_mem
                prior_fus_sq[name][a] += float(((fused - q_fut[sl]) ** 2).sum())

    chronos_mse = chronos_sq / n_pts
    shape_mem_mse = shape_mem_sq / n_pts
    shape_fus_mse = shape_fus_sq / n_pts

    rows = []
    for p in priors:
        fus = {0.0: shape_fus_mse}
        mem = {0.0: shape_mem_mse}
        for a in nonzero:
            fus[a] = prior_fus_sq[p.name][a] / n_pts
            mem[a] = prior_mem_sq[p.name][a] / n_pts
        best_a = min(fus, key=fus.get)
        rows.append(dict(name=p.name, best_alpha=best_a, val_mem_mse=mem[best_a],
                         val_fus_mse=fus[best_a],
                         improvement=(shape_fus_mse - fus[best_a]) / shape_fus_mse * 100.0))
    return rows, chronos_mse, shape_fus_mse


def retrieval_features(prior, series, mem_starts, mem_ctx, mem_fut, q_starts, q_ctx, q_fut, yc):
    mf, qf, _ = prior_feats(prior, series, mem_starts, q_starts)
    N = len(q_starts)
    feat = np.empty((N, 5), dtype=np.float32)
    y_mem = np.empty((N, PRED), dtype=np.float32)
    mem_ctx_mean = mem_ctx.mean(axis=1)
    q_ctx_mean = q_ctx.mean(axis=1)
    for q0 in range(0, N, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, N))
        S = feature_similarity(prior, qf[sl], mf)
        idx = _topk(S, K)
        sc = np.take_along_axis(S, idx, -1)
        fut = mem_fut[idx]
        cm = mem_ctx_mean[idx]
        w = softmax(sc)
        mem = q_ctx_mean[sl][:, None] + np.einsum("nk,nkh->nh", w, fut - cm[:, :, None])
        y_mem[sl] = mem
        feat[sl, 0] = np.abs(yc[sl] - mem).mean(axis=1)
        feat[sl, 1] = sc.max(axis=1)
        feat[sl, 2] = sc.mean(axis=1)
        feat[sl, 3] = sc.std(axis=1)
        feat[sl, 4] = fut.std(axis=1).mean(axis=1)
        del S
    return feat, y_mem


def main():
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dates, series, N, train_len, val_start, val_end = load_electricity()
    priors = [ExecutablePrior(**d) for d in
              json.load(open("results/cross_domain/electricity_pipeline.json"))["exec_priors"]]

    # ---- banks / queries ----
    mem_starts = np.arange(0, train_len - CONTEXT - PRED + 1)
    val_starts = np.arange(val_start, val_end - PRED + 1, VAL_STRIDE)
    mem_ctx, mem_fut, val_ctx, val_fut = build(series, mem_starts, val_starts)

    train_b_len = int(TRAIN_B_FRAC * train_len)
    train_a_end = train_len - train_b_len
    mem_starts_A = np.arange(0, train_a_end - CONTEXT - PRED + 1)
    q_starts_B = np.arange(train_a_end, train_len - PRED + 1, VAL_STRIDE)
    mem_ctx_A, mem_fut_A, q_ctx_B, q_fut_B = build(series, mem_starts_A, q_starts_B)

    # leak-safety
    assert int(mem_starts_A.max() + CONTEXT + PRED) <= int(q_starts_B.min()), "TRAIN_B leak"
    print(f"electricity: N={N} train={train_len} val=[{val_start},{val_end}) "
          f"TRAIN_A mem={len(mem_starts_A)} TRAIN_B q={len(q_starts_B)} VAL q={len(val_starts)}")

    # ---- Chronos (val + train-B), cached ----
    pipeline = load_pipeline()
    cval = Path("results/electricity_chronos.npz")
    if cval.exists():
        yc_val = np.load(cval)["y_chronos_val"].astype(np.float32)
    else:
        yc_val = run_chronos(pipeline, [c for c in val_ctx])
        np.savez(cval, y_chronos_val=yc_val)
    ctrb = Path("results/electricity_chronos_trainB.npz")
    if ctrb.exists():
        yc_B = np.load(ctrb)["y_chronos_trainB"].astype(np.float32)
    else:
        yc_B = run_chronos(pipeline, [c for c in q_ctx_B])
        np.savez(ctrb, y_chronos_trainB=yc_B)

    # ---- VAL prior selection ----
    rows, chronos_mse, shape_fus_mse = leave_one_in(
        priors, series, mem_starts, mem_ctx, mem_fut, val_starts, val_ctx, val_fut, yc_val)

    print("\nVAL prior selection (leave-one-in):")
    for r in rows:
        print(f"  {r['name']}: best_alpha={r['best_alpha']} fusedMSE={r['val_fus_mse']:.4f} "
              f"imp={r['improvement']:+.2f}%")
    best = min(rows, key=lambda r: r["val_fus_mse"])
    best_prior = next(p for p in priors if p.name == best["name"])
    agent_fixed_mse = None
    # recompute agent fixed-lambda=0.1 MSE (at alpha=1 pure prior, lambda=0.1)
    mf, qf, _ = prior_feats(best_prior, series, mem_starts, val_starts)
    S_prior_full = feature_similarity(best_prior, qf, mf)
    y_mem_full = _mem_forecast(S_prior_full, mem_fut, mem_ctx.mean(1), val_ctx.mean(1))
    agent_fixed_mse = float(((0.9 * yc_val + 0.1 * y_mem_full - val_fut) ** 2).mean())

    # ---- gate ----
    feat_B, ym_B = retrieval_features(best_prior, series, mem_starts_A, mem_ctx_A, mem_fut_A,
                                      q_starts_B, q_ctx_B, q_fut_B, yc_B)
    feat_V, ym_V = retrieval_features(best_prior, series, mem_starts, mem_ctx, mem_fut,
                                      val_starts, val_ctx, val_fut, yc_val)
    gate, mu, sd, best_val = train_gate(feat_B, yc_B, ym_B, q_fut_B,
                                        feat_V, yc_val, ym_V, val_fut, device)
    gate.eval()
    with torch.no_grad():
        Xv = torch.tensor((feat_V - mu) / sd, dtype=torch.float32, device=device)
        lam_v = gate(Xv).detach().cpu().numpy().flatten()
    gate_mse = best_val
    lam = np.asarray(lam_v)

    # ---- report ----
    print("\n" + "=" * 90)
    print("ELECTRICITY (cross-domain) — VAL-only results")
    print("=" * 90)
    spec = f"{best_prior.transform}/{best_prior.temporal_aggregation}/{best_prior.similarity} on {best_prior.input_variables}"
    print(f"Dataset:                      electricity")
    print(f"Chronos VAL MSE:              {chronos_mse:.4f}")
    print(f"Shape-Memory fusion MSE:      {shape_fus_mse:.4f}")
    print(f"Best Agent prior:             {best_prior.name}")
    print(f"Executable spec:              {spec}")
    print(f"Agent-Memory fixed-lambda MSE:{agent_fixed_mse:.4f}")
    print(f"Adaptive Gate MSE:            {gate_mse:.4f}")
    print(f"gain vs Chronos:              {(chronos_mse - gate_mse)/chronos_mse*100:+.2f}%")
    print(f"gain vs Shape:                {(shape_fus_mse - gate_mse)/shape_fus_mse*100:+.2f}%")
    print(f"gain vs fixed lambda:         {(agent_fixed_mse - gate_mse)/agent_fixed_mse*100:+.2f}%")
    print(f"gate mean/std lambda:         {lam.mean():.3f} / {lam.std():.3f}")
    print(f"proposed / accepted / rejected: 5 / 2 / 3")


if __name__ == "__main__":
    main()
