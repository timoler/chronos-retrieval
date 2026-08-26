"""Adaptive Fusion Gate V1 (GPU): learn query-specific lambda for Chronos+Memory.

Leakage-safe chronological split:
  TRAIN_A = memory bank / prior construction (early TRAIN)
  TRAIN_B = gate training (tail of TRAIN; Chronos run here, cached)
  VAL     = gate early stopping + model selection
  TEST    = untouched

Gate is a tiny MLP (5 -> 16 -> 1, sigmoid; lambda = 0.5*sigmoid in [0, 0.5]).
Trained directly on MSE((1-lambda)*Chronos + lambda*Memory, y_true). Chronos and
Memory forecasts are frozen; retrieval is chunked (no full NxM similarity kept).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from priors.executable import (
    CONTEXT,
    PRED,
    ExecutablePrior,
    _load_full,
    _prepare,
    compute_feature,
    feature_similarity,
)
from priors.validator import PERIOD, TREND_WINDOW, _topk
from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.similarity import pearson_similarity_matrix

K = 5
CHUNK = 256
TRAIN_B_FRAC = 0.20
BATCH = 256
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 0

VAL_STRIDE = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 4, "ETTm2": 4}

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
METHOD = {
    "ETTh1": ("prior", "Nonlinear Heating Regime"),
    "ETTh2": ("prior", "Accumulated Heat Persistence"),
    "ETTm1": ("prior", "Nonlinear Heating Regime"),
    "ETTm2": ("shape", None),
}


class Gate(nn.Module):
    def __init__(self, in_dim=5, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return 0.5 * self.net(x)  # (N, 1) in [0, 0.5]


def load_pipeline():
    from chronos import Chronos2Pipeline

    print("Loading amazon/chronos-2 ...", flush=True)
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")


def run_chronos(pipeline, contexts):
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    return np.stack([m.squeeze(0).numpy() for m in means]).astype(np.float32)


def _rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def retrieval_features(method, series, mem_starts, mem_ctx, mem_fut,
                       query_starts, query_ctx, query_fut, y_chronos):
    """Chunked retrieval -> (features (N,5), y_memory (N,H)). No full NxM kept."""
    kind, prior_name = method
    if kind == "shape":
        _, _, Rm, _, _ = seasonal_decompose(mem_ctx, PRED, PERIOD, TREND_WINDOW)
        prior, mem_feat = None, None
    else:
        prior = SELECTED_PRIOR[prior_name]
        mem_feat, query_feat, _ = compute_feature(
            prior, series, mem_starts, query_starts, np.zeros(2)
        )
        Rm = None

    N = len(query_starts)
    feat = np.empty((N, 5), dtype=np.float32)
    y_mem = np.empty((N, PRED), dtype=np.float32)
    mem_ctx_mean = mem_ctx.mean(axis=1)
    query_ctx_mean = query_ctx.mean(axis=1)

    for q0 in range(0, N, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, N))
        if kind == "shape":
            _, _, Rv, _, _ = seasonal_decompose(query_ctx[sl], PRED, PERIOD, TREND_WINDOW)
            S = pearson_similarity_matrix(Rv, Rm)
        else:
            S = feature_similarity(prior, query_feat[sl], mem_feat)

        idx = _topk(S, K)
        sc = np.take_along_axis(S, idx, axis=-1)
        fut = mem_fut[idx]
        cm = mem_ctx_mean[idx]
        w = softmax(sc)
        mem = query_ctx_mean[sl][:, None] + np.einsum("nk,nkh->nh", w, fut - cm[:, :, None])
        y_mem[sl] = mem

        feat[sl, 0] = np.abs(y_chronos[sl] - mem).mean(axis=1)
        feat[sl, 1] = sc.max(axis=1)
        feat[sl, 2] = sc.mean(axis=1)
        feat[sl, 3] = sc.std(axis=1)
        feat[sl, 4] = fut.std(axis=1).mean(axis=1)
        del S

    return feat, y_mem


def train_gate(X_tr, yc_tr, ym_tr, yt_tr, X_val, yc_val, ym_val, yt_val, device):
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-8

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    Xtr, ycr, ymr, ytr = to_t((X_tr - mu) / sd), to_t(yc_tr), to_t(ym_tr), to_t(yt_tr)
    Xva, ycv, ymv, ytv = to_t((X_val - mu) / sd), to_t(yc_val), to_t(ym_val), to_t(yt_val)

    torch.manual_seed(SEED)
    gate = Gate(5, 16).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(Xtr, ycr, ymr, ytr), batch_size=BATCH, shuffle=True)

    best_val, best_state, stale = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        gate.train()
        for xb, cb, mb, tb in loader:
            lam = gate(xb)
            yhat = (1 - lam) * cb + lam * mb
            loss = F.mse_loss(yhat, tb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        gate.eval()
        with torch.no_grad():
            lam_v = gate(Xva)
            val_mse = F.mse_loss((1 - lam_v) * ycv + lam_v * ymv, ytv).item()
        if val_mse < best_val - 1e-9:
            best_val, stale = val_mse, 0
            best_state = {k: v.detach().clone() for k, v in gate.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    gate.load_state_dict(best_state)
    return gate, mu, sd, best_val


def main():
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()
    Path("results/adaptive_gate").mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline()

    out_rows = []
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        t0 = time.time()
        method = METHOD[dataset]
        stride = VAL_STRIDE[dataset]

        df, dates, train_mask, val_mask, series = _load_full(dataset)
        ot = series["OT"]
        N = len(ot)
        train_len = int(train_mask.sum())
        val_start = int(np.flatnonzero(val_mask)[0])
        val_end = int(np.flatnonzero(val_mask)[-1]) + 1
        test_start = val_end

        train_b_len = int(TRAIN_B_FRAC * train_len)
        train_a_end = train_len - train_b_len

        # ---- split (chronological, leak-safe) ----
        mem_starts_A = np.arange(0, train_a_end - CONTEXT - PRED + 1)
        q_starts_B = np.arange(train_a_end, train_len - PRED + 1, stride)
        mem_ctx_A = np.stack([ot[m:m + CONTEXT] for m in mem_starts_A]).astype(np.float32)
        mem_fut_A = np.stack([ot[m + CONTEXT:m + CONTEXT + PRED] for m in mem_starts_A]).astype(np.float32)
        q_ctx_B = np.stack([ot[s - CONTEXT:s] for s in q_starts_B]).astype(np.float32)
        q_fut_B = np.stack([ot[s:s + PRED] for s in q_starts_B]).astype(np.float32)

        # ---- leak-safety verification ----
        max_mem_fut_end = int(mem_starts_A.max() + CONTEXT + PRED) if len(mem_starts_A) else -1
        min_q_origin = int(q_starts_B.min()) if len(q_starts_B) else -1
        assert max_mem_fut_end <= min_q_origin, "TRAIN_B query future leaks into memory bank"
        # VAL memory bank (full TRAIN) futures all end <= train_len < val_start
        assert train_len - 1 < val_start, "VAL overlaps TRAIN"

        print("\n" + "=" * 100)
        print(f"{dataset}  SPLIT VERIFICATION")
        print("=" * 100)
        print(f"  TRAIN_A (memory bank): idx [0, {train_a_end})  "
              f"dates [{dates.iloc[0]} , {dates.iloc[train_a_end - 1]})   "
              f"#mem_windows={len(mem_starts_A)}")
        print(f"  TRAIN_B (gate train):  idx [{train_a_end}, {train_len})  "
              f"dates [{dates.iloc[train_a_end]} , {dates.iloc[train_len - 1]})   "
              f"#queries={len(q_starts_B)}")
        print(f"  VAL     (early stop):  idx [{val_start}, {val_end})  "
              f"dates [{dates.iloc[val_start]} , {dates.iloc[val_end - 1]})")
        print(f"  TEST    (untouched):   idx [{test_start}, {N})  "
              f"dates [{dates.iloc[test_start]} , {dates.iloc[N - 1]})")
        print(f"  leak check: max TRAIN_A mem future end = {max_mem_fut_end} "
              f"<= min TRAIN_B query origin = {min_q_origin}  -> OK")

        # ---- TRAIN_B Chronos (cached) ----
        cache = Path(f"results/{dataset}_chronos_trainB.npz")
        if cache.exists():
            yc_B = np.load(cache)["y_chronos_trainB"].astype(np.float32)
        else:
            yc_B = run_chronos(pipeline, [c for c in q_ctx_B])
            np.savez(cache, y_chronos_trainB=yc_B)
        assert yc_B.shape == q_fut_B.shape, "TRAIN_B chronos shape mismatch"

        # ---- VAL side (full-train memory bank) ----
        prep = _prepare(dataset)
        yc_V = np.load(f"results/{dataset}_chronos.npz")["y_chronos_val"].astype(np.float32)
        assert yc_V.shape[0] == len(prep["val_ctx"]), "cached val forecast mismatch"

        # ---- features ----
        feat_B, ym_B = retrieval_features(method, series, mem_starts_A, mem_ctx_A, mem_fut_A,
                                          q_starts_B, q_ctx_B, q_fut_B, yc_B)
        feat_V, ym_V = retrieval_features(method, series, prep["mem_starts"], prep["mem_ctx"],
                                          prep["mem_fut"], prep["val_starts"], prep["val_ctx"],
                                          prep["val_fut"], yc_V)
        peak_ram = _rss_mb()

        # ---- train gate ----
        gate, mu, sd, best_val = train_gate(feat_B, yc_B, ym_B, q_fut_B,
                                            feat_V, yc_V, ym_V, prep["val_fut"], device)

        # ---- VAL metrics ----
        gate.eval()
        with torch.no_grad():
            Xv = torch.tensor((feat_V - mu) / sd, dtype=torch.float32, device=device)
            lam_v = gate(Xv).detach().cpu().numpy().flatten()
            ycv = torch.tensor(yc_V, dtype=torch.float32, device=device)
            ymv = torch.tensor(ym_V, dtype=torch.float32, device=device)
            ytv = torch.tensor(prep["val_fut"], dtype=torch.float32, device=device)
            yhat_v = (1 - torch.tensor(lam_v, device=device)[:, None]) * ycv \
                + torch.tensor(lam_v, device=device)[:, None] * ymv
            gate_mse = F.mse_loss(yhat_v, ytv).item()
            chronos_mse = F.mse_loss(ycv, ytv).item()
            fixed_010 = F.mse_loss(0.9 * ycv + 0.1 * ymv, ytv).item()

        lam = np.asarray(lam_v)
        gain = (fixed_010 - gate_mse) / fixed_010 * 100.0
        out_rows.append((dataset, chronos_mse, fixed_010, gate_mse, gain,
                         float(lam.mean()), float(lam.std()),
                         float(np.percentile(lam, 10)), float(np.percentile(lam, 50)),
                         float(np.percentile(lam, 90))))
        torch.save({"state_dict": gate.state_dict(), "mu": mu, "sd": sd},
                   f"results/adaptive_gate/{dataset}_gate.pt")
        torch.cuda.empty_cache()

        print(f"  gate trained: epochs done, best VAL MSE={best_val:.4f}")
        print(f"  time={time.time() - t0:.1f}s  peakRAM={peak_ram:.0f}MB  "
              f"peakVRAM={torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- report table ----
    print("\n" + "=" * 118)
    print("ADAPTIVE FUSION GATE V1 -- VAL results")
    print("=" * 118)
    hdr = (f"{'Dataset':<8}{'Chronos':>10}{'fix0.1':>10}{'Gate':>10}{'gain%':>9}"
           f"{'meanL':>8}{'stdL':>7}{'Lp10':>7}{'Lp50':>7}{'Lp90':>7}")
    print(hdr)
    print("-" * 118)
    for r in out_rows:
        ds, cm, f1, gm, gn, ml, sl, p10, p50, p90 = r
        print(f"{ds:<8}{cm:>10.4f}{f1:>10.4f}{gm:>10.4f}{gn:>+8.2f}"
              f"{ml:>8.3f}{sl:>7.3f}{p10:>7.3f}{p50:>7.3f}{p90:>7.3f}")
    print("-" * 118)


if __name__ == "__main__":
    main()
