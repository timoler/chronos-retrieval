#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 — validate whether the frozen v5 ACCEPT knowledge improves retrieval.

Reads the ALREADY-FROZEN closed-loop output (results/etth1_critic_v5_closed_loop/
final_accepted_knowledge.json). Does NOT re-run the Agent / Critic / Compiler and
does NOT regenerate experts.

Reuses the project's existing retrieval stack:
  - Shape-only:  residual Pearson, Top-K=5, mean-anchor (shape_retrieval).
  - Expert-MLP:  ConditionalExpertMLPRetriever (from run_conditional_expert_mlp_v4),
                 with the N ACCEPT experts as expert embeddings + compiled activations.

Runs:
  1. Shape-only MSE/MAE.
  2. Shape + ALL ACCEPT Expert (n_experts=10) MSE/MAE + gain%.
  3. Per-expert ablation (each expert alone).
  4. Leave-one-out (ALL minus one expert).
  5. Fair information control (ZERO / SHUFFLED / REAL, 3 seeds).

Data protocol: TRAIN = memory/training, VAL = prediction-value validation.
TEST never read. No Chronos/CNN-only/Attention/gate. No new architecture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_expert_attention_retrieval_v2 as attn                    # seed_all
from run_expert_embedding_mlp import build_windows, mae, mse        # noqa: E402
from run_solar_same_val_baselines import residualize, shape_retrieval  # noqa: E402
from run_conditional_expert_mlp_v4 import (                          # noqa: E402
    ConditionalExpertMLPRetriever,
    _linear_slope_vec,
    _lag1_autocorr_vec,
    _sigmoid,
    train_conditional,
    retrieve_conditional,
)
from sentence_transformers import SentenceTransformer                 # noqa: E402

ACC_POOL = Path("results/etth1_critic_v5_closed_loop/final_accepted_knowledge.json")
OUT_DIR = Path("results/etth1_critic_v5_prediction_validation")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
SHUFFLE_SEED = 777


# --------------------------------------------------------------------------- #
# activation helpers
# --------------------------------------------------------------------------- #
def compute_training_stats(X_train_all):
    """TRAIN-only feature stats for z-scoring recent_* features."""
    ctx_means = X_train_all.mean(axis=1)
    ctx_stds = X_train_all.std(axis=1)
    slopes = _linear_slope_vec(X_train_all)
    last_diffs = np.abs(X_train_all[:, -1] - X_train_all[:, -2])
    autocorrs = _lag1_autocorr_vec(X_train_all)
    return {
        "level_mean": float(ctx_means.mean()), "level_std": float(ctx_means.std()),
        "vol_mean": float(ctx_stds.mean()), "vol_std": float(ctx_stds.std()),
        "slope_abs_mean": float(np.abs(slopes).mean()), "slope_abs_std": float(np.abs(slopes).std()),
        "diff_abs_mean": float(last_diffs.mean()), "diff_abs_std": float(last_diffs.std()),
    }


def compute_window_feats(X, tss, stats):
    """Per-window raw feature values (calendar units + continuous z-scores)."""
    n = len(X)
    months = np.array([ts.month for ts in tss], dtype=np.int64)
    hours = np.array([ts.hour for ts in tss], dtype=np.int64)
    dows = np.array([ts.dayofweek for ts in tss], dtype=np.int64)
    ctx_means = X.mean(axis=1)
    ctx_stds = X.std(axis=1)
    slopes = np.abs(_linear_slope_vec(X))
    last_diffs = np.abs(X[:, -1] - X[:, -2])
    autocorrs = _lag1_autocorr_vec(X)
    return {
        "seasonal_level": months,
        "hour_of_day": hours,
        "working_day": dows,
        "day_of_week": dows,
        "recent_level": (ctx_means - stats["level_mean"]) / (stats["level_std"] + 1e-6),
        "recent_volatility": (ctx_stds - stats["vol_mean"]) / (stats["vol_std"] + 1e-6),
        "recent_diff_magnitude": (last_diffs - stats["diff_abs_mean"]) / (stats["diff_abs_std"] + 1e-6),
        "recent_slope_magnitude": (slopes - stats["slope_abs_mean"]) / (stats["slope_abs_std"] + 1e-6),
        "recent_autocorr": (autocorrs + 1.0) / 2.0,
    }


def expert_activation(compiled, feats, i):
    """Evaluate a compiled condition_spec at row i -> scalar in [0,1]."""
    prims = compiled["primitives"]
    logic = compiled.get("logic", "AND")
    vals = []
    for p in prims:
        feat = p["feature"]
        kind = p.get("kind", "calendar")
        if kind == "calendar":
            unit = int(feats[feat][i])
            vals.append(1.0 if unit in p["rule"] else 0.0)
        else:
            if feat == "recent_autocorr":
                v = float(feats[feat][i])                    # already [0,1]
            else:
                v = float(_sigmoid(feats[feat][i]))          # z-score -> [0,1]
            if p.get("qualifier") in ("high", "above_typical"):
                vals.append(v)
            else:                                            # low / below_typical
                vals.append(1.0 - v)
    if logic == "OR":
        return float(max(vals)) if vals else 0.0
    if len(vals) > 1:
        return float(min(vals))                              # binary AND = min
    return float(vals[0]) if vals else 0.0


def build_activation_matrix(experts, X, tss, stats):
    """[n, N_experts] activation matrix."""
    feats = compute_window_feats(X, tss, stats)
    n = len(X)
    M = np.zeros((n, len(experts)), dtype=np.float32)
    for j, e in enumerate(experts):
        compiled = e["compiled"]
        for i in range(n):
            M[i, j] = expert_activation(compiled, feats, i)
    return M


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    # ---- frozen ACCEPT pool (do NOT modify) ----
    accepted = json.loads(ACC_POOL.read_text(encoding="utf-8"))
    # sort by E-id for a stable order
    experts = [accepted[k] for k in sorted(accepted, key=lambda x: int(x[1:]))]
    names = [e["name"] for e in experts]
    n_exp = len(experts)
    print(f"Loaded {n_exp} frozen ACCEPT experts:")
    for e in experts:
        print(f"  {e['expert_id']}: {e['name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- ETTh1 data (TRAIN + VAL; TEST never read) ----
    ds = Path("results/multidataset/ETTh1")
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    X_bank, Y_bank = build_windows(series, bank_starts, CONTEXT, HORIZON)
    X_fit, Y_fit = build_windows(series, fit_starts, CONTEXT, HORIZON)
    X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)
    print(f"bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)} (TEST not read)")

    bank_ts = [dates[s + CONTEXT] for s in bank_starts]
    fit_ts = [dates[s + CONTEXT] for s in fit_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]

    # ---- TRAIN stats + activation matrix ----
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)
    bank_act = build_activation_matrix(experts, X_bank, bank_ts, stats)
    fit_act = build_activation_matrix(experts, X_fit, fit_ts, stats)
    val_act = build_activation_matrix(experts, X_val, val_ts, stats)
    print(f"activation matrix shapes: bank {bank_act.shape} fit {fit_act.shape} val {val_act.shape}")

    # ---- expert embeddings (MiniLM, frozen) ----
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    claims = [e["claim"] for e in experts]
    emb = enc.encode(claims, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    expert_dim = emb.shape[1]
    print(f"expert embeddings: {emb.shape}")

    # ---- A. Shape-only ----
    shape_pred = shape_retrieval(X_val, X_bank, Y_bank, topk=TOPK)
    shape_mse, shape_mae = mse(Y_val, shape_pred), mae(Y_val, shape_pred)
    print(f"\n[A] Shape-only          MSE={shape_mse:.2f} MAE={shape_mae:.2f}")

    def train_and_eval(emb_matrix, b_act, f_act, v_act, seed=None, tag=""):
        """Train ConditionalExpertMLPRetriever on a given expert subset and evaluate."""
        if seed is not None:
            attn.seed_all(seed)
        model = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=emb_matrix.shape[0],
            d_enc=128, d_exp=128, d_z=128, hidden=32,
        ).to(device)
        model.set_expert_embeddings(emb_matrix)
        train_conditional(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                          b_act, f_act, device=device, epochs=30, batch_size=64,
                          candidates=256, lr=1e-3)
        pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                    v_act, b_act, "expert", TOPK, device)
        return mse(Y_val, pred), mae(Y_val, pred)

    # ---- B. ALL ACCEPT ----
    all_mse, all_mae = train_and_eval(emb, bank_act, fit_act, val_act, seed=42, tag="ALL")
    gain_all = (shape_mse - all_mse) / shape_mse * 100.0
    print(f"[B] ALL ACCEPT (n={n_exp}) MSE={all_mse:.2f} MAE={all_mae:.2f} gain={gain_all:+.2f}%")

    # ---- C. Per-expert ablation ----
    per_expert = []
    print("\n[Per-expert ablation]")
    for j, e in enumerate(experts):
        eid = e["expert_id"]
        em = emb[j:j + 1]                                  # [1, E]
        b_act = bank_act[:, j:j + 1]
        f_act = fit_act[:, j:j + 1]
        v_act = val_act[:, j:j + 1]
        m, a = train_and_eval(em, b_act, f_act, v_act, seed=42, tag=eid)
        gain = (shape_mse - m) / shape_mse * 100.0
        per_expert.append({"expert_id": eid, "name": names[j], "claim": e["claim"],
                           "mse": m, "mae": a, "vs_shape_gain_pct": gain})
        print(f"  {eid} {names[j]:<28} MSE={m:.2f} MAE={a:.2f} gain={gain:+.2f}%")

    # ---- D. Leave-one-out ----
    leave_one_out = []
    print("\n[Leave-one-out]")
    for j, e in enumerate(experts):
        eid = e["expert_id"]
        keep = [k for k in range(n_exp) if k != j]
        em = emb[keep]                                     # [9, E]
        b_act = bank_act[:, keep]
        f_act = fit_act[:, keep]
        v_act = val_act[:, keep]
        m, a = train_and_eval(em, b_act, f_act, v_act, seed=42, tag=f"ALL-{eid}")
        gain = (shape_mse - m) / shape_mse * 100.0
        leave_one_out.append({"removed_expert_id": eid, "removed_name": names[j],
                              "mse": m, "mae": a, "vs_shape_gain_pct": gain,
                              "delta_vs_all": m - all_mse})
        print(f"  ALL-{eid} (removed {names[j]:<26}) MSE={m:.2f} MAE={a:.2f} "
              f"gain={gain:+.2f}%  ΔvsALL={m-all_mse:+.2f}")

    # ---- E. Fair information control (ZERO / SHUFFLED / REAL, 3 seeds) ----
    fair = {"zero": [], "shuffled": [], "real": []}
    n_bank, n_fit, n_val = len(bank_act), len(fit_act), len(val_act)
    # fixed shuffle permutation (one permutation per split, computed once)
    def perm_act(a, seed):
        rng = np.random.default_rng(seed)
        return a[rng.permutation(len(a))]

    bank_shuf = perm_act(bank_act, SHUFFLE_SEED)
    fit_shuf = perm_act(fit_act, SHUFFLE_SEED)
    val_shuf = perm_act(val_act, SHUFFLE_SEED)
    bank_zero = np.zeros_like(bank_act)
    fit_zero = np.zeros_like(fit_act)
    val_zero = np.zeros_like(val_act)

    print("\n[Fair information control (3 seeds)]")
    for seed in SEEDS:
        # shared initialization: create one model, copy state_dict to the three arms
        attn.seed_all(seed)
        base = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_exp,
            d_enc=128, d_exp=128, d_z=128, hidden=32,
        ).to(device)
        base.set_expert_embeddings(emb)
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

        for arm, b_act, f_act, v_act in (
            ("zero", bank_zero, fit_zero, val_zero),
            ("shuffled", bank_shuf, fit_shuf, val_shuf),
            ("real", bank_act, fit_act, val_act),
        ):
            model = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_exp,
                d_enc=128, d_exp=128, d_z=128, hidden=32,
            ).to(device)
            model.set_expert_embeddings(emb)
            model.load_state_dict(init_state)
            train_conditional(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                              b_act, f_act, device=device, epochs=30, batch_size=64,
                              candidates=256, lr=1e-3)
            pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                        v_act, b_act, "expert", TOPK, device)
            fair[arm].append({"seed": seed, "mse": mse(Y_val, pred), "mae": mae(Y_val, pred)})
            print(f"  seed {seed} [{arm:<8}] MSE={fair[arm][-1]['mse']:.2f} "
                  f"MAE={fair[arm][-1]['mae']:.2f}")

    def mean_std(vals):
        a = np.array([v["mse"] for v in vals], dtype=np.float64)
        b = np.array([v["mae"] for v in vals], dtype=np.float64)
        return float(a.mean()), float(a.std()), float(b.mean()), float(b.std())

    fair_summary = {}
    for arm in ("zero", "shuffled", "real"):
        mm, ms, am, as_ = mean_std(fair[arm])
        fair_summary[arm] = {"mse_mean": mm, "mse_std": ms, "mae_mean": am, "mae_std": as_,
                             "per_seed": fair[arm]}

    gain_real_vs_zero = (fair_summary["zero"]["mse_mean"] - fair_summary["real"]["mse_mean"]) / fair_summary["zero"]["mse_mean"] * 100.0
    gain_real_vs_shuf = (fair_summary["shuffled"]["mse_mean"] - fair_summary["real"]["mse_mean"]) / fair_summary["shuffled"]["mse_mean"] * 100.0

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": "ETTh1",
        "protocol": "multidataset ETTh1 VAL, residualize Pearson, Top-K=5 mean",
        "shape_only": {"mse": shape_mse, "mae": shape_mae},
        "all_accept": {"mse": all_mse, "mae": all_mae, "n_experts": n_exp,
                       "gain_vs_shape_pct": gain_all},
        "fair_information_control": fair_summary,
        "fair_gain_real_vs_zero_pct": gain_real_vs_zero,
        "fair_gain_real_vs_shuffled_pct": gain_real_vs_shuf,
        "seeds": SEEDS,
        "flags": {
            "critic_rerun": False,
            "expert_regenerated": False,
            "accepted_pool_modified": False,
            "test_used": False,
        },
    }
    (OUT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "per_expert_ablation.json").write_text(
        json.dumps(per_expert, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "leave_one_out.json").write_text(
        json.dumps(leave_one_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "fair_information_control.json").write_text(
        json.dumps(fair_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- report ----
    print("\n" + "=" * 78)
    print("ETTH1 PREDICTION VALIDATION (frozen v5 ACCEPT knowledge)")
    print("=" * 78)
    print(f"Shape-only      MSE = {shape_mse:.2f}  MAE = {shape_mae:.2f}")
    print(f"ALL ACCEPT      MSE = {all_mse:.2f}  MAE = {all_mae:.2f}  gain = {gain_all:+.2f}%")
    print("-" * 78)
    print("Fair control (mean±std over 3 seeds):")
    for arm in ("zero", "shuffled", "real"):
        s = fair_summary[arm]
        print(f"  {arm:<8} MSE = {s['mse_mean']:.2f} ± {s['mse_std']:.2f}  "
              f"MAE = {s['mae_mean']:.2f} ± {s['mae_std']:.2f}")
    print(f"  REAL vs ZERO     gain = {gain_real_vs_zero:+.2f}%")
    print(f"  REAL vs SHUFFLED gain = {gain_real_vs_shuf:+.2f}%")
    print("=" * 78)
    print(f"Saved to {OUT_DIR}")
    print("critic_rerun=false, expert_regenerated=false, accepted_pool_modified=false, test_used=false")


if __name__ == "__main__":
    main()
