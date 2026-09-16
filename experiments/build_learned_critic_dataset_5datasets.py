#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned Critic — Stage 1: Cross-Dataset Knowledge-Effect Dataset Builder.

Collects, on 5 univariate-OT time-series datasets, per Raw Expert:

  * dataset generic features            (TRAIN-only)
  * expert activation statistics        (TRAIN query windows)
  * Critic v6.1 audit decisions         (FROZEN critic, no dataset-specific rule)
  * real forecasting marginal contribution (deterministic Leave-One-Out)

No Learned Critic network is trained here.  The critic audit is only an input
feature; the ground-truth label is the LOO marginal effect, computed by zeroing a
single expert's activation column while keeping architecture / params / init /
optimizer / epochs / batch order / candidate memory identical.

Critic is FROZEN = v6.1 (v6 core in run_expert_agent_closed_loop_v6 + Post-Repair
Activation Safety Check).  No dataset-specific / expert-specific rule is added.  The
only per-dataset INPUT is metadata + TRAIN evidence + the frozen subset-proxy feature
for electricity's E7 (an observed covariate already frozen in its raw pool, not a
critic rule).

TEST is never read.  test_used = false.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# frozen reuse (verbatim imports; nothing here modifies the Critic)
# --------------------------------------------------------------------------- #
import run_expert_agent_closed_loop_v6 as v6mod                 # noqa: E402
import run_electricity_critic_v6_1_cross_dataset as elec        # noqa: E402  (helpers + elec evidence/activation)

from run_expert_embedding_mlp import build_windows, mse, mae    # noqa: E402
from run_solar_same_val_baselines import residualize            # noqa: E402
from run_conditional_expert_mlp_v4 import (                     # noqa: E402
    ConditionalExpertMLPRetriever,
)
from run_etth1_prediction_validation import (                   # noqa: E402
    compute_training_stats,
    build_activation_matrix,
)

seed_everything = elec.seed_everything
embed_claims = elec.embed_claims
_sorted_experts = elec._sorted_experts
_corr = elec._corr
_jaccard = elec._jaccard
train_arm = elec.train_arm
evaluate = elec.evaluate
n_params = elec.n_params
FEATURE_REGISTRY = elec.FEATURE_REGISTRY            # generic + x_subset (union)
build_electricity_evidence = elec.build_electricity_evidence
build_activation_matrix_elec = elec.build_activation_matrix_elec
LOAD_VARS = elec.LOAD_VARS
N_POPULATION = elec.N_POPULATION

COMPILER_CONFIG = v6mod.COMPILER_CONFIG
process_expert_v6 = v6mod.process_expert_v6
audit_activation_coverage = v6mod.audit_activation_coverage

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
CONTEXT, HORIZON = 512, 96
SEEDS = [42, 123, 2026]
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

OUT_ROOT = Path("results/learned_critic_dataset_v1")

# --------------------------------------------------------------------------- #
# dataset registry (univariate-OT inputs only)
# --------------------------------------------------------------------------- #
DATASETS = [
    {
        "key": "etth1", "dir": "ETTh1",
        "raw_pool": "results/etth1_critic_v5_closed_loop/expert_round_1.json",
        "sampling_str": "hourly", "sampling_minutes": 60,
        "metadata": {"target_description": "transformer oil temperature", "n_population": 1},
        "evidence": "generic",
    },
    {
        "key": "electricity", "dir": "electricity",
        "raw_pool": "results/electricity_critic_v5_closed_loop/expert_round_1.json",
        "sampling_str": "15 minutes", "sampling_minutes": 15,
        "metadata": {"target_description": "aggregate load of many consumers", "n_population": N_POPULATION},
        "evidence": "electricity",
    },
    {
        "key": "solar", "dir": "solar",
        "raw_pool": "results/cross_dataset_validation/solar/critic/expert_round_1.json",
        "sampling_str": "10 minutes", "sampling_minutes": 10,
        "metadata": {"target_description": "aggregate solar generation", "n_population": 1},
        "evidence": "generic",
    },
    {
        "key": "weather", "dir": "weather",
        "raw_pool": "results/cross_dataset_validation/weather/critic/expert_round_1.json",
        "sampling_str": "10 minutes", "sampling_minutes": 10,
        "metadata": {"target_description": "a single weather-station meteorological variable", "n_population": 1},
        "evidence": "generic",
    },
    {
        "key": "exchange_rate", "dir": "exchange_rate",
        "raw_pool": "results/cross_dataset_validation/exchange_rate/critic/expert_round_1.json",
        "sampling_str": "daily", "sampling_minutes": 1440,
        "metadata": {"target_description": "a daily foreign exchange rate", "n_population": 1},
        "evidence": "generic",
    },
]


# --------------------------------------------------------------------------- #
# generic TRAIN-only evidence (OT + timestamp) — same schema as
# build_electricity_evidence (feature_distributions with q10..q90), so it is
# compatible with compile_condition_v6 / critique_v6.
# --------------------------------------------------------------------------- #
def build_generic_evidence(series_csv: str) -> dict:
    df = pd.read_csv(series_csv)
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    dt = dates.iloc[:train_len]
    y = df["OT"].to_numpy(np.float64)[:train_len]
    months = dt.dt.month.to_numpy()
    hours = dt.dt.hour.to_numpy()
    dows = dt.dt.dayofweek.to_numpy()

    months_present = sorted(int(m) for m in set(months.tolist()))
    hours_present = sorted(int(h) for h in set(hours.tolist()))
    dows_present = sorted(int(d) for d in set(dows.tolist()))

    monthly_mean = {int(m): float(y[months == m].mean()) for m in months_present}
    monthly_vals = [monthly_mean[m] for m in months_present]
    mdir, nrev = {}, 0
    for i in range(1, len(monthly_vals)):
        d = "up" if monthly_vals[i] > monthly_vals[i - 1] else ("down" if monthly_vals[i] < monthly_vals[i - 1] else "flat")
        mdir[f"{months_present[i-1]}->{months_present[i]}"] = d
        if i >= 2 and mdir[f"{months_present[i-2]}->{months_present[i-1]}"] != d \
                and mdir[f"{months_present[i-2]}->{months_present[i-1]}"] != "flat" and d != "flat":
            nrev += 1

    def _safe_mean(vals):
        vals = np.asarray(vals, float)
        return float(vals.mean()) if len(vals) > 0 else 0.0

    hourly_mean = np.array([_safe_mean(y[hours == h]) for h in range(24)])
    dow_mean = np.array([_safe_mean(y[dows == d]) for d in range(7)])

    ctx = 64
    n_win = train_len - ctx
    step = max(1, n_win // 5000)
    t = np.linspace(-1, 1, ctx)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    feat_vals = {k: [] for k in ("recent_level", "recent_volatility", "recent_slope_magnitude",
                                 "recent_diff_magnitude", "recent_autocorr")}
    for i in range(0, n_win, step):
        x = y[i:i + ctx]
        feat_vals["recent_level"].append(float(x.mean()))
        feat_vals["recent_volatility"].append(float(x.std()))
        xm = x - x.mean()
        feat_vals["recent_slope_magnitude"].append(abs(float(np.sum(xm * tc) / denom)))
        feat_vals["recent_diff_magnitude"].append(abs(float(x[-1] - x[-2])))
        feat_vals["recent_autocorr"].append(_corr(x[1:], x[:-1]))

    feature_distributions = {}
    for feat, arr in feat_vals.items():
        a = np.asarray(arr, float)
        feature_distributions[feat] = {
            "q10": float(np.percentile(a, 10)), "q25": float(np.percentile(a, 25)),
            "q50": float(np.percentile(a, 50)), "q75": float(np.percentile(a, 75)),
            "q90": float(np.percentile(a, 90)), "mean": float(a.mean()), "std": float(a.std()),
        }

    return {
        "temporal_coverage": {
            "months_present": months_present,
            "months_missing": [m for m in range(1, 13) if m not in months_present],
            "month_fraction": round(len(months_present) / 12, 4),
            "hours_present": hours_present,
            "hours_missing": [h for h in range(24) if h not in hours_present],
            "hour_fraction": round(len(hours_present) / 24, 4),
            "dows_present": dows_present, "dow_fraction": round(len(dows_present) / 7, 4),
        },
        "diurnal": {"hourly_mean": [round(v, 2) for v in hourly_mean],
                    "eta2_hour_on_ot": round(_eta2(hours, y), 4)},
        "seasonal": {"monthly_mean": {k: round(v, 2) for k, v in monthly_mean.items()},
                     "monthly_mean_values": [round(v, 2) for v in monthly_vals],
                     "month_direction": mdir, "n_month_direction_reversals": nrev,
                     "eta2_month_on_ot": round(_eta2(months, y), 4)},
        "weekly": {"dow_mean": [round(v, 2) for v in dow_mean],
                   "eta2_dow_on_ot": round(_eta2(dows, y), 4)},
        "population": {"n_proxy": 1, "n_population": 1, "proxy_fraction": 1.0,
                       "corr_proxy_sum_vs_target": 1.0},
        "x_subset_representativeness": {},
        "feature_distributions": feature_distributions,
    }


def _eta2(groups, y):
    groups, y = np.asarray(groups), np.asarray(y, float)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    within = sum(float(((y[groups == g] - y[groups == g].mean()) ** 2).sum())
                 for g in np.unique(groups))
    return 1.0 - within / total


# --------------------------------------------------------------------------- #
# dataset generic features (Section 10; TRAIN-only)
# --------------------------------------------------------------------------- #
def _fft_period_features(y):
    y = np.asarray(y, float)
    n = len(y)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    try:
        y = y - np.polyval(np.polyfit(x, y, 1), x)   # remove linear trend
    except Exception:
        pass
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n)
    power = np.abs(Y) ** 2
    power[0] = 0.0
    total = float(power.sum())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    idx = int(np.argmax(power))
    dom_period = float(1.0 / freqs[idx]) if freqs[idx] > 0 else float("nan")
    dom_strength = float(power[idx] / total)
    pnorm = power / total
    pnorm = pnorm[pnorm > 0]
    ent = float(-np.sum(pnorm * np.log(pnorm)) / np.log(len(pnorm))) if len(pnorm) > 1 else 0.0
    return dom_period, dom_strength, ent


def _trend_strength(y):
    y = np.asarray(y, float)
    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return float("nan")
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _nonstationarity_score(y):
    y = np.asarray(y, float)
    n = len(y)
    nblocks = 4
    edges = np.linspace(0, n, nblocks + 1).astype(int)
    bm, bs = [], []
    for b in range(nblocks):
        seg = y[edges[b]:edges[b + 1]]
        if len(seg) > 0:
            bm.append(float(seg.mean()))
            bs.append(float(seg.std()))
    bm, bs = np.asarray(bm), np.asarray(bs)
    cv_m = float(bm.std() / bm.mean()) if bm.mean() != 0 else 0.0
    cv_s = float(bs.std() / bs.mean()) if bs.mean() != 0 else 0.0
    return float(cv_m + cv_s)


def dataset_generic_features(series: np.ndarray, train_end: int, cfg: dict) -> dict:
    y = series[:train_end]
    n = int(len(y))
    d = np.abs(np.diff(y)) if n > 1 else np.array([0.0])
    dom_period, dom_strength, spectral_entropy = _fft_period_features(y)
    return {
        "n_train_points": n,
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "normalized_mean_abs_diff": float(d.mean()) / (float(y.std()) + 1e-9) if n > 1 else float("nan"),
        "lag1_autocorrelation": _corr(y[1:], y[:-1]) if n > 2 else float("nan"),
        "trend_strength": _trend_strength(y),
        "nonstationarity_score": _nonstationarity_score(y) if n >= 4 else float("nan"),
        "dominant_period": dom_period,
        "dominant_period_strength": dom_strength,
        "spectral_entropy": spectral_entropy,
        "context_length": CONTEXT,
        "forecast_horizon": HORIZON,
        "sampling_interval": cfg["sampling_str"],
        "sampling_interval_minutes": cfg["sampling_minutes"],
    }


# --------------------------------------------------------------------------- #
# expert activation features (Section 11; TRAIN = bank + fit query windows)
# --------------------------------------------------------------------------- #
def activation_features(act_vec: np.ndarray) -> dict:
    a = np.asarray(act_vec, dtype=np.float64)
    cov = audit_activation_coverage(a, [], n_blocks=4)
    st = cov["stats"]
    binary = bool(st["binary"])
    block_means = [cov["chronological_blocks"][f"block{b}"]["activation_mean"] for b in range(1, 5)]
    bm = np.asarray(block_means, float)
    mean_a = float(st["activation_mean"])
    temporal_stability = float(1.0 - min(1.0, bm.std() / (mean_a + 1e-6))) if mean_a > 1e-6 else float("nan")
    return {
        "activation_mean": float(st["activation_mean"]),
        "activation_std": float(st["std"]),
        "activation_min": float(st["min"]),
        "activation_max": float(st["max"]),
        "activation_zero_ratio": float((a == 0.0).mean()),
        "activation_nonzero_ratio": float((a != 0.0).mean()),
        "activation_unique_ratio": float(st["unique_count"]) / max(1, len(a)),
        "activation_block1_mean": float(block_means[0]),
        "activation_block2_mean": float(block_means[1]),
        "activation_block3_mean": float(block_means[2]),
        "activation_block4_mean": float(block_means[3]),
        "activation_block_mean_std": float(bm.std()),
        "activation_temporal_stability": temporal_stability,
        "activation_positive_rate": float(st["active_ratio"]) if binary else float("nan"),
        "activation_binary": binary,
    }


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def make_model(n_experts: int, expert_dim: int, device):
    return ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=expert_dim, n_experts=n_experts,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)


def capture_init_state(seed: int, n_experts: int, expert_dim: int, device):
    seed_everything(seed)
    base = make_model(n_experts, expert_dim, device)
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    torch.cuda.empty_cache()
    return state


def train_eval_one(init_state, emb, device,
                   X_bank, Y_bank, X_fit, Y_fit, X_val, Y_val,
                   X_bank_res, X_fit_res, X_val_res, bank_act, fit_act, val_act):
    """Train one arm (init from init_state, use_shape=True) -> (mse, mae)."""
    n_experts = int(emb.shape[0])
    expert_dim = int(emb.shape[1])
    model = make_model(n_experts, expert_dim, device)
    model.load_state_dict(init_state)
    model.set_expert_embeddings(emb)
    train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, True, device)
    pred = evaluate(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                    val_act, bank_act, True, device)
    m, a = mse(Y_val, pred), mae(Y_val, pred)
    del model
    torch.cuda.empty_cache()
    return float(m), float(a)


# --------------------------------------------------------------------------- #
# data preparation (per dataset, no training)
# --------------------------------------------------------------------------- #
def load_and_prepare(cfg: dict, device):
    ds_dir = Path("results/multidataset") / cfg["dir"]
    series_csv = ds_dir / "series.csv"
    window_npz = ds_dir / "window_indices.npz"

    df = pd.read_csv(series_csv)
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])

    idx = np.load(window_npz)
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)          # TEST never read

    raw_experts = _sorted_experts(json.loads(Path(cfg["raw_pool"]).read_text(encoding="utf-8")))
    n_raw = len(raw_experts)

    # evidence
    if cfg["evidence"] == "electricity":
        evidence = build_electricity_evidence()
    else:
        evidence = build_generic_evidence(str(series_csv))

    # bank/fit split (80/20 of TRAIN), frozen
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    X_bank, Y_bank = build_windows(series, bank_starts, CONTEXT, HORIZON)
    X_fit, Y_fit = build_windows(series, fit_starts, CONTEXT, HORIZON)
    X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)

    bank_ts = [dates[s + CONTEXT] for s in bank_starts]
    fit_ts = [dates[s + CONTEXT] for s in fit_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]

    stats = compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    # raw activation matrices (frozen v5 compiled from expert_round_1.json)
    if cfg["evidence"] == "electricity":
        df_raw = pd.read_csv("data/electricity.csv")
        s6 = np.sum([df_raw[v].to_numpy(np.float64) for v in LOAD_VARS], axis=0)
        s6_train = s6[: int(0.6 * len(df_raw))]
        s6_mean, s6_std = float(s6_train.mean()), float(s6_train.std())
        d6 = np.abs(np.diff(s6))
        d6_train = d6[: max(1, int(0.6 * len(df_raw)) - 1)]
        d6_mean, d6_std = float(d6_train.mean()), float(d6_train.std())
        raw_bank_act = build_activation_matrix_elec(raw_experts, X_bank, bank_ts, stats, bank_starts,
                                                    s6, d6, s6_mean, s6_std, d6_mean, d6_std)
        raw_fit_act = build_activation_matrix_elec(raw_experts, X_fit, fit_ts, stats, fit_starts,
                                                   s6, d6, s6_mean, s6_std, d6_mean, d6_std)
        raw_val_act = build_activation_matrix_elec(raw_experts, X_val, val_ts, stats, val_starts,
                                                   s6, d6, s6_mean, s6_std, d6_mean, d6_std)
    else:
        raw_bank_act = build_activation_matrix(raw_experts, X_bank, bank_ts, stats)
        raw_fit_act = build_activation_matrix(raw_experts, X_fit, fit_ts, stats)
        raw_val_act = build_activation_matrix(raw_experts, X_val, val_ts, stats)

    # expert embeddings (MiniLM, already cached)
    emb = embed_claims([e["claim"] for e in raw_experts], device)
    expert_dim = int(emb.shape[1])

    return {
        "cfg": cfg, "series": series, "dates": dates,
        "train_starts": train_starts, "val_starts": val_starts,
        "bank_starts": bank_starts, "fit_starts": fit_starts,
        "raw_experts": raw_experts, "n_raw": n_raw, "evidence": evidence,
        "X_bank": X_bank, "Y_bank": Y_bank, "X_fit": X_fit, "Y_fit": Y_fit,
        "X_val": X_val, "Y_val": Y_val,
        "X_bank_res": X_bank_res, "X_fit_res": X_fit_res, "X_val_res": X_val_res,
        "bank_ts": bank_ts, "fit_ts": fit_ts, "val_ts": val_ts,
        "raw_bank_act": raw_bank_act, "raw_fit_act": raw_fit_act, "raw_val_act": raw_val_act,
        "emb": emb, "expert_dim": expert_dim,
    }


# --------------------------------------------------------------------------- #
# Critic v6.1 audit (per expert)
# --------------------------------------------------------------------------- #
def run_audit(cfg: dict, data: dict):
    raw_experts = data["raw_experts"]
    n_raw = data["n_raw"]
    evidence = data["evidence"]
    metadata = cfg["metadata"]

    audit_rows = []
    v6_compiled_list = []

    for e in raw_experts:
        e.setdefault("id", e.get("expert_id"))
        raw_prim = e.get("compiled", {}).get("primitives", [])
        pre_state = "inactive" if len(raw_prim) == 0 else "active"

        proc = process_expert_v6(e, evidence, metadata, FEATURE_REGISTRY, COMPILER_CONFIG)
        compiled = proc["compiled"]
        v6_compiled_list.append(compiled)
        post_state = "active" if compiled["status"] == "COMPILED" and len(compiled["primitives"]) >= 1 else "inactive"
        repair_happened = len(compiled.get("repairs", [])) > 0
        triggered = (pre_state == "inactive") and (post_state == "active") and repair_happened

        obs = proc["observability_audit"]
        checks = proc["checks"]
        proxy = proc["proxy_validation"]
        ev_pass = bool(obs["all_observable"] and checks["support"]["ok"] and checks["coverage"]["ok"]
                       and checks["stability"]["ok"] and checks["counter"]["ok"]
                       and checks["population"]["ok"])
        fid_pass = bool(proxy["faithful"] and compiled["status"] == "COMPILED")

        audit_rows.append({
            "expert_id": e["expert_id"], "name": e.get("name", ""),
            "pre_state": pre_state, "post_state": post_state,
            "repair_happened": repair_happened, "triggered": triggered,
            "ev_pass": ev_pass, "fid_pass": fid_pass,
            "compiled": compiled,
            "observability": obs, "proxy": proxy,
            "lineage": proc["lineage"],
            "final_decision_v6": proc["final_decision"],
        })
        if triggered:
            print(f"    TRIGGER: {e['expert_id']} ({pre_state} -> {post_state})", flush=True)

    # generalization (activation coverage on TRAIN raw activation)
    raw_train_act = np.concatenate([data["raw_bank_act"], data["raw_fit_act"]], axis=0)
    train_ts_all = data["bank_ts"] + data["fit_ts"]
    for i in range(n_raw):
        a = raw_train_act[:, i]
        cov = audit_activation_coverage(a, train_ts_all, n_blocks=4)
        audit_rows[i]["coverage"] = cov
        audit_rows[i]["gen_pass"] = bool((not cov["dead"]) and (not cov["constant"]))

    # redundancy (sequential dedup by expert_id order, same as electricity)
    passing = [i for i in range(n_raw) if audit_rows[i]["ev_pass"] and audit_rows[i]["fid_pass"] and audit_rows[i]["gen_pass"]]
    final_pool = []
    for i in passing:
        a = raw_train_act[:, i]
        dup = False
        for j in final_pool:
            b = raw_train_act[:, j]
            if abs(_corr(a, b)) > 0.95 or _jaccard(a, b) > 0.95:
                dup = True
                break
        if not dup:
            final_pool.append(i)
    keep = set(final_pool)

    for i in range(n_raw):
        audit_rows[i]["redundant"] = bool(i in passing and i not in keep)
        audit_rows[i]["in_accept_pool"] = bool(i in keep)

    return {
        "rows": audit_rows,
        "triggered_idx": [i for i in range(n_raw) if audit_rows[i]["triggered"]],
        "keep": keep, "final_pool": final_pool,
        "v6_compiled_list": v6_compiled_list,
        "n_repair_created_activation": sum(1 for i in range(n_raw) if audit_rows[i]["triggered"]),
    }


# --------------------------------------------------------------------------- #
# safety check (paired ON/OFF, only for repair-created activations)
# --------------------------------------------------------------------------- #
def build_v6_activation(cfg: dict, data: dict, audit: dict):
    """v6-compiled activation matrices (used only for the safety check)."""
    n_raw = data["n_raw"]
    v6_experts = [dict(e) for e in data["raw_experts"]]
    for i in range(n_raw):
        v6_experts[i]["compiled"] = audit["v6_compiled_list"][i]

    X_bank, X_fit, X_val = data["X_bank"], data["X_fit"], data["X_val"]
    bank_ts, fit_ts, val_ts = data["bank_ts"], data["fit_ts"], data["val_ts"]
    bank_starts, fit_starts, val_starts = data["bank_starts"], data["fit_starts"], data["val_starts"]
    stats = compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    if cfg["evidence"] == "electricity":
        df_raw = pd.read_csv("data/electricity.csv")
        s6 = np.sum([df_raw[v].to_numpy(np.float64) for v in LOAD_VARS], axis=0)
        s6_train = s6[: int(0.6 * len(df_raw))]
        s6_mean, s6_std = float(s6_train.mean()), float(s6_train.std())
        d6 = np.abs(np.diff(s6))
        d6_train = d6[: max(1, int(0.6 * len(df_raw)) - 1)]
        d6_mean, d6_std = float(d6_train.mean()), float(d6_train.std())
        v6_bank_act = build_activation_matrix_elec(v6_experts, X_bank, bank_ts, stats, bank_starts,
                                                   s6, d6, s6_mean, s6_std, d6_mean, d6_std)
        v6_fit_act = build_activation_matrix_elec(v6_experts, X_fit, fit_ts, stats, fit_starts,
                                                  s6, d6, s6_mean, s6_std, d6_mean, d6_std)
        v6_val_act = build_activation_matrix_elec(v6_experts, X_val, val_ts, stats, val_starts,
                                                  s6, d6, s6_mean, s6_std, d6_mean, d6_std)
    else:
        v6_bank_act = build_activation_matrix(v6_experts, X_bank, bank_ts, stats)
        v6_fit_act = build_activation_matrix(v6_experts, X_fit, fit_ts, stats)
        v6_val_act = build_activation_matrix(v6_experts, X_val, val_ts, stats)

    # mask to ACCEPT pool
    v6_bank = np.zeros_like(data["raw_bank_act"])
    v6_fit = np.zeros_like(data["raw_fit_act"])
    v6_val = np.zeros_like(data["raw_val_act"])
    for i in audit["keep"]:
        v6_bank[:, i] = v6_bank_act[:, i]
        v6_fit[:, i] = v6_fit_act[:, i]
        v6_val[:, i] = v6_val_act[:, i]
    return v6_bank, v6_fit, v6_val


def run_safety_check(cfg: dict, data: dict, audit: dict, device):
    triggered_idx = audit["triggered_idx"]
    safety = {}
    disabled = set()
    if not triggered_idx:
        return safety, disabled

    n_raw = data["n_raw"]
    expert_dim = data["expert_dim"]
    emb = data["emb"]
    X_bank, Y_bank = data["X_bank"], data["Y_bank"]
    X_fit, Y_fit = data["X_fit"], data["Y_fit"]
    X_val, Y_val = data["X_val"], data["Y_val"]
    X_bank_res, X_fit_res, X_val_res = data["X_bank_res"], data["X_fit_res"], data["X_val_res"]

    v6_bank, v6_fit, v6_val = build_v6_activation(cfg, data, audit)

    on_mse = {}
    for seed in SEEDS:
        st = capture_init_state(seed, n_raw, expert_dim, device)
        on_mse[seed], _ = train_eval_one(st, emb, device,
                                         X_bank, Y_bank, X_fit, Y_fit, X_val, Y_val,
                                         X_bank_res, X_fit_res, X_val_res, v6_bank, v6_fit, v6_val)

    for i in triggered_idx:
        bank_off = v6_bank.copy(); bank_off[:, i] = 0.0
        fit_off = v6_fit.copy(); fit_off[:, i] = 0.0
        val_off = v6_val.copy(); val_off[:, i] = 0.0
        off_mse = {}
        for seed in SEEDS:
            st = capture_init_state(seed, n_raw, expert_dim, device)
            off_mse[seed], _ = train_eval_one(st, emb, device,
                                              X_bank, Y_bank, X_fit, Y_fit, X_val, Y_val,
                                              X_bank_res, X_fit_res, X_val_res, bank_off, fit_off, val_off)
        deltas = {s: on_mse[s] - off_mse[s] for s in SEEDS}
        mean_d = float(np.mean([deltas[s] for s in SEEDS]))
        n_pos = sum(1 for s in SEEDS if deltas[s] > 0)
        fail = (n_pos >= 2 and mean_d > 0)
        safety[i] = {"deltas": {s: round(deltas[s], 4) for s in SEEDS},
                     "mean_delta": round(mean_d, 4),
                     "safety": "FAIL" if fail else "PASS"}
        if fail:
            disabled.add(i)
            print(f"    safety-FAIL {data['raw_experts'][i]['expert_id']}", flush=True)

    return safety, disabled


# --------------------------------------------------------------------------- #
# training (FULL_RAW + LOO) with checkpoint/resume
# --------------------------------------------------------------------------- #
def run_training(cfg: dict, data: dict, device, limit_experts: int = None):
    key = cfg["key"]
    n_raw = data["n_raw"]
    expert_dim = data["expert_dim"]
    emb = data["emb"]
    X_bank, Y_bank = data["X_bank"], data["Y_bank"]
    X_fit, Y_fit = data["X_fit"], data["Y_fit"]
    X_val, Y_val = data["X_val"], data["Y_val"]
    X_bank_res, X_fit_res, X_val_res = data["X_bank_res"], data["X_fit_res"], data["X_val_res"]
    raw_bank_act, raw_fit_act, raw_val_act = data["raw_bank_act"], data["raw_fit_act"], data["raw_val_act"]

    runs_dir = OUT_ROOT / key / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    def run_file(kind: str, eid: str, seed: int) -> Path:
        return runs_dir / f"{kind}__{eid}__seed{seed}.json"

    def done(kind: str, eid: str, seed: int) -> bool:
        return run_file(kind, eid, seed).exists()

    def save_run(kind: str, eid: str, seed: int, m, a, pcount, runtime):
        rec = {"dataset": key, "kind": kind, "expert_id": eid, "seed": seed,
               "mse": m, "mae": a, "param_count": pcount, "runtime_seconds": runtime,
               "status": "done", "use_shape": True}
        run_file(kind, eid, seed).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_run(kind: str, eid: str, seed: int):
        return json.loads(run_file(kind, eid, seed).read_text(encoding="utf-8"))

    param_count_full = n_params(make_model(n_raw, expert_dim, device))
    full_mse, full_mae = {}, {}
    wo_mse = {i: {} for i in range(n_raw)}
    wo_mae = {i: {} for i in range(n_raw)}

    init_states = {seed: capture_init_state(seed, n_raw, expert_dim, device) for seed in SEEDS}

    for seed in SEEDS:
        st = init_states[seed]
        # FULL_RAW
        if done("FULL", "ALL", seed):
            r = load_run("FULL", "ALL", seed)
            full_mse[seed] = r["mse"]; full_mae[seed] = r["mae"]
        else:
            t0 = time.time()
            full_mse[seed], full_mae[seed] = train_eval_one(
                st, emb, device, X_bank, Y_bank, X_fit, Y_fit, X_val, Y_val,
                X_bank_res, X_fit_res, X_val_res, raw_bank_act, raw_fit_act, raw_val_act)
            save_run("FULL", "ALL", seed, full_mse[seed], full_mae[seed], param_count_full, round(time.time() - t0, 2))
            print(f"  [{key}] FULL seed{seed} mse={full_mse[seed]:.4f} param={param_count_full}", flush=True)

        # LOO per expert
        n_loo = n_raw if limit_experts is None else min(n_raw, limit_experts)
        for i in range(n_loo):
            eid = data["raw_experts"][i]["expert_id"]
            bank_off = raw_bank_act.copy(); bank_off[:, i] = 0.0
            fit_off = raw_fit_act.copy(); fit_off[:, i] = 0.0
            val_off = raw_val_act.copy(); val_off[:, i] = 0.0
            if done("LOO", eid, seed):
                r = load_run("LOO", eid, seed)
                wo_mse[i][seed] = r["mse"]; wo_mae[i][seed] = r["mae"]
            else:
                t0 = time.time()
                wo_mse[i][seed], wo_mae[i][seed] = train_eval_one(
                    st, emb, device, X_bank, Y_bank, X_fit, Y_fit, X_val, Y_val,
                    X_bank_res, X_fit_res, X_val_res, bank_off, fit_off, val_off)
                save_run("LOO", eid, seed, wo_mse[i][seed], wo_mae[i][seed], param_count_full, round(time.time() - t0, 2))
                print(f"  [{key}] LOO {eid} seed{seed} mse={wo_mse[i][seed]:.4f} param={param_count_full}", flush=True)

    return {"full_mse": full_mse, "full_mae": full_mae,
            "wo_mse": wo_mse, "wo_mae": wo_mae, "param_count": param_count_full}


# --------------------------------------------------------------------------- #
# per-expert label computation
# --------------------------------------------------------------------------- #
def compute_labels(full_mse, wo_mse):
    n = len(wo_mse)
    out = []
    for i in range(n):
        d = {s: wo_mse[i][s] - full_mse[s] for s in SEEDS}
        rg = {s: d[s] / wo_mse[i][s] if wo_mse[i][s] != 0 else float("nan") for s in SEEDS}
        dv = np.array([d[s] for s in SEEDS], float)
        rgv = np.array([rg[s] for s in SEEDS], float)
        n_pos = int((dv > 0).sum())
        n_neg = int((dv < 0).sum())
        if n_pos >= 2 and dv.mean() > 0:
            label = "POSITIVE"
        elif n_neg >= 2 and dv.mean() < 0:
            label = "NEGATIVE"
        else:
            label = "MIXED"
        out.append({
            "without_i_mse": {f"without_i_mse_seed{s}": wo_mse[i][s] for s in SEEDS},
            "delta": {f"delta_seed{s}": d[s] for s in SEEDS},
            "relative_gain": {f"relative_gain_seed{s}": rg[s] for s in SEEDS},
            "mean_delta": float(dv.mean()), "std_delta": float(dv.std()),
            "mean_relative_gain": float(np.nanmean(rgv)), "std_relative_gain": float(np.nanstd(rgv)),
            "n_positive_seeds": n_pos, "n_negative_seeds": n_neg,
            "contribution_label": label,
        })
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None, help="only run this dataset key")
    ap.add_argument("--limit-experts", type=int, default=None, help="limit LOO to first N experts (smoke test)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| strict deterministic ON", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_samples = []
    dataset_summaries = []
    global_manifest = []
    total_runs = 0
    t_start = time.time()

    for cfg in DATASETS:
        key = cfg["key"]
        if args.only and key != args.only:
            continue

        ds_out = OUT_ROOT / key
        ds_out.mkdir(parents=True, exist_ok=True)

        # 0. raw pool existence check
        if not Path(cfg["raw_pool"]).exists():
            print(f"[{key}] RAW_EXPERT_POOL_NOT_FOUND: {cfg['raw_pool']}", flush=True)
            (ds_out / "summary.json").write_text(
                json.dumps({"dataset": key, "status": "RAW_EXPERT_POOL_NOT_FOUND"}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            continue

        print(f"\n===== dataset: {key} =====", flush=True)
        t_ds = time.time()

        # A. load + audit + features
        data = load_and_prepare(cfg, device)
        n_raw = data["n_raw"]
        print(f"[{key}] raw experts = {n_raw}: {[e['expert_id'] for e in data['raw_experts']]}", flush=True)
        print(f"[{key}] bank={len(data['X_bank'])} fit={len(data['X_fit'])} val={len(data['X_val'])} (TEST not read)", flush=True)

        audit = run_audit(cfg, data)
        print(f"[{key}] n_repair_created_activation = {audit['n_repair_created_activation']}", flush=True)
        print(f"[{key}] v6 ACCEPT pool = {[data['raw_experts'][i]['expert_id'] for i in audit['final_pool']]}", flush=True)

        # safety check (extra training, only for triggered experts)
        safety, disabled = run_safety_check(cfg, data, audit, device)

        gen_feats = dataset_generic_features(data["series"], int(0.6 * len(data["series"])), cfg)
        raw_train_act = np.concatenate([data["raw_bank_act"], data["raw_fit_act"]], axis=0)

        # persist audit + features
        audit_json = []
        for i in range(n_raw):
            r = audit["rows"][i]
            audit_json.append({
                "expert_id": r["expert_id"], "name": r["name"],
                "pre_repair_execution_state": r["pre_state"],
                "post_repair_execution_state": r["post_state"],
                "repair_happened": r["repair_happened"],
                "post_repair_safety_triggered": r["triggered"],
                "post_repair_safety_decision": safety.get(i, {}).get("safety"),
                "v6_1_final_decision": ("ACCEPT" if (i in audit["keep"] and i not in disabled) else "REJECT"),
                "final_active": bool(i in audit["keep"] and i not in disabled),
                "observability_decision": "PASS" if r["observability"]["all_observable"] else "FAIL",
                "evidence_decision": "PASS" if r["ev_pass"] else "FAIL",
                "compiler_decision": r["compiled"]["status"],
                "compiler_repair": r["compiled"].get("repairs", []),
                "fidelity_decision": "PASS" if r["fid_pass"] else "FAIL",
                "generalization_decision": "PASS" if r["gen_pass"] else "FAIL",
                "redundancy_decision": "DUPLICATE_ACTIVATION" if r["redundant"] else None,
                "activation_coverage_decision": ("DEAD" if r["coverage"]["dead"]
                                                 else ("CONSTANT_LOW_INFORMATION" if r["coverage"]["constant"]
                                                       else ("ACCEPTED_LOCAL_SCOPE" if r["coverage"]["local_scope"] else "OK"))),
                "mechanism_family": r["lineage"]["mechanism_family"],
                "state_variable_family": r["lineage"]["state_variable_family"],
            })
        (ds_out / "audit.json").write_text(json.dumps(audit_json, ensure_ascii=False, indent=2), encoding="utf-8")
        (ds_out / "dataset_features.json").write_text(json.dumps(gen_feats, ensure_ascii=False, indent=2), encoding="utf-8")

        # B/C. training (FULL_RAW + LOO) with checkpoint
        tr = run_training(cfg, data, device, args.limit_experts)
        labels = compute_labels(tr["full_mse"], tr["wo_mse"])
        total_runs += len(SEEDS) + n_raw * len(SEEDS)

        # assemble per-expert samples
        for i in range(n_raw):
            r = audit["rows"][i]
            e = data["raw_experts"][i]
            act_f = activation_features(raw_train_act[:, i])
            lab = labels[i]
            full_vals = [tr["full_mse"][s] for s in SEEDS]
            row = {
                "dataset_name": key,
                "expert_id": e["expert_id"],
                "expert_index": i,
                "pool_size": n_raw,
                "expert_text": e.get("claim", ""),
                "name": e.get("name", ""),
                "observable_features": json.dumps(e.get("observable_features", [])),
                "mechanism_family": r["lineage"]["mechanism_family"],
            }
            row.update(gen_feats)
            row.update(act_f)
            row.update(audit_json[i])
            row["full_raw_mse_mean"] = float(np.mean(full_vals))
            row["full_raw_mse_std"] = float(np.std(full_vals))
            for s in SEEDS:
                row[f"without_i_mse_seed{s}"] = tr["wo_mse"][i][s]
                row[f"delta_seed{s}"] = lab["delta"][f"delta_seed{s}"]
                row[f"relative_gain_seed{s}"] = lab["relative_gain"][f"relative_gain_seed{s}"]
            row["mean_delta"] = lab["mean_delta"]
            row["std_delta"] = lab["std_delta"]
            row["mean_relative_gain"] = lab["mean_relative_gain"]
            row["std_relative_gain"] = lab["std_relative_gain"]
            row["n_positive_seeds"] = lab["n_positive_seeds"]
            row["n_negative_seeds"] = lab["n_negative_seeds"]
            row["contribution_label"] = lab["contribution_label"]
            row["other_active_expert_count"] = sum(1 for j in range(n_raw) if j != i and raw_train_act[:, j].max() > 0.0)
            all_samples.append(row)

        # per-dataset summary
        label_counts = {}
        for lab in labels:
            label_counts[lab["contribution_label"]] = label_counts.get(lab["contribution_label"], 0) + 1
        ds_summary = {
            "dataset": key, "n_experts": n_raw,
            "full_raw_mse": {str(s): tr["full_mse"][s] for s in SEEDS},
            "full_raw_mse_mean": float(np.mean([tr["full_mse"][s] for s in SEEDS])),
            "full_raw_mse_std": float(np.std([tr["full_mse"][s] for s in SEEDS])),
            "param_count": tr["param_count"],
            "n_repair_created_activation": audit["n_repair_created_activation"],
            "final_accepted_experts": [data["raw_experts"][i]["expert_id"] for i in audit["final_pool"]],
            "label_counts": label_counts,
            "runtime_seconds": round(time.time() - t_ds, 2),
        }
        dataset_summaries.append(ds_summary)
        (ds_out / "summary.json").write_text(json.dumps(ds_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{key}] done in {ds_summary['runtime_seconds']}s | labels={label_counts}", flush=True)

    # ---- global samples ----
    if all_samples:
        df = pd.DataFrame(all_samples)
        df.to_csv(OUT_ROOT / "knowledge_effect_samples.csv", index=False, encoding="utf-8")
        try:
            df.to_parquet(OUT_ROOT / "knowledge_effect_samples.parquet", index=False)
        except Exception as e:
            print("parquet write skipped:", e, flush=True)

        expert_summary = []
        for r in all_samples:
            expert_summary.append({k: r[k] for k in (
                "dataset_name", "expert_id", "name", "mechanism_family",
                "evidence_decision", "fidelity_decision", "generalization_decision",
                "redundancy_decision", "activation_coverage_decision", "v6_1_final_decision",
                "full_raw_mse_mean", "mean_delta", "std_delta", "mean_relative_gain",
                "n_positive_seeds", "n_negative_seeds", "contribution_label")})
        (OUT_ROOT / "expert_summary.json").write_text(json.dumps(expert_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT_ROOT / "dataset_summary.json").write_text(json.dumps(dataset_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- run_manifest.json ----
    for cfg in DATASETS:
        if args.only and cfg["key"] != args.only:
            continue
        runs_dir = OUT_ROOT / cfg["key"] / "runs"
        if runs_dir.exists():
            for f in sorted(runs_dir.glob("*.json")):
                try:
                    global_manifest.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(global_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report.md ----
    write_report(all_samples, dataset_summaries, global_manifest, total_runs, time.time() - t_start)


def write_report(samples, dataset_summaries, manifest, total_runs, total_runtime):
    L = ["## 【结果】", ""]

    n_ds = len(dataset_summaries)
    n_experts_total = len(samples)
    labels = [s["contribution_label"] for s in samples]
    n_pos = labels.count("POSITIVE")
    n_neg = labels.count("NEGATIVE")
    n_mix = labels.count("MIXED")

    L.append(f"1. 成功处理数据集数 = {n_ds}")
    L.append("2. 每个数据集 Raw Expert 数：")
    for ds in dataset_summaries:
        L.append(f"   - {ds['dataset']}: {ds['n_experts']}")
    L.append(f"3. 总知识样本数量 = {n_experts_total}")
    L.append(f"4. contribution_label 分布：POSITIVE={n_pos}, NEGATIVE={n_neg}, MIXED={n_mix}")
    L.append("")
    L.append("5. 各数据集 FULL_RAW mean MSE：")
    L.append("| Dataset | n Expert | Positive | Negative | Mixed | Full Raw MSE |")
    L.append("|---|---|---|---|---|---|")
    for ds in dataset_summaries:
        lc = ds["label_counts"]
        L.append(f"| {ds['dataset']} | {ds['n_experts']} | {lc.get('POSITIVE', 0)} | "
                 f"{lc.get('NEGATIVE', 0)} | {lc.get('MIXED', 0)} | {ds['full_raw_mse_mean']:.4f} |")
    L.append("")

    L.append("6. 各数据集 Expert contribution 分布 (mean_delta)：")
    L.append("| Dataset | min | median | max |")
    L.append("|---|---|---|---|")
    by_ds = {}
    for s in samples:
        by_ds.setdefault(s["dataset_name"], []).append(s["mean_delta"])
    for ds, vals in by_ds.items():
        a = np.asarray(vals, float)
        L.append(f"| {ds} | {a.min():.4f} | {np.median(a):.4f} | {a.max():.4f} |")
    L.append("")

    same_dir = sum(1 for s in samples if s["n_positive_seeds"] == 3 or s["n_negative_seeds"] == 3)
    conflict = sum(1 for s in samples if s["n_positive_seeds"] >= 1 and s["n_negative_seeds"] >= 1)
    L.append(f"7. 三 seed 方向完全一致的 Expert 数 = {same_dir}")
    L.append(f"8. seed 间方向冲突 (有正有负) 的 Expert 数 = {conflict}")
    L.append("")

    L.append("9. Critic v6.1 ACCEPT/REJECT × contribution_label 交叉表：")
    L.append("| Critic v6.1 decision | Actual Positive | Actual Negative | Mixed |")
    L.append("|---|---|---|---|")
    for decision in ("ACCEPT", "REJECT"):
        sub = [s for s in samples if s["v6_1_final_decision"] == decision]
        L.append(f"| {decision} | {sum(1 for s in sub if s['contribution_label']=='POSITIVE')} | "
                 f"{sum(1 for s in sub if s['contribution_label']=='NEGATIVE')} | "
                 f"{sum(1 for s in sub if s['contribution_label']=='MIXED')} |")
    L.append("")

    L.append(f"10. 总训练次数 = {len(manifest)} (FULL + LOO；safety check 另计)")
    L.append(f"11. 总运行时间 = {total_runtime:.1f} s ({total_runtime/3600:.2f} h)")
    L.append("12. TEST 是否完全未访问 = 是 (test_used = false)")
    L.append("")

    L.append("## 【结论】")
    L.append("")
    for line in (
        'project_stage = "learned_critic_dataset_generation"',
        "datasets_requested = 5",
        'critic_version = "v6.1"',
        "critic_frozen = true",
        "expert_regenerated = false",
        'knowledge_label_method = "deterministic_leave_one_out"',
        "learned_critic_trained = false",
        "dataset_specific_rules = false",
        "expert_specific_rules = false",
        "train_resplit = false",
        "cert_used = false",
        "seeds = [42,123,2026]",
        "test_used = false",
    ):
        L.append(line)

    (OUT_ROOT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
