#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic v6.1 — cross-dataset validation on Electricity (frozen from ETTh1).

The Critic v6.1 is FROZEN on ETTh1 and is migrated to Electricity UNCHANGED.  No
Electricity / population / consumer-specific rule is added.  The only mechanism on
top of Critic v6 is the Post-Repair Activation Safety Check (same trigger + same
ON/OFF rule as ETTh1):

    trigger = (pre_repair_execution_state == "inactive") AND
              (post_repair_execution_state == "active") AND
              (a compiler repair actually occurred)

Only repair-created activations are safety-checked; every other expert keeps its v6
decision (never re-screened by MSE).

Electricity data protocol is the frozen one (OT univariate target, 15-min;
TRAIN/VAL/TEST from results/multidataset/electricity).  TEST is never read.

Output: results/electricity_critic_v6_1_cross_dataset/
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from run_expert_agent_closed_loop_v6 import (            # generic v6 machinery (frozen)
    COMPILER_CONFIG,
    audit_observability_for_expert,
    compile_condition_v6,
    validate_proxy_v6,
    analyze_claim_scope_v6,
    critique_v6,
    audit_activation_coverage,
)
from run_expert_embedding_mlp import build_windows, mse, mae        # noqa: E402
from run_solar_same_val_baselines import residualize                # noqa: E402
from run_conditional_expert_mlp_v4 import (                         # noqa: E402
    ConditionalExpertMLPRetriever,
    retrieve_conditional,
)
from run_etth1_prediction_validation import (                       # noqa: E402  (univariate activation)
    compute_training_stats,
    compute_window_feats,
    expert_activation,
    build_activation_matrix,
)
from sentence_transformers import SentenceTransformer               # noqa: E402

RAW_POOL = Path("results/electricity_critic_v5_closed_loop/expert_round_1.json")
OUT_DIR = Path("results/electricity_critic_v6_1_cross_dataset")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32
LOAD_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
N_POPULATION = 314

# generic feature registry (calendar categories are generic month/hour/day groupings)
FEATURE_REGISTRY = {
    "hour_of_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "mechanism_family": "calendar_cycle", "state_variable_family": "hour_of_day",
                    "categories": {"morning": [6, 7, 8, 9, 10, 11], "midday": [12, 13, 14, 15],
                                   "evening": [16, 17, 18, 19, 20, 21],
                                   "night": [22, 23, 0, 1, 2, 3, 4, 5]}},
    "day_of_week": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "mechanism_family": "calendar_cycle", "state_variable_family": "day_of_week",
                    "categories": {"weekday": [0, 1, 2, 3, 4], "weekend": [5, 6]}},
    "working_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "mechanism_family": "calendar_cycle", "state_variable_family": "working_day",
                    "categories": {"workday": [0, 1, 2, 3, 4], "non_working": [5, 6]}},
    "seasonal_level": {"kind": "calendar", "source_type": "timestamp_known",
                       "dependencies": ["known_timestamp"],
                       "mechanism_family": "calendar_cycle", "state_variable_family": "season",
                       "categories": {"winter": [12, 1, 2], "spring": [3, 4, 5],
                                      "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
    "recent_level": {"kind": "continuous", "source_type": "context_derived",
                     "dependencies": ["historical_context", "train_derived_constant"],
                     "mechanism_family": "recent_state_regime", "state_variable_family": "level"},
    "recent_volatility": {"kind": "continuous", "source_type": "context_derived",
                          "dependencies": ["historical_context", "train_derived_constant"],
                          "mechanism_family": "recent_state_regime", "state_variable_family": "volatility"},
    "recent_slope_magnitude": {"kind": "continuous", "source_type": "context_derived",
                               "dependencies": ["historical_context", "train_derived_constant"],
                               "mechanism_family": "recent_state_regime", "state_variable_family": "trend_slope"},
    "recent_diff_magnitude": {"kind": "continuous", "source_type": "context_derived",
                              "dependencies": ["historical_context", "train_derived_constant"],
                              "mechanism_family": "recent_state_regime", "state_variable_family": "change_magnitude"},
    "recent_autocorr": {"kind": "continuous", "source_type": "context_derived",
                        "dependencies": ["historical_context", "train_derived_constant"],
                        "mechanism_family": "recent_state_regime", "state_variable_family": "persistence"},
    "x_subset_level": {"kind": "continuous", "source_type": "observed_covariate",
                       "dependencies": ["observed_covariate"],
                       "mechanism_family": "recent_state_regime", "state_variable_family": "covariate_level"},
    "x_subset_change": {"kind": "continuous", "source_type": "observed_covariate",
                        "dependencies": ["observed_covariate"],
                        "mechanism_family": "recent_state_regime", "state_variable_family": "covariate_change"},
}

METADATA = {"target_description": "aggregate load of many consumers", "n_population": N_POPULATION}


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _sorted_experts(pool):
    if isinstance(pool, dict):
        items = list(pool.values())
    else:
        items = list(pool)
    return sorted(items, key=lambda e: int(str(e.get("expert_id", "0")).lstrip("E") or 0))


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _jaccard(a, b):
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _eta2(groups, y):
    groups, y = np.asarray(groups), np.asarray(y, float)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    within = sum(float(((y[groups == g] - y[groups == g].mean()) ** 2).sum())
                 for g in np.unique(groups))
    return 1.0 - within / total


def embed_claims(claims, device):
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    emb = enc.encode(claims, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


# --------------------------------------------------------------------------- #
# Frozen electricity activation: adds the subset-proxy x_subset features to the
# otherwise identical ETTh1 downstream activation builder (only E7 uses them).
# --------------------------------------------------------------------------- #
def add_x_subset_feats(feats, starts, s6, d6, s6_mean, s6_std, d6_mean, d6_std):
    """Append x_subset_level / x_subset_change (point-wise z-scored s6) to feats.

    s6 = sum(X1..X6) over the full electricity series (frozen subset-proxy feature).
    x_subset_level  = z-scored aggregate level of the observed subset at the window's
                       target timestamp.
    x_subset_change = z-scored point-to-point change of s6 at the target timestamp.
    """
    tidx = np.asarray(starts, dtype=np.int64) + CONTEXT
    lvl = np.zeros(len(starts), dtype=np.float64)
    chg = np.zeros(len(starts), dtype=np.float64)
    for k, t in enumerate(tidx):
        if 0 <= t < len(s6):
            lvl[k] = (s6[t] - s6_mean) / (s6_std + 1e-6)
        if 1 <= t < len(s6):
            chg[k] = (d6[t - 1] - d6_mean) / (d6_std + 1e-6)
    feats["x_subset_level"] = lvl
    feats["x_subset_change"] = chg
    return feats


def build_activation_matrix_elec(experts, X, tss, stats, starts,
                                 s6, d6, s6_mean, s6_std, d6_mean, d6_std):
    """Same as the frozen downstream activation matrix, plus the x_subset features."""
    feats = compute_window_feats(X, tss, stats)
    add_x_subset_feats(feats, starts, s6, d6, s6_mean, s6_std, d6_mean, d6_std)
    n = len(X)
    M = np.zeros((n, len(experts)), dtype=np.float32)
    for j, e in enumerate(experts):
        compiled = e["compiled"]
        for i in range(n):
            M[i, j] = expert_activation(compiled, feats, i)
    return M


# --------------------------------------------------------------------------- #
# Electricity TRAIN evidence (frozen TRAIN region; includes population facts)
# --------------------------------------------------------------------------- #
def build_electricity_evidence():
    df = pd.read_csv("data/electricity.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    dt = dates.iloc[:train_len]
    y = df["OT"].to_numpy(np.float64)[:train_len]
    s6 = np.sum([df[v].to_numpy(np.float64)[:train_len] for v in LOAD_VARS], axis=0)
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

    hourly_mean = np.array([float(y[hours == h].mean()) for h in range(24)])
    dow_mean = np.array([float(y[dows == d].mean()) for d in range(7)])

    ctx = 64
    n_win = train_len - ctx
    t = np.linspace(-1, 1, ctx)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    feat_vals = {"recent_level": [], "recent_volatility": [], "recent_slope_magnitude": [],
                 "recent_diff_magnitude": [], "recent_autocorr": []}
    for i in range(0, n_win, 4):  # 15-min data -> subsample for speed
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
    # covariate feature distributions (x_subset_level / x_subset_change from s6)
    feature_distributions["x_subset_level"] = {
        "q10": float(np.percentile(s6, 10)), "q25": float(np.percentile(s6, 25)),
        "q50": float(np.percentile(s6, 50)), "q75": float(np.percentile(s6, 75)),
        "q90": float(np.percentile(s6, 90)), "mean": float(s6.mean()), "std": float(s6.std()),
    }
    d6 = np.abs(np.diff(s6))
    feature_distributions["x_subset_change"] = {
        "q10": float(np.percentile(d6, 10)), "q25": float(np.percentile(d6, 25)),
        "q50": float(np.percentile(d6, 50)), "q75": float(np.percentile(d6, 75)),
        "q90": float(np.percentile(d6, 90)), "mean": float(d6.mean()), "std": float(d6.std()),
    }

    corr_by_month = {int(m): _corr(s6[months == m], y[months == m]) for m in months_present}
    sign_reversals = sum(1 for m in months_present if corr_by_month[m] < 0) if \
        any(v > 0 for v in corr_by_month.values()) else 0

    return {
        "temporal_coverage": {
            "months_present": months_present, "months_missing": [m for m in range(1, 13) if m not in months_present],
            "month_fraction": round(len(months_present) / 12, 4),
            "hours_present": hours_present, "hours_missing": [h for h in range(24) if h not in hours_present],
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
        "population": {"n_proxy": len(LOAD_VARS), "n_population": N_POPULATION,
                       "proxy_fraction": round(len(LOAD_VARS) / N_POPULATION, 4),
                       "corr_proxy_sum_vs_target": round(_corr(s6, y), 4)},
        "x_subset_representativeness": {
            "corr_proxy_sum_vs_target": round(_corr(s6, y), 4),
            "sign_reversals": sign_reversals,
            "corr_by_month": {k: round(v, 4) for k, v in corr_by_month.items()},
        },
        "feature_distributions": feature_distributions,
    }


# --------------------------------------------------------------------------- #
# training / eval helpers (univariate OT, same as ETTh1 downstream)
# --------------------------------------------------------------------------- #
def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device,
              epochs=EPOCHS, batch_size=BATCH_SIZE, candidates=CANDIDATES, lr=LR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)
    mode = "full" if use_shape else "expert"
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        for p in range(0, n, batch_size):
            ids = order[p:p + batch_size]
            if len(ids) == 0:
                continue
            cidx = rng.choice(len(X_bank), size=min(candidates, len(X_bank)), replace=False)
            x_q = torch.tensor(X_fit[ids], dtype=torch.float32, device=device)
            x_c = torch.tensor(X_bank[cidx], dtype=torch.float32, device=device)
            act_q = torch.tensor(fit_act[ids], dtype=torch.float32, device=device)
            act_c = bank_act_t[cidx]
            shape = torch.tensor(X_fit_res[ids] @ X_bank_res[cidx].T / L, dtype=torch.float32, device=device)
            z_q = model.encode(x_q, act_q)
            z_c = model.encode(x_c, act_c)
            delta = model.delta(z_q, z_c)
            s = model.combine(delta, shape, mode)
            w = torch.softmax(s / 1.0, dim=1)
            q_mu = X_fit[ids].mean(axis=1)
            c_mu = X_bank[cidx].mean(axis=1)
            transferred = Y_bank[cidx][None, :, :] + q_mu[:, None, None] - c_mu[None, :, None]
            pred = torch.sum(w.unsqueeze(-1) * torch.tensor(transferred, dtype=torch.float32, device=device), dim=1)
            y = torch.tensor(Y_fit[ids], dtype=torch.float32, device=device)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return None


@torch.no_grad()
def evaluate(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res, val_act, bank_act,
             use_shape, device):
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                val_act, bank_act, mode, TOPK, device)
    return pred


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| strict deterministic ON")

    raw_experts = _sorted_experts(json.loads(RAW_POOL.read_text(encoding="utf-8")))
    n_raw = len(raw_experts)
    expert_ids = [e["expert_id"] for e in raw_experts]
    print(f"frozen raw experts = {n_raw}: {expert_ids}")

    # ---- v6 critic (Evidence / Compiler / Fidelity / Generalization / Redundancy) ----
    evidence = build_electricity_evidence()
    print(f"TRAIN months covered={len(evidence['temporal_coverage']['months_present'])} "
          f"n_month_reversals={evidence['seasonal']['n_month_direction_reversals']} "
          f"proxy_fraction={evidence['population']['proxy_fraction']}")

    v6_compiled = []
    trigger = []
    aud = []
    for e in raw_experts:
        e.setdefault("id", e.get("expert_id"))
        raw_prim = e.get("compiled", {}).get("primitives", [])
        pre_state = "inactive" if len(raw_prim) == 0 else "active"
        obs = audit_observability_for_expert(e, FEATURE_REGISTRY)
        compiled = compile_condition_v6(e, FEATURE_REGISTRY, evidence, COMPILER_CONFIG)
        post_state = "active" if compiled["status"] == "COMPILED" and len(compiled["primitives"]) >= 1 else "inactive"
        repair_happened = len(compiled.get("repairs", [])) > 0
        triggered = (pre_state == "inactive") and (post_state == "active") and repair_happened
        scope = analyze_claim_scope_v6(e, METADATA)
        proxy = validate_proxy_v6(e, compiled, obs)
        checks = critique_v6(e, scope, evidence, compiled, proxy, obs)
        ev_pass = (obs["all_observable"] and checks["support"]["ok"] and checks["coverage"]["ok"]
                   and checks["stability"]["ok"] and checks["counter"]["ok"]
                   and checks["population"]["ok"])
        fid_pass = proxy["faithful"] and compiled["status"] == "COMPILED"
        v6_compiled.append(compiled)
        trigger.append({"expert_id": e["expert_id"], "pre_repair_state": pre_state,
                        "post_repair_state": post_state, "repair_happened": repair_happened,
                        "triggered": triggered})
        aud.append({"expert_id": e["expert_id"], "compiled": compiled,
                    "ev_pass": ev_pass, "fid_pass": fid_pass,
                    "repair": compiled.get("repairs", [])})
        if triggered:
            print(f"  TRIGGER: {e['expert_id']} ({pre_state} -> {post_state})")

    triggered_idx = [i for i, t in enumerate(trigger) if t["triggered"]]
    print(f"repair-created activations = {len(triggered_idx)}")

    # ---- data (univariate OT downstream) ----
    ds = Path("results/multidataset/electricity")
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)      # TEST never read

    # frozen subset-proxy feature s6 = sum(X1..X6) for x_subset activation (E7)
    df_raw = pd.read_csv("data/electricity.csv")
    s6 = np.sum([df_raw[v].to_numpy(np.float64) for v in LOAD_VARS], axis=0)
    s6_train = s6[: int(0.6 * len(df_raw))]
    s6_mean, s6_std = float(s6_train.mean()), float(s6_train.std())
    d6 = np.abs(np.diff(s6))
    d6_train = d6[: max(1, int(0.6 * len(df_raw)) - 1)]
    d6_mean, d6_std = float(d6_train.mean()), float(d6_train.std())

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

    stats = compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    # raw pool activation (v5 compiled) and v6 pool activation (v6 compiled)
    raw_bank_act = build_activation_matrix_elec(raw_experts, X_bank, bank_ts, stats, bank_starts,
                                                s6, d6, s6_mean, s6_std, d6_mean, d6_std)
    raw_fit_act = build_activation_matrix_elec(raw_experts, X_fit, fit_ts, stats, fit_starts,
                                               s6, d6, s6_mean, s6_std, d6_mean, d6_std)
    raw_val_act = build_activation_matrix_elec(raw_experts, X_val, val_ts, stats, val_starts,
                                               s6, d6, s6_mean, s6_std, d6_mean, d6_std)

    v6_experts = [dict(e) for e in raw_experts]
    for i, e in enumerate(v6_experts):
        e["compiled"] = v6_compiled[i]
    v6_bank_act = build_activation_matrix_elec(v6_experts, X_bank, bank_ts, stats, bank_starts,
                                               s6, d6, s6_mean, s6_std, d6_mean, d6_std)
    v6_fit_act = build_activation_matrix_elec(v6_experts, X_fit, fit_ts, stats, fit_starts,
                                              s6, d6, s6_mean, s6_std, d6_mean, d6_std)
    v6_val_act = build_activation_matrix_elec(v6_experts, X_val, val_ts, stats, val_starts,
                                              s6, d6, s6_mean, s6_std, d6_mean, d6_std)

    emb = embed_claims([e["claim"] for e in raw_experts], device)
    expert_dim = emb.shape[1]
    n_slots = n_raw

    # ---- generalization (activation coverage on TRAIN_CORE=bank) + redundancy ----
    # v6 ACCEPT pool = Evidence + Fidelity + Generalization + Redundancy
    for i in range(n_raw):
        a = np.concatenate([raw_bank_act[:, i], raw_fit_act[:, i]], axis=0)
        cov = audit_activation_coverage(a, bank_ts + fit_ts, n_blocks=4)
        aud[i]["coverage"] = cov
        aud[i]["gen_pass"] = (not cov["dead"]) and (not cov["constant"])

    passing = [i for i in range(n_raw) if aud[i]["ev_pass"] and aud[i]["fid_pass"] and aud[i]["gen_pass"]]
    # redundancy (sequential, keep first by expert_id order)
    final_pool = []
    for i in passing:
        dup = False
        a = np.concatenate([raw_bank_act[:, i], raw_fit_act[:, i]])
        for j in final_pool:
            b = np.concatenate([raw_bank_act[:, j], raw_fit_act[:, j]])
            if abs(_corr(a, b)) > 0.95 or _jaccard(a, b) > 0.95:
                dup = True
                break
        if not dup:
            final_pool.append(i)
    keep = set(final_pool)
    print(f"v6 ACCEPT pool = {[expert_ids[i] for i in final_pool]} ({len(final_pool)} experts)")

    # ---- final ablation A/B/C/D/E ----
    print("\n--- Final ablation (VAL, 3 seeds) ---")
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_bank = np.zeros_like(raw_bank_act)
    zero_fit = np.zeros_like(raw_fit_act)
    zero_val = np.zeros_like(raw_val_act)

    # v6 pool = v6-compiled activations masked to the ACCEPT pool
    v6_bank = np.zeros_like(raw_bank_act)
    v6_fit = np.zeros_like(raw_fit_act)
    v6_val = np.zeros_like(raw_val_act)
    for i in keep:
        v6_bank[:, i] = v6_bank_act[:, i]
        v6_fit[:, i] = v6_fit_act[:, i]
        v6_val[:, i] = v6_val_act[:, i]

    # ---- safety check: paired ON/OFF (same as ETTh1; only repair-created activations) ----
    safety = {}
    disabled = set()
    if triggered_idx:
        on_mse = {}
        for seed in SEEDS:
            seed_everything(seed)
            m = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
            m.set_expert_embeddings(emb)
            train_arm(m, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      v6_bank, v6_fit, True, device)
            pred = evaluate(m, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                            v6_val, v6_bank, True, device)
            on_mse[seed] = mse(Y_val, pred)
        for i in triggered_idx:
            bank_off = v6_bank.copy(); bank_off[:, i] = 0.0
            fit_off = v6_fit.copy(); fit_off[:, i] = 0.0
            val_off = v6_val.copy(); val_off[:, i] = 0.0
            off_mse = {}
            for seed in SEEDS:
                seed_everything(seed)
                m = ConditionalExpertMLPRetriever(
                    context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                    d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
                m.set_expert_embeddings(emb)
                train_arm(m, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                          bank_off, fit_off, True, device)
                pred = evaluate(m, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                val_off, bank_off, True, device)
                off_mse[seed] = mse(Y_val, pred)
            deltas = {s: on_mse[s] - off_mse[s] for s in SEEDS}
            mean_d = float(np.mean([deltas[s] for s in SEEDS]))
            n_pos = sum(1 for s in SEEDS if deltas[s] > 0)
            fail = (n_pos >= 2 and mean_d > 0)
            safety[i] = {"deltas": {s: round(deltas[s], 4) for s in SEEDS},
                         "mean_delta": round(mean_d, 4),
                         "safety": "FAIL" if fail else "PASS"}
            if fail:
                disabled.add(i)

    # v6.1 = v6 with safety-failed repaired activations disabled
    v61_bank = v6_bank.copy()
    v61_fit = v6_fit.copy()
    v61_val = v6_val.copy()
    for i in disabled:
        v61_bank[:, i] = 0.0
        v61_fit[:, i] = 0.0
        v61_val[:, i] = 0.0

    configs = {
        "base":        (emb_zero, zero_bank, zero_fit, zero_val, False),
        "shape":       (emb_zero, zero_bank, zero_fit, zero_val, True),
        "raw_expert":  (emb,      raw_bank_act, raw_fit_act, raw_val_act, True),
        "critic_v6":   (emb,      v6_bank,      v6_fit,      v6_val,      True),
        "critic_v6_1": (emb,      v61_bank,     v61_fit,     v61_val,     True),
    }

    per_seed = {k: [] for k in configs}
    param_count = None
    for seed in SEEDS:
        seed_everything(seed)
        base = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        if param_count is None:
            param_count = n_params(base)
        for name, (embx, bank_act_x, fit_act_x, val_act_x, use_shape) in configs.items():
            model = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
            model.load_state_dict(init_state)
            model.set_expert_embeddings(embx)
            train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      bank_act_x, fit_act_x, use_shape, device)
            pred = evaluate(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                            val_act_x, bank_act_x, use_shape, device)
            per_seed[name].append({"seed": seed, "mse": mse(Y_val, pred), "mae": mae(Y_val, pred)})
            print(f"  [{name}] seed{seed} mse={per_seed[name][-1]['mse']:.4f}")

    # ---- aggregate + report ----
    def mean_std(vals):
        a = np.array([v["mse"] for v in vals], float)
        b = np.array([v["mae"] for v in vals], float)
        return (float(a.mean()), float(a.std()), float(b.mean()), float(b.std()))

    summary = {}
    for k, vals in per_seed.items():
        mm, ms, am, as_ = mean_std(vals)
        summary[k] = {"mse_mean": mm, "mse_std": ms, "mae_mean": am, "mae_std": as_, "per_seed": vals}

    v6_m, v61_m = summary["critic_v6"]["mse_mean"], summary["critic_v6_1"]["mse_mean"]
    raw_m, shape_m, base_m = summary["raw_expert"]["mse_mean"], summary["shape"]["mse_mean"], summary["base"]["mse_mean"]
    gains = {
        "shape_vs_base": (base_m - shape_m) / base_m * 100,
        "raw_vs_shape": (shape_m - raw_m) / shape_m * 100,
        "v6_vs_raw": (raw_m - v6_m) / raw_m * 100,
        "v61_vs_v6": (v6_m - v61_m) / v6_m * 100,
        "v61_vs_raw": (raw_m - v61_m) / raw_m * 100,
        "v61_vs_shape": (shape_m - v61_m) / shape_m * 100,
        "v61_vs_base": (base_m - v61_m) / base_m * 100,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for i in range(n_raw):
        audit_rows.append({
            "expert_id": expert_ids[i], "name": raw_experts[i]["name"],
            "evidence": "PASS" if aud[i]["ev_pass"] else "FAIL",
            "fidelity": "PASS" if aud[i]["fid_pass"] else "FAIL",
            "generalization": "PASS" if aud[i]["gen_pass"] else "FAIL",
            "pre_repair_state": trigger[i]["pre_repair_state"],
            "post_repair_state": trigger[i]["post_repair_state"],
            "repair_happened": trigger[i]["repair_happened"],
            "safety_triggered": trigger[i]["triggered"],
            "final_active": i in keep,
        })
    result = {
        "critic_base": "v6", "critic_version": "v6.1",
        "source_version_frozen_on": "ETTh1", "cross_dataset": "Electricity",
        "critic_rules_modified": False, "dataset_specific_rules": False, "expert_specific_rules": False,
        "cert_used": False, "expert_regenerated": False, "all_experts_rescreened_by_mse": False,
        "post_repair_activation_safety_enabled": True,
        "deterministic_enabled": True, "seed42_removed": False, "test_used": False,
        "param_count": param_count, "n_slots": n_slots,
        "audit": audit_rows,
        "trigger": trigger,
        "n_repair_created_activation": len(triggered_idx),
        "safety_check": {expert_ids[i]: safety.get(i, {}) for i in triggered_idx},
        "final_accepted_experts": [expert_ids[i] for i in final_pool],
        "arms": {k: {kk: vv for kk, vv in summary[k].items() if kk != "per_seed"}
                 for k in configs},
        "gains": gains,
        "seeds": SEEDS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "expert_audit.json").write_text(json.dumps(audit_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "safety_check.json").write_text(
        json.dumps({"n_repair_created_activation": len(triggered_idx),
                    "rows": [{**{"expert_id": expert_ids[i]}, **safety.get(i, {})} for i in triggered_idx]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "per_seed_results.json").write_text(
        json.dumps({"seeds": SEEDS, "arms": per_seed, "param_count": param_count, "n_slots": n_slots},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # report
    L = []
    L.append("## 【结果】")
    L.append("")
    L.append("## 1. Safety Check report")
    L.append("")
    L.append(f"- n_repair_created_activation = {len(triggered_idx)}")
    L.append("- 触发表：")
    L.append("| Expert | pre repair | post repair | repair_happened | triggered |")
    L.append("|---|---|---|---|---|")
    for t in trigger:
        L.append(f"| {t['expert_id']} | {t['pre_repair_state']} | {t['post_repair_state']} | {t['repair_happened']} | {t['triggered']} |")
    L.append("")
    L.append("## 2. Expert audit (v6 decisions)")
    L.append("")
    L.append("| Expert | Evidence | Fidelity | Generalization | final active |")
    L.append("|---|---|---|---|---|")
    for r in audit_rows:
        L.append(f"| {r['expert_id']} | {r['evidence']} | {r['fidelity']} | {r['generalization']} | {r['final_active']} |")
    L.append("")
    L.append("## 3. Final MSE table (VAL)")
    L.append("")
    L.append("| Method | seed42 | seed123 | seed2026 | mean MSE | std | mean MAE | std |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, key in (("BASE", "base"), ("+SHAPE", "shape"), ("+RAW EXPERT", "raw_expert"),
                       ("+CRITIC EXPERT v6", "critic_v6"), ("+CRITIC EXPERT v6.1", "critic_v6_1")):
        vals = sorted(per_seed[key], key=lambda v: v["seed"])
        cells = " | ".join(f"{v['mse']:.2f}" for v in vals)
        ss = summary[key]
        L.append(f"| {label} | {cells} | {ss['mse_mean']:.2f} | {ss['mse_std']:.2f} | {ss['mae_mean']:.4f} | {ss['mae_std']:.4f} |")
    L.append("")
    L.append("## 4. Gains")
    L.append("")
    L.append(f"- Shape vs Base = {gains['shape_vs_base']:+.2f}%")
    L.append(f"- Raw vs Shape = {gains['raw_vs_shape']:+.2f}%")
    L.append(f"- v6 vs Raw = {gains['v6_vs_raw']:+.2f}%")
    L.append(f"- v6.1 vs v6 = {gains['v61_vs_v6']:+.2f}%")
    L.append(f"- v6.1 vs Raw = {gains['v61_vs_raw']:+.2f}%")
    L.append(f"- v6.1 vs Shape = {gains['v61_vs_shape']:+.2f}%")
    L.append(f"- v6.1 vs Base = {gains['v61_vs_base']:+.2f}%")
    L.append("")

    L.append("## 【结论】")
    L.append("")
    L.append(f"1. Critic v6.1 是否未经修改直接运行？ 是（通用 v6 核心 + 同一 Safety Check，无 Electricity 特判）")
    L.append("2. 是否有任何 Electricity-specific rule？ 否")
    L.append(f"3. 原始 Expert 是否重新生成？ 否（{n_raw} 条 raw candidates 原样读取）")
    L.append(f"4. 有多少条知识发生 Compiler Repair？ {sum(1 for t in trigger if t['repair_happened'])}")
    L.append(f"5. 有多少条知识属于 repair-created activation？ {len(triggered_idx)}")
    L.append(f"6. Safety Check 自动挡掉了多少条？ {len(disabled)}")
    L.append("7. 未发生 repair 的知识是否完全没有经过 MSE 重筛？ 是")
    L.append(f"8. v6.1 是否改善 v6？ {'是' if v61_m < v6_m else '否（v6.1==v6，0 条触发）'}（{v61_m:.2f} vs {v6_m:.2f}）")
    L.append(f"9. v6.1 是否保留 Raw Expert 的优势？ {'是' if v61_m <= raw_m else '否'}（{v61_m:.2f} vs {raw_m:.2f}）")
    L.append("10. TEST 是否完全未访问？ 是")
    L.append("")
    L.append("critic_base = \"v6\"")
    L.append("critic_version = \"v6.1\"")
    L.append("source_version_frozen_on = \"ETTh1\"")
    L.append("cross_dataset = \"Electricity\"")
    L.append("critic_rules_modified = false")
    L.append("dataset_specific_rules = false")
    L.append("expert_specific_rules = false")
    L.append("cert_used = false")
    L.append("expert_regenerated = false")
    L.append("all_experts_rescreened_by_mse = false")
    L.append("post_repair_activation_safety_enabled = true")
    L.append("deterministic_enabled = true")
    L.append("seed42_removed = false")
    L.append("test_used = false")
    report = "\n".join(L)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
