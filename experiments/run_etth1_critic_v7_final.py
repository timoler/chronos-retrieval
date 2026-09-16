#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic Critic v7 (final) — run end-to-end on ETTh1.

Critic v7 is fully dataset-agnostic: it never references ETTh1 / OT / any domain
phenomenon in its DECISION LOGIC.  It evaluates the frozen Round-1 Raw Expert
candidates through five abstract principles only:

    1. Evidence Sufficiency    (is the claim supported by TRAIN_CORE?)
    2. Execution Fidelity      (is the compiled proxy faithful to the claim?)
    3. Generalization Stability (is the activation stable across TRAIN_CORE blocks?)
    4. Predictive Value        (does the expert add incremental CERT value?)
    5. Redundancy              (is the activation duplicated by another accepted expert?)

Data protocol (no leakage):
    - TRAIN region split chronologically: TRAIN_CORE (first 80%) + CERT (last 20%).
    - Evidence / Compiler / Fidelity / Generalization / Redundancy use TRAIN_CORE only.
    - Predictive Value trains on TRAIN_CORE and evaluates on CERT only.
    - Final VAL is touched ONLY for the final A/B/C/D MSE ablation, AFTER the pool is
      frozen.  TEST is never read.

The Expert pool is the frozen Round-1 candidates (E1..E10); experts are NOT
regenerated, and no expert (incl. E9) is special-cased.  Predictive Value uses
strict incremental leave-one-out (WITHOUT Ei vs WITH Ei) over 3 deterministic seeds.

Output: results/etth1_critic_v7_final/
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

from run_expert_agent_closed_loop_v6 import (            # generic v6 machinery reused
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
from run_etth1_prediction_validation import (                       # noqa: E402
    compute_training_stats,
    build_activation_matrix,
)
from sentence_transformers import SentenceTransformer               # noqa: E402

RAW_POOL = Path("results/etth1_critic_v5_closed_loop/expert_round_1.json")
OUT_DIR = Path("results/etth1_critic_v7_final")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

# Generic feature registry (source metadata + mechanism family).  The calendar
# category labels are generic month/hour/day groupings (NOT dataset-specific rules).
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
}

METADATA = {"target_description": "univariate target series", "n_population": 1}


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
# TRAIN_CORE evidence (generic statistics over the first 80% of TRAIN rows)
# --------------------------------------------------------------------------- #
def build_train_core_evidence():
    df = pd.read_csv("data/ETTh1.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    core_len = int(0.8 * train_len)
    dt = dates.iloc[:core_len]
    y = df["OT"].to_numpy(np.float64)[:core_len]
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
    n_win = core_len - ctx
    t = np.linspace(-1, 1, ctx)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    feat_vals = {"recent_level": [], "recent_volatility": [], "recent_slope_magnitude": [],
                 "recent_diff_magnitude": [], "recent_autocorr": []}
    for i in range(n_win):
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
        "train_core_range": [str(dt.min()), str(dt.max())],
        "core_len": core_len, "train_len": train_len, "total_rows": N,
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
        "population": {"n_proxy": 1, "n_population": 1, "proxy_fraction": 1.0,
                       "corr_proxy_sum_vs_target": 1.0},
        "x_subset_representativeness": {},
        "feature_distributions": feature_distributions,
    }


# --------------------------------------------------------------------------- #
# training / eval helpers (deterministic)
# --------------------------------------------------------------------------- #
def train_arm(model, X_mem, Y_mem, X_q, Y_q, X_mem_res, X_q_res, mem_act, q_act,
              use_shape, device, epochs=EPOCHS, batch_size=BATCH_SIZE, candidates=CANDIDATES, lr=LR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_mem.shape[1]
    Yb = torch.tensor(Y_mem, dtype=torch.float32, device=device)
    mem_act_t = torch.tensor(mem_act, dtype=torch.float32, device=device)
    n = len(X_q)
    rng = np.random.default_rng(12345)
    mode = "full" if use_shape else "expert"
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        for p in range(0, n, batch_size):
            ids = order[p:p + batch_size]
            if len(ids) == 0:
                continue
            cidx = rng.choice(len(X_mem), size=min(candidates, len(X_mem)), replace=False)
            x_q = torch.tensor(X_q[ids], dtype=torch.float32, device=device)
            x_c = torch.tensor(X_mem[cidx], dtype=torch.float32, device=device)
            act_q = torch.tensor(q_act[ids], dtype=torch.float32, device=device)
            act_c = mem_act_t[cidx]
            shape = torch.tensor(X_q_res[ids] @ X_mem_res[cidx].T / L, dtype=torch.float32, device=device)
            z_q = model.encode(x_q, act_q)
            z_c = model.encode(x_c, act_c)
            delta = model.delta(z_q, z_c)
            s = model.combine(delta, shape, mode)
            w = torch.softmax(s / 1.0, dim=1)
            q_mu = X_q[ids].mean(axis=1)
            c_mu = X_mem[cidx].mean(axis=1)
            transferred = Y_mem[cidx][None, :, :] + q_mu[:, None, None] - c_mu[None, :, None]
            pred = torch.sum(w.unsqueeze(-1) * torch.tensor(transferred, dtype=torch.float32, device=device), dim=1)
            y = torch.tensor(Y_q[ids], dtype=torch.float32, device=device)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return None


@torch.no_grad()
def evaluate(model, X_eval, X_mem, Y_mem, X_eval_res, X_mem_res, eval_act, mem_act,
             use_shape, device):
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(model, X_eval, X_mem, Y_mem, X_eval_res, X_mem_res,
                                eval_act, mem_act, mode, TOPK, device)
    return pred


# --------------------------------------------------------------------------- #
# one predictive-value training run (WITH all experts, or WITHOUT a single one)
# --------------------------------------------------------------------------- #
def run_pv(seed, emb, zero_slot, X_mem, Y_mem, X_q, Y_q, X_mem_res, X_q_res,
           mem_act, q_act, X_cert, Y_cert, X_cert_res, cert_act, device):
    seed_everything(seed)
    if zero_slot is not None:
        mem_act = mem_act.copy(); mem_act[:, zero_slot] = 0.0
        q_act = q_act.copy(); q_act[:, zero_slot] = 0.0
        cert_act = cert_act.copy(); cert_act[:, zero_slot] = 0.0
    model = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=emb.shape[1], n_experts=emb.shape[0],
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    model.set_expert_embeddings(emb)
    train_arm(model, X_mem, Y_mem, X_q, Y_q, X_mem_res, X_q_res, mem_act, q_act, True, device)
    pred = evaluate(model, X_cert, X_mem, Y_mem, X_cert_res, X_mem_res, cert_act, mem_act, True, device)
    return mse(Y_cert, pred)


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

    # ---- data ----
    ds = Path("results/multidataset/ETTh1")
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

    cut = max(1, int(len(train_starts) * 0.80))       # TRAIN_CORE / CERT boundary
    core_starts = train_starts[:cut]
    cert_starts = train_starts[cut:]

    cut2 = max(1, int(len(core_starts) * 0.80))       # PV memory / PV-train-query boundary
    mem_starts = core_starts[:cut2]
    trainq_starts = core_starts[cut2:]

    X_mem, Y_mem = build_windows(series, mem_starts, CONTEXT, HORIZON)
    X_q, Y_q = build_windows(series, trainq_starts, CONTEXT, HORIZON)
    X_cert, Y_cert = build_windows(series, cert_starts, CONTEXT, HORIZON)
    X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)
    X_mem_res = residualize(X_mem)
    X_q_res = residualize(X_q)
    X_cert_res = residualize(X_cert)
    X_val_res = residualize(X_val)

    mem_ts = [dates[s + CONTEXT] for s in mem_starts]
    trainq_ts = [dates[s + CONTEXT] for s in trainq_starts]
    cert_ts = [dates[s + CONTEXT] for s in cert_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]

    stats = compute_training_stats(np.concatenate([X_mem, X_q], axis=0))
    mem_act = build_activation_matrix(raw_experts, X_mem, mem_ts, stats)
    q_act = build_activation_matrix(raw_experts, X_q, trainq_ts, stats)
    cert_act = build_activation_matrix(raw_experts, X_cert, cert_ts, stats)
    val_act = build_activation_matrix(raw_experts, X_val, val_ts, stats)

    # final ablation: memory = full TRAIN_CORE (core_starts), queries = CERT, eval = VAL
    X_bank_all, Y_bank_all = build_windows(series, core_starts, CONTEXT, HORIZON)
    X_bank_res = residualize(X_bank_all)
    bank_ts = [dates[s + CONTEXT] for s in core_starts]
    bank_all_act = build_activation_matrix(raw_experts, X_bank_all, bank_ts, stats)

    # save ORIGINAL raw-pool activation (E9 dead, since raw pool has uncompilable E9)
    orig_bank_all_act = bank_all_act.copy()
    orig_cert_act = cert_act.copy()
    orig_val_act = val_act.copy()

    emb = embed_claims([e["claim"] for e in raw_experts], device)
    expert_dim = emb.shape[1]
    n_slots = n_raw
    print(f"TRAIN_CORE windows={len(core_starts)} (PV mem={len(mem_starts)}+trainq={len(trainq_starts)}), "
          f"CERT windows={len(cert_starts)}, VAL windows={len(val_starts)} (TEST not read)")

    # ---- 1-4. Evidence / Compiler / Fidelity / Generalization (TRAIN_CORE) ----
    evidence = build_train_core_evidence()
    aud = []
    for e in raw_experts:
        e.setdefault("id", e.get("expert_id"))
        obs = audit_observability_for_expert(e, FEATURE_REGISTRY)
        compiled = compile_condition_v6(e, FEATURE_REGISTRY, evidence, COMPILER_CONFIG)
        e["compiled"] = compiled   # downstream activation uses the v7-compiled primitives
        scope = analyze_claim_scope_v6(e, METADATA)
        proxy = validate_proxy_v6(e, compiled, obs)
        checks = critique_v6(e, scope, evidence, compiled, proxy, obs)
        ev_pass = (obs["all_observable"]
                   and checks["support"]["ok"] and checks["coverage"]["ok"]
                   and checks["counter"]["ok"] and checks["population"]["ok"])
        fid_pass = proxy["faithful"] and compiled["status"] == "COMPILED"
        aud.append({"expert_id": e["expert_id"], "compiled": compiled,
                    "ev_pass": ev_pass, "fid_pass": fid_pass})

    # re-compute activation matrices with the v7-compiled primitives (E9 is now live)
    mem_act = build_activation_matrix(raw_experts, X_mem, mem_ts, stats)
    q_act = build_activation_matrix(raw_experts, X_q, trainq_ts, stats)
    cert_act = build_activation_matrix(raw_experts, X_cert, cert_ts, stats)
    val_act = build_activation_matrix(raw_experts, X_val, val_ts, stats)
    bank_all_act = build_activation_matrix(raw_experts, X_bank_all, bank_ts, stats)

    # generalization: activation coverage on TRAIN_CORE (mem+q), chronological
    for i in range(n_raw):
        a = np.concatenate([mem_act[:, i], q_act[:, i]], axis=0)
        cov = audit_activation_coverage(a, mem_ts + trainq_ts, n_blocks=4)
        aud[i]["coverage"] = cov
        aud[i]["gen_pass"] = (not cov["dead"]) and (not cov["constant"])

    # ---- 5. Predictive Value (WITHOUT vs WITH, 3 seeds, CERT) ----
    print("\n--- Predictive Value Critic (3 seeds, CERT) ---")
    passing = [i for i in range(n_raw) if aud[i]["ev_pass"] and aud[i]["fid_pass"] and aud[i]["gen_pass"]]
    print(f"passing Evidence/Fidelity/Generalization: {len(passing)}/{n_raw}")

    with_all = {}
    for seed in SEEDS:
        with_all[seed] = run_pv(seed, emb, None, X_mem, Y_mem, X_q, Y_q,
                                X_mem_res, X_q_res, mem_act, q_act,
                                X_cert, Y_cert, X_cert_res, cert_act, device)
        print(f"  WITH all experts, seed{seed}: CERT mse = {with_all[seed]:.4f}")

    pv = {}
    for i in passing:
        deltas = {}
        for seed in SEEDS:
            without = run_pv(seed, emb, i, X_mem, Y_mem, X_q, Y_q,
                             X_mem_res, X_q_res, mem_act, q_act,
                             X_cert, Y_cert, X_cert_res, cert_act, device)
            deltas[seed] = with_all[seed] - without      # delta = with - without (<0 improvement)
        d = [deltas[s] for s in SEEDS]
        mean_d = float(np.mean(d))
        n_neg = sum(1 for x in d if x < 0)
        n_pos = sum(1 for x in d if x > 0)
        if n_neg >= 2 and mean_d < 0:
            decision = "POSITIVE"
        elif n_pos >= 2 and mean_d > 0:
            decision = "NEGATIVE"
        else:
            decision = "MIXED"
        pv[i] = {"deltas": {s: round(deltas[s], 4) for s in SEEDS}, "decision": decision}
        print(f"  {expert_ids[i]}: deltas={pv[i]['deltas']} -> {decision}")

    # ---- 6. Redundancy (TRAIN_CORE activation) ----
    accepted_order = sorted([i for i in passing if pv[i]["decision"] == "POSITIVE"],
                            key=lambda i: float(np.mean([pv[i]["deltas"][s] for s in SEEDS])))
    final_pool = []
    for i in accepted_order:
        dup = False
        a = np.concatenate([mem_act[:, i], q_act[:, i]])
        for j in final_pool:
            b = np.concatenate([mem_act[:, j], q_act[:, j]])
            if abs(_corr(a, b)) > 0.95 or _jaccard(a, b) > 0.95:
                dup = True
                break
        if not dup:
            final_pool.append(i)

    keep = set(final_pool)
    final_expert_ids = [expert_ids[i] for i in final_pool]
    print(f"\nfinal certified v7 pool: {final_expert_ids} ({len(final_pool)} experts)")

    # ---- final ablation A/B/C/D on VAL (frozen pool) ----
    print("\n--- Final ablation (VAL, 3 seeds) ---")
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_bank = np.zeros_like(bank_all_act)
    zero_cert = np.zeros_like(cert_act)
    zero_val = np.zeros_like(val_act)

    emb_v7 = np.zeros_like(emb)
    bank_v7 = np.zeros_like(bank_all_act)
    cert_v7 = np.zeros_like(cert_act)
    val_v7 = np.zeros_like(val_act)
    for i in final_pool:
        emb_v7[i] = emb[i]
        bank_v7[:, i] = bank_all_act[:, i]
        cert_v7[:, i] = cert_act[:, i]
        val_v7[:, i] = val_act[:, i]

    configs = {
        "base":       (emb_zero, zero_bank, zero_cert, zero_val, False),
        "shape":      (emb_zero, zero_bank, zero_cert, zero_val, True),
        "raw_expert": (emb,      orig_bank_all_act, orig_cert_act, orig_val_act, True),
        "critic_v7":  (emb_v7,   bank_v7, cert_v7, val_v7, True),
    }

    per_seed = {k: [] for k in configs}
    for seed in SEEDS:
        seed_everything(seed)
        base = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        for name, (embx, bank_act_x, cert_act_x, val_act_x, use_shape) in configs.items():
            model = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
            model.load_state_dict(init_state)
            model.set_expert_embeddings(embx)
            train_arm(model, X_bank_all, Y_bank_all, X_cert, Y_cert,
                      X_bank_res, X_cert_res, bank_act_x, cert_act_x, use_shape, device)
            pred = evaluate(model, X_val, X_bank_all, Y_bank_all, X_val_res, X_bank_res,
                            val_act_x, bank_act_x, use_shape, device)
            per_seed[name].append({"seed": seed, "mse": mse(Y_val, pred), "mae": mae(Y_val, pred)})
            print(f"  [{name}] seed{seed} mse={per_seed[name][-1]['mse']:.4f}")

    # ---- aggregate + report ----
    def mean_std(vals):
        a = np.array([v["mse"] for v in vals], float)
        b = np.array([v["mae"] for v in vals], float)
        return (float(a.mean()), float(a.std()), float(b.mean()), float(b.std()))

    summary = {k: None for k in configs}
    for k, vals in per_seed.items():
        mm, ms, am, as_ = mean_std(vals)
        summary[k] = {"mse_mean": mm, "mse_std": ms, "mae_mean": am, "mae_std": as_, "per_seed": vals}

    b, s, r, c = summary["base"], summary["shape"], summary["raw_expert"], summary["critic_v7"]
    gains = {
        "shape_vs_base": (b["mse_mean"] - s["mse_mean"]) / b["mse_mean"] * 100,
        "raw_vs_shape": (s["mse_mean"] - r["mse_mean"]) / s["mse_mean"] * 100,
        "critic_v7_vs_raw": (r["mse_mean"] - c["mse_mean"]) / r["mse_mean"] * 100,
        "critic_v7_vs_shape": (s["mse_mean"] - c["mse_mean"]) / s["mse_mean"] * 100,
        "critic_v7_vs_base": (b["mse_mean"] - c["mse_mean"]) / b["mse_mean"] * 100,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for i in range(n_raw):
        pvd = pv.get(i, {})
        audit_rows.append({
            "expert_id": expert_ids[i], "name": raw_experts[i]["name"],
            "evidence": "PASS" if aud[i]["ev_pass"] else "FAIL",
            "fidelity": "PASS" if aud[i]["fid_pass"] else "FAIL",
            "generalization": "PASS" if aud[i]["gen_pass"] else "FAIL",
            "predictive": pvd.get("decision", "N/A"),
            "final": "ACCEPT" if i in keep else "REJECT",
        })
    result = {
        "critic_version": "v7_final",
        "dataset_specific_rules": False,
        "raw_expert_regenerated": False,
        "predictive_selection_used_final_val": False,
        "certification_data": "TRAIN_ONLY",
        "deterministic_enabled": True,
        "seed42_removed": False,
        "test_used": False,
        "train_core_range": evidence["train_core_range"],
        "cert_range": [str(cert_ts[0]), str(cert_ts[-1])],
        "val_range": [str(val_ts[0]), str(val_ts[-1])],
        "audit": audit_rows,
        "predictive_value": {expert_ids[i]: pv.get(i, {}) for i in range(n_raw)},
        "final_accepted_experts": final_expert_ids,
        "n_accepted": len(final_pool),
        "arms": {k: {kk: vv for kk, vv in summary[k].items() if kk != "per_seed"}
                 for k in ("base", "shape", "raw_expert", "critic_v7")},
        "gains": gains,
        "seeds": SEEDS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "final_accepted_experts.json").write_text(
        json.dumps([{"expert_id": expert_ids[i], "name": raw_experts[i]["name"],
                     "claim": raw_experts[i].get("claim", "")} for i in final_pool],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "per_seed_results.json").write_text(
        json.dumps({"seeds": SEEDS, "arms": per_seed}, ensure_ascii=False, indent=2), encoding="utf-8")

    # report
    L = []
    L.append("【结果】")
    L.append("")
    L.append("## 1. Raw Expert 10 条完整审核表")
    L.append("")
    L.append("| Expert | Evidence | Fidelity | Generalization | Predictive | Final |")
    L.append("|---|---|---|---|---|---|")
    for row in audit_rows:
        L.append(f"| {row['expert_id']} | {row['evidence']} | {row['fidelity']} | {row['generalization']} "
                 f"| {row['predictive']} | {row['final']} |")
    L.append("")
    L.append("## 2. Predictive Value 表")
    L.append("")
    L.append("| Expert | seed42 delta | seed123 delta | seed2026 delta | mean delta | decision |")
    L.append("|---|---|---|---|---|---|")
    for i in passing:
        d = pv[i]["deltas"]
        mean_d = float(np.mean([d[s] for s in SEEDS]))
        L.append(f"| {expert_ids[i]} | {d[42]:+.2f} | {d[123]:+.2f} | {d[2026]:+.2f} | {mean_d:+.2f} | {pv[i]['decision']} |")
    L.append("")
    L.append("## 3. E9 特别报告")
    L.append("")
    i9 = expert_ids.index("E9")
    r9 = audit_rows[i9]
    L.append(f"- Evidence={r9['evidence']} Fidelity={r9['fidelity']} Generalization={r9['generalization']} "
             f"Predictive={r9['predictive']} Final={r9['final']}")
    if i9 in passing:
        d9 = pv[i9]["deltas"]
        L.append(f"- deltas: seed42={d9[42]:+.2f} seed123={d9[123]:+.2f} seed2026={d9[2026]:+.2f} "
                 f"mean={np.mean([d9[s] for s in SEEDS]):+.2f}")
    L.append("")
    L.append(f"## 4. Final v7 ACCEPT 数量 = {len(final_pool)}")
    L.append("")
    L.append("## 5. Final MSE 表 (VAL)")
    L.append("")
    L.append("| Method | seed42 | seed123 | seed2026 | mean MSE | std | mean MAE | std |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, key in (("BASE", "base"), ("+SHAPE", "shape"), ("+RAW EXPERT", "raw_expert"), ("+CRITIC EXPERT v7", "critic_v7")):
        vals = sorted(per_seed[key], key=lambda v: v["seed"])
        cells = " | ".join(f"{v['mse']:.2f}" for v in vals)
        ss = summary[key]
        L.append(f"| {label} | {cells} | {ss['mse_mean']:.2f} | {ss['mse_std']:.2f} | {ss['mae_mean']:.4f} | {ss['mae_std']:.4f} |")
    L.append("")
    L.append("## 6. Gains")
    L.append("")
    L.append(f"- Shape vs Base = {gains['shape_vs_base']:+.2f}%")
    L.append(f"- Raw vs Shape = {gains['raw_vs_shape']:+.2f}%")
    L.append(f"- Critic v7 vs Raw = {gains['critic_v7_vs_raw']:+.2f}%")
    L.append(f"- Critic v7 vs Shape = {gains['critic_v7_vs_shape']:+.2f}%")
    L.append(f"- Critic v7 vs Base = {gains['critic_v7_vs_base']:+.2f}%")
    L.append("")

    L.append("【结论】")
    L.append("")
    L.append(f"A. Expert 是否有效？ {'是' if r['mse_mean'] < b['mse_mean'] else '否'}"
             f"（raw={r['mse_mean']:.2f} vs base={b['mse_mean']:.2f}）")
    blocked = sum(1 for i in range(n_raw) if i in passing and pv[i]['decision'] == 'NEGATIVE')
    L.append(f"B. Critic v7 是否成功挡住负贡献知识？ {'是' if blocked > 0 else '否'}（挡住 {blocked} 条 NEGATIVE）")
    L.append(f"C. E9 是否被 Predictive Value Critic 自动识别？ "
             f"{'是（NEGATIVE）' if i9 in passing and pv[i9]['decision'] == 'NEGATIVE' else '否/其他'}")
    L.append(f"D. Critic v7 的 MSE 是否优于 Raw Expert？ {'是' if c['mse_mean'] < r['mse_mean'] else '否'}"
             f"（{c['mse_mean']:.2f} vs {r['mse_mean']:.2f}，{gains['critic_v7_vs_raw']:+.2f}%）")
    L.append(f"E. Critic v7 是否优于 Shape / Base？ "
             f"{'是' if c['mse_mean'] < min(s['mse_mean'], b['mse_mean']) else '否'}"
             f"（critic={c['mse_mean']:.2f} shape={s['mse_mean']:.2f} base={b['mse_mean']:.2f}）")
    L.append("F. 本次有没有任何 ETTh1-specific rule？ 否（纯通用 Evidence/Fidelity/Generalization/Predictive/Redundancy）")
    L.append("G. 最终 VAL 是否完全没有参与 knowledge selection？ 是（Predictive Value 只用 CERT，VAL 仅在冻结后用于最终消融）")
    L.append("")
    L.append("critic_version = \"v7_final\"")
    L.append("dataset_specific_rules = false")
    L.append("raw_expert_regenerated = false")
    L.append("predictive_selection_used_final_val = false")
    L.append("certification_data = \"TRAIN_ONLY\"")
    L.append("deterministic_enabled = true")
    L.append("seed42_removed = false")
    L.append("test_used = false")
    report = "\n".join(L)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
