#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Critic v6.1 — minimal modification: Post-Repair Activation Safety Check.

v6.1 is Critic v6 plus ONE new mechanism.  It does NOT refactor the Critic, does
NOT introduce a CERT split, does NOT re-split TRAIN, does NOT regenerate experts,
and does NOT write any dataset / expert / feature-specific rule.

The single new check closes a gap in v6: after Compiler Repair turns a previously
inactive / uncompilable knowledge into an executable activation, v6 never verified
whether that newly-created activation stably HURTS the downstream model.

Trigger (fully generic, decided from execution state only):
    pre_repair_execution_state  = "inactive"  (raw pool compiled has 0 primitives)
    post_repair_execution_state = "active"    (v6 compiler produced >=1 primitive)
    + a compiler repair actually occurred (compile returned a non-empty repair log)

Only knowledge satisfying ALL THREE is safety-checked.  Every other expert keeps its
exact v6 decision (they are NOT re-screened by MSE).

Safety check (paired ON/OFF, v6's original train/val protocol unchanged):
    ON  = repaired activation active
    OFF = repaired activation forced to 0
    everything else identical; 3 strict-deterministic seeds {42,123,2026}
    delta = MSE_ON - MSE_OFF  (delta > 0 => ON worse)
    SAFETY_FAIL if >=2/3 seeds delta > 0 AND mean delta > 0, else SAFETY_PASS.

On SAFETY_FAIL: the repaired activation is disabled (final_active=false).  No
regeneration, no threshold tuning, no expert rewrite.

Final ETTh1 ablation (strict deterministic): A BASE / B +SHAPE / C +RAW EXPERT /
D +CRITIC EXPERT v6 / E +CRITIC EXPERT v6.1.
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

from run_expert_agent_closed_loop_v6 import (            # generic v6 machinery (reused, NOT modified)
    COMPILER_CONFIG,
    compile_condition_v6,
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
OUT_DIR = Path("results/etth1_critic_v6_1_final")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

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
}


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
# v6 TRAIN-region evidence (full TRAIN region, identical to v6; NOT re-split)
# --------------------------------------------------------------------------- #
def build_v6_evidence():
    df = pd.read_csv("data/ETTh1.csv")
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

    hourly_mean = np.array([float(y[hours == h].mean()) for h in range(24)])
    dow_mean = np.array([float(y[dows == d].mean()) for d in range(7)])

    ctx = 64
    n_win = train_len - ctx
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
# training / eval helpers (v6 protocol: train fit+bank, eval VAL)
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


def train_and_eval(seed, emb, bank_act, fit_act, val_act, use_shape,
                   X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res, X_val, Y_val, X_val_res, device):
    seed_everything(seed)
    model = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=emb.shape[1], n_experts=emb.shape[0],
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    model.set_expert_embeddings(emb)
    train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device)
    pred = evaluate(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res, val_act, bank_act, use_shape, device)
    return mse(Y_val, pred), mae(Y_val, pred)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| strict deterministic ON")

    raw_experts = _sorted_experts(json.loads(RAW_POOL.read_text(encoding="utf-8")))
    n_raw = len(raw_experts)
    expert_ids = [e["expert_id"] for e in raw_experts]
    print(f"frozen raw experts = {n_raw}")

    # ---- v6 compile (reused unchanged) -> v6 pool + trigger detection ----
    evidence = build_v6_evidence()
    v6_compiled = []
    trigger = []      # list of expert indices that satisfy the repair-created-activation trigger
    for e in raw_experts:
        e.setdefault("id", e.get("expert_id"))
        raw_prim = e.get("compiled", {}).get("primitives", [])
        pre_state = "inactive" if len(raw_prim) == 0 else "active"
        compiled = compile_condition_v6(e, FEATURE_REGISTRY, evidence, COMPILER_CONFIG)
        post_state = "active" if compiled["status"] == "COMPILED" and len(compiled["primitives"]) >= 1 else "inactive"
        repair_happened = len(compiled.get("repairs", [])) > 0
        triggered = (pre_state == "inactive") and (post_state == "active") and repair_happened
        v6_compiled.append(compiled)
        trigger.append({
            "expert_id": e["expert_id"],
            "pre_repair_state": pre_state,
            "post_repair_state": post_state,
            "repair_happened": repair_happened,
            "triggered": triggered,
        })
        if triggered:
            print(f"  TRIGGER repair-created activation: {e['expert_id']} ({pre_state} -> {post_state})")

    triggered_idx = [i for i, t in enumerate(trigger) if t["triggered"]]
    print(f"triggered experts (repair-created activation): {[expert_ids[i] for i in triggered_idx]}")

    # ---- data (v6 protocol: bank memory + fit queries + val eval) ----
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

    bank_ts = [dates[s + CONTEXT] for s in bank_starts]
    fit_ts = [dates[s + CONTEXT] for s in fit_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]

    stats = compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))

    # raw pool activation (v5 compiled: E9 dead) and v6 pool activation (v6 compiled: E9 live)
    raw_bank_act = build_activation_matrix(raw_experts, X_bank, bank_ts, stats)
    raw_fit_act = build_activation_matrix(raw_experts, X_fit, fit_ts, stats)
    raw_val_act = build_activation_matrix(raw_experts, X_val, val_ts, stats)

    # v6 pool = raw_experts with v6-compiled primitives
    v6_experts = [dict(e) for e in raw_experts]
    for i, e in enumerate(v6_experts):
        e["compiled"] = v6_compiled[i]
    v6_bank_act = build_activation_matrix(v6_experts, X_bank, bank_ts, stats)
    v6_fit_act = build_activation_matrix(v6_experts, X_fit, fit_ts, stats)
    v6_val_act = build_activation_matrix(v6_experts, X_val, val_ts, stats)

    emb = embed_claims([e["claim"] for e in raw_experts], device)
    expert_dim = emb.shape[1]
    n_slots = n_raw

    # ---- safety check: ON (v6 all active) vs OFF_i (expert i zeroed) ----
    print("\n--- Post-Repair Activation Safety Check (3 seeds, VAL) ---")
    # ON = D arm (v6 pool, all active)
    on_mse, on_mae = {}, {}
    for seed in SEEDS:
        on_mse[seed], on_mae[seed] = train_and_eval(
            seed, emb, v6_bank_act, v6_fit_act, v6_val_act, True,
            X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res, X_val, Y_val, X_val_res, device)
        print(f"  ON (v6 all active) seed{seed}: VAL mse = {on_mse[seed]:.4f}")

    safety = {}
    for i in triggered_idx:
        bank_off = v6_bank_act.copy(); bank_off[:, i] = 0.0
        fit_off = v6_fit_act.copy(); fit_off[:, i] = 0.0
        val_off = v6_val_act.copy(); val_off[:, i] = 0.0
        off_mse = {}
        for seed in SEEDS:
            off_mse[seed], _ = train_and_eval(
                seed, emb, bank_off, fit_off, val_off, True,
                X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res, X_val, Y_val, X_val_res, device)
            print(f"  OFF {expert_ids[i]} seed{seed}: VAL mse = {off_mse[seed]:.4f}")
        deltas = {s: on_mse[s] - off_mse[s] for s in SEEDS}
        mean_d = float(np.mean([deltas[s] for s in SEEDS]))
        n_pos = sum(1 for s in SEEDS if deltas[s] > 0)
        fail = (n_pos >= 2 and mean_d > 0)
        safety[i] = {"deltas": {s: round(deltas[s], 4) for s in SEEDS},
                     "mean_delta": round(mean_d, 4), "safety": "FAIL" if fail else "PASS"}
        print(f"  {expert_ids[i]}: deltas={safety[i]['deltas']} mean={mean_d:+.4f} -> {'SAFETY_FAIL' if fail else 'SAFETY_PASS'}")

    disabled = {i for i in triggered_idx if safety[i]["safety"] == "FAIL"}

    # ---- final ablation A/B/C/D/E ----
    print("\n--- Final ablation (VAL, 3 seeds) ---")
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_bank = np.zeros_like(raw_bank_act)
    zero_fit = np.zeros_like(raw_fit_act)
    zero_val = np.zeros_like(raw_val_act)

    # v6.1 pool = v6 pool with disabled experts zeroed
    v61_bank_act = v6_bank_act.copy()
    v61_fit_act = v6_fit_act.copy()
    v61_val_act = v6_val_act.copy()
    for i in disabled:
        v61_bank_act[:, i] = 0.0
        v61_fit_act[:, i] = 0.0
        v61_val_act[:, i] = 0.0

    configs = {
        "base":       (emb_zero, zero_bank, zero_fit, zero_val, False),
        "shape":      (emb_zero, zero_bank, zero_fit, zero_val, True),
        "raw_expert": (emb,      raw_bank_act, raw_fit_act, raw_val_act, True),
        "critic_v6":  (emb,      v6_bank_act,  v6_fit_act,  v6_val_act,  True),
        "critic_v6_1":(emb,      v61_bank_act, v61_fit_act, v61_val_act, True),
    }

    per_seed = {k: [] for k in configs}
    for seed in SEEDS:
        seed_everything(seed)
        base = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
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

    v6_m = summary["critic_v6"]["mse_mean"]
    v61_m = summary["critic_v6_1"]["mse_mean"]
    raw_m = summary["raw_expert"]["mse_mean"]
    shape_m = summary["shape"]["mse_mean"]
    base_m = summary["base"]["mse_mean"]
    gains = {
        "v61_vs_v6": (v6_m - v61_m) / v6_m * 100,
        "v61_vs_raw": (raw_m - v61_m) / raw_m * 100,
        "v61_vs_shape": (shape_m - v61_m) / shape_m * 100,
        "v61_vs_base": (base_m - v61_m) / base_m * 100,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "critic_base": "v6",
        "critic_version": "v6.1",
        "change_scope": "post_repair_activation_safety_only",
        "cert_used": False, "train_resplit": False, "expert_regenerated": False,
        "all_experts_rescreened_by_mse": False, "dataset_specific_rules": False,
        "expert_specific_rules": False, "test_used": False, "deterministic_enabled": True,
        "trigger": trigger,
        "safety_check": {expert_ids[i]: safety.get(i, {}) for i in triggered_idx},
        "disabled": [expert_ids[i] for i in disabled],
        "arms": {k: {kk: vv for kk, vv in summary[k].items() if kk != "per_seed"}
                 for k in configs},
        "gains": gains,
        "seeds": SEEDS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "per_seed_results.json").write_text(
        json.dumps({"seeds": SEEDS, "arms": per_seed}, ensure_ascii=False, indent=2), encoding="utf-8")

    # report
    L = []
    L.append("【结果】")
    L.append("")
    L.append("## 1. Triggered repair-created activations")
    L.append("")
    L.append("| Expert | pre_repair_state | post_repair_state | seed42 delta | seed123 delta | seed2026 delta | mean delta | safety | final active |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for i in triggered_idx:
        s = safety[i]
        d = s["deltas"]
        final_active = "false" if i in disabled else "true"
        L.append(f"| {expert_ids[i]} | {trigger[i]['pre_repair_state']} | {trigger[i]['post_repair_state']} "
                 f"| {d[42]:+.2f} | {d[123]:+.2f} | {d[2026]:+.2f} | {s['mean_delta']:+.2f} | {s['safety']} | {final_active} |")
    L.append("")
    L.append("## 2. Final MSE table (VAL)")
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
    L.append("## 3. Gains")
    L.append("")
    L.append(f"- v6.1 vs v6 = {gains['v61_vs_v6']:+.2f}%")
    L.append(f"- v6.1 vs Raw = {gains['v61_vs_raw']:+.2f}%")
    L.append(f"- v6.1 vs Shape = {gains['v61_vs_shape']:+.2f}%")
    L.append(f"- v6.1 vs Base = {gains['v61_vs_base']:+.2f}%")
    L.append("")

    L.append("【结论】")
    L.append("")
    L.append(f"1. v6 中有多少条知识触发 repair-created activation？ {len(triggered_idx)}（{ [expert_ids[i] for i in triggered_idx] or '无'}）")
    L.append(f"2. 哪些 repaired activation 被 Safety Check 判为稳定有害？ { [expert_ids[i] for i in disabled] or '无'}")
    L.append(f"3. v6.1 是否改善 v6 的 MSE？ {'是' if v61_m < v6_m else '否'}"
             f"（{v61_m:.2f} vs {v6_m:.2f}，{gains['v61_vs_v6']:+.2f}%）")
    L.append(f"4. v6.1 与 Raw Expert 的差距是多少？ {v61_m:.2f} vs {raw_m:.2f}（{gains['v61_vs_raw']:+.2f}%）")
    L.append(f"5. 未触发 repair 的 Expert 是否完全没有被重新筛选？ 是（只对 triggered 做 Safety Check，其余保持 v6 决定）")
    L.append("6. 是否存在任何 dataset-specific / expert-specific rule？ 否（trigger 只依据 pre/post execution state + repair log）")
    L.append("7. TEST 是否保持完全未访问？ 是")
    L.append("")
    L.append("critic_base = \"v6\"")
    L.append("critic_version = \"v6.1\"")
    L.append("change_scope = \"post_repair_activation_safety_only\"")
    L.append("cert_used = false")
    L.append("train_resplit = false")
    L.append("expert_regenerated = false")
    L.append("all_experts_rescreened_by_mse = false")
    L.append("dataset_specific_rules = false")
    L.append("expert_specific_rules = false")
    L.append("test_used = false")
    L.append("deterministic_enabled = true")
    report = "\n".join(L)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
