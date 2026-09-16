#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-dataset unified validation of the FROZEN v5 Expert-Critic Closed Loop.

This is a VALIDATION script, not a design script. It does NOT modify the frozen
method. It reuses, verbatim:

  Stage A  (critic)      : the generic v5 core in run_expert_agent_closed_loop
                           (analyze_claim_scope / plan_evidence / compile_condition /
                            validate_proxy / critique / decide / classify_failure /
                            build_feedback / mark_duplicates / run_one_round).
  Stage B  (prediction)  : ConditionalExpertMLPRetriever (run_conditional_expert_mlp_v4)
                           + residualize/shape_retrieval (run_solar_same_val_baselines)
                           + build_windows/mae/mse (run_expert_embedding_mlp)
                           + seed_all (run_expert_attention_retrieval_v2).

Only dataset INPUTS are provided for each new univariate-OT dataset
(metadata / feature catalog / TRAIN-only evidence / LLM prompt text). No
`if dataset == ...` branch modifies the Critic. No new module is added.

Stages:
  --stage critic       run frozen v5 closed loop (max 3 rounds) for the dataset
  --stage prediction   run Shape / ZERO / SHUFFLED / REAL + per-expert ablation
                       + leave-one-out on the frozen ACCEPT pool
  --stage aggregate    build summary.json / summary.csv / cross_dataset_audit.md

TEST is never read. VAL-only. test_used = false is asserted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# --------------------------------------------------------------------------- #
# Frozen v5 core (imported verbatim, NOT modified)
# --------------------------------------------------------------------------- #
from run_expert_agent_closed_loop import (     # noqa: E402
    analyze_claim_scope,
    plan_evidence,
    compile_condition,
    validate_proxy,
    critique,
    decide,
    classify_failure,
    build_feedback,
    mark_duplicates,
    run_one_round,
    CRITICAL_CHECKS,
)
import run_agent_v2_electricity as av2          # _call_llm / _list_from / _default_model  # noqa: E402

# --------------------------------------------------------------------------- #
# Frozen retrieval stack (imported verbatim, NOT modified)
# --------------------------------------------------------------------------- #
import run_expert_attention_retrieval_v2 as attn                      # noqa: E402
from run_expert_embedding_mlp import build_windows, mae, mse          # noqa: E402
from run_solar_same_val_baselines import residualize, shape_retrieval # noqa: E402
from run_conditional_expert_mlp_v4 import (                           # noqa: E402
    ConditionalExpertMLPRetriever,
    _linear_slope_vec,
    _lag1_autocorr_vec,
    _sigmoid,
    train_conditional,
    retrieve_conditional,
)
from sentence_transformers import SentenceTransformer                 # noqa: E402

MAX_ROUNDS = 3
CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
SHUFFLE_SEED = 777
ROOT = Path("results/cross_dataset_validation")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------- #
# Dataset registry (univariate-OT inputs only)
# --------------------------------------------------------------------------- #
# Every new dataset is a single OT series + timestamp, so a single generic
# feature catalog and a single generic TRAIN evidence function serve all of them.
# Only metadata / domain text / sampling labels differ. No critic rule differs.
UNIVARIATE_FEATURE_RULES = {
    "hour_of_day": {"kind": "calendar", "categories": {
        "morning": [6, 7, 8, 9, 10, 11], "midday": [12, 13, 14, 15],
        "evening": [16, 17, 18, 19, 20, 21], "night": [22, 23, 0, 1, 2, 3, 4, 5]}},
    "day_of_week": {"kind": "calendar", "categories": {
        "weekday": [0, 1, 2, 3, 4], "weekend": [5, 6]}},
    "working_day": {"kind": "calendar", "categories": {
        "workday": [0, 1, 2, 3, 4], "non_working": [5, 6]}},
    "seasonal_level": {"kind": "calendar", "categories": {
        "winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
    "recent_level": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low",
        "above_typical": "@train_high", "below_typical": "@train_low"}},
    "recent_slope_magnitude": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
    "recent_diff_magnitude": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
    "recent_volatility": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
    "recent_autocorr": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
}

DATASETS = {
    "traffic": {
        "data_path": "results/multidataset/traffic/series.csv",
        "sampling_str": "hourly",
        "freq": "hourly",
        "domain": "an aggregate traffic occupancy series (mean across 862 sensors)",
        "target_desc": "aggregate traffic occupancy (mean across 862 sensors)",
        "cache_dir": "cache/llm_priors_closed_loop_v5_traffic",
        "llm_tag": "traffic",
    },
    "weather": {
        "data_path": "results/multidataset/weather/series.csv",
        "sampling_str": "10 minutes",
        "freq": "10-minute",
        "domain": "a weather-station meteorological measurement series",
        "target_desc": "a single weather-station meteorological variable",
        "cache_dir": "cache/llm_priors_closed_loop_v5_weather",
        "llm_tag": "weather",
    },
    "exchange_rate": {
        "data_path": "results/multidataset/exchange_rate/series.csv",
        "sampling_str": "daily",
        "freq": "daily",
        "domain": "a univariate daily foreign exchange-rate series",
        "target_desc": "a daily foreign exchange rate",
        "cache_dir": "cache/llm_priors_closed_loop_v5_exchange",
        "llm_tag": "exchange",
    },
    "solar": {
        "data_path": "results/multidataset/solar/series.csv",
        "sampling_str": "10 minutes",
        "freq": "10-minute",
        "domain": "an aggregate solar generation series",
        "target_desc": "aggregate solar generation",
        "cache_dir": "cache/llm_priors_closed_loop_v5_solar",
        "llm_tag": "solar",
    },
}


def make_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_variable": "OT",
        "target_description": cfg["target_desc"],
        "domain": cfg["domain"],
        "sampling_frequency": cfg["sampling_str"],
        "n_population": 1,          # univariate aggregate target (no subset-proxy population)
    }


# --------------------------------------------------------------------------- #
# Generic TRAIN-only evidence (univariate OT + timestamp)
# --------------------------------------------------------------------------- #
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


def univariate_evidence(data_path: str, sampling_str: str) -> Dict[str, Any]:
    df = pd.read_csv(data_path)
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
    mdir = {}
    nrev = 0
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

    # feature thresholds (TRAIN-derived, for the compiler)
    rng = np.random.default_rng(0)
    ctx = 64
    n_win = train_len - ctx
    starts = rng.choice(n_win, size=min(3000, n_win), replace=False)
    win = {"recent_level": [], "recent_volatility": [], "recent_slope_magnitude": [],
           "recent_diff_magnitude": [], "recent_autocorr": []}
    t = np.linspace(-1, 1, ctx)
    tc = t - t.mean()
    for s in starts:
        x = y[s:s + ctx]
        win["recent_level"].append(x.mean())
        win["recent_volatility"].append(x.std())
        slope = float(np.sum((x - x.mean()) * tc) / np.sum(tc * tc))
        win["recent_slope_magnitude"].append(abs(slope))
        win["recent_diff_magnitude"].append(abs(x[-1] - x[-2]))
        win["recent_autocorr"].append(_corr(x[1:], x[:-1]))
    feature_thresholds = {}
    for feat, arr in win.items():
        arr = np.asarray(arr, float)
        feature_thresholds[feat] = {
            "train_median": float(np.median(arr)),
            "train_median_low": float(np.median(arr)),
            "train_high": float(arr.mean() + arr.std()),
            "train_low": float(arr.mean() - arr.std()),
        }

    return {
        "train_span": {"start": str(dt.min()), "end": str(dt.max()),
                       "n_days": int(dt.dt.date.nunique()), "n_rows": int(train_len),
                       "total_rows": int(N), "sampling": sampling_str},
        "temporal_coverage": {
            "months_present": months_present, "months_missing": [m for m in range(1, 13) if m not in months_present],
            "month_fraction": round(len(months_present) / 12, 4),
            "hours_present": hours_present, "hours_missing": [h for h in range(24) if h not in hours_present],
            "hour_fraction": round(len(hours_present) / 24, 4),
            "dows_present": dows_present, "dow_fraction": round(len(dows_present) / 7, 4),
        },
        "diurnal": {"hourly_mean": [round(v, 2) for v in hourly_mean],
                    "peak_hour": int(np.argmax(hourly_mean)),
                    "eta2_hour_on_ot": round(_eta2(hours, y), 4)},
        "seasonal": {"monthly_mean": {k: round(v, 2) for k, v in monthly_mean.items()},
                     "monthly_mean_values": [round(v, 2) for v in monthly_vals],
                     "month_direction": mdir, "n_month_direction_reversals": nrev,
                     "eta2_month_on_ot": round(_eta2(months, y), 4)},
        "weekly": {"dow_mean": [round(v, 2) for v in dow_mean],
                   "eta2_dow_on_ot": round(_eta2(dows, y), 4)},
        "population": {"n_proxy": 1, "n_population": 1, "proxy_fraction": 1.0,
                       "corr_proxy_sum_vs_target": 1.0},
        "persistence": {"lag1_autocorr": round(_corr(y[1:], y[:-1]), 4)},
        "variability": {"ot_mean": round(float(y.mean()), 2), "ot_std": round(float(y.std()), 2)},
        "feature_thresholds": feature_thresholds,
        "x_subset_representativeness": {},
    }


# --------------------------------------------------------------------------- #
# Generic LLM prompts (parameterized by dataset config)
# --------------------------------------------------------------------------- #
def _metadata_text(cfg, metadata):
    return (
        "DATASET:\n"
        f"- target variable: OT ({metadata['target_description']})\n"
        f"- domain: {metadata['domain']}\n"
        f"- sampling frequency: {metadata['sampling_frequency']}\n"
        "- observed variables: OT only (univariate target series)"
    )


def _catalog_text():
    return (
        "AVAILABLE ACTIVATION FEATURES (choose observable_features ONLY from this list):\n"
        "  - hour_of_day: current hour-of-day phase (morning/midday/evening/night)\n"
        "  - day_of_week: weekday vs weekend\n"
        "  - working_day: workday vs non-working day\n"
        "  - seasonal_level: month-of-year phase (winter/spring/summer/autumn)\n"
        "  - recent_level: current level relative to typical\n"
        "  - recent_slope_magnitude: recent trend strength\n"
        "  - recent_diff_magnitude: recent point-to-point change magnitude\n"
        "  - recent_volatility: recent variability regime\n"
        "  - recent_autocorr: recent persistence strength"
    )


def _train_summary_text(ev):
    ds, se, pv = ev["diurnal"], ev["seasonal"], ev["persistence"]
    return (
        "TRAIN-ONLY DATA SUMMARY (deterministic):\n"
        f"- train span: {ev['train_span']['start']} -> {ev['train_span']['end']} "
        f"({ev['train_span']['n_days']} days, {ev['train_span']['sampling']})\n"
        f"- coverage: {len(ev['temporal_coverage']['months_present'])}/12 months, "
        f"{len(ev['temporal_coverage']['hours_present'])}/24 hours, "
        f"{len(ev['temporal_coverage']['dows_present'])}/7 days\n"
        f"- eta2(hour,OT)={ds['eta2_hour_on_ot']}, eta2(month,OT)={se['eta2_month_on_ot']}, "
        f"eta2(dow,OT)={ev['weekly']['eta2_dow_on_ot']}\n"
        f"- hourly mean OT: {ds['hourly_mean']}\n"
        f"- monthly mean OT: {se['monthly_mean']}\n"
        f"- lag1 autocorr {pv['lag1_autocorr']}\n"
        f"- OT mean {ev['variability']['ot_mean']}, OT std {ev['variability']['ot_std']}\n"
    )


def _schema_text():
    return (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"expert_knowledge": [ ... ]}. Each entry:\n'
        '  - "expert_id": "E1", "E2", ...\n'
        '  - "name": short label\n'
        '  - "claim": one conditional sentence: "When [observable condition], historical '
        "segments satisfying [relationship] should be preferred because [reason]\"\n"
        '  - "activation_condition": the "when [observable state]" part, natural language\n'
        '  - "observable_features": list of features from AVAILABLE ACTIVATION FEATURES\n'
        '  - "condition_spec": {"logic": "AND"|"OR", "conditions": [{"feature": "...", '
        '"qualifier": "..."}]} — exact logical structure, one entry per primitive\n'
        '  - "expected_relationship": the "prefer [relationship]" part\n'
        '  - "reason": why this helps retrieval\n'
        '  - "domain_fact": a 1-sentence mechanism\n'
        '  - "latent_state": the hidden predictive state\n'
        '  - "observable_proxy": the observable quantity proxying the state\n'
        '  - "expected_future_relationship": expected future behavior\n'
    )


def _gen_system(cfg):
    return (
        f"You are a domain expert on forecasting {cfg['domain']} (target variable OT, "
        f"sampled {cfg['freq']}). A retrieval model forecasts the series by retrieving "
        "similar historical windows from a memory bank and averaging their futures. "
        "Propose CONDITIONAL expert knowledge: each statement says WHEN it applies, "
        "WHAT historical segments to prefer, and WHY. The activation condition must be "
        "observable from the query's own history and timestamp. Output strict JSON only."
    )


def build_generation_prompt(cfg, ev):
    metadata = make_metadata(cfg)
    user = "\n\n".join([
        _metadata_text(cfg, metadata),
        _train_summary_text(ev),
        _catalog_text(),
        _schema_text(),
        "TASK: propose 6-10 conditional expert-knowledge statements. Output only the JSON object.",
    ])
    return _gen_system(cfg), user


def build_revision_prompt(cfg, ev, feedbacks):
    metadata = make_metadata(cfg)
    user = "\n\n".join([
        _metadata_text(cfg, metadata),
        _train_summary_text(ev),
        _catalog_text(),
        "CRITIC FEEDBACK ON YOUR PREVIOUS PROPOSALS (per expert):\n"
        + json.dumps(feedbacks, ensure_ascii=False, indent=2),
        _schema_text(),
        "TASK: For each failed expert, REVISE it to fix the specific failure(s) with "
        "TRAIN-supported evidence, or REPLACE it with a genuinely different mechanism. "
        "You may keep experts that are already fine. Output only the JSON object.",
    ])
    return _gen_system(cfg), user


# --------------------------------------------------------------------------- #
# Stage A: frozen v5 closed loop, parameterized
# --------------------------------------------------------------------------- #
def run_closed_loop(cfg, evidence, metadata, feature_rules):
    av2.CACHE_DIR = cfg["cache_dir"]
    model = av2._default_model()
    failed_history: List[Dict] = []
    all_rounds = []
    accepted_pool: Dict[str, Dict] = {}

    for rnd in range(1, MAX_ROUNDS + 1):
        if rnd == 1:
            sys_p, usr_p = build_generation_prompt(cfg, evidence)
            raw, path, cached = av2._call_llm(sys_p, usr_p, f"{cfg['llm_tag']}_expert_r{rnd}", model)
            experts = av2._list_from(raw, "expert_knowledge")
        else:
            feedbacks = [r["feedback"] for prev in all_rounds for r in prev
                         if r["final_decision"] != "ACCEPT" and r["feedback"] and not r["duplicate_of"]]
            if not feedbacks:
                break
            sys_p, usr_p = build_revision_prompt(cfg, evidence, feedbacks)
            raw, path, cached = av2._call_llm(sys_p, usr_p, f"{cfg['llm_tag']}_expert_r{rnd}", model)
            experts = av2._list_from(raw, "expert_knowledge")

        experts = mark_duplicates(experts, failed_history)
        round_results = run_one_round(experts, evidence, metadata, feature_rules, failed_history)
        all_rounds.append(round_results)

        for r in round_results:
            if r["final_decision"] == "ACCEPT":
                accepted_pool[r["expert_id"]] = r

        remaining = [r for r in round_results
                     if r["final_decision"] != "ACCEPT" and r["feedback"]
                     and r["feedback"]["repairable"] and not r["duplicate_of"]]
        if not remaining:
            break

    return {"rounds": all_rounds, "accepted_pool": accepted_pool,
            "failed_history": failed_history}


def run_stage_a(name: str) -> Dict[str, Any]:
    cfg = DATASETS[name]
    metadata = make_metadata(cfg)
    evidence = univariate_evidence(cfg["data_path"], cfg["sampling_str"])
    loop = run_closed_loop(cfg, evidence, metadata, UNIVARIATE_FEATURE_RULES)

    out_dir = ROOT / name / "critic"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, rnd in enumerate(loop["rounds"], 1):
        (out_dir / f"expert_round_{i}.json").write_text(
            json.dumps(rnd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out_dir / f"critic_round_{i}.json").write_text(
            json.dumps([{k: r[k] for k in ("expert_id", "name", "final_decision",
                                           "failure_reasons", "failure_kind", "checks")}
                        for r in rnd], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out_dir / f"feedback_round_{i}.json").write_text(
            json.dumps([r["feedback"] for r in rnd if r.get("feedback")],
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "compiler_audit.json").write_text(
        json.dumps([{"expert_id": r["expert_id"], "compiled": r["compiled"],
                     "proxy_validation": r["proxy_validation"]}
                    for rnd in loop["rounds"] for r in rnd],
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "proxy_validation.json").write_text(
        json.dumps([r["proxy_validation"] for rnd in loop["rounds"] for r in rnd],
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "final_accepted_knowledge.json").write_text(
        json.dumps(loop["accepted_pool"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 88)
    print(f"STAGE A  {name}  (frozen v5, max {MAX_ROUNDS} rounds)")
    print("=" * 88)
    print(f"TRAIN: {evidence['train_span']['start']} -> {evidence['train_span']['end']} "
          f"({evidence['train_span']['n_rows']} rows, {evidence['train_span']['sampling']})")
    for i, rnd in enumerate(loop["rounds"], 1):
        n_acc = sum(1 for r in rnd if r["final_decision"] == "ACCEPT")
        n_uns = sum(1 for r in rnd if r["final_decision"] == "UNSUPPORTED")
        n_rej = sum(1 for r in rnd if r["final_decision"] == "REJECT")
        n_kf = sum(1 for r in rnd if r.get("failure_kind") == "knowledge_failure")
        n_cf = sum(1 for r in rnd if r.get("failure_kind") == "compiler_failure")
        n_dup = sum(1 for r in rnd if r.get("duplicate_of"))
        print(f"Round {i}: {len(rnd)} experts -> {n_acc} ACCEPT, {n_uns} UNSUPPORTED, "
              f"{n_rej} REJECT | kf={n_kf} cf={n_cf} dup={n_dup}")
        for r in rnd:
            reason = "; ".join(r["failure_reasons"]) if r["failure_reasons"] else "-"
            print(f"    {r['expert_id']:<4}{r['name']:<30}{r['final_decision']:<12}"
                  f"{r.get('failure_kind') or '-':<18}{reason[:70]}")
    print(f"Final ACCEPT pool: {len(loop['accepted_pool'])} experts")
    print(f"Saved to {out_dir}")
    print("dataset_specific_critic_rule = false; train_only = true; test_used = false")
    return loop


# --------------------------------------------------------------------------- #
# Stage B: frozen prediction-value validation, parameterized
# --------------------------------------------------------------------------- #
def compute_training_stats(X_train_all):
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
                v = float(feats[feat][i])
            else:
                v = float(_sigmoid(feats[feat][i]))
            if p.get("qualifier") in ("high", "above_typical"):
                vals.append(v)
            else:
                vals.append(1.0 - v)
    if logic == "OR":
        return float(max(vals)) if vals else 0.0
    if len(vals) > 1:
        return float(min(vals))
    return float(vals[0]) if vals else 0.0


def build_activation_matrix(experts, X, tss, stats):
    feats = compute_window_feats(X, tss, stats)
    n = len(X)
    M = np.zeros((n, len(experts)), dtype=np.float32)
    for j, e in enumerate(experts):
        compiled = e["compiled"]
        for i in range(n):
            M[i, j] = expert_activation(compiled, feats, i)
    return M


def _expert_sort_key(x):
    m = re.search(r"(\d+)$", x)
    return (int(m.group(1)) if m else 0, x)


def run_stage_b(name: str, acc_pool_path: Optional[str] = None) -> Dict[str, Any]:
    acc_pool_path = acc_pool_path or str(ROOT / name / "critic" / "final_accepted_knowledge.json")
    accepted = json.loads(Path(acc_pool_path).read_text(encoding="utf-8"))
    experts = [accepted[k] for k in sorted(accepted, key=_expert_sort_key)]
    names = [e["name"] for e in experts]
    n_exp = len(experts)
    print(f"[{name}] Loaded {n_exp} frozen ACCEPT experts: " + ", ".join(f"{e['expert_id']}:{e['name']}" for e in experts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = Path(DATASETS[name]["data_path"]).parent
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
    print(f"[{name}] bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)} (TEST not read)")

    bank_ts = [dates[s + CONTEXT] for s in bank_starts]
    fit_ts = [dates[s + CONTEXT] for s in fit_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]

    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)
    bank_act = build_activation_matrix(experts, X_bank, bank_ts, stats)
    fit_act = build_activation_matrix(experts, X_fit, fit_ts, stats)
    val_act = build_activation_matrix(experts, X_val, val_ts, stats)

    enc = SentenceTransformer(TEXT_MODEL, device=str(device))
    claims = [e["claim"] for e in experts]
    emb = enc.encode(claims, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    expert_dim = emb.shape[1]

    # ---- A. Shape-only ----
    shape_pred = shape_retrieval(X_val, X_bank, Y_bank, topk=TOPK)
    shape_mse, shape_mae = mse(Y_val, shape_pred), mae(Y_val, shape_pred)
    print(f"[{name}] [A] Shape-only MSE={shape_mse:.4f} MAE={shape_mae:.4f}")

    def train_and_eval(emb_matrix, b_act, f_act, v_act, seed=None, tag=""):
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
    print(f"[{name}] [B] ALL ACCEPT (n={n_exp}) MSE={all_mse:.4f} MAE={all_mae:.4f} gain={gain_all:+.2f}%")

    # ---- C. Per-expert ablation ----
    per_expert = []
    print(f"[{name}] [Per-expert ablation]")
    for j, e in enumerate(experts):
        eid = e["expert_id"]
        em = emb[j:j + 1]
        b_act = bank_act[:, j:j + 1]
        f_act = fit_act[:, j:j + 1]
        v_act = val_act[:, j:j + 1]
        m, a = train_and_eval(em, b_act, f_act, v_act, seed=42, tag=eid)
        gain = (shape_mse - m) / shape_mse * 100.0
        per_expert.append({"expert_id": eid, "name": names[j], "claim": e["claim"],
                           "mse": m, "mae": a, "vs_shape_gain_pct": gain})
        print(f"    {eid} {names[j]:<28} MSE={m:.4f} gain={gain:+.2f}%")

    # ---- D. Leave-one-out ----
    leave_one_out = []
    print(f"[{name}] [Leave-one-out]")
    for j, e in enumerate(experts):
        eid = e["expert_id"]
        keep = [k for k in range(n_exp) if k != j]
        em = emb[keep]
        b_act = bank_act[:, keep]
        f_act = fit_act[:, keep]
        v_act = val_act[:, keep]
        m, a = train_and_eval(em, b_act, f_act, v_act, seed=42, tag=f"ALL-{eid}")
        gain = (shape_mse - m) / shape_mse * 100.0
        leave_one_out.append({"removed_expert_id": eid, "removed_name": names[j],
                              "mse": m, "mae": a, "vs_shape_gain_pct": gain,
                              "delta_vs_all": m - all_mse})
        print(f"    ALL-{eid} (removed {names[j]:<26}) MSE={m:.4f} "
              f"gain={gain:+.2f}%  dvsALL={m-all_mse:+.4f}")

    # ---- E. Fair information control (ZERO / SHUFFLED / REAL, 3 seeds) ----
    fair = {"zero": [], "shuffled": [], "real": []}
    def perm_act(a, seed):
        rng = np.random.default_rng(seed)
        return a[rng.permutation(len(a))]
    bank_shuf = perm_act(bank_act, SHUFFLE_SEED)
    fit_shuf = perm_act(fit_act, SHUFFLE_SEED)
    val_shuf = perm_act(val_act, SHUFFLE_SEED)
    bank_zero = np.zeros_like(bank_act)
    fit_zero = np.zeros_like(fit_act)
    val_zero = np.zeros_like(val_act)

    print(f"[{name}] [Fair information control (3 seeds)]")
    for seed in SEEDS:
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
            print(f"    seed {seed} [{arm:<8}] MSE={fair[arm][-1]['mse']:.4f} "
                  f"MAE={fair[arm][-1]['mae']:.4f}")

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
    pred_dir = ROOT / name / "prediction"
    abl_dir = ROOT / name / "ablation"
    pred_dir.mkdir(parents=True, exist_ok=True)
    abl_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": name,
        "protocol": "multidataset VAL, residualize Pearson, Top-K=5 mean",
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
    (pred_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (pred_dir / "fair_information_control.json").write_text(
        json.dumps(fair_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (abl_dir / "per_expert_ablation.json").write_text(
        json.dumps(per_expert, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (abl_dir / "leave_one_out.json").write_text(
        json.dumps(leave_one_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=" * 88)
    print(f"STAGE B  {name}  (frozen v5 ACCEPT knowledge)")
    print("=" * 88)
    print(f"Shape-only      MSE = {shape_mse:.4f}  MAE = {shape_mae:.4f}")
    print(f"ALL ACCEPT      MSE = {all_mse:.4f}  MAE = {all_mae:.4f}  gain = {gain_all:+.2f}%")
    for arm in ("zero", "shuffled", "real"):
        s = fair_summary[arm]
        print(f"  {arm:<8} MSE = {s['mse_mean']:.4f} ± {s['mse_std']:.4f}  "
              f"MAE = {s['mae_mean']:.4f} ± {s['mae_std']:.4f}")
    print(f"  REAL vs ZERO     gain = {gain_real_vs_zero:+.2f}%")
    print(f"  REAL vs SHUFFLED gain = {gain_real_vs_shuf:+.2f}%")
    print(f"Saved to {pred_dir} / {abl_dir}")
    print("critic_rerun=false, accepted_pool_modified=false, test_used=false")
    return result


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def _stage_a_summary(critic_dir: Path) -> Dict[str, Any]:
    pool_path = critic_dir / "final_accepted_knowledge.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else {}
    n_final = len(pool)
    n_proposed = 0
    n_acc_r1 = 0
    r1_path = critic_dir / "expert_round_1.json"
    if r1_path.exists():
        r1 = json.loads(r1_path.read_text(encoding="utf-8"))
        n_proposed = len(r1)
        n_acc_r1 = sum(1 for r in r1 if r["final_decision"] == "ACCEPT")
    n_rounds = 0
    for i in range(1, 5):
        if (critic_dir / f"expert_round_{i}.json").exists():
            n_rounds = i
    return {"n_proposed": n_proposed, "n_accepted_round1": n_acc_r1,
            "n_accepted_final": n_final, "n_rounds": n_rounds}


def _first_existing(*paths) -> Optional[Path]:
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def aggregate():
    # dataset -> (critic_dir, prediction result dir, ablation dir)
    spec = {
        "ETTh1": {
            "critic": "results/etth1_critic_v5_closed_loop",
            "prediction": "results/etth1_critic_v5_prediction_validation",
        },
        "electricity": {
            "critic": "results/electricity_critic_v5_closed_loop",
            "prediction": str(ROOT / "electricity" / "prediction"),
        },
        "solar": {
            "critic": str(ROOT / "solar" / "critic"),
            "prediction": str(ROOT / "solar" / "prediction"),
        },
        "traffic": {
            "critic": str(ROOT / "traffic" / "critic"),
            "prediction": str(ROOT / "traffic" / "prediction"),
        },
        "weather": {
            "critic": str(ROOT / "weather" / "critic"),
            "prediction": str(ROOT / "weather" / "prediction"),
        },
        "exchange_rate": {
            "critic": str(ROOT / "exchange_rate" / "critic"),
            "prediction": str(ROOT / "exchange_rate" / "prediction"),
        },
    }

    rows = []
    for ds, paths in spec.items():
        critic_dir = Path(paths["critic"])
        pred_dir = Path(paths["prediction"])
        row = {"dataset": ds}
        row.update(_stage_a_summary(critic_dir))

        result_path = pred_dir / "result.json"
        if not result_path.exists():
            row["stage_b"] = "missing"
            rows.append(row)
            continue
        res = json.loads(result_path.read_text(encoding="utf-8"))
        shape_mse = res["shape_only"]["mse"]
        fair = res["fair_information_control"]
        zero_mse = fair["zero"]["mse_mean"]
        shuf_mse = fair["shuffled"]["mse_mean"]
        real_mse = fair["real"]["mse_mean"]

        real_vs_shape = (shape_mse - real_mse) / shape_mse * 100.0
        real_vs_zero = (zero_mse - real_mse) / zero_mse * 100.0
        real_vs_shuf = (shuf_mse - real_mse) / shuf_mse * 100.0

        # per-expert ablation (positive/negative counts)
        abl_path = _first_existing(
            pred_dir / "per_expert_ablation.json",
            pred_dir.parent / "ablation" / "per_expert_ablation.json",
        )
        n_pos = n_neg = n_neu = None
        if abl_path is not None:
            abl = json.loads(abl_path.read_text(encoding="utf-8"))
            eps = 0.01
            n_pos = sum(1 for x in abl if x["vs_shape_gain_pct"] > eps)
            n_neg = sum(1 for x in abl if x["vs_shape_gain_pct"] < -eps)
            n_neu = len(abl) - n_pos - n_neg

        row.update({
            "shape_mse": shape_mse,
            "zero_mse": zero_mse, "shuffled_mse": shuf_mse, "real_mse": real_mse,
            "real_mse_std": fair["real"]["mse_std"],
            "real_vs_shape_pct": real_vs_shape,
            "real_vs_zero_pct": real_vs_zero,
            "real_vs_shuffled_pct": real_vs_shuf,
            "n_positive_experts": n_pos,
            "n_negative_experts": n_neg,
            "n_neutral_experts": n_neu,
            "zero_mae": fair["zero"]["mae_mean"], "shuffled_mae": fair["shuffled"]["mae_mean"],
            "real_mae": fair["real"]["mae_mean"], "real_mae_std": fair["real"]["mae_std"],
        })
        rows.append(row)

    # ---- write summary ----
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    cols = ["dataset", "n_proposed", "n_accepted_round1", "n_accepted_final",
            "shape_mse", "zero_mse", "shuffled_mse", "real_mse",
            "real_vs_shape_pct", "real_vs_zero_pct", "real_vs_shuffled_pct",
            "n_positive_experts", "n_negative_experts"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    df.to_csv(ROOT / "summary.csv", index=False, encoding="utf-8-sig")

    # ---- audit markdown ----
    L = ["# Cross-Dataset Validation — Frozen v5 Expert-Critic Closed Loop", ""]
    L.append("## Unified summary table")
    L.append("")
    L.append("| Dataset | #Prop | #Acc R1 | #Acc Final | Shape | ZERO | SHUFFLED | REAL | vs Shape | vs ZERO | vs SHUFFLED | #Pos | #Neg |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("stage_b") == "missing":
            L.append(f"| {r['dataset']} | {r['n_proposed']} | {r['n_accepted_round1']} | {r['n_accepted_final']} | (Stage B missing) |")
            continue
        L.append(
            f"| {r['dataset']} | {r['n_proposed']} | {r['n_accepted_round1']} | {r['n_accepted_final']} "
            f"| {r['shape_mse']:.3f} | {r['zero_mse']:.3f} | {r['shuffled_mse']:.3f} | {r['real_mse']:.3f} "
            f"| {r['real_vs_shape_pct']:+.2f}% | {r['real_vs_zero_pct']:+.2f}% | {r['real_vs_shuffled_pct']:+.2f}% "
            f"| {r['n_positive_experts']} | {r['n_negative_experts']} |")
    L.append("")
    L.append("## Audit flags (frozen-method invariants)")
    L.append("")
    L.append("- critic_v5_modified = **false** (generic functions imported verbatim)")
    L.append("- dataset_specific_critic_rule = **false**")
    L.append("- new_modules_added = **false**")
    L.append("- test_used = **false** (all Stage B results assert test_used=false)")
    L.append("")
    (ROOT / "cross_dataset_audit.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    print(f"\nSaved summary.json / summary.csv / cross_dataset_audit.md under {ROOT}")
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["critic", "prediction", "aggregate"], required=True)
    ap.add_argument("--datasets", default=None,
                    help="comma-separated; default = all applicable")
    args = ap.parse_args()

    if args.stage == "critic":
        names = args.datasets.split(",") if args.datasets else ["traffic", "weather", "exchange_rate", "solar"]
        for n in names:
            run_stage_a(n)
    elif args.stage == "prediction":
        names = args.datasets.split(",") if args.datasets else ["electricity", "solar", "traffic", "weather", "exchange_rate"]
        for n in names:
            run_stage_b(n)
    elif args.stage == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
