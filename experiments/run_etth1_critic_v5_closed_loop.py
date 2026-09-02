#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 -> frozen Agentic Closed-Loop v5 (cross-dataset migration).

Reuses the Electricity-frozen v5 closed loop **unchanged** on ETTh1, to verify the
loop is cross-dataset stable. The Critic / Claim Scope Analyzer / Evidence Planner /
Compiler / Proxy Validator / ACCEPT-UNSUPPORTED-REJECT / duplicate check /
knowledge-vs-compiler-failure classification are all imported verbatim from
`run_expert_agent_closed_loop` (NOT modified).

Only the dataset INPUTS are provided for ETTh1:
  - metadata: target = OT, domain = transformer_temperature (from config.json),
    hourly sampling, univariate (no covariate semantics are invented for
    HUFL/HULL/MUFL/MULL/LUFL/LULL — the project config does not define them).
  - feature catalog: OT/timestamp deterministic features only (no x_subset).
  - TRAIN-only evidence: computed from data/ETTh1.csv OT column, TRAIN = first 60%.

No `if dataset == etth1` branch. No ETTh1-specific critic rule. No domain mechanism
is hand-fed to the Expert (no season/trend/period/temperature/load hints) — the
Expert proposes from metadata + TRAIN summaries alone.

VAL unused. TEST never read. No prediction / MLP / Chronos / MSE.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# frozen generic v5 core (imported verbatim, NOT modified)
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
import run_agent_v2_electricity as av2          # _call_llm / _list_from / _default_model

av2.CACHE_DIR = "cache/llm_priors_closed_loop_v5_etth1"

MAX_ROUNDS = 3
OUT_DIR = Path("results/etth1_critic_v5_closed_loop")

# --------------------------------------------------------------------------- #
# ETTh1 dataset INPUTS
# --------------------------------------------------------------------------- #
# Metadata is taken from the project's config.json (domain="transformer_temperature",
# target="OT", frequency="1h"). HUFL..LULL are NOT given invented semantics.
ETTH1_METADATA = {
    "target_variable": "OT",
    "target_description": "transformer temperature (target series)",
    "domain": "transformer_temperature",
    "sampling_frequency": "hourly",
    "n_population": 1,           # univariate target (no aggregate-of-units population)
}

# Univariate OT + timestamp deterministic features (no x_subset covariates).
ETTH1_FEATURE_RULES = {
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


def etth1_evidence():
    df = pd.read_csv("data/ETTh1.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)          # == 10452 (frozen split [0, 10452))
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

    hourly_mean = np.array([float(y[hours == h].mean()) for h in range(24)])
    dow_mean = np.array([float(y[dows == d].mean()) for d in range(7)])

    # feature thresholds (TRAIN-derived, for the compiler)
    rng = np.random.default_rng(0)
    ctx = 64
    starts = rng.choice(train_len - ctx, size=min(3000, train_len - ctx), replace=False)
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
                       "total_rows": int(N), "sampling": "hourly"},
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
        "persistence": {"lag1_autocorr": round(_corr(y[1:], y[:-1]), 4),
                        "lag24h_autocorr": round(_corr(y[24:], y[:-24]), 4)},
        "variability": {"ot_mean": round(float(y.mean()), 2), "ot_std": round(float(y.std()), 2)},
        "feature_thresholds": feature_thresholds,
        "x_subset_representativeness": {},
    }


# --------------------------------------------------------------------------- #
# ETTh1 LLM prompts (generic hourly time-series; no domain mechanisms hand-fed)
# --------------------------------------------------------------------------- #
def _metadata_text():
    return (
        "DATASET:\n"
        f"- target variable: {ETTH1_METADATA['target_variable']} "
        f"({ETTH1_METADATA['target_description']})\n"
        f"- domain: {ETTH1_METADATA['domain']}\n"
        f"- sampling frequency: {ETTH1_METADATA['sampling_frequency']}\n"
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
        f"- lag1 autocorr {pv['lag1_autocorr']}, lag24h autocorr {pv['lag24h_autocorr']}\n"
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


_GEN_SYSTEM = (
    "You are a domain expert on forecasting a univariate hourly time series (target "
    "variable OT). A retrieval model forecasts the series by retrieving similar "
    "historical windows from a memory bank and averaging their futures. Propose "
    "CONDITIONAL expert knowledge: each statement says WHEN it applies, WHAT historical "
    "segments to prefer, and WHY. The activation condition must be observable from the "
    "query's own history and timestamp. Output strict JSON only."
)


def build_generation_prompt(ev):
    user = "\n\n".join([
        _metadata_text(),
        _train_summary_text(ev),
        _catalog_text(),
        _schema_text(),
        "TASK: propose 6-10 conditional expert-knowledge statements. Output only the JSON object.",
    ])
    return _GEN_SYSTEM, user


def build_revision_prompt(ev, feedbacks):
    user = "\n\n".join([
        _metadata_text(),
        _train_summary_text(ev),
        _catalog_text(),
        "CRITIC FEEDBACK ON YOUR PREVIOUS PROPOSALS (per expert):\n"
        + json.dumps(feedbacks, ensure_ascii=False, indent=2),
        _schema_text(),
        "TASK: For each failed expert, REVISE it to fix the specific failure(s) with "
        "TRAIN-supported evidence, or REPLACE it with a genuinely different mechanism. "
        "You may keep experts that are already fine. Output only the JSON object.",
    ])
    return _GEN_SYSTEM, user


# --------------------------------------------------------------------------- #
# closed-loop driver (identical logic to frozen v5, ETTh1 inputs)
# --------------------------------------------------------------------------- #
def run_closed_loop(evidence, metadata, feature_rules):
    model = av2._default_model()
    failed_history: List[Dict] = []
    all_rounds = []
    accepted_pool: Dict[str, Dict] = {}

    for rnd in range(1, MAX_ROUNDS + 1):
        if rnd == 1:
            sys_p, usr_p = build_generation_prompt(evidence)
            raw, path, cached = av2._call_llm(sys_p, usr_p, f"etth1_expert_r{rnd}", model)
            experts = av2._list_from(raw, "expert_knowledge")
        else:
            feedbacks = [r["feedback"] for prev in all_rounds for r in prev
                         if r["final_decision"] != "ACCEPT" and r["feedback"] and not r["duplicate_of"]]
            if not feedbacks:
                break
            sys_p, usr_p = build_revision_prompt(evidence, feedbacks)
            raw, path, cached = av2._call_llm(sys_p, usr_p, f"etth1_expert_r{rnd}", model)
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


def render_report(loop, evidence):
    L = []
    L.append("# ETTh1 — Agentic Closed-Loop v5 (frozen) Audit")
    L.append("")
    L.append("The Electricity-frozen v5 closed loop is reused **unchanged** on ETTh1.")
    L.append("Only dataset inputs (metadata, feature catalog, TRAIN evidence) were swapped.")
    L.append("No Critic rule modified; no ETTh1-specific hard-code.")
    L.append("")
    for i, rnd in enumerate(loop["rounds"], 1):
        L.append(f"## Round {i}")
        L.append("")
        L.append("| ID | expert | decision | failure kind | reason |")
        L.append("|---|---|---|---|---|")
        for r in rnd:
            reason = "; ".join(r["failure_reasons"]) if r["failure_reasons"] else "all checks pass"
            dup = f" (dup of {r['duplicate_of']})" if r.get("duplicate_of") else ""
            L.append(f"| {r['expert_id']} | {r['name']} | **{r['final_decision']}** | "
                     f"{r.get('failure_kind') or '-'}{dup} | {reason} |")
        L.append("")
    L.append("## Final accepted knowledge pool")
    L.append("")
    if loop["accepted_pool"]:
        for eid, r in loop["accepted_pool"].items():
            L.append(f"- **{eid}** {r['name']}: {r['claim']}")
    else:
        L.append("- (empty)")
    L.append("")
    L.append("## Cross-dataset audit")
    L.append("")
    L.append("- critic_v5_modified = **false** (generic functions imported verbatim)")
    L.append("- etth1_specific_hardcode = **false**")
    L.append("- compound_condition_supported = **true**")
    L.append(f"- feedback_regeneration_used = **{str(len(loop['rounds']) > 1).lower()}**")
    L.append("- train_only_evidence = **true**; val_used = false; test_used = false; "
             "prediction_experiment_run = false")
    L.append("")
    return "\n".join(L)


def main():
    evidence = etth1_evidence()
    loop = run_closed_loop(evidence, ETTH1_METADATA, ETTH1_FEATURE_RULES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, rnd in enumerate(loop["rounds"], 1):
        (OUT_DIR / f"expert_round_{i}.json").write_text(
            json.dumps(rnd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (OUT_DIR / f"critic_round_{i}.json").write_text(
            json.dumps([{k: r[k] for k in ("expert_id", "name", "final_decision",
                                           "failure_reasons", "failure_kind", "checks")}
                        for r in rnd], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (OUT_DIR / f"feedback_round_{i}.json").write_text(
            json.dumps([r["feedback"] for r in rnd if r.get("feedback")],
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    (OUT_DIR / "compiler_audit.json").write_text(
        json.dumps([{"expert_id": r["expert_id"], "compiled": r["compiled"],
                     "proxy_validation": r["proxy_validation"]}
                    for rnd in loop["rounds"] for r in rnd],
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "proxy_validation.json").write_text(
        json.dumps([r["proxy_validation"] for rnd in loop["rounds"] for r in rnd],
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "final_accepted_knowledge.json").write_text(
        json.dumps(loop["accepted_pool"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "closed_loop_audit.md").write_text(render_report(loop, evidence), encoding="utf-8")

    print("=" * 90)
    print("ETTH1 CLOSED LOOP (frozen v5, cross-dataset)")
    print("=" * 90)
    print(f"TRAIN: {evidence['train_span']['start']} -> {evidence['train_span']['end']} "
          f"({evidence['train_span']['n_rows']} rows, hourly)")
    print(f"coverage: {len(evidence['temporal_coverage']['months_present'])}/12 months, "
          f"{len(evidence['temporal_coverage']['hours_present'])}/24 hours, "
          f"{len(evidence['temporal_coverage']['dows_present'])}/7 days")
    print()
    for i, rnd in enumerate(loop["rounds"], 1):
        n_acc = sum(1 for r in rnd if r["final_decision"] == "ACCEPT")
        n_uns = sum(1 for r in rnd if r["final_decision"] == "UNSUPPORTED")
        n_rej = sum(1 for r in rnd if r["final_decision"] == "REJECT")
        n_kf = sum(1 for r in rnd if r.get("failure_kind") == "knowledge_failure")
        n_cf = sum(1 for r in rnd if r.get("failure_kind") == "compiler_failure")
        print(f"Round {i}: {len(rnd)} experts -> {n_acc} ACCEPT, {n_uns} UNSUPPORTED, "
              f"{n_rej} REJECT | knowledge_failure={n_kf} compiler_failure={n_cf}")
        for r in rnd:
            reason = "; ".join(r["failure_reasons"]) if r["failure_reasons"] else "-"
            print(f"    {r['expert_id']:<4}{r['name']:<30}{r['final_decision']:<12}"
                  f"{r.get('failure_kind') or '-':<18}{reason[:80]}")
    print()
    print("Flags:")
    print("  critic_v5_modified = false")
    print("  closed_loop_implemented = true")
    print("  feedback_regeneration_used =", len(loop["rounds"]) > 1)
    print("  compound_condition_supported = true")
    print("  dataset_specific_critic_rule = false")
    print("  etth1_specific_hardcode = false")
    print("  train_only_evidence = true; val_used = false; test_used = false; "
          "prediction_experiment_run = false")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
