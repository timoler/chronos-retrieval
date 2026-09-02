#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Electricity Expert → Critic v4 (frozen) knowledge-certification run.

Goal: verify whether the Solar-frozen Critic v4 framework still works when pointed
at the Electricity dataset, WITHOUT modifying any judgment rule.

Pipeline (identical to Solar, imported verbatim):
    Expert Knowledge
      -> Claim Scope Analyzer
      -> Evidence Planner
      -> TRAIN-only Evidence Executor
      -> Arbitrary Fallback Detector
      -> Evidence Sufficiency Critic (conservative AND-gate A..H)
      -> ACCEPT / UNSUPPORTED / REJECT

FROZEN (imported from run_expert_knowledge_critic, byte-identical, NOT edited):
    analyze_claim_scope / plan_evidence / sufficiency_critic / decide /
    detect_arbitrary_fallback / critique_expert + all concept vocabularies
    (annual/diurnal/weekly, unobserved-driver, population, leakage).

SWAPPED INPUTS (dataset metadata / feature catalog — these are the same *kind* of
INPUTS the Solar script declares as inputs; the judgment rules are untouched):
    N_POPULATION         137 -> 314
    LOAD_VARS            P001..P006 -> HUFL..LULL  (X1..X6 = MT_001..MT_006)
    ACTIVATION_CATALOG   solar features -> electricity features
    FEATURE_DOMAIN       solar domains  -> electricity domains

No `if dataset == electricity` branch is added anywhere. No rule is injected that
tells the critic X1-X6 are non-representative. Only dataset facts are supplied
(Y = aggregate of 314 consumers; X1-X6 = first 6 of those 314; no representativeness
guarantee), and the critic must reach its own verdict.

No prediction / MLP / Chronos / retrieval / Attention. VAL unused. TEST never read.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_expert_knowledge_critic as kc       # frozen Critic v4 (generic judgment)
import run_agent_v2_electricity as av2          # electricity metadata + _call_llm / _list_from

# --------------------------------------------------------------------------- #
# Electricity dataset INPUTS (metadata + activation catalog + feature domain).
# These are inputs, not judgment rules.
# --------------------------------------------------------------------------- #
N_POPULATION = 314
LOAD_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]   # X1..X6 = MT_001..MT_006

ELECTRICITY_METADATA = av2.ELECTRICITY_METADATA_V2

ELECTRICITY_ACTIVATION_CATALOG = [
    "hour_of_day",            # current hour-of-day load phase (TRAIN hourly means)
    "day_of_week",            # current day-of-week phase
    "working_day",            # working vs non-working day
    "seasonal_level",         # current month-of-year phase (TRAIN monthly means)
    "recent_level",           # current level relative to typical (z-scored context mean)
    "recent_slope_magnitude", # recent trend strength
    "recent_diff_magnitude",  # recent point-to-point change magnitude
    "recent_volatility",      # recent variability regime
    "recent_autocorr",        # recent persistence strength
    "x_subset_level",         # X1-X6 aggregate level (z-scored)
    "x_subset_change",        # X1-X6 aggregate point-to-point change (z-scored)
]

ELECTRICITY_FEATURE_DOMAIN = {
    "hour_of_day": {"kind": "calendar_categorical", "unit": "hour", "n": 24},
    "day_of_week": {"kind": "calendar_categorical", "unit": "dow", "n": 7},
    "working_day": {"kind": "calendar_categorical", "unit": "dow", "n": 7},
    "seasonal_level": {"kind": "calendar_categorical", "unit": "month", "n": 12},
    "recent_level": {"kind": "continuous_zscore"},
    "recent_slope_magnitude": {"kind": "continuous_zscore"},
    "recent_diff_magnitude": {"kind": "continuous_zscore"},
    "recent_volatility": {"kind": "continuous_zscore"},
    "recent_autocorr": {"kind": "continuous_bounded_0_1"},
    "x_subset_level": {"kind": "continuous_zscore"},
    "x_subset_change": {"kind": "continuous_zscore"},
}

# Point the frozen critic's input globals at the electricity values. These are the
# same input constants the Solar critic declares; re-pointing them is supplying
# electricity inputs, NOT editing any judgment function.
kc.N_POPULATION = N_POPULATION
kc.ACTIVATION_CATALOG = ELECTRICITY_ACTIVATION_CATALOG
kc.FEATURE_DOMAIN = ELECTRICITY_FEATURE_DOMAIN

# isolated LLM cache so we never reuse a Solar-cache entry by accident
av2.CACHE_DIR = "cache/llm_priors_critic_v4_electricity"

OUT_DIR = Path("results/electricity_critic_v4")


# --------------------------------------------------------------------------- #
# TRAIN-only data access + evidence executor (electricity; same schema as Solar)
# --------------------------------------------------------------------------- #
def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _eta2(groups, y) -> float:
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=np.float64)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    within = 0.0
    for g in np.unique(groups):
        yg = y[groups == g]
        within += float(((yg - yg.mean()) ** 2).sum())
    return 1.0 - within / total


def load_train():
    df = pd.read_csv("data/electricity.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)          # == 84153, matches the frozen split [0, 84153)
    dt = dates.iloc[:train_len]
    y = df["OT"].to_numpy(np.float64)[:train_len]
    plants = {v: df[v].to_numpy(np.float64)[:train_len] for v in LOAD_VARS}
    return df, dates, dt, y, plants, N, train_len


def compute_train_evidence():
    """Deterministic TRAIN-only evidence (electricity), same general schema."""
    _, _, dt, y, plants, N, train_len = load_train()

    months = dt.dt.month.to_numpy()
    hours = dt.dt.hour.to_numpy()
    dows = dt.dt.dayofweek.to_numpy()

    months_present = sorted(int(m) for m in set(months.tolist()))
    hours_present = sorted(int(h) for h in set(hours.tolist()))
    dows_present = sorted(int(d) for d in set(dows.tolist()))

    monthly_mean = {int(m): float(y[months == m].mean()) for m in months_present}
    monthly_daily_peak = {}
    for m in months_present:
        mm = months == m
        hm = np.array([y[mm & (hours == h)].mean() for h in range(24)])
        monthly_daily_peak[m] = float(np.nanmax(hm)) if hm.size else float("nan")

    hourly_mean = np.array([float(y[hours == h].mean()) for h in range(24)])
    peak_hour = int(np.argmax(hourly_mean))

    # electricity load is never zero at night (people still draw power)
    night_mask = (hours >= 19) | (hours < 6)
    night_zero_frac = float((y[night_mask] == 0).mean()) if night_mask.sum() else float("nan")

    # month -> month direction (reversal detection, identical logic to Solar)
    mvals = [monthly_mean[m] for m in months_present]
    month_direction = {}
    n_reversals = 0
    for i in range(1, len(mvals)):
        d = "up" if mvals[i] > mvals[i - 1] else ("down" if mvals[i] < mvals[i - 1] else "flat")
        month_direction[f"{months_present[i-1]}->{months_present[i]}"] = d
        if i >= 2 and month_direction[f"{months_present[i-2]}->{months_present[i-1]}"] != d \
                and month_direction[f"{months_present[i-2]}->{months_present[i-1]}"] != "flat" \
                and d != "flat":
            n_reversals += 1

    s6 = np.sum([plants[v] for v in LOAD_VARS], axis=0)
    lag1 = _corr(y[1:], y[:-1])
    lag96 = _corr(y[96:], y[:-96])   # 24h at 15-min resolution

    # X1-X6 vs Y representativeness (informational counter evidence)
    corr_s6_y_by_month = {int(m): round(_corr(s6[months == m], y[months == m]), 4)
                          for m in months_present}
    corr_s6_y_by_tod = {}
    for b in range(4):
        mask = (hours // 6) == b
        corr_s6_y_by_tod[f"{(b*6):02d}-{((b+1)*6)%24:02d}h"] = round(
            _corr(s6[mask], y[mask]), 4)

    return {
        "train_span": {
            "start": str(dt.min()), "end": str(dt.max()),
            "n_days": int(dt.dt.date.nunique()), "n_rows": int(train_len),
            "total_rows": int(N), "sampling": "15 minutes",
        },
        "temporal_coverage": {
            "months_present": months_present,
            "months_missing": [m for m in range(1, 13) if m not in months_present],
            "month_fraction": round(len(months_present) / 12, 4),
            "hours_present": hours_present,
            "hours_missing": [h for h in range(24) if h not in hours_present],
            "hour_fraction": round(len(hours_present) / 24, 4),
            "dows_present": dows_present,
            "dow_fraction": round(len(dows_present) / 7, 4),
        },
        "diurnal": {
            "hourly_mean": [round(float(v), 2) for v in hourly_mean],
            "peak_hour": peak_hour,
            "night_zero_fraction": round(night_zero_frac, 4),
            "eta2_hour_on_ot": round(_eta2(hours, y), 4),
        },
        "seasonal": {
            "monthly_mean": {k: round(v, 2) for k, v in monthly_mean.items()},
            "monthly_daily_peak": {k: round(v, 2) for k, v in monthly_daily_peak.items()},
            "month_direction": month_direction,
            "n_month_direction_reversals": n_reversals,
            "eta2_month_on_ot": round(_eta2(months, y), 4),
        },
        "persistence": {
            "lag1_autocorr": round(lag1, 4),
            "lag96_autocorr_24h": round(lag96, 4),
        },
        "population": {
            "n_proxy": len(LOAD_VARS),
            "n_population": N_POPULATION,
            "proxy_fraction": round(len(LOAD_VARS) / N_POPULATION, 4),
            "corr_proxy_sum_vs_target": round(_corr(s6, y), 4),
        },
        "variability": {
            "ot_mean": round(float(y.mean()), 2),
            "ot_std": round(float(y.std()), 2),
            "diff_abs_mean": round(float(np.abs(np.diff(y)).mean()), 4),
            "diff_std": round(float(np.diff(y).std()), 4),
        },
        # informational counter evidence (not consumed as a global per-expert check)
        "x_subset_representativeness": {
            "corr_proxy_sum_vs_target": round(_corr(s6, y), 4),
            "diff_corr": round(_corr(np.diff(s6), np.diff(y)), 4),
            "proxy_sum_target_ratio_mean": round(float((s6 / y).mean()), 5),
            "corr_by_month": corr_s6_y_by_month,
            "corr_by_time_of_day_6h": corr_s6_y_by_tod,
        },
    }


def _train_summary_text(ev) -> str:
    ds = ev["diurnal"]
    se = ev["seasonal"]
    pv = ev["persistence"]
    pop = ev["population"]
    xr = ev["x_subset_representativeness"]
    return (
        "TRAIN-ONLY DATA SUMMARY (deterministic):\n"
        f"- train span: {ev['train_span']['start']} -> {ev['train_span']['end']} "
        f"({ev['train_span']['n_days']} days, {ev['train_span']['sampling']})\n"
        f"- temporal coverage: {len(ev['temporal_coverage']['months_present'])}/12 months, "
        f"{len(ev['temporal_coverage']['hours_present'])}/24 hours, "
        f"{len(ev['temporal_coverage']['dows_present'])}/7 days present\n"
        f"- hourly mean OT: peak hour {ds['peak_hour']} ({ds['hourly_mean'][ds['peak_hour']]}), "
        f"night zero fraction {ds['night_zero_fraction']}\n"
        f"- eta2(hour,OT)={ds['eta2_hour_on_ot']}, eta2(month,OT)={se['eta2_month_on_ot']}\n"
        f"- monthly mean OT: {se['monthly_mean']}\n"
        f"- lag1 autocorr {pv['lag1_autocorr']}, lag96(24h) autocorr {pv['lag96_autocorr_24h']}\n"
        f"- OT mean {ev['variability']['ot_mean']}, OT std {ev['variability']['ot_std']}\n"
        f"- sum(X1..X6) vs Y: corr {xr['corr_proxy_sum_vs_target']}, "
        f"diff corr {xr['diff_corr']}, ratio mean {xr['proxy_sum_target_ratio_mean']}, "
        f"coverage {pop['proxy_fraction']}\n"
        f"- corr(sum X1..X6, Y) by month: {xr['corr_by_month']}\n"
        f"- corr(sum X1..X6, Y) by 6h block: {xr['corr_by_time_of_day_6h']}\n"
    )


# --------------------------------------------------------------------------- #
# Expert Generator (LLM) — electricity domain expert proposes candidates
# --------------------------------------------------------------------------- #
_GEN_SYSTEM = (
    "You are a domain expert on aggregate electricity load forecasting. A retrieval "
    "model forecasts a univariate electricity load series (OT, 15-minute resolution) by "
    "retrieving similar historical windows from a memory bank and averaging their "
    "futures.\n\n"
    "You propose CONDITIONAL expert knowledge: each statement must say WHEN it applies "
    "(based on the current query's observable state), WHAT historical segments should "
    "then be preferred, and WHY. Do NOT propose unconditional statements like "
    "'electricity load persists' — always attach an activation condition.\n\n"
    "The activation condition must be observable from the query's own history and "
    "timestamp, or from the observed variables listed in the metadata. Output strict "
    "JSON only."
)


def build_expert_generation_prompt(ev):
    catalog_text = (
        "AVAILABLE ACTIVATION FEATURES (choose observable_features ONLY from this list, "
        "1-2 items):\n"
        "  - hour_of_day: current hour-of-day load phase (TRAIN hourly means)\n"
        "  - day_of_week: current day-of-week phase\n"
        "  - working_day: working vs non-working day\n"
        "  - seasonal_level: current month-of-year phase (TRAIN monthly means)\n"
        "  - recent_level: current load level relative to typical (z-scored context mean)\n"
        "  - recent_slope_magnitude: recent trend strength\n"
        "  - recent_diff_magnitude: recent point-to-point change magnitude\n"
        "  - recent_volatility: recent variability regime\n"
        "  - recent_autocorr: recent persistence strength\n"
        "  - x_subset_level: aggregate level of X1-X6 (z-scored)\n"
        "  - x_subset_change: aggregate point-to-point change of X1-X6 (z-scored)"
    )
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"expert_knowledge": [ ... ]} with 6 to 10 entries. Each entry:\n'
        '  - "expert_id": "E1", "E2", ...\n'
        '  - "name": short label\n'
        '  - "claim": one conditional sentence: "When [observable condition], historical '
        "segments satisfying [relationship] should be preferred because [reason]\"\n"
        '  - "activation_condition": the "when [observable state]" part, natural language\n'
        '  - "observable_features": list of 1-2 features from the AVAILABLE ACTIVATION FEATURES list\n'
        '  - "expected_relationship": the "prefer [relationship]" part\n'
        '  - "reason": why this relationship helps retrieval\n'
        '  - "domain_fact": a 1-sentence mechanism (same idea as claim, for review)\n'
        '  - "latent_state": the hidden predictive state\n'
        '  - "observable_proxy": the observable quantity proxying the state\n'
        '  - "expected_future_relationship": expected future behavior\n'
    )
    user = "\n\n".join([
        av2._metadata_text(),
        _train_summary_text(ev),
        catalog_text,
        "DOMAIN FACTS (physical mechanisms, for grounding):\n" + av2.ELECTRICITY_EXPERT_MECHANISM,
        schema,
        "TASK: propose 6-10 conditional expert-knowledge statements. Output only the JSON object.",
    ])
    return _GEN_SYSTEM, user


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    evidence = compute_train_evidence()

    # ---- Expert generation (LLM) ----
    gen_sys, gen_user = build_expert_generation_prompt(evidence)
    model = av2._default_model()
    raw, cache_path, cached = av2._call_llm(gen_sys, gen_user, "electricity_expert", model)
    experts = av2._list_from(raw, "expert_knowledge")
    print(f"Expert Generator produced {len(experts)} candidates "
          f"(cache={'hit' if cached else 'miss'} {cache_path})")

    # map to the schema the frozen critic consumes (id + knowledge alias of claim)
    for i, e in enumerate(experts, 1):
        e.setdefault("expert_id", f"E{i}")
        e["id"] = e.get("expert_id", e.get("name"))
        e["knowledge"] = e.get("claim", e.get("knowledge", ""))

    # ---- frozen Critic v4 on each expert ----
    results = [kc.critique_expert(e, evidence, metadata=ELECTRICITY_METADATA)
               for e in experts]

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "expert_candidates.json").write_text(
        json.dumps(experts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    scope_analysis = {
        e["id"]: {
            "name": e.get("name"),
            "claim": e.get("claim"),
            "claimed_scope": next(r for r in results if r["expert_id"] == e["id"])["claim_scope"],
            "required_coverage": next(r for r in results if r["expert_id"] == e["id"])["required_coverage"],
            "potential_failure_modes": next(r for r in results if r["expert_id"] == e["id"])["potential_failure_modes"],
        }
        for e in experts
    }
    (OUT_DIR / "scope_analysis.json").write_text(
        json.dumps(scope_analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    evidence_execution = {
        "train_span": evidence["train_span"],
        "temporal_coverage": evidence["temporal_coverage"],
        "diurnal": evidence["diurnal"],
        "seasonal": evidence["seasonal"],
        "persistence": evidence["persistence"],
        "population": evidence["population"],
        "variability": evidence["variability"],
        "x_subset_representativeness": evidence["x_subset_representativeness"],
        "per_expert_executed_evidence": {
            r["expert_id"]: r["executed_evidence"] for r in results
        },
    }
    (OUT_DIR / "evidence_execution.json").write_text(
        json.dumps(evidence_execution, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = {
        "n_experts": len(results),
        "n_accept": sum(1 for r in results if r["final_decision"] == "ACCEPT"),
        "n_unsupported": sum(1 for r in results if r["final_decision"] == "UNSUPPORTED"),
        "n_reject": sum(1 for r in results if r["final_decision"] == "REJECT"),
    }
    false_accepts = []
    for r in results:
        if r["final_decision"] == "ACCEPT":
            bad = [c for c in kc.CRITICAL_CHECKS
                   if r[{ "support":"support_sufficiency","coverage":"coverage_sufficiency",
                          "stability":"stability_sufficiency","counter":"counter_evidence",
                          "observability":"observability","executability":"future_executability",
                          "population":"population_validity","proxy_fidelity":"proxy_fidelity"}[c]]["status"] != "PASS"]
            if bad:
                false_accepts.append(r["name"])

    audit = {
        "dataset": "electricity",
        "critic_version": "v4 (expert_knowledge_critic, frozen)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_span": evidence["train_span"],
        "temporal_coverage": evidence["temporal_coverage"],
        "population": evidence["population"],
        "x_subset_representativeness": evidence["x_subset_representativeness"],
        "experts": results,
        "false_accepts": false_accepts,
        "summary": summary,
        # final confirmation flags (section 十五.10)
        "critic_modified": False,
        "electricity_specific_rule": False,
        "val_used": False,
        "test_used": False,
        "prediction_experiment_run": False,
        "expert_generator": {
            "system": gen_sys,
            "user": gen_user,
            "raw_response": raw,
            "cache_path": cache_path,
            "from_cache": cached,
            "model": model,
        },
    }
    (OUT_DIR / "critic_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "critic_audit.md").write_text(
        render_report(results, evidence, summary, false_accepts), encoding="utf-8")

    # ---- console report (10-item summary) ----
    print("\n" + "=" * 90)
    print("ELECTRICITY EXPERT -> CRITIC v4 (frozen)")
    print("=" * 90)
    print(f"TRAIN: {evidence['train_span']['start']} -> {evidence['train_span']['end']} "
          f"({evidence['train_span']['n_rows']} rows, {evidence['train_span']['sampling']})")
    print(f"coverage: {len(evidence['temporal_coverage']['months_present'])}/12 months, "
          f"{len(evidence['temporal_coverage']['hours_present'])}/24 hours, "
          f"{len(evidence['temporal_coverage']['dows_present'])}/7 days")
    print(f"population: {evidence['population']['n_proxy']}/{evidence['population']['n_population']} "
          f"= {evidence['population']['proxy_fraction']:.0%}, "
          f"corr(sum X1-X6, Y)={evidence['population']['corr_proxy_sum_vs_target']}")
    print()
    print(f"{'ID':<5}{'expert':<30}{'decision':<13}reason")
    print("-" * 90)
    for r in results:
        reason = "; ".join(r["failure_reasons"]) if r["failure_reasons"] else "all checks pass"
        print(f"{r['expert_id']:<5}{r['name']:<30}{r['final_decision']:<13}{reason}")
    print()
    print(f"Final: {summary['n_accept']} ACCEPT, {summary['n_unsupported']} UNSUPPORTED, "
          f"{summary['n_reject']} REJECT (among {summary['n_experts']} experts)")
    print(f"False Accept (ACCEPT with a FAILed critical check): {len(false_accepts)}")
    print()
    print("Guardrails: critic_modified=false, electricity_specific_rule=false, "
          "val_used=false, test_used=false, prediction_experiment_run=false")
    print(f"Saved to {OUT_DIR}")
    return audit


def render_report(results, evidence, summary, false_accepts):
    L = []
    L.append("# Electricity Expert → Critic v4 (frozen) — Knowledge Certification")
    L.append("")
    L.append("This run reuses the Solar-frozen Critic v4 framework **unchanged**. Only the")
    L.append("dataset inputs (metadata, feature catalog, TRAIN evidence) were swapped for")
    L.append("Electricity. No judgment rule was modified; no `if dataset == electricity`")
    L.append("branch was added; no rule was injected telling the critic that X1-X6 are")
    L.append("non-representative.")
    L.append("")
    L.append("## 1. Dataset facts supplied (metadata only, not verdicts)")
    L.append("")
    L.append("```json")
    L.append(json.dumps(av2.ELECTRICITY_METADATA_V2, ensure_ascii=False, indent=2))
    L.append("```")
    L.append("")
    L.append(f"- TRAIN span: {evidence['train_span']['start']} -> {evidence['train_span']['end']}")
    L.append(f"- Temporal coverage: {len(evidence['temporal_coverage']['months_present'])}/12 months, "
             f"{len(evidence['temporal_coverage']['hours_present'])}/24 hours, "
             f"{len(evidence['temporal_coverage']['dows_present'])}/7 days")
    L.append(f"- Population: {evidence['population']['n_proxy']}/{evidence['population']['n_population']} "
             f"({evidence['population']['proxy_fraction']:.0%}), "
             f"corr(sum X1-X6, Y) = {evidence['population']['corr_proxy_sum_vs_target']}")
    L.append("")
    L.append("## 2. Final decisions")
    L.append("")
    L.append("| ID | expert | decision | reason |")
    L.append("|---|---|---|---|")
    for r in results:
        reason = ("; ".join(r["failure_reasons"]) if r["failure_reasons"]
                  else "all critical checks pass")
        L.append(f"| {r['expert_id']} | {r['name']} | **{r['final_decision']}** | {reason} |")
    L.append("")
    L.append(f"Counts: {summary['n_accept']} ACCEPT, {summary['n_unsupported']} UNSUPPORTED, "
             f"{summary['n_reject']} REJECT.")
    L.append("")
    L.append("## 3. Per-expert evidence chain")
    L.append("")
    for r in results:
        L.append(f"### {r['expert_id']} — {r['name']}  ->  {r['final_decision']}")
        L.append("")
        L.append(f"- **claimed_scope**: `{json.dumps(r['claim_scope'], ensure_ascii=False)}`")
        L.append(f"- **required_coverage**: `{json.dumps(r['required_coverage'], ensure_ascii=False)}`")
        L.append(f"- **potential_failure_modes**: `{json.dumps(r['potential_failure_modes'], ensure_ascii=False)}`")
        for chk in kc.CRITICAL_CHECKS:
            key = {"support": "support_sufficiency", "coverage": "coverage_sufficiency",
                   "stability": "stability_sufficiency", "counter": "counter_evidence",
                   "observability": "observability", "executability": "future_executability",
                   "population": "population_validity", "proxy_fidelity": "proxy_fidelity"}[chk]
            c = r[key]
            L.append(f"- **{chk}**: `{c['status']}` — {c['note']}")
        L.append(f"- **failure_reasons**: `{json.dumps(r['failure_reasons'], ensure_ascii=False)}`")
        L.append("")
    L.append("## 4. Guardrails")
    L.append("")
    L.append("- critic_modified = **false** (judgment functions imported verbatim)")
    L.append("- electricity_specific_rule = **false**")
    L.append("- val_used = **false**")
    L.append("- test_used = **false**")
    L.append("- prediction_experiment_run = **false**")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
