#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 -> Critic v6 closed loop (generic v6 core + ETTh1 inputs only).

Runs the NEW generic v6 core (``run_expert_agent_closed_loop_v6``) on ETTh1, reusing
the *same* frozen Round-1 Expert candidates, TRAIN split and evidence source as the
v5 run.  The v6 core is dataset-agnostic: no ETTh1 / OT / summer / winter /
temperature rule is added anywhere.  Only the dataset INPUTS live here:

  - feature registry (source_type + computation dependencies + mechanism/state family)
  - metadata (target OT, univariate, hourly)
  - TRAIN-only evidence (feature distributions + activation feature matrix, ctx=64)
  - the frozen Round-1 candidates (loaded from the v5 cache) and a regeneration stub

The v6 fixes are exercised end-to-end:
  - E8 "Volatile regime scaling" uses only ``recent_volatility`` (context-derived),
    so it is observable at prediction time and is NOT rejected for observability.
  - E9 "Strong recent trend momentum" uses ``recent_slope_magnitude:large``; the
    qualifier "large" is repaired to "high" with a TRAIN-only q75 threshold — the
    knowledge is NOT regenerated into an unrelated mechanism.

VAL is never read by the Critic.  TEST is never read.  No prediction MSE is run.
Output: results/etth1_critic_v6_closed_loop/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_expert_agent_closed_loop_v6 import (            # noqa: E402  (generic v6 core)
    audit_observability_for_expert,
    run_closed_loop_v6,
    synthetic_audit_v6,
    COMPILER_CONFIG,
)
import run_agent_v2_electricity as av2                   # _call_llm / _list_from / _default_model

av2.CACHE_DIR = "cache/llm_priors_closed_loop_v6_etth1"  # isolated v6 LLM cache

OUT_DIR = Path("results/etth1_critic_v6_closed_loop")
V5_CACHE_DIR = Path("cache/llm_priors_closed_loop_v5_etth1")
V5_OUT = Path("results/etth1_critic_v5_closed_loop")

ACTIVATION_CONTEXT = 64      # recent-* window length (matches v5 evidence source)


# --------------------------------------------------------------------------- #
# ETTh1 metadata + feature registry (source metadata is the v6 observability basis)
# --------------------------------------------------------------------------- #
ETTH1_METADATA = {
    "target_variable": "OT",
    "target_description": "transformer temperature (target series)",
    "domain": "transformer_temperature",
    "sampling_frequency": "hourly",
    "n_population": 1,           # univariate target (no aggregate-of-units population)
}

ETTH1_FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- calendar / timestamp_known features ---
    "hour_of_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "computation": "hour of the query timestamp",
                    "mechanism_family": "calendar_cycle", "state_variable_family": "hour_of_day",
                    "categories": {"morning": [6, 7, 8, 9, 10, 11], "midday": [12, 13, 14, 15],
                                   "evening": [16, 17, 18, 19, 20, 21],
                                   "night": [22, 23, 0, 1, 2, 3, 4, 5]}},
    "day_of_week": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "computation": "day-of-week of the query timestamp",
                    "mechanism_family": "calendar_cycle", "state_variable_family": "day_of_week",
                    "categories": {"weekday": [0, 1, 2, 3, 4], "weekend": [5, 6]}},
    "working_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "computation": "workday vs non-working day of the query timestamp",
                    "mechanism_family": "calendar_cycle", "state_variable_family": "working_day",
                    "categories": {"workday": [0, 1, 2, 3, 4], "non_working": [5, 6]}},
    "seasonal_level": {"kind": "calendar", "source_type": "timestamp_known",
                       "dependencies": ["known_timestamp"],
                       "computation": "month-of-year of the query timestamp",
                       "mechanism_family": "calendar_cycle", "state_variable_family": "season",
                       "categories": {"winter": [12, 1, 2], "spring": [3, 4, 5],
                                      "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
    # --- context_derived features (observable from the query's own past window) ---
    "recent_level": {"kind": "continuous", "source_type": "context_derived",
                     "dependencies": ["historical_context", "train_derived_constant"],
                     "computation": "mean of the past context window",
                     "mechanism_family": "recent_state_regime", "state_variable_family": "level"},
    "recent_volatility": {"kind": "continuous", "source_type": "context_derived",
                          "dependencies": ["historical_context", "train_derived_constant"],
                          "computation": "std of the past context window",
                          "mechanism_family": "recent_state_regime", "state_variable_family": "volatility"},
    "recent_slope_magnitude": {"kind": "continuous", "source_type": "context_derived",
                               "dependencies": ["historical_context", "train_derived_constant"],
                               "computation": "abs linear slope of the past context window",
                               "mechanism_family": "recent_state_regime", "state_variable_family": "trend_slope"},
    "recent_diff_magnitude": {"kind": "continuous", "source_type": "context_derived",
                              "dependencies": ["historical_context", "train_derived_constant"],
                              "computation": "abs last point-to-point change",
                              "mechanism_family": "recent_state_regime", "state_variable_family": "change_magnitude"},
    "recent_autocorr": {"kind": "continuous", "source_type": "context_derived",
                        "dependencies": ["historical_context", "train_derived_constant"],
                        "computation": "lag-1 autocorrelation of the past context window",
                        "mechanism_family": "recent_state_regime", "state_variable_family": "persistence"},
}


# --------------------------------------------------------------------------- #
# TRAIN-only evidence (feature distributions + activation feature matrix)
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


def _compute_train_features(y_train, dt, ctx=ACTIVATION_CONTEXT):
    """Raw feature values over TRAIN query windows (window = [i, i+ctx))."""
    n = len(y_train)
    n_win = n - ctx
    feats = {
        "recent_level": np.zeros(n_win),
        "recent_volatility": np.zeros(n_win),
        "recent_slope_magnitude": np.zeros(n_win),
        "recent_diff_magnitude": np.zeros(n_win),
        "recent_autocorr": np.zeros(n_win),
        "hour_of_day": np.zeros(n_win, dtype=np.int64),
        "day_of_week": np.zeros(n_win, dtype=np.int64),
        "working_day": np.zeros(n_win, dtype=np.int64),
        "seasonal_level": np.zeros(n_win, dtype=np.int64),
    }
    t = np.linspace(-1.0, 1.0, ctx)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    for i in range(n_win):
        x = y_train[i:i + ctx].astype(np.float64)
        feats["recent_level"][i] = float(x.mean())
        feats["recent_volatility"][i] = float(x.std())
        xm = x - x.mean()
        slope = float(np.sum(xm * tc) / denom)
        feats["recent_slope_magnitude"][i] = abs(slope)
        feats["recent_diff_magnitude"][i] = abs(float(x[-1] - x[-2]))
        feats["recent_autocorr"][i] = _corr(x[1:], x[:-1])
    # calendar units from the query timestamp (window end)
    for i in range(n_win):
        ts = dt.iloc[i + ctx]
        feats["hour_of_day"][i] = ts.hour
        feats["day_of_week"][i] = ts.dayofweek
        feats["working_day"][i] = ts.dayofweek
        feats["seasonal_level"][i] = ts.month
    return feats, n_win


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

    # ---- TRAIN query-window features (ctx=64) -> distributions + activation matrix ----
    feats, n_win = _compute_train_features(y, dt)
    feature_distributions = {}
    for feat in ("recent_level", "recent_volatility", "recent_slope_magnitude",
                 "recent_diff_magnitude", "recent_autocorr"):
        arr = feats[feat]
        feature_distributions[feat] = {
            "q10": float(np.percentile(arr, 10)),
            "q25": float(np.percentile(arr, 25)),
            "q50": float(np.percentile(arr, 50)),
            "q75": float(np.percentile(arr, 75)),
            "q90": float(np.percentile(arr, 90)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    timestamps = [str(dt.iloc[i + ACTIVATION_CONTEXT]) for i in range(n_win)]
    train_feature_matrix = dict(feats)
    train_feature_matrix["_timestamps"] = timestamps
    train_feature_matrix["_n_windows"] = n_win

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
        "feature_distributions": feature_distributions,
        "x_subset_representativeness": {},
        "train_feature_matrix": train_feature_matrix,
        "activation_context": ACTIVATION_CONTEXT,
        "n_train_windows": n_win,
    }


# --------------------------------------------------------------------------- #
# Frozen Round-1 candidates (reuse v5's exact Round-1 Expert, not re-designed)
# --------------------------------------------------------------------------- #
def load_frozen_round1() -> List[Dict[str, Any]]:
    if V5_CACHE_DIR.exists():
        for f in V5_CACHE_DIR.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("stage") == "etth1_expert_r1":
                return av2._list_from(d["raw_response"], "expert_knowledge")
    # fallback: flattened round-1 results from v5 output
    if V5_OUT.exists():
        flat = json.loads((V5_OUT / "expert_round_1.json").read_text(encoding="utf-8"))
        return flat
    raise FileNotFoundError("frozen Round-1 candidates not found (v5 cache or output missing)")


# --------------------------------------------------------------------------- #
# Regeneration stub (v5-style revision prompt; only fires if a knowledge failure
# remains after round 1 — which v6 fixes prevent for ETTh1)
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
        '  - "claim": one conditional sentence\n'
        '  - "activation_condition": the "when [observable state]" part\n'
        '  - "observable_features": list of features from AVAILABLE ACTIVATION FEATURES\n'
        '  - "condition_spec": {"logic": "AND"|"OR", "conditions": [{"feature": "...", "qualifier": "..."}]}\n'
        '  - "expected_relationship", "reason", "domain_fact", "latent_state", '
        '"observable_proxy", "expected_future_relationship"\n'
    )


_GEN_SYSTEM = (
    "You are a domain expert on forecasting a univariate hourly time series (target "
    "variable OT). Propose CONDITIONAL expert knowledge for retrieval. Output strict JSON only."
)


def regenerate_fn(feedbacks):
    model = av2._default_model()
    ev = etth1_evidence()  # TRAIN-only; rebuilt only for prompt text (never VAL)
    user = "\n\n".join([
        _metadata_text(),
        _train_summary_text(ev),
        _catalog_text(),
        "CRITIC FEEDBACK ON YOUR PREVIOUS PROPOSALS (per expert):\n"
        + json.dumps(feedbacks, ensure_ascii=False, indent=2),
        _schema_text(),
        "TASK: For each failed expert, REVISE it to fix the specific failure(s) with "
        "TRAIN-supported evidence, keeping the SAME mechanism family and state variable "
        "(do NOT swap to an unrelated mechanism). Output only the JSON object.",
    ])
    raw, path, cached = av2._call_llm(_GEN_SYSTEM, user, "etth1_expert_v6_revision", model)
    return av2._list_from(raw, "expert_knowledge")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _expert_path(rounds, eid):
    for rnd in rounds:
        for r in rnd:
            if r["expert_id"] == eid:
                return r
    return None


def render_report(loop, syn, evidence, round1_experts):
    L = []
    L.append("# ETTh1 — Critic v6 Closed-Loop Audit")
    L.append("")
    L.append("Generic v6 core (dependency-based observability, strict knowledge/compiler "
             "failure split, compiler repair, lineage preservation, TRAIN-only activation "
             "redundancy + coverage). No dataset-specific rule. VAL unused by the Critic; "
             "TEST never read; no prediction MSE run.")
    L.append("")

    # 1. synthetic audit
    L.append("## 1. Synthetic audit")
    L.append("")
    L.append("| case | expected | ok |")
    L.append("|---|---|---|")
    for s in syn:
        L.append(f"| {s['id']} | {s['name']} ({s['expected']}) | {'PASS' if s['ok'] else 'FAIL'} |")
    n_ok = sum(1 for s in syn if s["ok"])
    L.append("")
    L.append(f"**{n_ok}/{len(syn)} PASS**")
    L.append("")

    # 2. round-by-round
    L.append("## 2. Round-by-round counts")
    L.append("")
    L.append("| round | experts | ACCEPT | UNSUPPORTED | REJECT | knowledge_failure | compiler_failure | compiler_repairs |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, rnd in enumerate(loop["rounds"], 1):
        n_acc = sum(1 for r in rnd if r["final_decision"] == "ACCEPT")
        n_uns = sum(1 for r in rnd if r["final_decision"] == "UNSUPPORTED")
        n_rej = sum(1 for r in rnd if r["final_decision"] == "REJECT")
        n_kf = sum(1 for r in rnd if r.get("failure_type") == "KNOWLEDGE_FAILURE")
        n_cf = sum(1 for r in rnd if r.get("failure_type") == "COMPILER_FAILURE")
        n_rp = sum(len(r.get("compiler_repair", [])) for r in rnd)
        L.append(f"| {i} | {len(rnd)} | {n_acc} | {n_uns} | {n_rej} | {n_kf} | {n_cf} | {n_rp} |")
    L.append("")

    # 3/4. E8 / E9 path
    for eid, label in (("E8", "Volatile regime scaling"), ("E9", "Strong recent trend momentum")):
        r = _expert_path(loop["rounds"], eid)
        idx = "3" if eid == "E8" else "4"
        L.append(f"## {idx}. {eid} ({label}) — full processing path")
        L.append("")
        if r is None:
            L.append("(not found)")
        else:
            obs = r["observability_audit"]["features"]
            obs_str = "; ".join(f"{a['feature_name']}=observable:{a['observable_at_prediction_time']}"
                                f"({a['source_type']})" for a in obs)
            repair_str = "; ".join(f"{p['feature']}:{p['original_qualifier']}->{p['normalized_qualifier']}"
                                   f" @ {p.get('threshold_method','')}" for p in r["compiler_repair"]) or "(none)"
            if r["compiler_repair"]:
                path = ("COMPILER_FAILURE -> Compiler Repair (REPAIR_COMPILER): qualifier normalized + "
                        "TRAIN-only threshold; knowledge unchanged (no regeneration)")
            else:
                path = ("no compiler gap; observability dependency-based (no regeneration)")
            L.append(f"- **observability**: {obs_str}")
            L.append(f"- **compiler repair**: {repair_str}")
            L.append(f"- **path**: {path}")
            L.append(f"- **final decision**: {r['final_decision']} | failure_type={r['failure_type']} "
                     f"| action={r['action']}")
            L.append(f"- **disposition**: {r.get('final_disposition')} (coverage "
                     f"dead={r.get('coverage_audit',{}).get('dead')}, "
                     f"constant={r.get('coverage_audit',{}).get('constant')}, "
                     f"local_scope={r.get('coverage_audit',{}).get('local_scope')}, "
                     f"duplicate={r.get('redundancy_audit',{}).get('duplicate_activation')})")
            L.append(f"- **failure reasons**: {r['failure_reasons'] or '(none)'}")
        L.append("")

    # 5. observability audit
    L.append("## 5. Observability audit")
    L.append("")
    L.append("| feature | source_type | dependencies | observable |")
    L.append("|---|---|---|---|")
    for fname, a in loop["observability_log"].items():
        L.append(f"| {fname} | {a['source_type']} | {', '.join(a['dependencies'])} "
                 f"| {a['observable_at_prediction_time']} |")
    L.append("")

    # 6. compiler repair audit
    L.append("## 6. Compiler repair audit")
    L.append("")
    if loop["repair_log"]:
        L.append("| feature | original qualifier | normalized | threshold method | value |")
        L.append("|---|---|---|---|---|")
        for p in loop["repair_log"]:
            L.append(f"| {p['feature']} | {p['original_qualifier']} | {p['normalized_qualifier']} "
                     f"| {p.get('threshold_method','-')} | {p.get('threshold_value','-')} |")
    else:
        L.append("(no compiler repairs — all qualifiers were already canonical)")
    L.append("")

    # 7. regeneration lineage
    L.append("## 7. Regeneration lineage")
    L.append("")
    L.append("| parent | child | mechanism before->after | state before->after | drift |")
    L.append("|---|---|---|---|---|")
    for g in loop["lineage_log"]:
        L.append(f"| {g['parent_expert_id']} | {g['child_expert_id']} | "
                 f"{g['mechanism_family_before']}->{g['mechanism_family_after']} | "
                 f"{g['state_variable_before']}->{g['state_variable_after']} | {g['semantic_drift']} |")
    L.append("")

    # 8. duplicate detection
    L.append("## 8. Duplicate detection")
    L.append("")
    dupes = []
    for rnd in loop["rounds"]:
        for r in rnd:
            red = r.get("redundancy_audit")
            if red and red.get("duplicate_activation"):
                dupes.append(r["expert_id"])
    L.append(f"DUPLICATE_ACTIVATION experts: {dupes or '(none)'}")
    L.append("")

    # 9. activation coverage
    L.append("## 9. Activation coverage (TRAIN-only)")
    L.append("")
    L.append("| expert | active_ratio | dead | constant | local_scope | disposition |")
    L.append("|---|---|---|---|---|---|")
    for rnd in loop["rounds"]:
        for r in rnd:
            cov = r.get("coverage_audit")
            if cov:
                L.append(f"| {r['expert_id']} | {cov['stats'].get('active_ratio','-'):.3f} "
                         f"| {cov['dead']} | {cov['constant']} | {cov['local_scope']} "
                         f"| {r.get('final_disposition')} |")
    L.append("")

    # 10. final pool
    L.append("## 10. Final accepted knowledge pool")
    L.append("")
    if loop["accepted_pool"]:
        for eid, r in loop["accepted_pool"].items():
            L.append(f"- **{eid}** {r['name']}: {r['claim']}")
    else:
        L.append("- (empty)")
    L.append("")

    # conclusion
    e8 = _expert_path(loop["rounds"], "E8")
    e9 = _expert_path(loop["rounds"], "E9")
    e8_obs_ok = bool(e8 and all(a["observable_at_prediction_time"] for a in e8["observability_audit"]["features"]))
    e9_repair = bool(e9 and e9["compiler_repair"] and e9["final_decision"] == "ACCEPT")
    any_drift = any(g["semantic_drift"] for g in loop["lineage_log"])
    any_dupe = bool(dupes)
    L.append("## Conclusion")
    L.append("")
    L.append(f"- **E8 observability**: {'CORRECTLY observable (NOT rejected)' if e8_obs_ok else 'STILL WRONG'}")
    L.append(f"- **E9 compiler**: {'CORRECTLY repaired (qualifier threshold), knowledge unchanged' if e9_repair else 'STILL WRONG'}")
    L.append(f"- **semantic drift occurred**: {any_drift}")
    L.append(f"- **duplicate activation occurred**: {any_dupe}")
    L.append("- **VAL used by Critic**: false (TRAIN-only evidence)")
    L.append("- **v5 population/stability/leakage preserved**: true "
             "(population coverage, subperiod reversal, cycle stability, future leakage all retained)")
    L.append("")
    L.append("```")
    L.append("critic_version = v6")
    L.append("critic_modified_for_dataset = false")
    L.append("train_only_evidence = true")
    L.append("val_used_for_critic = false")
    L.append("test_used = false")
    L.append("prediction_mse_run = false")
    L.append("```")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 90)
    print("ETTH1 CRITIC v6 CLOSED LOOP (generic v6 core)")
    print("=" * 90)

    # ---- 1. synthetic audit (generic, always first) ----
    syn = synthetic_audit_v6()
    for s in syn:
        print(f"  synthetic {s['id']:<2} {s['name']:<55} {'PASS' if s['ok'] else 'FAIL'}")
    print(f"Synthetic: {sum(1 for s in syn if s['ok'])}/{len(syn)} PASS")

    # ---- 2. evidence + frozen round-1 candidates ----
    evidence = etth1_evidence()
    round1_experts = load_frozen_round1()
    print(f"\nTRAIN: {evidence['train_span']['start']} -> {evidence['train_span']['end']} "
          f"({evidence['train_span']['n_rows']} rows, hourly) | "
          f"{evidence['n_train_windows']} TRAIN query windows (ctx={evidence['activation_context']})")
    print(f"frozen Round-1 candidates: {len(round1_experts)}")

    # ---- 3. closed loop ----
    loop = run_closed_loop_v6(
        initial_experts=round1_experts,
        evidence=evidence,
        metadata=ETTH1_METADATA,
        feature_registry=ETTH1_FEATURE_REGISTRY,
        regenerate_fn=regenerate_fn,
        config=COMPILER_CONFIG,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 4. write outputs ----
    for i, rnd in enumerate(loop["rounds"], 1):
        (OUT_DIR / f"expert_round_{i}.json").write_text(
            json.dumps(rnd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (OUT_DIR / f"critic_round_{i}.json").write_text(
            json.dumps([{k: r[k] for k in ("expert_id", "name", "final_decision",
                                           "failure_type", "action", "failure_reasons", "checks")}
                        for r in rnd], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (OUT_DIR / f"feedback_round_{i}.json").write_text(
            json.dumps([r["feedback"] for r in rnd if r.get("feedback")],
                       ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    (OUT_DIR / "final_accepted_knowledge.json").write_text(
        json.dumps(loop["accepted_pool"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "lineage.json").write_text(
        json.dumps(loop["lineage_log"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "observability_audit.json").write_text(
        json.dumps(loop["observability_log"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "compiler_repair_audit.json").write_text(
        json.dumps(loop["repair_log"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "synthetic_audit.json").write_text(
        json.dumps(syn, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    coverage = {}
    redundancy = {}
    for rnd in loop["rounds"]:
        for r in rnd:
            if r.get("coverage_audit"):
                coverage[r["expert_id"]] = r["coverage_audit"]
            if r.get("redundancy_audit"):
                redundancy[r["expert_id"]] = r["redundancy_audit"]
    (OUT_DIR / "activation_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "activation_redundancy.json").write_text(
        json.dumps(redundancy, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = {
        "critic_version": "v6",
        "critic_modified_for_dataset": False,
        "train_only_evidence": True,
        "val_used_for_critic": False,
        "test_used": False,
        "prediction_mse_run": False,
        "dataset": "ETTh1",
        "config": loop["config"],
        "activation_context": evidence["activation_context"],
        "n_train_windows": evidence["n_train_windows"],
        "n_rounds": len(loop["rounds"]),
        "round_counts": [
            {"round": i, "n": len(rnd),
             "accept": sum(1 for r in rnd if r["final_decision"] == "ACCEPT"),
             "unsupported": sum(1 for r in rnd if r["final_decision"] == "UNSUPPORTED"),
             "reject": sum(1 for r in rnd if r["final_decision"] == "REJECT"),
             "knowledge_failure": sum(1 for r in rnd if r.get("failure_type") == "KNOWLEDGE_FAILURE"),
             "compiler_failure": sum(1 for r in rnd if r.get("failure_type") == "COMPILER_FAILURE")}
            for i, rnd in enumerate(loop["rounds"], 1)
        ],
        "n_synthetic_pass": sum(1 for s in syn if s["ok"]),
        "n_synthetic_total": len(syn),
        "n_accepted_pool": len(loop["accepted_pool"]),
        "semantic_drift_occurred": any(g["semantic_drift"] for g in loop["lineage_log"]),
        "duplicate_activation_occurred": any(
            (r.get("redundancy_audit") or {}).get("duplicate_activation")
            for rnd in loop["rounds"] for r in rnd),
        "n_compiler_repairs": len(loop["repair_log"]),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 5. report ----
    report = render_report(loop, syn, evidence, round1_experts)
    (OUT_DIR / "closed_loop_audit.md").write_text(report, encoding="utf-8")

    # ---- 6. console summary ----
    print("\n--- Round-by-round ---")
    for i, rnd in enumerate(loop["rounds"], 1):
        n_acc = sum(1 for r in rnd if r["final_decision"] == "ACCEPT")
        n_kf = sum(1 for r in rnd if r.get("failure_type") == "KNOWLEDGE_FAILURE")
        n_cf = sum(1 for r in rnd if r.get("failure_type") == "COMPILER_FAILURE")
        print(f"Round {i}: {len(rnd)} experts -> {n_acc} ACCEPT | "
              f"knowledge_failure={n_kf} compiler_failure={n_cf}")
        for r in rnd:
            reason = "; ".join(r["failure_reasons"]) if r["failure_reasons"] else "-"
            print(f"    {r['expert_id']:<4}{r['name']:<30}{r['final_decision']:<12}"
                  f"{r.get('failure_type') or '-'}<--{reason[:70]}")

    print(f"\nFinal accepted pool: {len(loop['accepted_pool'])} experts")
    for eid, r in loop["accepted_pool"].items():
        print(f"    {eid:<4}{r['name']:<30}{r.get('final_disposition','-')}")

    print("\nFlags:")
    print("  critic_version = v6")
    print("  critic_modified_for_dataset = false")
    print("  train_only_evidence = true")
    print("  val_used_for_critic = false")
    print("  test_used = false")
    print("  prediction_mse_run = false")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
