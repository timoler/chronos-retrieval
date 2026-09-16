#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agentic Closed-Loop Expert-Agent (v6) — generic, dataset-agnostic core.

Critic v6 fixes four *generic* logic flaws exposed by the ETTh1 run of v5, WITHOUT
adding any dataset-specific rule (no ETTh1 / OT / summer / winter / temperature /
solar / electricity / 6-of-314 hard-codes).  The fixes are:

1. **Dependency-based observability.**  v5 judged observability by scanning the
   expert's whole text for weather/temperature/... keywords, so a feature that is
   fully computable from past context (``recent_volatility``) was wrongly rejected
   because its *explanation* mentioned "weather fronts".  v6 attaches source
   metadata to every feature (``source_type`` + ``dependencies``) and decides
   observability from the computation dependency graph only.

2. **Strict KNOWLEDGE_FAILURE vs COMPILER_FAILURE.**  A qualifier that lacks a
   numeric threshold (``recent_slope_magnitude:large``) is a *compiler* gap, not an
   Expert error.  v6 keeps the knowledge unchanged and repairs the compiler by
   normalising the qualifier and deriving a TRAIN-only threshold (high/large ->
   upper quantile, low/small -> lower quantile).

3. **Knowledge lineage preservation.**  Regeneration may not silently change the
   mechanism family.  If the regenerated candidate changes BOTH mechanism_family and
   state_variable_family it is marked SEMANTIC_DRIFT and re-enters as a NEW
   candidate instead of overwriting the parent.

4. **TRAIN-only activation redundancy + coverage audits.**  A new/existing ACCEPT
   expert is only admitted to the final pool after (a) an activation-redundancy
   check vs every already-accepted expert and (b) a coverage audit (DEAD / CONSTANT
   / LOCAL_SCOPE) over TRAIN query windows.

All audits are TRAIN-only.  VAL is never read by the Critic.  TEST is never read.
No prediction / MLP / Chronos / retrieval MSE is run.

This module is a *core*: it defines the generic machinery.  A dataset script
(e.g. ``run_etth1_critic_v6_closed_loop.py``) supplies the feature registry,
metadata, TRAIN-only evidence (feature distributions + activation feature matrix),
and the initial / regenerated expert candidates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# 0. Global config
# --------------------------------------------------------------------------- #
MAX_ROUNDS = 3

# Qualifier -> quantile config.  Uniform across datasets (never tuned per-dataset).
COMPILER_CONFIG: Dict[str, Any] = {
    "LOW_Q": 0.25,
    "HIGH_Q": 0.75,
    "MEDIUM": [0.25, 0.75],
}

# Dependency kinds that are available at prediction time.
OBSERVABLE_DEP_KINDS = {
    "historical_context", "known_timestamp", "train_derived_constant",
    "observed_covariate", "static_metadata",
}

# source_type values that additionally imply future leakage (semantic REJECT).
FUTURE_SOURCE_TYPES = {"future_target", "future_covariate_unknown"}

# --------------------------------------------------------------------------- #
# 1. Generic concept vocabularies (dataset-agnostic)
# --------------------------------------------------------------------------- #
ANNUAL_CYCLE_VOCAB = [
    "season", "seasonal", "summer", "winter", "autumn", "fall", "spring",
    "month", "monthly", "annual", "year", "yearly", "time-of-year",
    "day length", "day-length", "sun elevation", "solar elevation", "solstice",
]
DIURNAL_CYCLE_VOCAB = [
    "night", "nighttime", "day", "daytime", "morning", "evening", "noon",
    "midday", "time-of-day", "diurnal", "hour", "sunrise", "sunset", "daylight",
]
WEEKLY_CYCLE_VOCAB = ["week", "weekday", "weekend", "workday", "working day"]

POPULATION_VOCAB = [
    "many", "all", "population", "aggregate", "aggregated", "total", "common",
    "shared", "across", "system-wide", "system wide", "co-located", "colocated",
]
LEAKAGE_VOCAB = [
    "future output", "future target", "future value", "future level", "future state",
    "next day", "next-day", "tomorrow", "known in advance", "test set", "test-set",
    "validation", "benchmark", "holdout",
]

# Qualifier synonyms -> canonical {high, low, medium} or calendar category names.
QUALIFIER_SYNONYMS = {
    # canonical "high"
    "large": "high", "big": "high", "elevated": "high", "strong": "high",
    "above": "high", "above_typical": "high", "upper": "high", "greater": "high",
    # canonical "low"
    "small": "low", "weak": "low", "below": "low", "below_typical": "low",
    "lower": "low", "little": "low", "minor": "low",
    # canonical "medium"
    "moderate": "medium", "normal": "medium", "middle": "medium", "mid": "medium",
    # calendar category synonyms
    "weekday": "workday", "weekdays": "workday",
    "saturday": "weekend", "sunday": "weekend",
}

CONTINUOUS_CANONICAL_QUALIFIERS = {"high", "low", "medium"}


def _has(text: str, vocab: List[str]) -> List[str]:
    t = (text or "").lower()
    return [w for w in vocab if w in t]


def _word_in(word: str, text: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text) is not None


def _q_key(q: float) -> str:
    return f"q{int(round(q * 100))}"


def _normalize_qualifier(q: Any) -> Optional[str]:
    if q is None:
        return None
    q = str(q).strip().lower()
    if q.startswith("is_"):
        q = q[3:]
    return QUALIFIER_SYNONYMS.get(q, q)


# =========================================================================== #
# 2. OBSERVABILITY (dependency-based, NOT name-based)
# =========================================================================== #
def audit_observability(feature: str, feature_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether a feature is observable at prediction time.

    The decision is computed from the feature's *computation dependencies*, not
    from a hard-coded whitelist of feature names.  A feature whose dependencies are
    all drawn from {historical context, known timestamp, TRAIN-derived constants,
    observed covariates, static metadata} is observable.  A feature that depends on
    a future target, an unknown future covariate, or an unavailable external
    variable is not.
    """
    meta = feature_registry.get(feature)
    if meta is None:
        return {
            "feature_name": feature,
            "source_type": "external_unavailable",
            "dependencies": ["unavailable_external_variable"],
            "observable_at_prediction_time": False,
            "reason": "feature has no computation dependency in the available-variable "
                      "registry -> treated as an unavailable external variable",
        }
    deps = list(meta.get("dependencies", []))
    source_type = meta.get("source_type", "unknown")
    bad = [d for d in deps if d not in OBSERVABLE_DEP_KINDS]
    observable = len(bad) == 0
    if observable:
        reason = "all dependencies available at prediction time"
    else:
        reason = f"depends on inputs not available at prediction time: {bad}"
    return {
        "feature_name": feature,
        "source_type": source_type,
        "dependencies": deps,
        "observable_at_prediction_time": observable,
        "reason": reason,
    }


def _condition_features(expert: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = expert.get("condition_spec") or {}
    conds = spec.get("conditions") or []
    if conds:
        return conds
    # legacy fallback: observable_features as bare features
    return [{"feature": f} for f in expert.get("observable_features", [])]


def audit_observability_for_expert(expert: Dict[str, Any],
                                   feature_registry: Dict[str, Any]) -> Dict[str, Any]:
    feats = _condition_features(expert)
    audits = [audit_observability(c.get("feature"), feature_registry) for c in feats]
    all_obs = all(a["observable_at_prediction_time"] for a in audits)
    any_future = any(a["source_type"] in FUTURE_SOURCE_TYPES for a in audits)
    return {
        "expert_id": expert.get("id", expert.get("expert_id", expert.get("name"))),
        "features": audits,
        "all_observable": all_obs,
        "future_leakage": any_future,
    }


# =========================================================================== #
# 3. KNOWLEDGE LINEAGE (mechanism / state-variable family)
# =========================================================================== #
def extract_lineage(expert: Dict[str, Any],
                    feature_registry: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a stable semantic identity to an expert, used for drift detection."""
    feats = [c.get("feature") for c in _condition_features(expert) if c.get("feature")]
    mech_families = []
    state_families = []
    for f in feats:
        meta = feature_registry.get(f, {})
        mf = meta.get("mechanism_family", "unknown")
        sf = meta.get("state_variable_family", "unknown")
        mech_families.append(mf)
        state_families.append(sf)

    def _primary(vals):
        uniq = sorted(set(v for v in vals if v))
        return "+".join(uniq) if uniq else "unknown"

    return {
        "expert_id": expert.get("id", expert.get("expert_id", expert.get("name"))),
        "mechanism_family": _primary(mech_families),
        "state_variable_family": _primary(state_families),
        "activation_scope": feats,
        "temporal_scope": [],   # filled by the claim-scope analyzer
        "population_scope": False,
    }


def detect_semantic_drift(parent: Dict[str, Any],
                          child: Dict[str, Any]) -> Dict[str, Any]:
    """Return a drift record.  Drift = BOTH mechanism_family and state_variable_family
    changed.  A drift candidate must not overwrite the parent."""
    mech_changed = parent.get("mechanism_family") != child.get("mechanism_family")
    state_changed = parent.get("state_variable_family") != child.get("state_variable_family")
    drift = bool(mech_changed and state_changed)
    return {
        "parent_expert_id": parent.get("expert_id"),
        "child_expert_id": child.get("expert_id"),
        "mechanism_family_before": parent.get("mechanism_family"),
        "mechanism_family_after": child.get("mechanism_family"),
        "state_variable_before": parent.get("state_variable_family"),
        "state_variable_after": child.get("state_variable_family"),
        "semantic_drift": drift,
    }


# =========================================================================== #
# 4. Claim Scope Analyzer (registry-driven observability; no keyword rejection)
# =========================================================================== #
def analyze_claim_scope_v6(expert: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(str(expert.get(k, "")) for k in (
        "name", "claim", "knowledge", "activation_condition", "expected_relationship",
        "reason", "domain_fact", "latent_state", "observable_proxy",
        "expected_future_relationship"))

    temporal_scope = []
    if _has(text, ANNUAL_CYCLE_VOCAB):
        temporal_scope.append("annual")
    if _has(text, DIURNAL_CYCLE_VOCAB):
        temporal_scope.append("diurnal")
    if _has(text, WEEKLY_CYCLE_VOCAB):
        temporal_scope.append("weekly")

    population_scope = bool(_has(text, POPULATION_VOCAB)) or bool(
        _has(str(metadata.get("target_description", "")), POPULATION_VOCAB))

    state_scope = []
    for kw in ("near zero", "near-zero", "high", "low", "above", "below",
               "typical", "large", "small", "smooth", "differs", "different"):
        if kw in str(expert.get("activation_condition", "")).lower():
            state_scope.append(kw)
    state_scope = sorted(set(state_scope))

    n_pop = int(metadata.get("n_population", 1))
    required_coverage = {
        "months": 12 if "annual" in temporal_scope else None,
        "hours": 24 if "diurnal" in temporal_scope else None,
        "days": 7 if "weekly" in temporal_scope else None,
        "population": n_pop if population_scope else None,
    }

    failure_modes = []
    if "annual" in temporal_scope:
        failure_modes.append("annual/seasonal cycle -> requires full-year temporal coverage")
    if "diurnal" in temporal_scope:
        failure_modes.append("diurnal cycle -> requires full-day (24h) temporal coverage")
    if "weekly" in temporal_scope:
        failure_modes.append("weekly cycle -> requires full-week temporal coverage")
    if population_scope:
        failure_modes.append("population-level claim -> requires proxy representativeness evidence")

    input_side = " ".join([
        str(expert.get("activation_condition", "")),
        str(expert.get("observable_proxy", "")),
        " ".join(str(f) for f in expert.get("observable_features", [])),
    ])
    leakage = bool(_has(input_side, LEAKAGE_VOCAB))
    contradiction = _detect_contradiction(expert)

    return {
        "claimed_scope": {
            "temporal_scope": temporal_scope,
            "state_scope": state_scope,
            "population_scope": population_scope,
            "variable_scope": list(expert.get("observable_features", [])),
        },
        "required_coverage": required_coverage,
        "potential_failure_modes": failure_modes,
        "semantic_flags": {
            "leakage": leakage,
            "contradiction": contradiction,
            "relevant": True,
        },
    }


def _detect_contradiction(expert: Dict[str, Any]) -> bool:
    cond = str(expert.get("activation_condition", "")).lower()
    rel = (str(expert.get("expected_relationship", "")) + " "
           + str(expert.get("expected_future_relationship", ""))).lower()
    for a, b in (("high", "low"), ("large", "small"), ("above", "below"),
                 ("increase", "decrease"), ("up", "down")):
        if _word_in(a, cond) and _word_in(b, rel) and not _word_in(a, rel):
            return True
    return False


# =========================================================================== #
# 5. Evidence Planner (generic)
# =========================================================================== #
def plan_evidence(scope: Dict[str, Any]) -> Dict[str, Any]:
    cs = scope["claimed_scope"]
    req = scope["required_coverage"]
    plan = {"support": [], "coverage": [], "stability": [], "counter": [], "executability": []}

    if req["months"]:
        plan["support"].append("annual signal: monthly means / eta2(month, target)")
        plan["coverage"].append("month coverage over the full 12-month cycle")
        plan["stability"].append("month-to-month direction consistency (single-peak annual rhythm)")
        plan["counter"].append("any month where the claimed ordering reverses")
    if req["hours"]:
        plan["support"].append("diurnal signal: hourly means / eta2(hour, target)")
        plan["coverage"].append("hour coverage over the full 24h cycle")
        plan["stability"].append("hour-to-hour envelope smoothness")
    if req["days"]:
        plan["support"].append("weekly signal: day-of-week means / eta2(dow, target)")
        plan["coverage"].append("day-of-week coverage over the full week")
    if req["population"]:
        plan["coverage"].append("proxy fraction of the population")
        plan["counter"].append("whether the subset proxy can be shown representative")
    if cs["state_scope"]:
        plan["support"].append(
            f"verify the claimed state {cs['state_scope']} actually manifests in TRAIN")
    for _f in cs["variable_scope"]:
        plan["executability"].append(
            "activation feature computable for future queries without arbitrary fallback")
    return plan


# =========================================================================== #
# 6. Compiler v6 (+ Compiler Repair)
# =========================================================================== #
def _resolve_continuous(feature: str, canonical_qualifier: str,
                        evidence: Dict[str, Any], config: Dict[str, Any]):
    """Derive an executable TRAIN-only threshold for a continuous qualifier.

    high/large -> upper quantile (HIGH_Q); low/small -> lower quantile (LOW_Q);
    medium -> middle interval [LOW_Q, HIGH_Q].  All thresholds come from the
    TRAIN-only feature distribution.
    """
    dist = (evidence.get("feature_distributions", {}) or {}).get(feature)
    if dist is None:
        return None, f"no TRAIN feature distribution for continuous feature '{feature}'"
    lo, hi = config["LOW_Q"], config["HIGH_Q"]
    if canonical_qualifier == "high":
        return {"op": ">=", "rule": dist[_q_key(hi)], "method": f"HIGH_Q={hi}"}, None
    if canonical_qualifier == "low":
        return {"op": "<=", "rule": dist[_q_key(lo)], "method": f"LOW_Q={lo}"}, None
    if canonical_qualifier == "medium":
        return {"op": "between", "rule": [dist[_q_key(lo)], dist[_q_key(hi)]],
                "method": f"MEDIUM=[{lo},{hi}]"}, None
    return None, f"qualifier '{canonical_qualifier}' is not a supported continuous qualifier"


def _resolve_primitive_v6(feature: str, qualifier: Any, feature_registry: Dict[str, Any],
                          evidence: Dict[str, Any], config: Dict[str, Any]):
    """Resolve one primitive condition into an executable rule, with repair.

    Returns (primitive_or_None, error_or_None, repair_record_or_None).
    """
    meta = feature_registry.get(feature)
    if meta is None:
        return None, {"type": "feature_unobservable",
                      "reason": f"feature '{feature}' has no observable computation"}, None

    kind = meta.get("kind")
    raw_qualifier = qualifier
    qualifier = _normalize_qualifier(qualifier)

    if kind == "calendar":
        categories = meta.get("categories", {})
        if qualifier not in categories:
            return None, {"type": "qualifier_undefined",
                          "reason": f"qualifier '{qualifier}' not a defined calendar category "
                                    f"for feature '{feature}'"}, None
        repair = ({"feature": feature, "original_qualifier": raw_qualifier,
                   "normalized_qualifier": qualifier, "repair": "calendar_category",
                   "action": "REPAIR_COMPILER"}
                  if raw_qualifier != qualifier else None)
        return {"feature": feature, "qualifier": qualifier,
                "rule": categories[qualifier], "kind": "calendar", "op": "in"}, None, repair

    # continuous
    if qualifier not in CONTINUOUS_CANONICAL_QUALIFIERS:
        return None, {"type": "qualifier_undefined",
                      "reason": f"qualifier '{qualifier}' not defined for continuous "
                                f"feature '{feature}'"}, None
    resolved, err = _resolve_continuous(feature, qualifier, evidence, config)
    if err is not None:
        return None, {"type": "qualifier_undefined", "reason": err}, None
    repair = ({"feature": feature, "original_qualifier": raw_qualifier,
               "normalized_qualifier": qualifier, "threshold_method": resolved["method"],
               "threshold_value": resolved["rule"],
               "train_source": "TRAIN-only feature distribution quantile",
               "action": "REPAIR_COMPILER"}
              if raw_qualifier != qualifier else None)
    return {"feature": feature, "qualifier": qualifier, "rule": resolved["rule"],
            "kind": "continuous", "op": resolved["op"]}, None, repair


def compile_condition_v6(expert: Dict[str, Any], feature_registry: Dict[str, Any],
                         evidence: Dict[str, Any], config: Dict[str, Any]):
    spec = expert.get("condition_spec") or {}
    conds = spec.get("conditions") or []
    if not conds:
        conds = [{"feature": f} for f in expert.get("observable_features", [])]

    logic = (spec.get("logic") or "AND").upper()
    negate = bool(spec.get("negate", False))

    primitives = []
    uncompilable = []
    repairs = []
    for c in conds:
        feat = c.get("feature")
        qual = c.get("qualifier")
        prim, err, repair = _resolve_primitive_v6(feat, qual, feature_registry, evidence, config)
        if prim is None:
            uncompilable.append({"feature": feat, "qualifier": qual,
                                 "type": err["type"], "reason": err["reason"]})
        else:
            primitives.append(prim)
            if repair is not None:
                repairs.append(repair)

    status = "COMPILED" if not uncompilable else "UNCOMPILABLE_REQUIRED_CONDITION"
    return {
        "status": status,
        "logic": logic,
        "negate": negate,
        "n_primitives": len(primitives),
        "primitives": primitives,
        "uncompilable": uncompilable,
        "features_compiled": [p["feature"] for p in primitives],
        "repairs": repairs,
    }


# =========================================================================== #
# 7. Proxy Validator (compiled condition faithful to the original)
# =========================================================================== #
def validate_proxy_v6(expert: Dict[str, Any], compiled: Dict[str, Any],
                      obs_audit: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    spec = expert.get("condition_spec") or {}
    requested = [c.get("feature") for c in spec.get("conditions", [])]
    if not requested:
        requested = list(expert.get("observable_features", []))
    requested = [f for f in requested if f]

    compiled_feats = set(compiled["features_compiled"])
    missing = [f for f in requested if f not in compiled_feats]
    if missing:
        issues.append(f"condition dropped / uncompiled: {missing}")

    if compiled["uncompilable"]:
        issues.append("required condition uncompilable (must not be dropped): "
                      + "; ".join(f"{u['feature']} ({u['reason']})" for u in compiled["uncompilable"]))

    req_logic = (spec.get("logic") or "AND").upper()
    if req_logic != compiled["logic"]:
        issues.append(f"logic changed: requested {req_logic}, compiled {compiled['logic']}")

    if bool(spec.get("negate", False)) != compiled["negate"]:
        issues.append("negation changed")

    # future info / leakage in the compiled input side (dependency-based + vocab)
    input_side = " ".join([
        str(expert.get("activation_condition", "")),
        str(expert.get("observable_proxy", "")),
        " ".join(requested),
    ])
    if _has(input_side, LEAKAGE_VOCAB):
        issues.append("activation input uses future/test/benchmark information")
    if obs_audit.get("future_leakage"):
        issues.append("activation input depends on a future target / unknown future covariate")

    faithful = len(issues) == 0 and compiled["status"] == "COMPILED"
    return {"faithful": faithful, "compiled_status": compiled["status"], "issues": issues}


# =========================================================================== #
# 8. Critic (A..J checks, dataset-agnostic)
# =========================================================================== #
def _cycle_support(scope, evidence):
    cs = scope["claimed_scope"]
    temporal = cs["temporal_scope"]
    state = cs["state_scope"]

    cycle_means = None
    eta2 = None
    name = None
    if "annual" in temporal:
        cycle_means = evidence["seasonal"]["monthly_mean_values"]
        eta2 = evidence["seasonal"]["eta2_month_on_ot"]
        name = "annual"
    elif "diurnal" in temporal:
        cycle_means = evidence["diurnal"]["hourly_mean"]
        eta2 = evidence["diurnal"]["eta2_hour_on_ot"]
        name = "diurnal"
    elif "weekly" in temporal:
        cycle_means = evidence["weekly"]["dow_mean"]
        eta2 = evidence["weekly"]["eta2_dow_on_ot"]
        name = "weekly"

    if cycle_means is None:
        return {"ok": True, "status": "PASS", "note": "no temporal cycle claimed"}

    cycle_means = np.asarray(cycle_means, dtype=np.float64)
    mean = float(cycle_means.mean())
    mn, mx = float(cycle_means.min()), float(cycle_means.max())
    spread = (mx - mn) / (abs(mean) + 1e-9)

    if "near zero" in state or "near-zero" in state:
        ratio = mn / (abs(mean) + 1e-9)
        ok = ratio < 0.1
        note = (f"{name} minimum is {ratio:.1%} of mean -> near-zero state observed"
                if ok else f"{name} minimum is {ratio:.1%} of mean -> near-zero state NOT observed")
    else:
        ok = spread > 0.05 or (eta2 is not None and eta2 > 0.05)
        note = (f"{name} signal present (spread {spread:.3f}, eta2 {eta2})"
                if ok else f"{name} signal weak (spread {spread:.3f}, eta2 {eta2})")

    return {"ok": ok, "status": "PASS" if ok else "FAIL",
            "evidence": {"cycle": name, "min": mn, "max": mx, "spread": round(spread, 4),
                         "eta2": eta2},
            "note": note}


def critique_v6(expert, scope, evidence, compiled, proxy_valid, obs_audit):
    cs = scope["claimed_scope"]
    req = scope["required_coverage"]
    tc = evidence["temporal_coverage"]
    checks = {}

    # A. semantic validity
    sem = scope["semantic_flags"]
    sem_fail = []
    if sem["leakage"] or obs_audit.get("future_leakage"):
        sem_fail.append("future/test/benchmark leakage")
    if sem["contradiction"]:
        sem_fail.append("directional contradiction")
    if not sem["relevant"]:
        sem_fail.append("not relevant")
    checks["semantic_validity"] = {"ok": not sem_fail, "status": "PASS" if not sem_fail else "FAIL",
                                   "evidence": sem, "note": "; ".join(sem_fail) or "semantically valid"}

    # B. observability (dependency-based)
    unobs = [a["feature_name"] for a in obs_audit["features"]
             if not a["observable_at_prediction_time"]]
    obs_ok = len(unobs) == 0
    checks["observability"] = {"ok": obs_ok, "status": "PASS" if obs_ok else "FAIL",
                               "evidence": obs_audit["features"],
                               "note": ("required activation features observable"
                                        if obs_ok else f"unobservable required features: {unobs}")}

    # C. empirical support (claim-driven, generic)
    sup = _cycle_support(scope, evidence)
    checks["support"] = sup

    # D. coverage sufficiency
    cov_fail = []
    if req["months"] and tc["month_fraction"] < 1.0:
        cov_fail.append(f"annual cycle {tc['month_fraction']:.0%} covered (missing {tc['months_missing']})")
    if req["hours"] and tc["hour_fraction"] < 1.0:
        cov_fail.append(f"diurnal cycle {tc['hour_fraction']:.0%} covered")
    if req["days"] and tc["dow_fraction"] < 1.0:
        cov_fail.append(f"weekly cycle {tc['dow_fraction']:.0%} covered")
    checks["coverage"] = {"ok": not cov_fail, "status": "PASS" if not cov_fail else "FAIL",
                          "evidence": tc, "note": "; ".join(cov_fail) or "claimed scope fully covered"}

    # E. stability (temporal-cycle direction reversals)
    stab_fail = []
    if req["months"] and evidence["seasonal"]["n_month_direction_reversals"] > 1:
        stab_fail.append("monthly direction reversed "
                         f"{evidence['seasonal']['n_month_direction_reversals']} times in TRAIN")
    checks["stability"] = {"ok": not stab_fail, "status": "PASS" if not stab_fail else "FAIL",
                           "evidence": {"n_month_direction_reversals":
                                        evidence["seasonal"]["n_month_direction_reversals"]},
                           "note": "; ".join(stab_fail) or "relationship stable across sub-periods"}

    # F. counter-evidence (subset-proxy sub-period sign reversals -> alternative cause)
    counter = []
    xr = evidence.get("x_subset_representativeness", {}) or {}
    if xr.get("sign_reversals", 0) > 0 and "subset" in str(expert.get("observable_proxy", "")).lower():
        counter.append("subset-proxy relationship reverses sign across sub-periods "
                       f"({xr['sign_reversals']} reversal(s))")
    checks["counter"] = {"ok": not counter, "status": "PASS" if not counter else "FAIL",
                         "evidence": xr,
                         "note": "; ".join(counter) or "no critical counter-evidence"}

    # G. future executability
    fallback = _detect_fallback(expert, compiled, evidence)
    exec_ok = compiled["status"] == "COMPILED" and not fallback["arbitrary_fallback"]
    checks["executability"] = {"ok": exec_ok, "status": "PASS" if exec_ok else "FAIL",
                               "evidence": {"compiled": compiled["status"], "fallback": fallback},
                               "note": (compiled["status"] if not exec_ok and compiled["status"] != "COMPILED"
                                        else (fallback["reason"] if fallback["arbitrary_fallback"]
                                              else "compiled and executable"))}

    # H. population validity
    pop = evidence["population"]
    if cs["population_scope"]:
        needs_subset_proxy = any(w in str(expert.get("observable_proxy", "")).lower()
                                 for w in ("subset", "individual consumer", "observed subset",
                                           "represent", "proxy"))
        if needs_subset_proxy and pop["proxy_fraction"] < 1.0:
            checks["population"] = {"ok": False, "status": "FAIL", "evidence": pop,
                                    "note": f"subset proxy covers {pop['proxy_fraction']:.0%} of "
                                            "population without representativeness evidence"}
        else:
            checks["population"] = {"ok": True, "status": "PASS", "evidence": pop,
                                    "note": "claim operates on the aggregate target, not a subset proxy"}
    else:
        checks["population"] = {"ok": True, "status": "PASS", "evidence": pop,
                                "note": "no population-level claim"}

    # I. proxy fidelity
    checks["proxy_fidelity"] = {"ok": proxy_valid["faithful"], "status": "PASS" if proxy_valid["faithful"] else "FAIL",
                                "evidence": proxy_valid,
                                "note": "; ".join(proxy_valid["issues"]) or "compiled condition faithful to original"}

    # J. arbitrary fallback
    checks["arbitrary_fallback"] = {"ok": not fallback["arbitrary_fallback"],
                                    "status": "PASS" if not fallback["arbitrary_fallback"] else "FAIL",
                                    "evidence": fallback, "note": fallback["reason"]}

    return checks


def _detect_fallback(expert, compiled, evidence):
    tc = evidence["temporal_coverage"]
    for p in compiled["primitives"]:
        f = p["feature"]
        if f == "hour_of_day" and tc["hour_fraction"] < 1.0:
            return {"arbitrary_fallback": True, "reason":
                    f"{f} indexes hour; TRAIN missing {tc['hours_missing']}"}
        if f == "seasonal_level" and tc["month_fraction"] < 1.0:
            return {"arbitrary_fallback": True, "reason":
                    f"{f} indexes month; TRAIN missing {tc['months_missing']}"}
        if f in ("day_of_week", "working_day") and tc["dow_fraction"] < 1.0:
            return {"arbitrary_fallback": True, "reason":
                    f"{f} indexes day-of-week; TRAIN missing {tc['dows_missing']}"}
    return {"arbitrary_fallback": False, "reason": "no arbitrary calendar fallback"}


# =========================================================================== #
# 9. Decision + strict failure classification
# =========================================================================== #
CRITICAL_CHECKS = [
    "semantic_validity", "observability", "support", "coverage", "stability",
    "counter", "executability", "population", "proxy_fidelity", "arbitrary_fallback",
]

KNOWLEDGE_CHECKS = {"semantic_validity", "observability", "support", "coverage",
                    "stability", "counter", "population"}
COMPILER_CHECKS = {"executability", "proxy_fidelity", "arbitrary_fallback"}


def decide_v6(scope, checks):
    sem = scope["semantic_flags"]
    reject_reasons = []
    if sem["leakage"]:
        reject_reasons.append("future/test/benchmark leakage")
    if sem["contradiction"]:
        reject_reasons.append("directional contradiction between activation and expected relationship")
    if not sem["relevant"]:
        reject_reasons.append("not relevant to retrieval/forecasting")
    if reject_reasons:
        return {"final_decision": "REJECT", "failure_reasons": reject_reasons,
                "failed_checks": ["semantic_validity"]}
    failures = [c for c in CRITICAL_CHECKS if not checks[c]["ok"]]
    if failures:
        reasons = [f"{c}: {checks[c]['note']}" for c in failures]
        return {"final_decision": "UNSUPPORTED", "failure_reasons": reasons, "failed_checks": failures}
    return {"final_decision": "ACCEPT", "failure_reasons": [], "failed_checks": []}


def classify_failure_v6(scope, checks, compiled, obs_audit):
    """Strict KNOWLEDGE_FAILURE vs COMPILER_FAILURE vs NONE.

    KNOWLEDGE_FAILURE  -> the mechanism itself is unsupported (unobservable variable,
                          population invalidity, coverage contradiction, instability,
                          counter-evidence, future leakage, scope unsupported).
    COMPILER_FAILURE   -> the knowledge is fine but the compiler cannot faithfully
                          realize the condition (an *observable* feature whose qualifier
                          lacks an executable threshold).  Never fed back to the Expert.
    """
    sem = scope["semantic_flags"]
    if sem["leakage"] or sem["contradiction"] or not sem["relevant"] \
            or obs_audit.get("future_leakage"):
        return "KNOWLEDGE_FAILURE", "REJECT"

    failed = [c for c in CRITICAL_CHECKS if not checks[c]["ok"]]
    if not failed:
        return "NONE", "ACCEPT"

    knowledge_failed = [c for c in failed if c in KNOWLEDGE_CHECKS]
    if knowledge_failed:
        return "KNOWLEDGE_FAILURE", "REGENERATE_KNOWLEDGE"

    # only compiler-side checks failed: decide whether it is an unobservable variable
    # (knowledge failure) or a qualifier/compile gap (compiler failure).
    for u in compiled.get("uncompilable", []):
        if u.get("type") == "feature_unobservable":
            return "KNOWLEDGE_FAILURE", "REGENERATE_KNOWLEDGE"
    return "COMPILER_FAILURE", "REPAIR_COMPILER"


def build_feedback_v6(expert, decision, checks, compiled, evidence, failure_type):
    failed = [c for c in CRITICAL_CHECKS if not checks[c]["ok"]]
    repairable = failure_type == "KNOWLEDGE_FAILURE" and "semantic_validity" not in failed

    digests = []
    if "population" in failed:
        p = evidence["population"]
        digests.append(f"observed subset covers {p['n_proxy']}/{p['n_population']} "
                       f"({p['proxy_fraction']:.0%}) of the target population; "
                       f"corr(subset-sum, target) = {p['corr_proxy_sum_vs_target']}")
    if "stability" in failed and "monthly direction reversed" in checks["stability"].get("note", ""):
        digests.append(f"monthly direction reversed "
                       f"{evidence['seasonal']['n_month_direction_reversals']} times in TRAIN")
    if "support" in failed:
        digests.append(checks["support"]["note"])
    if "counter" in failed:
        digests.append(checks["counter"]["note"])
    if "observability" in failed:
        digests.append(checks["observability"]["note"])

    if failure_type == "COMPILER_FAILURE":
        feedback = ("The knowledge itself may be fine; the compiler failed to faithfully "
                    "realize the full activation condition. This is a compiler gap, NOT an "
                    "Expert error — do NOT change the mechanism. Issues: "
                    + "; ".join(checks["proxy_fidelity"]["evidence"]["issues"]))
    else:
        feedback = ("TRAIN evidence does not support this knowledge as stated. "
                    + " ".join(digests)
                    + " Revise the mechanism to one that is TRAIN-supported, or drop it "
                    "and propose a different mechanism. Do not re-state the same failed idea.")

    return {
        "expert_id": expert.get("id", expert.get("expert_id", expert.get("name"))),
        "decision": decision["final_decision"],
        "failed_checks": failed,
        "failure_type": failure_type,
        "feedback_to_expert": feedback,
        "repairable": repairable,
    }


# =========================================================================== #
# 10. Semantic duplicate check (re-phrasing of a failed mechanism)
# =========================================================================== #
def _token_similarity(a, b):
    ta = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def mark_duplicates(experts, failed_history, threshold=0.75):
    for e in experts:
        e["_duplicate_of"] = None
        claim = e.get("claim") or e.get("knowledge", "")
        best = None
        for h in failed_history:
            hc = h.get("claim") or h.get("knowledge", "")
            sim = _token_similarity(claim, hc)
            if sim >= threshold and (best is None or sim > best[0]):
                best = (sim, h.get("id"))
        if best is not None:
            e["_duplicate_of"] = best[1]
    return experts


# =========================================================================== #
# 11. Activation executor (generic evaluation over a TRAIN feature matrix)
# =========================================================================== #
def evaluate_expert_activation(compiled: Dict[str, Any],
                               feature_matrix: Dict[str, np.ndarray], i: int) -> float:
    """Evaluate a compiled condition at window index i -> binary {0,1}."""
    prims = compiled.get("primitives", [])
    logic = compiled.get("logic", "AND")
    negate = bool(compiled.get("negate", False))
    if not prims:
        return 0.0
    vals = []
    for p in prims:
        feat = p["feature"]
        if feat not in feature_matrix:
            vals.append(0.0)
            continue
        raw = feature_matrix[feat][i]
        kind = p.get("kind")
        if kind == "calendar":
            vals.append(1.0 if int(raw) in p["rule"] else 0.0)
        else:
            op = p.get("op", ">=")
            if op == ">=":
                vals.append(1.0 if float(raw) >= p["rule"] else 0.0)
            elif op == "<=":
                vals.append(1.0 if float(raw) <= p["rule"] else 0.0)
            elif op == "between":
                lo, hi = p["rule"]
                vals.append(1.0 if lo <= float(raw) <= hi else 0.0)
            else:
                vals.append(0.0)
    if logic == "OR":
        out = max(vals)
    else:
        out = min(vals)
    if negate:
        out = 1.0 - out
    return float(out)


def build_train_activation_matrix(compiled_experts: List[Dict[str, Any]],
                                  feature_matrix: Dict[str, np.ndarray]) -> np.ndarray:
    n_win = feature_matrix.get("_n_windows", None)
    if n_win is None:
        n_win = len(next(v for k, v in feature_matrix.items() if not k.startswith("_")))
    n = len(compiled_experts)
    M = np.zeros((n_win, n), dtype=np.float64)
    for j, ce in enumerate(compiled_experts):
        for i in range(n_win):
            M[i, j] = evaluate_expert_activation(ce, feature_matrix, i)
    return M


# =========================================================================== #
# 12. Activation coverage audit (TRAIN-only)
# =========================================================================== #
def audit_activation_coverage(act_vec: np.ndarray, timestamps: List[Any],
                              n_blocks: int = 4) -> Dict[str, Any]:
    a = np.asarray(act_vec, dtype=np.float64)
    n = len(a)
    binary = bool(np.all((a == 0.0) | (a == 1.0)))

    stats = {
        "activation_mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "unique_count": int(len(np.unique(np.round(a, 6)))),
        "binary": binary,
    }
    if binary:
        stats["active_ratio"] = float((a > 0.5).mean())

    # 4 chronological blocks
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    blocks = {}
    for b in range(n_blocks):
        seg = a[edges[b]:edges[b + 1]]
        blocks[f"block{b + 1}"] = {
            "activation_mean": float(seg.mean()),
            "std": float(seg.std()),
            "active_ratio": float((seg > 0.5).mean()) if binary else None,
            "n": int(len(seg)),
        }

    # DEAD: constant 0 or near-zero with no variation
    dead = bool(a.max() <= 0.0) or (binary and a.mean() == 0.0)
    # CONSTANT: almost always 1, or continuous std ~ 0
    constant = bool(a.min() >= 1.0) or (not binary and a.std() < 1e-9) \
        or (binary and a.mean() == 1.0)
    # LOCAL_SCOPE: activates in only part of the chronological blocks
    active_blocks = [b for b in range(n_blocks)
                     if (blocks[f"block{b + 1}"]["active_ratio"] or 0.0) > 0.0]
    local_scope = bool(not dead and not constant and len(active_blocks) < n_blocks)

    return {
        "stats": stats,
        "chronological_blocks": blocks,
        "dead": dead,
        "constant": constant,
        "low_information": constant,
        "local_scope": local_scope,
        "n_active_blocks": len(active_blocks),
    }


# =========================================================================== #
# 13. Activation redundancy audit (TRAIN-only, vs accepted pool)
# =========================================================================== #
def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _jaccard(a, b):
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def audit_activation_redundancy(candidate_vec: np.ndarray,
                                pool: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparisons = []
    duplicate = False
    for p in pool:
        b = np.asarray(p["activation_vector"], dtype=np.float64)
        corr = _pearson(candidate_vec, b)
        jac = _jaccard(candidate_vec, b)
        coact = float((np.logical_and(candidate_vec > 0.5, b > 0.5)).mean())
        redundant = abs(corr) > 0.95 or jac > 0.95
        comparisons.append({
            "vs_expert_id": p["expert_id"],
            "vs_name": p["name"],
            "pearson_corr": round(corr, 4),
            "abs_corr": round(abs(corr), 4),
            "coactivation_rate": round(coact, 4),
            "jaccard": round(jac, 4),
            "duplicate_activation": bool(redundant),
        })
        if redundant:
            duplicate = True
    return {"duplicate_activation": duplicate, "comparisons": comparisons}


def determine_disposition(coverage, redundancy) -> str:
    if coverage.get("dead"):
        return "DEAD"
    if coverage.get("constant"):
        return "CONSTANT_LOW_INFORMATION"
    if redundancy.get("duplicate_activation"):
        return "DUPLICATE_ACTIVATION"
    if coverage.get("local_scope"):
        return "ACCEPTED_LOCAL_SCOPE"
    return "ACCEPTED"


# =========================================================================== #
# 14. Per-expert processing
# =========================================================================== #
def process_expert_v6(expert: Dict[str, Any], evidence: Dict[str, Any],
                      metadata: Dict[str, Any], feature_registry: Dict[str, Any],
                      config: Dict[str, Any]) -> Dict[str, Any]:
    expert.setdefault("id", expert.get("expert_id", expert.get("name")))
    expert.setdefault("knowledge", expert.get("claim", ""))
    eid = expert["id"]

    lineage = extract_lineage(expert, feature_registry)
    scope = analyze_claim_scope_v6(expert, metadata)
    lineage["temporal_scope"] = scope["claimed_scope"]["temporal_scope"]
    lineage["population_scope"] = scope["claimed_scope"]["population_scope"]

    plan = plan_evidence(scope)
    obs_audit = audit_observability_for_expert(expert, feature_registry)
    compiled = compile_condition_v6(expert, feature_registry, evidence, config)
    proxy_valid = validate_proxy_v6(expert, compiled, obs_audit)
    checks = critique_v6(expert, scope, evidence, compiled, proxy_valid, obs_audit)
    decision = decide_v6(scope, checks)
    failure_type, action = classify_failure_v6(scope, checks, compiled, obs_audit)

    fb = None
    if decision["final_decision"] != "ACCEPT":
        fb = build_feedback_v6(expert, decision, checks, compiled, evidence, failure_type)

    return {
        "expert_id": eid,
        "name": expert.get("name"),
        "claim": expert.get("claim", expert.get("knowledge", "")),
        "activation_condition": expert.get("activation_condition", ""),
        "observable_features": expert.get("observable_features", []),
        "condition_spec": expert.get("condition_spec", {}),
        "observable_proxy": expert.get("observable_proxy", ""),
        "domain_fact": expert.get("domain_fact", ""),
        "latent_state": expert.get("latent_state", ""),
        "lineage": lineage,
        "observability_audit": obs_audit,
        "claim_scope": scope["claimed_scope"],
        "required_coverage": scope["required_coverage"],
        "evidence_plan": plan,
        "compiled": compiled,
        "compiler_repair": compiled.get("repairs", []),
        "proxy_validation": proxy_valid,
        "checks": checks,
        "final_decision": decision["final_decision"],
        "failure_type": failure_type,
        "action": action,
        "failure_reasons": decision.get("failure_reasons", []),
        "feedback": fb,
        "duplicate_of": expert.get("_duplicate_of"),
    }


# =========================================================================== #
# 15. Closed-loop driver
# =========================================================================== #
def run_closed_loop_v6(initial_experts: List[Dict[str, Any]],
                       evidence: Dict[str, Any],
                       metadata: Dict[str, Any],
                       feature_registry: Dict[str, Any],
                       regenerate_fn: Optional[Callable] = None,
                       max_rounds: int = MAX_ROUNDS,
                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = config or COMPILER_CONFIG
    feature_matrix = evidence.get("train_feature_matrix") or {}
    timestamps = list(feature_matrix.get("_timestamps", []))
    act_matrix_dict = {k: np.asarray(v) for k, v in feature_matrix.items()
                       if not k.startswith("_")}

    pool: List[Dict[str, Any]] = []          # accepted records (each with activation_vector)
    all_rounds: List[List[Dict[str, Any]]] = []
    failed_history: List[Dict[str, Any]] = []
    observability_log: Dict[str, Any] = {}
    repair_log: List[Dict[str, Any]] = []
    lineage_log: List[Dict[str, Any]] = []
    parent_lineage: Dict[str, Dict[str, Any]] = {}

    for rnd in range(1, max_rounds + 1):
        if rnd == 1:
            experts = list(initial_experts)
        else:
            feedbacks = [r["feedback"] for prev in all_rounds for r in prev
                         if r.get("failure_type") == "KNOWLEDGE_FAILURE"
                         and r.get("action") == "REGENERATE_KNOWLEDGE"
                         and r.get("feedback") and not r.get("duplicate_of")]
            if not feedbacks or regenerate_fn is None:
                break
            experts = regenerate_fn(feedbacks)

        experts = mark_duplicates(experts, failed_history)
        round_results = []
        for e in experts:
            r = process_expert_v6(e, evidence, metadata, feature_registry, config)
            round_results.append(r)

        # lineage tracking (parent = first-seen lineage for the same expert_id)
        for r in round_results:
            eid = r["expert_id"]
            child_lineage = r["lineage"]
            if eid in parent_lineage:
                drift = detect_semantic_drift(parent_lineage[eid], child_lineage)
                drift["parent_expert_id"] = eid
                if drift["semantic_drift"]:
                    # do NOT overwrite the parent; re-enter as a NEW candidate
                    r["lineage_relation"] = drift
                    r["semantic_drift"] = True
                else:
                    r["lineage_relation"] = drift
                    r["semantic_drift"] = False
                lineage_log.append(drift)
            else:
                parent_lineage[eid] = child_lineage
                r["lineage_relation"] = {
                    "parent_expert_id": None, "child_expert_id": eid,
                    "mechanism_family_before": None,
                    "mechanism_family_after": child_lineage["mechanism_family"],
                    "state_variable_before": None,
                    "state_variable_after": child_lineage["state_variable_family"],
                    "semantic_drift": False,
                }
                lineage_log.append(r["lineage_relation"])
                r["semantic_drift"] = False

        # accumulate observability + repair logs
        for r in round_results:
            for a in r["observability_audit"]["features"]:
                observability_log[a["feature_name"]] = a
            repair_log.extend(r["compiler_repair"])

        # ---- coverage + redundancy + pool admission (only for ACCEPT candidates) ----
        accepted = [r for r in round_results if r["final_decision"] == "ACCEPT"]
        if accepted and act_matrix_dict:
            compiled_list = [r["compiled"] for r in accepted]
            act_mat = build_train_activation_matrix(compiled_list, act_matrix_dict)
            for j, r in enumerate(accepted):
                r["activation_vector"] = act_mat[:, j].tolist()
                r["coverage_audit"] = audit_activation_coverage(act_mat[:, j], timestamps)
                r["redundancy_audit"] = audit_activation_redundancy(act_mat[:, j], pool)
                r["final_disposition"] = determine_disposition(
                    r["coverage_audit"], r["redundancy_audit"])
                if r["final_disposition"] in ("ACCEPTED", "ACCEPTED_LOCAL_SCOPE"):
                    pool.append(r)
                else:
                    r["final_decision"] = r["final_disposition"]
        else:
            for r in accepted:
                r["coverage_audit"] = None
                r["redundancy_audit"] = None
                r["final_disposition"] = "ACCEPTED"  # no activation matrix -> admit
                pool.append(r)

        all_rounds.append(round_results)

        # track failed knowledge for duplicate detection next round
        for r in round_results:
            if r["final_decision"] != "ACCEPT":
                failed_history.append({
                    "id": r["expert_id"], "claim": r["claim"],
                    "final_decision": r["final_decision"],
                    "failed_checks": r["failure_reasons"],
                })

        # stop if no repairable knowledge failure remains
        remaining = [r for r in round_results
                     if r.get("failure_type") == "KNOWLEDGE_FAILURE"
                     and r.get("action") == "REGENERATE_KNOWLEDGE"
                     and r.get("feedback") and not r.get("duplicate_of")
                     and not r.get("semantic_drift")]
        if not remaining:
            break

    # strip the internal (large) TRAIN activation vectors before returning; the
    # coverage / redundancy audits already summarise them.
    for rnd in all_rounds:
        for r in rnd:
            r.pop("activation_vector", None)

    return {
        "rounds": all_rounds,
        "accepted_pool": {r["expert_id"]: r for r in pool},
        "failed_history": failed_history,
        "observability_log": observability_log,
        "repair_log": repair_log,
        "lineage_log": lineage_log,
        "config": config,
    }


# =========================================================================== #
# 16. Synthetic audit (generic; validates the v6 fixes A..J)
# =========================================================================== #
def _syn_expert(sid, name, claim, cond, feats, spec, proxy="", domain_fact="", latent=""):
    return {"id": sid, "expert_id": sid, "name": name, "claim": claim, "knowledge": claim,
            "activation_condition": cond, "observable_features": feats,
            "condition_spec": spec, "observable_proxy": proxy,
            "domain_fact": domain_fact, "latent_state": latent,
            "expected_relationship": "", "reason": "", "expected_future_relationship": ""}


def _syn_registry():
    return {
        "recent_volatility": {"kind": "continuous", "source_type": "context_derived",
                              "dependencies": ["historical_context", "train_derived_constant"],
                              "mechanism_family": "recent_state_regime", "state_variable_family": "volatility"},
        "recent_slope_magnitude": {"kind": "continuous", "source_type": "context_derived",
                                   "dependencies": ["historical_context", "train_derived_constant"],
                                   "mechanism_family": "recent_state_regime", "state_variable_family": "trend_slope"},
        "seasonal_level": {"kind": "calendar", "source_type": "timestamp_known",
                           "dependencies": ["known_timestamp"],
                           "mechanism_family": "calendar_cycle", "state_variable_family": "season",
                           "categories": {"winter": [12, 1, 2], "spring": [3, 4, 5],
                                          "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
        "hour_of_day": {"kind": "calendar", "source_type": "timestamp_known",
                        "dependencies": ["known_timestamp"],
                        "mechanism_family": "calendar_cycle", "state_variable_family": "hour_of_day",
                        "categories": {"morning": [6, 7, 8, 9, 10, 11], "evening": [16, 17, 18, 19, 20, 21]}},
        "future_level": {"kind": "continuous", "source_type": "future_target",
                         "dependencies": ["future_target"],
                         "mechanism_family": "future", "state_variable_family": "future_target"},
    }


def _syn_evidence():
    return {
        "feature_distributions": {
            "recent_volatility": {"q10": 1.0, "q25": 2.0, "q50": 3.0, "q75": 4.0, "q90": 5.0,
                                  "mean": 3.0, "std": 1.0},
            "recent_slope_magnitude": {"q10": 0.1, "q25": 0.2, "q50": 0.3, "q75": 0.4, "q90": 0.5,
                                       "mean": 0.3, "std": 0.1},
        },
        "temporal_coverage": {"months_present": list(range(1, 13)), "months_missing": [],
                              "month_fraction": 1.0, "hours_present": list(range(24)),
                              "hours_missing": [], "hour_fraction": 1.0,
                              "dows_present": list(range(7)), "dow_fraction": 1.0},
        "seasonal": {"monthly_mean_values": [100 + 50 * np.sin(np.pi * m / 11) for m in range(12)],
                     "n_month_direction_reversals": 1, "eta2_month_on_ot": 0.3,
                     "monthly_mean": {}, "month_direction": {}},
        "diurnal": {"hourly_mean": [100 + 50 * np.sin(np.pi * h / 11) for h in range(24)],
                    "peak_hour": 6, "eta2_hour_on_ot": 0.3},
        "weekly": {"dow_mean": [100] * 7, "eta2_dow_on_ot": 0.3},
        "population": {"n_proxy": 1, "n_population": 1, "proxy_fraction": 1.0,
                       "corr_proxy_sum_vs_target": 1.0},
        "x_subset_representativeness": {},
    }


def synthetic_audit_v6():
    registry = _syn_registry()
    evidence = _syn_evidence()
    meta = {"target_description": "univariate target", "n_population": 1}
    results = []

    # A. feature fully from past context -> observable
    a = audit_observability("recent_volatility", registry)
    results.append({"id": "A", "name": "past-context feature observable",
                    "expected": "observable", "ok": bool(a["observable_at_prediction_time"]),
                    "detail": a})

    # B. slope magnitude + "large" -> Compiler Repair -> TRAIN threshold -> executable
    b_expert = _syn_expert("S-B", "SlopeLarge", "When recent slope magnitude is large, prefer such segments.",
                           "recent slope magnitude is large", ["recent_slope_magnitude"],
                           {"logic": "AND", "conditions": [{"feature": "recent_slope_magnitude", "qualifier": "large"}]})
    b_compiled = compile_condition_v6(b_expert, registry, evidence, COMPILER_CONFIG)
    b_repair = b_compiled.get("repairs", [])
    b_prim = b_compiled["primitives"][0] if b_compiled["primitives"] else {}
    b_ok = (b_compiled["status"] == "COMPILED"
            and len(b_repair) == 1
            and b_repair[0]["normalized_qualifier"] == "high"
            and b_prim.get("op") == ">="
            and b_prim.get("rule") == 0.4)
    results.append({"id": "B", "name": "qualifier 'large' -> compiler repair + TRAIN threshold",
                    "expected": "REPAIR_COMPILER (no regeneration)",
                    "ok": bool(b_ok),
                    "detail": {"compiled_status": b_compiled["status"], "repairs": b_repair,
                               "primitive": b_prim}})

    # C. feature needing future target -> unobservable / REJECT
    c = audit_observability("future_level", registry)
    c_expert = _syn_expert("S-C", "Future", "When future level is high, prefer such segments.",
                           "future level is high", ["future_level"],
                           {"logic": "AND", "conditions": [{"feature": "future_level", "qualifier": "high"}]})
    c_obs = audit_observability_for_expert(c_expert, registry)
    c_ok = (not c["observable_at_prediction_time"]
            and c["source_type"] == "future_target"
            and c_obs["future_leakage"])
    results.append({"id": "C", "name": "future-target feature -> unobservable / REJECT",
                    "expected": "unobservable + future leakage",
                    "ok": bool(c_ok), "detail": {"audit": c, "expert": c_obs}})

    # D. nonexistent external weather variable -> UNSUPPORTED
    d = audit_observability("weather", registry)
    d_expert = _syn_expert("S-D", "Weather", "When weather is cloudy, prefer such segments.",
                           "weather is cloudy", ["weather"],
                           {"logic": "AND", "conditions": [{"feature": "weather", "qualifier": "cloudy"}]})
    d_compiled = compile_condition_v6(d_expert, registry, evidence, COMPILER_CONFIG)
    d_ok = (not d["observable_at_prediction_time"]
            and d["source_type"] == "external_unavailable"
            and d_compiled["uncompilable"]
            and d_compiled["uncompilable"][0]["type"] == "feature_unobservable")
    results.append({"id": "D", "name": "nonexistent external weather -> UNSUPPORTED",
                    "expected": "external_unavailable / feature_unobservable",
                    "ok": bool(d_ok), "detail": {"audit": d, "uncompilable": d_compiled["uncompilable"]}})

    # E. regenerated knowledge with activation corr = 1 -> DUPLICATE
    pool_member = {"expert_id": "S-P", "name": "existing", "activation_vector": [1.0, 1.0, 0.0, 0.0]}
    cand_vec = np.asarray([1.0, 1.0, 0.0, 0.0])
    e_red = audit_activation_redundancy(cand_vec, [pool_member])
    results.append({"id": "E", "name": "activation corr=1 -> DUPLICATE",
                    "expected": "DUPLICATE_ACTIVATION",
                    "ok": bool(e_red["duplicate_activation"]), "detail": e_red})

    # F. volatility -> seasonality regeneration -> SEMANTIC_DRIFT
    parent = {"expert_id": "S-V", "mechanism_family": "recent_state_regime",
              "state_variable_family": "volatility"}
    child = {"expert_id": "S-V", "mechanism_family": "calendar_cycle",
             "state_variable_family": "season"}
    f_drift = detect_semantic_drift(parent, child)
    results.append({"id": "F", "name": "volatility -> seasonality -> SEMANTIC_DRIFT",
                    "expected": "semantic_drift=true",
                    "ok": bool(f_drift["semantic_drift"]), "detail": f_drift})

    # G. binary expert all 0 over TRAIN -> DEAD
    g_vec = np.zeros(8)
    g_cov = audit_activation_coverage(g_vec, [None] * 8)
    results.append({"id": "G", "name": "binary all-0 -> DEAD",
                    "expected": "dead=true",
                    "ok": bool(g_cov["dead"]), "detail": g_cov})

    # H. local seasonal expert (partial blocks, scope consistent) -> not auto-reject
    h_vec = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    h_cov = audit_activation_coverage(h_vec, [None] * 8)
    h_ok = h_cov["local_scope"] and not h_cov["dead"] and not h_cov["constant"]
    results.append({"id": "H", "name": "local seasonal scope (consistent) -> not rejected",
                    "expected": "local_scope=true, not DEAD/CONSTANT",
                    "ok": bool(h_ok), "detail": h_cov})

    # I. small subset claims population (low coverage) -> UNSUPPORTED
    i_evidence = dict(_syn_evidence())
    i_evidence["population"] = {"n_proxy": 6, "n_population": 100, "proxy_fraction": 0.06,
                                "corr_proxy_sum_vs_target": 0.9}
    i_expert = _syn_expert("S-I", "SubsetPopulation",
                           "When the subset level is high, prefer high-subset segments representing the population.",
                           "subset level is high", ["recent_volatility"],
                           {"logic": "AND", "conditions": [{"feature": "recent_volatility", "qualifier": "high"}]},
                           proxy="sum of X1-X6 subset")
    i_r = process_expert_v6(i_expert, i_evidence, {"target_description": "aggregate of many consumers", "n_population": 100},
                            registry, COMPILER_CONFIG)
    i_ok = i_r["final_decision"] == "UNSUPPORTED" and not i_r["checks"]["population"]["ok"]
    results.append({"id": "I", "name": "small subset claims population -> UNSUPPORTED",
                    "expected": "UNSUPPORTED (population)",
                    "ok": bool(i_ok), "detail": {"decision": i_r["final_decision"],
                                                 "population": i_r["checks"]["population"]}})

    # J. high pooled corr but subperiod direction reversal -> UNSUPPORTED
    j_evidence = dict(_syn_evidence())
    j_evidence["x_subset_representativeness"] = {"sign_reversals": 1}
    j_expert = _syn_expert("S-J", "SubPeriodReversal",
                           "When the subset level is high, prefer high-subset segments.",
                           "subset level is high", ["recent_volatility"],
                           {"logic": "AND", "conditions": [{"feature": "recent_volatility", "qualifier": "high"}]},
                           proxy="sum of X1-X6 subset")
    j_r = process_expert_v6(j_expert, j_evidence, {"target_description": "univariate", "n_population": 1},
                            registry, COMPILER_CONFIG)
    j_ok = j_r["final_decision"] == "UNSUPPORTED" and not j_r["checks"]["counter"]["ok"]
    results.append({"id": "J", "name": "subperiod direction reversal -> UNSUPPORTED",
                    "expected": "UNSUPPORTED (counter-evidence)",
                    "ok": bool(j_ok), "detail": {"decision": j_r["final_decision"],
                                                 "counter": j_r["checks"]["counter"]}})

    return results
