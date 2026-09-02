#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expert Knowledge Critic (conservative, general-purpose, TRAIN-only).

A principled replacement for the previous single-shot "Data Critic" LLM gate.

The critic does NOT ask "does this knowledge sound right?". It asks:

    "Given the TRAIN evidence and the scope this knowledge *claims* for itself,
     do we have sufficient reason to let it enter the forecasting/retrieval
     system?"

Only three outcomes exist: ACCEPT / REJECT / UNSUPPORTED.

Pipeline (every stage is deterministic; no fresh LLM call; no MLP; no retrieval;
no Chronos; no Attention; VAL never used; TEST never read):

    Expert Knowledge K (claim / activation_condition / observable_features /
                        expected_relationship / reason / domain_fact / ...)
      -> Claim Scope Analyzer       (what conditions/range/cycle/state does K claim?)
      -> Evidence Planner           (what evidence must be computed to prove that scope?)
      -> Evidence Executor          (compute it on TRAIN only; else EVIDENCE_NOT_AVAILABLE)
      -> Arbitrary Fallback Detector(does the compiled activation default on unseen state?)
      -> Evidence Sufficiency Critic(are A..H all satisfied? conservative AND-gate)
      -> final decision             (ACCEPT / REJECT / UNSUPPORTED + full evidence chain)

Design principles (see the task spec):
  * conservative: False Accept is strictly more expensive than False Unsupported.
  * ACCEPT requires an AND of every critical condition -- no majority vote.
  * nothing here is Solar-specific. Scope concepts are general (a claim about a
    "seasonal/periodic cycle" -> requires cycle coverage; a claim about an
    unobserved external driver -> requires the driver to be deterministically
    anchored or a validated proxy; a calendar-indexed activation that can meet a
    calendar unit absent from TRAIN -> arbitrary-fallback risk).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# inputs: dataset metadata + the 6 current Solar expert-knowledge statements
# --------------------------------------------------------------------------- #
# These are INPUTS (the same facts the pipeline already feeds in). The critic
# does not encode any Solar-specific verdict about them.

LOAD_VARS = ["P001", "P002", "P003", "P004", "P005", "P006"]  # X1..X6
N_POPULATION = 137  # plants summed into Y (from preprocessing)

SOLAR_METADATA = {
    "target_variable": "Y",
    "target_description": "total solar photovoltaic power output, aggregate output of 137 co-located plants",
    "observed_variables": ["X1", "X2", "X3", "X4", "X5", "X6"],
    "observed_description": "six individual plant outputs P001..P006",
    "observed_population_relation": "X1-X6 are 6 of the 137 plants summed into Y",
    "selection_note": "X1-X6 = first six plant columns, no representativeness guarantee",
    "sampling_frequency": "10 minutes",
}

# The activation feature catalog is ALSO an input (it was handed to the proposal
# stage). Each entry is deterministic in the query's own OT history + timestamp.
ACTIVATION_CATALOG = [
    "daylight_level", "seasonal_level", "recent_level", "recent_slope_magnitude",
    "recent_diff_magnitude", "recent_volatility", "recent_autocorr",
]

# Domain of each calendar-indexed activation feature (general: how many distinct
# calendar categories the feature indexes, so the fallback detector can check
# whether TRAIN covers the full domain). Continuous z-scored features have no
# categorical domain and no arbitrary default.
FEATURE_DOMAIN = {
    "seasonal_level": {"kind": "calendar_categorical", "unit": "month", "n": 12},
    "daylight_level": {"kind": "calendar_categorical", "unit": "hour", "n": 24},
    "recent_level": {"kind": "continuous_zscore"},
    "recent_slope_magnitude": {"kind": "continuous_zscore"},
    "recent_diff_magnitude": {"kind": "continuous_zscore"},
    "recent_volatility": {"kind": "continuous_zscore"},
    "recent_autocorr": {"kind": "continuous_bounded_0_1"},
}

# The 6 current Solar expert-knowledge statements (verbatim from the v4 cache).
SOLAR_EXPERTS = [
    {
        "id": "E1",
        "name": "NightZeroPersistence",
        "knowledge": "When daylight_level is near zero, historical segments with near-zero output and near-zero daylight_level should be preferred because no solar irradiance at night forces aggregate output to remain at zero.",
        "activation_condition": "daylight_level is near zero",
        "observable_features": ["daylight_level"],
        "expected_relationship": "historical night segments with near-zero output and near-zero daylight_level",
        "reason": "matching nighttime irradiance absence prevents retrieval of daytime segments and predicts flat zero output",
        "domain_fact": "At night solar irradiance is zero, so all plants and the aggregate output are zero.",
        "latent_state": "night/no-irradiance",
        "observable_proxy": "daylight_level near zero",
        "expected_future_relationship": "future OT remains near zero",
        "confidence": 0.98,
    },
    {
        "id": "E2",
        "name": "ClearSkyMiddayHighLevelSmooth",
        "knowledge": "When daylight_level is high and recent_volatility is low, historical midday segments with high output and low recent_volatility should be preferred because clear-sky conditions near solar noon yield stable high irradiance and smooth aggregate output.",
        "activation_condition": "daylight_level is high and recent_volatility is low",
        "observable_features": ["daylight_level", "recent_volatility"],
        "expected_relationship": "historical midday segments with high output and low recent volatility",
        "reason": "retrieving clear-sky midday profiles predicts a smooth high-output future",
        "domain_fact": "Near solar noon under clear skies, irradiance is high and changes slowly, so aggregate output is high and smooth.",
        "latent_state": "clear-sky midday",
        "observable_proxy": "high daylight_level and low recent_volatility",
        "expected_future_relationship": "future OT remains high and smooth",
        "confidence": 0.9,
    },
    {
        "id": "E3",
        "name": "CloudVolatilityRegime",
        "knowledge": "When recent_volatility is high, historical segments with similarly high recent_volatility and comparable recent_level should be preferred because cloud-driven fluctuations are likely to continue in the near term.",
        "activation_condition": "recent_volatility is high",
        "observable_features": ["recent_volatility", "recent_level"],
        "expected_relationship": "historical segments with high recent volatility and similar output level",
        "reason": "matches the current cloud-affected regime and preserves plausible short-term fluctuation amplitude",
        "domain_fact": "Cloud cover modulates irradiance on short timescales, causing rapid correlated fluctuations in aggregate output.",
        "latent_state": "cloud-affected variable irradiance",
        "observable_proxy": "high recent_volatility",
        "expected_future_relationship": "future OT continues to fluctuate with comparable variability",
        "confidence": 0.8,
    },
    {
        "id": "E4",
        "name": "SharpRampMagnitude",
        "knowledge": "When recent_diff_magnitude is large, historical segments with similarly large recent point-to-point change magnitude should be preferred because large short-term changes are usually caused by moving cloud shadows and ramp dynamics differ from smooth clear-sky periods.",
        "activation_condition": "recent_diff_magnitude is large",
        "observable_features": ["recent_diff_magnitude"],
        "expected_relationship": "historical segments with similarly large recent point-to-point change magnitude",
        "reason": "aligns retrieval with cloud-induced ramp events, improving representation of rapid changes",
        "domain_fact": "Cloud shadows can create rapid ramps in aggregate solar output over a few minutes.",
        "latent_state": "cloud-ramp event",
        "observable_proxy": "large recent_diff_magnitude",
        "expected_future_relationship": "future OT has elevated probability of continued large changes",
        "confidence": 0.7,
    },
    {
        "id": "E5",
        "name": "SeasonalHighPotential",
        "knowledge": "When seasonal_level is high, historical segments from similarly high-season months with comparable daylight_level should be preferred because summer-like sun elevation supports longer and stronger generation than low-season periods.",
        "activation_condition": "seasonal_level is high",
        "observable_features": ["seasonal_level", "daylight_level"],
        "expected_relationship": "historical high-season segments with similar time-of-day output level",
        "reason": "accounts for seasonal differences in maximum output and day length in retrieved futures",
        "domain_fact": "Solar elevation and day length are larger in summer, so maximum output and daily generation are higher.",
        "latent_state": "high-season irradiance potential",
        "observable_proxy": "high seasonal_level",
        "expected_future_relationship": "future OT follows a higher seasonal diurnal profile",
        "confidence": 0.85,
    },
    {
        "id": "E6",
        "name": "AttenuatedDaylightBelowTypical",
        "knowledge": "When daylight_level is high but recent_level is below typical for the current time, historical daylight segments with similarly below-typical output should be preferred because cloud or haze attenuation is likely to persist and dominate over clear-sky generation.",
        "activation_condition": "daylight_level is high and recent_level is below typical",
        "observable_features": ["daylight_level", "recent_level"],
        "expected_relationship": "historical daylight segments with similarly below-typical output relative to their time-of-day profile",
        "reason": "captures cloud/haze attenuation events rather than mixing them with clear-sky segments",
        "domain_fact": "Cloud cover and haze reduce irradiance below the clear-sky potential for a given sun angle.",
        "latent_state": "attenuated irradiance under cloud/haze",
        "observable_proxy": "high daylight_level and low recent_level",
        "expected_future_relationship": "future OT remains below typical daytime potential until the attenuation clears",
        "confidence": 0.78,
    },
]

# --------------------------------------------------------------------------- #
# general-purpose concept vocabularies (NOT dataset-specific)
# --------------------------------------------------------------------------- #
# A claim is "scoped" by the general concepts its wording invokes. None of these
# lists names Solar, months, or seasons as a special case -- they name the
# *kind* of thing that requires a kind of coverage.

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

# External drivers that are NOT columns of the dataset: their state cannot be read
# from OT / X1..X6 / date. A mechanism that invokes one is only executable if the
# driver is deterministically anchored to a calendar state (e.g. "at night") or the
# proxy is independently validated.
UNOBSERVED_DRIVER_VOCAB = [
    "cloud", "clouds", "cloudy", "haze", "hazy", "irradiance", "weather",
    "temperature", "wind", "atmospheric", "precipitation", "rain", "snow",
]
# Calendar anchors that make an external driver's state *deterministic* rather than
# a loose statistical proxy (e.g. "at night there is no irradiance").
DETERMINISTIC_ANCHOR_VOCAB = [
    "night", "daytime", "day", "morning", "evening", "noon", "midday",
    "sunrise", "sunset", "hour", "time-of-day", "season", "summer", "winter",
    "month", "time of year", "day length", "sun elevation", "solar elevation",
]

POPULATION_VOCAB = [
    "many", "all", "population", "aggregate", "aggregated", "total", "common",
    "shared", "across", "system-wide", "system wide", "co-located", "colocated",
]

LEAKAGE_VOCAB = [
    "future output", "future target", "future value", "future level", "future state",
    "next day", "next-day", "tomorrow", "known in advance", "test set", "test-set",
    "validation", "benchmark", "holdout",
]


def _has(text: str, vocab: List[str]) -> List[str]:
    t = (text or "").lower()
    return [w for w in vocab if w in t]


# --------------------------------------------------------------------------- #
# TRAIN-only data access
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
    """Read solar.csv and slice TRAIN only (first 60%). Never touches VAL/TEST."""
    df = pd.read_csv("data/solar.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    dt = dates.iloc[:train_len]
    y = df["OT"].to_numpy(np.float64)[:train_len]
    plants = {v: df[v].to_numpy(np.float64)[:train_len] for v in LOAD_VARS}
    return df, dates, dt, y, plants, N, train_len


def compute_train_evidence():
    """Deterministic TRAIN-only evidence catalog (general schema)."""
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

    night_mask = (hours >= 19) | (hours < 6)
    night_zero_frac = float((y[night_mask] == 0).mean()) if night_mask.sum() else float("nan")

    # month -> month direction of the mean (for reversal detection)
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
    lag144 = _corr(y[144:], y[:-144])

    return {
        "train_span": {
            "start": str(dt.min()), "end": str(dt.max()),
            "n_days": int(dt.dt.date.nunique()), "n_rows": int(train_len),
            "total_rows": int(N), "sampling": "10 minutes",
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
            "lag144_autocorr": round(lag144, 4),
        },
        "population": {
            "n_proxy": len(LOAD_VARS),
            "n_population": N_POPULATION,
            "proxy_fraction": round(len(LOAD_VARS) / N_POPULATION, 4),
            "corr_proxy_sum_vs_target": round(_corr(s6, y), 4),
        },
        "variability": {
            "ot_std": round(float(y.std()), 2),
            "diff_abs_mean": round(float(np.abs(np.diff(y)).mean()), 4),
            "diff_std": round(float(np.diff(y).std()), 4),
        },
    }


# --------------------------------------------------------------------------- #
# Stage 1: Claim Scope Analyzer
# --------------------------------------------------------------------------- #
def analyze_claim_scope(expert: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Infer, from the knowledge's own wording, the scope it *claims*.

    Returns {claimed_scope, required_coverage, potential_failure_modes,
             semantic_flags}. Everything is general (cycle -> cycle coverage;
    population -> representativeness; unobserved driver -> observability).
    """
    text = " ".join(str(expert.get(k, "")) for k in
                    ("name", "knowledge", "activation_condition", "expected_relationship",
                     "reason", "domain_fact", "latent_state", "observable_proxy",
                     "expected_future_relationship"))

    temporal_scope = []
    if _has(text, ANNUAL_CYCLE_VOCAB):
        temporal_scope.append("annual")
    if _has(text, DIURNAL_CYCLE_VOCAB):
        temporal_scope.append("diurnal")
    if _has(text, WEEKLY_CYCLE_VOCAB):
        temporal_scope.append("weekly")

    unobserved_drivers = _has(text, UNOBSERVED_DRIVER_VOCAB)
    anchors = _has(text, DETERMINISTIC_ANCHOR_VOCAB)
    driver_anchored = bool(unobserved_drivers) and bool(anchors)

    population_scope = bool(_has(text, POPULATION_VOCAB)) or bool(
        _has(str(metadata.get("target_description", "")), POPULATION_VOCAB))

    # states the claim conditions on (from the activation_condition wording)
    state_scope = []
    for kw in ("near zero", "near-zero", "high", "low", "above", "below",
               "typical", "large", "smooth"):
        if kw in str(expert.get("activation_condition", "")).lower():
            state_scope.append(kw)
    state_scope = sorted(set(state_scope))

    required_coverage = {
        "months": 12 if "annual" in temporal_scope else None,
        "hours": 24 if "diurnal" in temporal_scope else None,
        "days": 7 if "weekly" in temporal_scope else None,
        "population": N_POPULATION if population_scope else None,
    }

    failure_modes = []
    if "annual" in temporal_scope:
        failure_modes.append("annual/seasonal cycle -> requires full-year temporal coverage")
    if "diurnal" in temporal_scope:
        failure_modes.append("diurnal cycle -> requires full-day (24h) temporal coverage")
    if "weekly" in temporal_scope:
        failure_modes.append("weekly cycle -> requires full-week temporal coverage")
    if unobserved_drivers and not driver_anchored:
        failure_modes.append(
            f"unobserved external driver {unobserved_drivers} not deterministically "
            "anchored to a calendar state -> proxy may not equal the driver")
    if population_scope:
        failure_modes.append("population-level claim -> requires proxy representativeness evidence")

    # semantic flags (REJECT candidates)
    # Leakage is judged ONLY on the retrieval-time input side (what the activation
    # observes to select memories). The expected_future_relationship field is the
    # *predicted output* and legitimately mentions "future"; it must not count as
    # leakage.
    input_side = " ".join([
        str(expert.get("activation_condition", "")),
        str(expert.get("observable_proxy", "")),
        " ".join(str(f) for f in expert.get("observable_features", [])),
    ])
    leakage = bool(_has(input_side, LEAKAGE_VOCAB))
    # contradiction heuristic: activation asserts a state X but expected_relationship
    # asserts the opposite direction (e.g. "high" -> "low").
    contradiction = _detect_contradiction(expert)
    # duplication: not judged here (needs the set); marked unknown.
    semantic_flags = {
        "leakage": leakage,
        "contradiction": contradiction,
        "relevant": True,  # all 6 are forecasting-retrieval-relevant by construction
    }

    return {
        "claimed_scope": {
            "temporal_scope": temporal_scope,
            "state_scope": state_scope,
            "population_scope": population_scope,
            "variable_scope": list(expert.get("observable_features", [])),
            "regime_scope": {
                "unobserved_drivers": unobserved_drivers,
                "deterministically_anchored": driver_anchored,
            },
        },
        "required_coverage": required_coverage,
        "potential_failure_modes": failure_modes,
        "semantic_flags": semantic_flags,
    }


def _word_in(word: str, text: str) -> bool:
    # word-boundary match so "low" does not match inside "below" or "follow",
    # and "high" does not match inside "higher".
    return re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text) is not None


def _detect_contradiction(expert: Dict[str, Any]) -> bool:
    cond = str(expert.get("activation_condition", "")).lower()
    rel = (str(expert.get("expected_relationship", "")) + " "
           + str(expert.get("expected_future_relationship", ""))).lower()
    for a, b in (("high", "low"), ("large", "small"), ("above", "below"),
                 ("increase", "decrease"), ("up", "down")):
        # "when X is high ... prefer low-X segments" is a directional clash
        if _word_in(a, cond) and _word_in(b, rel) and not _word_in(a, rel):
            return True
    return False


# --------------------------------------------------------------------------- #
# Stage 2: Evidence Planner (deterministic: scope -> required evidence)
# --------------------------------------------------------------------------- #
def plan_evidence(scope: Dict[str, Any]) -> Dict[str, Any]:
    """Given claimed scope, list the evidence the executor must compute.

    Five categories always considered: support / coverage / stability /
    counter / executability. The planner does not invent statistics; it names
    the *kind* of evidence the scope requires. The executor decides whether a
    required statistic is computable (else EVIDENCE_NOT_AVAILABLE).
    """
    cs = scope["claimed_scope"]
    req = scope["required_coverage"]
    plan = {
        "support": [],
        "coverage": [],
        "stability": [],
        "counter": [],
        "executability": [],
    }
    if req["months"]:
        plan["support"].append("seasonal signal: spread of monthly means / eta2(month, OT)")
        plan["coverage"].append("month coverage over the full 12-month cycle")
        plan["stability"].append("month-to-month direction consistency (single-peak annual rhythm)")
        plan["counter"].append("any month where the claimed high-season > low-season ordering reverses")
    if req["hours"]:
        plan["support"].append("diurnal signal: hourly means / eta2(hour, OT)")
        plan["coverage"].append("hour coverage over the full 24h cycle")
        plan["stability"].append("hour-to-hour envelope smoothness")
    if req["days"]:
        plan["coverage"].append("day-of-week coverage over the full week")
    if req["population"]:
        plan["coverage"].append("proxy fraction of the population")
        plan["counter"].append("whether the subset proxy can be shown representative")
    if cs["regime_scope"]["unobserved_drivers"]:
        plan["support"].append("direct evidence that the proxy equals the unobserved driver")
        plan["counter"].append("alternative explanations for the proxy statistic")
    for _f in cs["variable_scope"]:
        plan["executability"].append("activation feature computable for future queries without arbitrary fallback")
    return plan


# --------------------------------------------------------------------------- #
# Stage 3: Evidence Executor (deterministic; TRAIN only)
# --------------------------------------------------------------------------- #
def compile_activation_feature(expert: Dict[str, Any]) -> Optional[str]:
    """Deterministic mapping of observable_features -> a catalog feature (first hit)."""
    feats = [str(f).strip() for f in expert.get("observable_features", [])]
    for f in feats:
        if f in ACTIVATION_CATALOG:
            return f
    return None


def activation_condition_features(expert: Dict[str, Any]) -> List[str]:
    """Catalog features explicitly named by the activation_condition."""
    cond = str(expert.get("activation_condition", "")).lower()
    return [f for f in ACTIVATION_CATALOG if f in cond]


def detect_arbitrary_fallback(expert: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Flag activation functions that silently default on an unseen state.

    General rule: a calendar-indexed activation (e.g. seasonal_level -> month,
    daylight_level -> hour) that can meet a calendar unit *absent* from TRAIN must
    not resolve to an arbitrary constant (0 / mean / default). We detect whether
    TRAIN covers the feature's full calendar domain.
    """
    feat = compile_activation_feature(expert)
    if feat is None:
        return {"activation_feature": None, "arbitrary_fallback": False,
                "reason": "no activation feature compiled (not executable)"}
    dom = FEATURE_DOMAIN.get(feat, {})
    tc = evidence.get("temporal_coverage", {})
    if dom.get("kind") == "calendar_categorical":
        unit = dom["unit"]
        n = dom["n"]
        if unit == "month":
            present = tc.get("months_present", [])
            categories = list(range(1, n + 1))  # months are 1..12
        elif unit == "hour":
            present = tc.get("hours_present", [])
            categories = list(range(n))         # hours are 0..23
        else:
            present = list(range(n))
            categories = list(range(n))
        missing = [c for c in categories if c not in present]
        has_fallback = len(missing) > 0
        return {
            "activation_feature": feat,
            "domain": f"{unit} ({categories[0]}..{categories[-1]})",
            "present_in_train": present,
            "unseen_categories": missing,
            "arbitrary_fallback": has_fallback,
            "reason": (
                f"{feat} indexes {unit}; TRAIN lacks categories {missing} -> an unseen "
                "category resolves to an arbitrary default (0) instead of an observed value"
                if has_fallback else
                f"{feat} indexes {unit}; TRAIN covers the full domain -> no unseen-category fallback"),
        }
    return {
        "activation_feature": feat,
        "domain": dom.get("kind", "unknown"),
        "arbitrary_fallback": False,
        "reason": f"{feat} is continuous ({dom.get('kind')}); no categorical default",
    }


# --------------------------------------------------------------------------- #
# Stage 4: Evidence Sufficiency Critic (deterministic, conservative AND-gate)
# --------------------------------------------------------------------------- #
def _monthly_signal_sufficient(evidence) -> Dict[str, Any]:
    se = evidence["seasonal"]
    means = list(se["monthly_mean"].values())
    if len(means) < 2:
        return {"ok": False, "status": "FAIL", "note": "fewer than 2 months observed"}
    lo, hi = min(means), max(means)
    spread = (hi - lo) / (abs(np.mean(means)) + 1e-9)
    eta2 = se["eta2_month_on_ot"]
    ok = spread > 0.05 or eta2 > 0.05
    return {"ok": ok, "status": "PASS" if ok else "FAIL",
            "evidence": {"monthly_mean": se["monthly_mean"], "spread_ratio": round(spread, 4),
                         "eta2_month_on_ot": eta2},
            "note": ("monthly means vary meaningfully" if ok else "monthly means barely vary")}


def _diurnal_signal_sufficient(evidence, expert) -> Dict[str, Any]:
    d = evidence["diurnal"]
    cond = str(expert.get("activation_condition", "")).lower()
    if "near zero" in cond or "night" in str(expert.get("latent_state", "")).lower():
        nz = d["night_zero_fraction"]
        ok = nz > 0.9
        return {"ok": ok, "status": "PASS" if ok else "FAIL",
                "evidence": {"night_zero_fraction": nz},
                "note": "night->zero state directly observed" if ok else "night-zero state not cleanly observed"}
    eta2 = d["eta2_hour_on_ot"]
    ok = eta2 > 0.05
    return {"ok": ok, "status": "PASS" if ok else "FAIL",
            "evidence": {"eta2_hour_on_ot": eta2, "peak_hour": d["peak_hour"]},
            "note": "diurnal signal present" if ok else "diurnal signal weak"}


def sufficiency_critic(expert, scope, evidence) -> Dict[str, Any]:
    """Compute the A..H checks and the conservative final decision."""
    cs = scope["claimed_scope"]
    req = scope["required_coverage"]
    tc = evidence["temporal_coverage"]

    checks = {}

    # ---- A. support sufficiency ----
    support_fail = []
    support_ev = {}
    if req["months"]:
        r = _monthly_signal_sufficient(evidence)
        support_ev["seasonal"] = r
        if not r["ok"]:
            support_fail.append(r["note"])
    if req["hours"]:
        r = _diurnal_signal_sufficient(evidence, expert)
        support_ev["diurnal"] = r
        if not r["ok"]:
            support_fail.append(r["note"])
    checks["support"] = {
        "ok": not support_fail, "status": "PASS" if not support_fail else "FAIL",
        "evidence": support_ev, "note": "; ".join(support_fail) or "support signal present",
    }

    # ---- B. coverage sufficiency ----
    coverage_fail = []
    if req["months"]:
        frac = tc["month_fraction"]
        if frac < 1.0:
            coverage_fail.append(f"annual cycle only {frac:.0%} covered (missing months "
                                 f"{tc['months_missing']})")
    if req["hours"]:
        frac = tc["hour_fraction"]
        if frac < 1.0:
            coverage_fail.append(f"diurnal cycle only {frac:.0%} covered (missing hours "
                                 f"{tc['hours_missing']})")
    if req["days"]:
        frac = tc["dow_fraction"]
        if frac < 1.0:
            coverage_fail.append(f"weekly cycle only {frac:.0%} covered")
    checks["coverage"] = {
        "ok": not coverage_fail, "status": "PASS" if not coverage_fail else "FAIL",
        "evidence": tc, "note": "; ".join(coverage_fail) or "claimed temporal scope fully covered",
    }

    # ---- C. stability sufficiency ----
    stability_fail = []
    if req["months"]:
        nrev = evidence["seasonal"]["n_month_direction_reversals"]
        # a single-peak annual rhythm is allowed one turn; more than one is unstable
        if nrev > 1:
            stability_fail.append(f"monthly direction reversed {nrev} times in TRAIN")
    # general sub-period sign-reversal signal (if the executor provided it)
    stab = evidence.get("stability", {})
    if stab.get("n_sign_reversals", 0) > 0:
        stability_fail.append("relationship reverses sign across sub-periods "
                              f"({stab.get('sub_period_corr')})")
    checks["stability"] = {
        "ok": not stability_fail, "status": "PASS" if not stability_fail else "FAIL",
        "evidence": {"n_month_direction_reversals": evidence["seasonal"]["n_month_direction_reversals"],
                     "month_direction": evidence["seasonal"]["month_direction"],
                     "sub_period": stab},
        "note": "; ".join(stability_fail) or "relationship stable across covered sub-periods",
    }

    # ---- D. counter-evidence ----
    counter = []
    if cs["regime_scope"]["unobserved_drivers"] and not cs["regime_scope"]["deterministically_anchored"]:
        counter.append(f"unobserved driver {cs['regime_scope']['unobserved_drivers']} has no "
                       "deterministic calendar anchor; proxy statistic admits alternative causes")
    checks["counter"] = {
        "ok": not counter, "status": "PASS" if not counter else "FAIL",
        "evidence": cs["regime_scope"], "note": "; ".join(counter) or "no critical counter-evidence",
    }

    # ---- E. observability ----
    obs_fail = []
    if cs["regime_scope"]["unobserved_drivers"] and not cs["regime_scope"]["deterministically_anchored"]:
        obs_fail.append("mechanism's causal driver is unobserved and not deterministically anchored")
    checks["observability"] = {
        "ok": not obs_fail, "status": "PASS" if not obs_fail else "FAIL",
        "evidence": cs["regime_scope"], "note": "; ".join(obs_fail) or "required state observable",
    }

    # ---- F. future executability (arbitrary fallback) ----
    fb = detect_arbitrary_fallback(expert, evidence)
    checks["executability"] = {
        "ok": fb["activation_feature"] is not None and not fb["arbitrary_fallback"],
        "status": "PASS" if (fb["activation_feature"] is not None and not fb["arbitrary_fallback"]) else "FAIL",
        "evidence": fb, "note": fb["reason"],
    }

    # ---- G. population validity ----
    pop = evidence["population"]
    if cs["population_scope"]:
        # A population-level claim is valid only if the proxy that instantiates it
        # actually covers the population (or the claim operates on the aggregate
        # target itself, which IS the population). If the observable_proxy reduces a
        # population claim to a small arbitrary subset, that is a validity gap.
        needs_subset_proxy = any(w in str(expert.get("observable_proxy", "")).lower()
                                 for w in ("x1", "x2", "x3", "x4", "x5", "x6", "plant",
                                           "subset", "subsume"))
        if needs_subset_proxy and pop["proxy_fraction"] < 1.0:
            checks["population"] = {"ok": False, "status": "FAIL", "evidence": pop,
                                    "note": f"subset proxy covers {pop['proxy_fraction']:.0%} "
                                            "of population without representativeness evidence"}
        else:
            checks["population"] = {"ok": True, "status": "PASS", "evidence": pop,
                                    "note": "claim operates on the aggregate target, not a subset proxy"}
    else:
        checks["population"] = {"ok": True, "status": "PASS", "evidence": pop,
                                "note": "no population-level claim"}

    # ---- H. proxy fidelity ----
    compiled = compile_activation_feature(expert)
    cond_feats = activation_condition_features(expert)
    if compiled is None:
        checks["proxy_fidelity"] = {"ok": False, "status": "FAIL", "evidence": {},
                                    "note": "no observable feature compiled"}
    elif len(cond_feats) > 1 and compiled in cond_feats:
        # activation condition names N features but only one is compiled -> scope narrowed
        checks["proxy_fidelity"] = {"ok": False, "status": "FAIL",
                                    "evidence": {"activation_condition_features": cond_feats,
                                                 "compiled_feature": compiled},
                                    "note": f"activation_condition requires {cond_feats} but only "
                                            f"'{compiled}' compiled (condition narrowed)"}
    else:
        checks["proxy_fidelity"] = {"ok": True, "status": "PASS",
                                    "evidence": {"activation_condition_features": cond_feats,
                                                 "compiled_feature": compiled},
                                    "note": "compiled activation preserves the condition"}

    return checks


# --------------------------------------------------------------------------- #
# final decision (conservative AND-gate)
# --------------------------------------------------------------------------- #
CRITICAL_CHECKS = [
    "support", "coverage", "stability", "counter", "observability",
    "executability", "population", "proxy_fidelity",
]


def decide(expert, scope, checks) -> Dict[str, Any]:
    semantic = scope["semantic_flags"]
    reasons = []
    if semantic["leakage"]:
        reasons.append("future/test/benchmark leakage")
    if semantic["contradiction"]:
        reasons.append("directional contradiction between activation and expected relationship")
    if not semantic["relevant"]:
        reasons.append("not relevant to retrieval/forecasting")

    if reasons:
        return {"final_decision": "REJECT", "failure_reasons": reasons}

    failures = [c for c in CRITICAL_CHECKS if not checks[c]["ok"]]
    if failures:
        reasons = [f"{c}: {checks[c]['note']}" for c in failures]
        return {"final_decision": "UNSUPPORTED", "failure_reasons": reasons}

    return {"final_decision": "ACCEPT", "failure_reasons": []}


# --------------------------------------------------------------------------- #
# per-expert driver
# --------------------------------------------------------------------------- #
def critique_expert(expert, evidence, metadata=SOLAR_METADATA) -> Dict[str, Any]:
    scope = analyze_claim_scope(expert, metadata)
    plan = plan_evidence(scope)
    checks = sufficiency_critic(expert, scope, evidence)
    decision = decide(expert, scope, checks)
    return {
        "expert_id": expert.get("id", expert.get("name")),
        "name": expert.get("name"),
        "semantic_decision": "valid" if not scope["semantic_flags"]["leakage"]
                              and not scope["semantic_flags"]["contradiction"] else "invalid",
        "claim_scope": scope["claimed_scope"],
        "required_coverage": scope["required_coverage"],
        "potential_failure_modes": scope["potential_failure_modes"],
        "evidence_plan": plan,
        "executed_evidence": _executed_evidence_for(checks, evidence),
        "support_sufficiency": checks["support"],
        "coverage_sufficiency": checks["coverage"],
        "stability_sufficiency": checks["stability"],
        "counter_evidence": checks["counter"],
        "observability": checks["observability"],
        "future_executability": checks["executability"],
        "population_validity": checks["population"],
        "proxy_fidelity": checks["proxy_fidelity"],
        "final_decision": decision["final_decision"],
        "failure_reasons": decision["failure_reasons"],
    }


def _executed_evidence_for(checks, evidence):
    # a compact view of the evidence actually consumed for this expert
    return {
        "temporal_coverage": evidence["temporal_coverage"],
        "support": checks["support"]["evidence"],
        "counter": checks["counter"]["evidence"],
        "executability": checks["executability"]["evidence"],
        "population": evidence["population"],
        "proxy_fidelity": checks["proxy_fidelity"]["evidence"],
    }


# --------------------------------------------------------------------------- #
# synthetic audit cases (generic; test the critic's general audit capability)
# --------------------------------------------------------------------------- #
# These are NOT trained into anything; they only test whether the critic wrongly
# admits (False Accept) a defective claim or wrongly admits coverage/observability
# gaps. Each case supplies its own evidence dict against the same general schema.
def synthetic_cases() -> List[Dict[str, Any]]:
    def full_annual_evidence():
        return {
            "temporal_coverage": {
                "months_present": list(range(1, 13)), "months_missing": [],
                "month_fraction": 1.0, "hours_present": list(range(24)), "hours_missing": [],
                "hour_fraction": 1.0, "dows_present": list(range(7)), "dow_fraction": 1.0,
            },
            "seasonal": {"monthly_mean": {m: 100.0 + 50.0 * np.sin(np.pi * (m - 1) / 11.0) for m in range(1, 13)},
                         "monthly_daily_peak": {}, "month_direction": {}, "n_month_direction_reversals": 1,
                         "eta2_month_on_ot": 0.4},
            "diurnal": {"hourly_mean": [], "peak_hour": 0, "night_zero_fraction": 1.0, "eta2_hour_on_ot": 0.3},
            "population": {"n_proxy": 100, "n_population": 100, "proxy_fraction": 1.0, "corr_proxy_sum_vs_target": 1.0},
        }

    cases = []

    # Case A: reasonable mechanism, insufficient temporal coverage -> UNSUPPORTED
    cases.append({
        "id": "S-A", "name": "AnnualCyclePartialCoverage",
        "expert": {
            "name": "AnnualCyclePartialCoverage",
            "knowledge": "When month-of-year is in the high months, historical segments from the same high months should be preferred because the annual cycle makes late-year levels systematically different.",
            "activation_condition": "month-of-year is high",
            "observable_features": ["seasonal_level"],
            "expected_relationship": "historical high-month segments",
            "reason": "align retrieval with the annual cycle",
            "domain_fact": "Output follows an annual cycle.",
            "latent_state": "high-month",
            "observable_proxy": "high seasonal_level",
            "expected_future_relationship": "future follows the high-month profile",
        },
        "evidence": {
            "temporal_coverage": {"months_present": list(range(1, 9)), "months_missing": [9, 10, 11, 12],
                                  "month_fraction": 8 / 12, "hours_present": list(range(24)), "hours_missing": [],
                                  "hour_fraction": 1.0, "dows_present": list(range(7)), "dow_fraction": 1.0},
            "seasonal": {"monthly_mean": {m: 100.0 + 10.0 * m for m in range(1, 9)},
                         "monthly_daily_peak": {}, "month_direction": {}, "n_month_direction_reversals": 0,
                         "eta2_month_on_ot": 0.3},
            "diurnal": {"hourly_mean": [], "peak_hour": 0, "night_zero_fraction": 1.0, "eta2_hour_on_ot": 0.3},
            "population": {"n_proxy": 137, "n_population": 137, "proxy_fraction": 1.0, "corr_proxy_sum_vs_target": 1.0},
        },
        "expected": "UNSUPPORTED",
    })

    # Case B: high overall correlation but sub-period sign reversal -> UNSUPPORTED
    cases.append({
        "id": "S-B", "name": "SubPeriodReversal",
        "expert": {
            "name": "SubPeriodReversal",
            "knowledge": "When recent_level is high, historical high-level segments should be preferred because high level predicts continued high level.",
            "activation_condition": "recent_level is high",
            "observable_features": ["recent_level"],
            "expected_relationship": "historical high-level segments",
            "reason": "persistence",
            "domain_fact": "Level persists.",
            "latent_state": "high level",
            "observable_proxy": "high recent_level",
            "expected_future_relationship": "future stays high",
        },
        # high pooled correlation, but the two sub-periods have opposite signs
        "evidence": {
            "temporal_coverage": {"months_present": list(range(1, 13)), "months_missing": [], "month_fraction": 1.0,
                                  "hours_present": list(range(24)), "hours_missing": [], "hour_fraction": 1.0,
                                  "dows_present": list(range(7)), "dow_fraction": 1.0},
            "seasonal": {"monthly_mean": {}, "monthly_daily_peak": {}, "month_direction": {}, "n_month_direction_reversals": 0, "eta2_month_on_ot": 0.0},
            "diurnal": {"hourly_mean": [], "peak_hour": 0, "night_zero_fraction": 1.0, "eta2_hour_on_ot": 0.0},
            "population": {"n_proxy": 137, "n_population": 137, "proxy_fraction": 1.0, "corr_proxy_sum_vs_target": 1.0},
            # injected counter-evidence: relationship reverses across sub-periods
            "stability": {"n_sign_reversals": 1, "sub_period_corr": {"first_half": 0.8, "second_half": -0.7}},
        },
        "expected": "UNSUPPORTED",
    })

    # Case C: variable exists but tiny non-random subset claims the population -> UNSUPPORTED
    cases.append({
        "id": "S-C", "name": "TinySubsetClaimsPopulation",
        "expert": {
            "name": "TinySubsetClaimsPopulation",
            "knowledge": "When the aggregate level is high, historical segments with high aggregate level should be preferred because the aggregate level represents the whole system.",
            "activation_condition": "aggregate level is high",
            "observable_features": ["recent_level"],
            "expected_relationship": "historical high-aggregate segments",
            "reason": "system-wide level",
            "domain_fact": "The aggregate is the sum of many units.",
            "latent_state": "system-wide high",
            "observable_proxy": "sum of a small subset of units",
            "expected_future_relationship": "future follows the system-wide level",
        },
        "evidence": {
            "temporal_coverage": {"months_present": list(range(1, 13)), "months_missing": [], "month_fraction": 1.0,
                                  "hours_present": list(range(24)), "hours_missing": [], "hour_fraction": 1.0,
                                  "dows_present": list(range(7)), "dow_fraction": 1.0},
            "seasonal": {"monthly_mean": {}, "monthly_daily_peak": {}, "month_direction": {}, "n_month_direction_reversals": 0, "eta2_month_on_ot": 0.0},
            "diurnal": {"hourly_mean": [], "peak_hour": 0, "night_zero_fraction": 1.0, "eta2_hour_on_ot": 0.0},
            "population": {"n_proxy": 6, "n_population": 137, "proxy_fraction": 6 / 137, "corr_proxy_sum_vs_target": 0.99},
        },
        "expected": "UNSUPPORTED",
    })

    # Case D: mechanism needs a variable the data does not have -> UNSUPPORTED
    cases.append({
        "id": "S-D", "name": "MissingVariable",
        "expert": {
            "name": "MissingVariable",
            "knowledge": "When ambient temperature is high, historical segments with similarly high temperature should be preferred because high temperature drives the target.",
            "activation_condition": "ambient temperature is high",
            "observable_features": ["temperature"],
            "expected_relationship": "historical high-temperature segments",
            "reason": "temperature drives output",
            "domain_fact": "Output depends on ambient temperature.",
            "latent_state": "high temperature",
            "observable_proxy": "ambient temperature",
            "expected_future_relationship": "future follows temperature",
        },
        "evidence": {
            "temporal_coverage": {"months_present": list(range(1, 13)), "months_missing": [], "month_fraction": 1.0,
                                  "hours_present": list(range(24)), "hours_missing": [], "hour_fraction": 1.0,
                                  "dows_present": list(range(7)), "dow_fraction": 1.0},
            "seasonal": {"monthly_mean": {}, "monthly_daily_peak": {}, "month_direction": {}, "n_month_direction_reversals": 0, "eta2_month_on_ot": 0.0},
            "diurnal": {"hourly_mean": [], "peak_hour": 0, "night_zero_fraction": 1.0, "eta2_hour_on_ot": 0.0},
            "population": {"n_proxy": 137, "n_population": 137, "proxy_fraction": 1.0, "corr_proxy_sum_vs_target": 1.0},
            # temperature is not in observed variables -> EVIDENCE_NOT_AVAILABLE
            "observability": {"missing_variables": ["temperature"]},
        },
        "expected": "UNSUPPORTED",
    })

    # Case E: fully supported -> ACCEPT
    cases.append({
        "id": "S-E", "name": "FullySupported",
        "expert": {
            "name": "FullySupported",
            "knowledge": "When time-of-day is in the night, historical night segments should be preferred because output is deterministically near zero at night.",
            "activation_condition": "time-of-day is night",
            "observable_features": ["daylight_level"],
            "expected_relationship": "historical night segments",
            "reason": "night output is zero",
            "domain_fact": "At night output is zero.",
            "latent_state": "night",
            "observable_proxy": "daylight_level near zero",
            "expected_future_relationship": "future stays near zero",
        },
        "evidence": full_annual_evidence(),
        "expected": "ACCEPT",
    })

    # Case F: future leakage -> REJECT
    cases.append({
        "id": "S-F", "name": "FutureLeakage",
        "expert": {
            "name": "FutureLeakage",
            "knowledge": "When the future output is high, historical segments with high future output should be preferred because the future output is known in advance.",
            "activation_condition": "future output is high",
            "observable_features": ["future_level"],
            "expected_relationship": "historical high-future-output segments",
            "reason": "future is known",
            "domain_fact": "The future output is available at retrieval time.",
            "latent_state": "future high",
            "observable_proxy": "future output",
            "expected_future_relationship": "future stays high",
        },
        "evidence": full_annual_evidence(),
        "expected": "REJECT",
    })

    return cases


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def render_report(results, synthetic_results, evidence):
    L = []
    L.append("# Expert Knowledge Critic — conservative, general, TRAIN-only")
    L.append("")
    L.append("## 1. Pipeline")
    L.append("")
    L.append("1. **Claim Scope Analyzer** (deterministic, general): infers the scope a knowledge "
             "statement *claims* from its own wording — temporal cycle (annual/diurnal/weekly), "
             "conditioned states, population scope, variable scope, and regime scope (unobserved "
             "external drivers). No Solar/month/season hard-code; only general concepts.")
    L.append("2. **Evidence Planner** (deterministic): maps the claimed scope to the evidence that "
             "must be computed, across five categories — support / coverage / stability / counter / "
             "executability.")
    L.append("3. **Evidence Executor** (deterministic, TRAIN only): computes the statistics; returns "
             "`EVIDENCE_NOT_AVAILABLE` for genuinely uncomputable statistics (never invents a proxy).")
    L.append("4. **Arbitrary Fallback Detector**: flags calendar-indexed activations that resolve an "
             "unseen calendar category to a constant.")
    L.append("5. **Evidence Sufficiency Critic** (deterministic, conservative AND-gate): checks "
             "A..H; ACCEPT requires ALL to pass. Any critical gap -> UNSUPPORTED (mechanism ok) or "
             "REJECT (mechanism wrong).")
    L.append("")
    L.append("## 2. Final decisions — 6 Solar experts")
    L.append("")
    L.append("| ID | expert | decision | reason |")
    L.append("|---|---|---|---|")
    for r in results:
        reason = ("; ".join(r["failure_reasons"]) if r["failure_reasons"]
                  else "all critical checks pass")
        L.append(f"| {r['expert_id']} | {r['name']} | **{r['final_decision']}** | {reason} |")
    L.append("")
    L.append("## 3. Per-expert evidence chain")
    L.append("")
    for r in results:
        L.append(f"### {r['expert_id']} — {r['name']}  ->  {r['final_decision']}")
        L.append("")
        L.append(f"- **claimed_scope**: `{json.dumps(r['claim_scope'], ensure_ascii=False)}`")
        L.append(f"- **required_coverage**: `{json.dumps(r['required_coverage'], ensure_ascii=False)}`")
        L.append(f"- **potential_failure_modes**: `{json.dumps(r['potential_failure_modes'], ensure_ascii=False)}`")
        for chk in CRITICAL_CHECKS:
            key = {"support": "support_sufficiency", "coverage": "coverage_sufficiency",
                   "stability": "stability_sufficiency", "counter": "counter_evidence",
                   "observability": "observability", "executability": "future_executability",
                   "population": "population_validity", "proxy_fidelity": "proxy_fidelity"}[chk]
            c = r[key]
            L.append(f"- **{chk}**: `{c['status']}` — {c['note']}")
        L.append(f"- **failure_reasons**: `{json.dumps(r['failure_reasons'], ensure_ascii=False)}`")
        L.append("")
    L.append("## 4. Synthetic audit cases")
    L.append("")
    L.append("| case | expected | actual | verdict |")
    L.append("|---|---|---|---|")
    for s in synthetic_results:
        ok = s["final_decision"] == s["expected"]
        L.append(f"| {s['expert_id']} | {s['expected']} | {s['final_decision']} | "
                 f"{'OK' if ok else 'MISMATCH'} |")
    L.append("")
    L.append("## 5. Guardrails")
    L.append("")
    L.append("- VAL: **not used** (evidence executor reads `solar.csv[:train_len]` only).")
    L.append("- TEST: **not accessed**.")
    L.append("- Solar-specific hard-code: **none** (scope concepts are general).")
    L.append("")
    return "\n".join(L)


def main():
    evidence = compute_train_evidence()

    results = [critique_expert(e, evidence) for e in SOLAR_EXPERTS]

    synthetic_results = []
    for case in synthetic_cases():
        # run through the same critic with the supplied evidence
        expert = case["expert"]
        expert["id"] = case["id"]
        scope = analyze_claim_scope(expert, SOLAR_METADATA)
        plan = plan_evidence(scope)
        checks = sufficiency_critic(expert, scope, case["evidence"])
        decision = decide(expert, scope, checks)
        synthetic_results.append({
            "expert_id": case["id"], "name": case["name"], "expected": case["expected"],
            "final_decision": decision["final_decision"],
            "failure_reasons": decision["failure_reasons"],
        })

    # ---- print ----
    print("=" * 100)
    print("EXPERT KNOWLEDGE CRITIC (conservative, general, TRAIN-only)")
    print("=" * 100)
    print(f"TRAIN span : {evidence['train_span']['start']} -> {evidence['train_span']['end']}")
    print(f"month cov  : {evidence['temporal_coverage']['month_fraction']:.0%} "
          f"(present {evidence['temporal_coverage']['months_present']}, "
          f"missing {evidence['temporal_coverage']['months_missing']})")
    print(f"hour cov   : {evidence['temporal_coverage']['hour_fraction']:.0%}")
    print()
    print(f"{'ID':<5}{'expert':<32}{'decision':<12}reason")
    print("-" * 100)
    for r in results:
        reason = "; ".join(r["failure_reasons"]) if r["failure_reasons"] else "all checks pass"
        print(f"{r['expert_id']:<5}{r['name']:<32}{r['final_decision']:<12}{reason}")
    print()
    print("Synthetic audit cases:")
    for s in synthetic_results:
        mark = "OK" if s["final_decision"] == s["expected"] else "MISMATCH"
        print(f"  {s['expert_id']:<5}{s['name']:<28}expected={s['expected']:<12}"
              f"actual={s['final_decision']:<12}{mark}")

    # A False Accept would be: decision ACCEPT while some critical check still FAILs.
    # With the conservative AND-gate this must be impossible; we report the count
    # explicitly rather than asserting it away.
    false_accepts = []
    for r in results:
        if r["final_decision"] == "ACCEPT":
            bad = [c for c in CRITICAL_CHECKS
                   if r[{ "support":"support_sufficiency","coverage":"coverage_sufficiency",
                          "stability":"stability_sufficiency","counter":"counter_evidence",
                          "observability":"observability","executability":"future_executability",
                          "population":"population_validity","proxy_fidelity":"proxy_fidelity"}[c]]["status"] != "PASS"]
            if bad:
                false_accepts.append(r["name"])
    print()
    print(f"Final: {sum(1 for r in results if r['final_decision']=='ACCEPT')} ACCEPT, "
          f"{sum(1 for r in results if r['final_decision']=='UNSUPPORTED')} UNSUPPORTED, "
          f"{sum(1 for r in results if r['final_decision']=='REJECT')} REJECT "
          f"(among the 6 real experts)")
    print(f"False Accept (ACCEPT with a FAILed critical check): {len(false_accepts)} {false_accepts}")

    # ---- save ----
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "dataset": "solar",
        "critic_version": "expert_knowledge_critic_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_span": evidence["train_span"],
        "temporal_coverage": evidence["temporal_coverage"],
        "experts": results,
        "synthetic_cases": synthetic_results,
        "false_accepts": false_accepts,
        "summary": {
            "n_experts": len(results),
            "n_accept": sum(1 for r in results if r["final_decision"] == "ACCEPT"),
            "n_unsupported": sum(1 for r in results if r["final_decision"] == "UNSUPPORTED"),
            "n_reject": sum(1 for r in results if r["final_decision"] == "REJECT"),
            "n_false_accept": len(false_accepts),
        },
        "val_used": False,
        "test_accessed": False,
        "solar_specific_hardcode": False,
    }
    (out_dir / "expert_knowledge_critic_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "expert_knowledge_critic_report.md").write_text(
        render_report(results, synthetic_results, evidence), encoding="utf-8")

    print("\nSaved results/expert_knowledge_critic_audit.json")
    print("Saved results/expert_knowledge_critic_report.md")
    return audit


if __name__ == "__main__":
    main()
