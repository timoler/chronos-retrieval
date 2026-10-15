# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional
import numpy as np

# Source: experiments/run_expert_agent_closed_loop_v6.py:56-60
COMPILER_CONFIG: Dict[str, Any] = {
    "LOW_Q": 0.25,
    "HIGH_Q": 0.75,
    "MEDIUM": [0.25, 0.75],
}

# Source: experiments/run_expert_agent_closed_loop_v6.py:96-108
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

# Source: experiments/run_expert_agent_closed_loop_v6.py:110-110
CONTINUOUS_CANONICAL_QUALIFIERS = {"high", "low", "medium"}

# Source: experiments/run_expert_agent_closed_loop_v6.py:122-123
def _q_key(q: float) -> str:
    return f"q{int(round(q * 100))}"

# Source: experiments/run_expert_agent_closed_loop_v6.py:126-132
def _normalize_qualifier(q: Any) -> Optional[str]:
    if q is None:
        return None
    q = str(q).strip().lower()
    if q.startswith("is_"):
        q = q[3:]
    return QUALIFIER_SYNONYMS.get(q, q)

# Source: experiments/run_expert_agent_closed_loop_v6.py:362-381
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

# Source: experiments/run_expert_agent_closed_loop_v6.py:384-427
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

# Source: experiments/run_expert_agent_closed_loop_v6.py:430-465
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
