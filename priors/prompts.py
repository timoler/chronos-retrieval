"""Prompt builders for the three LLM prior conditions.

The LLM's task is ONLY to propose candidate retrieval priors. It must not
forecast, must not pick fusion weights / hyperparameters, and must never see or
be conditioned on test / benchmark performance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .schema import ALLOWED_TYPES

PROMPT_VERSION = "v1"

# Generic (built-in) expert guidance, used by the "Generic Expert-guided LLM"
# condition. This is a domain-agnostic instruction, not a per-dataset hint.
GENERIC_EXPERT_HINT = (
    "Reason from domain knowledge about which temporal, operating, trend, state, "
    "environmental, or event relationships may make one historical segment useful "
    "for predicting another."
)

_SYSTEM = (
    "You are a time-series domain expert and retrieval-system designer.\n\n"
    "Your job is to propose CANDIDATE RETRIEVAL PRIORS: structured hypotheses about "
    "what kinds of historical relationships are plausibly useful when retrieving "
    "predictive historical examples for a target variable.\n\n"
    "You are NOT asked to, and must NOT:\n"
    "- predict the future or produce forecast values\n"
    "- choose fusion weights (e.g. a lambda) or any model hyperparameter\n"
    "- use, assume, or optimize against any benchmark / test performance\n\n"
    "Your priors are HYPOTHESES to be checked against numerical training data later, "
    "not verified facts. Prefer priors that are physically or operationally plausible "
    "and that could in principle be checked against data.\n\n"
    "Output STRICT JSON only (no prose, no markdown outside the JSON)."
)

_TYPE_DEFS = {
    "periodic": "a repeating rhythm in the target (e.g. a daily operating rhythm)",
    "calendar": "a calendar-indexed effect (e.g. working-day vs weekend)",
    "trend": "a slow drift or a persistence / reversal tendency",
    "state": "a discrete operating state (e.g. high-load vs low-load)",
    "operational": "a process-driven cycle (e.g. an equipment heating/cooling cycle)",
    "environmental": "an external environmental driver (e.g. ambient temperature)",
    "event": "an occasional event or anomaly",
    "other": "anything not covered above",
}


def _metadata_block(metadata: Dict[str, Any]) -> str:
    lines = ["DATASET METADATA:"]
    for key, label in (
        ("dataset_name", "dataset name"),
        ("description", "description"),
        ("sampling_frequency", "sampling frequency"),
        ("domain", "domain"),
    ):
        if metadata.get(key):
            lines.append(f"- {label}: {metadata[key]}")
    lines.append(
        f"- target variable: {metadata.get('target_variable')} "
        f"({metadata.get('target_description', '')})"
    )
    variables = metadata.get("variables") or {}
    lines.append("- available variables:")
    for name in metadata.get("available_variables", []):
        desc = variables.get(name)
        lines.append(f"    * {name}: {desc}" if desc else f"    * {name}")
    return "\n".join(lines)


def _schema_block() -> str:
    lines = ["OUTPUT SCHEMA (strict JSON):"]
    lines.append(
        'Return a single JSON object with a "priors" list of 3 to 8 entries. '
        "Each entry has exactly these keys:"
    )
    lines.append('  - "name": short human-readable label')
    lines.append(f'  - "type": one of {sorted(ALLOWED_TYPES)}')
    lines.append(
        '  - "timescale": the time scale the prior operates on '
        '(e.g. daily, weekly, seasonal, minutes-to-hours, state-duration)'
    )
    lines.append(
        '  - "condition": a precise, checkable description of WHEN the prior is expected '
        'to hold (e.g. "same hour-of-day and same load state")'
    )
    lines.append(
        '  - "confidence": a number in [0,1] — your subjective confidence that this '
        "relationship is real"
    )
    lines.append('  - "reason": 1-2 sentences of physical / domain rationale')
    lines.append("\nAllowed types and their meaning:")
    for t in sorted(ALLOWED_TYPES):
        lines.append(f"  - {t}: {_TYPE_DEFS[t]}")
    lines.append("\nExample (illustrative only, do not copy):")
    example = {
        "priors": [
            {
                "name": "daily operating rhythm",
                "type": "periodic",
                "timescale": "daily",
                "condition": "same hour-of-day",
                "confidence": 0.8,
                "reason": "load and temperature follow a diurnal cycle",
            }
        ]
    }
    lines.append(json.dumps(example, indent=2))
    return "\n".join(lines)


def build_prompt(
    metadata: Dict[str, Any],
    expert_hint: Optional[str] = None,
    prompt_version: str = PROMPT_VERSION,
) -> Dict[str, str]:
    """Build the system + user prompt for a single prior-generation call."""
    user_blocks = [_metadata_block(metadata)]
    if expert_hint:
        user_blocks.append(
            "EXPERT HINT (hypothesis guidance, NOT verified facts):\n" + expert_hint
        )
    else:
        user_blocks.append("No expert hint provided.")
    user_blocks.append(_schema_block())
    user_blocks.append(
        "TASK: Propose 3-8 candidate retrieval priors for the target variable above. "
        "Do not predict values. Output only the JSON object."
    )
    user = "\n\n".join(user_blocks)

    return {
        "prompt_version": prompt_version,
        "system": _SYSTEM,
        "user": user,
        "full": f"SYSTEM:\n{_SYSTEM}\n\nUSER:\n{user}",
    }


# --------------------------------------------------------------------------- #
# Prompt V2 -- mechanism-driven chain, no calendar/trend/lag priming
# --------------------------------------------------------------------------- #

PROMPT_VERSION_V2 = "v2"

_V2_SYSTEM = (
    "You are a time-series domain expert and retrieval-system designer.\n\n"
    "Your job is to propose candidate retrieval priors for a target variable. A "
    "retrieval prior is a structured hypothesis about what makes one historical "
    "segment predictive for another.\n\n"
    "For each prior you must reason through a causal chain, from mechanism to "
    "retrieval:\n"
    "  domain fact -> latent predictive state -> observable proxy -> "
    "retrieval similarity -> expected future relationship\n\n"
    "You are NOT asked to, and must NOT:\n"
    "- forecast the future or produce forecast values\n"
    "- choose fusion weights or any model hyperparameter\n"
    "- use, assume, or optimize against any benchmark / test performance\n\n"
    "Do NOT default to generic calendar features (hour-of-day, day-of-week, month) "
    "unless they follow directly from a supplied physical mechanism. Derive each prior "
    "from a mechanism, not from a fixed list of feature types.\n\n"
    "Your priors are HYPOTHESES to be checked against numerical data later, not "
    "verified facts.\n\n"
    "Output STRICT JSON only (no prose, no markdown outside the JSON)."
)


def _v2_schema_block() -> str:
    example = {
        "priors": [
            {
                "name": "accumulated heat state",
                "domain_fact": "thermal mass integrates heat input, so temperature changes gradually",
                "latent_state": "stored thermal energy in the system",
                "observable_proxy": "recent cumulative heat-generating input",
                "retrieval_similarity": "match segments by similarity of recent cumulative input",
                "expected_future_relationship": "similar stored energy -> similar short-term trajectory",
                "confidence": 0.7,
            }
        ]
    }
    return (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return a single JSON object with a "priors" list of 3 to 8 entries. '
        "Each entry has exactly these keys:\n"
        '  - "name": short human-readable label\n'
        '  - "domain_fact": the governing domain fact / mechanism (1 sentence)\n'
        '  - "latent_state": the hidden, unobserved predictive state implied by that fact\n'
        '  - "observable_proxy": the observable quantity (from the available variables or the '
        "target's own history) that proxies this latent state\n"
        '  - "retrieval_similarity": how similarity between two segments should be measured '
        "from that proxy\n"
        '  - "expected_future_relationship": what future behavior is expected for segments '
        "with similar latent state\n"
        '  - "confidence": a number in [0,1] -- subjective confidence that the mechanism is real\n\n'
        "Example (illustrative only, do not copy):\n"
        + json.dumps(example, indent=2)
    )


def build_prompt_v2(
    metadata: Dict[str, Any],
    expert_hint: Optional[str] = None,
    prompt_version: str = PROMPT_VERSION_V2,
) -> Dict[str, str]:
    """Build the Prompt V2 (mechanism-driven) system + user prompt."""
    user_blocks = [_metadata_block(metadata)]
    if expert_hint:
        user_blocks.append(
            "EXPERT MECHANISM (domain facts / physical mechanisms; not retrieval features):\n"
            + expert_hint
        )
    else:
        user_blocks.append("No expert mechanism provided.")
    user_blocks.append(_v2_schema_block())
    user_blocks.append(
        "TASK: Propose 3-8 candidate retrieval priors by reasoning from mechanism to "
        "retrieval. Do not predict values. Output only the JSON object."
    )
    user = "\n\n".join(user_blocks)

    return {
        "prompt_version": prompt_version,
        "system": _V2_SYSTEM,
        "user": user,
        "full": f"SYSTEM:\n{_V2_SYSTEM}\n\nUSER:\n{user}",
    }


# --------------------------------------------------------------------------- #
# Prompt V2.1 -- mechanism -> structured executable specification
# --------------------------------------------------------------------------- #

PROMPT_VERSION_V21 = "v21"

_V21_SYSTEM = (
    "You are a time-series domain expert and retrieval-system designer.\n\n"
    "Your job is to propose candidate retrieval priors as EXECUTABLE specifications. "
    "Each prior must state, in structured form, how to construct a retrieval feature "
    "from the available data.\n\n"
    "Reason from a physical mechanism: domain fact -> latent predictive state -> "
    "observable proxy -> executable retrieval feature.\n\n"
    "You must use ONLY the allowed operations listed below. Do not invent new "
    "operations or new variables.\n\n"
    "You are NOT asked to, and must NOT: forecast values, choose fusion weights or "
    "hyperparameters, or use/optimize against any benchmark or test performance.\n\n"
    "Do NOT default to generic calendar features (hour-of-day, day-of-week, month) "
    "unless they follow directly from a supplied physical mechanism.\n\n"
    "Your priors are HYPOTHESES to be checked against data later, not verified facts.\n\n"
    "Output STRICT JSON only."
)


def _v21_schema_block() -> str:
    example = {
        "priors": [
            {
                "name": "accumulated heat state",
                "input_variables": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
                "transform": "ewma",
                "temporal_aggregation": "last",
                "similarity": "abs_diff",
                "expected_mechanism": "thermal mass integrates load, so EWMA of load proxies stored heat",
                "confidence": 0.7,
            }
        ]
    }
    return (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return a single JSON object with a "priors" list of 3 to 8 entries. Each entry has exactly these keys:\n'
        '  - "name": short human-readable label\n'
        '  - "input_variables": a list of variable names from '
        "{OT, HUFL, HULL, MUFL, MULL, LUFL, LULL}\n"
        '  - "transform": one of {raw, sum, square, difference, slope, ewma, '
        "residual_from_train_fitted_equilibrium, concatenate}\n"
        '  - "temporal_aggregation": one of {window, mean, last}\n'
        '  - "similarity": one of {correlation, abs_diff}\n'
        '  - "expected_mechanism": the domain mechanism this feature operationalizes\n'
        '  - "confidence": a number in [0,1]\n\n'
        "ALLOWED OPERATIONS (use only these):\n"
        "  - raw: use the variable as-is\n"
        "  - sum: sum the input variables into one series\n"
        "  - square: square each value\n"
        "  - difference: first difference (point-to-point change)\n"
        "  - slope: linear slope over the recent context\n"
        "  - ewma: exponentially weighted moving average\n"
        "  - residual_from_train_fitted_equilibrium: residual of OT against a train-fitted "
        "function of load\n"
        "  - concatenate: join multiple variables into a multivariate profile\n"
        "  - temporal_aggregation: window = keep full context (use with similarity=correlation); "
        "mean = average over context; last = final value\n"
        "  - similarity: correlation (for window profiles) or abs_diff (for scalars)\n\n"
        "Example (illustrative only, do not copy):\n"
        + json.dumps(example, indent=2)
    )


def build_prompt_v21(
    metadata: Dict[str, Any],
    expert_hint: Optional[str] = None,
    prompt_version: str = PROMPT_VERSION_V21,
) -> Dict[str, str]:
    user_blocks = [_metadata_block(metadata)]
    if expert_hint:
        user_blocks.append(
            "EXPERT MECHANISM (domain facts / physical mechanisms; not retrieval features):\n"
            + expert_hint
        )
    else:
        user_blocks.append("No expert mechanism provided.")
    user_blocks.append(_v21_schema_block())
    user_blocks.append(
        "TASK: Propose 3-8 candidate executable retrieval priors using only the allowed "
        "operations. Output only the JSON object."
    )
    user = "\n\n".join(user_blocks)
    return {
        "prompt_version": prompt_version,
        "system": _V21_SYSTEM,
        "user": user,
        "full": f"SYSTEM:\n{_V21_SYSTEM}\n\nUSER:\n{user}",
    }


# --------------------------------------------------------------------------- #
# Prompt V2-blind -- anonymized metadata, mechanism -> executable retrieval
# criterion. Used by the blind prompt-control experiment to test whether the LLM
# recognizes the benchmark identity vs. reasons from the mechanism itself.
# --------------------------------------------------------------------------- #

PROMPT_VERSION_BLIND = "v2_blind_1"

_BLIND_SYSTEM = (
    "You are a time-series domain expert and retrieval-system designer.\n\n"
    "Your job is to propose candidate retrieval priors for a target variable. "
    "A retrieval prior is a structured hypothesis about what makes one historical "
    "segment predictive for another.\n\n"
    "For each prior you must reason through a causal chain:\n"
    "  domain fact -> latent predictive state -> observable proxy -> "
    "executable retrieval criterion -> expected future relationship\n\n"
    "You are NOT asked to, and must NOT:\n"
    "- forecast the future or produce forecast values\n"
    "- choose fusion weights or any model hyperparameter\n"
    "- use, assume, or optimize against any benchmark / test performance\n\n"
    "Use ONLY the allowed operations listed in the schema. Do not invent new "
    "operations or new variables.\n\n"
    "Your priors are HYPOTHESES to be checked against data later, not verified facts.\n\n"
    "Output STRICT JSON only (no prose, no markdown outside the JSON)."
)


def _blind_metadata_block(metadata: Dict[str, Any]) -> str:
    lines = ["DATASET (anonymized):"]
    lines.append(
        f"- target variable: {metadata.get('target_variable', 'Y')} "
        f"({metadata.get('target_description', '')})"
    )
    obs = metadata.get("observed_variables", [])
    desc = metadata.get("observed_description", "")
    lines.append(f"- observed variables: {', '.join(obs)} ({desc})")
    lines.append(f"- sampling frequency: {metadata.get('sampling_frequency', '')}")
    return "\n".join(lines)


def _blind_schema_block() -> str:
    example = {
        "priors": [
            {
                "name": "accumulated thermal state",
                "domain_fact": "thermal mass integrates heat input, so temperature changes gradually",
                "latent_state": "stored thermal energy in the system",
                "input_variables": ["X1", "X2", "X3", "X4", "X5", "X6"],
                "transform": "ewma",
                "temporal_aggregation": "last",
                "similarity": "abs_diff",
                "expected_future_relationship": "similar stored energy -> similar short-term trajectory",
                "confidence": 0.7,
            }
        ]
    }
    return (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return a single JSON object with a "priors" list of 3 to 8 entries. Each entry has exactly these keys:\n'
        '  - "name": short human-readable label\n'
        '  - "domain_fact": the governing domain fact / mechanism (1 sentence)\n'
        '  - "latent_state": the hidden predictive state implied by that fact\n'
        '  - "input_variables": a list of variable names from {Y, X1, X2, X3, X4, X5, X6}\n'
        '  - "transform": one of {raw, sum, square, signed_power, difference, slope, ewma, '
        "residual_from_train_fitted_equilibrium, concatenate}\n"
        '  - "temporal_aggregation": one of {last, mean, window}\n'
        '  - "similarity": one of {abs_diff, correlation}\n'
        '  - "expected_future_relationship": what future behavior is expected for segments '
        "with similar latent state\n"
        '  - "confidence": a number in [0,1]\n\n'
        "ALLOWED OPERATIONS (use only these):\n"
        "  - raw: use the variable as-is\n"
        "  - sum: sum the input variables into one series\n"
        "  - square: square each value\n"
        "  - signed_power: sign(x) * |x|^p for a chosen power p\n"
        "  - difference: first difference (point-to-point change)\n"
        "  - slope: linear slope over the recent context\n"
        "  - ewma: exponentially weighted moving average\n"
        "  - residual_from_train_fitted_equilibrium: residual of Y against a train-fitted "
        "function of the load variables\n"
        "  - concatenate: join multiple variables into a multivariate profile\n"
        "  - temporal_aggregation: last = final value; mean = average over context; "
        "window = keep full context (use with similarity=correlation)\n"
        "  - similarity: abs_diff (for scalars) or correlation (for window profiles)\n\n"
        "Example (illustrative only, do not copy):\n"
        + json.dumps(example, indent=2)
    )


def build_prompt_blind(
    metadata: Dict[str, Any],
    expert_hint: Optional[str] = None,
    prompt_version: str = PROMPT_VERSION_BLIND,
) -> Dict[str, str]:
    """Build the anonymized (blind) prompt. ``metadata`` must already be anonymized."""
    user_blocks = [_blind_metadata_block(metadata)]
    if expert_hint:
        user_blocks.append(
            "DOMAIN FACTS (physical mechanisms; NOT retrieval features):\n" + expert_hint
        )
    else:
        user_blocks.append("No domain facts provided.")
    user_blocks.append(_blind_schema_block())
    user_blocks.append(
        "TASK: Propose 3-8 candidate retrieval priors by reasoning from mechanism to an "
        "executable retrieval criterion. Do not predict values. Output only the JSON object."
    )
    user = "\n\n".join(user_blocks)

    return {
        "prompt_version": prompt_version,
        "system": _BLIND_SYSTEM,
        "user": user,
        "full": f"SYSTEM:\n{_BLIND_SYSTEM}\n\nUSER:\n{user}",
    }
