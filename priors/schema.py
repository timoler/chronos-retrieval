"""Structured schema for retrieval priors.

A Prior is a structured hypothesis about what historical relationship makes one
historical segment predictive for another. It is explicitly NOT a period
discovery result, NOT a forecast, and NOT a tuned hyperparameter.

This module only defines the schema + LLM-output parsing. Data-driven
validation (``validation_score`` / ``validated`` / ``operationalizable`` /
``evidence``) is added in a later step by ``priors.validator``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

ALLOWED_TYPES = {
    "periodic",
    "calendar",
    "trend",
    "state",
    "operational",
    "environmental",
    "event",
    "other",
}

ALLOWED_SOURCES = {"llm", "expert", "expert+llm", "llm+expert"}

# Fields the LLM is required to produce. ``source`` is stamped by the caller
# (it is determined by the experiment condition, not trusted from the LLM).
LLM_FIELDS = ("name", "type", "timescale", "condition", "confidence", "reason")


@dataclass
class Prior:
    name: str
    type: str
    timescale: str
    condition: str
    confidence: float
    reason: str
    source: str

    def __post_init__(self):
        if self.type not in ALLOWED_TYPES:
            raise ValueError(f"invalid prior type {self.type!r}")
        if self.source not in ALLOWED_SOURCES:
            raise ValueError(f"invalid prior source {self.source!r}")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Prior":
        d = dict(d)
        return cls(
            name=str(d.get("name", "")).strip(),
            type=str(d.get("type", "")).strip().lower(),
            timescale=str(d.get("timescale", "")).strip(),
            condition=str(d.get("condition", "")).strip(),
            confidence=float(d.get("confidence", 0.0)),
            reason=str(d.get("reason", "")).strip(),
            source=str(d.get("source", "")).strip().lower(),
        )


def _close_unbalanced(text: str):
    """Close unbalanced brackets in a JSON fragment, or ``None`` if truncated mid-string.

    Handles the common LLM truncation where a JSON object/array is cut off just
    before its final closing brace(s). Returns ``None`` (unrepairable) when the
    text ends inside an unterminated string, since closing brackets then would not
    produce a trustworthy value.
    """
    pairs = {"{": "}", "[": "]"}
    stack = []
    in_str = False
    esc = False
    for c in text:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack and pairs[stack[-1]] == c:
                    stack.pop()
    if in_str:
        return None
    if not stack:
        return text
    suffix = "".join(pairs[o] for o in reversed(stack))
    candidate = text.rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1].rstrip()
    return candidate + suffix


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a JSON object/list from an LLM response."""
    text = (text or "").strip()

    # strip markdown code fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # fallback: locate the outermost balanced JSON object or array anywhere in text
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        break

    # fallback: repair truncation by closing unbalanced brackets (only at a
    # bracket/whitespace boundary; mid-string truncation is left to the caller)
    repaired = _close_unbalanced(text)
    if repaired is not None:
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            pass

    raise ValueError("could not extract JSON from LLM response: " + repr(text[:800]))


def parse_priors(text: str, source: str) -> List[Prior]:
    """Parse an LLM response into a list of :class:`Prior`, stamping ``source``.

    Accepts either a bare JSON list or a ``{"priors": [...]}`` object.
    """
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"invalid source {source!r}")

    data = _extract_json(text)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "priors" in data:
        items = data["priors"]
    else:
        raise ValueError("expected a JSON object with a 'priors' list")

    priors: List[Prior] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"prior entry is not an object: {item!r}")
        item = dict(item)
        item["source"] = source
        priors.append(Prior.from_dict(item))
    return priors


# --------------------------------------------------------------------------- #
# Prompt V2 schema -- mechanism -> latent state -> proxy -> similarity -> future
# --------------------------------------------------------------------------- #

@dataclass
class PriorV2:
    name: str
    domain_fact: str
    latent_state: str
    observable_proxy: str
    retrieval_similarity: str
    expected_future_relationship: str
    confidence: float
    source: str

    def __post_init__(self):
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PriorV2":
        d = dict(d)
        return cls(
            name=str(d.get("name", "")).strip(),
            domain_fact=str(d.get("domain_fact", "")).strip(),
            latent_state=str(d.get("latent_state", "")).strip(),
            observable_proxy=str(d.get("observable_proxy", "")).strip(),
            retrieval_similarity=str(d.get("retrieval_similarity", "")).strip(),
            expected_future_relationship=str(d.get("expected_future_relationship", "")).strip(),
            confidence=float(d.get("confidence", 0.0)),
            source=str(d.get("source", "")).strip().lower(),
        )


def parse_priors_v2(text: str, source: str) -> List[PriorV2]:
    """Parse an LLM response into a list of :class:`PriorV2`, stamping ``source``."""
    data = _extract_json(text)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "priors" in data:
        items = data["priors"]
    else:
        raise ValueError("expected a JSON object with a 'priors' list")

    priors: List[PriorV2] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"prior entry is not an object: {item!r}")
        item = dict(item)
        item["source"] = source
        priors.append(PriorV2.from_dict(item))
    return priors


# --------------------------------------------------------------------------- #
# Prompt V2-blind schema -- anonymized metadata, mechanism -> executable
# retrieval criterion (for the blind prompt-control experiment)
# --------------------------------------------------------------------------- #

@dataclass
class BlindPrior:
    name: str
    domain_fact: str
    latent_state: str
    input_variables: List[str]
    transform: str
    temporal_aggregation: str
    similarity: str
    expected_future_relationship: str
    confidence: float
    source: str

    def __post_init__(self):
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BlindPrior":
        d = dict(d)
        return cls(
            name=str(d.get("name", "")).strip(),
            domain_fact=str(d.get("domain_fact", "")).strip(),
            latent_state=str(d.get("latent_state", "")).strip(),
            input_variables=[str(v).strip().upper() for v in d.get("input_variables", [])],
            transform=str(d.get("transform", "")).strip().lower(),
            temporal_aggregation=str(d.get("temporal_aggregation", "")).strip().lower(),
            similarity=str(d.get("similarity", "")).strip().lower(),
            expected_future_relationship=str(d.get("expected_future_relationship", "")).strip(),
            confidence=float(d.get("confidence", 0.0)),
            source=str(d.get("source", "")).strip().lower(),
        )


def parse_blind_priors(text: str, source: str) -> List[BlindPrior]:
    """Parse an LLM response into a list of :class:`BlindPrior`, stamping ``source``."""
    data = _extract_json(text)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "priors" in data:
        items = data["priors"]
    else:
        raise ValueError("expected a JSON object with a 'priors' list")

    priors: List[BlindPrior] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"prior entry is not an object: {item!r}")
        item = dict(item)
        item["source"] = source
        priors.append(BlindPrior.from_dict(item))
    return priors
