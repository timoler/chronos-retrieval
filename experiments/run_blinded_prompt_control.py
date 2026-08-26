"""Blind prompt control (generation only, two LLM calls).

Question under test: does the LLM propose its retrieval priors because it
recognizes the ETTh1 benchmark identity, or does the expert mechanism itself
let it derive those priors?

Conditions (same anonymized metadata, two LLM calls total):
  A = blind_metadata  -- Y + X1..X6, no dataset name, no domain facts.
  B = blind_expert    -- same metadata + domain facts (no retrieval solution).

This experiment ONLY generates prompts + priors. It does NOT load ETT data,
Chronos, the memory bank, the validator, or the retriever; it does not run
alpha/lambda sweeps and never touches val/test labels.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.executable import (
    ALLOWED_AGGREGATIONS,
    ALLOWED_SIMILARITIES,
    ALLOWED_TRANSFORMS,
)
from priors.llm_prior import generate_priors
from priors.prompts import PROMPT_VERSION_BLIND, build_prompt_blind

# ---------------------------------------------------------------------------
# Anonymized metadata: minimum necessary semantics only. No dataset name, no
# variable names, no previous results.
# ---------------------------------------------------------------------------
BLIND_METADATA = {
    "target_variable": "Y",
    "target_description": "transformer oil temperature",
    "observed_variables": ["X1", "X2", "X3", "X4", "X5", "X6"],
    "observed_description": "transformer load-related measurements",
    "sampling_frequency": "1 hour",
}

# Domain facts only -- no retrieval solution, no "use square / load^2 /
# correlation / profile / lag / trend / hour-month-week / previous best".
# The LLM must derive the retrieval criterion from the mechanism itself.
BLIND_EXPERT_MECHANISM = (
    "transformer oil temperature reflects accumulated thermal state\n"
    "load changes may influence temperature with delay\n"
    "load-related heating can be nonlinear\n"
    "identical current temperatures may correspond to different heating/cooling states\n"
    "both heat accumulation and dissipation matter"
)

# Benchmark-identity / prior-result tokens that must never appear in a prompt.
FORBIDDEN_IDENTITY = [
    "ETTh1",
    "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL",
    "squared_load_profile",
    "7.7623", "8.0155", "6.1968",
]

# Retrieval-solution terms that must NOT appear in the expert mechanism text.
FORBIDDEN_IN_EXPERT = [
    "square", "load^2", "load²", "correlation", "profile", "lag",
    "trend", "hour", "month", "week", "previous best",
]

BLIND_VARS = {"Y", "X1", "X2", "X3", "X4", "X5", "X6"}
LOAD_VARS = {"X1", "X2", "X3", "X4", "X5", "X6"}


def identity_leaks(full_prompt_text: str):
    """Benchmark-identity / prior-result tokens found in a full prompt."""
    text = full_prompt_text.lower()
    hits = [t for t in FORBIDDEN_IDENTITY if t.lower() in text]
    if "mse" in text:
        hits.append("mse")
    return hits


def expert_leaks(expert_hint: str):
    """Retrieval-solution terms found in the expert mechanism text."""
    text = expert_hint.lower()
    return [t for t in FORBIDDEN_IN_EXPERT if t in text]


def whitelist_check(p):
    """Return (supported, reason). Mark unsupported; never rewrite the prior."""
    if p.transform not in ALLOWED_TRANSFORMS:
        return False, f"unsupported transform {p.transform!r}"
    if p.temporal_aggregation not in ALLOWED_AGGREGATIONS:
        return False, f"unsupported aggregation {p.temporal_aggregation!r}"
    if p.similarity not in ALLOWED_SIMILARITIES:
        return False, f"unsupported similarity {p.similarity!r}"
    if not p.input_variables:
        return False, "no input_variables"
    for v in p.input_variables:
        if v not in BLIND_VARS:
            return False, f"unknown variable {v!r}"
    return True, "ok"


def _feature(p):
    return (p.transform, p.temporal_aggregation, p.similarity,
            tuple(sorted(p.input_variables)))


def _feat_label(f):
    t, a, s, vars_ = f
    v = "+".join(vars_) if vars_ else "-"
    return f"{t}/{a}/{s} on {v}"


def _save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    # ---- build + leak-check the two prompts BEFORE any LLM call ----
    prompt_a = build_prompt_blind(BLIND_METADATA, expert_hint=None,
                                  prompt_version=PROMPT_VERSION_BLIND)
    prompt_b = build_prompt_blind(BLIND_METADATA, expert_hint=BLIND_EXPERT_MECHANISM,
                                  prompt_version=PROMPT_VERSION_BLIND)

    failed = False
    for label, prompt in (("blind_metadata", prompt_a), ("blind_expert", prompt_b)):
        hits = identity_leaks(prompt["full"])
        print(f"identity leak check [{label}]: "
              f"{'PASS' if not hits else 'FAIL -> ' + str(hits)}")
        failed = failed or bool(hits)

    exp_hits = expert_leaks(BLIND_EXPERT_MECHANISM)
    print(f"expert-mechanism leak check [blind_expert]: "
          f"{'PASS' if not exp_hits else 'FAIL -> ' + str(exp_hits)}")
    failed = failed or bool(exp_hits)

    if failed:
        print("LEAK CHECK FAILED -- stopping before any LLM call.")
        sys.exit(1)

    # ---- two LLM calls (cache-aware: same inputs never re-billed) ----
    result_a = generate_priors(
        BLIND_METADATA, expert_hint=None, prompt_type="blind_metadata",
        prompt_version=PROMPT_VERSION_BLIND,
    )
    result_b = generate_priors(
        BLIND_METADATA, expert_hint=BLIND_EXPERT_MECHANISM, prompt_type="blind_expert",
        prompt_version=PROMPT_VERSION_BLIND,
    )

    pa = result_a["priors"]
    pb = result_b["priors"]

    print(f"\nmodel={result_a['model']}  prompt_version={PROMPT_VERSION_BLIND}")
    print(f"blind_metadata: from_cache={result_a['from_cache']}  "
          f"blind_expert: from_cache={result_b['from_cache']}")

    # ---- side-by-side table: Prior | Blind Metadata | Blind + Expert ----
    fa = {_feature(p) for p in pa}
    fb = {_feature(p) for p in pb}
    all_feats = sorted(fa | fb, key=str)

    def cell(priors, feat):
        matched = [p for p in priors if _feature(p) == feat]
        if not matched:
            return "-"
        parts = []
        for p in matched:
            sup, reason = whitelist_check(p)
            s = f"{p.name} ({p.confidence:.2f})"
            if not sup:
                s += f" [UNSUPPORTED: {reason}]"
            parts.append(s)
        return "; ".join(parts)

    print("\n" + "=" * 120)
    print("SIDE-BY-SIDE PRIOR SETS")
    print("=" * 120)
    print(f"{'Prior (transform/agg/sim on vars)':<34} | {'Blind Metadata':<42} | Blind + Expert")
    print("-" * 120)
    for f in all_feats:
        print(f"{_feat_label(f):<34} | {cell(pa, f):<42} | {cell(pb, f)}")
    print("-" * 120)

    # ---- factual answers ----
    def nonlinear_load(priors):
        """Load-var priors whose transform OR mechanism text is nonlinear."""
        out = []
        for p in priors:
            if not (set(p.input_variables) & LOAD_VARS):
                continue
            transform_nl = p.transform in ("square", "signed_power")
            text = " ".join([p.domain_fact, p.latent_state,
                             p.expected_future_relationship]).lower()
            text_nl = any(k in text for k in
                          ("nonlinear", "non-linear", "quadratic", "squared", "square"))
            if transform_nl or text_nl:
                out.append(p.name)
        return out

    def squared_load(priors):
        return [p.name for p in priors
                if p.transform in ("square", "signed_power")
                and (set(p.input_variables) & LOAD_VARS)]

    nl_a = nonlinear_load(pa)
    nl_b = nonlinear_load(pb)
    sq_b = squared_load(pb)

    print("\nFACTUAL ANSWERS")
    print("---------------")
    print(f"1. Blind Metadata proposes a nonlinear load-related mechanism: "
          f"{'yes -> ' + ', '.join(nl_a) if nl_a else 'no'}")
    print(f"2. Blind + Expert proposes a nonlinear load-related mechanism: "
          f"{'yes -> ' + ', '.join(nl_b) if nl_b else 'no'}")
    print(f"3. Blind + Expert derives a squared/load^2-like transformation "
          f"(transform=square/signed_power on load): "
          f"{'yes -> ' + ', '.join(sq_b) if sq_b else 'no'}")
    overlap = fa & fb
    print(f"4. Hypothesis sets clearly differ: A has {len(fa)} unique features, "
          f"B has {len(fb)}, overlap {len(overlap)}. "
          f"A-only: {sorted(_feat_label(f) for f in fa - fb)}; "
          f"B-only: {sorted(_feat_label(f) for f in fb - fa)}")

    # ---- save ----
    ts = datetime.now().isoformat(timespec="seconds")

    def blob(result):
        return {
            "model": result["model"],
            "prompt_version": result["prompt_version"],
            "prompt_type": result["prompt_type"],
            "from_cache": result["from_cache"],
            "timestamp": ts,
            "prompt": result["prompt"],
            "raw_response": result["raw_response"],
            "parsed_priors": [p.to_dict() for p in result["priors"]],
        }

    _save("results/blinding/blind_metadata.json", blob(result_a))
    _save("results/blinding/blind_expert.json", blob(result_b))
    _save("results/blinding/prompts.json", {
        "model": result_a["model"],
        "prompt_version": PROMPT_VERSION_BLIND,
        "timestamp": ts,
        "blind_metadata": BLIND_METADATA,
        "blind_expert_mechanism": BLIND_EXPERT_MECHANISM,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
    })
    print("\nSaved: results/blinding/blind_metadata.json, "
          "results/blinding/blind_expert.json, results/blinding/prompts.json")


if __name__ == "__main__":
    main()
