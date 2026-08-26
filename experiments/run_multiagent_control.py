"""Single-LLM vs Multi-Agent prior generation (benchmark-blinded, one pass).

Two conditions, same anonymized metadata + expert mechanism, same model:

  A  Single LLM  -- one model directly proposes executable retrieval priors
                    (reuses priors.prompts.build_prompt_blind + the LLM cache).
  B  Multi-Agent -- fixed one-pass Proposal -> Critic -> Compiler (3 LLM calls,
                    no iterative chatting).

Both sets then run through the existing TRAIN/VAL V2.1 leave-one-in validator
(priors.executable.leave_one_in_executable) with identical settings. No Chronos
execution (cached val forecasts only), no TEST, no prior combination, no
adaptive fusion. Every prompt + raw output is saved.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.executable import (
    ALLOWED_AGGREGATIONS,
    ALLOWED_SIMILARITIES,
    ALLOWED_TRANSFORMS,
    ExecutablePrior,
    leave_one_in_executable,
)
from priors.llm_prior import _default_model, _get_client, generate_priors
from priors.prompts import PROMPT_VERSION_BLIND
from priors.schema import _extract_json

# ---------------------------------------------------------------------------
# anonymized metadata + expert mechanism (identical to the blind control)
# ---------------------------------------------------------------------------
BLIND_METADATA = {
    "target_variable": "Y",
    "target_description": "transformer oil temperature",
    "observed_variables": ["X1", "X2", "X3", "X4", "X5", "X6"],
    "observed_description": "transformer load-related measurements",
    "sampling_frequency": "1 hour",
}
BLIND_EXPERT_MECHANISM = (
    "transformer oil temperature reflects accumulated thermal state\n"
    "load changes may influence temperature with delay\n"
    "load-related heating can be nonlinear\n"
    "identical current temperatures may correspond to different heating/cooling states\n"
    "both heat accumulation and dissipation matter"
)

BLIND_TO_REAL = {
    "Y": "OT",
    "X1": "HUFL", "X2": "HULL", "X3": "MUFL",
    "X4": "MULL", "X5": "LUFL", "X6": "LULL",
}


def _metadata_text():
    return (
        "DATASET (anonymized):\n"
        f"- target variable: Y ({BLIND_METADATA['target_description']})\n"
        "- observed variables: X1, X2, X3, X4, X5, X6 "
        f"({BLIND_METADATA['observed_description']})\n"
        f"- sampling frequency: {BLIND_METADATA['sampling_frequency']}"
    )


def _whitelist_block():
    return (
        "ALLOWED OPERATIONS (use ONLY these):\n"
        f"  transforms: {', '.join(sorted(ALLOWED_TRANSFORMS))}\n"
        f"  temporal_aggregation: {', '.join(sorted(ALLOWED_AGGREGATIONS))}\n"
        f"  similarity: {', '.join(sorted(ALLOWED_SIMILARITIES))}\n"
        "\n"
        "  - raw: use the variable as-is\n"
        "  - sum: sum the input variables into one series\n"
        "  - square: square each value (x^2) -- the canonical nonlinear load transform\n"
        "  - signed_power: sign(x)*|x|^p (specify power p)\n"
        "  - difference: first difference (point-to-point change)\n"
        "  - slope: linear slope over the recent context\n"
        "  - ewma: exponentially weighted moving average\n"
        "  - residual_from_train_fitted_equilibrium: residual of Y against a LINEAR "
        "fit of total load (LINEAR ONLY)\n"
        "  - concatenate: join multiple variables into a multivariate profile\n"
        "  - temporal_aggregation: last=final value; mean=average; window=full context "
        "(use with similarity=correlation)\n"
        "  - similarity: abs_diff (scalars) or correlation (window profiles)\n"
        "\n"
        "NONLINEAR WARNING: residual_from_train_fitted_equilibrium is a LINEAR fit.\n"
        "If a hypothesis needs a nonlinear relationship (squared/curved load-equilibrium,\n"
        "nonlinear heating), you MUST either (a) use `square`/`signed_power` if that\n"
        "faithfully captures the mechanism, or (b) set supported=false. NEVER silently\n"
        "map a nonlinear mechanism to the linear residual transform."
    )


# ---------------------------------------------------------------------------
# LLM call helper (lightweight cache; reuses the existing client/model)
# ---------------------------------------------------------------------------
def _call_llm(system: str, user: str, stage: str, model: str):
    key = hashlib.sha256(
        json.dumps({"stage": stage, "system": system, "user": user, "model": model},
                   sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path("cache/llm_priors_multi")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached["raw_response"], str(cache_path), True

    client = _get_client()
    resp = client.messages.create(
        model=model, max_tokens=20000, system=system,
        messages=[{"role": "user", "content": user}],
    )
    content = resp.content
    raw = (
        content
        if isinstance(content, str)
        else "".join(getattr(b, "text", "")
                     for b in content if getattr(b, "type", "text") == "text")
    )
    if not raw.strip():
        raise RuntimeError("LLM returned no text")

    cache_path.write_text(json.dumps({
        "stage": stage, "model": model, "system": system, "user": user,
        "raw_response": raw,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw, str(cache_path), False


def _list_from(text, key):
    data = _extract_json(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and key in data:
        return data[key]
    raise ValueError(f"expected JSON object with a '{key}' list")


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------
_PROP_SYSTEM = (
    "You are a time-series domain expert. Given an anonymized physical system and a "
    "set of domain facts, propose 4-6 mechanism-aware RETRIEVAL HYPOTHESES. A "
    "retrieval hypothesis is a causal claim: domain fact -> latent predictive state "
    "-> observable proxy -> expected future relationship. Reason from the domain "
    "facts, not from a fixed list of feature types. Do NOT propose executable "
    "operations (no transforms/aggregations/similarities) and do NOT reference any "
    "benchmark, dataset name, or previous results. Output strict JSON only."
)


def build_proposal_prompt():
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"hypotheses": [ ... ]} with 4 to 6 entries. Each entry:\n'
        '  - "name": short label\n'
        '  - "domain_fact": the governing physical mechanism (1 sentence)\n'
        '  - "latent_state": the hidden predictive state implied by that fact\n'
        '  - "observable_proxy": the observable quantity (from Y / X1..X6) proxying it\n'
        '  - "expected_future_relationship": expected future behavior for similar latent state\n'
        '  - "confidence": number in [0,1]\n'
        "Do NOT include transforms/aggregations/similarities."
    )
    user = "\n\n".join([
        _metadata_text(),
        "DOMAIN FACTS (physical mechanisms; NOT retrieval features):\n" + BLIND_EXPERT_MECHANISM,
        schema,
        "TASK: propose 4-6 retrieval hypotheses. Output only the JSON object.",
    ])
    return _PROP_SYSTEM, user


_CRIT_SYSTEM = (
    "You are a critical reviewer of retrieval hypotheses. For each hypothesis assess "
    "five criteria and give a verdict. Be strict: reject hypotheses that are "
    "physically/logically incoherent, unobservable from the available variables, "
    "benchmark-specific, duplicative, or not faithfully compilable by the whitelist. "
    "Output strict JSON only."
)


def build_critique_prompt(hypotheses):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"critiques": [ ... ]}, one entry per hypothesis. Each entry:\n'
        '  - "name": the hypothesis name (copy exactly)\n'
        '  - "verdict": "accept" or "reject"\n'
        '  - "consistency": note on physical/logical consistency\n'
        '  - "observability": note on whether the proxy is observable from Y / X1..X6\n'
        '  - "leakage": note on benchmark/dataset/prior-result leakage\n'
        '  - "duplication": note on duplication with another hypothesis\n'
        '  - "compiler_fidelity": note on whether the whitelist can faithfully represent it\n'
        '  - "summary": 1-sentence decision rationale\n'
        'Set verdict "accept" only if ALL five checks pass; otherwise "reject".'
    )
    user = "\n\n".join([
        _metadata_text(),
        "DOMAIN FACTS:\n" + BLIND_EXPERT_MECHANISM,
        _whitelist_block(),
        "HYPOTHESES TO REVIEW:\n" + json.dumps(hypotheses, ensure_ascii=False, indent=2),
        schema,
        "TASK: critique every hypothesis. Output only the JSON object.",
    ])
    return _CRIT_SYSTEM, user


_COMP_SYSTEM = (
    "You convert accepted retrieval hypotheses into EXECUTABLE retrieval priors using "
    "ONLY the whitelist of operations below. If a hypothesis cannot be faithfully "
    "represented by the whitelist, mark it supported=false with a reason. NEVER "
    "silently approximate: in particular, never represent a nonlinear relationship "
    "using the LINEAR equilibrium residual. Output strict JSON only."
)


def build_compile_prompt(accepted):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"priors": [ ... ]}, one entry per accepted hypothesis. Each entry:\n'
        '  - "name": short label\n'
        '  - "input_variables": list from {Y, X1, X2, X3, X4, X5, X6}\n'
        '  - "transform": one of the allowed transforms\n'
        '  - "temporal_aggregation": one of the allowed aggregations\n'
        '  - "similarity": one of the allowed similarities\n'
        '  - "power": a positive number only if transform == "signed_power" (else omit)\n'
        '  - "expected_mechanism": how this executable feature operationalizes the hypothesis\n'
        '  - "confidence": number in [0,1]\n'
        '  - "supported": true or false\n'
        '  - "unsupported_reason": short reason if supported is false (else "")'
    )
    user = "\n\n".join([
        _metadata_text(),
        _whitelist_block(),
        "ACCEPTED HYPOTHESES:\n" + json.dumps(accepted, ensure_ascii=False, indent=2),
        schema,
        "TASK: compile each accepted hypothesis into an executable prior, marking any "
        "that cannot be faithfully represented as supported=false. Output only the JSON.",
    ])
    return _COMP_SYSTEM, user


# ---------------------------------------------------------------------------
# mapping helpers
# ---------------------------------------------------------------------------
def blind_to_exec(p):
    inputs = []
    for v in p.input_variables:
        if v not in BLIND_TO_REAL:
            return None  # unmappable variable -> rejected
        inputs.append(BLIND_TO_REAL[v])
    return ExecutablePrior(
        name=p.name, source=p.source, input_variables=inputs,
        transform=p.transform, temporal_aggregation=p.temporal_aggregation,
        similarity=p.similarity,
        expected_mechanism=p.expected_future_relationship or p.domain_fact,
        power=None, confidence=p.confidence,
    )


def compiled_to_exec(d):
    inputs = []
    for v in d.get("input_variables", []):
        v = str(v).strip().upper()
        if v not in BLIND_TO_REAL:
            return None
        inputs.append(BLIND_TO_REAL[v])
    power = None
    if d.get("power") is not None:
        try:
            power = float(d["power"])
        except (TypeError, ValueError):
            power = None
    return ExecutablePrior(
        name=str(d.get("name", "")).strip(),
        source="multi_agent",
        input_variables=inputs,
        transform=str(d.get("transform", "")).strip().lower(),
        temporal_aggregation=str(d.get("temporal_aggregation", "")).strip().lower(),
        similarity=str(d.get("similarity", "")).strip().lower(),
        expected_mechanism=str(d.get("expected_mechanism", "")).strip(),
        power=power,
        confidence=float(d.get("confidence", 0.0)),
    )


def _save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(rows, n_proposed):
    exec_rows = [r for r in rows if r["feature"] is not None]
    n_exec = len(exec_rows)
    if exec_rows:
        best = min(exec_rows, key=lambda r: r["val_fus_mse"])
        top3 = sorted(exec_rows, key=lambda r: r["val_fus_mse"])[:3]
        best_fus, best_name = best["val_fus_mse"], best["name"]
        top3_mean = sum(r["val_fus_mse"] for r in top3) / len(top3)
    else:
        best_fus = best_name = top3_mean = None
    return {
        "n_proposed": n_proposed,
        "n_executable": n_exec,
        "n_rejected_unsupported": n_proposed - n_exec,
        "unique_mechanisms": len({r["feature"] for r in exec_rows}),
        "best_val_fused_mse": best_fus,
        "top3_mean_val_fused_mse": top3_mean,
        "best_prior_name": best_name,
    }


def main():
    model = _default_model()
    ts = datetime.now().isoformat(timespec="seconds")

    # ---- A. Single LLM (blind + expert; cache-aware) ----
    res_a = generate_priors(
        BLIND_METADATA, expert_hint=BLIND_EXPERT_MECHANISM,
        prompt_type="blind_expert", prompt_version=PROMPT_VERSION_BLIND,
    )
    exec_a = [blind_to_exec(p) for p in res_a["priors"]]
    exec_a = [p for p in exec_a if p is not None]
    print(f"[A] single-LLM: {len(exec_a)} priors (from_cache={res_a['from_cache']})")

    # ---- B. Multi-Agent (one pass) ----
    prop_sys, prop_user = build_proposal_prompt()
    prop_raw, _, prop_cached = _call_llm(prop_sys, prop_user, "proposal", model)
    hypotheses = _list_from(prop_raw, "hypotheses")
    print(f"[B] proposal: {len(hypotheses)} hypotheses (from_cache={prop_cached})")

    crit_sys, crit_user = build_critique_prompt(hypotheses)
    crit_raw, _, crit_cached = _call_llm(crit_sys, crit_user, "critique", model)
    critiques = _list_from(crit_raw, "critiques")

    verdict_by_name = {}
    for c in critiques:
        verdict_by_name[str(c.get("name", "")).strip().lower()] = c
    accepted = []
    for h in hypotheses:
        crit = verdict_by_name.get(str(h.get("name", "")).strip().lower())
        if crit is None or str(crit.get("verdict", "accept")).strip().lower() == "accept":
            h_with_crit = dict(h)
            h_with_crit["_critique"] = crit if crit else {"verdict": "accept (no critique)"}
            accepted.append(h_with_crit)
    print(f"[B] critique: {len(critiques)} reviews -> {len(accepted)} accepted "
          f"(from_cache={crit_cached})")

    comp_sys, comp_user = build_compile_prompt(accepted)
    comp_raw, _, comp_cached = _call_llm(comp_sys, comp_user, "compile", model)
    compiled = _list_from(comp_raw, "priors")

    exec_b, unsupported_b = [], []
    for d in compiled:
        supported = str(d.get("supported", "true")).strip().lower() in ("true", "1", "yes")
        e = compiled_to_exec(d) if supported else None
        if supported and e is not None:
            exec_b.append(e)
        else:
            unsupported_b.append({
                "name": d.get("name"),
                "reason": d.get("unsupported_reason", ""),
                "unmappable": (supported and e is None),
            })
    print(f"[B] compile: {len(compiled)} entries -> {len(exec_b)} executable, "
          f"{len(unsupported_b)} unsupported (from_cache={comp_cached})")

    # ---- validate (TRAIN/VAL V2.1 leave-one-in; no Chronos run, no TEST) ----
    rows_a = leave_one_in_executable(exec_a, dataset="ETTh1")
    rows_b = leave_one_in_executable(exec_b, dataset="ETTh1") if exec_b else []

    # ---- report ----
    n_prop_a = len(res_a["priors"])
    n_prop_b = len(hypotheses)
    sa = summarize(rows_a, n_prop_a)
    sb = summarize(rows_b, n_prop_b)

    def f(x, nd=4):
        return "-" if x is None else f"{x:.{nd}f}"

    print("\n" + "=" * 100)
    print("SINGLE-LLM vs MULTI-AGENT (ETTh1 TRAIN/VAL V2.1, identical settings)")
    print("=" * 100)
    header = ["Condition", "# prop", "# exec", "# rej/unsup", "uniq mech",
              "best fusMSE", "top-3 mean fusMSE", "best prior"]
    print(f"{header[0]:<14}{header[1]:>8}{header[2]:>8}{header[3]:>12}{header[4]:>10}"
          f"{header[5]:>13}{header[6]:>18}{header[7]:>30}")
    print("-" * 100)
    for label, s in (("A Single-LLM", sa), ("B Multi-Agent", sb)):
        print(f"{label:<14}{s['n_proposed']:>8}{s['n_executable']:>8}"
              f"{s['n_rejected_unsupported']:>12}{s['unique_mechanisms']:>10}"
              f"{f(s['best_val_fused_mse']):>13}{f(s['top3_mean_val_fused_mse']):>18}"
              f"{str(s['best_prior_name'] or '-'):>30}")
    print("-" * 100)

    # ---- save every prompt/output ----
    _save("results/multiagent/single_llm.json", {
        "condition": "A_single_llm", "model": model, "timestamp": ts,
        "from_cache": res_a["from_cache"],
        "prompt": res_a["prompt"], "raw_response": res_a["raw_response"],
        "parsed_priors": [p.to_dict() for p in res_a["priors"]],
    })
    _save("results/multiagent/multiagent_stages.json", {
        "condition": "B_multi_agent", "model": model, "timestamp": ts,
        "proposal": {"system": prop_sys, "user": prop_user,
                     "raw_response": prop_raw, "parsed": hypotheses},
        "critique": {"system": crit_sys, "user": crit_user,
                     "raw_response": crit_raw, "parsed": critiques,
                     "accepted": accepted},
        "compile": {"system": comp_sys, "user": comp_user,
                    "raw_response": comp_raw, "parsed": compiled},
        "executable": [p.to_dict() for p in exec_b],
        "unsupported": unsupported_b,
    })
    _save("results/multiagent/comparison.json", {
        "model": model, "timestamp": ts,
        "A_single_llm": sa, "B_multi_agent": sb,
        "rows_a": rows_a, "rows_b": rows_b,
    })
    print("\nSaved to results/multiagent/ (single_llm.json, multiagent_stages.json, "
          "comparison.json)")


if __name__ == "__main__":
    main()
