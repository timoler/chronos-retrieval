"""Agent v2 -- electricity load (UCI Electricity Load Diagrams).

Re-run of the electricity Multi-Agent pipeline with two added review gates so the
Agent itself can reject an unsupported operationalization instead of silently
compiling it.

Old flow:   Expert facts -> Hypothesis Generator -> Critic -> Compiler -> Retrieval
New flow:   Expert facts -> Hypothesis Generator -> Semantic Critic -> Data-Support
            Critic -> Compiler -> Proxy Validator -> Retrieval Validation

What changed vs. the old pipeline:

  * metadata now states the population relation (X1-X6 are only 6 of the 314
    consumers summed into Y) and the selection note (first six retained columns,
    no representativeness guarantee), plus explicit rules that variable existence
    is NOT proxy validity and that an observed subset must not be assumed to
    represent a population.
  * the old single Critic is split into a Semantic Critic (mechanism coherence /
    contradiction / leakage / duplication / existence / expressibility ONLY) and a
    Data-Support Critic (TRAIN-only empirical support / coverage / stability /
    unsupported assumptions -> ACCEPT / REJECT / UNSUPPORTED).
  * a Proxy Validator checks that the compiled executable spec still preserves the
    semantic scope of the mechanism (many/population/all must not silently become
    an arbitrary small subset).

X1..X6 are NOT changed: they remain MT_001 .. MT_006 (the first six retained
consumer columns). We do not repair the proxy, re-select representative consumers,
or hard-code any verdict. Every verdict is produced by the Agent from the metadata
rules plus deterministic TRAIN-only statistics. TEST is never accessed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from priors.executable import ExecutablePrior, compute_feature, feature_similarity
from priors.llm_prior import _default_model, _get_client
from priors.schema import _extract_json
from priors.validator import (
    CONTEXT,
    PRED,
    PERIOD,
    TREND_WINDOW,
    _mem_forecast,
)
from retrieval.decompose import seasonal_decompose
from retrieval.similarity import pearson_similarity_matrix

# --------------------------------------------------------------------------- #
# frozen experiment inputs (do NOT change X1..X6 / expert facts / split)
# --------------------------------------------------------------------------- #

# Blind variable -> real electricity.csv column.
#   Y  = OT                       (sum of the 314 retained consumers)
#   X1..X6 = HUFL..LULL           (MT_001 .. MT_006, first six retained columns)
BLIND_TO_REAL = {
    "Y": "OT",
    "X1": "HUFL", "X2": "HULL", "X3": "MUFL",
    "X4": "MULL", "X5": "LUFL", "X6": "LULL",
}
LOAD_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]  # == X1..X6
N_POPULATION = 314  # consumers summed into Y (established at preprocessing)

ELECTRICITY_METADATA_V2 = {
    "target_variable": "Y",
    "target_description": "total electricity demand, equal to the aggregate load of 314 retained consumers",
    "observed_variables": ["X1", "X2", "X3", "X4", "X5", "X6"],
    "observed_description": "six individual consumer loads corresponding to MT_001 through MT_006",
    "observed_population_relation": "X1-X6 are only 6 consumers among the 314 consumers included in Y",
    "selection_note": "X1-X6 were selected as the first six retained columns, with no representativeness guarantee",
    "sampling_frequency": "15 minutes",
}

# Expert mechanism facts (UNCHANGED from the old electricity run).
ELECTRICITY_EXPERT_MECHANISM = (
    "electricity demand has a strong recurring temporal usage structure\n"
    "recent demand tends to persist into the near future because consumption patterns change gradually\n"
    "working and non-working periods correspond to systematically different demand regimes\n"
    "aggregate demand is the sum of many individual consumers whose loads rise and fall together\n"
    "aggregate demand changes smoothly because it averages over many independent consumers"
)

VAL_STRIDE = 4          # 15-min data, like ETTm
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAMBDA = 0.1
CHUNK = 256
CACHE_DIR = "cache/llm_priors_v2"

# --------------------------------------------------------------------------- #
# shared prompt blocks
# --------------------------------------------------------------------------- #

def _metadata_text() -> str:
    m = ELECTRICITY_METADATA_V2
    return (
        "DATASET (anonymized):\n"
        f"- target variable: {m['target_variable']} ({m['target_description']})\n"
        f"- observed variables: {', '.join(m['observed_variables'])} ({m['observed_description']})\n"
        f"- observed population relation: {m['observed_population_relation']}\n"
        f"- selection note: {m['selection_note']}\n"
        f"- sampling frequency: {m['sampling_frequency']}"
    )


RULES_BLOCK = (
    "CRITICAL RULES:\n"
    "1. Do not assume that X1-X6 are representative of the 314-consumer population.\n"
    "2. Do not infer population-level coverage from variable availability.\n"
    "3. If a hypothesis requires X1-X6 to represent the full population, that assumption "
    "must be explicitly justified by TRAIN-only evidence.\n"
    "4. A plausible mechanism is not automatically a valid executable prior.\n"
    "5. Variable existence is not equivalent to proxy validity.\n"
    "6. Do not assume that an observed subset represents a larger population.\n"
    "7. Whenever a hypothesis refers to \"many\", \"population\", \"aggregate\", "
    "\"system-wide\", \"common factor\", or equivalent concepts, explicitly verify whether "
    "the available variables cover or validly proxy that population-level mechanism.\n"
    "8. If the mechanism is plausible but the available variables do not faithfully "
    "instantiate it, return UNSUPPORTED.\n"
    "9. Do not repair the hypothesis by silently changing variables.\n"
    "10. Do not search for alternative variables merely to make the hypothesis work.\n"
    "11. The role of the critic is to reject unsupported operationalizations, not to "
    "rescue every hypothesis.\n"
    "The critic should be willing to say: \"The idea is reasonable, but this dataset "
    "cannot support it.\""
)


def _whitelist_block() -> str:
    from priors.executable import (
        ALLOWED_AGGREGATIONS,
        ALLOWED_SIMILARITIES,
        ALLOWED_TRANSFORMS,
    )

    return (
        "ALLOWED OPERATIONS (use ONLY these):\n"
        f"  transforms: {', '.join(sorted(ALLOWED_TRANSFORMS))}\n"
        f"  temporal_aggregation: {', '.join(sorted(ALLOWED_AGGREGATIONS))}\n"
        f"  similarity: {', '.join(sorted(ALLOWED_SIMILARITIES))}\n"
        "\n"
        "  - raw: use the variable as-is\n"
        "  - sum: sum the input variables into one series\n"
        "  - square: square each value (x^2)\n"
        "  - signed_power: sign(x)*|x|^p (specify power p)\n"
        "  - difference: first difference (point-to-point change)\n"
        "  - slope: linear slope over the recent context\n"
        "  - ewma: exponentially weighted moving average\n"
        "  - residual_from_train_fitted_equilibrium: residual of Y against a LINEAR fit\n"
        "  - concatenate: join multiple variables into a multivariate profile\n"
        "  - temporal_aggregation: last=final value; mean=average; window=full context "
        "(use with similarity=correlation)\n"
        "  - similarity: abs_diff (scalars) or correlation (window profiles)\n"
    )


# --------------------------------------------------------------------------- #
# TRAIN-only evidence (deterministic, never from the LLM)
# --------------------------------------------------------------------------- #

def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _load_train() -> tuple:
    df = pd.read_csv("data/electricity.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    series = {"OT": df["OT"].to_numpy(np.float64)}
    for v in LOAD_VARS:
        series[v] = df[v].to_numpy(np.float64)
    return df, dates, series, N, train_len


def compute_train_evidence() -> dict:
    """Deterministic TRAIN-only statistics used by the Data-Support Critic."""
    df, dates, series, N, train_len = _load_train()
    dt = dates.iloc[:train_len]

    y = series["OT"][:train_len].astype(np.float64)
    s6 = np.sum([series[v][:train_len].astype(np.float64) for v in LOAD_VARS], axis=0)

    # -- S6 = sum(X1..X6) vs Y --
    corr_s6y = _corr(s6, y)
    diff_corr_s6y = _corr(np.diff(s6), np.diff(y))
    ratio = s6 / y
    ratio_stats = {
        "mean": float(np.mean(ratio)),
        "std": float(np.std(ratio)),
        "min": float(np.min(ratio)),
        "max": float(np.max(ratio)),
    }

    months = dt.dt.month.to_numpy()
    monthly = {int(m): round(_corr(s6[months == m], y[months == m]), 4)
               for m in sorted(set(months.tolist()))}

    hours = dt.dt.hour.to_numpy()
    tod = {}
    for lo, hi in ((0, 6), (6, 12), (12, 18), (18, 24)):
        mask = (hours >= lo) & (hours < hi)
        tod[f"{lo:02d}-{hi:02d}h"] = round(_corr(s6[mask], y[mask]), 4)

    coverage = {
        "mechanism_scope": f"{N_POPULATION} consumers (population-level aggregate)",
        "proxy_scope": f"{len(LOAD_VARS)} consumers",
        "fraction": round(len(LOAD_VARS) / N_POPULATION, 4),
    }

    # -- Y persistence (proxy == target) --
    lag1 = _corr(y[1:], y[:-1])
    lag96 = _corr(y[96:], y[:-96])  # same clock-time, previous day
    yc = y[:-96]
    yf = np.array([y[i + 1 : i + 1 + 96].mean() for i in range(len(y) - 96)])
    cur_vs_future = _corr(yc, yf)

    return {
        "train_len": int(train_len),
        "N": int(N),
        "s6_vs_y": {
            "description": "sum(X1..X6) vs Y (population total), TRAIN only",
            "pearson_corr": round(corr_s6y, 4),
            "first_difference_corr": round(diff_corr_s6y, 4),
            "relative_magnitude": ratio_stats,
            "coverage": coverage,
            "monthly_corr": monthly,
            "time_of_day_corr": tod,
        },
        "y_persistence": {
            "description": "Y vs its own past/future, TRAIN only",
            "lag1_autocorr": round(lag1, 4),
            "same_clock_prev_day_corr": round(lag96, 4),
            "current_vs_next96_mean_corr": round(cur_vs_future, 4),
        },
    }


# --------------------------------------------------------------------------- #
# LLM call helper (cache-aware; DeepSeek anthropic-compatible endpoint)
# --------------------------------------------------------------------------- #

def _call_llm(system: str, user: str, stage: str, model: str):
    key = hashlib.sha256(
        json.dumps({"stage": stage, "system": system, "user": user, "model": model},
                   sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path(CACHE_DIR)
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
        raise RuntimeError(
            f"LLM returned no text (stop_reason={getattr(resp, 'stop_reason', None)})"
        )

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


# --------------------------------------------------------------------------- #
# Stage 1: Hypothesis Generator (proposal) -- unchanged from the old pipeline
# --------------------------------------------------------------------------- #

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
        "DOMAIN FACTS (physical mechanisms; NOT retrieval features):\n" + ELECTRICITY_EXPERT_MECHANISM,
        schema,
        "TASK: propose 4-6 retrieval hypotheses. Output only the JSON object.",
    ])
    return _PROP_SYSTEM, user


# --------------------------------------------------------------------------- #
# Stage 2: Semantic Critic -- mechanism coherence ONLY (no proxy validity)
# --------------------------------------------------------------------------- #

_SEMANTIC_SYSTEM = (
    "You are a SEMANTIC reviewer of retrieval hypotheses. You assess ONLY whether a "
    "mechanism is internally coherent and logically sound. You do NOT judge whether "
    "the available data actually supports the mechanism, and you do NOT decide whether "
    "a proxy is valid -- that is a separate data-support review.\n\n"
    "For each hypothesis assess: (1) mechanism plausibility, (2) logical contradiction, "
    "(3) leakage of benchmark/dataset/prior results, (4) duplication with another "
    "hypothesis, (5) whether the referenced variables exist, (6) whether the mechanism "
    "is expressible in principle. IMPORTANT: variable existence is NOT proxy validity -- "
    "the fact that X1-X6 exist does not mean they validly represent a population-level "
    "mechanism. A plausible mechanism is not automatically a valid executable prior.\n\n"
    "Output strict JSON only."
)


def build_semantic_prompt(hypotheses):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"critiques": [ ... ]}, one entry per hypothesis. Each entry:\n'
        '  - "name": hypothesis name (copy exactly)\n'
        '  - "mechanism_plausible": true or false\n'
        '  - "verdict": "accept" or "reject"\n'
        '  - "consistency": note on physical/logical consistency\n'
        '  - "contradiction": note on any logical contradiction\n'
        '  - "leakage": note on benchmark/dataset/prior-result leakage\n'
        '  - "duplication": note on duplication with another hypothesis\n'
        '  - "variable_existence": note on whether referenced variables exist (existence ONLY, NOT proxy validity)\n'
        '  - "theoretical_expressibility": note on whether the mechanism can be expressed in principle\n'
        '  - "summary": 1-sentence rationale\n'
        'Set "verdict" to "reject" ONLY if the mechanism is implausible, self-contradictory, '
        'leaks, duplicative, references nonexistent variables, or cannot be expressed at all. '
        'A plausible-but-data-unsupported proxy is still "accept" here (data support is decided later).'
    )
    user = "\n\n".join([
        _metadata_text(),
        "DOMAIN FACTS:\n" + ELECTRICITY_EXPERT_MECHANISM,
        "HYPOTHESES TO REVIEW:\n" + json.dumps(hypotheses, ensure_ascii=False, indent=2),
        schema,
        "TASK: critique every hypothesis at the SEMANTIC level only. Output only the JSON object.",
    ])
    return _SEMANTIC_SYSTEM, user


# --------------------------------------------------------------------------- #
# Stage 3: Data-Support Critic -- TRAIN evidence -> ACCEPT/REJECT/UNSUPPORTED
# --------------------------------------------------------------------------- #

_DATA_SUPPORT_SYSTEM = (
    "You are a DATA-SUPPORT reviewer. You judge whether the TRAIN data actually "
    "supports turning a hypothesis into an executable retrieval prior. A plausible "
    "mechanism is not automatically a valid executable prior. Variable existence is "
    "not equivalent to proxy validity. Do not assume an observed subset represents a "
    "larger population.\n\n"
    "You are given TRAIN-ONLY statistical evidence (computed deterministically, NOT "
    "by you). Base your decision on that evidence and the coverage facts. Do not "
    "invent statistics. Output strict JSON only."
)


def build_data_support_prompt(entries, evidence):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"assessments": [ ... ]}, one entry per hypothesis. Each entry:\n'
        '  - "hypothesis_id": short id (copy exactly from input)\n'
        '  - "name": hypothesis name (copy exactly)\n'
        '  - "mechanism_plausible": true or false (echo the semantic assessment)\n'
        '  - "required_assumptions": list of strings -- the implicit assumptions the hypothesis makes\n'
        '  - "proxy_candidate": the concrete observable proxy this hypothesis implies (e.g. "sum(X1..X6)")\n'
        '  - "coverage_check": {"status": "PASS"|"WEAK"|"FAIL", "evidence": {...}}\n'
        '  - "empirical_support": {"status": "PASS"|"WEAK"|"FAIL", "evidence": {...}}\n'
        '  - "stability_check": {"status": "PASS"|"WEAK"|"FAIL", "evidence": {...}}\n'
        '  - "proxy_validity": "SUPPORTED"|"WEAK"|"UNSUPPORTED"\n'
        '  - "decision": "ACCEPT"|"REJECT"|"UNSUPPORTED"\n'
        '  - "reason": 1-3 sentences\n\n'
        "DECISION MEANINGS (use exactly):\n"
        "  ACCEPT      = mechanism plausible AND the current data supports the corresponding proxy.\n"
        "  REJECT      = mechanism itself is invalid (leakage, duplicate, contradiction, logically invalid).\n"
        "  UNSUPPORTED = mechanism may be reasonable, but the current data has insufficient evidence "
        "to reliably turn it into an executable retrieval prior.\n\n"
        "If a mechanism refers to 'many / population / aggregate / system-wide / common factor' but the "
        "available proxy only covers a small arbitrary subset and there is no representativeness "
        "evidence, mark coverage WEAK/FAIL, proxy_validity UNSUPPORTED, decision UNSUPPORTED."
    )
    user = "\n\n".join([
        _metadata_text(),
        RULES_BLOCK,
        "TRAIN-ONLY EVIDENCE (deterministic):\n" + json.dumps(evidence, ensure_ascii=False, indent=2),
        "HYPOTHESES TO ASSESS (with hypothesis_id and semantic assessment):\n"
        + json.dumps(entries, ensure_ascii=False, indent=2),
        schema,
        "TASK: assess every hypothesis at the data-support level. Output only the JSON object.",
    ])
    return _DATA_SUPPORT_SYSTEM, user


# --------------------------------------------------------------------------- #
# Stage 4: Compiler -- accepted -> executable spec (unchanged from old pipeline)
# --------------------------------------------------------------------------- #

_COMP_SYSTEM = (
    "You convert accepted retrieval hypotheses into EXECUTABLE retrieval priors using "
    "ONLY the whitelist of operations below. If a hypothesis cannot be faithfully "
    "represented by the whitelist, mark it supported=false with a reason. NEVER "
    "silently approximate. Output strict JSON only."
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
        "ACCEPTED HYPOTHESES (semantic + data-support accepted):\n"
        + json.dumps(accepted, ensure_ascii=False, indent=2),
        schema,
        "TASK: compile each accepted hypothesis into an executable prior, marking any "
        "that cannot be faithfully represented as supported=false. Output only the JSON.",
    ])
    return _COMP_SYSTEM, user


# --------------------------------------------------------------------------- #
# Stage 5: Proxy Validator -- compiled spec still faithful to mechanism scope?
# --------------------------------------------------------------------------- #

_PROXY_VALIDATOR_SYSTEM = (
    "You validate that a compiled executable proxy still faithfully represents the "
    "original mechanism. Specifically, check whether the compiler preserved the "
    "semantic scope of the mechanism, or silently narrowed a population-level "
    "mechanism to an arbitrary small subset.\n\n"
    "If a mechanism refers to 'many / population / all / aggregate / common factor' "
    "and the compiled proxy reduces it to an arbitrary small subset of observed "
    "variables with no representativeness evidence, you must FAIL it.\n\n"
    "Output strict JSON only."
)


def build_proxy_validator_prompt(compiled_entries):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"validations": [ ... ]}, one entry per compiled prior. Each entry:\n'
        '  - "hypothesis_id": short id (copy exactly from input)\n'
        '  - "name": hypothesis name (copy exactly)\n'
        '  - "original_mechanism": the population-level mechanism from the hypothesis\n'
        '  - "compiled_proxy": the compiled executable spec (input_variables/transform/temporal_aggregation/similarity)\n'
        '  - "semantic_fidelity": "HIGH"|"MEDIUM"|"LOW"\n'
        '  - "coverage_mismatch": true or false\n'
        '  - "unsupported_assumptions": list of strings\n'
        '  - "decision": "PASS"|"FAIL"\n'
        '  - "reason": 1-3 sentences\n\n'
        "RULE: mechanism scope 'many / population / all' compiled to 'arbitrary small subset' "
        "with no representativeness evidence -> coverage_mismatch=true, semantic_fidelity=LOW, decision=FAIL."
    )
    user = "\n\n".join([
        _metadata_text(),
        RULES_BLOCK,
        "COMPILED PRIORS TO VALIDATE:\n" + json.dumps(compiled_entries, ensure_ascii=False, indent=2),
        schema,
        "TASK: validate every compiled prior. Output only the JSON object.",
    ])
    return _PROXY_VALIDATOR_SYSTEM, user


# --------------------------------------------------------------------------- #
# mapping helpers (blind -> real)
# --------------------------------------------------------------------------- #

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
        source="agent_v2",
        input_variables=inputs,
        transform=str(d.get("transform", "")).strip().lower(),
        temporal_aggregation=str(d.get("temporal_aggregation", "")).strip().lower(),
        similarity=str(d.get("similarity", "")).strip().lower(),
        expected_mechanism=str(d.get("expected_mechanism", "")).strip(),
        power=power,
        confidence=float(d.get("confidence", 0.0)),
    )


# --------------------------------------------------------------------------- #
# Stage 6: Retrieval validation (TRAIN memory + VAL query; TEST never touched)
# --------------------------------------------------------------------------- #

def run_retrieval_validation(exec_priors):
    """VAL leave-one-in sweep for the supported priors, numpy-only (cached Chronos)."""
    np.random.seed(0)
    df = pd.read_csv("data/electricity.csv")
    N = len(df)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    val_start, val_end = train_len, train_len + val_len
    series = {"OT": df["OT"].to_numpy(np.float32)}
    for v in LOAD_VARS:
        series[v] = df[v].to_numpy(np.float32)
    ot = series["OT"]

    mem_starts = np.arange(0, train_len - CONTEXT - PRED + 1)
    val_starts = np.arange(val_start, val_end - PRED + 1, VAL_STRIDE)
    mem_ctx = np.stack([ot[m:m + CONTEXT] for m in mem_starts]).astype(np.float32)
    mem_fut = np.stack([ot[m + CONTEXT:m + CONTEXT + PRED] for m in mem_starts]).astype(np.float32)
    val_ctx = np.stack([ot[s - CONTEXT:s] for s in val_starts]).astype(np.float32)
    val_fut = np.stack([ot[s:s + PRED] for s in val_starts]).astype(np.float32)

    yc = np.load("results/electricity_chronos.npz")["y_chronos_val"].astype(np.float32)
    assert yc.shape[0] == len(val_starts), (
        f"cached chronos mismatch: {yc.shape[0]} vs {len(val_starts)}"
    )

    _, _, Rm, _, _ = seasonal_decompose(mem_ctx, PRED, PERIOD, TREND_WINDOW)
    mem_ctx_mean = mem_ctx.mean(axis=1)
    val_ctx_mean = val_ctx.mean(axis=1)
    n_val = len(val_ctx)
    n_pts = n_val * PRED
    nonzero = [a for a in ALPHAS if a != 0.0]

    pf = {}
    for p in exec_priors:
        mf, qf, _ = compute_feature(p, series, mem_starts, val_starts, np.zeros(2))
        pf[p.name] = (p, mf, qf)

    chronos_sq = shape_mem_sq = shape_fus_sq = 0.0
    prior_mem_sq = {p.name: {a: 0.0 for a in nonzero} for p in exec_priors}
    prior_fus_sq = {p.name: {a: 0.0 for a in nonzero} for p in exec_priors}

    for q0 in range(0, n_val, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, n_val))
        _, _, Rv, _, _ = seasonal_decompose(val_ctx[sl], PRED, PERIOD, TREND_WINDOW)
        S_shape = pearson_similarity_matrix(Rv, Rm)
        y_mem_shape = _mem_forecast(S_shape, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
        shape_mem_sq += float(((y_mem_shape - val_fut[sl]) ** 2).sum())
        yc_sl = yc[sl]
        fused_shape = (1 - LAMBDA) * yc_sl + LAMBDA * y_mem_shape
        shape_fus_sq += float(((fused_shape - val_fut[sl]) ** 2).sum())
        chronos_sq += float(((yc_sl - val_fut[sl]) ** 2).sum())
        for name, (p, mf, qf) in pf.items():
            S_prior = feature_similarity(p, qf[sl], mf)
            for a in nonzero:
                S = (1 - a) * S_shape + a * S_prior
                y_mem = _mem_forecast(S, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
                prior_mem_sq[name][a] += float(((y_mem - val_fut[sl]) ** 2).sum())
                fused = (1 - LAMBDA) * yc_sl + LAMBDA * y_mem
                prior_fus_sq[name][a] += float(((fused - val_fut[sl]) ** 2).sum())

    chronos_mse = chronos_sq / n_pts
    shape_mem_mse = shape_mem_sq / n_pts
    shape_fus_mse = shape_fus_sq / n_pts

    rows = []
    for p in exec_priors:
        fus = {0.0: shape_fus_mse}
        mem = {0.0: shape_mem_mse}
        for a in nonzero:
            fus[a] = prior_fus_sq[p.name][a] / n_pts
            mem[a] = prior_mem_sq[p.name][a] / n_pts
        best_a = min(fus, key=fus.get)
        rows.append(dict(
            name=p.name, best_alpha=best_a,
            val_mem_mse=float(mem[best_a]), val_fus_mse=float(fus[best_a]),
            improvement=float((shape_fus_mse - fus[best_a]) / shape_fus_mse * 100.0),
        ))
    return dict(chronos_mse=float(chronos_mse), shape_mem_mse=float(shape_mem_mse),
                shape_fus_mse=float(shape_fus_mse), rows=rows)


# --------------------------------------------------------------------------- #
# pipeline driver
# --------------------------------------------------------------------------- #

def _spec(p):
    v = p.input_variables
    if set(v) == set(LOAD_VARS):
        inp = "X1..X6"
    elif v == ["OT"]:
        inp = "Y"
    else:
        inp = "+".join(v)
    return f"{p.transform}/{p.temporal_aggregation}/{p.similarity} on {inp}"


def run_pipeline(model):
    evidence = compute_train_evidence()

    # ---- 1. proposal ----
    prop_sys, prop_user = build_proposal_prompt()
    prop_raw, prop_path, prop_cached = _call_llm(prop_sys, prop_user, "proposal", model)
    hypotheses = _list_from(prop_raw, "hypotheses")
    for i, h in enumerate(hypotheses, 1):
        h["hypothesis_id"] = f"H{i}"

    # ---- 2. semantic critic ----
    sem_sys, sem_user = build_semantic_prompt(hypotheses)
    sem_raw, sem_path, sem_cached = _call_llm(sem_sys, sem_user, "semantic_critic", model)
    critiques = _list_from(sem_raw, "critiques")
    by_name = {str(c.get("name", "")).strip().lower(): c for c in critiques}
    for h in hypotheses:
        c = by_name.get(str(h.get("name", "")).strip().lower())
        h["_semantic"] = c if c else {"verdict": "accept (no review)"}

    sem_accepted = [h for h in hypotheses
                    if str(h["_semantic"].get("verdict", "accept")).strip().lower() == "accept"]

    # ---- 3. data-support critic (semantic-accepted only) ----
    ds_entries = [{"hypothesis_id": h["hypothesis_id"], "name": h["name"],
                   "domain_fact": h.get("domain_fact", ""),
                   "observable_proxy": h.get("observable_proxy", ""),
                   "mechanism_plausible": h["_semantic"].get("mechanism_plausible", True)}
                  for h in sem_accepted]
    ds_sys, ds_user = build_data_support_prompt(ds_entries, evidence)
    ds_raw, ds_path, ds_cached = _call_llm(ds_sys, ds_user, "data_support_critic", model)
    ds_assessments = _list_from(ds_raw, "assessments")
    ds_by_id = {str(a.get("hypothesis_id", "")).strip().upper(): a for a in ds_assessments}
    for h in sem_accepted:
        a = ds_by_id.get(str(h.get("hypothesis_id", "")).strip().upper())
        if a is None:
            a = ds_by_id.get(str(h.get("name", "")).strip().lower())
        h["_data_support"] = a if a else {"decision": "UNSUPPORTED", "reason": "no assessment"}

    data_accepted = [h for h in sem_accepted
                     if str(h["_data_support"].get("decision", "")).strip().upper() == "ACCEPT"]

    # ---- 4. compiler (data-support ACCEPT only) ----
    compiled = []
    comp_sys = comp_user = comp_raw = comp_path = comp_cached = None
    if data_accepted:
        comp_sys, comp_user = build_compile_prompt(data_accepted)
        comp_raw, comp_path, comp_cached = _call_llm(comp_sys, comp_user, "compiler", model)
        compiled = _list_from(comp_raw, "priors")

    exec_by_name = {}
    unsupported_compile = []
    for d in compiled:
        supported = str(d.get("supported", "true")).strip().lower() in ("true", "1", "yes")
        e = compiled_to_exec(d) if supported else None
        if supported and e is not None:
            exec_by_name[str(d.get("name", "")).strip().lower()] = (d, e)
        else:
            unsupported_compile.append({"name": d.get("name"),
                                        "reason": d.get("unsupported_reason", "")})

    # attach compiled spec to hypotheses
    for h in hypotheses:
        h["_compiled"] = None
        if h in data_accepted:
            d = exec_by_name.get(str(h.get("name", "")).strip().lower())
            h["_compiled"] = d[0] if d else None

    compiled_hyps = [h for h in data_accepted if h["_compiled"] is not None]

    # ---- 5. proxy validator (compiled only) ----
    pv_by_id = {}
    pv_sys = pv_user = pv_raw = pv_path = pv_cached = None
    if compiled_hyps:
        pv_entries = [{"hypothesis_id": h["hypothesis_id"], "name": h["name"],
                       "original_mechanism": h.get("domain_fact", ""),
                       "compiled_proxy": h["_compiled"]} for h in compiled_hyps]
        pv_sys, pv_user = build_proxy_validator_prompt(pv_entries)
        pv_raw, pv_path, pv_cached = _call_llm(pv_sys, pv_user, "proxy_validator", model)
        pv_validations = _list_from(pv_raw, "validations")
        pv_by_id = {str(v.get("hypothesis_id", "")).strip().upper(): v for v in pv_validations}

    for h in compiled_hyps:
        v = pv_by_id.get(str(h.get("hypothesis_id", "")).strip().upper())
        if v is None:
            v = pv_by_id.get(str(h.get("name", "")).strip().lower())
        h["_proxy_validator"] = v if v else {"decision": "FAIL", "reason": "no validation"}

    # ---- final decision + supported priors ----
    supported_hyps = [h for h in compiled_hyps
                      if str(h["_proxy_validator"].get("decision", "")).strip().upper() == "PASS"]

    exec_priors = [exec_by_name[str(h.get("name", "")).strip().lower()][1]
                   for h in supported_hyps]

    def final_decision(h):
        if str(h["_semantic"].get("verdict", "accept")).strip().lower() != "accept":
            return "REJECTED (semantic)"
        d = str(h["_data_support"].get("decision", "")).strip().upper()
        if d == "REJECT":
            return "REJECTED (data)"
        if d == "UNSUPPORTED":
            return "UNSUPPORTED"
        if h["_compiled"] is None:
            return "REJECTED (compiler)"
        if str(h["_proxy_validator"].get("decision", "")).strip().upper() != "PASS":
            return "REJECTED (proxy validator)"
        return "ACCEPTED (supported prior)"

    rows = []
    for h in hypotheses:
        rows.append({
            "id": h["hypothesis_id"],
            "name": h.get("name", ""),
            "hypothesis": h.get("domain_fact", ""),
            "semantic_critic": str(h["_semantic"].get("verdict", "")).strip().lower(),
            "data_support": str(h["_data_support"].get("decision", "")).strip().upper()
            if h.get("_data_support") else "n/a",
            "compiler": _spec(exec_by_name[str(h.get("name", "")).strip().lower()][1])
            if str(h.get("name", "")).strip().lower() in exec_by_name else "not compiled",
            "proxy_validator": str(h["_proxy_validator"].get("decision", "")).strip().upper()
            if h.get("_proxy_validator") else "n/a",
            "final_decision": final_decision(h),
        })

    return {
        "model": model,
        "metadata": ELECTRICITY_METADATA_V2,
        "expert_mechanism": ELECTRICITY_EXPERT_MECHANISM,
        "evidence": evidence,
        "proposal": {"system": prop_sys, "user": prop_user, "raw_response": prop_raw,
                     "parsed": hypotheses, "cache_path": prop_path, "from_cache": prop_cached},
        "semantic_critic": {"system": sem_sys, "user": sem_user, "raw_response": sem_raw,
                            "parsed": critiques, "cache_path": sem_path, "from_cache": sem_cached},
        "data_support_critic": {"system": ds_sys, "user": ds_user, "raw_response": ds_raw,
                                "parsed": ds_assessments, "cache_path": ds_path, "from_cache": ds_cached},
        "compiler": {"system": comp_sys, "user": comp_user, "raw_response": comp_raw,
                     "parsed": compiled, "cache_path": comp_path, "from_cache": comp_cached},
        "proxy_validator": {"system": pv_sys if compiled_hyps else None,
                            "user": pv_user if compiled_hyps else None,
                            "raw_response": pv_raw, "parsed": list(pv_by_id.values()),
                            "cache_path": pv_path, "from_cache": pv_cached},
        "unsupported_compile": unsupported_compile,
        "supported_hyps": [h for h in supported_hyps],
        "supported_priors": [p.to_dict() for p in exec_priors],
        "rows": rows,
        "n_proposed": len(hypotheses),
        "n_semantic_accepted": len(sem_accepted),
        "n_data_supported": len(data_accepted),
        "n_data_unsupported": len([h for h in sem_accepted
                                   if str(h["_data_support"].get("decision", "")).strip().upper() == "UNSUPPORTED"]),
        "n_data_rejected": len([h for h in sem_accepted
                                if str(h["_data_support"].get("decision", "")).strip().upper() == "REJECT"]),
        "n_compiled": len(compiled_hyps),
        "n_proxy_passed": len(supported_hyps),
    }


def main():
    model = _default_model()
    result = run_pipeline(model)

    # ---- retrieval validation (only supported priors) ----
    retrieval = None
    if result["supported_priors"]:
        try:
            exec_priors = [ExecutablePrior(**d) for d in result["supported_priors"]]
            retrieval = run_retrieval_validation(exec_priors)
        except Exception as e:  # noqa: BLE001 -- surface, do not abort the audit
            retrieval = {"error": f"{type(e).__name__}: {e}"}

    # ---- print report table ----
    print("=" * 100)
    print("ELECTRICITY Agent v2 (Semantic Critic -> Data-Support Critic -> Proxy Validator)")
    print("=" * 100)
    print(f"model = {model}")
    print(f"proposed          : {result['n_proposed']}")
    print(f"semantic accepted : {result['n_semantic_accepted']}")
    print(f"data supported    : {result['n_data_supported']}")
    print(f"data unsupported  : {result['n_data_unsupported']}")
    print(f"data rejected     : {result['n_data_rejected']}")
    print(f"compiled          : {result['n_compiled']}")
    print(f"proxy passed      : {result['n_proxy_passed']}")

    print("\n" + "-" * 100)
    hdr = f"{'ID':<4}{'hypothesis':<22}{'semantic':<10}{'data':<13}{'compiler':<30}{'proxy':<7}{'final'}"
    print(hdr)
    print("-" * 100)
    for r in result["rows"]:
        print(f"{r['id']:<4}{r['name']:<22}{r['semantic_critic']:<10}{r['data_support']:<13}"
              f"{r['compiler']:<30}{r['proxy_validator']:<7}{r['final_decision']}")

    if retrieval:
        print("\nRETRIEVAL VALIDATION (VAL only):")
        if "error" in retrieval:
            print(f"  error: {retrieval['error']}")
        else:
            print(f"  Chronos VAL MSE   : {retrieval['chronos_mse']:.4f}")
            print(f"  Shape Memory MSE  : {retrieval['shape_mem_mse']:.4f}")
            print(f"  Shape Fusion MSE  : {retrieval['shape_fus_mse']:.4f}")
            for r in retrieval["rows"]:
                print(f"  [{r['name']}] alpha={r['best_alpha']} memMSE={r['val_mem_mse']:.4f} "
                      f"fusMSE={r['val_fus_mse']:.4f} imp={r['improvement']:+.2f}%")
    else:
        print("\nRETRIEVAL VALIDATION: no supported expert retrieval prior -> "
              "expert_prior_available = false (legal result, not a pipeline failure)")

    # ---- audit JSON ----
    ts = datetime.now().isoformat(timespec="seconds")
    audit = {
        "dataset": "electricity",
        "agent_version": "v2",
        "model": model,
        "timestamp": ts,
        "metadata_sent_to_llm": result["metadata"],
        "expert_facts": result["expert_mechanism"],
        "train_evidence": result["evidence"],
        "proposal": {k: v for k, v in result["proposal"].items() if k != "parsed"},
        "hypotheses": result["proposal"]["parsed"],
        "semantic_critic": result["semantic_critic"],
        "data_support_critic": result["data_support_critic"],
        "compiler": result["compiler"],
        "proxy_validator": result["proxy_validator"],
        "unsupported_compile": result["unsupported_compile"],
        "supported_priors": result["supported_priors"],
        "summary": {
            "n_proposed": result["n_proposed"],
            "n_semantic_accepted": result["n_semantic_accepted"],
            "n_data_supported": result["n_data_supported"],
            "n_data_unsupported": result["n_data_unsupported"],
            "n_data_rejected": result["n_data_rejected"],
            "n_compiled": result["n_compiled"],
            "n_proxy_passed": result["n_proxy_passed"],
            "expert_prior_available": len(result["supported_priors"]) > 0,
        },
        "rows": result["rows"],
        "retrieval_validation": retrieval,
        "test_access": "NO (TRAIN/VAL only; TEST untouched)",
        "leakage_check": "TRAIN evidence + VAL retrieval only; TEST never read",
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/electricity_agent_v2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nSaved results/electricity_agent_v2_audit.json")

    return audit


if __name__ == "__main__":
    main()
