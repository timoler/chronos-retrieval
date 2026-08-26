"""Executable retrieval priors (Prompt V2.1).

Converts a mechanism-derived LLM prior into a concrete retrieval feature using
ONLY a whitelist of operations. The executable specification (input_variables /
transform / temporal_aggregation / similarity) is produced by the LLM and must
pass compiler validation; we never hand-pick an implementation for a better MSE.

Features are constructed from the TRAIN/VAL-observable series only -- the
equilibrium model is fit on TRAIN, and no future or test data is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import _extract_json
from .validator import (
    CONTEXT,
    PRED,
    K,
    PERIOD,
    TREND_WINDOW,
    _corr,
    _mem_forecast,
    _spearman,
    _topk,
)

LOAD_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
ALL_VARS = LOAD_VARS + ["OT"]

ALLOWED_TRANSFORMS = {
    "raw",
    "sum",
    "square",
    "difference",
    "slope",
    "signed_power",
    "ewma",
    "residual_from_train_fitted_equilibrium",
    "concatenate",
}
ALLOWED_AGGREGATIONS = {"window", "mean", "last"}
ALLOWED_SIMILARITIES = {"correlation", "abs_diff"}

# transforms whose feature is a "change" (structural score uses future_change)
_CHANGE_TRANSFORMS = {"difference", "slope", "residual_from_train_fitted_equilibrium"}

# validation/query window stride per dataset, matching the cached Chronos forecast
# windowing in run_temporal_ablation_all_ett.py (hourly ETTh stride 1, 15-min ETTm
# stride 4). Memory-bank construction stays stride 1.
VAL_STRIDE = {"ETTh1": 1, "ETTh2": 1, "ETTm1": 4, "ETTm2": 4}


@dataclass
class ExecutablePrior:
    name: str
    source: str
    input_variables: List[str]
    transform: str
    temporal_aggregation: str
    similarity: str
    expected_mechanism: str
    power: Optional[float] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Fixed control prior: metadata-only autoregressive prior on OT. Included in the
# V2.1 leave-one-in experiment as a baseline against which the mechanism-derived
# priors are compared. It is a hand-specified control (not LLM-generated), so it
# does NOT go through compiler hand-picking -- it simply encodes "OT is
# autoregressive" as an executable spec (raw OT window, correlation similarity).
AR_OT_CONTROL = ExecutablePrior(
    name="AR-OT (metadata-only control)",
    source="metadata_llm",
    input_variables=["OT"],
    transform="raw",
    temporal_aggregation="window",
    similarity="correlation",
    expected_mechanism=(
        "OT is autoregressive: its own recent trajectory is the retrieval key; "
        "similar recent OT histories should have similar near-term futures."
    ),
    confidence=1.0,
)


def parse_executable(text: str, source: str) -> List[ExecutablePrior]:
    data = _extract_json(text)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "priors" in data:
        items = data["priors"]
    else:
        raise ValueError("expected a JSON object with a 'priors' list")

    out: List[ExecutablePrior] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"prior entry is not an object: {item!r}")
        out.append(
            ExecutablePrior(
                name=str(item.get("name", "")).strip(),
                source=source,
                input_variables=[str(v).strip().upper() for v in item.get("input_variables", [])],
                transform=str(item.get("transform", "")).strip().lower(),
                temporal_aggregation=str(item.get("temporal_aggregation", "")).strip().lower(),
                similarity=str(item.get("similarity", "")).strip().lower(),
                power=float(item["power"]) if item.get("power") is not None else None,
                expected_mechanism=str(item.get("expected_mechanism", "")).strip(),
                confidence=float(item.get("confidence", 0.0)),
            )
        )
    return out


def compile_prior(prior: ExecutablePrior) -> Tuple[Optional[str], str]:
    """Validate a prior against the operation whitelist.

    Returns (feature_key, reason). feature_key is None when rejected.
    """
    if prior.transform not in ALLOWED_TRANSFORMS:
        return None, f"unsupported transform {prior.transform!r}"
    if prior.temporal_aggregation not in ALLOWED_AGGREGATIONS:
        return None, f"unsupported aggregation {prior.temporal_aggregation!r}"
    if prior.similarity not in ALLOWED_SIMILARITIES:
        return None, f"unsupported similarity {prior.similarity!r}"
    if not prior.input_variables:
        return None, "no input_variables"
    for v in prior.input_variables:
        if v not in ALL_VARS:
            return None, f"unknown variable {v!r}"
    if prior.transform == "concatenate" and prior.similarity != "correlation":
        return None, "concatenate requires similarity=correlation"
    if prior.temporal_aggregation == "window" and prior.similarity != "correlation":
        return None, "window aggregation requires similarity=correlation"
    if prior.transform == "residual_from_train_fitted_equilibrium":
        if "OT" not in prior.input_variables:
            return None, "residual_from_train_fitted_equilibrium requires OT"
        if not set(prior.input_variables) & set(LOAD_VARS):
            return None, "residual_from_train_fitted_equilibrium requires a load variable"
    if prior.transform == "signed_power":
        if prior.power is None or not (float(prior.power) > 0.0):
            return None, "signed_power requires power > 0"
    # canonical key for display / dispatch
    if prior.transform == "signed_power":
        key = f"signed_power(p={prior.power})|{prior.temporal_aggregation}|{'/'.join(prior.input_variables)}"
    else:
        key = f"{prior.transform}|{prior.temporal_aggregation}|{'/'.join(prior.input_variables)}"
    return key, "ok"


# --------------------------------------------------------------------------- #
# data + feature computation (per-point transform, then per-window slice)
# --------------------------------------------------------------------------- #

def _load_full(dataset: str):
    df = pd.read_csv(f"data/{dataset}.csv")
    dates = pd.to_datetime(df["date"])
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()
    series = {"OT": df["OT"].to_numpy(dtype=np.float32)}
    for v in LOAD_VARS:
        series[v] = df[v].to_numpy(dtype=np.float32)
    series["total_load"] = np.sum([series[v] for v in LOAD_VARS], axis=0)
    return df, dates, train_mask, val_mask, series


def _ewma(x, span: int = 24):
    x = np.asarray(x, dtype=np.float64)
    alpha = 2.0 / (span + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _slope(x):
    r = np.asarray(x, dtype=np.float64)[-168:]
    t = np.arange(len(r), dtype=np.float64)
    t = t - t.mean()
    denom = float((t ** 2).sum())
    if denom == 0:
        return 0.0
    return float(((r - r.mean()) * t).sum() / denom)


def _equilibrium_fit(series, train_len):
    """Fit OT ~ total_load linearly on TRAIN only."""
    ot = series["OT"][:train_len].astype(np.float64)
    tl = series["total_load"][:train_len].astype(np.float64)
    A = np.vstack([tl, np.ones_like(tl)]).T
    coef, *_ = np.linalg.lstsq(A, ot, rcond=None)
    return coef  # [a, b]


def compute_feature(
    prior: ExecutablePrior,
    series: Dict[str, np.ndarray],
    mem_starts: np.ndarray,
    val_starts: np.ndarray,
    eq_coef: np.ndarray,
):
    """Return (mem_feat, val_feat, kind). kind in {"scalar", "vector"}."""
    t = prior.transform
    agg = prior.temporal_aggregation

    if t == "concatenate":
        base = np.stack([series[v].astype(np.float64) for v in prior.input_variables], axis=1)  # (N, n)
        mem = np.stack([base[s : s + CONTEXT].ravel() for s in mem_starts])  # (M, CONTEXT*n)
        val = np.stack([base[s - CONTEXT : s].ravel() for s in val_starts])  # (V, CONTEXT*n)
        return mem, val, "vector"

    # residual-from-equilibrium uses the train-fitted model (no future/test info)
    if t == "residual_from_train_fitted_equilibrium":
        a, b = eq_coef
        s = series["OT"].astype(np.float64) - (a * series["total_load"].astype(np.float64) + b)
    else:
        if len(prior.input_variables) == 1:
            s = series[prior.input_variables[0]].astype(np.float64)
        else:
            s = np.sum([series[v].astype(np.float64) for v in prior.input_variables], axis=0)

        if t == "square":
            s = s ** 2
        elif t == "signed_power":
            # sign-preserving power: sign(x) * |x|^p (safe for negative x)
            s = np.sign(s) * np.abs(s) ** float(prior.power)
        elif t == "difference":
            s = np.concatenate([[s[0]], np.diff(s)])
        elif t == "ewma":
            s = _ewma(s, span=24)

    if t == "slope":
        mem = np.array([_slope(s[st : st + CONTEXT]) for st in mem_starts])
        val = np.array([_slope(s[st - CONTEXT : st]) for st in val_starts])
        return mem, val, "scalar"

    if agg == "window":
        mem = np.stack([s[st : st + CONTEXT] for st in mem_starts])
        val = np.stack([s[st - CONTEXT : st] for st in val_starts])
        return mem, val, "vector"
    if agg == "mean":
        mem = np.array([s[st : st + CONTEXT].mean() for st in mem_starts])
        val = np.array([s[st - CONTEXT : st].mean() for st in val_starts])
        return mem, val, "scalar"
    if agg == "last":
        mem = np.array([s[st + CONTEXT - 1] for st in mem_starts])
        val = np.array([s[st - 1] for st in val_starts])
        return mem, val, "scalar"
    raise ValueError(f"bad aggregation {agg!r}")


def feature_similarity(prior: ExecutablePrior, q_feat, m_feat):
    from retrieval.similarity import pearson_similarity_matrix

    if prior.similarity == "correlation":
        r = pearson_similarity_matrix(q_feat, m_feat)
        return (r + 1.0) / 2.0
    # abs_diff -> standardized Gaussian kernel
    q = np.asarray(q_feat, dtype=np.float64)[:, None]
    m = np.asarray(m_feat, dtype=np.float64)[None, :]
    sigma = float(np.asarray(m_feat, dtype=np.float64).std())
    if sigma == 0:
        sigma = 1.0
    return np.exp(-0.5 * ((q - m) / sigma) ** 2)


def _structural(prior: ExecutablePrior, mem_feat, kind, ot_level, future_mean, future_change):
    if kind == "vector":
        scalar = mem_feat.mean(axis=1)  # reduce profile to a scalar proxy
    else:
        scalar = np.asarray(mem_feat, dtype=np.float64)
    ref = future_change if prior.transform in _CHANGE_TRANSFORMS else future_mean
    return abs(_corr(scalar, ref))


# --------------------------------------------------------------------------- #
# TRAIN validation (structural + Spearman) and VAL leave-one-in
# --------------------------------------------------------------------------- #

def _prepare(dataset: str):
    df, dates, train_mask, val_mask, series = _load_full(dataset)
    train_len = int(train_mask.sum())

    mem_starts = np.arange(0, train_len - CONTEXT - PRED + 1)
    val_start = int(np.flatnonzero(val_mask)[0])
    val_end = int(np.flatnonzero(val_mask)[-1]) + 1
    val_stride = VAL_STRIDE.get(dataset, 1)
    val_starts = np.arange(val_start, val_end - PRED + 1, val_stride)

    ot = series["OT"]
    mem_ctx = np.stack([ot[s : s + CONTEXT] for s in mem_starts]).astype(np.float32)
    mem_fut = np.stack([ot[s + CONTEXT : s + CONTEXT + PRED] for s in mem_starts]).astype(np.float32)
    val_ctx = np.stack([ot[s - CONTEXT : s] for s in val_starts]).astype(np.float32)
    val_fut = np.stack([ot[s : s + PRED] for s in val_starts]).astype(np.float32)

    mem_ctx_mean = mem_ctx.mean(axis=1)
    val_ctx_mean = val_ctx.mean(axis=1)

    eq_coef = _equilibrium_fit(series, train_len)

    # scalar OT stats for structural (on memory windows)
    future_mean = mem_fut.mean(axis=1)
    future_change = mem_fut[:, -1] - mem_ctx[:, -1]
    ot_level = mem_ctx.mean(axis=1)

    return dict(
        series=series, mem_starts=mem_starts, val_starts=val_starts,
        mem_ctx=mem_ctx, mem_fut=mem_fut, val_ctx=val_ctx, val_fut=val_fut,
        mem_ctx_mean=mem_ctx_mean, val_ctx_mean=val_ctx_mean, eq_coef=eq_coef,
        future_mean=future_mean, future_change=future_change, ot_level=ot_level,
        dataset=dataset,
    )


def _validate_one(prior, key, prep):
    from retrieval.decompose import seasonal_decompose
    from retrieval.similarity import pearson_similarity_matrix

    mem_feat, val_feat, kind = compute_feature(
        prior, prep["series"], prep["mem_starts"], prep["val_starts"], prep["eq_coef"]
    )
    structural = _structural(
        prior, mem_feat, kind, prep["ot_level"], prep["future_mean"], prep["future_change"]
    )

    # train-based Spearman over sampled (query, memory) pairs
    q_idx = np.unique(np.round(np.linspace(0, len(prep["mem_ctx"]) - 1, 256)).astype(int))
    A = prep["mem_fut"][q_idx] - prep["mem_ctx_mean"][q_idx, None]
    B = prep["mem_fut"] - prep["mem_ctx_mean"][:, None]
    meanA2 = (A ** 2).mean(axis=1)
    meanB2 = (B ** 2).mean(axis=1)
    future_mse = meanA2[:, None] + meanB2[None, :] - 2.0 * (A @ B.T) / PRED
    valid = np.abs(prep["mem_starts"][None, :] - prep["mem_starts"][q_idx, None]) >= (CONTEXT + PRED)

    sim = feature_similarity(prior, mem_feat[q_idx], mem_feat)
    spearman = _spearman(sim[valid], -future_mse[valid])
    return dict(feature=key, structural=structural, spearman=spearman,
                mem_feat=mem_feat, val_feat=val_feat, kind=kind)


def leave_one_in_executable(
    priors: List[ExecutablePrior],
    dataset: str = "ETTh1",
    alpha_grid=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    lambda_fixed: float = 0.1,
    chunk: int = 256,
):
    from retrieval.decompose import seasonal_decompose
    from retrieval.similarity import pearson_similarity_matrix

    prep = _prepare(dataset)
    y_chronos_val = np.load(f"results/{dataset}_chronos.npz")["y_chronos_val"]
    assert y_chronos_val.shape[0] == len(prep["val_ctx"]), (
        f"cached val forecasts mismatch for {dataset}: "
        f"y_chronos_val={y_chronos_val.shape[0]} vs val_ctx={len(prep['val_ctx'])} "
        f"(val_stride={VAL_STRIDE.get(dataset, 1)})"
    )

    _, _, Rm, _, _ = seasonal_decompose(prep["mem_ctx"], PRED, PERIOD, TREND_WINDOW)
    _, _, Rv, _, _ = seasonal_decompose(prep["val_ctx"], PRED, PERIOD, TREND_WINDOW)

    n_val = len(prep["val_ctx"])
    n_pts = n_val * PRED
    nonzero = [a for a in alpha_grid if a != 0.0]

    chronos_sq = 0.0
    shape_mem_sq = 0.0
    shape_fus_sq = 0.0
    # per prior: {(a): mem_sq}, {(a): fus_sq}
    prior_mem_sq: Dict[str, Dict[float, float]] = {}
    prior_fus_sq: Dict[str, Dict[float, float]] = {}
    # precompute per-prior feature (full val + memory)
    prior_feats = {}
    for p in priors:
        key, reason = compile_prior(p)
        if key is None:
            prior_feats[p.name] = None
            continue
        mem_feat, val_feat, kind = compute_feature(
            p, prep["series"], prep["mem_starts"], prep["val_starts"], prep["eq_coef"]
        )
        prior_feats[p.name] = (p, mem_feat, val_feat, kind)
        prior_mem_sq[p.name] = {a: 0.0 for a in nonzero}
        prior_fus_sq[p.name] = {a: 0.0 for a in nonzero}

    for q_start in range(0, n_val, chunk):
        q_end = min(q_start + chunk, n_val)
        sl = slice(q_start, q_end)
        S_shape = pearson_similarity_matrix(Rv[sl], Rm)

        y_mem_shape = _mem_forecast(S_shape, prep["mem_fut"], prep["mem_ctx_mean"], prep["val_ctx_mean"][sl])
        shape_mem_sq += float(((y_mem_shape - prep["val_fut"][sl]) ** 2).sum())
        yc = y_chronos_val[sl]
        fused_shape = (1 - lambda_fixed) * yc + lambda_fixed * y_mem_shape
        shape_fus_sq += float(((fused_shape - prep["val_fut"][sl]) ** 2).sum())
        chronos_sq += float(((yc - prep["val_fut"][sl]) ** 2).sum())

        for name, entry in prior_feats.items():
            if entry is None:
                continue
            p, mem_feat, val_feat, kind = entry
            S_prior = feature_similarity(p, val_feat[sl], mem_feat)
            for a in nonzero:
                S = (1 - a) * S_shape + a * S_prior
                y_mem = _mem_forecast(S, prep["mem_fut"], prep["mem_ctx_mean"], prep["val_ctx_mean"][sl])
                prior_mem_sq[name][a] += float(((y_mem - prep["val_fut"][sl]) ** 2).sum())
                fused = (1 - lambda_fixed) * yc + lambda_fixed * y_mem
                prior_fus_sq[name][a] += float(((fused - prep["val_fut"][sl]) ** 2).sum())

    chronos_mse = chronos_sq / n_pts
    shape_mem_mse = shape_mem_sq / n_pts
    shape_fus_mse = shape_fus_sq / n_pts

    rows = []
    for p in priors:
        key, reason = compile_prior(p)
        if key is None:
            rows.append(dict(name=p.name, source=p.source, feature=None, reason=reason,
                             structural=None, spearman=None, best_alpha=None,
                             val_mem_mse=None, val_fus_mse=None, improvement=None,
                             shape_mem_mse=shape_mem_mse, shape_fus_mse=shape_fus_mse,
                             chronos_val_mse=chronos_mse))
            continue
        v = _validate_one(p, key, prep)
        fus = {0.0: shape_fus_mse}
        mem = {0.0: shape_mem_mse}
        for a in nonzero:
            fus[a] = prior_fus_sq[p.name][a] / n_pts
            mem[a] = prior_mem_sq[p.name][a] / n_pts
        best_a = min(fus, key=fus.get)
        improvement = (shape_fus_mse - fus[best_a]) / shape_fus_mse * 100.0
        rows.append(dict(name=p.name, source=p.source, feature=key, reason=reason,
                         structural=v["structural"], spearman=v["spearman"], best_alpha=best_a,
                         val_mem_mse=mem[best_a], val_fus_mse=fus[best_a], improvement=improvement,
                         shape_mem_mse=shape_mem_mse, shape_fus_mse=shape_fus_mse,
                         chronos_val_mse=chronos_mse))
    return rows


def multi_prior_combination(
    priors: List[ExecutablePrior],
    dataset: str = "ETTh1",
    alpha_grid=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    lambda_fixed: float = 0.1,
    chunk: int = 256,
    structural_min: float = 0.05,
    best_single_name: str = "squared_load_profile",
):
    """Combine validated priors into one retrieval score (TRAIN/VAL only).

    Candidate selection keeps only operationalizable priors that pass the TRAIN
    validation gate (structural >= ``structural_min`` and Spearman > 0), then
    deduplicates by feature key (first occurrence wins). Strategies:

      A  shape-only baseline (alpha = 0)
      B  best single prior (one-hot on ``best_single_name``)
      C  equal-weight accepted priors
      D  LLM-confidence weighted priors
      E  validation-score (TRAIN Spearman) weighted priors

    For C/D/E the combined prior score is S_prior = sum_i w_i S_i (sum_i w_i = 1);
    the final score is S = (1-alpha) S_shape + alpha S_prior, swept over the fixed
    VAL alpha grid. TEST is never accessed.
    """
    from retrieval.decompose import seasonal_decompose
    from retrieval.similarity import pearson_similarity_matrix

    prep = _prepare(dataset)
    y_chronos_val = np.load(f"results/{dataset}_chronos.npz")["y_chronos_val"]
    assert y_chronos_val.shape[0] == len(prep["val_ctx"]), (
        f"cached val forecasts mismatch for {dataset}: "
        f"y_chronos_val={y_chronos_val.shape[0]} vs val_ctx={len(prep['val_ctx'])} "
        f"(val_stride={VAL_STRIDE.get(dataset, 1)})"
    )

    _, _, Rm, _, _ = seasonal_decompose(prep["mem_ctx"], PRED, PERIOD, TREND_WINDOW)
    _, _, Rv, _, _ = seasonal_decompose(prep["val_ctx"], PRED, PERIOD, TREND_WINDOW)

    n_val = len(prep["val_ctx"])
    n_pts = n_val * PRED
    nonzero = [a for a in alpha_grid if a != 0.0]

    # ---- per-prior TRAIN validation + dedup + accept gate ----
    seen: Dict[str, bool] = {}
    candidates = []
    rejected = []
    for p in priors:
        key, reason = compile_prior(p)
        if key is None:
            rejected.append(dict(name=p.name, key=None, reason=reason))
            continue
        if key in seen:
            continue
        seen[key] = True
        v = _validate_one(p, key, prep)
        structural, spearman = v["structural"], v["spearman"]
        if structural < structural_min:
            rejected.append(dict(name=p.name, key=key,
                                 reason=f"structural={structural:.3f} < {structural_min}"))
            continue
        if spearman <= 0.0:
            rejected.append(dict(name=p.name, key=key,
                                 reason=f"Spearman={spearman:.3f} <= 0"))
            continue
        candidates.append(dict(prior=p, key=key, mem_feat=v["mem_feat"],
                               val_feat=v["val_feat"], kind=v["kind"],
                               structural=structural, spearman=spearman))

    n = len(candidates)

    def _norm(w):
        w = np.asarray(w, dtype=np.float64)
        s = w.sum()
        return w / s if s > 0 else np.full(n, 1.0 / n)

    equal_w = _norm(np.ones(n))
    conf_w = _norm(np.array([c["prior"].confidence for c in candidates]))
    val_w = _norm(np.array([c["spearman"] for c in candidates]))

    best_idx = next((i for i, c in enumerate(candidates)
                     if c["prior"].name == best_single_name), None)
    if best_idx is None:
        best_idx = next((i for i, c in enumerate(candidates)
                         if c["prior"].transform == "square"
                         and c["prior"].temporal_aggregation == "window"), None)
    one_hot = np.zeros(n)
    if best_idx is not None:
        one_hot[best_idx] = 1.0

    strategy_specs = [
        ("B_best_single", one_hot, "best single prior"),
        ("C_equal_weight", equal_w, "equal-weight accepted priors"),
        ("D_confidence_weight", conf_w, "LLM-confidence weighted priors"),
        ("E_validation_weight", val_w, "validation-score (TRAIN Spearman) weighted priors"),
    ]

    # ---- chunked VAL sweep ----
    chronos_sq = 0.0
    shape_mem_sq = 0.0
    shape_fus_sq = 0.0
    strat_mem_sq = {name: {a: 0.0 for a in nonzero} for name, _, _ in strategy_specs}
    strat_fus_sq = {name: {a: 0.0 for a in nonzero} for name, _, _ in strategy_specs}

    for q_start in range(0, n_val, chunk):
        q_end = min(q_start + chunk, n_val)
        sl = slice(q_start, q_end)
        S_shape = pearson_similarity_matrix(Rv[sl], Rm)

        y_mem_shape = _mem_forecast(
            S_shape, prep["mem_fut"], prep["mem_ctx_mean"], prep["val_ctx_mean"][sl]
        )
        shape_mem_sq += float(((y_mem_shape - prep["val_fut"][sl]) ** 2).sum())
        yc = y_chronos_val[sl]
        fused_shape = (1 - lambda_fixed) * yc + lambda_fixed * y_mem_shape
        shape_fus_sq += float(((fused_shape - prep["val_fut"][sl]) ** 2).sum())
        chronos_sq += float(((yc - prep["val_fut"][sl]) ** 2).sum())

        for name, wvec, _label in strategy_specs:
            S_prior = None
            for i, c in enumerate(candidates):
                w = wvec[i]
                if w == 0.0:
                    continue
                Si = feature_similarity(c["prior"], c["val_feat"][sl], c["mem_feat"])
                S_prior = w * Si if S_prior is None else S_prior + w * Si
            for a in nonzero:
                S = (1 - a) * S_shape + a * S_prior
                y_mem = _mem_forecast(
                    S, prep["mem_fut"], prep["mem_ctx_mean"], prep["val_ctx_mean"][sl]
                )
                strat_mem_sq[name][a] += float(((y_mem - prep["val_fut"][sl]) ** 2).sum())
                fused = (1 - lambda_fixed) * yc + lambda_fixed * y_mem
                strat_fus_sq[name][a] += float(((fused - prep["val_fut"][sl]) ** 2).sum())

    chronos_mse = chronos_sq / n_pts
    shape_mem_mse = shape_mem_sq / n_pts
    shape_fus_mse = shape_fus_sq / n_pts

    # ---- build rows ----
    names = [c["prior"].name for c in candidates]
    rows = [
        dict(method="A_shape_only", included=[], weights=[], best_alpha=0.0,
             val_mem_mse=shape_mem_mse, val_fus_mse=shape_fus_mse, imp_shape=0.0)
    ]
    for name, wvec, _label in strategy_specs:
        fus = {0.0: shape_fus_mse}
        mem = {0.0: shape_mem_mse}
        for a in nonzero:
            fus[a] = strat_fus_sq[name][a] / n_pts
            mem[a] = strat_mem_sq[name][a] / n_pts
        best_a = min(fus, key=fus.get)
        included = [nm for nm, w in zip(names, wvec) if w > 0]
        weights = [round(float(w), 4) for w in wvec if w > 0]
        rows.append(dict(method=name, included=included, weights=weights,
                         best_alpha=best_a, val_mem_mse=mem[best_a],
                         val_fus_mse=fus[best_a],
                         imp_shape=(shape_fus_mse - fus[best_a]) / shape_fus_mse * 100.0))

    best_single_fus = next(r["val_fus_mse"] for r in rows if r["method"] == "B_best_single")
    for r in rows:
        r["imp_best"] = (best_single_fus - r["val_fus_mse"]) / best_single_fus * 100.0

    return dict(
        candidates=[dict(name=c["prior"].name, source=c["prior"].source, key=c["key"],
                         structural=c["structural"], spearman=c["spearman"],
                         confidence=c["prior"].confidence) for c in candidates],
        rejected=rejected,
        strategies=rows,
        shape_mem_mse=shape_mem_mse,
        shape_fus_mse=shape_fus_mse,
        chronos_mse=chronos_mse,
        best_single_fus_mse=best_single_fus,
    )
