"""TRAIN-ONLY data-driven validation of candidate retrieval priors.

The core validation question is NOT "does the proposed pattern exist in the
signal", but rather: **does higher prior similarity between a query and a memory
predict more compatible historical futures** (lower MSE after the existing
mean-anchor transfer)?

For every operationalizable prior we report two scores:

  * ``structural_score`` ∈ [0,1] -- does the proposed relation exist in the
    observed training data (ACF / eta-squared / correlation evidence)?
  * ``predictive_relevance_score`` ∈ [0,1] -- do higher-prior-similarity pairs
    have more compatible futures? Defined as ``max(0, Spearman)`` where
    Spearman = Spearman(prior_similarity, -future_MSE) pooled over sampled
    (query, memory) pairs.

We also compare Top-K future MSE of prior-based vs shape-based (residual
Pearson) retrieval. Only TRAIN is used. Official test data is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import Prior

CONTEXT = 512
PRED = 96
K = 5
PERIOD = 24
TREND_WINDOW = 48

# --------------------------------------------------------------------------- #
# feature registry
# --------------------------------------------------------------------------- #

FEATURES: Dict[str, Dict[str, Any]] = {
    "hour": {"kind": "circular", "n": 24, "label": "hour-of-day"},
    "dow": {"kind": "circular", "n": 7, "label": "day-of-week"},
    "month": {"kind": "circular", "n": 12, "label": "month-of-year"},
    "ot_level": {"kind": "scalar", "label": "recent OT level"},
    "ot_slope": {"kind": "scalar", "label": "recent OT slope"},
    "load_level": {"kind": "scalar", "label": "recent total-load level"},
    "load_ramp": {"kind": "scalar", "label": "recent total-load change"},
    "load_profile": {"kind": "profile", "label": "lagged load profile"},
}


# --------------------------------------------------------------------------- #
# data loading (train only)
# --------------------------------------------------------------------------- #

def load_train(dataset: str = "ETTh1"):
    df = pd.read_csv(f"data/{dataset}.csv")
    dates = pd.to_datetime(df["date"])
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    dates = dates[train_mask].reset_index(drop=True)
    ot = df["OT"].to_numpy(dtype=np.float32)[train_mask]
    # total useful load = high + middle + low useful load (the heating driver)
    load = (
        df["HUFL"].to_numpy(dtype=np.float32)
        + df["MUFL"].to_numpy(dtype=np.float32)
        + df["LUFL"].to_numpy(dtype=np.float32)
    )[train_mask]
    return dates, ot, load


def build_windows(ot, load, dates):
    n = len(ot)
    starts = np.arange(0, n - CONTEXT - PRED + 1)
    contexts = np.stack([ot[s : s + CONTEXT] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s + CONTEXT : s + CONTEXT + PRED] for s in starts]).astype(np.float32)
    load_ctx = np.stack([load[s : s + CONTEXT] for s in starts]).astype(np.float32)
    ts = dates.iloc[starts + CONTEXT]  # forecast-start timestamp
    return {
        "starts": starts,
        "contexts": contexts,
        "futures": futures,
        "load_ctx": load_ctx,
        "hour": ts.dt.hour.to_numpy(),
        "dow": ts.dt.dayofweek.to_numpy(),
        "month": ts.dt.month.to_numpy() - 1,
    }


def _scalar_features(windows):
    ctx = windows["contexts"]
    fut = windows["futures"]
    load_ctx = windows["load_ctx"]

    ot_level = ctx.mean(axis=1)
    future_mean = fut.mean(axis=1)
    future_change = fut[:, -1] - ctx[:, -1]

    recent = ctx[:, -168:]
    t = np.arange(168, dtype=np.float64)
    t_c = t - t.mean()
    ot_slope = ((recent - recent.mean(axis=1, keepdims=True)) * t_c).sum(axis=1) / (t_c ** 2).sum()

    load_level = load_ctx.mean(axis=1)
    load_ramp = load_ctx[:, -1] - load_ctx[:, -25]  # 24-hour load change

    return {
        "ot_level": ot_level.astype(np.float64),
        "future_mean": future_mean.astype(np.float64),
        "future_change": future_change.astype(np.float64),
        "ot_slope": ot_slope.astype(np.float64),
        "load_level": load_level.astype(np.float64),
        "load_ramp": load_ramp.astype(np.float64),
    }


# --------------------------------------------------------------------------- #
# prior -> feature operationalization (explicit, transparent rules)
# --------------------------------------------------------------------------- #

def operationalize(prior: Prior) -> Optional[str]:
    """Map a Prior to a concrete feature key, or None if not operationalizable.

    A prior is not operationalizable when its meaning requires a variable that
    is unavailable in the data (e.g. ambient temperature). We do NOT silently
    substitute a calendar proxy for such a prior.
    """
    t = prior.type
    text = f"{prior.name or ''} {prior.condition or ''}".lower()

    if t == "periodic":
        return "hour"  # diurnal rhythm
    if t == "calendar":
        if any(k in text for k in ("week", "workday", "weekend", "working day", "day-of-week")):
            return "dow"
        if any(k in text for k in ("month", "season")):
            return "month"
        return "dow"
    if t == "trend":
        return "ot_slope"
    if t == "state":
        return "load_level" if "load" in text else "ot_level"
    if t == "operational":
        if any(k in text for k in ("profile", "lagged", "accumulated")):
            return "load_profile"
        if any(k in text for k in ("ramp", "spike", "increase", "decrease", "event")):
            return "load_ramp"
        if "night" in text or "cooling" in text:
            return "load_level"
        return "load_ramp"
    if t == "environmental":
        return None  # ambient temperature is not a variable in the ETT data
    if t == "event":
        return "load_ramp"
    return None  # "other"


# --------------------------------------------------------------------------- #
# similarity / statistical helpers
# --------------------------------------------------------------------------- #

def _circular_lut(n):
    lut = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            lut[i, j] = 1.0 - d / (n // 2)
    return lut


def _feature_values(feature, windows, scalar):
    if feature in ("hour", "dow", "month"):
        return windows[feature]
    if feature == "load_profile":
        return windows["load_ctx"]
    return scalar[feature]


def _feature_similarity(feature, q_feats, m_feats):
    from retrieval.similarity import pearson_similarity_matrix

    if feature in ("hour", "dow", "month"):
        lut = _circular_lut(FEATURES[feature]["n"])
        return lut[np.asarray(q_feats, dtype=int)][:, np.asarray(m_feats, dtype=int)]
    if feature == "load_profile":
        r = pearson_similarity_matrix(q_feats, m_feats)
        return (r + 1.0) / 2.0
    # scalar: standardized Gaussian kernel
    q = np.asarray(q_feats, dtype=np.float64)[:, None]
    m = np.asarray(m_feats, dtype=np.float64)[None, :]
    sigma = float(np.asarray(m_feats, dtype=np.float64).std())
    if sigma == 0:
        sigma = 1.0
    return np.exp(-0.5 * ((q - m) / sigma) ** 2)


def _linear_detrend(x):
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x), dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - (A @ coef)


def _acf(x, lag):
    x = x - x.mean()
    den = float((x * x).sum())
    if den == 0:
        return 0.0
    return float((x[lag:] * x[:-lag]).sum()) / den


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _eta2(groups, y):
    y = np.asarray(y, dtype=np.float64)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    within = 0.0
    for g in np.unique(groups):
        yg = y[groups == g]
        within += float(((yg - yg.mean()) ** 2).sum())
    return 1.0 - within / total


def _lagged_cross_corr(load, ot, max_lag=24):
    load = np.asarray(load, dtype=np.float64)
    ot = np.asarray(ot, dtype=np.float64)
    best = 0.0
    for lag in range(0, max_lag + 1):
        r = _corr(load, ot) if lag == 0 else _corr(load[:-lag], ot[lag:])
        best = max(best, abs(r))
    return best


def _spearman(a, b):
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).statistic)
    except Exception:
        pass
    # fallback: Pearson on average ranks (with tie handling)
    def rank_avg(x):
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ranks = np.empty(len(x), dtype=np.float64)
        i = 0
        n = len(x)
        while i < n:
            j = i
            while j + 1 < n and xs[j + 1] == xs[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return ranks

    ra, rb = rank_avg(np.asarray(a, dtype=np.float64)), rank_avg(np.asarray(b, dtype=np.float64))
    return float(np.corrcoef(ra, rb)[0, 1])


def _topk(scores, k):
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


# --------------------------------------------------------------------------- #
# structural score (does the proposed relation exist in train?)
# --------------------------------------------------------------------------- #

def _structural_score(feature, dates, ot, load, windows, scalar):
    if feature == "hour":
        return abs(_acf(_linear_detrend(ot), 24))
    if feature == "dow":
        return _eta2(dates.dt.dayofweek.to_numpy(), _linear_detrend(ot))
    if feature == "month":
        return _eta2(dates.dt.month.to_numpy() - 1, ot)
    if feature == "ot_level":
        return abs(_corr(scalar["ot_level"], scalar["future_mean"]))
    if feature == "ot_slope":
        return abs(_corr(scalar["ot_slope"], scalar["future_change"]))
    if feature == "load_level":
        return abs(_corr(scalar["load_level"], scalar["ot_level"]))
    if feature == "load_ramp":
        return abs(_corr(scalar["load_ramp"], scalar["future_change"]))
    if feature == "load_profile":
        return _lagged_cross_corr(load, ot, max_lag=24)
    return 0.0


# --------------------------------------------------------------------------- #
# validation driver
# --------------------------------------------------------------------------- #

def _decide(feature, structural, spearman):
    if feature is None:
        return "not_operationalizable_v1", "required variable (e.g. ambient temperature) unavailable"
    if structural < 0.05:
        return "reject", f"structural_score={structural:.3f} < 0.05 (pattern not evident in train)"
    if spearman <= 0.0:
        return "reject", f"Spearman={spearman:.3f} <= 0 (similarity does not predict better futures)"
    return "accept", "pattern present and positively predicts future compatibility"


def validate_priors(priors: List[Prior], dataset: str = "ETTh1", n_queries: int = 256):
    from retrieval.decompose import seasonal_decompose
    from retrieval.fusion import softmax
    from retrieval.similarity import pearson_similarity_matrix

    dates, ot, load = load_train(dataset)
    windows = build_windows(ot, load, dates)
    scalar = _scalar_features(windows)
    N = windows["contexts"].shape[0]

    # query subset (evenly spaced across train)
    q_idx = np.unique(np.round(np.linspace(0, N - 1, n_queries)).astype(int))

    # mean-anchor transfer pieces
    m_ctx_mean = windows["contexts"].mean(axis=1)
    q_ctx_mean = windows["contexts"][q_idx].mean(axis=1)
    A = windows["futures"][q_idx] - q_ctx_mean[:, None]  # (Q, H)
    B = windows["futures"] - m_ctx_mean[:, None]          # (N, H)
    meanA2 = (A ** 2).mean(axis=1)                         # (Q,)
    meanB2 = (B ** 2).mean(axis=1)                         # (N,)
    # future_MSE(q, m) = mean_h (transferred_m - future_q)^2  (exact, vectorized)
    future_mse = meanA2[:, None] + meanB2[None, :] - 2.0 * (A @ B.T) / PRED  # (Q, N)

    # shape-retrieval baseline: residual Pearson
    _, _, Rq, _, _ = seasonal_decompose(windows["contexts"][q_idx], PRED, PERIOD, TREND_WINDOW)
    _, _, Rm, _, _ = seasonal_decompose(windows["contexts"], PRED, PERIOD, TREND_WINDOW)
    shape_sim = pearson_similarity_matrix(Rq, Rm).astype(np.float64)  # (Q, N)

    # overlap-exclusion mask (avoid a memory overlapping its own query)
    valid = np.abs(windows["starts"][None, :] - windows["starts"][q_idx, None]) >= (CONTEXT + PRED)

    def topk_mse(scores):
        s = np.where(valid, scores, -np.inf)
        idx = _topk(s, K)
        sc = np.take_along_axis(s, idx, axis=-1)
        fut = windows["futures"][idx]          # (Q, K, H)
        cm = m_ctx_mean[idx]                   # (Q, K)
        w = softmax(sc)
        shape = fut - cm[:, :, None]
        y_mem = q_ctx_mean[:, None] + np.einsum("qk,qkh->qh", w, shape)
        return float(((y_mem - windows["futures"][q_idx]) ** 2).mean())

    shape_topk_mse = topk_mse(shape_sim)

    results: List[Dict[str, Any]] = []
    for p in priors:
        feature = operationalize(p)
        if feature is None:
            results.append(
                {
                    "name": p.name,
                    "type": p.type,
                    "llm_confidence": p.confidence,
                    "operationalizable": False,
                    "feature": None,
                    "structural_score": None,
                    "predictive_relevance_score": None,
                    "spearman": None,
                    "prior_topk_mse": None,
                    "shape_topk_mse": shape_topk_mse,
                    "status": "not_operationalizable_v1",
                    "reason": "required variable (e.g. ambient temperature) unavailable in data",
                }
            )
            continue

        prior_sim = _feature_similarity(
            feature,
            _feature_values(feature, windows, scalar)[q_idx],
            _feature_values(feature, windows, scalar),
        ).astype(np.float64)  # (Q, N)

        structural = _structural_score(feature, dates, ot, load, windows, scalar)

        # pooled Spearman over valid pairs: Spearman(prior_similarity, -future_MSE),
        # so a POSITIVE value means higher prior similarity -> LOWER future MSE
        # (more compatible futures).
        ps = prior_sim[valid]
        fm = future_mse[valid]
        spearman = _spearman(ps, -fm)
        predictive = max(0.0, spearman)

        prior_topk_mse = topk_mse(prior_sim)
        status, reason = _decide(feature, structural, spearman)

        results.append(
            {
                "name": p.name,
                "type": p.type,
                "llm_confidence": p.confidence,
                "operationalizable": True,
                "feature": feature,
                "structural_score": structural,
                "predictive_relevance_score": predictive,
                "spearman": spearman,
                "prior_topk_mse": prior_topk_mse,
                "shape_topk_mse": shape_topk_mse,
                "status": status,
                "reason": reason,
            }
        )

    return results


def _mem_forecast(S, mem_futures, mem_ctx_mean, q_ctx_mean, k=K):
    """Mean-anchor + softmax memory forecast from a (Q, M) similarity matrix."""
    from retrieval.fusion import softmax

    idx = _topk(S, k)
    sc = np.take_along_axis(S, idx, axis=-1)
    fut = mem_futures[idx]              # (Q, k, H)
    cm = mem_ctx_mean[idx]              # (Q, k)
    w = softmax(sc)                     # (Q, k)
    shape = fut - cm[:, :, None]
    return q_ctx_mean[:, None] + np.einsum("qk,qkh->qh", w, shape)


def leave_one_in(
    priors: List[Prior],
    dataset: str = "ETTh1",
    alpha_grid=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    lambda_fixed: float = 0.1,
    chunk: int = 256,
    cache_dir: str = "results",
):
    """Leave-one-in: add ONE prior to shape retrieval, select alpha on VAL only.

    For each prior, sweeps alpha over ``S = (1-alpha)*S_shape + alpha*S_prior``
    with the memory bank / Top-K / mean-anchor transfer / softmax aggregation
    held fixed. alpha is selected by VAL final-fusion MSE (Chronos fused at a
    FIXED ``lambda_fixed``). Never touches test data.

    Computation is chunked over validation queries so no full (val x memory)
    pairwise tensor is materialized. Returns one row per prior (same order).
    """
    from retrieval.decompose import seasonal_decompose
    from retrieval.similarity import pearson_similarity_matrix

    df = pd.read_csv(f"data/{dataset}.csv")
    dates = pd.to_datetime(df["date"])
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    val_mask = ((dates >= "2017-07-01") & (dates < "2017-11-01")).to_numpy()

    ot = df["OT"].to_numpy(dtype=np.float32)
    load = (
        df["HUFL"].to_numpy(dtype=np.float32)
        + df["MUFL"].to_numpy(dtype=np.float32)
        + df["LUFL"].to_numpy(dtype=np.float32)
    )

    mem_windows = build_windows(
        ot[train_mask], load[train_mask], dates[train_mask].reset_index(drop=True)
    )
    # Val windows use FULL-series context (may extend back into train), matching
    # the cached Chronos forecasts and the existing pipeline's windowing.
    val_start = int(np.flatnonzero(val_mask)[0])
    val_end = int(np.flatnonzero(val_mask)[-1]) + 1
    v_s = np.arange(val_start, val_end - PRED + 1)
    v_ts = dates.iloc[v_s]
    val_windows = {
        "starts": v_s,
        "contexts": np.stack([ot[i - CONTEXT : i] for i in v_s]).astype(np.float32),
        "futures": np.stack([ot[i : i + PRED] for i in v_s]).astype(np.float32),
        "load_ctx": np.stack([load[i - CONTEXT : i] for i in v_s]).astype(np.float32),
        "hour": v_ts.dt.hour.to_numpy(),
        "dow": v_ts.dt.dayofweek.to_numpy(),
        "month": v_ts.dt.month.to_numpy() - 1,
    }

    mem_scalar = _scalar_features(mem_windows)
    val_scalar = _scalar_features(val_windows)

    _, _, Rm, _, _ = seasonal_decompose(mem_windows["contexts"], PRED, PERIOD, TREND_WINDOW)
    _, _, Rv, _, _ = seasonal_decompose(val_windows["contexts"], PRED, PERIOD, TREND_WINDOW)

    y_chronos_val = np.load(f"{cache_dir}/{dataset}_chronos.npz")["y_chronos_val"]
    assert y_chronos_val.shape[0] == len(val_windows["contexts"]), "cached val forecasts mismatch"

    mem_futures = mem_windows["futures"]
    mem_ctx_mean = mem_windows["contexts"].mean(axis=1)
    val_futures = val_windows["futures"]
    val_ctx_mean = val_windows["contexts"].mean(axis=1)

    mem_feat = {f: _feature_values(f, mem_windows, mem_scalar) for f in FEATURES}
    val_feat = {f: _feature_values(f, val_windows, val_scalar) for f in FEATURES}

    used_features = sorted({operationalize(p) for p in priors if operationalize(p) is not None})
    nonzero = [a for a in alpha_grid if a != 0.0]

    n_val = len(val_windows["contexts"])
    n_pts = n_val * PRED

    chronos_sq = 0.0
    shape_mem_sq = 0.0
    shape_fus_sq = 0.0
    mem_sq = {f: {a: 0.0 for a in nonzero} for f in used_features}
    fus_sq = {f: {a: 0.0 for a in nonzero} for f in used_features}

    for q_start in range(0, n_val, chunk):
        q_end = min(q_start + chunk, n_val)
        sl = slice(q_start, q_end)

        S_shape = pearson_similarity_matrix(Rv[sl], Rm)  # (Q, M)

        y_mem_shape = _mem_forecast(S_shape, mem_futures, mem_ctx_mean, val_ctx_mean[sl])
        shape_mem_sq += float(((y_mem_shape - val_futures[sl]) ** 2).sum())

        yc = y_chronos_val[sl]
        fused_shape = (1 - lambda_fixed) * yc + lambda_fixed * y_mem_shape
        shape_fus_sq += float(((fused_shape - val_futures[sl]) ** 2).sum())
        chronos_sq += float(((yc - val_futures[sl]) ** 2).sum())

        for f in used_features:
            S_prior = _feature_similarity(f, val_feat[f][sl], mem_feat[f])  # (Q, M)
            for a in nonzero:
                S = (1 - a) * S_shape + a * S_prior
                y_mem = _mem_forecast(S, mem_futures, mem_ctx_mean, val_ctx_mean[sl])
                mem_sq[f][a] += float(((y_mem - val_futures[sl]) ** 2).sum())
                fused = (1 - lambda_fixed) * yc + lambda_fixed * y_mem
                fus_sq[f][a] += float(((fused - val_futures[sl]) ** 2).sum())

    chronos_mse = chronos_sq / n_pts
    shape_mem_mse = shape_mem_sq / n_pts
    shape_fus_mse = shape_fus_sq / n_pts

    rows = []
    for p in priors:
        f = operationalize(p)
        if f is None:
            rows.append(
                {
                    "name": p.name,
                    "type": p.type,
                    "feature": None,
                    "best_alpha": None,
                    "val_mem_mse": None,
                    "val_fus_mse": None,
                    "shape_mem_mse": shape_mem_mse,
                    "shape_fus_mse": shape_fus_mse,
                    "chronos_val_mse": chronos_mse,
                    "improvement": None,
                }
            )
            continue

        fus = {0.0: shape_fus_mse}
        mem = {0.0: shape_mem_mse}
        for a in nonzero:
            fus[a] = fus_sq[f][a] / n_pts
            mem[a] = mem_sq[f][a] / n_pts
        best_a = min(fus, key=fus.get)
        improvement = (shape_fus_mse - fus[best_a]) / shape_fus_mse * 100.0
        rows.append(
            {
                "name": p.name,
                "type": p.type,
                "feature": f,
                "best_alpha": best_a,
                "val_mem_mse": mem[best_a],
                "val_fus_mse": fus[best_a],
                "shape_mem_mse": shape_mem_mse,
                "shape_fus_mse": shape_fus_mse,
                "chronos_val_mse": chronos_mse,
                "improvement": improvement,
            }
        )

    return rows
