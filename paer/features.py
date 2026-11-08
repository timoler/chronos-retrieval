"""Observable window features and soft conditions used by the frozen retriever."""
from typing import Any, Optional

import numpy as np


FEATURE_REGISTRY = {
    "hour_of_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "categories": {"morning": [6, 7, 8, 9, 10, 11], "midday": [12, 13, 14, 15],
                                   "evening": [16, 17, 18, 19, 20, 21],
                                   "night": [22, 23, 0, 1, 2, 3, 4, 5]}},
    "day_of_week": {"kind": "calendar", "source_type": "timestamp_known",
                    "categories": {"weekday": [0, 1, 2, 3, 4], "weekend": [5, 6]}},
    "working_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "categories": {"workday": [0, 1, 2, 3, 4], "non_working": [5, 6]}},
    "seasonal_level": {"kind": "calendar", "source_type": "timestamp_known",
                       "categories": {"winter": [12, 1, 2], "spring": [3, 4, 5],
                                      "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
    "recent_level": {"kind": "continuous", "source_type": "context_derived"},
    "recent_volatility": {"kind": "continuous", "source_type": "context_derived"},
    "recent_slope_magnitude": {"kind": "continuous", "source_type": "context_derived"},
    "recent_diff_magnitude": {"kind": "continuous", "source_type": "context_derived"},
    "recent_autocorr": {"kind": "continuous", "source_type": "context_derived"},
}


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


def _normalize_qualifier(q: Any) -> Optional[str]:
    if q is None:
        return None
    q = str(q).strip().lower()
    if q.startswith("is_"):
        q = q[3:]
    return QUALIFIER_SYNONYMS.get(q, q)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _linear_slope_vec(X):
    """每行 [L] 的线性斜率（向量化）。"""
    L = X.shape[1]
    t = np.linspace(-1.0, 1.0, L, dtype=np.float32)
    tc = t - t.mean()
    denom = np.sum(tc * tc)
    xm = X.mean(axis=1, keepdims=True)
    return np.sum((X - xm) * tc[None, :], axis=1) / denom


def _lag1_autocorr_vec(X):
    x1 = X[:, 1:]
    x0 = X[:, :-1]
    m1 = x1.mean(axis=1, keepdims=True)
    m0 = x0.mean(axis=1, keepdims=True)
    num = ((x1 - m1) * (x0 - m0)).sum(axis=1)
    den = np.sqrt(((x1 - m1) ** 2).sum(axis=1) * ((x0 - m0) ** 2).sum(axis=1))
    return num / (den + 1e-9)


def compute_training_stats(X_train_all):
    """TRAIN-only feature stats for z-scoring recent_* features."""
    ctx_means = X_train_all.mean(axis=1)
    ctx_stds = X_train_all.std(axis=1)
    slopes = _linear_slope_vec(X_train_all)
    last_diffs = np.abs(X_train_all[:, -1] - X_train_all[:, -2])
    autocorrs = _lag1_autocorr_vec(X_train_all)
    return {
        "level_mean": float(ctx_means.mean()), "level_std": float(ctx_means.std()),
        "vol_mean": float(ctx_stds.mean()), "vol_std": float(ctx_stds.std()),
        "slope_abs_mean": float(np.abs(slopes).mean()), "slope_abs_std": float(np.abs(slopes).std()),
        "diff_abs_mean": float(last_diffs.mean()), "diff_abs_std": float(last_diffs.std()),
    }


def compute_window_feats(X, tss, stats):
    """Per-window raw feature values (calendar units + continuous z-scores)."""
    n = len(X)
    months = np.array([ts.month for ts in tss], dtype=np.int64)
    hours = np.array([ts.hour for ts in tss], dtype=np.int64)
    dows = np.array([ts.dayofweek for ts in tss], dtype=np.int64)
    ctx_means = X.mean(axis=1)
    ctx_stds = X.std(axis=1)
    slopes = np.abs(_linear_slope_vec(X))
    last_diffs = np.abs(X[:, -1] - X[:, -2])
    autocorrs = _lag1_autocorr_vec(X)
    return {
        "seasonal_level": months,
        "hour_of_day": hours,
        "working_day": dows,
        "day_of_week": dows,
        "recent_level": (ctx_means - stats["level_mean"]) / (stats["level_std"] + 1e-6),
        "recent_volatility": (ctx_stds - stats["vol_mean"]) / (stats["vol_std"] + 1e-6),
        "recent_diff_magnitude": (last_diffs - stats["diff_abs_mean"]) / (stats["diff_abs_std"] + 1e-6),
        "recent_slope_magnitude": (slopes - stats["slope_abs_mean"]) / (stats["slope_abs_std"] + 1e-6),
        "recent_autocorr": (autocorrs + 1.0) / 2.0,
    }


def expert_activation(compiled, feats, i):
    """Evaluate a compiled condition_spec at row i -> scalar in [0,1]."""
    prims = compiled["primitives"]
    logic = compiled.get("logic", "AND")
    vals = []
    for p in prims:
        feat = p["feature"]
        kind = p.get("kind", "calendar")
        if kind == "calendar":
            unit = int(feats[feat][i])
            vals.append(1.0 if unit in p["rule"] else 0.0)
        else:
            if feat == "recent_autocorr":
                v = float(feats[feat][i])                    # already [0,1]
            else:
                v = float(_sigmoid(feats[feat][i]))          # z-score -> [0,1]
            if p.get("qualifier") in ("high", "above_typical"):
                vals.append(v)
            else:                                            # low / below_typical
                vals.append(1.0 - v)
    if logic == "OR":
        return float(max(vals)) if vals else 0.0
    if len(vals) > 1:
        return float(min(vals))                              # binary AND = min
    return float(vals[0]) if vals else 0.0


def residualize(x):
    """
    轻量 residual shape：
    每个窗口去线性趋势，再标准化。
    用于 Pearson/余弦 shape matching。
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    tc = t - t.mean()
    denom = np.sum(tc * tc)

    xm = x.mean(axis=-1, keepdims=True)
    slope = np.sum((x - xm) * tc[None, :], axis=-1, keepdims=True) / denom
    trend = xm + slope * tc[None, :]
    r = x - trend

    r = r - r.mean(axis=-1, keepdims=True)
    sd = r.std(axis=-1, keepdims=True)
    r = r / np.maximum(sd, 1e-6)
    return r


def build_windows(series, starts, context, horizon):
    starts = np.asarray(starts, dtype=np.int64)
    X = np.stack([series[s:s + context] for s in starts]).astype("float32")
    Y = np.stack([series[s + context:s + context + horizon] for s in starts]).astype("float32")
    return X, Y
