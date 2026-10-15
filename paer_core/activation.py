# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import numpy as np
from .model import _linear_slope_vec, _lag1_autocorr_vec, _sigmoid

# Source: experiments/run_etth1_prediction_validation.py:62-74
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

# Source: experiments/run_etth1_prediction_validation.py:77-98
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

# Source: experiments/run_etth1_prediction_validation.py:101-125
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

# Source: experiments/run_etth1_prediction_validation.py:128-137
def build_activation_matrix(experts, X, tss, stats):
    """[n, N_experts] activation matrix."""
    feats = compute_window_feats(X, tss, stats)
    n = len(X)
    M = np.zeros((n, len(experts)), dtype=np.float32)
    for j, e in enumerate(experts):
        compiled = e["compiled"]
        for i in range(n):
            M[i, j] = expert_activation(compiled, feats, i)
    return M
