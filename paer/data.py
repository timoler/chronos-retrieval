"""Causal bank, fit and query windows; condition evaluation at the prediction origin."""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .features import (
    FEATURE_REGISTRY, _normalize_qualifier, build_windows, compute_training_stats,
    compute_window_feats, expert_activation, residualize,
)


def knowledge_folds(train_end, context, horizon):
    selection_end = int(train_end * 0.8)
    warmup = max(int(selection_end / 5), math.ceil((context + horizon + 64) / 0.8))
    edges = np.linspace(warmup, selection_end, 5).astype(int)
    if np.any(np.diff(edges) < horizon):
        raise ValueError("TRAIN is too short for four folds with complete target horizons.")
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def stage_indices(cutoff, end, context, horizon, outer_starts=None):
    boundary = int(cutoff * 0.8)
    bank = np.arange(0, boundary - context - horizon + 1)
    fit = np.arange(boundary - context, cutoff - context - horizon + 1)
    query = (np.arange(cutoff - context, end - context - horizon + 1)
             if outer_starts is None else np.asarray(outer_starts, dtype=np.int64))
    if min(map(len, (bank, fit, query))) < 5:
        raise ValueError("Each bank, fit and query set must contain at least five windows.")
    if (min(bank[0], fit[0], query.min()) < 0 or query.max() + context + horizon > end
            or bank[-1] + horizon > fit[0] or fit[-1] + horizon > query.min()):
        raise ValueError("Window bounds violate the historical availability constraints.")
    return {"bank": bank, "fit": fit, "query": query}


def slot(expert):
    eid = expert["id"]
    if eid not in {f"E{i}" for i in range(1, 9)}:
        raise ValueError(f"Invalid expert ID: {eid}")
    return int(eid[1:]) - 1


def compile_pool(pool, calendar=True):
    result = deepcopy(pool)
    if len({slot(e) for e in pool}) != len(pool):
        raise ValueError("Duplicate expert ID in pool")
    for expert in result:
        if not isinstance(expert.get("mechanism"), str) or not expert["mechanism"].strip():
            raise ValueError(f"Missing mechanism for {expert['id']}")
        spec = expert["condition_spec"]
        if spec.get("negate") or spec["logic"] not in ("AND", "OR") or not spec["conditions"]:
            raise ValueError(f"Unsupported condition for {expert['id']}")
        primitives = []
        for cond in spec["conditions"]:
            feature = cond["feature"]
            registry = FEATURE_REGISTRY[feature]
            qualifier = _normalize_qualifier(cond["qualifier"])
            if registry["kind"] == "calendar":
                if not calendar or qualifier not in registry["categories"]:
                    raise ValueError(f"Calendar condition unavailable: {feature}/{qualifier}")
                rule = registry["categories"][qualifier]
                op = "in"
            else:
                if qualifier not in ("high", "low"):
                    raise ValueError(f"Unsupported qualifier: {qualifier}")
                rule, op = None, "soft_sigmoid"
            primitives.append(dict(feature=feature, qualifier=qualifier, kind=registry["kind"], rule=rule, op=op))
        expert["claim"] = expert["mechanism"]
        expert["observable_features"] = expert.get("observables", [])
        expert["compiled"] = dict(status="COMPILED", logic=spec["logic"], negate=False,
                                  primitives=primitives, features_compiled=[p["feature"] for p in primitives],
                                  uncompilable=[])
    return result


@dataclass
class Series:
    values: np.ndarray
    dates: pd.DatetimeIndex
    context: int
    horizon: int
    train_end: int
    calendar: bool = True

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        self.dates = pd.DatetimeIndex(self.dates)
        if (self.values.ndim != 1 or not np.isfinite(self.values).all() or self.context < 2
                or self.horizon <= 0 or len(self.dates) != len(self.values)
                or self.dates.hasnans or not 0 < self.train_end <= len(self.values)):
            raise ValueError("Invalid series, dates, TRAIN boundary, context or horizon")

    def stage(self, cutoff, end, outer_starts=None):
        if not 0 < cutoff <= end <= len(self.values):
            raise ValueError("Invalid stage boundaries")
        starts = stage_indices(cutoff, end, self.context, self.horizon, outer_starts)
        x, y, residual, feats = {}, {}, {}, {}
        for side, indices in starts.items():
            x[side], y[side] = build_windows(self.values, indices, self.context, self.horizon)
            residual[side] = residualize(x[side])
        stats = compute_training_stats(x["bank"])
        for side, indices in starts.items():
            feats[side] = compute_window_feats(x[side], self.dates[indices + self.context], stats)
        return {**starts, "X": x, "Y": y, "R": residual, "feats": feats}


def activations(stage, pool, drop=None):
    out = {}
    for side in ("bank", "fit", "query"):
        n = len(stage[side])
        values = np.zeros((n, 8), dtype=np.float32)
        for expert in pool:
            if expert["id"] != drop:
                values[:, slot(expert)] = [expert_activation(expert["compiled"], stage["feats"][side], i)
                                           for i in range(n)]
        out[side] = values
    return out
