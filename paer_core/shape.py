# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import numpy as np

# Source: experiments/run_solar_same_val_baselines.py:56-76
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
