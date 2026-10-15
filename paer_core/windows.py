# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import numpy as np

# Source: experiments/run_expert_embedding_mlp.py:117-121
def build_windows(series, starts, context, horizon):
    starts = np.asarray(starts, dtype=np.int64)
    X = np.stack([series[s:s + context] for s in starts]).astype("float32")
    Y = np.stack([series[s + context:s + context + horizon] for s in starts]).astype("float32")
    return X, Y
