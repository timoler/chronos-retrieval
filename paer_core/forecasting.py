# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from . import experts as v1
from . import training as v2p
from .model import ConditionalExpertMLPRetriever
from .windows import build_windows
from .shape import residualize
from .activation import compute_training_stats

# Source: experiments/external_baselines/run_ours_fusion_v2.py:57-57
D_ENC, D_EXP, D_Z, HIDDEN = v2p.D_ENC, v2p.D_EXP, v2p.D_Z, v2p.HIDDEN

# Source: experiments/external_baselines/run_ours_fusion_v2.py:60-60
BATCH = 256

# Source: experiments/external_baselines/run_ours_fusion_v2.py:117-121
def run_chronos(pipeline, contexts, horizon):
    """Chronos-2 零样本 mean 点预测（确定性，seed 无关）。"""
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=horizon, batch_size=BATCH)
    return np.stack([m.squeeze(0).numpy() for m in means]).astype(np.float32)

# Source: experiments/external_baselines/run_ours_fusion_v2.py:149-154
def _init_state(seed, context, expert_dim, n_slots, device):
    v2p.seed_everything(seed)
    m = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

# Input-path adapter only: all seven initial Raw8 files are mandatory.
# The unified runtime supplies the explicit input root; there is no historical fallback.
FROZEN_ROOT = None

def load_raw_pool(name):
    if FROZEN_ROOT is None:
        raise RuntimeError('Initial Raw8 root must be configured by unified/runtime.py')
    return json.loads((Path(FROZEN_ROOT) / name / 'raw_experts.json').read_text(encoding='utf-8'))
