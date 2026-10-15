# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import torch
import torch.nn as nn

# Source: experiments/run_expert_attention_retrieval_v2.py:75-78
def zscore_window(x, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

# Source: experiments/run_expert_attention_retrieval_v2.py:217-234
class QueryEncoder(nn.Module):
    """原始 512 点时间序列 -> d_model。无任何人工时间/状态特征。"""
    def __init__(self, context, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = zscore_window(x).unsqueeze(1)
        return self.norm(self.net(x).squeeze(-1))
