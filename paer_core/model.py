# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from . import encoder as attn

# Source: experiments/run_conditional_expert_mlp_v4.py:48-49
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# Source: experiments/run_conditional_expert_mlp_v4.py:52-59
def _linear_slope_vec(X):
    """每行 [L] 的线性斜率（向量化）。"""
    L = X.shape[1]
    t = np.linspace(-1.0, 1.0, L, dtype=np.float32)
    tc = t - t.mean()
    denom = np.sum(tc * tc)
    xm = X.mean(axis=1, keepdims=True)
    return np.sum((X - xm) * tc[None, :], axis=1) / denom

# Source: experiments/run_conditional_expert_mlp_v4.py:62-69
def _lag1_autocorr_vec(X):
    x1 = X[:, 1:]
    x0 = X[:, :-1]
    m1 = x1.mean(axis=1, keepdims=True)
    m0 = x0.mean(axis=1, keepdims=True)
    num = ((x1 - m1) * (x0 - m0)).sum(axis=1)
    den = np.sqrt(((x1 - m1) ** 2).sum(axis=1) * ((x0 - m0) ** 2).sum(axis=1))
    return num / (den + 1e-9)

# Source: experiments/run_conditional_expert_mlp_v4.py:124-180
class ConditionalExpertMLPRetriever(nn.Module):
    """z = cond([CNN(x), ExpertVector(x)])；ExpertVector(x) = concat(a_i(x)·e_i)。无 attention。"""

    def __init__(self, context, expert_dim, n_experts, d_enc=128, d_exp=128, d_z=128,
                 hidden=32, beta_init=0.2):
        super().__init__()
        self.ts_encoder = attn.QueryEncoder(context, d_enc)          # [B,512]->[B,128]
        self.n_experts = n_experts
        self.expert_proj = nn.Linear(n_experts * expert_dim, d_exp)  # concat 后降维
        self.cond = nn.Sequential(nn.Linear(d_enc + d_exp, d_z), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_z, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.logit_gamma = nn.Parameter(torch.tensor(-2.0))
        beta_init = float(np.clip(beta_init, 1e-4, 1 - 1e-4))
        self.logit_beta = nn.Parameter(
            torch.tensor(np.log(beta_init / (1 - beta_init)), dtype=torch.float32))
        self.register_buffer("expert_emb_matrix", torch.zeros(n_experts, expert_dim))

    def set_expert_embeddings(self, emb_matrix):
        """emb_matrix: [N, E] 冻结的专家文本 embedding。"""
        self.expert_emb_matrix.copy_(torch.from_numpy(emb_matrix.astype(np.float32)))

    def expert_vector(self, x, act):
        """act: [B, N]。返回 ExpertVector [B, N*E]（投影前，供诊断）。"""
        weighted = act.unsqueeze(-1) * self.expert_emb_matrix.unsqueeze(0)  # [B, N, E]
        return weighted.reshape(len(x), -1)                                 # [B, N*E]

    def encode(self, x, act):
        q = self.ts_encoder(x)                                            # [B, d_enc]
        ev = self.expert_vector(x, act)                                   # [B, N*E]
        ev = self.expert_proj(ev)                                         # [B, d_exp]
        return self.cond(torch.cat([q, ev], dim=-1))                      # [B, d_z]

    def delta(self, z_q, z_c):
        B = z_q.shape[0]
        C = z_c.shape[0]
        qq = z_q[:, None, :].expand(B, C, -1)
        cc = z_c[None, :, :].expand(B, C, -1)
        return self.mlp(torch.cat([qq, cc, torch.abs(qq - cc)], dim=-1)).squeeze(-1)

    def score(self, shape, x_q, x_c, act_q, act_c):
        z_q = self.encode(x_q, act_q)
        z_c = self.encode(x_c, act_c)
        delta = self.delta(z_q, z_c)
        return torch.sigmoid(self.logit_gamma) * delta + torch.sigmoid(self.logit_beta) * shape

    def combine(self, delta, shape, mode):
        gamma = torch.sigmoid(self.logit_gamma)
        beta = torch.sigmoid(self.logit_beta)
        if mode == "shape":
            return shape
        if mode == "expert":
            return gamma * delta
        return gamma * delta + beta * shape
