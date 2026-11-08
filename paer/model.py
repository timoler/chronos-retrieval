"""Numerical encoder and eight-slot conditional retrieval scorer."""
import numpy as np
import torch
from torch import nn


def zscore_window(x, eps=1e-6):
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd


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


class ConditionalExpertMLPRetriever(nn.Module):
    """z = cond([CNN(x), ExpertVector(x)])；ExpertVector(x) = concat(a_i(x)·e_i)。无 attention。"""

    def __init__(self, context, expert_dim, n_experts, d_enc=128, d_exp=128, d_z=128,
                 hidden=32, beta_init=0.2):
        super().__init__()
        self.ts_encoder = QueryEncoder(context, d_enc)          # [B,512]->[B,128]
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
