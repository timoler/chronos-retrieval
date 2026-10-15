# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import os
import random
import numpy as np
import torch
from . import llm_transport as av2

# Source: experiments/run_reward_feedback_critic_v2_poolaware_pilot.py:65-65
MIN_COVERAGE = 0.01

# Source: experiments/run_reward_feedback_critic_v2_poolaware_pilot.py:68-68
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3

# Source: experiments/run_reward_feedback_critic_v2_poolaware_pilot.py:69-69
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

# Source: experiments/run_reward_feedback_critic_v2_poolaware_pilot.py:79-89
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# Source: experiments/run_reward_feedback_critic_v2_poolaware_pilot.py:119-156
def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(seed)
    mode = "full" if use_shape else "expert"
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(n)
        for p in range(0, n, BATCH_SIZE):
            ids = order[p:p + BATCH_SIZE]
            if len(ids) == 0:
                continue
            cidx = rng.choice(len(X_bank), size=min(CANDIDATES, len(X_bank)), replace=False)
            x_q = torch.tensor(X_fit[ids], dtype=torch.float32, device=device)
            x_c = torch.tensor(X_bank[cidx], dtype=torch.float32, device=device)
            act_q = torch.tensor(fit_act[ids], dtype=torch.float32, device=device)
            act_c = bank_act_t[cidx]
            shape = torch.tensor(X_fit_res[ids] @ X_bank_res[cidx].T / L,
                                 dtype=torch.float32, device=device)
            z_q = model.encode(x_q, act_q)
            z_c = model.encode(x_c, act_c)
            delta = model.delta(z_q, z_c)
            s = model.combine(delta, shape, mode)
            w = torch.softmax(s / 1.0, dim=1)
            q_mu = X_fit[ids].mean(axis=1)
            c_mu = X_bank[cidx].mean(axis=1)
            transferred = Y_bank[cidx][None, :, :] + q_mu[:, None, None] - c_mu[None, :, None]
            pred = torch.sum(w.unsqueeze(-1) * torch.tensor(transferred, dtype=torch.float32, device=device), dim=1)
            y = torch.tensor(Y_fit[ids], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

# Source: experiments/run_reward_feedback_critic_v2_poolaware_pilot.py:239-251
def v2_reward_rule(st, coverage):
    if coverage is not None and coverage < MIN_COVERAGE:
        return 0, "UNSUPPORTED"
    pfr = st["positive_fold_ratio"]
    mean_pr = st["mean_pool_reward"]
    median_pr = st["median_pool_reward"]
    if pfr >= 0.75 and median_pr > 0:
        return 2, "KEEP"
    if pfr >= 0.50 and mean_pr > 0:
        return 1, "KEEP"
    if pfr <= 0.25 and median_pr < 0:
        return -2, "REJECT"
    return -1, "REVISE"
