#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 Critic v6 — deterministic reproducibility test (D arm only).

Re-runs ONLY the D arm (+CRITIC EXPERT v6) of ``run_etth1_module_ablation_v6``
under STRICT deterministic settings, and repeats seeds 7 / 42 / 123 twice each to
test whether the previously-observed same-seed MSE divergence (seed 7: 90.05 vs
38.16) is attributable to CUDA/cuDNN non-determinism.

The D-arm data / network / optimizer / LR / epochs / batch / Shape / Expert
activation / fixed slots / preprocessing are copied VERBATIM.  The ONLY addition is
a strict ``seed_everything`` (PYTHONHASHSEED + random/numpy/torch seeds +
cudnn.deterministic=True + cudnn.benchmark=False + use_deterministic_algorithms(True)).

The original D arm uses manual numpy batch iteration (``rng = np.random.default_rng(12345)``),
NOT a DataLoader, so there is no DataLoader shuffle / worker_init_fn to fix here.

Does NOT modify the Critic / knowledge / network / optimizer / LR / epochs /
clipping / normalization.  Does NOT re-select seeds, does NOT early-stop, does NOT
read TEST.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import run_expert_attention_retrieval_v2 as attn                    # (unused; kept for parity)
from run_expert_embedding_mlp import build_windows, mse, mae        # noqa: E402
from run_solar_same_val_baselines import residualize                # noqa: E402
from run_conditional_expert_mlp_v4 import (                         # noqa: E402
    ConditionalExpertMLPRetriever,
    retrieve_conditional,
)
from run_etth1_prediction_validation import (                       # noqa: E402
    compute_training_stats,
    compute_window_feats,
    expert_activation,
    build_activation_matrix,
)
from sentence_transformers import SentenceTransformer               # noqa: E402

CRITIC_POOL = Path("results/etth1_critic_v6_closed_loop/final_accepted_knowledge.json")
OUT_DIR = Path("results/etth1_v6_deterministic_reproducibility")

CONTEXT, HORIZON, TOPK = 512, 96, 5
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

SEEDS = [7, 42, 123]
REPEATS = [1, 2]


# --------------------------------------------------------------------------- #
# strict deterministic seeding
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _sorted_experts(pool):
    if isinstance(pool, dict):
        items = list(pool.values())
    else:
        items = list(pool)
    return sorted(items, key=lambda e: int(str(e.get("expert_id", "0")).lstrip("E") or 0))


def embed_claims(claims, device):
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    emb = enc.encode(claims, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


def _checksum(model) -> str:
    h = hashlib.sha256()
    for name in sorted(model.state_dict().keys()):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        t = model.state_dict()[name].detach().cpu().numpy()
        h.update(np.ascontiguousarray(t.astype(np.float32)).tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate_val(model, X_val, X_bank, Y_bank, Y_val, X_val_res, X_bank_res,
                 val_act, bank_act, device):
    pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                val_act, bank_act, "full", TOPK, device)
    return mse(Y_val, pred), mae(Y_val, pred)


# --------------------------------------------------------------------------- #
# one full training run (fresh model / optimizer; frozen knowledge reloaded)
# --------------------------------------------------------------------------- #
def train_one_run(seed, emb, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                  bank_act, fit_act, X_val, Y_val, X_val_res, val_act, device):
    seed_everything(seed)

    model = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=emb.shape[1], n_experts=emb.shape[0],
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    model.set_expert_embeddings(emb)
    initial_checksum = _checksum(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)                 # fixed batch/candidate order
    mode = "full"

    epoch_train_loss = []
    epoch_val_mse = []
    epoch_val_mae = []
    unsupported_deterministic_op = None

    try:
        for ep in range(1, EPOCHS + 1):
            model.train()
            order = rng.permutation(n)
            losses = []
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
                loss = F.mse_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(loss.item())

            val_mse, val_mae = evaluate_val(model, X_val, X_bank, Y_bank, Y_val,
                                            X_val_res, X_bank_res, val_act, bank_act, device)
            epoch_train_loss.append(float(np.mean(losses)))
            epoch_val_mse.append(val_mse)
            epoch_val_mae.append(val_mae)
    except RuntimeError as e:
        unsupported_deterministic_op = str(e)
        return {"seed": seed, "unsupported_deterministic_op": unsupported_deterministic_op}

    final_checksum = _checksum(model)
    best_ep = int(np.argmin(epoch_val_mse)) + 1
    return {
        "seed": seed,
        "initial_checksum": initial_checksum,
        "final_checksum": final_checksum,
        "final_val_mse": float(epoch_val_mse[-1]),
        "final_val_mae": float(epoch_val_mae[-1]),
        "best_val_mse": float(min(epoch_val_mse)),
        "best_val_epoch": best_ep,
        "final_train_loss": float(epoch_train_loss[-1]),
        "epoch_train_loss": [float(x) for x in epoch_train_loss],
        "epoch_val_mse": [float(x) for x in epoch_val_mse],
        "epoch_val_mae": [float(x) for x in epoch_val_mae],
        "unsupported_deterministic_op": None,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("cudnn.deterministic will be forced True; benchmark False; use_deterministic_algorithms True")

    critic_experts = _sorted_experts(json.loads(CRITIC_POOL.read_text(encoding="utf-8")))
    n_slots = len(critic_experts)
    print(f"critic experts={n_slots}")

    # ---- data (identical to run_etth1_module_ablation_v6) ----
    ds = Path("results/multidataset/ETTh1")
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)      # TEST never read

    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    X_bank, Y_bank = build_windows(series, bank_starts, CONTEXT, HORIZON)
    X_fit, Y_fit = build_windows(series, fit_starts, CONTEXT, HORIZON)
    X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)
    print(f"bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)} (TEST not read)")

    bank_ts = [dates[s + CONTEXT] for s in bank_starts]
    fit_ts = [dates[s + CONTEXT] for s in fit_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]

    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)
    bank_act = build_activation_matrix(critic_experts, X_bank, bank_ts, stats)
    fit_act = build_activation_matrix(critic_experts, X_fit, fit_ts, stats)
    val_act = build_activation_matrix(critic_experts, X_val, val_ts, stats)

    # frozen embeddings (shared, deterministic)
    emb = embed_claims([e["claim"] for e in critic_experts], device)

    # ---- run each seed twice ----
    results = {}
    for seed in SEEDS:
        results[seed] = {}
        for rep in REPEATS:
            print(f"\n{'='*64}\nSEED {seed} run{rep}\n{'='*64}")
            r = train_one_run(seed, emb, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                              bank_act, fit_act, X_val, Y_val, X_val_res, val_act, device)
            results[seed][rep] = r
            if r.get("unsupported_deterministic_op"):
                print(f"    UNSUPPORTED_DETERMINISTIC_OP: {r['unsupported_deterministic_op'][:200]}")
                break
            print(f"    final_val_mse={r['final_val_mse']:.6f} best={r['best_val_mse']:.6f}@ep{r['best_val_epoch']} "
                  f"final_train_loss={r['final_train_loss']:.6f}")

    # ---- compare per seed ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw_results.json").write_text(
        json.dumps({str(s): {str(r): v for r, v in results[s].items()} for s in SEEDS},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    comparison = {}
    for seed in SEEDS:
        r1 = results[seed].get(1, {})
        r2 = results[seed].get(2, {})
        if r1.get("unsupported_deterministic_op") or r2.get("unsupported_deterministic_op"):
            comparison[seed] = {"unsupported_deterministic_op": True}
            continue
        m1, m2 = r1["final_val_mse"], r2["final_val_mse"]
        abs_diff = abs(m1 - m2)
        rel_diff = abs_diff / ((m1 + m2) / 2.0) * 100.0 if (m1 + m2) > 0 else 0.0
        max_tl = max(abs(a - b) for a, b in zip(r1["epoch_train_loss"], r2["epoch_train_loss"]))
        max_vm = max(abs(a - b) for a, b in zip(r1["epoch_val_mse"], r2["epoch_val_mse"]))
        init_equal = r1["initial_checksum"] == r2["initial_checksum"]
        final_equal = r1["final_checksum"] == r2["final_checksum"]
        reproducible = bool(init_equal and final_equal and abs_diff < 1e-8)
        numerically_reproducible = bool(abs_diff < 1e-4 and not reproducible)
        comparison[seed] = {
            "run1_final_mse": m1, "run2_final_mse": m2,
            "abs_diff": abs_diff, "rel_diff_pct": rel_diff,
            "run1_best_mse": r1["best_val_mse"], "run2_best_mse": r2["best_val_mse"],
            "best_mse_abs_diff": abs(r1["best_val_mse"] - r2["best_val_mse"]),
            "best_epoch_run1": r1["best_val_epoch"], "best_epoch_run2": r2["best_val_epoch"],
            "final_train_loss_diff": abs(r1["final_train_loss"] - r2["final_train_loss"]),
            "initial_checksum_equal": init_equal,
            "final_checksum_equal": final_equal,
            "max_epoch_train_loss_abs_diff": max_tl,
            "max_epoch_val_mse_abs_diff": max_vm,
            "reproducible": reproducible,
            "numerically_reproducible": numerically_reproducible,
        }

    (OUT_DIR / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report ----
    L = []
    L.append("【结果】")
    L.append("")
    L.append("## 1. three seeds x two runs (final MSE/MAE)")
    L.append("")
    L.append("| seed | run1 MSE | run1 MAE | run2 MSE | run2 MAE |")
    L.append("|---|---|---|---|---|")
    for seed in SEEDS:
        if comparison[seed].get("unsupported_deterministic_op"):
            L.append(f"| {seed} | UNSUPPORTED_DETERMINISTIC_OP | | | |")
            continue
        r1, r2 = results[seed][1], results[seed][2]
        L.append(f"| {seed} | {r1['final_val_mse']:.6f} | {r1['final_val_mae']:.4f} | "
                 f"{r2['final_val_mse']:.6f} | {r2['final_val_mae']:.4f} |")
    L.append("")
    L.append("## 2. per-seed MSE difference")
    L.append("")
    L.append("| seed | abs diff | rel diff % | best_mse diff |")
    L.append("|---|---|---|---|")
    for seed in SEEDS:
        c = comparison[seed]
        if c.get("unsupported_deterministic_op"):
            L.append(f"| {seed} | - | - | - |")
            continue
        L.append(f"| {seed} | {c['abs_diff']:.3e} | {c['rel_diff_pct']:.4f}% | {c['best_mse_abs_diff']:.3e} |")
    L.append("")
    L.append("## 3. initial / final checksum equal")
    L.append("")
    L.append("| seed | initial equal | final equal |")
    L.append("|---|---|---|")
    for seed in SEEDS:
        c = comparison[seed]
        if c.get("unsupported_deterministic_op"):
            L.append(f"| {seed} | - | - |")
            continue
        L.append(f"| {seed} | {c['initial_checksum_equal']} | {c['final_checksum_equal']} |")
    L.append("")
    L.append("## 4. max per-epoch curve difference")
    L.append("")
    L.append("| seed | max train-loss diff | max val-mse diff |")
    L.append("|---|---|---|")
    for seed in SEEDS:
        c = comparison[seed]
        if c.get("unsupported_deterministic_op"):
            L.append(f"| {seed} | - | - |")
            continue
        L.append(f"| {seed} | {c['max_epoch_train_loss_abs_diff']:.3e} | {c['max_epoch_val_mse_abs_diff']:.3e} |")
    L.append("")
    L.append("## 5/6/7. per-seed reproducibility verdict")
    L.append("")
    L.append("| seed | reproducible | numerically_reproducible |")
    L.append("|---|---|---|")
    for seed in SEEDS:
        c = comparison[seed]
        if c.get("unsupported_deterministic_op"):
            L.append(f"| {seed} | unsupported op | |")
            continue
        L.append(f"| {seed} | {c['reproducible']} | {c['numerically_reproducible']} |")
    L.append("")

    L.append("【结论】")
    L.append("")
    all_supported = all(not comparison[s].get("unsupported_deterministic_op") for s in SEEDS)
    op_list = "; ".join(str(s) for s in SEEDS if comparison[s].get("unsupported_deterministic_op")) or "无"
    L.append(f"A. deterministic 设置是否成功？ {'是' if all_supported else '否'}"
             f"（unsupported_deterministic_op = {op_list}）")
    n_rep = sum(1 for s in SEEDS if comparison[s].get("reproducible"))
    L.append(f"B. 同一个 seed 是否可以稳定复现？ {'是' if n_rep == len(SEEDS) else '否'}"
             f"（{n_rep}/{len(SEEDS)} 个 seed bitwise 可复现）")
    s7 = comparison[7]
    L.append(f"C. 之前 seed7 90 vs 38 是否可合理归因于非确定训练？ "
             f"{'是' if s7.get('reproducible') or s7.get('numerically_reproducible') else '否'}"
             f"（deterministic 下 seed7 run1={s7.get('run1_final_mse', float('nan')):.3f} "
             f"run2={s7.get('run2_final_mse', float('nan')):.3f}）")
    L.append(f"D. 后续多 seed 实验是否应全部采用该 deterministic 设置？ "
             f"{'是' if n_rep == len(SEEDS) else '否'}")
    L.append("")
    L.append("critic_modified = false")
    L.append("network_modified = false")
    L.append("hyperparameters_modified = false")
    L.append("test_used = false")
    L.append("deterministic_enabled = true")
    report = "\n".join(L)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
