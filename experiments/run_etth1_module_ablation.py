#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 — frozen module ablation for the Conditional Expert-MLP retrieval.

Goal: decompose the full method into its modules and measure each module's
independent contribution, without re-running the Critic, without regenerating
experts, without modifying the network structure, and without reading TEST.

Modules under test (score(q,c) = gamma * delta + beta * shape):
  - delta  = pairwise MLP on z = cond([CNN(x), expert_proj(expert_vector(x))])
  - shape  = frozen Residual Pearson similarity (Top-K=5)
  - expert = conditional expert activation a_i(q) * text embedding e_i

Arms (same architecture / params / init / optimizer / epochs / batch order):
  A. BASE           CNN-only retrieval, no Shape, no Expert
  B. +SHAPE         CNN retrieval + frozen Shape (Residual Pearson)
  C. +RAW EXPERT    B + raw expert knowledge (expert_round_1, NO Critic filter)
  D. +CRITIC EXPERT B + Critic v5 final ACCEPT knowledge  (full method)

Auxiliary mechanism control (full Critic-Expert structure, same params/init):
  ZERO      expert activation all zeroed
  SHUFFLED  expert activation rows shuffled (query<->activation decoupled)
  REAL      normal Critic Expert (== arm D)

Fairness: one shared random init per seed is copied to every arm; a FIXED number
of expert slots (= max(n_raw, n_critic)) with a zero mask keeps the parameter
count identical even if Raw/Critic expert counts differed.

Data: results/multidataset/ETTh1 (TRAIN 80/20 bank/fit, VAL 2877, TEST never read).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_expert_attention_retrieval_v2 as attn                          # seed_all
from run_expert_embedding_mlp import build_windows, mse, mae             # noqa: E402
from run_solar_same_val_baselines import residualize                     # noqa: E402
from run_conditional_expert_mlp_v4 import (                              # noqa: E402
    ConditionalExpertMLPRetriever,
    retrieve_conditional,
)
from run_etth1_prediction_validation import (                            # noqa: E402
    compute_training_stats,
    compute_window_feats,
    expert_activation,
    build_activation_matrix,
)
from sentence_transformers import SentenceTransformer                    # noqa: E402

RAW_POOL = Path("results/etth1_critic_v5_closed_loop/expert_round_1.json")
CRITIC_POOL = Path("results/etth1_critic_v5_closed_loop/final_accepted_knowledge.json")
OUT_DIR = Path("results/etth1_module_ablation")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
SHUFFLE_SEED = 777
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3


# --------------------------------------------------------------------------- #
# frozen training loop (mirrors train_conditional, but score mode is explicit)
# --------------------------------------------------------------------------- #
def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device, epochs=EPOCHS,
              batch_size=BATCH_SIZE, candidates=CANDIDATES, lr=LR, temp=1.0):
    """Faithful copy of train_conditional; `use_shape` selects the score mode.

    mode = "full"  -> gamma * delta + beta * shape   (Shape ON)
    mode = "expert" -> gamma * delta                 (Shape OFF)
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)                 # fixed batch/candidate order
    mode = "full" if use_shape else "expert"

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for p in range(0, n, batch_size):
            ids = order[p:p + batch_size]
            if len(ids) == 0:
                continue
            cidx = rng.choice(len(X_bank), size=min(candidates, len(X_bank)), replace=False)

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

            w = torch.softmax(s / temp, dim=1)
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

        gamma = float(torch.sigmoid(model.logit_gamma).detach().cpu())
        beta = float(torch.sigmoid(model.logit_beta).detach().cpu())
        print(f"    epoch {ep:02d}/{epochs} loss={np.mean(losses):.6f} gamma={gamma:.4f} beta={beta:.4f}")
    return {"gamma": gamma, "beta": beta}


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# expert loading / slot padding (fixed slots + mask)
# --------------------------------------------------------------------------- #
def _sorted_experts(pool):
    """Return a list of expert dicts ordered by numeric expert id (E1,E2,...)."""
    if isinstance(pool, dict):
        items = list(pool.values())
    else:
        items = list(pool)
    return sorted(items, key=lambda e: int(str(e.get("expert_id", "0")).lstrip("E") or 0))


def pad_to_slots(emb, act_dict, n_slots):
    """Zero-pad an expert set to `n_slots` so network params stay constant.

    Returns (padded_emb, padded_act_dict, mask) where mask[j]=1 for used slots.
    """
    n = emb.shape[0]
    mask = np.zeros(n_slots, dtype=np.float32)
    mask[:n] = 1.0
    if n == n_slots:
        return emb, act_dict, mask
    pad_emb = np.zeros((n_slots - n, emb.shape[1]), dtype=np.float32)
    emb = np.concatenate([emb, pad_emb], axis=0)
    padded_act = {}
    for k, a in act_dict.items():
        pad_act = np.zeros((a.shape[0], n_slots - n), dtype=np.float32)
        padded_act[k] = np.concatenate([a, pad_act], axis=1)
    return emb, padded_act, mask


def embed_claims(claims, device):
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    emb = enc.encode(claims, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    epochs = args.epochs
    seeds = list(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- 1. frozen expert pools (NO Critic re-run, NO regeneration) ----
    raw_experts = _sorted_experts(json.loads(RAW_POOL.read_text(encoding="utf-8")))
    critic_experts = _sorted_experts(json.loads(CRITIC_POOL.read_text(encoding="utf-8")))
    n_raw, n_critic = len(raw_experts), len(critic_experts)
    n_slots = max(n_raw, n_critic)
    print(f"raw experts={n_raw}, critic experts={n_critic}, fixed slots={n_slots}")
    for e in raw_experts:
        comp = e.get("compiled", {})
        n_prims = len(comp.get("primitives", []))
        print(f"  [raw] {e['expert_id']}: {e['name']} | prims={n_prims} | status={comp.get('status')}")
    for e in critic_experts:
        print(f"  [critic] {e['expert_id']}: {e['name']}")

    # ---- 2. ETTh1 data (TRAIN + VAL; TEST never read) ----
    ds = Path("results/multidataset/ETTh1")
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

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

    # ---- 3. TRAIN stats + activation matrices (raw / critic) ----
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)

    def act_mats(experts):
        bank = build_activation_matrix(experts, X_bank, bank_ts, stats)
        fit = build_activation_matrix(experts, X_fit, fit_ts, stats)
        val = build_activation_matrix(experts, X_val, val_ts, stats)
        return {"bank": bank, "fit": fit, "val": val}

    raw_act = act_mats(raw_experts)
    critic_act = act_mats(critic_experts)

    # ---- 4. expert text embeddings (frozen MiniLM) ----
    emb_raw = embed_claims([e["claim"] for e in raw_experts], device)
    emb_critic = embed_claims([e["claim"] for e in critic_experts], device)
    expert_dim = emb_raw.shape[1]
    print(f"expert embeddings raw={emb_raw.shape} critic={emb_critic.shape} dim={expert_dim}")

    # ---- 5. fixed slots + mask (identical param count across Raw/Critic) ----
    emb_raw, raw_act, raw_mask = pad_to_slots(emb_raw, raw_act, n_slots)
    emb_critic, critic_act, critic_mask = pad_to_slots(emb_critic, critic_act, n_slots)

    # zero / shuffled controls (full Critic-Expert structure, same params/init)
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    def _zero(a):
        return np.zeros_like(a)
    zero_act = {k: _zero(critic_act[k]) for k in critic_act}

    def _perm(a, seed):
        rng = np.random.default_rng(seed)
        return a[rng.permutation(len(a))]
    shuffled_act = {k: _perm(critic_act[k], SHUFFLE_SEED) for k in critic_act}

    # ---- 6. per-seed arms + aux control ----
    # config: name -> (emb, act, use_shape)
    configs = {
        "base":          (emb_zero,    zero_act,      False),   # A
        "shape":         (emb_zero,    zero_act,      True),    # B
        "raw_expert":    (emb_raw,     raw_act,       True),    # C
        "critic_expert": (emb_critic,  critic_act,    True),    # D
        "zero":          (emb_critic,  zero_act,      True),    # aux
        "shuffled":      (emb_critic,  shuffled_act,  True),    # aux
    }

    per_seed = {name: [] for name in configs}
    param_count = None

    for seed in seeds:
        print(f"\n{'='*72}\nSEED {seed}\n{'='*72}")
        attn.seed_all(seed)
        base = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=128, d_exp=128, d_z=128, hidden=32,
        ).to(device)
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        param_count = n_params(base)

        for name, (emb, act, use_shape) in configs.items():
            print(f"\n[{name}] use_shape={use_shape}")
            model = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                d_enc=128, d_exp=128, d_z=128, hidden=32,
            ).to(device)
            model.load_state_dict(init_state)
            model.set_expert_embeddings(emb)
            train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      act["bank"], act["fit"], use_shape, device, epochs=epochs)
            mode = "full" if use_shape else "expert"
            pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                        act["val"], act["bank"], mode, TOPK, device)
            per_seed[name].append({"seed": seed, "mse": mse(Y_val, pred), "mae": mae(Y_val, pred)})
            print(f"    [{name}] MSE={per_seed[name][-1]['mse']:.2f} MAE={per_seed[name][-1]['mae']:.2f}")

    # REAL is by definition arm D (Critic Expert, real activation, shape ON) — alias
    # rather than retrain, so ZERO/SHUFFLED/REAL share the exact same REAL values.
    per_seed["real"] = [dict(v) for v in per_seed["critic_expert"]]

    # ---- 7. aggregate mean ± std ----
    def mean_std(vals):
        a = np.array([v["mse"] for v in vals], dtype=np.float64)
        b = np.array([v["mae"] for v in vals], dtype=np.float64)
        return (float(a.mean()), float(a.std()), float(b.mean()), float(b.std()))

    summary = {}
    for name, vals in per_seed.items():
        mm, ms, am, as_ = mean_std(vals)
        summary[name] = {"mse_mean": mm, "mse_std": ms, "mae_mean": am, "mae_std": as_,
                         "per_seed": vals}

    # adjacent gains (positive = lower MSE = improvement)
    b_mse = summary["base"]["mse_mean"]
    s_mse = summary["shape"]["mse_mean"]
    r_mse = summary["raw_expert"]["mse_mean"]
    c_mse = summary["critic_expert"]["mse_mean"]
    gains = {
        "shape_vs_base_pct":       (b_mse - s_mse) / b_mse * 100.0,
        "raw_vs_shape_pct":        (s_mse - r_mse) / s_mse * 100.0,
        "critic_vs_raw_pct":       (r_mse - c_mse) / r_mse * 100.0,
        "critic_vs_base_pct":      (b_mse - c_mse) / b_mse * 100.0,
        "real_vs_zero_pct":        (summary["zero"]["mse_mean"] - summary["real"]["mse_mean"]) / summary["zero"]["mse_mean"] * 100.0,
        "real_vs_shuffled_pct":    (summary["shuffled"]["mse_mean"] - summary["real"]["mse_mean"]) / summary["shuffled"]["mse_mean"] * 100.0,
    }

    # ---- 8. save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "dataset": "ETTh1",
        "protocol": "multidataset ETTh1 VAL, residualize Pearson, Top-K=5 mean",
        "modules": {"shape": "residual Pearson (frozen)", "expert": "conditional a_i(q)*e_i",
                    "cnn": "QueryEncoder + pairwise MLP"},
        "expert_sources": {
            "raw": {"file": str(RAW_POOL), "n": n_raw, "note": "expert_round_1, NO Critic filter"},
            "critic": {"file": str(CRITIC_POOL), "n": n_critic, "note": "final ACCEPT pool"},
            "n_slots": n_slots,
            "mask_raw": [int(x) for x in raw_mask],
            "mask_critic": [int(x) for x in critic_mask],
            "param_count": param_count,
        },
        "arms": {k: {kk: vv for kk, vv in summary[k].items() if kk != "per_seed"}
                 for k in ("base", "shape", "raw_expert", "critic_expert")},
        "gains": gains,
        "aux_control": {k: {kk: vv for kk, vv in summary[k].items() if kk != "per_seed"}
                        for k in ("zero", "shuffled", "real")},
        "seeds": seeds,
        "hyperparameters": {"context": CONTEXT, "horizon": HORIZON, "topk": TOPK,
                            "epochs": epochs, "batch_size": BATCH_SIZE,
                            "candidates": CANDIDATES, "lr": LR,
                            "d_enc": 128, "d_exp": 128, "d_z": 128, "hidden": 32},
        "flags": {"critic_rerun": False, "expert_regenerated": False,
                  "network_modified": False, "test_used": False},
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "per_seed_results.json").write_text(
        json.dumps({"seeds": seeds, "arms": per_seed,
                    "param_count": param_count, "n_slots": n_slots},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # summary.csv
    rows = [["arm", "mse_mean", "mse_std", "mae_mean", "mae_std"]]
    for name in ("base", "shape", "raw_expert", "critic_expert", "zero", "shuffled", "real"):
        s = summary[name]
        rows.append([name, f"{s['mse_mean']:.6f}", f"{s['mse_std']:.6f}",
                     f"{s['mae_mean']:.6f}", f"{s['mae_std']:.6f}"])
    rows.append(["gain_shape_vs_base_pct", f"{gains['shape_vs_base_pct']:.4f}", "", "", ""])
    rows.append(["gain_raw_vs_shape_pct", f"{gains['raw_vs_shape_pct']:.4f}", "", "", ""])
    rows.append(["gain_critic_vs_raw_pct", f"{gains['critic_vs_raw_pct']:.4f}", "", "", ""])
    rows.append(["gain_critic_vs_base_pct", f"{gains['critic_vs_base_pct']:.4f}", "", "", ""])
    (OUT_DIR / "summary.csv").write_text(
        "\n".join(",".join(r) for r in rows) + "\n", encoding="utf-8")

    audit = {
        "dataset": "ETTh1",
        "critic_rerun": False,
        "expert_regenerated": False,
        "network_modified": False,
        "test_used": False,
        "val_used": True,
        "raw_expert_source": str(RAW_POOL),
        "critic_expert_source": str(CRITIC_POOL),
        "raw_expert_decision": {e["expert_id"]: e.get("final_decision", "N/A") for e in raw_experts},
        "critic_expert_decision": {e["expert_id"]: "ACCEPT" for e in critic_experts},
        "n_slots": n_slots, "n_raw": n_raw, "n_critic": n_critic,
        "param_count": param_count,
        "shared_init_per_seed": True,
        "fixed_batch_order": "np.random.default_rng(12345)",
        "seeds": seeds,
        "shuffle_seed": SHUFFLE_SEED,
        "arms": {"base": "CNN only (no shape, no expert)",
                 "shape": "CNN + frozen shape",
                 "raw_expert": "CNN + shape + raw expert (no Critic filter)",
                 "critic_expert": "CNN + shape + Critic expert (full method)"},
        "aux_control": {"zero": "Critic structure, activation zeroed",
                        "shuffled": "Critic structure, activation shuffled",
                        "real": "Critic structure, real activation (== critic_expert)"},
        "note": "Raw expert E9 (UNCOMPILABLE) contributes activation=0 by construction; "
                "raw arm therefore includes an uncompilable slot, which is the point of "
                "the no-Critic-filter ablation.",
    }
    (OUT_DIR / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 9. report ----
    def fmt(name):
        s = summary[name]
        return f"{s['mse_mean']:.2f} ± {s['mse_std']:.2f}   /   {s['mae_mean']:.4f} ± {s['mae_std']:.4f}"

    print("\n" + "=" * 78)
    print("ETTH1 MODULE ABLATION (frozen, 3-seed mean ± std)")
    print("=" * 78)
    print(f"BASE           (A)  MSE/MAE = {fmt('base')}")
    print(f"+SHAPE         (B)  MSE/MAE = {fmt('shape')}")
    print(f"+RAW EXPERT    (C)  MSE/MAE = {fmt('raw_expert')}")
    print(f"+CRITIC EXPERT (D)  MSE/MAE = {fmt('critic_expert')}")
    print("-" * 78)
    print(f"Shape vs Base            = {gains['shape_vs_base_pct']:+.2f}%")
    print(f"Raw Expert vs Shape      = {gains['raw_vs_shape_pct']:+.2f}%")
    print(f"Critic Expert vs Raw     = {gains['critic_vs_raw_pct']:+.2f}%")
    print(f"Critic Expert vs Base    = {gains['critic_vs_base_pct']:+.2f}%")
    print("-" * 78)
    print(f"ZERO                     = {fmt('zero')}")
    print(f"SHUFFLED                 = {fmt('shuffled')}")
    print(f"REAL                     = {fmt('real')}")
    print(f"REAL vs ZERO             = {gains['real_vs_zero_pct']:+.2f}%")
    print(f"REAL vs SHUFFLED         = {gains['real_vs_shuffled_pct']:+.2f}%")
    print("=" * 78)
    print(f"Saved to {OUT_DIR}")
    print("critic_rerun=false, expert_regenerated=false, network_modified=false, test_used=false")


if __name__ == "__main__":
    main()
