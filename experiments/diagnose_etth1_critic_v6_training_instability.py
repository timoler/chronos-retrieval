#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose why the ETTh1 Critic v6 D-arm (Conditional Expert-MLP) is multi-seed unstable.

Read-only diagnosis.  Re-runs ONLY the D arm of ``run_etth1_module_ablation_v6``
with the SAME network / init / optimizer / LR / epochs / batch / Shape / Expert
activation / fixed slots / preprocessing, but INSTRUMENTS the loop to record, per
epoch: train-loss stats, VAL MSE/MAE, per-branch param norms, raw (pre-clip)
gradient norms, gamma/beta, and the CNN vs Expert branch output magnitudes on a
fixed diagnostic batch.  After training, each Expert slot is individually zeroed
on a fixed VAL subset to measure its marginal contribution (sensitivity).

Four seeds are compared: GOOD = {123, 2026}, BAD = {42, 7}.

This script does NOT modify the Critic / knowledge / network / hyper-params, does
NOT add early stopping or gradient clipping (the original clip_grad_norm_=5.0 is
kept untouched — only the RAW pre-clip norm is recorded), does NOT re-select seeds,
and does NOT read TEST.
"""

from __future__ import annotations

import json
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

import run_expert_attention_retrieval_v2 as attn                    # seed_all
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
OUT_DIR = Path("results/etth1_critic_v6_training_instability")

CONTEXT, HORIZON, TOPK = 512, 96, 5
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

SEEDS = [123, 2026, 42, 7]      # GOOD: 123, 2026 ; BAD: 42, 7
GOOD_SEEDS = {123, 2026}
BAD_SEEDS = {42, 7}

GRAD_EXPLOSION_THRESH = 100.0   # raw (pre-clip) global grad norm beyond this -> explosion


# --------------------------------------------------------------------------- #
# norm / branch helpers
# --------------------------------------------------------------------------- #
def _l2(params):
    return float(sum((p * p).sum().item() for p in params)) ** 0.5


def _param_norms(model):
    total = 0.0
    branch = {"expert": 0.0, "cnn": 0.0, "mlp": 0.0}
    for name, p in model.named_parameters():
        pn = float((p * p).sum().item())
        total += pn
        if name.startswith("ts_encoder."):
            branch["cnn"] += pn
        elif name.startswith("expert_proj."):
            branch["expert"] += pn
        elif name.startswith("mlp."):
            branch["mlp"] += pn
    return {"total": total ** 0.5,
            "expert": branch["expert"] ** 0.5,
            "cnn": branch["cnn"] ** 0.5,
            "mlp": branch["mlp"] ** 0.5}


def _grad_norms(model):
    """Raw (pre-clip) L2 gradient norms, global + per branch."""
    total = 0.0
    branch = {"expert": 0.0, "cnn": 0.0, "mlp": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        gn = float((g * g).sum().item())
        total += gn
        if name.startswith("ts_encoder."):
            branch["cnn"] += gn
        elif name.startswith("expert_proj."):
            branch["expert"] += gn
        elif name.startswith("mlp."):
            branch["mlp"] += gn
    return {"total": total ** 0.5,
            "expert": branch["expert"] ** 0.5,
            "cnn": branch["cnn"] ** 0.5,
            "mlp": branch["mlp"] ** 0.5}


def _out_stats(t):
    t = t.detach().float()
    return {"mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "max_abs": float(t.abs().max().item()),
            "norm": float(t.norm().item())}


@torch.no_grad()
def _branch_outputs(model, x_diag, act_diag):
    model.eval()
    cnn = model.ts_encoder(x_diag)                                   # [B, d_enc]
    expert = model.expert_proj(model.expert_vector(x_diag, act_diag))  # [B, d_exp]
    fused = model.encode(x_diag, act_diag)                            # [B, d_z]
    return _out_stats(cnn), _out_stats(expert), _out_stats(fused)


@torch.no_grad()
def evaluate_val(model, X_val, X_bank, Y_bank, Y_val, X_val_res, X_bank_res,
                 val_act, bank_act, device):
    pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                val_act, bank_act, "full", TOPK, device)
    return mse(Y_val, pred), mae(Y_val, pred)


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


# --------------------------------------------------------------------------- #
# instrumented D-arm training loop (faithful to run_etth1_module_ablation_v6)
# --------------------------------------------------------------------------- #
def train_and_record(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                     bank_act, fit_act, X_val, Y_val, X_val_res, val_act,
                     x_diag, act_diag, device,
                     epochs=EPOCHS, batch_size=BATCH_SIZE, candidates=CANDIDATES, lr=LR, temp=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)
    mode = "full"

    records = []
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        grad_global = []
        grad_branch = {"expert": [], "cnn": [], "mlp": []}
        nan_detected = inf_detected = False
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

            # record RAW pre-clip gradient norms (clipping is kept unchanged below)
            gn = _grad_norms(model)
            grad_global.append(gn["total"])
            for k in ("expert", "cnn", "mlp"):
                grad_branch[k].append(gn[k])
            if np.isnan(gn["total"]) or np.isnan(loss.item()):
                nan_detected = True
            if np.isinf(gn["total"]) or np.isinf(loss.item()):
                inf_detected = True

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # unchanged from original
            opt.step()
            losses.append(loss.item())

        # --- per-epoch VAL + param + gate + branch-output recording ---
        val_mse, val_mae = evaluate_val(model, X_val, X_bank, Y_bank, Y_val,
                                        X_val_res, X_bank_res, val_act, bank_act, device)
        pn = _param_norms(model)
        cnn_o, expert_o, fused_o = _branch_outputs(model, x_diag, act_diag)
        gamma = float(torch.sigmoid(model.logit_gamma).detach().cpu())
        beta = float(torch.sigmoid(model.logit_beta).detach().cpu())
        records.append({
            "epoch": ep,
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_std": float(np.std(losses)),
            "train_loss_min": float(np.min(losses)),
            "train_loss_max": float(np.max(losses)),
            "val_mse": round(val_mse, 6),
            "val_mae": round(val_mae, 6),
            "total_param_norm": round(pn["total"], 4),
            "expert_branch_param_norm": round(pn["expert"], 4),
            "cnn_param_norm": round(pn["cnn"], 4),
            "mlp_param_norm": round(pn["mlp"], 4),
            "gamma": round(gamma, 4),
            "beta": round(beta, 4),
            "logit_gamma": round(float(model.logit_gamma.detach().cpu()), 4),
            "logit_beta": round(float(model.logit_beta.detach().cpu()), 4),
            "global_grad_norm_mean": round(float(np.mean(grad_global)), 4),
            "global_grad_norm_max": round(float(np.max(grad_global)), 4),
            "expert_branch_grad_norm_mean": round(float(np.mean(grad_branch["expert"])), 4),
            "cnn_grad_norm_mean": round(float(np.mean(grad_branch["cnn"])), 4),
            "mlp_grad_norm_mean": round(float(np.mean(grad_branch["mlp"])), 4),
            "expert_branch_output": expert_o,
            "cnn_output": cnn_o,
            "fused_output": fused_o,
            "expert_cnn_output_norm_ratio": round(expert_o["norm"] / (cnn_o["norm"] + 1e-9), 4),
        })

    # final gate values
    return records


# --------------------------------------------------------------------------- #
# slot sensitivity (zero each expert slot once)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def slot_sensitivity(model, expert_ids, X_val, X_bank, Y_bank, Y_val, X_val_res, X_bank_res,
                     val_act, bank_act, device):
    full_pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                     val_act, bank_act, "full", TOPK, device)
    full_mse = mse(Y_val, full_pred)
    out = []
    for i in range(len(expert_ids)):
        va = val_act.copy(); va[:, i] = 0.0
        ba = bank_act.copy(); ba[:, i] = 0.0
        pred_i = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                      va, ba, "full", TOPK, device)
        mse_i = mse(Y_val, pred_i)
        out.append({"slot": i, "expert_id": expert_ids[i],
                    "zero_mse": round(mse_i, 4), "delta": round(mse_i - full_mse, 4)})
    return round(full_mse, 4), out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    critic_experts = _sorted_experts(json.loads(CRITIC_POOL.read_text(encoding="utf-8")))
    expert_ids = [e["expert_id"] for e in critic_experts]
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

    emb = embed_claims([e["claim"] for e in critic_experts], device)
    expert_dim = emb.shape[1]

    # ---- section 4: Expert activation scale (seed-independent, computed once) ----
    act_scale = {}
    for tag, mat in (("train", np.concatenate([bank_act, fit_act], axis=0)), ("val", val_act)):
        rows = []
        for i, eid in enumerate(expert_ids):
            a = mat[:, i]
            binary = bool(np.all((a == 0.0) | (a == 1.0)))
            rows.append({
                "expert_id": eid, "name": critic_experts[i]["name"],
                "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "p01": float(np.percentile(a, 1)), "p10": float(np.percentile(a, 10)),
                "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
                "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
                "p99": float(np.percentile(a, 99)),
                "binary": binary,
            })
        act_scale[tag] = rows

    nonz_std = [r["std"] for r in act_scale["train"] if r["std"] > 1e-9]
    scale_info = {
        "max_std": max(nonz_std) if nonz_std else 0.0,
        "min_nonzero_std": min(nonz_std) if nonz_std else 0.0,
        "max_abs_value": max(r["max"] for r in act_scale["train"]),
        "largest_scale_expert": max(act_scale["train"], key=lambda r: r["std"])["expert_id"],
        "smallest_scale_expert": min(act_scale["train"], key=lambda r: (r["std"],))["expert_id"],
        "max_std_ratio": (max(nonz_std) / min(nonz_std)) if nonz_std and min(nonz_std) > 1e-9 else None,
    }

    # ---- diagnostic batch (fixed) ----
    diag_n = 64
    x_diag = torch.tensor(X_fit[:diag_n], dtype=torch.float32, device=device)
    act_diag = torch.tensor(fit_act[:diag_n], dtype=torch.float32, device=device)

    # ---- run the 4 seeds ----
    per_seed = {}
    for seed in SEEDS:
        tag = "GOOD" if seed in GOOD_SEEDS else "BAD"
        print(f"\n{'='*66}\nSEED {seed} ({tag})\n{'='*66}")
        attn.seed_all(seed)
        model = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        model.set_expert_embeddings(emb)
        records = train_and_record(
            model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
            bank_act, fit_act, X_val, Y_val, X_val_res, val_act,
            x_diag, act_diag, device)

        full_mse, sensitivity = slot_sensitivity(
            model, expert_ids, X_val, X_bank, Y_bank, Y_val, X_val_res, X_bank_res,
            val_act, bank_act, device)

        per_seed[seed] = {
            "records": records,
            "sensitivity": sensitivity,
            "full_val_mse": full_mse,
        }
        print(f"    final val_mse={records[-1]['val_mse']:.2f} full_model_mse={full_mse:.2f}")

    # ---- analysis ----
    summary = {}
    for seed in SEEDS:
        rec = per_seed[seed]["records"]
        val_mse = [r["val_mse"] for r in rec]
        best_ep = int(np.argmin(val_mse)) + 1
        best_mse = float(np.min(val_mse))
        final_mse = rec[-1]["val_mse"]
        max_grad = max(r["global_grad_norm_max"] for r in rec)
        max_expert_ratio = max(r["expert_cnn_output_norm_ratio"] for r in rec)
        first_mse = rec[0]["val_mse"]
        # late degradation
        fmb = (final_mse - best_mse) / best_mse if best_mse > 0 else 0.0
        # harmful expert slot (largest positive delta = removing it helps the most)
        sens = per_seed[seed]["sensitivity"]
        harmful = min(sens, key=lambda s: s["delta"])  # most negative delta = removal helps most
        summary[seed] = {
            "best_val_mse": round(best_mse, 2),
            "final_val_mse": round(final_mse, 2),
            "best_epoch": best_ep,
            "first_epoch_mse": round(first_mse, 2),
            "final_minus_best": round(final_mse - best_mse, 2),
            "final_minus_best_pct": round(fmb * 100, 2),
            "late_training_degradation": bool(fmb > 0.20),
            "max_grad_norm": round(max_grad, 2),
            "nan_detected": any(np.isnan(r["global_grad_norm_max"]) for r in rec),
            "inf_detected": any(np.isinf(r["global_grad_norm_max"]) for r in rec),
            "gradient_explosion_detected": bool(max_grad > GRAD_EXPLOSION_THRESH),
            "max_expert_cnn_output_norm_ratio": round(max_expert_ratio, 3),
            "expert_branch_domination": bool(max_expert_ratio > 10.0),
            "expert_param_norm_final": rec[-1]["expert_branch_param_norm"],
            "cnn_param_norm_final": rec[-1]["cnn_param_norm"],
            "gamma_final": rec[-1]["gamma"],
            "beta_final": rec[-1]["beta"],
            "harmful_expert_slot": harmful["expert_id"],
            "harmful_expert_delta": harmful["delta"],
        }

    # ---- ROOT_CAUSE heuristic ----
    bad = [s for s in SEEDS if s in BAD_SEEDS]
    good = [s for s in SEEDS if s in GOOD_SEEDS]
    bad_first_high = all(summary[s]["first_epoch_mse"] > 60 for s in bad)
    good_first_low = all(summary[s]["first_epoch_mse"] < 60 for s in good)
    bad_late = any(summary[s]["late_training_degradation"] for s in bad)
    bad_grad = any(summary[s]["gradient_explosion_detected"] for s in bad)
    bad_dom = any(summary[s]["expert_branch_domination"] for s in bad)
    bad_harmful = any(summary[s]["harmful_expert_delta"] < -20 for s in bad)

    if bad_first_high and not bad_grad and not bad_dom:
        cause = "BAD_INITIALIZATION_OR_LOCAL_OPTIMUM"
    elif bad_late:
        cause = "LATE_TRAINING_DEGRADATION"
    elif bad_grad:
        cause = "GRADIENT_INSTABILITY"
    elif bad_dom:
        cause = "EXPERT_BRANCH_DOMINATION"
    elif bad_harmful:
        cause = "SPECIFIC_EXPERT_OVERRELIANCE"
    else:
        cause = "UNCLEAR"

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_seed_records.json").write_text(
        json.dumps({str(s): per_seed[s] for s in SEEDS}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "activation_scale.json").write_text(
        json.dumps({"scale": act_scale, "scale_info": scale_info}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"seeds": SEEDS, "per_seed": summary,
                    "activation_scale_info": scale_info, "root_cause": cause},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report ----
    L = []
    L.append("【结果】")
    L.append("")
    L.append("## 1. per-seed train/VAL curve summary (first / best / final epoch)")
    L.append("")
    L.append("| seed | first_ep MSE | best_ep | best MSE | final MSE | final-best |")
    L.append("|---|---|---|---|---|---|")
    for seed in SEEDS:
        s = summary[seed]
        L.append(f"| {seed} | {s['first_epoch_mse']:.2f} | {s['best_epoch']} | "
                 f"{s['best_val_mse']:.2f} | {s['final_val_mse']:.2f} | {s['final_minus_best']:+.2f} |")
    L.append("")
    L.append("## 2. best epoch vs final epoch")
    L.append("")
    L.append("| seed | best MSE | final MSE | final-best | pct | late_degradation |")
    L.append("|---|---|---|---|---|---|")
    for seed in SEEDS:
        s = summary[seed]
        L.append(f"| {seed} | {s['best_val_mse']:.2f} | {s['final_val_mse']:.2f} | "
                 f"{s['final_minus_best']:+.2f} | {s['final_minus_best_pct']:+.1f}% | "
                 f"{s['late_training_degradation']} |")
    L.append("")
    L.append("## 3. gradient norm (raw pre-clip)")
    L.append("")
    L.append("| seed | max_global_grad_norm | nan | inf | explosion |")
    L.append("|---|---|---|---|---|")
    for seed in SEEDS:
        s = summary[seed]
        L.append(f"| {seed} | {s['max_grad_norm']:.2f} | {s['nan_detected']} | "
                 f"{s['inf_detected']} | {s['gradient_explosion_detected']} |")
    L.append("")
    L.append("## 4. Expert activation scale (TRAIN queries; seed-independent)")
    L.append("")
    L.append("| expert | binary | mean | std | min | max | p50 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in act_scale["train"]:
        L.append(f"| {r['expert_id']} | {r['binary']} | {r['mean']:.3f} | {r['std']:.3f} | "
                 f"{r['min']:.2f} | {r['max']:.2f} | {r['p50']:.2f} |")
    L.append("")
    L.append(f"- max_std = {scale_info['max_std']:.3f} (expert {scale_info['largest_scale_expert']})")
    L.append(f"- min_nonzero_std = {scale_info['min_nonzero_std']:.3f} (expert {scale_info['smallest_scale_expert']})")
    L.append(f"- max_abs_value = {scale_info['max_abs_value']:.3f}")
    L.append(f"- max_std / min_nonzero_std = {scale_info['max_std_ratio']}")
    L.append("")
    L.append("## 5. Expert/CNN branch output norm (final epoch, fixed diagnostic batch)")
    L.append("")
    L.append("| seed | cnn_norm | expert_norm | expert/cnn ratio | fused_norm |")
    L.append("|---|---|---|---|---|")
    for seed in SEEDS:
        rec = per_seed[seed]["records"][-1]
        L.append(f"| {seed} | {rec['cnn_output']['norm']:.2f} | {rec['expert_branch_output']['norm']:.2f} | "
                 f"{rec['expert_cnn_output_norm_ratio']:.2f} | {rec['fused_output']['norm']:.2f} |")
    L.append("")
    L.append("## 6. per-expert slot zero-out sensitivity (delta = zero_mse - full_mse)")
    L.append("")
    L.append("| seed | full_mse | " + " | ".join(f"ΔE{i+1}" for i in range(n_slots)) + " |")
    L.append("|---|" + "|".join(["---"] * (n_slots + 1)) + "|")
    for seed in SEEDS:
        sens = per_seed[seed]["sensitivity"]
        cells = " | ".join(f"{s['delta']:+.1f}" for s in sens)
        L.append(f"| {seed} | {per_seed[seed]['full_val_mse']:.2f} | {cells} |")
    L.append("")
    L.append("## 7. GOOD vs BAD comparison")
    L.append("")
    hdr = "| metric | " + " | ".join(f"seed{s}" for s in SEEDS) + " |"
    sep = "|---|" + "|".join(["---"] * len(SEEDS)) + "|"
    L.append(hdr); L.append(sep)
    for metric, key in (("best_val_mse", "best_val_mse"), ("final_val_mse", "final_val_mse"),
                        ("best_epoch", "best_epoch"), ("final_minus_best", "final_minus_best"),
                        ("max_grad_norm", "max_grad_norm"), ("expert_param_norm", "expert_param_norm_final"),
                        ("cnn_param_norm", "cnn_param_norm_final"),
                        ("expert/cnn output norm ratio", "max_expert_cnn_output_norm_ratio"),
                        ("gamma", "gamma_final"), ("beta", "beta_final"),
                        ("largest harmful expert slot", "harmful_expert_slot")):
        vals = " | ".join(str(summary[s][key]) for s in SEEDS)
        L.append(f"| {metric} | {vals} |")
    L.append("")

    L.append("【结论】")
    L.append("")
    bad_first = {s: summary[s]["first_epoch_mse"] for s in bad}
    L.append(f"A. 坏 seed 是不是训练后期恶化？ "
             f"{'是' if bad_late else '否'}"
             f"（final_minus_best_pct = "
             + ", ".join(f"seed{s}={summary[s]['final_minus_best_pct']:.1f}%" for s in bad) + "）")
    L.append(f"B. 是否存在梯度爆炸？ {'是' if bad_grad else '否'}"
             f"（max_grad_norm: "
             + ", ".join(f"seed{s}={summary[s]['max_grad_norm']:.1f}" for s in bad)
             + f"，clip=5.0）")
    L.append(f"C. 是否存在 Expert 输入尺度异常？ "
             f"{'是' if (scale_info['max_std_ratio'] or 0) > 100 else '否'}"
             f"（max_std/min_nonzero_std={scale_info['max_std_ratio']}）")
    L.append(f"D. 是否存在 Expert branch 过度主导？ {'是' if bad_dom else '否'}"
             f"（expert/cnn norm ratio: "
             + ", ".join(f"seed{s}={summary[s]['max_expert_cnn_output_norm_ratio']:.2f}" for s in bad) + "）")
    L.append(f"E. 是否存在某个 Expert slot 被错误过度依赖？ {'是' if bad_harmful else '否'}"
             f"（最有害 slot delta: "
             + ", ".join(f"seed{s}: {summary[s]['harmful_expert_slot']} {summary[s]['harmful_expert_delta']:+.1f}" for s in bad) + "）")
    L.append(f"F. ROOT_CAUSE = **{cause}**")
    L.append("")
    L.append(f"ROOT_CAUSE = {cause}")
    if cause == "BAD_INITIALIZATION_OR_LOCAL_OPTIMUM":
        L.append("RECOMMENDED_FIX = (diagnosis only) multiple inits per seed / warm-up; "
                 "do not change Critic or network")
    elif cause == "LATE_TRAINING_DEGRADATION":
        L.append("RECOMMENDED_FIX = (diagnosis only) investigate overfitting; more fit data / "
                 "weight-decay tuning (not applied here)")
    elif cause == "GRADIENT_INSTABILITY":
        L.append("RECOMMENDED_FIX = (diagnosis only) investigate LR/init scale (not applied here)")
    elif cause == "EXPERT_BRANCH_DOMINATION":
        L.append("RECOMMENDED_FIX = (diagnosis only) balance expert vs CNN branch scale (not applied here)")
    elif cause == "SPECIFIC_EXPERT_OVERRELIANCE":
        L.append("RECOMMENDED_FIX = (diagnosis only) inspect the flagged slot (not applied here)")
    else:
        L.append("RECOMMENDED_FIX = (diagnosis only) further investigation needed")
    L.append("")
    L.append("critic_modified = false")
    L.append("network_modified = false")
    L.append("hyperparameters_modified = false")
    L.append("test_used = false")
    L.append("diagnosis_only = true")
    report = "\n".join(L)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
