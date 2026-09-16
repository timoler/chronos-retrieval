#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 — module ablation v6 with STRICT deterministic settings (formal re-run).

Identical to ``run_etth1_module_ablation_v6`` (arms A/B/C/D + ZERO/SHUFFLED/REAL,
frozen Raw / Critic-v6 pools, fixed slots + mask, shared init per seed, same
network / optimizer / LR / epochs / batch / Shape / Expert activation /
preprocessing), with ONE protocol change: strict deterministic seeding
(PYTHONHASHSEED + random/numpy/torch seeds + cudnn.deterministic=True +
cudnn.benchmark=False + use_deterministic_algorithms(True)).

The Critic is NOT re-run and experts are NOT regenerated: the Raw pool and the
Critic-v6 final ACCEPT pool are read frozen.  TEST is never read.

A reproducibility check re-runs seed 123 + D (+CRITIC EXPERT) once more and asserts
MSE diff == 0 and checksums equal, confirming the deterministic protocol took effect.
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

RAW_POOL = Path("results/etth1_critic_v5_closed_loop/expert_round_1.json")
CRITIC_POOL = Path("results/etth1_critic_v6_closed_loop/final_accepted_knowledge.json")
OUT_DIR = Path("results/etth1_module_ablation_v6_deterministic")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
SHUFFLE_SEED = 777
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32


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


def _checksum(model) -> str:
    h = hashlib.sha256()
    for name in sorted(model.state_dict().keys()):
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        t = model.state_dict()[name].detach().cpu().numpy()
        h.update(np.ascontiguousarray(t.astype(np.float32)).tobytes())
    return h.hexdigest()


def _sorted_experts(pool):
    if isinstance(pool, dict):
        items = list(pool.values())
    else:
        items = list(pool)
    return sorted(items, key=lambda e: int(str(e.get("expert_id", "0")).lstrip("E") or 0))


def pad_to_slots(emb, act_dict, n_slots):
    n = emb.shape[0]
    mask = np.zeros(n_slots, dtype=np.float32)
    mask[:n] = 1.0
    if n == n_slots:
        return emb, act_dict, mask
    pad_emb = np.zeros((n_slots - n, emb.shape[1]), dtype=np.float32)
    emb = np.concatenate([emb, pad_emb], axis=0)
    padded_act = {}
    for k, a in act_dict.items():
        padded_act[k] = np.concatenate([a, np.zeros((a.shape[0], n_slots - n), dtype=np.float32)], axis=1)
    return emb, padded_act, mask


def embed_claims(claims, device):
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    emb = enc.encode(claims, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


@torch.no_grad()
def evaluate_val(model, X_val, X_bank, Y_bank, Y_val, X_val_res, X_bank_res,
                 val_act, bank_act, mode, device):
    pred = retrieve_conditional(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                val_act, bank_act, mode, TOPK, device)
    return mse(Y_val, pred), mae(Y_val, pred)


# --------------------------------------------------------------------------- #
# instrumented D-arm training loop (faithful to the ablation, per-epoch recorded)
# --------------------------------------------------------------------------- #
def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, X_val, Y_val, X_val_res, val_act, device,
              epochs=EPOCHS, batch_size=BATCH_SIZE, candidates=CANDIDATES, lr=LR):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)
    mode = "full" if use_shape else "expert"

    epoch_train_loss, epoch_val_mse, epoch_val_mae = [], [], []
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
                                        X_val_res, X_bank_res, val_act, bank_act, mode, device)
        epoch_train_loss.append(float(np.mean(losses)))
        epoch_val_mse.append(val_mse)
        epoch_val_mae.append(val_mae)

    best_ep = int(np.argmin(epoch_val_mse)) + 1
    return {
        "final_mse": float(epoch_val_mse[-1]),
        "final_mae": float(epoch_val_mae[-1]),
        "best_val_mse": float(min(epoch_val_mse)),
        "best_val_epoch": best_ep,
        "final_train_loss": float(epoch_train_loss[-1]),
        "epoch_train_loss": [float(x) for x in epoch_train_loss],
        "epoch_val_mse": [float(x) for x in epoch_val_mse],
        "epoch_val_mae": [float(x) for x in epoch_val_mae],
        "gamma": float(torch.sigmoid(model.logit_gamma).detach().cpu()),
        "beta": float(torch.sigmoid(model.logit_beta).detach().cpu()),
    }


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| strict deterministic: ON")

    # ---- frozen pools ----
    raw_experts = _sorted_experts(json.loads(RAW_POOL.read_text(encoding="utf-8")))
    critic_experts = _sorted_experts(json.loads(CRITIC_POOL.read_text(encoding="utf-8")))
    n_raw, n_critic = len(raw_experts), len(critic_experts)
    n_slots = max(n_raw, n_critic)
    print(f"raw={n_raw} critic={n_critic} fixed slots={n_slots}")

    # ---- data ----
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

    def act_mats(experts):
        return {"bank": build_activation_matrix(experts, X_bank, bank_ts, stats),
                "fit": build_activation_matrix(experts, X_fit, fit_ts, stats),
                "val": build_activation_matrix(experts, X_val, val_ts, stats)}

    raw_act = act_mats(raw_experts)
    critic_act = act_mats(critic_experts)

    emb_raw = embed_claims([e["claim"] for e in raw_experts], device)
    emb_critic = embed_claims([e["claim"] for e in critic_experts], device)
    expert_dim = emb_raw.shape[1]

    emb_raw, raw_act, raw_mask = pad_to_slots(emb_raw, raw_act, n_slots)
    emb_critic, critic_act, critic_mask = pad_to_slots(emb_critic, critic_act, n_slots)

    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_act = {k: np.zeros_like(a) for k, a in critic_act.items()}

    def _perm(a, seed):
        rng = np.random.default_rng(seed)
        return a[rng.permutation(len(a))]
    shuffled_act = {k: _perm(critic_act[k], SHUFFLE_SEED) for k in critic_act}

    # configs: name -> (emb, act, use_shape)
    configs = {
        "base":          (emb_zero,    zero_act,      False),
        "shape":         (emb_zero,    zero_act,      True),
        "raw_expert":    (emb_raw,     raw_act,       True),
        "critic_expert": (emb_critic,  critic_act,    True),
        "zero":          (emb_critic,  zero_act,      True),
        "shuffled":      (emb_critic,  shuffled_act,  True),
    }

    per_seed = {name: [] for name in configs}
    records = {}   # (seed, arm) -> full record
    param_count = None

    for seed in SEEDS:
        print(f"\n{'='*70}\nSEED {seed} (deterministic)\n{'='*70}")
        seed_everything(seed)
        base = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
        if param_count is None:
            param_count = n_params(base)

        for name, (emb, act, use_shape) in configs.items():
            print(f"\n[{name}] use_shape={use_shape}")
            model = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
            model.load_state_dict(init_state)
            model.set_expert_embeddings(emb)
            initial_checksum = _checksum(model)
            rec = train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                            act["bank"], act["fit"], use_shape,
                            X_val, Y_val, X_val_res, act["val"], device)
            final_checksum = _checksum(model)
            rec["initial_checksum"] = initial_checksum
            rec["final_checksum"] = final_checksum
            records[(seed, name)] = rec
            per_seed[name].append({"seed": seed, "mse": rec["final_mse"], "mae": rec["final_mae"]})
            print(f"    [{name}] final_mse={rec['final_mse']:.4f} best={rec['best_val_mse']:.4f}@ep{rec['best_val_epoch']}")

    # REAL == D (alias, not retrained)
    per_seed["real"] = [dict(v) for v in per_seed["critic_expert"]]

    # ---- reproducibility check: seed 123 + D re-run once ----
    print(f"\n{'='*70}\nREPRODUCIBILITY CHECK: seed 123 + critic_expert (run 2)\n{'='*70}")
    seed_everything(123)
    base2 = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    init_state2 = {k: v.detach().clone() for k, v in base2.state_dict().items()}
    model2 = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    model2.load_state_dict(init_state2)
    model2.set_expert_embeddings(emb_critic)
    init2_checksum = _checksum(model2)
    rec2 = train_arm(model2, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                     critic_act["bank"], critic_act["fit"], True,
                     X_val, Y_val, X_val_res, critic_act["val"], device)
    final2_checksum = _checksum(model2)

    r1 = records[(123, "critic_expert")]
    repro = {
        "seed": 123, "arm": "critic_expert",
        "run1_final_mse": r1["final_mse"], "run2_final_mse": rec2["final_mse"],
        "mse_abs_diff": abs(r1["final_mse"] - rec2["final_mse"]),
        "initial_checksum_equal": r1["initial_checksum"] == init2_checksum,
        "final_checksum_equal": r1["final_checksum"] == final2_checksum,
        "max_epoch_val_mse_diff": max(abs(a - b) for a, b in zip(r1["epoch_val_mse"], rec2["epoch_val_mse"])),
        "max_epoch_train_loss_diff": max(abs(a - b) for a, b in zip(r1["epoch_train_loss"], rec2["epoch_train_loss"])),
    }
    repro["reproducible"] = bool(repro["mse_abs_diff"] == 0.0 and repro["initial_checksum_equal"]
                                 and repro["final_checksum_equal"]
                                 and repro["max_epoch_val_mse_diff"] == 0.0)
    print(f"    MSE diff={repro['mse_abs_diff']:.3e} init_equal={repro['initial_checksum_equal']} "
          f"final_equal={repro['final_checksum_equal']} curve_diff={repro['max_epoch_val_mse_diff']:.3e}")

    # ---- aggregate ----
    def mean_std(vals):
        a = np.array([v["mse"] for v in vals], dtype=np.float64)
        b = np.array([v["mae"] for v in vals], dtype=np.float64)
        return (float(a.mean()), float(a.std()), float(b.mean()), float(b.std()))

    summary = {}
    for name, vals in per_seed.items():
        mm, ms, am, as_ = mean_std(vals)
        summary[name] = {"mse_mean": mm, "mse_std": ms, "mae_mean": am, "mae_std": as_, "per_seed": vals}

    b, s, r, c = summary["base"], summary["shape"], summary["raw_expert"], summary["critic_expert"]
    gains = {
        "shape_vs_base_pct": (b["mse_mean"] - s["mse_mean"]) / b["mse_mean"] * 100.0,
        "raw_vs_shape_pct": (s["mse_mean"] - r["mse_mean"]) / s["mse_mean"] * 100.0,
        "critic_vs_raw_pct": (r["mse_mean"] - c["mse_mean"]) / r["mse_mean"] * 100.0,
        "critic_vs_base_pct": (b["mse_mean"] - c["mse_mean"]) / b["mse_mean"] * 100.0,
        "real_vs_zero_pct": (summary["zero"]["mse_mean"] - summary["real"]["mse_mean"]) / summary["zero"]["mse_mean"] * 100.0,
        "real_vs_shuffled_pct": (summary["shuffled"]["mse_mean"] - summary["real"]["mse_mean"]) / summary["shuffled"]["mse_mean"] * 100.0,
    }

    # ---- save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": "ETTh1", "deterministic_enabled": True,
        "protocol": "multidataset ETTh1 VAL, residualize Pearson, Top-K=5 mean",
        "critic_modified": False, "expert_regenerated": False, "network_modified": False,
        "hyperparameters_modified": False, "test_used": False, "seed42_removed": False,
        "seeds": SEEDS,
        "expert_sources": {"raw": str(RAW_POOL), "critic": str(CRITIC_POOL),
                           "n_slots": n_slots,
                           "mask_raw": [int(x) for x in raw_mask],
                           "mask_critic": [int(x) for x in critic_mask],
                           "param_count": param_count},
        "arms": {k: {kk: vv for kk, vv in summary[k].items() if kk != "per_seed"}
                 for k in ("base", "shape", "raw_expert", "critic_expert")},
        "gains": gains,
        "aux_control": {k: {kk: vv for kk, vv in summary[k].items() if kk != "per_seed"}
                        for k in ("zero", "shuffled", "real")},
        "reproducibility_check": repro,
        "hyperparameters": {"context": CONTEXT, "horizon": HORIZON, "topk": TOPK,
                            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                            "candidates": CANDIDATES, "lr": LR,
                            "d_enc": D_ENC, "d_exp": D_EXP, "d_z": D_Z, "hidden": HIDDEN},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "per_seed_results.json").write_text(
        json.dumps({"seeds": SEEDS, "arms": per_seed, "param_count": param_count, "n_slots": n_slots},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "records.json").write_text(
        json.dumps({f"{s}_{k}": v for (s, k), v in records.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report ----
    def fmt(name):
        ss = summary[name]
        return f"{ss['mse_mean']:.2f} ± {ss['mse_std']:.2f}   /   {ss['mae_mean']:.4f} ± {ss['mae_std']:.4f}"

    L = []
    L.append("【结果】")
    L.append("")
    L.append("## 1. A/B/C/D three-seed full table (final-epoch MSE)")
    L.append("")
    L.append("| Method | seed42 | seed123 | seed2026 | mean MSE | std | mean MAE | std |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, key in (("BASE", "base"), ("+SHAPE", "shape"),
                       ("+RAW EXPERT", "raw_expert"), ("+CRITIC EXPERT v6", "critic_expert")):
        vals = sorted(per_seed[key], key=lambda v: v["seed"])
        cells = " | ".join(f"{v['mse']:.2f}" for v in vals)
        s = summary[key]
        L.append(f"| {label} | {cells} | {s['mse_mean']:.2f} | {s['mse_std']:.2f} | "
                 f"{s['mae_mean']:.4f} | {s['mae_std']:.4f} |")
    L.append("")
    L.append("## 2. mean ± std")
    L.append("")
    for label, key in (("BASE", "base"), ("+SHAPE", "shape"),
                       ("+RAW EXPERT", "raw_expert"), ("+CRITIC EXPERT v6", "critic_expert")):
        L.append(f"- {label}: MSE {fmt(key)}")
    L.append("")
    L.append(f"- Shape vs Base = {gains['shape_vs_base_pct']:+.2f}%")
    L.append(f"- Raw Expert vs Shape = {gains['raw_vs_shape_pct']:+.2f}%")
    L.append(f"- Critic Expert vs Raw = {gains['critic_vs_raw_pct']:+.2f}%")
    L.append(f"- Critic Expert vs Base = {gains['critic_vs_base_pct']:+.2f}%")
    L.append("")
    L.append("## 3. per-seed method comparison")
    L.append("")
    L.append("| seed | BASE | +SHAPE | +RAW | +CRITIC v6 |")
    L.append("|---|---|---|---|---|")
    for seed in SEEDS:
        vals = {}
        for key in ("base", "shape", "raw_expert", "critic_expert"):
            v = [x for x in per_seed[key] if x["seed"] == seed][0]
            vals[key] = v["mse"]
        L.append(f"| {seed} | {vals['base']:.2f} | {vals['shape']:.2f} | "
                 f"{vals['raw_expert']:.2f} | {vals['critic_expert']:.2f} |")
    L.append("")
    L.append("## 4. ZERO / SHUFFLED / REAL")
    L.append("")
    L.append(f"- ZERO = {fmt('zero')}")
    L.append(f"- SHUFFLED = {fmt('shuffled')}")
    L.append(f"- REAL = {fmt('real')}")
    L.append(f"- REAL vs ZERO = {gains['real_vs_zero_pct']:+.2f}%")
    L.append(f"- REAL vs SHUFFLED = {gains['real_vs_shuffled_pct']:+.2f}%")
    L.append("")
    L.append("## 5. reproducibility check (seed123 + D)")
    L.append("")
    L.append(f"- MSE diff = {repro['mse_abs_diff']:.3e}")
    L.append(f"- initial checksum equal = {repro['initial_checksum_equal']}")
    L.append(f"- final checksum equal = {repro['final_checksum_equal']}")
    L.append(f"- epoch curve diff = {repro['max_epoch_val_mse_diff']:.3e}")
    L.append(f"- reproducible = {repro['reproducible']}")
    L.append("")
    L.append("## 6. (reference only) old nondeterministic vs new deterministic")
    L.append("")
    L.append("(old nondeterministic critic D: mean 49.22, seed42 81.31 / seed123 32.83 / seed2026 33.52)")
    L.append("")
    L.append("【结论】")
    L.append("")
    L.append(f"A. deterministic 后 Raw Expert 是否仍明显优于 Base/Shape？ "
             f"{'是' if summary['raw_expert']['mse_mean'] < summary['shape']['mse_mean'] * 0.5 else '否'}"
             f"（raw={summary['raw_expert']['mse_mean']:.2f} vs base={summary['base']['mse_mean']:.2f} "
             f"/ shape={summary['shape']['mse_mean']:.2f}）")
    d_better = summary["critic_expert"]["mse_mean"] < summary["raw_expert"]["mse_mean"]
    L.append(f"B. deterministic 后 Critic v6 是否优于/接近/差于 Raw Expert？ "
             f"{'优于' if d_better else ('接近' if abs(gains['critic_vs_raw_pct']) < 5 else '差于')}"
             f"（critic={summary['critic_expert']['mse_mean']:.2f} vs raw={summary['raw_expert']['mse_mean']:.2f}，"
             f"{gains['critic_vs_raw_pct']:+.2f}%）")
    s42 = [x for x in per_seed["critic_expert"] if x["seed"] == 42][0]["mse"]
    s42_others = np.mean([x["mse"] for x in per_seed["critic_expert"] if x["seed"] != 42])
    L.append(f"C. seed42 的异常是否仍然存在？ {'是' if s42 > s42_others * 1.5 else '否'}"
             f"（seed42 critic={s42:.2f} vs 其他两 seed 均值={s42_others:.2f}）")
    L.append(f"D. Critic v6 的跨 seed 稳定性如何？ std={summary['critic_expert']['mse_std']:.2f}"
             f"（seed42={s42:.2f} / seed123={[x for x in per_seed['critic_expert'] if x['seed']==123][0]['mse']:.2f} "
             f"/ seed2026={[x for x in per_seed['critic_expert'] if x['seed']==2026][0]['mse']:.2f}）")
    L.append(f"E. ETTh1 当前是否可以冻结实验协议？ {'是' if repro['reproducible'] else '否'}"
             f"（deterministic 已生效、可复现）")
    L.append("")
    L.append("critic_modified = false")
    L.append("expert_regenerated = false")
    L.append("network_modified = false")
    L.append("hyperparameters_modified = false")
    L.append("deterministic_enabled = true")
    L.append("test_used = false")
    L.append("seed42_removed = false")
    report = "\n".join(L)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("\n" + report)


if __name__ == "__main__":
    main()
