#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 Critic v6 — additional-seed training-stability test (D arm only).

Re-runs ONLY the D arm (+CRITIC EXPERT v6) of ``run_etth1_module_ablation_v6`` on
5 NEW seeds (7, 43, 100, 3407, 9999) and merges them with the already-computed
seeds 42 / 123 / 2026, to judge whether seed 42's MSE=81.31 is a one-off outlier or
a sign of systematic training instability in the Critic v6 arm.

The training loop, network, hyper-params, TRAIN/VAL split, candidate pool (Critic v6
final accepted knowledge), activation code, Shape code, fixed expert slots, and
preprocessing are copied VERBATIM from the D arm — only per-epoch TRAIN-loss
recording is added (the forward/backward/optimizer steps are identical).

Does NOT re-run A/B/C. Does NOT re-run 42/123/2026. Does NOT touch TEST. Does NOT
modify the Critic / network / hyper-params / optimizer / LR / epochs / clipping.
No early stopping.
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
EXISTING = Path("results/etth1_module_ablation_v6/per_seed_results.json")
OUT_DIR = Path("results/etth1_critic_v6_seed_stability")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SHUFFLE_SEED = 777
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

EXISTING_SEEDS = [42, 123, 2026]
NEW_SEEDS = [7, 43, 100, 3407, 9999]
ALL_SEEDS = sorted(NEW_SEEDS + EXISTING_SEEDS)


# --------------------------------------------------------------------------- #
# faithful copy of run_etth1_module_ablation_v6.train_arm, use_shape=True
# (mode="full"), with per-epoch TRAIN-loss recording added.
# --------------------------------------------------------------------------- #
def train_critic_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                     bank_act, fit_act, device, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     candidates=CANDIDATES, lr=LR, temp=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)                 # fixed batch/candidate order
    mode = "full"                                      # D arm: Shape ON

    epoch_losses = []
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
        epoch_losses.append(float(np.mean(losses)))
    return epoch_losses


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


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- 1. frozen Critic v6 pool (D arm) ----
    critic_experts = _sorted_experts(json.loads(CRITIC_POOL.read_text(encoding="utf-8")))
    n_slots = len(critic_experts)
    print(f"critic experts={n_slots} (fixed slots={n_slots})")

    # ---- 2. data (identical to run_etth1_module_ablation_v6) ----
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
    print(f"expert_dim={expert_dim} n_slots={n_slots}")

    # ---- 3. run the 5 NEW seeds only ----
    new_results = []
    param_count = None
    for seed in NEW_SEEDS:
        print(f"\n{'='*64}\nSEED {seed} (new)\n{'='*64}")
        attn.seed_all(seed)
        model = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN,
        ).to(device)
        model.set_expert_embeddings(emb)
        if param_count is None:
            param_count = n_params(model)

        epoch_losses = train_critic_arm(
            model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
            bank_act, fit_act, device,
        )

        pred = retrieve_conditional(
            model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
            val_act, bank_act, "full", TOPK, device,
        )
        val_mse = mse(Y_val, pred)
        val_mae = mae(Y_val, pred)
        print(f"    val_mse={val_mse:.2f} val_mae={val_mae:.4f}")

        min_idx = int(np.argmin(epoch_losses))
        new_results.append({
            "seed": seed,
            "val_mse": round(val_mse, 6),
            "val_mae": round(val_mae, 6),
            "best_val_mse": round(val_mse, 6),       # final epoch (no early stopping)
            "final_train_loss": round(epoch_losses[-1], 6),
            "min_train_loss": round(epoch_losses[min_idx], 6),
            "epoch_of_min_train_loss": min_idx + 1,
            "min_val_mse": None,                     # no per-epoch val curve in original loop
            "epoch_of_min_val_mse": None,
            "train_loss_curve": [round(x, 6) for x in epoch_losses],
        })

    # ---- 4. merge with existing 42/123/2026 ----
    per_seed = json.loads(EXISTING.read_text(encoding="utf-8"))
    critic_rows = per_seed["arms"]["critic_expert"]
    existing_results = [
        {"seed": int(r["seed"]), "val_mse": round(float(r["mse"]), 6),
         "val_mae": round(float(r["mae"]), 6),
         "best_val_mse": round(float(r["mse"]), 6),
         "final_train_loss": None, "min_train_loss": None, "epoch_of_min_train_loss": None,
         "min_val_mse": None, "epoch_of_min_val_mse": None, "train_loss_curve": None,
         "source": "existing"}
        for r in critic_rows
    ]
    for r in new_results:
        r["source"] = "new"

    merged = {r["seed"]: r for r in existing_results + new_results}
    assert set(merged.keys()) == set(ALL_SEEDS), f"seed mismatch: {set(merged.keys())} vs {set(ALL_SEEDS)}"

    # ---- 5. aggregate stats ----
    mse_by_seed = {s: merged[s]["val_mse"] for s in ALL_SEEDS}
    mae_by_seed = {s: merged[s]["val_mae"] for s in ALL_SEEDS}

    mse_vals = np.array([mse_by_seed[s] for s in ALL_SEEDS], dtype=np.float64)
    sorted_seeds = sorted(ALL_SEEDS, key=lambda s: mse_by_seed[s])

    def mean(v):
        return float(np.mean(v))

    def std(v):
        return float(np.std(v))

    def median(v):
        return float(np.median(v))

    mean8, median8, std8 = mean(mse_vals), median(mse_vals), std(mse_vals)
    min8, max8 = float(mse_vals.min()), float(mse_vals.max())

    existing_mse = np.array([mse_by_seed[s] for s in EXISTING_SEEDS], dtype=np.float64)
    mean3 = mean(existing_mse)

    # seed42 vs other 7
    other7 = [s for s in ALL_SEEDS if s != 42]
    other7_mse = np.array([mse_by_seed[s] for s in other7], dtype=np.float64)
    other7_mean = mean(other7_mse)
    other7_median = median(other7_mse)
    ratio42 = mse_by_seed[42] / other7_mean if other7_mean else float("nan")
    ratio42_median = mse_by_seed[42] / other7_median if other7_median else float("nan")

    # distribution counts
    below40 = [s for s in ALL_SEEDS if mse_by_seed[s] < 40]
    between40_60 = [s for s in ALL_SEEDS if 40 <= mse_by_seed[s] <= 60]
    above60 = [s for s in ALL_SEEDS if mse_by_seed[s] > 60]
    frac_below40 = len(below40) / len(ALL_SEEDS)
    frac_above60 = len(above60) / len(ALL_SEEDS)

    # pattern
    if len(below40) >= 6 and len(above60) == 1 and 42 in above60:
        pattern = "single_outlier"
    elif len(above60) >= 2:
        pattern = "multi_seed_instability"
    else:
        pattern = "mixed"

    # seed42 outlier / instability flags
    new_above70 = [s for s in NEW_SEEDS if mse_by_seed[s] > 70]
    seed42_likely_outlier = bool(ratio42 > 1.5 and len(new_above70) == 0)
    training_instability_detected = bool(len(new_above70) >= 1)

    # ---- 6. save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_seed_new.json").write_text(
        json.dumps(new_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "merged_results.json").write_text(
        json.dumps({"seeds": ALL_SEEDS, "results": {str(s): merged[s] for s in ALL_SEEDS}},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "critic_modified": False, "network_modified": False,
        "hyperparameters_modified": False, "test_used": False,
        "existing_seeds_replaced": False,
        "new_seeds": NEW_SEEDS, "existing_seeds": EXISTING_SEEDS, "all_seeds": ALL_SEEDS,
        "param_count": param_count, "n_slots": n_slots,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "candidates": CANDIDATES, "lr": LR,
        "critic_pool": str(CRITIC_POOL),
        "mse_by_seed": {str(s): round(mse_by_seed[s], 6) for s in ALL_SEEDS},
        "mae_by_seed": {str(s): round(mae_by_seed[s], 6) for s in ALL_SEEDS},
        "mean_mse": round(mean8, 6), "median_mse": round(median8, 6),
        "std_mse": round(std8, 6), "min_mse": round(min8, 6), "max_mse": round(max8, 6),
        "mean3_mse": round(mean3, 6), "mean8_mse": round(mean8, 6),
        "seed42_mse": round(mse_by_seed[42], 6),
        "other7_mean_mse": round(other7_mean, 6), "other7_median_mse": round(other7_median, 6),
        "ratio42": round(ratio42, 6), "ratio42_median": round(ratio42_median, 6),
        "num_seeds_mse_below_40": len(below40),
        "num_seeds_mse_40_to_60": len(between40_60),
        "num_seeds_mse_above_60": len(above60),
        "fraction_below_40": round(frac_below40, 4),
        "fraction_above_60": round(frac_above60, 4),
        "pattern": pattern,
        "seed42_likely_outlier": seed42_likely_outlier,
        "training_instability_detected": training_instability_detected,
        "new_seeds_above_70": new_above70,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 7. report ----
    L = []
    L.append("【结果】")
    L.append("")
    L.append("## 1. 8-seed full MSE/MAE table (sorted by MSE)")
    L.append("")
    L.append("| seed | MSE | MAE | rank |")
    L.append("|---|---|---|---|")
    for rank, s in enumerate(sorted_seeds, 1):
        L.append(f"| {s} | {mse_by_seed[s]:.2f} | {mae_by_seed[s]:.4f} | {rank} |")
    L.append("")
    L.append("## 2. aggregate stats")
    L.append("")
    L.append(f"- mean MSE = {mean8:.2f}")
    L.append(f"- median MSE = {median8:.2f}")
    L.append(f"- std MSE = {std8:.2f}")
    L.append(f"- min MSE = {min8:.2f} (seed {sorted_seeds[0]})")
    L.append(f"- max MSE = {max8:.2f} (seed {sorted_seeds[-1]})")
    L.append("")
    L.append("## 3. 3-seed mean vs 8-seed mean")
    L.append("")
    L.append(f"- 3-seed mean (42/123/2026) = {mean3:.2f}")
    L.append(f"- 8-seed mean (all) = {mean8:.2f}")
    L.append(f"- 8-seed median = {median8:.2f}")
    L.append("")
    L.append("## 4. seed42 / other7 ratio")
    L.append("")
    L.append(f"- other7 mean MSE = {other7_mean:.2f}")
    L.append(f"- other7 median MSE = {other7_median:.2f}")
    L.append(f"- ratio42 (seed42 / other7_mean) = {ratio42:.3f}")
    L.append(f"- ratio42_median (seed42 / other7_median) = {ratio42_median:.3f}")
    L.append("")
    L.append("## 5. MSE distribution counts")
    L.append("")
    L.append(f"- seeds MSE < 40: {len(below40)} ({frac_below40:.2%})  -> {below40}")
    L.append(f"- seeds MSE 40-60: {len(between40_60)}  -> {between40_60}")
    L.append(f"- seeds MSE > 60: {len(above60)} ({frac_above60:.2%})  -> {above60}")
    L.append("")
    L.append(f"## 6. pattern = **{pattern}**")
    L.append("")
    L.append(f"- seed42_likely_outlier = {seed42_likely_outlier}")
    L.append(f"- training_instability_detected = {training_instability_detected}")
    L.append(f"- new seeds with MSE > 70: {new_above70 or 'none'}")
    L.append("")
    L.append("【结论】")
    L.append("")
    L.append(f"A. seed42 是否大概率是偶发异常？ "
             f"{'是' if seed42_likely_outlier else '否'}（ratio42={ratio42:.2f}，"
             f"新增 seed 中 MSE>70 的有 {len(new_above70)} 个）")
    L.append(f"B. Critic v6 是否存在系统性训练不稳定？ "
             f"{'是' if training_instability_detected else '否'}（{len(above60)}/{len(ALL_SEEDS)} "
             f"个 seed MSE>60）")
    L.append(f"C. 大多数 seed 的典型 MSE 大约是多少？ 约 {median8:.2f}（median），"
             f"{len(below40)}/{len(ALL_SEEDS)} 个 seed < 40")
    L.append(f"D. 是否值得进一步查 seed42 的训练曲线？ "
             f"{'是' if (ratio42 > 1.5) else '否'}（seed42 MSE={mse_by_seed[42]:.2f} vs "
             f"other7 median={other7_median:.2f}）")
    L.append("")
    L.append("critic_modified = false")
    L.append("network_modified = false")
    L.append("hyperparameters_modified = false")
    L.append("test_used = false")
    L.append("existing_seeds_replaced = false")
    report = "\n".join(L)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("\n" + report)


if __name__ == "__main__":
    main()
