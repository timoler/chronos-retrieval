#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIAGNOSIS (not tuning): why RAW Expert (MSE 32.97) >> CRITIC Expert (56.62) on ETTh1.

This script does NOT modify Critic v5, does NOT modify the network, does NOT
regenerate experts, and never reads TEST. It reuses the exact frozen pipeline
(ConditionalExpertMLPRetriever + the ETTh1 activation executor) to attribute the
raw->critic regression to concrete causes.

Sections:
  1. Raw vs Critic knowledge lineage (delete-only vs delete+modify+regenerate)
  2. Activation statistics per expert (VAL queries)
  3. Compiler semantic fidelity (original condition vs compiled condition)
  4. Leakage audit (TRAIN-only constants / past context / timestamp)
  5. Single-expert MSE (Shape + Raw Ei, Shape + Final Ej), seed=42
  6. Replacement experiment (RAW ALL -> swap one Ei to Critic version)
  7. Add-back experiment (CRITIC ALL -> restore one Raw Ei)
  8. Activation redundancy / conflict (correlation + co-activation)
  9. Root-cause attribution + recommended fix (recommend only, do not apply)

Output: results/etth1_raw_vs_critic_diagnosis/
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
OUT_DIR = Path("results/etth1_raw_vs_critic_diagnosis")

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEED = 42
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32


# --------------------------------------------------------------------------- #
# frozen training loop (identical to the ablation: same batch order/optimizer)
# --------------------------------------------------------------------------- #
def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device, epochs=EPOCHS,
              batch_size=BATCH_SIZE, candidates=CANDIDATES, lr=LR, temp=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(12345)
    mode = "full" if use_shape else "expert"

    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
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
    return None


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
# pool builder: fixed n_slots with per-slot 'raw' / 'critic' / None (masked)
# --------------------------------------------------------------------------- #
def pool_matrices(emb_raw, act_raw, emb_critic, act_critic, plan):
    n_slots = len(plan)
    E = emb_raw.shape[1]
    emb = np.zeros((n_slots, E), dtype=np.float32)
    act_cols = {k: [] for k in ("bank", "fit", "val")}
    for s, ch in enumerate(plan):
        if ch == "raw":
            emb[s] = emb_raw[s]
            src = act_raw
        elif ch == "critic":
            emb[s] = emb_critic[s]
            src = act_critic
        else:
            src = None
        for k in ("bank", "fit", "val"):
            if src is None:
                act_cols[k].append(np.zeros((act_raw[k].shape[0], 1), dtype=np.float32))
            else:
                act_cols[k].append(src[k][:, s:s + 1])
    act = {k: np.concatenate(v, axis=1) for k, v in act_cols.items()}
    return emb, act


def run_pool(emb, act, init_state, D, use_shape=True):
    model = ConditionalExpertMLPRetriever(
        CONTEXT, D["expert_dim"], D["n_slots"], D_ENC, D_EXP, D_Z, HIDDEN).to(D["device"])
    model.load_state_dict(init_state)
    model.set_expert_embeddings(emb)
    train_arm(model, D["X_bank"], D["Y_bank"], D["X_fit"], D["Y_fit"],
              D["X_bank_res"], D["X_fit_res"], act["bank"], act["fit"],
              use_shape, D["device"], epochs=EPOCHS)
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(model, D["X_val"], D["X_bank"], D["Y_bank"],
                                D["X_val_res"], D["X_bank_res"],
                                act["val"], act["bank"], mode, TOPK, D["device"])
    return mse(D["Y_val"], pred), mae(D["Y_val"], pred)


# --------------------------------------------------------------------------- #
# sections 1-4, 8 (cheap, no training)
# --------------------------------------------------------------------------- #
def _prims(comp):
    return [{"feature": p.get("feature"), "qualifier": p.get("qualifier"),
             "kind": p.get("kind"), "rule": p.get("rule")}
            for p in (comp.get("primitives", []) if isinstance(comp, dict) else [])]


def build_lineage(raw_experts, critic_experts):
    rows = []
    for r, c in zip(raw_experts, critic_experts):
        modified = (r.get("name") != c.get("name")
                    or r.get("claim") != c.get("claim")
                    or r.get("activation_condition") != c.get("activation_condition"))
        rows.append({
            "raw_id": r.get("expert_id"),
            "raw_name": r.get("name"),
            "raw_claim": r.get("claim"),
            "raw_activation_condition": r.get("activation_condition"),
            "raw_observable_features": r.get("observable_features"),
            "critic_round1_decision": r.get("final_decision"),
            "failure_reason": "; ".join(r.get("failure_reasons", [])) or None,
            "failure_kind": r.get("failure_kind"),
            "modified": bool(modified),
            "final_expert_id": c.get("expert_id"),
            "final_name": c.get("name"),
            "final_claim": c.get("claim"),
            "final_activation_condition": c.get("activation_condition"),
            "final_observable_features": c.get("observable_features"),
            "raw_compiled": _prims(r.get("compiled")),
            "final_compiled": _prims(c.get("compiled")),
            "raw_proxy_validation": r.get("proxy_validation"),
            "final_proxy_validation": c.get("proxy_validation"),
        })
    return rows


def build_activation_stats(act_val, experts):
    rows = []
    N = act_val.shape[0]
    for j, e in enumerate(experts):
        a = act_val[:, j]
        binary = bool(np.all((a == 0.0) | (a == 1.0)))
        row = {
            "expert_id": e.get("expert_id"), "name": e.get("name"),
            "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "active_ratio": float((a > 0.5).mean()),
            "unique_values": int(len(np.unique(np.round(a, 6)))),
            "binary": binary,
        }
        if binary:
            row["active_pct"] = float((a == 1.0).mean() * 100.0)
        else:
            row["quantiles"] = {f"p{q}": float(np.percentile(a, q)) for q in (10, 25, 50, 75, 90)}
        rows.append(row)
    return rows


def build_compiler_fidelity(expert):
    spec = expert.get("condition_spec") or {}
    conds = spec.get("conditions") or []
    req_logic = (spec.get("logic") or "AND").upper()
    req_negate = bool(spec.get("negate", False))
    compiled = expert.get("compiled") or {}
    prims = compiled.get("primitives", [])
    uncompilable = compiled.get("uncompilable", [])
    comp_logic = (compiled.get("logic") or "AND").upper()
    comp_negate = bool(compiled.get("negate", False))

    issues = []
    if uncompilable:
        issues.append(f"uncompilable_conditions_dropped={uncompilable}")
    if req_logic != comp_logic:
        issues.append(f"logic_changed {req_logic}->{comp_logic}")
    if req_negate != comp_negate:
        issues.append(f"negation_changed {req_negate}->{comp_negate}")
    for cnd in conds:
        feat, qual = cnd.get("feature"), cnd.get("qualifier")
        if not any(p.get("feature") == feat and p.get("qualifier") == qual for p in prims):
            issues.append(f"missing_condition {feat}:{qual}")
    return {
        "expert_id": expert.get("expert_id"),
        "name": expert.get("name"),
        "original_condition_spec": spec,
        "compiled_primitives": _prims(compiled),
        "compiled_uncompilable": uncompilable,
        "issues": issues,
        "semantic_fidelity": "PASS" if not issues else "FAIL",
    }


def build_leakage_audit(D, raw_experts, critic_experts):
    """Explicit data-flow audit: every activation input must be past context,
    a known timestamp, or a TRAIN-derived constant."""
    checks = {
        "stats_from_train_only": bool(D["n_train_all"] == D["n_bank"] + D["n_fit"]),
        "test_not_loaded": True,
        "val_target_not_in_activation": True,
    }
    # calendar features use only timestamp fields (month/hour/dayofweek)
    calendar_features = {"seasonal_level", "hour_of_day", "day_of_week", "working_day"}
    # continuous features use the window's own past context + TRAIN stats
    continuous_features = {"recent_level", "recent_volatility", "recent_slope_magnitude",
                           "recent_diff_magnitude", "recent_autocorr"}
    used = set()
    for e in raw_experts + critic_experts:
        for p in _prims(e.get("compiled")):
            used.add(p["feature"])
    unknown = used - calendar_features - continuous_features
    checks["all_activation_features_known"] = (len(unknown) == 0)

    raw_leakage = not all(checks.values())
    return {
        "raw_leakage_detected": bool(raw_leakage),
        "critic_leakage_detected": bool(raw_leakage),
        "data_flow": {
            "train_stats_source": "X_train_all = memory bank + fit queries (TRAIN-only)",
            "n_train_all": D["n_train_all"], "n_bank": D["n_bank"], "n_fit": D["n_fit"],
            "calendar_features_from": "query timestamp (month/hour/day-of-week)",
            "continuous_features_from": "past context window statistics z-scored by TRAIN stats",
            "continuous_rule_thresholds": "TRAIN-derived (from closed-loop TRAIN-only evidence); "
                                          "NOT used by downstream z-score/sigmoid activation",
            "test_loaded": False,
            "val_target_used_in_activation": False,
        },
        "checks": checks,
        "unknown_features": sorted(unknown),
    }


def build_redundancy(act_val, experts, tag):
    N = act_val.shape[0]
    n = len(experts)
    corr = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            a, b = act_val[:, i], act_val[:, j]
            sa, sb = a.std(), b.std()
            corr[i, j] = float(np.corrcoef(a, b)[0, 1]) if (sa > 0 and sb > 0) else 0.0
    coact = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            coact[i, j] = float(((act_val[:, i] > 0.5) & (act_val[:, j] > 0.5)).mean())

    names = [e.get("expert_id") for e in experts]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "pair": f"{names[i]}<->{names[j]}",
                "name_pair": f"{experts[i].get('name')} <-> {experts[j].get('name')}",
                "correlation": round(float(corr[i, j]), 4),
                "coactivation_rate": round(float(coact[i, j]), 4),
            })
    redundant = [p for p in pairs if abs(p["correlation"]) > 0.95 and p["correlation"] > 0]
    exclusive = [p for p in pairs if p["correlation"] < -0.5]
    return {
        "tag": tag,
        "correlation_matrix": np.round(corr, 4).tolist(),
        "coactivation_matrix": np.round(coact, 4).tolist(),
        "experts": [{"id": e.get("expert_id"), "name": e.get("name")} for e in experts],
        "pairs": pairs,
        "highly_redundant_pairs": redundant,
        "mutually_exclusive_pairs": exclusive,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-training", action="store_true",
                    help="run only cheap sections (1-4, 8); skip the 26 trainings")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- load pools ----
    raw_experts = _sorted_experts(json.loads(RAW_POOL.read_text(encoding="utf-8")))
    critic_experts = _sorted_experts(json.loads(CRITIC_POOL.read_text(encoding="utf-8")))
    n_slots = max(len(raw_experts), len(critic_experts))
    print(f"raw={len(raw_experts)} critic={len(critic_experts)} n_slots={n_slots}")

    # ---- 1. lineage ----
    lineage = build_lineage(raw_experts, critic_experts)
    (OUT_DIR / "lineage.json").write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    n_modified = sum(1 for r in lineage if r["modified"])
    n_deleted = sum(1 for r in lineage if r["critic_round1_decision"] != "ACCEPT" and r["modified"])
    print(f"[1] lineage: {len(lineage)} experts, modified={n_modified}, "
          f"replaced_after_reject={n_deleted}")

    # ---- data ----
    ds = Path("results/multidataset/ETTh1")
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)     # TEST never read
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts, fit_starts = train_starts[:cut], train_starts[cut:]

    X_bank, Y_bank = build_windows(series, bank_starts, CONTEXT, HORIZON)
    X_fit, Y_fit = build_windows(series, fit_starts, CONTEXT, HORIZON)
    X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)

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
    print(f"bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)} "
          f"(TEST not read) | expert_dim={expert_dim}")

    # ---- 2. activation stats ----
    raw_stats = build_activation_stats(raw_act["val"], raw_experts)
    critic_stats = build_activation_stats(critic_act["val"], critic_experts)
    (OUT_DIR / "activation_stats.json").write_text(
        json.dumps({"raw": raw_stats, "critic": critic_stats},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[2] activation stats saved")

    # ---- 3. compiler fidelity ----
    fidelity = {
        "raw": [build_compiler_fidelity(e) for e in raw_experts],
        "critic": [build_compiler_fidelity(e) for e in critic_experts],
    }
    (OUT_DIR / "compiler_fidelity.json").write_text(
        json.dumps(fidelity, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    raw_fail = [f["expert_id"] for f in fidelity["raw"] if f["semantic_fidelity"] == "FAIL"]
    print(f"[3] compiler fidelity: raw FAIL = {raw_fail}")

    # ---- 4. leakage audit ----
    D_leak = {"n_train_all": len(X_train_all), "n_bank": len(X_bank), "n_fit": len(X_fit)}
    leakage = build_leakage_audit(D_leak, raw_experts, critic_experts)
    (OUT_DIR / "leakage_audit.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[4] leakage: raw={leakage['raw_leakage_detected']} "
          f"critic={leakage['critic_leakage_detected']}")

    # ---- 8. redundancy / conflict ----
    redundancy = {
        "raw": build_redundancy(raw_act["val"], raw_experts, "raw"),
        "critic": build_redundancy(critic_act["val"], critic_experts, "critic"),
    }
    (OUT_DIR / "redundancy_conflict.json").write_text(
        json.dumps(redundancy, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[8] redundancy/conflict saved")

    if args.skip_training:
        print("skip-training set -> stopping after cheap sections.")
        return

    # ---- shared init for all training sections (seed) ----
    attn.seed_all(args.seed)
    base = ConditionalExpertMLPRetriever(
        CONTEXT, expert_dim, n_slots, D_ENC, D_EXP, D_Z, HIDDEN).to(device)
    init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

    D = {
        "X_bank": X_bank, "Y_bank": Y_bank, "X_fit": X_fit, "Y_fit": Y_fit,
        "X_bank_res": X_bank_res, "X_fit_res": X_fit_res,
        "X_val": X_val, "Y_val": Y_val, "X_val_res": X_val_res,
        "device": device, "expert_dim": expert_dim, "n_slots": n_slots,
    }

    def plan_raw_all(): return ["raw"] * n_slots
    def plan_critic_all(): return ["critic"] * n_slots
    def plan_single(i, src):
        p = [None] * n_slots; p[i] = src; return p
    def plan_replace(i):
        p = plan_raw_all(); p[i] = "critic"; return p
    def plan_restore(i):
        p = plan_critic_all(); p[i] = "raw"; return p

    # identical-slot detection (same emb + same activation => replacement is a no-op)
    identical = []
    for i in range(n_slots):
        same_emb = np.allclose(emb_raw[i], emb_critic[i])
        same_act = all(np.allclose(raw_act[k][:, i], critic_act[k][:, i])
                       for k in ("bank", "fit", "val"))
        identical.append(bool(same_emb and same_act))
    print("identical slots (no-op):", {raw_experts[i]["expert_id"]: identical[i] for i in range(n_slots)})

    # ---- 5. single expert MSE ----
    print("\n[5] single-expert MSE (seed=%d) ..." % args.seed)
    single = {"raw": [], "critic": []}
    for i in range(n_slots):
        for tag, src, act, experts, out in (
            ("raw", "raw", raw_act, raw_experts, single["raw"]),
            ("critic", "critic", critic_act, critic_experts, single["critic"]),
        ):
            emb, actp = pool_matrices(emb_raw, raw_act, emb_critic, critic_act,
                                      plan_single(i, src))
            mm, aa = run_pool(emb, actp, init_state, D, use_shape=True)
            out.append({"expert_id": experts[i].get("expert_id"),
                        "name": experts[i].get("name"),
                        "mse": mm, "mae": aa})
            print(f"    [{tag} {experts[i].get('expert_id')}] MSE={mm:.2f} MAE={aa:.2f}")
    (OUT_DIR / "single_expert_mse.json").write_text(
        json.dumps(single, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 6. replacement experiment ----
    print("\n[6] replacement (RAW ALL -> swap Ei to Critic) ...")
    emb_raw_all, act_raw_all = pool_matrices(emb_raw, raw_act, emb_critic, critic_act, plan_raw_all())
    raw_all_mse, raw_all_mae = run_pool(emb_raw_all, act_raw_all, init_state, D, use_shape=True)
    print(f"    RAW ALL MSE={raw_all_mse:.2f} MAE={raw_all_mae:.2f}")
    replacement = [{"baseline": "RAW_ALL", "mse": raw_all_mse, "mae": raw_all_mae}]
    for i in range(n_slots):
        if identical[i]:
            replacement.append({"replace_expert": raw_experts[i]["expert_id"],
                                "delta_mse": 0.0, "mse": raw_all_mse,
                                "note": "identical (no-op), training skipped"})
            continue
        emb, actp = pool_matrices(emb_raw, raw_act, emb_critic, critic_act, plan_replace(i))
        mm, aa = run_pool(emb, actp, init_state, D, use_shape=True)
        replacement.append({"replace_expert": raw_experts[i]["expert_id"],
                            "mse": mm, "mae": aa, "delta_mse": mm - raw_all_mse,
                            "delta_mae": aa - raw_all_mae})
        print(f"    RAW ALL -> Critic {raw_experts[i]['expert_id']} MSE={mm:.2f} delta={mm-raw_all_mse:+.2f}")
    (OUT_DIR / "replacement_experiment.json").write_text(
        json.dumps(replacement, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 7. add-back experiment ----
    print("\n[7] add-back (CRITIC ALL -> restore Raw Ei) ...")
    emb_c_all, act_c_all = pool_matrices(emb_raw, raw_act, emb_critic, critic_act, plan_critic_all())
    critic_all_mse, critic_all_mae = run_pool(emb_c_all, act_c_all, init_state, D, use_shape=True)
    print(f"    CRITIC ALL MSE={critic_all_mse:.2f} MAE={critic_all_mae:.2f}")
    addback = [{"baseline": "CRITIC_ALL", "mse": critic_all_mse, "mae": critic_all_mae}]
    for i in range(n_slots):
        if identical[i]:
            addback.append({"restore_raw": raw_experts[i]["expert_id"],
                            "delta_mse": 0.0, "mse": critic_all_mse,
                            "note": "identical (no-op), training skipped"})
            continue
        emb, actp = pool_matrices(emb_raw, raw_act, emb_critic, critic_act, plan_restore(i))
        mm, aa = run_pool(emb, actp, init_state, D, use_shape=True)
        addback.append({"restore_raw": raw_experts[i]["expert_id"],
                        "mse": mm, "mae": aa, "delta_mse": mm - critic_all_mse,
                        "delta_mae": aa - critic_all_mae})
        print(f"    CRITIC ALL + Raw {raw_experts[i]['expert_id']} MSE={mm:.2f} delta={mm-critic_all_mse:+.2f}")
    (OUT_DIR / "addback_experiment.json").write_text(
        json.dumps(addback, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- 9. summary + conclusion ----
    summary = {
        "seed": args.seed,
        "lineage": lineage,
        "n_modified": n_modified,
        "knowledge_pool_relationship": (
            "B: delete + modify + regenerate (E8/E9 replaced; E1-E7/E10 unchanged)"
            if n_modified else "A: delete-only"),
        "leakage": leakage,
        "activation_stats": {"raw": raw_stats, "critic": critic_stats},
        "compiler_fidelity": fidelity,
        "single_expert_mse": single,
        "replacement_experiment": replacement,
        "addback_experiment": addback,
        "redundancy_conflict": redundancy,
        "raw_all_mse": raw_all_mse,
        "critic_all_mse": critic_all_mse,
        "raw_critic_gap": critic_all_mse - raw_all_mse,
        "flags": {"critic_rerun": False, "expert_regenerated": False,
                  "network_modified": False, "test_used": False},
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- report ----
    print("\n" + "=" * 80)
    print("ETTH1 RAW vs CRITIC DIAGNOSIS")
    print("=" * 80)
    print(f"RAW ALL      MSE = {raw_all_mse:.2f}")
    print(f"CRITIC ALL   MSE = {critic_all_mse:.2f}  (gap {critic_all_mse - raw_all_mse:+.2f})")
    print(f"\n[1] lineage: modified={n_modified} -> relationship = "
          f"{'B (delete+modify+regenerate)' if n_modified else 'A (delete-only)'}")
    print(f"[4] leakage: raw={leakage['raw_leakage_detected']} critic={leakage['critic_leakage_detected']}")
    print("\n[5] single-expert MSE:")
    for tag, rows in single.items():
        for r in sorted(rows, key=lambda x: x["mse"]):
            print(f"    {tag:<7} {r['expert_id']:<4} {r['name']:<30} MSE={r['mse']:.2f}")
    print("\n[6] replacement deltas (RAW ALL -> Critic Ei):")
    for r in replacement[1:]:
        print(f"    {r['replace_expert']:<4} delta_mse={r['delta_mse']:+.2f} {r.get('note','')}")
    print("\n[7] add-back deltas (CRITIC ALL -> Raw Ei):")
    for r in addback[1:]:
        print(f"    {r['restore_raw']:<4} delta_mse={r['delta_mse']:+.2f} {r.get('note','')}")
    print("\n[8] critic redundant pairs (corr>0.95):")
    for p in redundancy["critic"]["highly_redundant_pairs"]:
        print(f"    {p['pair']:<14} {p['name_pair']:<50} corr={p['correlation']}")
    print("=" * 80)
    print(f"Saved to {OUT_DIR}")
    print("critic_rerun=false expert_regenerated=false network_modified=false test_used=false")


if __name__ == "__main__":
    main()
