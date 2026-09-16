#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence Critic v1 (Solar) — TRAIN-only retrieval evidence to select expert priors.

NOT a Learned Critic. No RF/MLP/Ridge training. Critic v6.1 frozen. TEST never read.

Pipeline:
  1. Reuse the frozen Solar pipeline (multidataset series/window_indices, context=512,
     horizon=96, residualize-Pearson shape, Top-K=5 mean transfer).
  2. Read all existing Solar expert priors (expert_round_1.json) + frozen v6.1 decisions.
  3. Semantic gate (observable / executable / compilable) — only "can we verify", never
     predicts whether a prior lowers MSE.
  4. TRAIN-only evidence: TRAIN_A = memory bank, TRAIN_B = eval queries. For each
     semantic-pass prior, run Shape+Prior_i retrieval vs Shape-only; evidence decision =
     ACCEPT iff relative_improvement > epsilon (epsilon=0.0, fixed).
  5. VAL (never used for selection): A Shape-only / B Raw / C v6.1 / D Evidence / E Random.
  6. TRAIN->VAL correlation (Pearson + Spearman) of per-prior relative_improvement.

Retrieval mechanism (non-parametric, documented): the prior's soft activation vector
A(w) (from the compiled primitives) gates the shape score by regime agreement
score(q,b) = residual_pearson(q,b) * (1 - mean_j |A_q[j] - A_b[j]|). Same mechanism for
single prior (evidence) and multi prior (VAL), so B/C/D differ only in WHICH priors.

Outputs under results/evidence_critic_solar_v1/.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from run_solar_same_val_baselines import residualize, shape_retrieval, mse, mae, build_windows  # noqa: E402
from run_etth1_prediction_validation import compute_training_stats, build_activation_matrix  # noqa: E402

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
CONTEXT, HORIZON = 512, 96
TOPK = 5
EPSILON = 0.0          # fixed; never tuned on VAL
N_RANDOM = 20
SEED_BASE = 0

DATA_ROOT = Path("results/multidataset/solar")
PRIOR_PATH = Path("results/cross_dataset_validation/solar/critic/expert_round_1.json")
V61_AUDIT_PATH = Path("results/learned_critic_dataset_v1/solar/audit.json")
OUT_ROOT = Path("results/evidence_critic_solar_v1")


# --------------------------------------------------------------------------- #
# semantic gate + prior loading
# --------------------------------------------------------------------------- #
def semantic_gate(prior: dict):
    """observable/executable/compilable -> (pass, reason). Only checks verifiability."""
    compiled = prior.get("compiled", {})
    prims = compiled.get("primitives", [])
    status = compiled.get("status", "")
    if not prims:
        return False, "no compiled primitives"
    if status != "COMPILED":
        un = compiled.get("uncompilable", [])
        feats = ", ".join(u.get("feature", "?") for u in un) or "?"
        return False, f"compiler status={status} (uncompilable: {feats})"
    feats = [p.get("feature", "?") for p in prims]
    for f in feats:
        if f not in ("seasonal_level", "hour_of_day", "day_of_week", "working_day",
                     "recent_level", "recent_slope_magnitude", "recent_diff_magnitude",
                     "recent_volatility", "recent_autocorr"):
            return False, f"feature '{f}' not observable from solar"
    return True, ""


def load_priors():
    priors = json.loads(PRIOR_PATH.read_text(encoding="utf-8"))
    # frozen v6.1 decisions (recorded only; NOT used for evidence decision)
    v61 = {}
    if V61_AUDIT_PATH.exists():
        for x in json.loads(V61_AUDIT_PATH.read_text(encoding="utf-8")):
            v61[x["expert_id"]] = x["v6_1_final_decision"]
    out = []
    for e in priors:
        pid = e.get("expert_id", e.get("id"))
        ok, reason = semantic_gate(e)
        out.append({
            "prior_id": pid,
            "prior_text": e.get("claim", "") or e.get("name", ""),
            "source": str(PRIOR_PATH),
            "v6_1_decision": v61.get(pid, "UNKNOWN"),
            "compiled": e.get("compiled", {}),
            "semantic_pass": ok,
            "semantic_reject_reason": reason if not ok else "",
            "compile_failed": (not ok),
        })
    return out


# --------------------------------------------------------------------------- #
# gated retrieval (shape * regime-agreement)
# --------------------------------------------------------------------------- #
def gated_retrieval(Xq, Xbank, Ybank, Aq, Abank, topk=TOPK, q_chunk=128, bank_chunk=2048):
    Q = residualize(Xq)
    B = residualize(Xbank)
    L = Q.shape[1]
    n_priors = Aq.shape[1]
    preds = []
    for qs in range(0, len(Q), q_chunk):
        q = Q[qs:qs + q_chunk]
        aq = Aq[qs:qs + q_chunk]
        nq = len(q)
        best_scores = np.full((nq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nq, topk), -1, dtype=np.int64)
        for bs in range(0, len(B), bank_chunk):
            b = B[bs:bs + bank_chunk]
            ab = Abank[bs:bs + bank_chunk]
            shape = (q @ b.T) / L
            # regime agreement gate (mean over priors, soft)
            gate = 1.0 - np.abs(aq[:, None, :] - ab[None, :, :]).mean(axis=-1)
            scores = shape * gate
            kk = min(topk, scores.shape[1])
            part = np.argpartition(scores, -kk, axis=1)[:, -kk:]
            local_scores = np.take_along_axis(scores, part, axis=1)
            local_idx = part + bs
            comb_scores = np.concatenate([best_scores, local_scores], axis=1)
            comb_idx = np.concatenate([best_idx, local_idx], axis=1)
            part2 = np.argpartition(comb_scores, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(comb_scores, part2, axis=1)
            best_idx = np.take_along_axis(comb_idx, part2, axis=1)
        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_idx = np.take_along_axis(best_idx, order, axis=1)
        preds.append(Ybank[best_idx].mean(axis=1))
    return np.concatenate(preds, axis=0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Evidence Critic v1 (Solar): TRAIN-only evidence", flush=True)

    # ---- data ----
    df = pd.read_csv(DATA_ROOT / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(DATA_ROOT / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    # TEST (idx["test"]) intentionally never read

    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]      # TRAIN_A = memory
    fit_starts = train_starts[cut:]       # TRAIN_B = evidence eval queries

    X_bank, Y_bank = build_windows(series, bank_starts, CONTEXT, HORIZON)
    X_fit, Y_fit = build_windows(series, fit_starts, CONTEXT, HORIZON)
    X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)

    bank_ts = [dates[s + CONTEXT] for s in bank_starts]
    fit_ts = [dates[s + CONTEXT] for s in fit_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]
    print(f"bank={len(X_bank)} fit={len(X_fit)} val={len(X_val)} (test NOT read)", flush=True)

    # ---- priors + semantic gate ----
    priors = load_priors()
    sem = [p for p in priors if p["semantic_pass"]]
    print(f"priors total={len(priors)} semantic_pass={len(sem)}", flush=True)
    for p in priors:
        if not p["semantic_pass"]:
            print(f"  [SEM-FAIL] {p['prior_id']}: {p['semantic_reject_reason']}", flush=True)

    # ---- activation matrix (soft, from compiled primitives) ----
    stats = compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))
    # build activation for the semantic-pass priors only
    sem_experts = [{"compiled": p["compiled"]} for p in sem]
    A_bank = build_activation_matrix(sem_experts, X_bank, bank_ts, stats)
    A_fit = build_activation_matrix(sem_experts, X_fit, fit_ts, stats)
    A_val = build_activation_matrix(sem_experts, X_val, val_ts, stats)
    print(f"activation matrices: bank={A_bank.shape} fit={A_fit.shape} val={A_val.shape}", flush=True)

    # ---- TRAIN evidence (Shape-only baseline + per-prior) ----
    print("\n[TRAIN evidence] Shape-only baseline on TRAIN_B ...", flush=True)
    base_fit = shape_retrieval(X_fit, X_bank, Y_bank, topk=TOPK)
    baseline_mse = mse(Y_fit, base_fit)
    baseline_mae = mae(Y_fit, base_fit)
    print(f"  baseline_mse={baseline_mse:.4f} baseline_mae={baseline_mae:.4f}", flush=True)

    evidence_rows = []
    for j, p in enumerate(sem):
        aq = A_fit[:, j:j + 1]
        ab = A_bank[:, j:j + 1]
        pred = gated_retrieval(X_fit, X_bank, Y_bank, aq, ab, topk=TOPK)
        pm = mse(Y_fit, pred)
        rel = (baseline_mse - pm) / baseline_mse if baseline_mse else 0.0
        evidence_rows.append({
            "prior_id": p["prior_id"], "prior_text": p["prior_text"],
            "semantic_pass": True, "semantic_reject_reason": "",
            "v6_1_decision": p["v6_1_decision"],
            "baseline_mse": baseline_mse, "prior_mse": pm,
            "delta_mse": pm - baseline_mse, "relative_improvement": rel,
            "evidence_decision": "ACCEPT" if rel > EPSILON else "REJECT",
        })
        print(f"  {p['prior_id']}: rel_imp={rel:+.4f} -> {evidence_rows[-1]['evidence_decision']}", flush=True)

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df.to_csv(OUT_ROOT / "prior_evidence.csv", index=False, encoding="utf-8")

    # ---- VAL comparison (A/B/C/D/E), selection frozen from TRAIN ----
    print("\n[VAL comparison] selection frozen (no VAL-based tuning)", flush=True)
    shape_val_pred = shape_retrieval(X_val, X_bank, Y_bank, topk=TOPK)
    shape_val_mse = mse(Y_val, shape_val_pred)

    sem_idx = [j for j, p in enumerate(sem)]
    raw_mse, raw_mae = _val_gated(A_val, A_bank, sem_idx, X_val, X_bank, Y_val, Y_bank)

    v61_idx = [j for j, p in enumerate(sem) if p["v6_1_decision"] == "ACCEPT"]
    v61_mse, v61_mae = _val_gated(A_val, A_bank, v61_idx, X_val, X_bank, Y_val, Y_bank) if v61_idx else (float("nan"), float("nan"))

    ev_idx = [j for j, p in enumerate(sem) if evidence_df.set_index("prior_id").loc[p["prior_id"], "evidence_decision"] == "ACCEPT"]
    ev_mse, ev_mae = _val_gated(A_val, A_bank, ev_idx, X_val, X_bank, Y_val, Y_bank) if ev_idx else (float("nan"), float("nan"))

    # random control (same size as evidence accept, 20 seeds)
    n_ev = len(ev_idx)
    rng = random.Random(SEED_BASE)
    random_mses, random_maes = [], []
    for s in range(N_RANDOM):
        if n_ev == 0:
            random_mses.append(shape_val_mse)
            random_maes.append(mae(Y_val, shape_val_pred))
            continue
        ridx = rng.sample(sem_idx, n_ev)
        rm, ra = _val_gated(A_val, A_bank, ridx, X_val, X_bank, Y_val, Y_bank)
        random_mses.append(rm)
        random_maes.append(ra)
    random_mses = np.asarray(random_mses)
    random_maes = np.asarray(random_maes)

    val_rows = [
        {"method": "shape_only", "val_mse": shape_val_mse, "val_mae": mae(Y_val, shape_val_pred),
         "improvement_vs_shape": 0.0, "n_priors": 0},
        {"method": "raw_expert", "val_mse": raw_mse, "val_mae": raw_mae,
         "improvement_vs_shape": (shape_val_mse - raw_mse) / shape_val_mse, "n_priors": len(sem_idx)},
        {"method": "v6_1_expert", "val_mse": v61_mse, "val_mae": v61_mae,
         "improvement_vs_shape": (shape_val_mse - v61_mse) / shape_val_mse if v61_idx else float("nan"),
         "n_priors": len(v61_idx)},
        {"method": "evidence_expert", "val_mse": ev_mse, "val_mae": ev_mae,
         "improvement_vs_shape": (shape_val_mse - ev_mse) / shape_val_mse if ev_idx else float("nan"),
         "n_priors": n_ev},
        {"method": "random_control", "val_mse": float(np.mean(random_mses)), "val_mae": float(np.mean(random_maes)),
         "improvement_vs_shape": (shape_val_mse - np.mean(random_mses)) / shape_val_mse, "n_priors": n_ev},
    ]
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(OUT_ROOT / "val_method_comparison.csv", index=False, encoding="utf-8")
    rc_df = pd.DataFrame({"seed": range(N_RANDOM), "val_mse": random_mses,
                          "improvement_vs_shape": (shape_val_mse - random_mses) / shape_val_mse})
    rc_df.to_csv(OUT_ROOT / "random_control.csv", index=False, encoding="utf-8")

    # ---- TRAIN -> VAL correlation (per-prior) ----
    print("\n[TRAIN->VAL correlation] per-prior", flush=True)
    corr_rows = []
    for j, p in enumerate(sem):
        aq = A_val[:, j:j + 1]
        ab = A_bank[:, j:j + 1]
        vpred = gated_retrieval(X_val, X_bank, Y_bank, aq, ab, topk=TOPK)
        vm = mse(Y_val, vpred)
        val_rel = (shape_val_mse - vm) / shape_val_mse
        train_rel = evidence_df.set_index("prior_id").loc[p["prior_id"], "relative_improvement"]
        corr_rows.append({"prior_id": p["prior_id"], "train_relative_improvement": train_rel,
                          "val_relative_improvement": val_rel})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_ROOT / "train_val_prior_correlation.csv", index=False, encoding="utf-8")
    pear = float(corr_df[["train_relative_improvement", "val_relative_improvement"]].corr().iloc[0, 1]) \
        if len(corr_df) > 1 else float("nan")
    spear = float(corr_df[["train_relative_improvement", "val_relative_improvement"]].corr(method="spearman").iloc[0, 1]) \
        if len(corr_df) > 1 else float("nan")
    print(f"  Pearson={pear:.4f} Spearman={spear:.4f}", flush=True)

    # ---- config + report ----
    config = {
        "dataset": "solar", "context": CONTEXT, "horizon": HORIZON, "topk": TOPK,
        "epsilon": EPSILON, "n_random": N_RANDOM,
        "n_priors_total": len(priors), "n_semantic_pass": len(sem),
        "n_evidence_accept": int((evidence_df["evidence_decision"] == "ACCEPT").sum()),
        "n_v6_1_accept": len(v61_idx),
        "retrieval_mechanism": "residual_pearson * (1 - mean|A_q - A_b|) over selected priors; Top-K mean transfer",
        "train_val_split": "bank=first 80% train (TRAIN_A), fit=last 20% train (TRAIN_B)",
    }
    (OUT_ROOT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(evidence_df, val_df, rc_df, corr_df, config, shape_val_mse, pear, spear)


def _val_gated(A_val, A_bank, idxs, X_val, X_bank, Y_val, Y_bank):
    if not idxs:
        return float("nan"), float("nan")
    aq = A_val[:, idxs]
    ab = A_bank[:, idxs]
    pred = gated_retrieval(X_val, X_bank, Y_bank, aq, ab, topk=TOPK)
    return mse(Y_val, pred), mae(Y_val, pred)


def write_report(evidence_df, val_df, rc_df, corr_df, config, shape_mse, pear, spear):
    n_ev_acc = config["n_evidence_accept"]
    n_total = config["n_priors_total"]
    ev_val = val_df[val_df["method"] == "evidence_expert"]["val_mse"].iloc[0]
    raw_val = val_df[val_df["method"] == "raw_expert"]["val_mse"].iloc[0]
    v61_val = val_df[val_df["method"] == "v6_1_expert"]["val_mse"].iloc[0]
    rc_mean = rc_df["val_mse"].mean()
    rc_std = rc_df["val_mse"].std()

    # judgment: A (clear value) / B (some signal unstable) / C (no signal)
    ev_better = ev_val < shape_mse
    if ev_better and pear > 0.3:
        verdict = "A. Evidence Critic 有明显价值"
    elif ev_better or (pear > 0.0 and not np.isnan(pear)):
        verdict = "B. 有一定信号但不稳定"
    else:
        verdict = "C. Evidence Critic 无有效信号"

    L = ["# Evidence Critic v1 (Solar)", "", "## 结果", ""]
    L.append(f"1. Shape VAL MSE = {shape_mse:.4f}")
    L.append(f"2. Raw Expert VAL MSE = {raw_val:.4f}")
    L.append(f"3. v6.1 VAL MSE = {v61_val:.4f}")
    L.append(f"4. Evidence Expert VAL MSE = {ev_val:.4f}")
    L.append(f"5. Random Control mean±std = {rc_mean:.4f} ± {rc_std:.4f}")
    L.append(f"6. Evidence ACCEPT 数量 / 总数量 = {n_ev_acc} / {n_total}")
    L.append(f"7. TRAIN→VAL Pearson = {pear:.4f}")
    L.append(f"8. TRAIN→VAL Spearman = {spear:.4f}")
    L.append("")
    L.append("### v6.1 ACCEPT priors 的 TRAIN evidence improvement 分布")
    L.append(evidence_df[evidence_df["v6_1_decision"] == "ACCEPT"]
             [["prior_id", "relative_improvement", "evidence_decision"]].to_markdown(index=False))
    L.append("")
    L.append("### v6.1 REJECT priors 的 TRAIN evidence improvement 分布")
    L.append(evidence_df[evidence_df["v6_1_decision"] == "REJECT"]
             [["prior_id", "relative_improvement", "evidence_decision"]].to_markdown(index=False))
    L.append("")
    L.append("### 最有效 Top-5 prior")
    L.append(evidence_df.sort_values("relative_improvement", ascending=False).head(5)
             [["prior_id", "relative_improvement", "evidence_decision"]].to_markdown(index=False))
    L.append("")
    L.append("### 最有害 Bottom-5 prior")
    L.append(evidence_df.sort_values("relative_improvement").head(5)
             [["prior_id", "relative_improvement", "evidence_decision"]].to_markdown(index=False))
    L.append("")
    L.append("## 结论")
    L.append("")
    L.append(verdict)
    (OUT_ROOT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)

    print("\n===== AUDIT =====", flush=True)
    print("TEST_USED = FALSE", flush=True)
    print("LEARNED_CRITIC_USED = FALSE", flush=True)
    print("CRITIC_V6_1_FROZEN = TRUE", flush=True)
    print("EVIDENCE_SPLIT = TRAIN_ONLY", flush=True)
    print("VAL_USED_FOR_SELECTION = FALSE", flush=True)


if __name__ == "__main__":
    main()
