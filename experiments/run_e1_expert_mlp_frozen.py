#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 (NightZeroPersistence) conditional-expert MLP — FROZEN reuse experiment.

Goal: verify whether the single expert that passed Critic v4 (E1) gives the
Expert-MLP an independent predictive gain over the CNN-only MLP.

Everything below is REUSED from existing modules — no new architecture, no new
module, no new loss, no Attention, no mean-pool, no softmax gating. Only the
accepted-expert set is frozen to ["E1"].

Arms (identical TRAIN bank/queries, VAL queries, candidate pool, optimizer, lr,
epochs, seed, Top-K=5, mean transfer, metric — only the Expert input changes):
  A. Shape-only           = residual Pearson retrieval baseline
  B. CNN-only MLP         = original ExpertEmbeddingMLPRetriever(use_expert=False)
  C. CNN + E1             = ConditionalExpertMLPRetriever(n_experts=1), a_E1=e_E1
  D. B-param-control      = same as C but random (unit-norm) embedding instead of
                            E1's text embedding; identical trainable param count.

E1 activation (reused from the Critic v4 compiler):
    a_E1(q) = daylight_level(q) = TRAIN_hourly_mean_OT[hour(q)] / max(TRAIN_hourly_mean_OT)
    TRAIN covers 24/24 hours -> no unseen-hour fallback.

TEST is never read. Critic v4 is not modified. Experts are not regenerated.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_expert_attention_retrieval_v2 as attn                       # QueryEncoder / seed_all
from run_expert_embedding_mlp import (                                 # original CNN-only MLP
    ExpertEmbeddingMLPRetriever,
    build_windows,
    mae,
    mse,
    retrieve_topk_mean,
    train,
)
from run_solar_same_val_baselines import residualize, shape_retrieval  # shape baseline
from run_conditional_expert_mlp_v4 import (                            # conditional-expert path
    ConditionalExpertMLPRetriever,
    build_activation_executor,
    retrieve_conditional,
    train_conditional,
)
from sentence_transformers import SentenceTransformer


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/multidataset"))
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--d-enc", type=int, default=128)
    ap.add_argument("--d-exp", type=int, default=128)
    ap.add_argument("--d-z", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-bank", type=int, default=25000)
    ap.add_argument("--max-train-query", type=int, default=6000)
    ap.add_argument("--n-diag", type=int, default=100)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--audit", type=Path, default=Path("results/solar_agent_v4_audit.json"))
    ap.add_argument("--critic-audit", type=Path,
                    default=Path("results/expert_knowledge_critic_audit.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/multidataset/solar/e1_expert_mlp_frozen"))
    args = ap.parse_args()

    attn.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ------------------------------------------------------------------ #
    # 1. E1 raw text + activation from the frozen pipeline; confirm that the
    #    refactored Critic's final review accepts ONLY E1 (E2..E6 unsupported).
    # ------------------------------------------------------------------ #
    # E1 raw text / activation_condition / observable_features / activation_function
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    accepted = audit.get("accepted_experts", [])
    e1 = next(e for e in accepted if e.get("name") == "NightZeroPersistence")
    assert e1.get("activation_feature") == "daylight_level"

    # "only E1" confirmation from the refactored Critic's final review
    critic_audit = json.loads(args.critic_audit.read_text(encoding="utf-8"))
    crit_dec = {e.get("expert_id"): e.get("final_decision")
                for e in critic_audit.get("experts", [])}
    assert crit_dec.get("E1") == "ACCEPT", f"E1 critic decision = {crit_dec.get('E1')}"
    for eid in ("E2", "E3", "E4", "E5", "E6"):
        assert crit_dec.get(eid) != "ACCEPT", f"{eid} critic decision = {crit_dec.get(eid)}"

    print("E1 accepted:", e1["name"], "| activation_feature:", e1["activation_feature"])
    print("Critic decisions (E1..E6):", {f"E{i}": crit_dec.get(f"E{i}") for i in range(1, 7)})

    # frozen compiler statistics (daylight_level lookup)
    hourly_mean_OT = np.array(
        audit["evidence_executor"]["diurnal_seasonal"]["hourly_mean_OT"], dtype=np.float64)
    monthly_mean_OT = {int(k): float(v) for k, v in
                       audit["evidence_executor"]["diurnal_seasonal"]["monthly_mean_OT"].items()}

    # ------------------------------------------------------------------ #
    # 2. E1 text embedding (frozen MiniLM, normalized)
    # ------------------------------------------------------------------ #
    enc = SentenceTransformer(args.text_model, device=str(device))
    e_E1 = enc.encode([e1["knowledge"]], normalize_embeddings=True,
                      convert_to_numpy=True).astype("float32")   # [1, 384]
    expert_dim = e_E1.shape[1]
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"e_E1 shape {e_E1.shape}, ||e_E1||={float(np.linalg.norm(e_E1)):.4f}")

    # ------------------------------------------------------------------ #
    # 3. data (same multidataset protocol as v4; TEST never read)
    # ------------------------------------------------------------------ #
    ds = args.root / args.dataset
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]
    if len(bank_starts) > args.max_bank:
        bank_starts = bank_starts[np.linspace(0, len(bank_starts) - 1, args.max_bank).astype(int)]
    if len(fit_starts) > args.max_train_query:
        fit_starts = fit_starts[np.linspace(0, len(fit_starts) - 1, args.max_train_query).astype(int)]

    X_bank, Y_bank = build_windows(series, bank_starts, args.context, args.horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, args.context, args.horizon)
    X_val, Y_val = build_windows(series, val_starts, args.context, args.horizon)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)
    print(f"bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)} "
          f"(TEST not read)")

    # ------------------------------------------------------------------ #
    # Arm A: Shape-only
    # ------------------------------------------------------------------ #
    shape_pred = shape_retrieval(X_val, X_bank, Y_bank, topk=args.topk)
    shape_mse = mse(Y_val, shape_pred)
    shape_mae = mae(Y_val, shape_pred)
    print(f"\n[Arm A] Shape-only          MSE={shape_mse:.2f} MAE={shape_mae:.2f}")

    # ------------------------------------------------------------------ #
    # Arm B: CNN-only MLP (original, no expert)
    # ------------------------------------------------------------------ #
    cnn_only = ExpertEmbeddingMLPRetriever(
        context=args.context, expert_dim=expert_dim, d_enc=args.d_enc, d_z=args.d_z,
        hidden=args.hidden, use_expert=False,
    ).to(device)
    train(cnn_only, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res, e_E1[0],
          device=device, epochs=args.epochs, batch_size=args.batch_size,
          candidates=args.candidates, lr=args.lr)
    cnn_pred = retrieve_topk_mean(cnn_only, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                  e_E1[0], "expert", args.topk, device)
    cnn_mse = mse(Y_val, cnn_pred)
    cnn_mae = mae(Y_val, cnn_pred)
    print(f"[Arm B] CNN-only MLP       MSE={cnn_mse:.2f} MAE={cnn_mae:.2f}")

    # ------------------------------------------------------------------ #
    # activation executor (reused compiler) -> a_E1(q)
    # ------------------------------------------------------------------ #
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    activation_fn, act_stats = build_activation_executor(X_train_all, hourly_mean_OT, monthly_mean_OT)
    bank_ts = [dates[s + args.context] for s in bank_starts]
    fit_ts = [dates[s + args.context] for s in fit_starts]
    val_ts = [dates[s + args.context] for s in val_starts]
    feat_names = [e1["activation_feature"]]  # ["daylight_level"]

    def compute_act(X, tss):
        M = np.zeros((len(X), len(feat_names)), dtype=np.float32)
        for i, feat in enumerate(feat_names):
            for r in range(len(X)):
                M[r, i] = activation_fn(feat, X[r], tss[r])
        return M

    bank_act = compute_act(X_bank, bank_ts)
    fit_act = compute_act(X_fit, fit_ts)
    val_act = compute_act(X_val, val_ts)

    # ------------------------------------------------------------------ #
    # Arm C: CNN + E1 (single conditional expert)
    # ------------------------------------------------------------------ #
    conditional = ConditionalExpertMLPRetriever(
        context=args.context, expert_dim=expert_dim, n_experts=1,
        d_enc=args.d_enc, d_exp=args.d_exp, d_z=args.d_z, hidden=args.hidden,
    ).to(device)
    conditional.set_expert_embeddings(e_E1)  # [1, 384]

    # fair init: copy the layers shared with B (ts_encoder / mlp / logit gates)
    sd_b = cnn_only.state_dict()
    sd_c = conditional.state_dict()
    for k in list(sd_c.keys()):
        if k in sd_b:
            sd_c[k] = sd_b[k].clone()
    conditional.load_state_dict(sd_c)

    train_conditional(conditional, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      bank_act, fit_act, device=device, epochs=args.epochs,
                      batch_size=args.batch_size, candidates=args.candidates, lr=args.lr)
    cond_pred = retrieve_conditional(conditional, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                     val_act, bank_act, "expert", args.topk, device)
    cond_mse = mse(Y_val, cond_pred)
    cond_mae = mae(Y_val, cond_pred)
    print(f"[Arm C] CNN + E1           MSE={cond_mse:.2f} MAE={cond_mae:.2f}")

    # ------------------------------------------------------------------ #
    # Arm D (optional): B-param-control — same architecture / same trainable
    # params as C, but a random unit-norm embedding instead of E1's text.
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(0)
    e_rand = rng.normal(0.0, 1.0, size=(1, expert_dim)).astype("float32")
    e_rand = e_rand / (np.linalg.norm(e_rand, axis=1, keepdims=True) + 1e-8)
    control = ConditionalExpertMLPRetriever(
        context=args.context, expert_dim=expert_dim, n_experts=1,
        d_enc=args.d_enc, d_exp=args.d_exp, d_z=args.d_z, hidden=args.hidden,
    ).to(device)
    control.set_expert_embeddings(e_rand)
    sd_ctrl = control.state_dict()
    for k in list(sd_ctrl.keys()):
        if k in sd_b:
            sd_ctrl[k] = sd_b[k].clone()
    control.load_state_dict(sd_ctrl)
    train_conditional(control, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      bank_act, fit_act, device=device, epochs=args.epochs,
                      batch_size=args.batch_size, candidates=args.candidates, lr=args.lr)
    ctrl_pred = retrieve_conditional(control, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                     val_act, bank_act, "expert", args.topk, device)
    ctrl_mse = mse(Y_val, ctrl_pred)
    ctrl_mae = mae(Y_val, ctrl_pred)
    print(f"[Arm D] B-param-control    MSE={ctrl_mse:.2f} MAE={ctrl_mae:.2f}")

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    # --- activation diagnostic (>=100 uniform VAL queries) ---
    n_diag = min(args.n_diag, len(X_val))
    diag_idx = np.linspace(0, len(X_val) - 1, n_diag).astype(int)
    diag_act = val_act[diag_idx, 0]
    rng2 = np.random.default_rng(0)
    c_idx = rng2.choice(len(bank_act), min(2000, len(bank_act)), replace=False)
    q_act = val_act[diag_idx, 0][:, None]
    c_act = bank_act[c_idx, 0][None, :]
    qc_diff = float(np.abs(q_act - c_act).mean())
    act_diag = {
        "activation_feature": "daylight_level",
        "n_val_queries": int(n_diag),
        "mean": float(diag_act.mean()),
        "std": float(diag_act.std()),
        "min": float(diag_act.min()),
        "max": float(diag_act.max()),
        "query_candidate_abs_diff_mean": qc_diff,
        "dynamic": bool(diag_act.std() > 1e-4),
    }
    print(f"[diag] daylight_level over {n_diag} VAL queries: "
          f"mean={act_diag['mean']:.4f} std={act_diag['std']:.4f} "
          f"min={act_diag['min']:.4f} max={act_diag['max']:.4f} "
          f"|q-c|={qc_diff:.4f} dynamic={act_diag['dynamic']}")

    # --- model diagnostic (is E1 branch actually used, or learned to ~0?) ---
    with torch.no_grad():
        x_diag = torch.tensor(X_val[diag_idx], dtype=torch.float32, device=device)
        act_diag_t = torch.tensor(val_act[diag_idx], dtype=torch.float32, device=device)
        ev = conditional.expert_vector(x_diag, act_diag_t)   # [B, N*E]=[B,384]
        proj = conditional.expert_proj(ev)                    # [B, d_exp]
        model_diag = {
            "expert_projection_param_norm": float(
                sum(p.data.norm().item() for p in conditional.expert_proj.parameters())),
            "expert_branch_output_abs_mean": float(proj.abs().mean().item()),
            "expert_branch_output_abs_std": float(proj.abs().std().item()),
            "expert_vector_abs_mean": float(ev.abs().mean().item()),
            "embedding_norm": float(np.linalg.norm(e_E1)),
        }
    print(f"[diag] expert_proj param norm = {model_diag['expert_projection_param_norm']:.4f}; "
          f"branch |out| mean={model_diag['expert_branch_output_abs_mean']:.5f} "
          f"std={model_diag['expert_branch_output_abs_std']:.5f}")

    # ------------------------------------------------------------------ #
    # results
    # ------------------------------------------------------------------ #
    e1_gain = (cnn_mse - cond_mse) / cnn_mse * 100.0
    b_params = n_params(cnn_only)
    c_params = n_params(conditional)
    ctrl_params = n_params(control)

    res = {
        "dataset": args.dataset,
        "protocol": "multidataset 9905-window VAL, residualize Pearson, Top-K=5 mean",
        "critic_version": "v4",
        "accepted_experts": ["E1"],
        "rejected_or_unsupported_experts": ["E2", "E3", "E4", "E5", "E6"],
        "attention_used": False,
        "test_used": False,
        "critic_modified": False,
        "expert_regenerated": False,
        "e1": {
            "name": e1["name"],
            "knowledge": e1["knowledge"],
            "activation_condition": e1["activation_condition"],
            "observable_features": e1["observable_features"],
            "activation_feature": e1["activation_feature"],
        },
        "shape_only": {"mse": shape_mse, "mae": shape_mae},
        "cnn_only_mlp": {"mse": cnn_mse, "mae": cnn_mae},
        "cnn_e1_conditional": {"mse": cond_mse, "mae": cond_mae},
        "b_param_control": {"mse": ctrl_mse, "mae": ctrl_mae},
        "e1_gain_vs_cnn_only_pct": e1_gain,
        "param_control_gain_vs_cnn_only_pct": (cnn_mse - ctrl_mse) / cnn_mse * 100.0,
        "parameter_counts": {
            "cnn_only_mlp": b_params,
            "cnn_e1_conditional": c_params,
            "b_param_control": ctrl_params,
        },
        "activation_diagnostic": act_diag,
        "model_diagnostic": model_diag,
        "baseline_shape_mse": 219891.45,
    }

    # ------------------------------------------------------------------ #
    # save
    # ------------------------------------------------------------------ #
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "cnn_only_state": cnn_only.state_dict(),
        "conditional_state": conditional.state_dict(),
        "control_state": control.state_dict(),
        "e_E1": e_E1,
        "e_rand": e_rand,
        "args": vars(args),
    }, out / "models.pt")
    np.savez_compressed(out / "val_predictions.npz",
                        y_true=Y_val, val_starts=val_starts,
                        shape_pred=shape_pred, cnn_only_pred=cnn_pred,
                        cond_pred=cond_pred, control_pred=ctrl_pred)
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                                     encoding="utf-8")
    (out / "activation_diagnostic.json").write_text(
        json.dumps(act_diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "model_diagnostic.json").write_text(
        json.dumps(model_diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = [
        "# E1 Expert-MLP Frozen Experiment Audit",
        "",
        "- critic_version = v4",
        "- accepted_experts = [\"E1\"]",
        "- rejected_or_unsupported_experts = [\"E2\",\"E3\",\"E4\",\"E5\",\"E6\"]",
        "- attention_used = false",
        "- test_used = false",
        "- critic_modified = false",
        "- expert_regenerated = false",
        "",
        "## Arms (identical protocol except Expert input)",
        "",
        f"- A. Shape-only MSE={shape_mse:.2f} MAE={shape_mae:.2f}",
        f"- B. CNN-only MLP MSE={cnn_mse:.2f} MAE={cnn_mae:.2f} ({b_params} params)",
        f"- C. CNN + E1 MSE={cond_mse:.2f} MAE={cond_mae:.2f} ({c_params} params)",
        f"- D. B-param-control (random emb) MSE={ctrl_mse:.2f} MAE={ctrl_mae:.2f} ({ctrl_params} params)",
        "",
        f"- E1 gain vs CNN-only = {e1_gain:+.2f}%",
        "",
        "## E1 activation",
        "",
        f"a_E1(q) = daylight_level(q) = TRAIN_hourly_mean_OT[hour(q)] / max(TRAIN_hourly_mean_OT), "
        "range [0,1], TRAIN covers 24/24 hours.",
        "",
        f"VAL activation: mean={act_diag['mean']:.4f} std={act_diag['std']:.4f} "
        f"min={act_diag['min']:.4f} max={act_diag['max']:.4f}, dynamic={act_diag['dynamic']}.",
        "",
        f"ExpertProjection param norm = {model_diag['expert_projection_param_norm']:.4f}; "
        f"branch |out| mean={model_diag['expert_branch_output_abs_mean']:.5f}.",
    ]
    (out / "experiment_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("E1 EXPERT-MLP FROZEN EXPERIMENT")
    print("=" * 72)
    print(f"A. Shape-only          MSE = {shape_mse:.2f}  MAE = {shape_mae:.2f}")
    print(f"B. CNN-only MLP        MSE = {cnn_mse:.2f}  MAE = {cnn_mae:.2f}  ({b_params} params)")
    print(f"C. CNN + E1            MSE = {cond_mse:.2f}  MAE = {cond_mae:.2f}  ({c_params} params)")
    print(f"D. B-param-control     MSE = {ctrl_mse:.2f}  MAE = {ctrl_mae:.2f}  ({ctrl_params} params)")
    print("-" * 72)
    print(f"E1 gain vs CNN-only = {e1_gain:+.2f}%")
    verdict = "POSITIVE" if e1_gain > 0.0 else ("NEUTRAL" if abs(e1_gain) < 1e-9 else "NEGATIVE")
    print(f"verdict = {verdict}")
    print("=" * 72)
    print("Saved to:", out)
    print("TEST was NOT evaluated. Critic v4 NOT modified. No Attention. No regenerated expert.")


if __name__ == "__main__":
    main()
