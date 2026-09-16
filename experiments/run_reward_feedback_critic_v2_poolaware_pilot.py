#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward-Feedback Critic v2 小规模验证（Pool-Aware Reward）。

唯一方法改动：把 v1 的「逐条 Expert 孤立 reward」改成「Expert 对整个 Expert Pool
的边际贡献 reward」。其余全部冻结（Expert/Critic prompt、Compiler、retrieval_action、
activation、Retriever 架构、text embedding、Top-K、optimizer、epochs、batch、数据划分、
deterministic protocol、seeds=[42,123,2026]）。

v2 reward：在 TRAIN-only chronological folds 里，用与最终 C/D arm 一致的 learned
retriever（embedding + activation + learned retriever + pool interaction）比较
FULL_POOL vs POOL_MINUS_i：
    pool_reward_i = (MSE_pool_minus_i - MSE_full_pool) / MSE_pool_minus_i

只跑 3 个数据集：ETTh1（正例）/ Electricity（反例）/ Weather（正例）。
每个跑 A/B/C/D1(v1)/D2(v2) 5 arms × 3 seeds，严格 deterministic。

输出：results/reward_feedback_critic_v2_poolaware_pilot/
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from run_expert_embedding_mlp import build_windows, mse, mae
from run_solar_same_val_baselines import residualize
from run_conditional_expert_mlp_v4 import ConditionalExpertMLPRetriever, retrieve_conditional
from run_etth1_prediction_validation import compute_training_stats, build_activation_matrix

import run_etth1_reward_feedback_critic_v1 as v1
import run_agent_v2_electricity as av2

REPO_ROOT = HERE.parent
OUT_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_poolaware_pilot"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LLM_CACHE_ROOT = REPO_ROOT / "cache/llm_priors_reward_feedback_critic_v2_pilot"

V1_CROSS = REPO_ROOT / "results/reward_feedback_critic_v1_cross_dataset"
V1_ETTH1 = REPO_ROOT / "results/reward_feedback_critic_v1_etth1"

SEEDS = [42, 123, 2026]
TOPK = 5
N_RAW_TARGET = 8
N_FOLD_BLOCKS = 5
MIN_COVERAGE = 0.01
MAX_REVISION_ROUNDS = 2

EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32

# 数据集：name + (context,horizon) 来自 config.json；domain/freq 仅用于 expert 生成
DATASETS = [
    {"name": "etth1", "domain": "transformer_temperature", "freq": "hourly"},
    {"name": "electricity", "domain": "aggregate_electricity_load", "freq": "15-minute"},
    {"name": "weather", "domain": "weather_meteorological", "freq": "10-minute"},
]


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


def mask_emb(emb, keep=None, drop=None):
    e = np.zeros_like(emb, dtype=np.float32)
    idx = keep if keep is not None else [i for i in range(emb.shape[0]) if i not in (drop or [])]
    for i in idx:
        e[i] = emb[i]
    return e


def mask_act(act, keep=None, drop=None):
    a = {k: np.zeros_like(v, dtype=np.float32) for k, v in act.items()}
    n = act["bank"].shape[1]
    idx = keep if keep is not None else [i for i in range(n) if i not in (drop or [])]
    for i in idx:
        for k in act:
            a[k][:, i] = act[k][:, i]
    return a


def mask_act2d(a, drop=None, keep=None):
    b = np.zeros_like(a, dtype=np.float32)
    n = a.shape[1]
    idx = keep if keep is not None else [i for i in range(n) if i not in (drop or [])]
    for i in idx:
        b[:, i] = a[:, i]
    return b


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


def train_eval(emb, act_bank, act_fit, act_q, X_sb, Y_sb, X_sf, Y_sf, X_q, Y_q,
               X_sb_res, X_sf_res, X_q_res, expert_dim, n_slots, device, seed, context):
    seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.set_expert_embeddings(emb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_arm(mdl, X_sb, Y_sb, X_sf, Y_sf, X_sb_res, X_sf_res,
                  act_bank, act_fit, True, device, seed)
    pred = retrieve_conditional(mdl, X_q, X_sb, Y_sb, X_q_res, X_sb_res,
                                act_q, act_bank, "full", TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(mse(Y_q, pred))


# --------------------------------------------------------------------------- #
# v2 pool-aware reward（TRAIN-only，4 folds，RETRAIN full + minus_i）
# --------------------------------------------------------------------------- #
def compute_pool_reward(experts, train_starts, series, dates, stats, device, context, horizon, seed=42):
    n = len(train_starts)
    edges = np.linspace(0, n, N_FOLD_BLOCKS + 1).astype(int)
    expert_ids = [e["id"] for e in experts]
    n_exp = len(experts)
    n_slots = max(n_exp, 1)
    emb = v1.embed_claims([e["claim"] for e in experts], device)
    expert_dim = emb.shape[1]

    fold_full = []
    fold_minus = {eid: [] for eid in expert_ids}
    for f in range(1, N_FOLD_BLOCKS):
        bank_idx = train_starts[:edges[f]]
        q_idx = train_starts[edges[f]:edges[f + 1]]
        cut = int(len(bank_idx) * 0.8)
        sb_idx = bank_idx[:cut]
        sf_idx = bank_idx[cut:]

        X_sb, Y_sb = build_windows(series, sb_idx, context, horizon)
        X_sf, Y_sf = build_windows(series, sf_idx, context, horizon)
        X_q, Y_q = build_windows(series, q_idx, context, horizon)
        X_sb_res = residualize(X_sb)
        X_sf_res = residualize(X_sf)
        X_q_res = residualize(X_q)
        sb_ts = [dates[s + context] for s in sb_idx]
        sf_ts = [dates[s + context] for s in sf_idx]
        q_ts = [dates[s + context] for s in q_idx]
        act_bank = build_activation_matrix(experts, X_sb, sb_ts, stats)
        act_fit = build_activation_matrix(experts, X_sf, sf_ts, stats)
        act_q = build_activation_matrix(experts, X_q, q_ts, stats)

        mse_full = train_eval(emb, act_bank, act_fit, act_q, X_sb, Y_sb, X_sf, Y_sf,
                              X_q, Y_q, X_sb_res, X_sf_res, X_q_res, expert_dim, n_slots,
                              device, seed, context)
        fold_full.append(mse_full)
        for i, eid in enumerate(expert_ids):
            mse_minus = train_eval(mask_emb(emb, drop=[i]), mask_act2d(act_bank, drop=[i]),
                                   mask_act2d(act_fit, drop=[i]), mask_act2d(act_q, drop=[i]),
                                   X_sb, Y_sb, X_sf, Y_sf, X_q, Y_q, X_sb_res, X_sf_res,
                                   X_q_res, expert_dim, n_slots, device, seed, context)
            pr = (mse_minus - mse_full) / mse_minus if mse_minus > 0 else 0.0
            fold_minus[eid].append(pr)
            print(f"    fold{f} {eid}: mse_full={mse_full:.4e} mse_minus={mse_minus:.4e} pool_reward={pr:+.4f}")

    stats_out = {}
    for eid in expert_ids:
        prs = np.array(fold_minus[eid], dtype=float)
        cov = None
        stats_out[eid] = {
            "fold_rewards": fold_minus[eid],
            "mean_pool_reward": float(prs.mean()),
            "median_pool_reward": float(np.median(prs)),
            "positive_fold_ratio": float((prs > 0).mean()),
            "worst_pool_reward": float(prs.min()),
        }
    return stats_out, fold_full, emb, expert_dim


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


# --------------------------------------------------------------------------- #
# v2 critic / revision（LLM，pool-aware evidence）
# --------------------------------------------------------------------------- #
def _evidence_text(rec):
    e = rec["expert"]
    st = rec["pool_stats"]
    lines = [
        f"expert {e['id']}: mechanism={e.get('mechanism','')}",
        f"  scope={e.get('scope','')} | activation_condition={e.get('activation_condition','')}",
        f"  observables={e.get('observables',[])} | retrieval_action={e.get('retrieval_action','')}",
        f"  pool_reward(mean)={st['mean_pool_reward']:+.4f} median={st['median_pool_reward']:+.4f} "
        f"positive_fold_ratio={st['positive_fold_ratio']:.2f} worst={st['worst_pool_reward']:+.4f}",
        f"  fold_pool_rewards={['%.4f' % x for x in st['fold_rewards']]}",
        f"  single_effect={rec.get('single_effect')} synergy_flag={rec.get('synergy_flag')}",
        f"  current_pool_size={rec.get('pool_size')}",
    ]
    return "\n".join(lines)


def llm_critic_v2(round_records, model):
    body = "\n\n".join(_evidence_text(r) for r in round_records)
    system = (
        "You are a rigorous critic for a retrieval-based forecaster using a POOL-AWARE reward. "
        "The reward measures each expert's MARGINAL contribution to the full expert pool in the "
        "learned retriever (embedding + activation + pool interaction), NOT its isolated retrieval "
        "action. pool_reward>0 means removing the expert hurts the pool (positive contribution). "
        "Distinguish single-expert effect from pool contribution: an expert with single_effect<=0 "
        "but pool_reward>0 is a SYNERGY expert (valuable only in the team). Do NOT recompute data. "
        "Output strict JSON only."
    )
    user = "\n\n".join([
        "EVIDENCE (per expert):\n" + body,
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"critiques": [ ... ]}. Each entry:\n'
        '  - "expert_id"\n'
        '  - "verdict": must equal the given reward verdict\n'
        '  - "pool_contribution": "positive|negative|neutral"\n'
        '  - "synergy_interpretation": whether this is a synergy expert\n'
        '  - "diagnosis": 2-3 sentences citing the pool_reward numbers\n'
        '  - "revision_instruction": if REVISE/REJECT/UNSUPPORTED, what to change (only refine '
        "based on evidence; do NOT invent data patterns)\n"
        "TASK: output only the JSON object.",
    ])
    return v1._call_llm_list(system, user, f"rf_critic_v2_pilot_critic", "critiques", model)


def llm_revise_v2(to_revise, revision_round, model):
    feedbacks = []
    for r in to_revise:
        e = r["expert"]
        feedbacks.append({
            "expert_id": e["id"],
            "previous_knowledge": {k: e.get(k) for k in (
                "mechanism", "scope", "activation_condition", "failure_condition",
                "observables", "retrieval_action", "expected_effect", "reason", "condition_spec")},
            "pool_reward_evidence": r["pool_stats"],
            "critic_diagnosis": r["critic"].get("diagnosis", ""),
            "revision_instruction": r["critic"].get("revision_instruction", ""),
        })
    system = (
        "You are a domain expert revising conditional retrieval knowledge after a pool-aware critic "
        "review. Revise each failed expert so that its MARGINAL contribution to the full expert pool "
        "(in the learned retriever) becomes positive. You may change retrieval_action, mechanism, "
        "scope, or activation — but do NOT merely add AND conditions. If the mechanism cannot be "
        "verified on TRAIN (UNSUPPORTED), you MAY switch to a different TRAIN-verifiable mechanism "
        "and MUST set 'replaced': true. Do NOT invent data patterns. Output strict JSON only."
    )
    user = "\n\n".join([
        v1._catalog_text(), v1._expert_schema_text(),
        "REVISION ROUND: " + str(revision_round),
        "FEEDBACK (per expert):\n" + json.dumps(feedbacks, ensure_ascii=False, indent=2),
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"experts": [ ... ]} with the SAME keys as the expert schema PLUS "replaced": true/false. '
        "Keep the same 'id'. Output only the JSON object.",
    ])
    experts, cached = v1._call_llm_list(system, user, f"rf_critic_v2_pilot_revision", "experts", model)
    return experts, cached


# --------------------------------------------------------------------------- #
# 单数据集 v2 闭环
# --------------------------------------------------------------------------- #
def run_v2_loop(name, cfg, device):
    ds_dir = REPO_ROOT / "results/multidataset" / name
    cfg_json = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg_json["context"])
    horizon = int(cfg_json["horizon"])
    v1.CONTEXT = context
    v1.HORIZON = horizon
    av2.CACHE_DIR = str(LLM_CACHE_ROOT / name)

    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]
    X_train_all_bank = build_windows(series, bank_starts, context, horizon)[0]
    X_train_all_fit = build_windows(series, fit_starts, context, horizon)[0]
    X_train_all = np.concatenate([X_train_all_bank, X_train_all_fit], axis=0)
    stats = compute_training_stats(X_train_all)
    evidence = {
        "feature_distributions": v1._raw_feature_quantiles(X_train_all),
        "temporal_coverage": v1._temporal_coverage(dates.iloc[:int(len(train_starts) + context)]),
    }

    # raw experts（复用 v1 已生成的 raw）
    if name == "etth1":
        raw_src = V1_ETTH1 / "raw_experts.json"
    else:
        raw_src = V1_CROSS / name / "raw_experts.json"
    raw_experts = v1._sorted_experts(json.loads(raw_src.read_text(encoding="utf-8")))[:N_RAW_TARGET]
    for e in raw_experts:
        e["claim"] = e.get("mechanism", e.get("reason", ""))
        e["observable_features"] = list(e.get("observables", []))
        e["replaced"] = False
    model = av2._default_model()

    # v2 闭环
    lineage_best = {}
    lineage_coverage = {}
    experts = list(raw_experts)
    round_records_all = []
    for rnd in range(0, MAX_REVISION_ROUNDS + 1):
        print(f"  [v2 Round {rnd}] pool={len(experts)} experts")
        for e in experts:
            e["compiled"] = v1.compile_condition_v6(e, v1.FEATURE_REGISTRY, evidence, v1.COMPILER_CONFIG)
            e["retrieval_spec"] = v1.derive_retrieval_spec(e, v1.FEATURE_REGISTRY)
        pool_stats, fold_full, emb_pool, expert_dim = compute_pool_reward(
            experts, train_starts, series, dates, stats, device, context, horizon, seed=42)
        # coverage（TRAIN 全窗 activation）
        act_train = build_activation_matrix(experts, X_train_all,
                                            [dates[s + context] for s in train_starts], stats)
        coverage = {e["id"]: float((act_train[:, j] >= 0.5).mean())
                    for j, e in enumerate(experts)}
        round_records = []
        for j, e in enumerate(experts):
            eid = e["id"]
            st = pool_stats[eid]
            reward, verdict = v2_reward_rule(st, coverage[eid])
            rec = {"expert": e, "pool_stats": st, "reward": reward, "verdict": verdict,
                   "single_effect": None, "synergy_flag": None, "pool_size": len(experts),
                   "round": rnd, "lineage_id": eid}
            round_records.append(rec)
            lineage_best[eid] = rec
            lineage_coverage[eid] = coverage[eid]
            print(f"    {eid}: pool_reward mean={st['mean_pool_reward']:+.4f} "
                  f"pfr={st['positive_fold_ratio']:.2f} reward={reward:+d} {verdict} cov={coverage[eid]:.3f}")

        critiques, _ = llm_critic_v2(round_records, model)
        crit_by_id = {c.get("expert_id"): c for c in critiques}
        for rec in round_records:
            crit = crit_by_id.get(rec["lineage_id"], {})
            crit.setdefault("verdict", rec["verdict"])
            crit.setdefault("pool_contribution", "")
            crit.setdefault("synergy_interpretation", "")
            crit.setdefault("diagnosis", "")
            crit.setdefault("revision_instruction", "")
            rec["critic"] = crit
            rec["critic"]["verdict"] = rec["verdict"]
        round_records_all.append(round_records)

        to_revise = [r for r in round_records if r["verdict"] in ("REVISE", "UNSUPPORTED", "REJECT")]
        if rnd >= MAX_REVISION_ROUNDS or not to_revise:
            break
        revised_experts, _ = llm_revise_v2(to_revise, rnd + 1, model)
        revised_by_id = {e.get("id"): e for e in revised_experts}
        next_experts = []
        for rec in to_revise:
            ne = revised_by_id.get(rec["lineage_id"])
            if ne is None:
                ne = dict(rec["expert"])
            ne["id"] = rec["lineage_id"]
            ne["claim"] = ne.get("mechanism", ne.get("reason", ""))
            ne["observable_features"] = list(ne.get("observables", []))
            ne["revision_round"] = rnd + 1
            ne["replaced"] = bool(ne.get("replaced", False))
            next_experts.append(ne)
        experts = next_experts

    # v2 final pool = reward>0
    final_pool = []
    for lid, rec in lineage_best.items():
        if rec["reward"] > 0:
            final_pool.append(rec["expert"])
    final_pool = v1._sorted_experts(final_pool)
    print(f"  [v2] final pool: {len(final_pool)} -> {[e['id'] for e in final_pool]}")

    # 写 lineage
    pool_rows = []
    for rec in round_records_all:
        for r in rec:
            e = r["expert"]
            pool_rows.append({
                "expert_id": r["lineage_id"], "round": r["round"], "reward": r["reward"],
                "verdict": r["verdict"], "mean_pool_reward": r["pool_stats"]["mean_pool_reward"],
                "median_pool_reward": r["pool_stats"]["median_pool_reward"],
                "positive_fold_ratio": r["pool_stats"]["positive_fold_ratio"],
                "worst_pool_reward": r["pool_stats"]["worst_pool_reward"],
                "coverage": lineage_coverage.get(r["lineage_id"]),
                "mechanism": e.get("mechanism", ""), "scope": e.get("scope", ""),
                "replaced": bool(e.get("replaced", False)),
            })
    with (OUT_ROOT / name / "pool_reward_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pool_rows[0].keys()))
        w.writeheader()
        for r in pool_rows:
            w.writerow(r)

    return raw_experts, final_pool


# --------------------------------------------------------------------------- #
# VAL ablation A/B/C/D1/D2
# --------------------------------------------------------------------------- #
def run_val(name, cfg, raw_experts, v1_pool, v2_pool, device):
    ds_dir = REPO_ROOT / "results/multidataset" / name
    cfg_json = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg_json["context"])
    horizon = int(cfg_json["horizon"])

    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    global X_bank, Y_bank, X_fit, Y_fit, X_val, Y_val, X_bank_res, X_fit_res, X_val_res
    X_bank, Y_bank = build_windows(series, bank_starts, context, horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, context, horizon)
    X_val, Y_val = build_windows(series, val_starts, context, horizon)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)
    bank_ts = [dates[s + context] for s in bank_starts]
    fit_ts = [dates[s + context] for s in fit_starts]
    val_ts = [dates[s + context] for s in val_starts]
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)

    emb_raw = v1.embed_claims([e["claim"] for e in raw_experts], device)
    emb_v1 = v1.embed_claims([e["claim"] for e in v1_pool], device) if v1_pool else np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    emb_v2 = v1.embed_claims([e["claim"] for e in v2_pool], device) if v2_pool else np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    expert_dim = emb_raw.shape[1]
    n_slots = max(len(raw_experts), len(v1_pool), len(v2_pool))

    raw_act = v1.act_mats(raw_experts, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    v1_act = v1.act_mats(v1_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats) if v1_pool else {"bank": np.zeros((len(X_bank), 0), dtype=np.float32), "fit": np.zeros((len(X_fit), 0), dtype=np.float32), "val": np.zeros((len(X_val), 0), dtype=np.float32)}
    v2_act = v1.act_mats(v2_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats) if v2_pool else {"bank": np.zeros((len(X_bank), 0), dtype=np.float32), "fit": np.zeros((len(X_fit), 0), dtype=np.float32), "val": np.zeros((len(X_val), 0), dtype=np.float32)}
    emb_raw, raw_act, _ = v1.pad_to_slots(emb_raw, raw_act, n_slots)
    emb_v1, v1_act, _ = v1.pad_to_slots(emb_v1, v1_act, n_slots)
    emb_v2, v2_act, _ = v1.pad_to_slots(emb_v2, v2_act, n_slots)
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_act = {k: np.zeros_like(raw_act[k]) for k in raw_act}

    configs = {
        "base": (emb_zero, zero_act, False),
        "shape": (emb_zero, zero_act, True),
        "raw": (emb_raw, raw_act, True),
        "v1": (emb_v1, v1_act, True),
        "v2": (emb_v2, v2_act, True),
    }

    per_seed = {nm: {} for nm in configs}
    for seed in SEEDS:
        seed_everything(seed)
        base_model = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        for nm, (emb, act, use_shape) in configs.items():
            mm = run_val_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed, context)
            per_seed[nm][seed] = mm
            print(f"    seed={seed} [{nm}] MSE={mm:.4f}")

    summary = {}
    for nm in configs:
        ms = [per_seed[nm][s] for s in SEEDS]
        summary[nm] = {"mean": float(np.mean(ms)), "std": float(np.std(ms)),
                       "seeds": {s: per_seed[nm][s] for s in SEEDS}}
    return summary, per_seed


def run_val_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed, context):
    seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                  act["bank"], act["fit"], use_shape, device, seed)
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(mdl, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                act["val"], act["bank"], mode, TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(mse(Y_val, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    targets = [d for d in DATASETS if (args.datasets is None or d["name"] in args.datasets)]
    results = []
    for cfg in targets:
        name = cfg["name"]
        (OUT_ROOT / name).mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\nDATASET {name}\n{'=' * 70}")
        try:
            raw_experts, v2_pool = run_v2_loop(name, cfg, device)
            # 载入 v1 final pool（冻结）
            if name == "etth1":
                v1_pool = v1._sorted_experts(json.loads(
                    (V1_ETTH1 / "final_expert_pool.json").read_text(encoding="utf-8")))
            else:
                v1_pool = v1._sorted_experts(json.loads(
                    (V1_CROSS / name / "final_expert_pool.json").read_text(encoding="utf-8")))
            print(f"  v1 final pool (frozen): {len(v1_pool)} -> {[e['id'] for e in v1_pool]}")
            summary, per_seed = run_val(name, cfg, raw_experts, v1_pool, v2_pool, device)
            results.append({"dataset": name, "summary": summary, "per_seed": per_seed,
                            "n_v1": len(v1_pool), "n_v2": len(v2_pool)})
        except Exception as e:
            import traceback
            print(f"  [FAILED] {name}: {e}")
            traceback.print_exc()
            results.append({"dataset": name, "error": str(e)})

    # ---- 汇总 ----
    rows = []
    for r in results:
        if "error" in r:
            continue
        s = r["summary"]
        raw_m = s["raw"]["mean"]
        v1_m = s["v1"]["mean"]
        v2_m = s["v2"]["mean"]
        v2_wins_raw = sum(1 for sd in SEEDS if s["v2"]["seeds"][sd] < s["raw"]["seeds"][sd])
        v2_wins_v1 = sum(1 for sd in SEEDS if s["v2"]["seeds"][sd] < s["v1"]["seeds"][sd])
        rows.append({
            "dataset": r["dataset"], "raw_mean_mse": raw_m, "v1_mean_mse": v1_m,
            "v2_mean_mse": v2_m, "v2_vs_raw_pct": (raw_m - v2_m) / raw_m * 100.0,
            "v2_vs_v1_pct": (v1_m - v2_m) / v1_m * 100.0,
            "v1_pool_size": r["n_v1"], "v2_pool_size": r["n_v2"],
            "v2_wins_vs_raw_seeds": v2_wins_raw, "v2_wins_vs_v1_seeds": v2_wins_v1,
        })

    with (OUT_ROOT / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "raw_mean_mse", "v1_mean_mse", "v2_mean_mse",
                                          "v2_vs_raw_pct", "v2_vs_v1_pct", "v1_pool_size",
                                          "v2_pool_size", "v2_wins_vs_raw_seeds", "v2_wins_vs_v1_seeds"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # val ablation 明细
    arm_rows = []
    for r in results:
        if "error" in r:
            continue
        for nm in ("base", "shape", "raw", "v1", "v2"):
            for sd in SEEDS:
                arm_rows.append({"dataset": r["dataset"], "arm": nm, "seed": sd,
                                 "mse": r["per_seed"][nm][sd]})
    with (OUT_ROOT / "pilot_val_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "arm", "seed", "mse"])
        w.writeheader()
        for r in arm_rows:
            w.writerow(r)

    report = render_report(results, rows)
    (OUT_ROOT / "pilot_report.md").write_text(report, encoding="utf-8")
    print("\n" + "=" * 70)
    for r in rows:
        print(f"{r['dataset']:<12} raw={r['raw_mean_mse']:.3e} v1={r['v1_mean_mse']:.3e} "
              f"v2={r['v2_mean_mse']:.3e} v2_vs_raw={r['v2_vs_raw_pct']:+.1f}% "
              f"v2_vs_v1={r['v2_vs_v1_pct']:+.1f}% pool={r['v1_pool_size']}->{r['v2_pool_size']} "
              f"wins_vs_raw={r['v2_wins_vs_raw_seeds']}/3")
    for r in results:
        if "error" in r:
            print(f"{r['dataset']} FAILED: {r['error']}")
    print(f"Saved to {OUT_ROOT}")


def render_report(results, rows):
    L = []
    L.append("# Reward-Feedback Critic v2 小规模验证（Pool-Aware Reward）")
    L.append("")
    L.append("只改 reward 机制（逐条 → pool-aware 边际贡献），其余冻结。3 数据集：ETTh1/Electricity/Weather。")
    L.append("")
    L.append("## 结果")
    L.append("")
    by = {r["dataset"]: r for r in rows}
    elec = by.get("electricity")
    etth1 = by.get("etth1")
    weather = by.get("weather")
    if elec:
        L.append(f"- Electricity pool size：{elec['v1_pool_size']} → {elec['v2_pool_size']}（v1 塌缩到 1，v2 是否保住组合）。")
    L.append("")
    L.append("主表：")
    L.append("")
    L.append("| Dataset | Raw | v1 | v2 | v2 vs Raw | v2 vs v1 | Pool v1→v2 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['dataset']} | {r['raw_mean_mse']:.4e} | {r['v1_mean_mse']:.4e} | {r['v2_mean_mse']:.4e} "
                 f"| {r['v2_vs_raw_pct']:+.1f}% | {r['v2_vs_v1_pct']:+.1f}% | {r['v1_pool_size']}→{r['v2_pool_size']} |")
    L.append("")

    # 判定
    ok = True
    reasons = []
    if elec:
        if elec["v2_mean_mse"] < elec["v1_mean_mse"] and elec["v2_pool_size"] > elec["v1_pool_size"]:
            reasons.append("Electricity v2 明显优于 v1 且 pool 不再塌缩")
        else:
            ok = False
            reasons.append("Electricity v2 未修复")
    if etth1:
        delta = (etth1["v2_mean_mse"] - etth1["v1_mean_mse"]) / etth1["v1_mean_mse"]
        if delta < 0.05:
            reasons.append(f"ETTh1 v2 未明显变坏（{delta*100:+.1f}%）")
        else:
            ok = False
            reasons.append(f"ETTh1 v2 明显变坏（{delta*100:+.1f}%）")
    if weather:
        delta = (weather["v2_mean_mse"] - weather["v1_mean_mse"]) / weather["v1_mean_mse"]
        if delta < 0.05:
            reasons.append(f"Weather v2 未明显变坏（{delta*100:+.1f}%）")
        else:
            ok = False
            reasons.append(f"Weather v2 明显变坏（{delta*100:+.1f}%）")

    L.append("## 结论")
    L.append("")
    if not rows:
        L.append("- **FAIL**（无成功数据集）")
    elif ok and elec and elec["v2_mean_mse"] < elec["v1_mean_mse"]:
        L.append("- **PASS**（v2 修 Electricity，同时基本保住 ETTh1/Weather）")
    elif elec and elec["v2_mean_mse"] < elec["v1_mean_mse"]:
        L.append("- **PARTIAL**（v2 有帮助，但正例有一定退化）")
    else:
        L.append("- **FAIL**（v2 未修反例或明显破坏正例）")
    for r in reasons:
        L.append(f"  - {r}")
    L.append("")
    L.append("```")
    L.append("test_used = false")
    L.append("deterministic_protocol = true")
    L.append("only_change = pool_aware_reward")
    L.append("```")
    return "\n".join(L)


if __name__ == "__main__":
    main()
