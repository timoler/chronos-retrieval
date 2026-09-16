#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward-Feedback Critic v1 — 跨数据集正式验证（冻结方法迁移）。

把 ETTh1 上已正式跑通的 Reward-Feedback Critic v1 冻结后迁移到 6 个数据集：
ETTh2 / electricity / solar / weather / exchange_rate / traffic。

冻结（不修改）：Expert/Critic prompt、Reward 规则与阈值、Compiler、retrieval_action、
activation 逻辑、Retriever 架构、Top-K、optimizer、epochs、batch、training protocol、
deterministic protocol。不访问 TEST、不挑/调 seed、不加 Attention/Gate/Learned Critic。

唯一允许的 per-dataset 输入：数据集名称 + domain/采样频率 metadata（来自各数据集
config.json）+ context/horizon（各数据集 config.json 里已定好的窗口长度，是数据准备
属性而非方法超参；ETTh2/traffic=168/24，其余=512/96）。Expert 生成用同一套通用 prompt，
不写任何数据集专用 Expert 模板。

流程（每数据集，与 v1 一致）：
  Dataset -> Raw Expert 生成 -> Compiler -> TRAIN-only retrieval 执行 -> Reward
  -> Critic 诊断 -> Expert revision -> 再执行 -> Final Pool -> 冻结 -> 首次访问 VAL。
Reward loop 只看 TRAIN；VAL 在闭环完全结束后首次访问；TEST 全程不用。

输出：results/reward_feedback_critic_v1_cross_dataset/
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
import traceback
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
from run_etth1_prediction_validation import compute_training_stats

import run_etth1_reward_feedback_critic_v1 as v1
import run_agent_v2_electricity as av2

REPO_ROOT = HERE.parent
OUT_ROOT = REPO_ROOT / "results/reward_feedback_critic_v1_cross_dataset"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LLM_CACHE_ROOT = REPO_ROOT / "cache/llm_priors_reward_feedback_critic_v1_cross_dataset"

SEEDS = [42, 123, 2026]
TOPK = 5
N_RAW_TARGET = 8

# 冻结自 v1 的超参（不修改）
EPOCHS, BATCH_SIZE, CANDIDATES, LR = v1.EPOCHS, v1.BATCH_SIZE, v1.CANDIDATES, v1.LR
D_ENC, D_EXP, D_Z, HIDDEN = v1.D_ENC, v1.D_EXP, v1.D_Z, v1.HIDDEN

# 数据集注册：name + domain/freq（来自各 config.json 的 metadata，非专家模板）
DATASETS = [
    {"name": "etth2",        "domain": "transformer_temperature",  "freq": "hourly"},
    {"name": "electricity",  "domain": "aggregate_electricity_load", "freq": "15-minute"},
    {"name": "solar",        "domain": "aggregate_solar_generation", "freq": "10-minute"},
    {"name": "weather",      "domain": "weather_meteorological",    "freq": "10-minute"},
    {"name": "exchange_rate","domain": "daily_exchange_rate",       "freq": "daily"},
    {"name": "traffic",      "domain": "road_traffic_occupancy",    "freq": "hourly"},
]


# --------------------------------------------------------------------------- #
# deterministic seeding（与 ETTh1 正式协议一致）
# --------------------------------------------------------------------------- #
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


def model_init_hash(state_dict):
    import hashlib
    h = hashlib.sha256()
    for p in state_dict.values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device, seed):
    """与 v1.train_arm 完全一致，唯一差异：rng 由 seed 参数化（deterministic）。"""
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


def run_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed):
    seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=v1.CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
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
    return float(mse(Y_val, pred)), float(mae(Y_val, pred))


# --------------------------------------------------------------------------- #
# 通用 Expert 生成（同一 prompt，domain/freq 来自 metadata，非专用模板）
# --------------------------------------------------------------------------- #
def _metadata_text(cfg):
    return (
        "TARGET AND DATA:\n"
        "- target variable: OT (the univariate series to forecast)\n"
        f"- domain: {cfg['domain']}\n"
        f"- sampling frequency: {cfg['freq']}\n"
        "- observed variables: OT only (univariate target)\n"
        f"- context window: {cfg['context']} steps, forecast horizon: {cfg['horizon']} steps\n"
    )


def _expert_hint_text(cfg):
    return (
        "EXPERT HINT (mechanism hypotheses, NOT verified facts):\n"
        f"For this univariate {cfg['freq']} series in the domain '{cfg['domain']}', propose "
        "conditional retrieval knowledge about WHEN historical segments are more predictive and "
        "HOW retrieval should change in that regime. Consider: level / trend / volatility regimes, "
        "calendar cycles (daily / weekly / seasonal, only where the sampling frequency makes them "
        "meaningful), and how recent dynamics predict the near future. Each entry is a hypothesis "
        "to be checked against TRAIN data."
    )


def generate_raw_experts(cfg, model):
    system = (
        "You are a domain expert on forecasting a univariate time series (target variable OT). "
        "Propose CONDITIONAL expert knowledge for a retrieval-based forecaster: each piece states "
        "a mechanism about WHEN historical segments are more predictive and HOW retrieval should "
        "change in that regime. These are hypotheses to be checked against data, not established "
        "facts. Output strict JSON only."
    )
    user = "\n\n".join([
        _metadata_text(cfg), _expert_hint_text(cfg), v1._catalog_text(), v1._expert_schema_text(),
        f"TASK: Propose exactly {N_RAW_TARGET} conditional expert knowledge entries. "
        "Output only the JSON object.",
    ])
    experts, cached = v1._call_llm_list(system, user, f"rf_critic_v1_raw_{cfg['name']}", "experts", model)
    for i, e in enumerate(experts, 1):
        e["id"] = e.get("id") or f"E{i}"
    return experts, cached


# --------------------------------------------------------------------------- #
# 单数据集完整流程
# --------------------------------------------------------------------------- #
def run_dataset(cfg, device):
    name = cfg["name"]
    ds_dir = REPO_ROOT / "results/multidataset" / name
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_json = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context = int(cfg_json["context"])
    horizon = int(cfg_json["horizon"])
    cfg["context"], cfg["horizon"] = context, horizon

    # 注入 per-dataset context/horizon 到 v1（方法核心冻结，仅窗口长度按数据准备）
    v1.CONTEXT = context
    v1.HORIZON = horizon
    av2.CACHE_DIR = str(LLM_CACHE_ROOT / name)

    print(f"\n{'=' * 78}\nDATASET {name}  domain={cfg['domain']} freq={cfg['freq']} "
          f"context={context} horizon={horizon}\n{'=' * 78}")

    # ---- 数据（TEST 不加载）----
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
    train_ts = [dates[s + context] for s in train_starts]
    stats = compute_training_stats(X_train_all)

    evidence = {
        "feature_distributions": v1._raw_feature_quantiles(X_train_all),
        "temporal_coverage": v1._temporal_coverage(dates.iloc[:int(len(train_starts) + context)]),
    }

    print(f"data: bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)}")
    print(f"TEST accessed = False")

    # ---- 1. Raw Experts 生成（LLM）----
    model = av2._default_model()
    raw_experts, cached = generate_raw_experts(cfg, model)
    raw_experts = v1._sorted_experts(raw_experts)[:N_RAW_TARGET]
    for e in raw_experts:
        e["claim"] = e.get("mechanism", e.get("reason", ""))
        e["observable_features"] = list(e.get("observables", []))
        e["replaced"] = False
        e["compiled"] = v1.compile_condition_v6(e, v1.FEATURE_REGISTRY, evidence, v1.COMPILER_CONFIG)
        e["retrieval_spec"] = v1.derive_retrieval_spec(e, v1.FEATURE_REGISTRY)
    print(f"[Round 0] generated {len(raw_experts)} raw experts")

    # ---- 2. 闭环（Round 0/1/2）----
    lineage_best = {}
    lineage_coverage = {}
    reward_history_rows = []
    lineage_rows = []
    round_files = []

    experts = list(raw_experts)
    for rnd in range(0, v1.MAX_REVISION_ROUNDS + 1):
        print(f"\n[Round {rnd}] process {len(experts)} experts")
        round_records = []
        for e in experts:
            e["compiled"] = v1.compile_condition_v6(e, v1.FEATURE_REGISTRY, evidence, v1.COMPILER_CONFIG)
            e["retrieval_spec"] = v1.derive_retrieval_spec(e, v1.FEATURE_REGISTRY)
            parent_cov = lineage_coverage.get(e["id"])
            ev = v1.verify_one_v1(e, evidence, train_starts, series, dates, stats,
                                  X_train_all, train_ts, parent_coverage=parent_cov)
            reward = ev["reward"]
            verdict = v1.verdict_from_reward_v1(reward)
            lid = e["id"]
            rec = {"expert": e, "evidence": ev, "reward": reward, "verdict": verdict,
                   "round": rnd, "lineage_id": lid}
            round_records.append(rec)
            lineage_best[lid] = rec
            lineage_coverage[lid] = ev["activation_coverage"]
            print(f"  {lid:<4} reward={reward:+d} verdict={verdict:<10} "
                  f"cov={ev['activation_coverage']:.3f} "
                  f"ae={ev['action_effect_size'] if ev['action_effect_size'] is not None else float('nan'):+.4f}")

        critiques, _ = v1.llm_critic(round_records, model)
        crit_by_id = {c.get("expert_id"): c for c in critiques}
        for rec in round_records:
            lid = rec["lineage_id"]
            crit = crit_by_id.get(lid, {})
            crit.setdefault("verdict", rec["verdict"])
            crit.setdefault("diagnosis", v1._fallback_diagnosis(rec))
            crit.setdefault("supported_parts", [])
            crit.setdefault("failed_parts", [])
            crit.setdefault("retrieval_action_effect", "")
            crit.setdefault("coverage_effect", "")
            crit.setdefault("revision_instruction", "")
            rec["critic"] = crit
            rec["critic"]["verdict"] = rec["verdict"]

        round_files.append({"round": rnd,
                            "experts": [dict(r["expert"]) for r in round_records],
                            "evidence": [r["evidence"] for r in round_records],
                            "critic": [r["critic"] for r in round_records]})

        for rec in round_records:
            ev = rec["evidence"]
            e = rec["expert"]
            reward_history_rows.append({
                "lineage_id": rec["lineage_id"], "round": rnd, "reward": rec["reward"],
                "evidence_status": ev["evidence_status"], "retrieval_status": ev["retrieval_status"],
                "action_effect_size": ev.get("action_effect_size"),
                "positive_fold_ratio": ev.get("positive_fold_ratio"),
                "median_gain": ev.get("median_gain"), "worst_gain": ev.get("worst_gain"),
                "mean_gain": ev.get("mean_gain"), "activation_coverage": ev.get("activation_coverage"),
                "coverage_change_vs_parent": ev.get("coverage_change_vs_parent"),
                "verdict": rec["verdict"],
            })
            lineage_rows.append({
                "lineage_id": rec["lineage_id"], "round": rnd,
                "mechanism": e.get("mechanism", ""), "scope": e.get("scope", ""),
                "reward": rec["reward"], "verdict": rec["verdict"],
                "mean_gain": ev.get("mean_gain"), "median_gain": ev.get("median_gain"),
                "positive_fold_ratio": ev.get("positive_fold_ratio"),
                "worst_gain": ev.get("worst_gain"), "action_effect_size": ev.get("action_effect_size"),
                "activation_coverage": ev.get("activation_coverage"),
                "coverage_change_vs_parent": ev.get("coverage_change_vs_parent"),
                "retrieval_status": ev["retrieval_status"], "evidence_status": ev["evidence_status"],
                "replaced": bool(e.get("replaced", False)),
            })

        to_revise = [r for r in round_records if r["verdict"] in ("REVISE", "UNSUPPORTED", "REJECT")]
        if rnd >= v1.MAX_REVISION_ROUNDS or not to_revise:
            break
        print(f"  -> revise {len(to_revise)} experts")
        revised_experts, _ = v1.llm_revise(to_revise, rnd + 1, model)
        revised_by_id = {e.get("id"): e for e in revised_experts}
        next_experts = []
        for rec in to_revise:
            lid = rec["lineage_id"]
            ne = revised_by_id.get(lid)
            if ne is None:
                ne = dict(rec["expert"])
            ne["id"] = lid
            ne["claim"] = ne.get("mechanism", ne.get("reason", ""))
            ne["observable_features"] = list(ne.get("observables", []))
            ne["revision_round"] = rnd + 1
            ne["replaced"] = bool(ne.get("replaced", False))
            ne["retrieval_spec"] = v1.derive_retrieval_spec(ne, v1.FEATURE_REGISTRY)
            next_experts.append(ne)
        experts = next_experts

    # ---- 3. Final pool = reward>0 ----
    final_pool = []
    for lid, rec in lineage_best.items():
        if rec["reward"] > 0:
            final_pool.append(rec["expert"])
    final_pool = v1._sorted_experts(final_pool)
    print(f"\nFinal pool: {len(final_pool)} experts -> {[e['id'] for e in final_pool]}")

    # ---- 4. VAL ablation（冻结后首次访问 VAL，deterministic）----
    emb_raw = v1.embed_claims([e["claim"] for e in raw_experts], device)
    emb_final = v1.embed_claims([e["claim"] for e in final_pool], device) if final_pool else \
        np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    expert_dim = emb_raw.shape[1]
    n_raw, n_final = len(raw_experts), len(final_pool)
    n_slots = max(n_raw, n_final)

    raw_act = v1.act_mats(raw_experts, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    final_act = v1.act_mats(final_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats) if final_pool else {
        "bank": np.zeros((len(X_bank), 0), dtype=np.float32),
        "fit": np.zeros((len(X_fit), 0), dtype=np.float32),
        "val": np.zeros((len(X_val), 0), dtype=np.float32)}
    emb_raw, raw_act, _ = v1.pad_to_slots(emb_raw, raw_act, n_slots)
    emb_final, final_act, _ = v1.pad_to_slots(emb_final, final_act, n_slots)
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_act = {k: np.zeros_like(raw_act[k]) for k in raw_act}

    configs = {
        "base": (emb_zero, zero_act, False),
        "shape": (emb_zero, zero_act, True),
        "raw_expert": (emb_raw, raw_act, True),
        "reward_feedback_expert": (emb_final, final_act, True),
    }

    # 复现检查：D seed2026 x2
    det_ok = True
    det_mse = []
    emb_d, act_d, _ = configs["reward_feedback_expert"]
    for _ in range(2):
        seed_everything(2026)
        base_model = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        mm, _ = run_arm(init_state, emb_d, act_d, True, expert_dim, n_slots, device, 2026)
        det_mse.append(mm)
    if abs(det_mse[0] - det_mse[1]) > 1e-4:
        det_ok = False
        print(f"  [WARN] determinism check FAILED: {det_mse}")
    print(f"  determinism check: D seed2026 x2 = {det_mse} -> {'PASS' if det_ok else 'FAIL'}")

    per_seed = {nm: {} for nm in configs}
    param_count = None
    for seed in SEEDS:
        seed_everything(seed)
        base_model = ConditionalExpertMLPRetriever(
            context=context, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        param_count = v1.n_params(base_model)
        for nm, (emb, act, use_shape) in configs.items():
            mm, aa = run_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed)
            per_seed[nm][seed] = mm
            print(f"  seed={seed} [{nm}] MSE={mm:.4f}")

    summary = {}
    for nm in configs:
        ms = [per_seed[nm][s] for s in SEEDS]
        summary[nm] = {"mean": float(np.mean(ms)), "std": float(np.std(ms)),
                       "seeds": {s: per_seed[nm][s] for s in SEEDS}}

    # ---- 5. 写 per-dataset 输出 ----
    (out_dir / "raw_experts.json").write_text(
        json.dumps(raw_experts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "final_expert_pool.json").write_text(
        json.dumps(final_pool, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for rf in round_files:
        rnd = rf["round"]
        (out_dir / f"round{rnd}_evidence.json").write_text(
            json.dumps(rf["evidence"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out_dir / f"round{rnd}_critic.json").write_text(
            json.dumps(rf["critic"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if rnd >= 1:
            (out_dir / f"round{rnd}_experts.json").write_text(
                json.dumps(rf["experts"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _csv(fname, rows, cols):
        with (out_dir / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})

    _csv("reward_history.csv", reward_history_rows,
         ["lineage_id", "round", "reward", "evidence_status", "retrieval_status",
          "action_effect_size", "positive_fold_ratio", "median_gain", "worst_gain",
          "mean_gain", "activation_coverage", "coverage_change_vs_parent", "verdict"])
    _csv("knowledge_lineage.csv", lineage_rows,
         ["lineage_id", "round", "mechanism", "scope", "reward", "verdict", "mean_gain",
          "median_gain", "positive_fold_ratio", "worst_gain", "action_effect_size",
          "activation_coverage", "coverage_change_vs_parent", "retrieval_status",
          "evidence_status", "replaced"])

    with (out_dir / "val_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "mse"])
        for nm in configs:
            for s in SEEDS:
                w.writerow([nm, s, f"{per_seed[nm][s]:.6f}"])
        w.writerow([])
        w.writerow(["arm", "mse_mean", "mse_std"])
        for nm in configs:
            w.writerow([nm, f"{summary[nm]['mean']:.6f}", f"{summary[nm]['std']:.6f}"])

    # 统计：raw/final 条数 + verdict 计数
    keep = sum(1 for r in lineage_rows if r["verdict"] == "KEEP")
    revise = sum(1 for r in lineage_rows if r["verdict"] == "REVISE")
    reject = sum(1 for r in lineage_rows if r["verdict"] == "REJECT")
    unsupported = sum(1 for r in lineage_rows if r["verdict"] == "UNSUPPORTED")

    # 每 seed 各 arm 完整记录（供跨数据集汇总）
    arm_seed = []
    for nm in configs:
        for s in SEEDS:
            arm_seed.append({"dataset": name, "arm": nm, "seed": s, "mse": per_seed[nm][s]})

    result = {
        "dataset": name, "context": context, "horizon": horizon, "domain": cfg["domain"],
        "freq": cfg["freq"], "det_ok": det_ok, "param_count": param_count,
        "summary": summary, "n_raw": n_raw, "n_final": n_final,
        "n_keep": keep, "n_revise": revise, "n_reject": reject, "n_unsupported": unsupported,
        "arm_seed": arm_seed,
        "d_vs_c_pct": (summary["raw_expert"]["mean"] - summary["reward_feedback_expert"]["mean"])
                       / summary["raw_expert"]["mean"] * 100.0 if summary["raw_expert"]["mean"] else None,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  -> saved to {out_dir}")
    return result


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def aggregate(results):
    rows = []
    for r in results:
        s = r["summary"]
        raw_m = s["raw_expert"]["mean"]
        rf_m = s["reward_feedback_expert"]["mean"]
        wins = sum(1 for sd in SEEDS if s["reward_feedback_expert"]["seeds"][sd] < s["raw_expert"]["seeds"][sd])
        rows.append({
            "dataset": r["dataset"],
            "base_mean_mse": s["base"]["mean"],
            "shape_mean_mse": s["shape"]["mean"],
            "raw_mean_mse": raw_m,
            "reward_feedback_mean_mse": rf_m,
            "reward_feedback_std": s["reward_feedback_expert"]["std"],
            "D_vs_C_pct": (raw_m - rf_m) / raw_m * 100.0,
            "D_vs_B_pct": (s["shape"]["mean"] - rf_m) / s["shape"]["mean"] * 100.0,
            "D_vs_A_pct": (s["base"]["mean"] - rf_m) / s["base"]["mean"] * 100.0,
            "wins_vs_raw_seeds": wins,
            "n_raw_experts": r["n_raw"],
            "n_final_experts": r["n_final"],
            "n_keep": r["n_keep"],
            "n_revise": r["n_revise"],
            "n_reject": r["n_reject"],
            "n_unsupported": r["n_unsupported"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of dataset names (default: all)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    targets = [d for d in DATASETS if (args.datasets is None or d["name"] in args.datasets)]

    results = []
    for cfg in targets:
        try:
            r = run_dataset(dict(cfg), device)
            results.append(r)
        except Exception as e:
            print(f"\n[DATASET FAILED] {cfg['name']}: {e}")
            traceback.print_exc()
            results.append({
                "dataset": cfg["name"], "error": str(e),
                "summary": {nm: {"mean": None, "std": None,
                                 "seeds": {s: None for s in SEEDS}} for nm in
                            ("base", "shape", "raw_expert", "reward_feedback_expert")},
                "n_raw": None, "n_final": None, "n_keep": None, "n_revise": None,
                "n_reject": None, "n_unsupported": None, "det_ok": False,
            })

    # ---- 跨数据集汇总 ----
    ok = [r for r in results if "error" not in r]
    sum_rows = aggregate(ok)

    arm_rows = []
    for r in results:
        if "arm_seed" in r:
            arm_rows.extend(r["arm_seed"])

    def _csv(fname, rows, cols):
        with (OUT_ROOT / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})

    _csv("cross_dataset_val_ablation.csv", arm_rows, ["dataset", "arm", "seed", "mse"])
    _csv("cross_dataset_summary.csv", sum_rows, [
        "dataset", "base_mean_mse", "shape_mean_mse", "raw_mean_mse",
        "reward_feedback_mean_mse", "reward_feedback_std", "D_vs_C_pct", "D_vs_B_pct",
        "D_vs_A_pct", "wins_vs_raw_seeds", "n_raw_experts", "n_final_experts",
        "n_keep", "n_revise", "n_reject", "n_unsupported"])

    # reward stats + expert pool stats
    reward_rows = []
    pool_rows = []
    for r in results:
        if "error" in r:
            continue
        d = r["dataset"]
        rp = OUT_ROOT / d / "reward_history.csv"
        if rp.exists():
            for row in csv.DictReader(rp.open()):
                row["dataset"] = d
                reward_rows.append(row)
        pool_rows.append({"dataset": d, "n_raw_experts": r["n_raw"], "n_final_experts": r["n_final"],
                          "n_keep": r["n_keep"], "n_revise": r["n_revise"],
                          "n_reject": r["n_reject"], "n_unsupported": r["n_unsupported"]})
    if reward_rows:
        cols = ["dataset", "lineage_id", "round", "reward", "verdict", "action_effect_size",
                "positive_fold_ratio", "median_gain", "mean_gain", "activation_coverage"]
        _csv("cross_dataset_reward_stats.csv", reward_rows, cols)
    _csv("cross_dataset_expert_pool_stats.csv", pool_rows,
         ["dataset", "n_raw_experts", "n_final_experts", "n_keep", "n_revise", "n_reject", "n_unsupported"])

    report = render_report(results, sum_rows)
    (OUT_ROOT / "cross_dataset_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("CROSS-DATASET SUMMARY")
    print("=" * 78)
    for row in sum_rows:
        print(f"{row['dataset']:<14} D={row['reward_feedback_mean_mse']:.2f} C={row['raw_mean_mse']:.2f} "
              f"D_vs_C={row['D_vs_C_pct']:+.2f}%  wins={row['wins_vs_raw_seeds']}/3  "
              f"final={row['n_final_experts']}")
    for r in results:
        if "error" in r:
            print(f"{r['dataset']:<14} FAILED: {r['error']}")
    print(f"\nSaved to {OUT_ROOT}")


def render_report(results, sum_rows):
    ok = [r for r in results if "error" not in r]
    ri = []
    wins_all = 0
    wins_23 = 0
    for row in sum_rows:
        ri.append(row["D_vs_C_pct"] / 100.0)
        if row["D_vs_C_pct"] > 0:
            wins_all += 1
        if row["wins_vs_raw_seeds"] >= 2:
            wins_23 += 1
    mean_ri = float(np.mean(ri)) if ri else 0.0
    median_ri = float(np.median(ri)) if ri else 0.0
    positive_ratio = wins_all / len(ri) if ri else 0.0

    # D std < C std 计数（需要原始 std）
    d_std_lower = 0
    for r in ok:
        s = r["summary"]
        if s["reward_feedback_expert"]["std"] < s["raw_expert"]["std"]:
            d_std_lower += 1

    full_wins = sum(1 for row in sum_rows if row["wins_vs_raw_seeds"] == 3)

    n_ok = len(ok)
    if mean_ri > 0 and median_ri > 0 and wins_all > n_ok / 2 and wins_23 >= n_ok / 2:
        concl = "STRONG SUCCESS"
    elif mean_ri > 0:
        concl = "PARTIAL SUCCESS"
    else:
        concl = "FAIL"

    L = []
    L.append("# Reward-Feedback Critic v1 — 跨数据集正式验证（冻结方法迁移）")
    L.append("")
    L.append("ETTh1 上冻结的 Reward-Feedback Critic v1 迁移到 6 个数据集，验证跨数据集泛化。")
    L.append("不访问 TEST、不改方法、不挑 seed、失败数据集保留并标记。")
    L.append("")

    L.append("## 结果")
    L.append("")
    L.append(f"- 一共跑了 {n_ok} 个数据集（失败 {len(results) - n_ok} 个）。")
    L.append(f"- D 比 C 好的数据集：{wins_all}/{n_ok}。")
    L.append(f"- mean relative improvement (RI_d = (MSE_C - MSE_D)/MSE_C) = {mean_ri*100:+.2f}%。")
    L.append(f"- median relative improvement = {median_ri*100:+.2f}%。")
    L.append(f"- 3/3 seeds 全胜（D<C）的数据集：{full_wins}/{n_ok}。")
    L.append(f"- 2/3 或 3/3 seeds 优于 C 的数据集：{wins_23}/{n_ok}。")
    L.append(f"- D 的 std < C 的 std 的数据集：{d_std_lower}/{n_ok}。")
    if full_wins < n_ok:
        fail_ds = [row["dataset"] for row in sum_rows if row["wins_vs_raw_seeds"] < 3]
        L.append(f"- 未全胜（部分失败）数据集：{fail_ds}。")
    L.append("")

    L.append("主表：")
    L.append("")
    L.append("| Dataset | BASE | Shape | Raw Expert | Reward-Feedback | D vs C | wins/seeds |")
    L.append("|---|---|---|---|---|---|---|")
    for row in sum_rows:
        L.append(f"| {row['dataset']} | {row['base_mean_mse']:.2f} | {row['shape_mean_mse']:.2f} "
                 f"| {row['raw_mean_mse']:.2f} | {row['reward_feedback_mean_mse']:.2f} "
                 f"| {row['D_vs_C_pct']:+.2f}% | {row['wins_vs_raw_seeds']}/3 |")
    L.append("")

    L.append("| Dataset | Raw experts | Final experts | KEEP | REVISE | REJECT | UNSUPPORTED |")
    L.append("|---|---|---|---|---|---|---|")
    for row in sum_rows:
        L.append(f"| {row['dataset']} | {row['n_raw_experts']} | {row['n_final_experts']} "
                 f"| {row['n_keep']} | {row['n_revise']} | {row['n_reject']} | {row['n_unsupported']} |")
    L.append("")

    for r in results:
        if "error" in r:
            L.append(f"### FAIL_DATASET: {r['dataset']}")
            L.append(f"- error: {r['error']}")
            L.append("- possible causes: no useful expert knowledge / action not executable / "
                     "reward mismatch / weak activation coverage / optimization sensitivity")
            L.append("")

    L.append("## 结论")
    L.append("")
    L.append(f"- **{concl}**")
    if concl == "STRONG SUCCESS":
        L.append("  （多数数据集 D<C，mean_RI>0，median_RI>0，至少一半数据集 2/3 或 3/3 seeds 优于 C。）")
    elif concl == "PARTIAL SUCCESS":
        L.append("  （mean_RI>0，但 dataset / seed 稳定性一般。）")
    else:
        L.append("  （mean_RI<=0，或多数数据集 D 明显劣于 C。）")
    L.append("")
    L.append("```")
    L.append("test_used = false")
    L.append("deterministic_protocol = true")
    L.append("method_frozen = true")
    L.append("```")
    return "\n".join(L)


if __name__ == "__main__":
    main()
