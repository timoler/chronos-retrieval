#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward-Feedback Critic v1 (ETTh1) — reward actually EXECUTES retrieval_action.

v0 的核心缺陷：reward 只读 activation condition 的 coverage 与 per-fold gain，
`retrieval_action`（“启用后到底怎么改变检索”）完全不进入 reward。于是 Expert
唯一能改变 reward 的杠杆是“加 AND 条件、收窄 coverage”，而真正有意义的检索动作
改动（短/长 lag、cross-season、cross-day-type、prefer similar level …）对 reward 不可见。

v1 修复：
- 编译 `retrieval_action` 成可执行的 retrieval_transform（activation_fn 与 retrieval_transform 分离）。
- evidence 里同时算 ACTION-ON（activation + retrieval_action）与 ACTION-OFF（仅 activation），
  得到 action_effect_size = 真正执行 action 带来的增量。
- reward 依据 action_effect_size / coverage_change_vs_parent / fold 指标，不再只奖励局部 gain。

约束（与 v0 一致）：
- 只 ETTh1；TRAIN 生成证据/反馈；VAL 仅闭环冻结后访问一次；TEST=False。
- 不训练 Critic 网络；不改 Chronos 主干；不加 Attention/Gate。
- Raw Expert 对照完全复用 v0 的 raw_experts.json（不重新生成、不改内容）。
- 3 seeds {42,123,2026}；同结构/参数量/初始化。

输出：results/reward_feedback_critic_v1_etth1/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

import run_expert_attention_retrieval_v2 as attn            # seed_all
from run_expert_embedding_mlp import build_windows, mse, mae
from run_solar_same_val_baselines import residualize
from run_conditional_expert_mlp_v4 import (
    ConditionalExpertMLPRetriever,
    retrieve_conditional,
    _linear_slope_vec,
    _lag1_autocorr_vec,
)
from run_etth1_prediction_validation import (
    compute_training_stats,
    compute_window_feats,
    build_activation_matrix,
)
from run_expert_agent_closed_loop_v6 import (
    compile_condition_v6,
    COMPILER_CONFIG,
    audit_observability_for_expert,
)
import run_agent_v2_electricity as av2                      # _call_llm / _list_from / _default_model
from sentence_transformers import SentenceTransformer

REPO_ROOT = HERE.parent
av2.CACHE_DIR = str(REPO_ROOT / "cache/llm_priors_reward_feedback_critic_v1")   # 隔离的 v1 LLM 缓存

OUT_DIR = REPO_ROOT / "results/reward_feedback_critic_v1_etth1"
V0_DIR = REPO_ROOT / "results/reward_feedback_critic_v0_etth1"
DATA_DIR = REPO_ROOT / "results/multidataset/ETTh1"

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32
N_RAW_TARGET = 8

# ---- Data Verifier / Reward 常量（绝不根据 VAL 调参）----
N_FOLD_BLOCKS = 5                 # warmup + 4 个 chronological folds
MIN_IN_SCOPE = 20                 # 每个 fold 至少多少 in-scope query 才算有效
MAX_REVISION_ROUNDS = 2           # 最多 2 轮 revision（round 1 / round 2）
# retrieval action 执行常量（固定，不看 VAL）
ACTION_WEIGHT = 1.0               # score = shape + ACTION_WEIGHT * action_score
PREF_SIM_WEIGHT = 0.5             # prefer_similar / prefer_same / lag 的默认权重
PREF_SAME_WEIGHT = 0.3
LAG_CAP_HOURS = 24.0 * 30.0       # lag "short" 偏好的归一化上限（30 天）
# v1 reward 阈值（固定，不看 VAL）
REWARD_AE_HIGH_PFR = 0.75
REWARD_AE_LOW_PFR = 0.50
REWARD_WORST_TOL = -0.05
REWARD_COV_COLLAPSE = -0.40       # coverage_change < -0.40 判定“只靠收窄刷分”

# --------------------------------------------------------------------------- #
# ETTh1 feature registry（与 v0 完全一致）
# --------------------------------------------------------------------------- #
FEATURE_REGISTRY = {
    "hour_of_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "categories": {"morning": [6, 7, 8, 9, 10, 11], "midday": [12, 13, 14, 15],
                                   "evening": [16, 17, 18, 19, 20, 21],
                                   "night": [22, 23, 0, 1, 2, 3, 4, 5]}},
    "day_of_week": {"kind": "calendar", "source_type": "timestamp_known",
                    "categories": {"weekday": [0, 1, 2, 3, 4], "weekend": [5, 6]}},
    "working_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "categories": {"workday": [0, 1, 2, 3, 4], "non_working": [5, 6]}},
    "seasonal_level": {"kind": "calendar", "source_type": "timestamp_known",
                       "categories": {"winter": [12, 1, 2], "spring": [3, 4, 5],
                                      "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
    "recent_level": {"kind": "continuous", "source_type": "context_derived"},
    "recent_volatility": {"kind": "continuous", "source_type": "context_derived"},
    "recent_slope_magnitude": {"kind": "continuous", "source_type": "context_derived"},
    "recent_diff_magnitude": {"kind": "continuous", "source_type": "context_derived"},
    "recent_autocorr": {"kind": "continuous", "source_type": "context_derived"},
}

CONTINUOUS_FEATURES = [f for f, m in FEATURE_REGISTRY.items() if m["kind"] == "continuous"]
CALENDAR_FEATURES = [f for f, m in FEATURE_REGISTRY.items() if m["kind"] == "calendar"]


# --------------------------------------------------------------------------- #
# 通用工具（复用 v0）
# --------------------------------------------------------------------------- #
def _topk_mean_forecast(Xq, Xbank, Ybank, topk, q_chunk=256, bank_chunk=4096):
    """残差 Pearson Top-K 均值（memory-only，无 mean-anchor，quiet）。"""
    Q = residualize(Xq)
    B = residualize(Xbank)
    L = Q.shape[1]
    out = np.empty((len(Q), Ybank.shape[1]), dtype=np.float32)
    for qs in range(0, len(Q), q_chunk):
        q = Q[qs:qs + q_chunk]
        nq = len(q)
        best_scores = np.full((nq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nq, topk), -1, dtype=np.int64)
        for bs in range(0, len(B), bank_chunk):
            b = B[bs:bs + bank_chunk]
            s = (q @ b.T) / L
            kk = min(topk, s.shape[1])
            lp = np.argpartition(s, -kk, axis=1)[:, -kk:]
            ls = np.take_along_axis(s, lp, axis=1)
            li = lp + bs
            comb_s = np.concatenate([best_scores, ls], axis=1)
            comb_i = np.concatenate([best_idx, li], axis=1)
            part = np.argpartition(comb_s, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(comb_s, part, axis=1)
            best_idx = np.take_along_axis(comb_i, part, axis=1)
        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_idx = np.take_along_axis(best_idx, order, axis=1)
        out[qs:qs + q_chunk] = Ybank[best_idx].mean(axis=1)
    return out


def _topk_forecast_action(Xq, Xbank, Ybank, topk, ops, feats_q, feats_bank,
                          lag_q, lag_bank, scope_mask, q_chunk=256, bank_chunk=4096):
    """ACTION-ON forecast：shape score + 可执行 retrieval_transform。

    - scope_mask: per-candidate bool（activation scope）。cross op 会把 scope 放宽到全 bank。
    - soft ops（prefer_similar/prefer_same/lag）作为加性分；hard op（restrict_same）作为 per-(q,c) mask。
    """
    Q = residualize(Xq)
    B = residualize(Xbank)
    L = Q.shape[1]
    nq = len(Q)
    nb = len(B)
    soft = [o for o in ops if o["type"] in ("prefer_similar", "prefer_same", "lag")]
    hard = [o for o in ops if o["type"] == "restrict_same"]
    cross = any(o.get("type") == "cross" for o in ops)
    base_ok = np.ones(nb, dtype=bool) if cross else np.asarray(scope_mask, dtype=bool).copy()
    out = np.empty((nq, Ybank.shape[1]), dtype=np.float32)
    for qs in range(0, nq, q_chunk):
        q = Q[qs:qs + q_chunk]
        nqq = len(q)
        best_scores = np.full((nqq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nqq, topk), -1, dtype=np.int64)
        for bs in range(0, nb, bank_chunk):
            b = B[bs:bs + bank_chunk]
            s = (q @ b.T) / L
            # soft additive
            for o in soft:
                t = o["type"]
                w = float(o.get("weight", PREF_SIM_WEIGHT))
                if t == "prefer_similar":
                    f = o["feature"]
                    d = np.abs(feats_q[f][qs:qs + q_chunk][:, None]
                               - feats_bank[f][bs:bs + bank_chunk][None, :])
                    s = s - w * np.minimum(d, 3.0) / 3.0
                elif t == "prefer_same":
                    f = o["feature"]
                    eq = (feats_q[f][qs:qs + q_chunk][:, None]
                          == feats_bank[f][bs:bs + bank_chunk][None, :]).astype(np.float32)
                    s = s + w * (2.0 * eq - 1.0)
                elif t == "lag":
                    if o.get("direction") == "short":
                        cap = float(o.get("cap", LAG_CAP_HOURS))
                        lag = np.abs(lag_q[qs:qs + q_chunk][:, None]
                                     - lag_bank[bs:bs + bank_chunk][None, :])
                        s = s - w * np.minimum(lag, cap) / cap
                    # allow_long: no-op
            # base candidate mask
            ok = np.broadcast_to(base_ok[bs:bs + bank_chunk][None, :], s.shape).copy()
            for o in hard:
                f = o["feature"]
                eq = (feats_q[f][qs:qs + q_chunk][:, None]
                      == feats_bank[f][bs:bs + bank_chunk][None, :])
                ok = ok & eq
            s = np.where(ok, s, -np.inf)
            kk = min(topk, s.shape[1])
            lp = np.argpartition(s, -kk, axis=1)[:, -kk:]
            ls = np.take_along_axis(s, lp, axis=1)
            li = lp + bs
            comb_s = np.concatenate([best_scores, ls], axis=1)
            comb_i = np.concatenate([best_idx, li], axis=1)
            part = np.argpartition(comb_s, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(comb_s, part, axis=1)
            best_idx = np.take_along_axis(comb_i, part, axis=1)
        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_idx = np.take_along_axis(best_idx, order, axis=1)
        out[qs:qs + q_chunk] = Ybank[best_idx].mean(axis=1)
    return out


def _raw_feature_quantiles(X):
    xm = X.mean(axis=1)
    xs = X.std(axis=1)
    slopes = np.abs(_linear_slope_vec(X))
    diffs = np.abs(X[:, -1] - X[:, -2])
    autocorr = _lag1_autocorr_vec(X)
    feats = {"recent_level": xm, "recent_volatility": xs,
             "recent_slope_magnitude": slopes, "recent_diff_magnitude": diffs,
             "recent_autocorr": autocorr}
    dist = {}
    for f, a in feats.items():
        a = np.asarray(a, float)
        dist[f] = {"q10": float(np.percentile(a, 10)), "q25": float(np.percentile(a, 25)),
                   "q50": float(np.percentile(a, 50)), "q75": float(np.percentile(a, 75)),
                   "q90": float(np.percentile(a, 90)), "mean": float(a.mean()), "std": float(a.std())}
    return dist


def _temporal_coverage(dt_train):
    months = dt_train.dt.month.to_numpy()
    hours = dt_train.dt.hour.to_numpy()
    dows = dt_train.dt.dayofweek.to_numpy()
    return {
        "months_present": sorted(int(m) for m in set(months.tolist())),
        "hours_present": sorted(int(h) for h in set(hours.tolist())),
        "dows_present": sorted(int(d) for d in set(dows.tolist())),
    }


def _sorted_experts(pool):
    items = list(pool.values()) if isinstance(pool, dict) else list(pool)
    return sorted(items, key=lambda e: int(str(e.get("id", e.get("expert_id", "0"))).lstrip("E") or 0))


def _hours_since(ts_list, ref):
    return np.asarray([(ts - ref).total_seconds() / 3600.0 for ts in ts_list], dtype=np.float64)


# --------------------------------------------------------------------------- #
# retrieval_action 编译（确定性启发式；无法执行 → COMPILER_UNSUPPORTED）
# --------------------------------------------------------------------------- #
def derive_retrieval_spec(expert, registry):
    """从 retrieval_action + observables + condition_spec 确定性推导可执行 ops。

    只依赖 query/candidate 的当前/历史可观测特征。若 action 引用了 registry 外的特征
    或未来信息，标记 COMPILER_UNSUPPORTED（不静默忽略）。
    """
    action = (expert.get("retrieval_action") or "").lower()
    observables = set(expert.get("observables") or [])
    cond = (expert.get("condition_spec") or {}).get("conditions", [])
    cond_features = {c.get("feature") for c in cond}
    ops = []
    notes = []
    unsupported = []

    # 引用 registry 外特征 -> unsupported
    for f in observables:
        if f not in registry:
            unsupported.append(f"observable '{f}' not in registry")
    for m in re.findall(r"recent_\w+|hour_of_day|working_day|seasonal_level|day_of_week", action):
        if m not in registry:
            unsupported.append(f"retrieval_action references unknown feature '{m}'")

    if unsupported:
        return {"status": "COMPILER_UNSUPPORTED", "ops": [], "notes": [],
                "uncompilable": unsupported}

    # “do-nothing” revision（回退 baseline）→ 空 ops，不误判为主动作
    if re.search(r"do not apply|do not use|fall back to (the )?default|default shape retrieval|no longer apply|remove this rule", action):
        return {"status": "COMPILED", "ops": [],
                "notes": ["action is to fall back to baseline (no-op)"], "uncompilable": []}

    # cross-season / cross-day-type（放宽 activation scope）——仅肯定式，避免 "instead of pooling all seasons" 误判
    if re.search(
        r"(?:from|across|use|over)\s+all[- ]?seasons"
        r"|all[- ]?season\s+retrieval"
        r"|cross[- ]?season"
        r"|(?:do\s+not|don't|no\s+longer)\s+restrict\b.{0,30}summer"
        r"|not\b.{0,20}summer[- ]?only", action):
        if "seasonal_level" in observables or "seasonal_level" in cond_features:
            ops.append({"type": "cross", "feature": "seasonal_level"})
    if re.search(
        r"(?:from|across|use|over)\s+both\b.{0,30}(?:workday|day[- ]?types)"
        r"|both\s+workday\s+and\s+non[- ]?working"
        r"|cross[- ]?day"
        r"|across\s+day[- ]?types"
        r"|(?:do\s+not|don't)\s+restrict\b.{0,30}workday", action):
        if "working_day" in observables or "working_day" in cond_features:
            ops.append({"type": "cross", "feature": "working_day"})

    # lag preference
    if re.search(r"short\b.{0,15}\b(?:window|horizon|lag)|recent local|shorter lag|only short|do not extend", action):
        ops.append({"type": "lag", "direction": "short", "weight": PREF_SIM_WEIGHT, "cap": LAG_CAP_HOURS})
    elif re.search(r"longer (retrieval )?window|allow longer|longer lag|extend retrieval", action):
        ops.append({"type": "lag", "direction": "allow_long"})

    # prefer_similar：连续特征；需相似动词在 feature 之前；跳过已在 activation condition 里的 feature（避免与 scope 重复）
    for f in CONTINUOUS_FEATURES:
        if f in cond_features:
            continue
        if re.search(rf"(?:similar|match|matching|same|prioritiz\w*|prefer|align)\b.{{0,20}}\b{f}\b", action):
            ops.append({"type": "prefer_similar", "feature": f, "weight": PREF_SIM_WEIGHT})

    # prefer_same / restrict_same：日历特征（restrict 只对“restrict/limit to <feature>”生效）
    for f in CALENDAR_FEATURES:
        if re.search(rf"(?:restrict|limit)\b.{{0,30}}\b{f}\b", action):
            ops.append({"type": "restrict_same", "feature": f})
        elif re.search(rf"(?:same|similar|match|matching|prioritiz\w*|prefer|align)\b.{{0,20}}\b{f}\b", action):
            ops.append({"type": "prefer_same", "feature": f, "weight": PREF_SAME_WEIGHT})

    # 去重（同 type+feature）
    seen, dedup = set(), []
    for o in ops:
        k = (o.get("type"), o.get("feature"), o.get("direction"))
        if k not in seen:
            seen.add(k)
            dedup.append(o)
    ops = dedup

    if not ops:
        notes.append("retrieval_action subsumed by activation scope; no additional executable transform")

    return {"status": "COMPILED", "ops": ops, "notes": notes, "uncompilable": []}


# --------------------------------------------------------------------------- #
# v1 Reward（固定常量，绝不根据 VAL 调参）
# --------------------------------------------------------------------------- #
def compute_reward_v1(evidence_status, retrieval_status, action_effect_size,
                      positive_fold_ratio, worst_gain, coverage_change_vs_parent, median_gain):
    if evidence_status in ("UNSUPPORTED",) or retrieval_status == "COMPILER_UNSUPPORTED":
        return 0
    if action_effect_size is None or positive_fold_ratio is None:
        return 0
    if action_effect_size > 0 and positive_fold_ratio >= REWARD_AE_HIGH_PFR and worst_gain >= REWARD_WORST_TOL:
        return 2
    if action_effect_size > 0 and positive_fold_ratio >= REWARD_AE_LOW_PFR:
        return 1
    if coverage_change_vs_parent is not None and coverage_change_vs_parent < REWARD_COV_COLLAPSE and action_effect_size <= 0:
        return -2
    if median_gain < 0:
        return -1
    return 0


def verdict_from_reward_v1(r):
    if r in (1, 2):
        return "KEEP"
    if r == 0:
        return "UNSUPPORTED"
    if r == -1:
        return "REVISE"
    return "REJECT"


# --------------------------------------------------------------------------- #
# TRAIN-only Data Verifier（v1：执行 retrieval_action，算 action_effect）
# --------------------------------------------------------------------------- #
def verify_one_v1(expert, evidence, train_starts, series, dates, stats, X_train_all, train_ts,
                  parent_coverage=None):
    lid = expert.get("id", expert.get("expert_id"))

    observables = list(expert.get("observables", []))
    obs_exist = all(f in FEATURE_REGISTRY for f in observables)
    obs_audit = audit_observability_for_expert(expert, FEATURE_REGISTRY)
    all_observable = bool(obs_audit["all_observable"])
    compiled = expert.get("compiled", {})
    compilable = compiled.get("status") == "COMPILED"
    retrieval = expert.get("retrieval_spec", {})
    retrieval_status = retrieval.get("status", "COMPILER_UNSUPPORTED")
    ops = retrieval.get("ops", [])

    unverifiable_reasons = []
    if not obs_exist:
        unverifiable_reasons.append("observable feature not in registry")
    if not all_observable:
        unverifiable_reasons.append("observable depends on unavailable/future input")
    if not compilable:
        unverifiable_reasons.append("activation condition uncompilable: " + str(compiled.get("uncompilable", [])))
    if retrieval_status == "COMPILER_UNSUPPORTED":
        unverifiable_reasons.append("retrieval_action uncompilable: " + str(retrieval.get("uncompilable", [])))

    act_all = build_activation_matrix([expert], X_train_all, train_ts, stats)[:, 0]
    activation_coverage = float((act_all >= 0.5).mean())
    n_supported_queries = int((act_all >= 0.5).sum())

    base = {
        "expert_id": lid,
        "evidence_status": "UNSUPPORTED" if unverifiable_reasons else "SUPPORTED",
        "unverifiable_reasons": unverifiable_reasons,
        "retrieval_status": retrieval_status,
        "retrieval_ops": ops,
        "activation_coverage": activation_coverage,
        "n_supported_queries": n_supported_queries,
        "fold_gains": [],
        "mean_gain": None, "median_gain": None, "positive_fold_ratio": None,
        "worst_gain": None, "std_gain": None, "action_effect_size": None,
        "coverage_change_vs_parent": None, "reward": 0,
    }

    if parent_coverage is not None and parent_coverage > 0:
        base["coverage_change_vs_parent"] = (activation_coverage - parent_coverage) / parent_coverage
    else:
        base["coverage_change_vs_parent"] = 0.0

    if unverifiable_reasons:
        base["reward"] = compute_reward_v1(base["evidence_status"], retrieval_status, None,
                                           None, None, base["coverage_change_vs_parent"], None)
        return base

    n = len(train_starts)
    edges = np.linspace(0, n, N_FOLD_BLOCKS + 1).astype(int)
    ref = dates[0]
    fold_gains = []
    for f in range(1, N_FOLD_BLOCKS):
        bank_idx = train_starts[:edges[f]]
        q_idx = train_starts[edges[f]:edges[f + 1]]
        X_bank, Y_bank = build_windows(series, bank_idx, CONTEXT, HORIZON)
        X_q, Y_q = build_windows(series, q_idx, CONTEXT, HORIZON)
        bank_ts = [dates[s + CONTEXT] for s in bank_idx]
        q_ts = [dates[s + CONTEXT] for s in q_idx]
        act_bank = build_activation_matrix([expert], X_bank, bank_ts, stats)[:, 0]
        act_q = build_activation_matrix([expert], X_q, q_ts, stats)[:, 0]
        in_scope = act_q >= 0.5
        n_in = int(in_scope.sum())
        row = {"fold": f, "n_in_scope": n_in, "covered": n_in >= MIN_IN_SCOPE}
        if not row["covered"]:
            fold_gains.append(row)
            continue

        scope_bank = act_bank >= 0.5
        shape_pred = _topk_mean_forecast(X_q, X_bank, Y_bank, TOPK)

        # ACTION-OFF：仅 activation scope 限制 bank，shape score
        act_only_pred = shape_pred.copy()
        if int(scope_bank.sum()) >= TOPK:
            act_only_pred[in_scope] = _topk_mean_forecast(
                X_q[in_scope], X_bank[scope_bank], Y_bank[scope_bank], TOPK)

        # ACTION-ON：activation scope + retrieval_transform（scope 不足 TOPK 时回退 shape，与 act_only 同 guard）
        action_pred = act_only_pred[in_scope].copy()
        if int(scope_bank.sum()) >= TOPK:
            feats_q = compute_window_feats(X_q, q_ts, stats)
            feats_bank = compute_window_feats(X_bank, bank_ts, stats)
            lag_q = _hours_since(q_ts, ref)
            lag_bank = _hours_since(bank_ts, ref)
            feats_q_in = {k: v[in_scope] for k, v in feats_q.items()}
            action_pred = _topk_forecast_action(
                X_q[in_scope], X_bank, Y_bank, TOPK, ops,
                feats_q_in, feats_bank, lag_q[in_scope], lag_bank, scope_bank)

        ms = mse(Y_q[in_scope], shape_pred[in_scope])
        mo = mse(Y_q[in_scope], act_only_pred[in_scope])
        ma = mse(Y_q[in_scope], action_pred)

        row.update({
            "mse_shape": ms, "mse_act_only": mo, "mse_action_on": ma,
            "gain_act_only": (ms - mo) / ms if ms > 0 else 0.0,
            "gain_action_on": (ms - ma) / ms if ms > 0 else 0.0,
            "candidate_retention": float(scope_bank.mean())
            if not any(o.get("type") == "cross" for o in ops) else 1.0,
        })
        row["action_effect"] = row["gain_action_on"] - row["gain_act_only"]
        fold_gains.append(row)

    covered = [g for g in fold_gains if g.get("covered")]
    if len(covered) == 0:
        base["evidence_status"] = "UNSUPPORTED"
        base["unverifiable_reasons"].append("insufficient TRAIN activation coverage")
        base["reward"] = compute_reward_v1(base["evidence_status"], retrieval_status, None,
                                           None, None, base["coverage_change_vs_parent"], None)
        return base

    gains = np.array([g["gain_action_on"] for g in covered], float)
    ae = np.array([g["action_effect"] for g in covered], float)
    mean_gain = float(gains.mean())
    median_gain = float(np.median(gains))
    positive_fold_ratio = float((gains > 0).mean())
    worst_gain = float(gains.min())
    std_gain = float(gains.std())
    action_effect_size = float(ae.mean())

    base.update({
        "fold_gains": fold_gains,
        "mean_gain": mean_gain, "median_gain": median_gain,
        "positive_fold_ratio": positive_fold_ratio, "worst_gain": worst_gain,
        "std_gain": std_gain, "action_effect_size": action_effect_size,
    })
    base["reward"] = compute_reward_v1(base["evidence_status"], retrieval_status, action_effect_size,
                                       positive_fold_ratio, worst_gain,
                                       base["coverage_change_vs_parent"], median_gain)
    return base


# --------------------------------------------------------------------------- #
# LLM 提示词：Expert 生成 / Critic / Revision
# --------------------------------------------------------------------------- #
def _catalog_text():
    lines = ["AVAILABLE OBSERVABLES (choose 'observables' and 'condition_spec.feature' ONLY from this list):"]
    for f, meta in FEATURE_REGISTRY.items():
        if meta["kind"] == "calendar":
            cats = ", ".join(sorted(meta["categories"].keys()))
            lines.append(f"  - {f}: calendar (categories: {cats})")
        else:
            lines.append(f"  - {f}: continuous (qualifier: high or low)")
    return "\n".join(lines)


def _expert_schema_text():
    return (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"experts": [ ... ]}. Each entry has exactly these keys:\n'
        '  - "id": "E1", "E2", ...\n'
        '  - "mechanism": the physical/domain mechanism (1 sentence)\n'
        '  - "scope": where it applies (e.g. global / summer / high volatility / workday)\n'
        '  - "activation_condition": the precise "when [observable state]" condition\n'
        '  - "failure_condition": when it is expected to FAIL or should NOT apply\n'
        '  - "observables": list of feature names from AVAILABLE OBSERVABLES (only current/history info)\n'
        '  - "retrieval_action": how it concretely changes historical retrieval\n'
        '  - "expected_effect": why it should make the retrieved future better match\n'
        '  - "reason": mechanism rationale (1-2 sentences)\n'
        '  - "condition_spec": {"logic": "AND"|"OR", "conditions": [{"feature": "...", "qualifier": "..."}]}\n'
        '  - "retrieval_spec": {"ops": [{"type": ..., "feature": ...}]}  (OPTIONAL structured form of retrieval_action)\n'
        "\n"
        'retrieval_spec.ops use ONLY these types:\n'
        '  - {"type":"prefer_similar","feature":"recent_level|recent_volatility|recent_slope_magnitude|recent_diff_magnitude|recent_autocorr"}\n'
        '  - {"type":"prefer_same","feature":"hour_of_day|working_day|seasonal_level"}\n'
        '  - {"type":"restrict_same","feature":"hour_of_day|working_day|seasonal_level"}\n'
        '  - {"type":"cross","feature":"working_day|seasonal_level"}\n'
        '  - {"type":"lag","direction":"short"}\n'
        "\n"
        "For condition_spec.qualifier: continuous features use \"high\" or \"low\"; "
        "calendar features use the category name (e.g. \"summer\", \"night\", \"workday\").\n"
        "Do NOT invent features outside the catalog. Do NOT use future/target values as observables. "
        "Output only the JSON object."
    )


def _call_llm_list(system, user, stage, key, model, max_retries=3):
    for attempt in range(1, max_retries + 1):
        raw, path, cached = av2._call_llm(system, user, stage, model)
        try:
            return av2._list_from(raw, key), cached
        except (ValueError, KeyError) as e:
            if os.path.exists(path):
                os.remove(path)
            if attempt >= max_retries:
                raise
            user = user + (
                f"\n\n[IMPORTANT retry {attempt}: your previous output was truncated or "
                f"incomplete. Output the COMPLETE JSON object, ending with the final "
                f"closing brace. Do not stop mid-array or mid-string.]"
            )
    raise RuntimeError(f"LLM parse failed after {max_retries} retries (stage={stage})")


def load_raw_experts():
    """复用 v0 的 raw experts（不重新生成、不改内容），仅补 retrieval_spec。"""
    path = V0_DIR / "raw_experts.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for e in raw:
        keep = {k: e.get(k) for k in (
            "id", "mechanism", "scope", "activation_condition", "failure_condition",
            "observables", "retrieval_action", "expected_effect", "reason", "condition_spec")}
        keep["claim"] = keep.get("mechanism", keep.get("reason", ""))
        keep["observable_features"] = list(keep.get("observables", []))
        keep["replaced"] = False
        out.append(keep)
    return _sorted_experts(out)[:N_RAW_TARGET]


def _evidence_summary_text(rec):
    ev = rec["evidence"]
    e = rec["expert"]
    lines = [
        f"expert {e['id']}: mechanism={e.get('mechanism','')}",
        f"  scope={e.get('scope','')} | activation_condition={e.get('activation_condition','')}",
        f"  observables={e.get('observables',[])} | retrieval_action={e.get('retrieval_action','')}",
        f"  retrieval_status={ev.get('retrieval_status')} | ops={ev.get('retrieval_ops')}",
        f"  evidence_status={ev['evidence_status']} | auto_reward={rec['reward']}",
        f"  activation_coverage={ev['activation_coverage']:.4f} | n_supported_queries={ev['n_supported_queries']}",
    ]
    if ev["evidence_status"] == "SUPPORTED":
        lines += [
            f"  mean_gain(action_on)={ev['mean_gain']:.4f} | median_gain={ev['median_gain']:.4f} | "
            f"positive_fold_ratio={ev['positive_fold_ratio']:.4f} | worst_gain={ev['worst_gain']:.4f}",
            f"  action_effect_size={ev['action_effect_size']:.4f} | "
            f"coverage_change_vs_parent={ev['coverage_change_vs_parent']:.4f}",
            f"  fold_gains={ev['fold_gains']}",
        ]
    else:
        lines.append(f"  unverifiable_reasons={ev['unverifiable_reasons']}")
    return "\n".join(lines)


def llm_critic(round_records, model):
    auto_verdicts = {r["expert"]["id"]: r["verdict"] for r in round_records}
    body = "\n\n".join(_evidence_summary_text(r) for r in round_records)
    system = (
        "You are a rigorous critic for a retrieval-based forecaster. Given (a) an expert's "
        "structured knowledge, (b) deterministic TRAIN-only evidence INCLUDING an "
        "ACTION-ON vs ACTION-OFF comparison, and (c) an automatic reward, produce a per-expert "
        "diagnosis. You must NOT recompute data and must NOT invent evidence. "
        "Distinguish whether a gain comes from the retrieval_action itself (action_effect_size>0) "
        "or merely from narrowing the activation scope (coverage_change<-0.40 with action_effect<=0). "
        "If evidence is UNSUPPORTED/COMPILER_UNSUPPORTED you must say the evidence is insufficient — "
        "NOT that the mechanism is wrong. Output strict JSON only."
    )
    user = "\n\n".join([
        "AUTO VERDICTS (reward -> verdict: 2=KEEP, 1=KEEP, 0=UNSUPPORTED, -1=REVISE, -2=REJECT):\n"
        + json.dumps(auto_verdicts, ensure_ascii=False),
        "EVIDENCE (per expert):\n" + body,
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"critiques": [ ... ]}. Each entry:\n'
        '  - "expert_id": the expert id\n'
        '  - "verdict": must equal the AUTO VERDICT above (do not override the data)\n'
        '  - "supported_parts": [which claims the data supports]\n'
        '  - "failed_parts": [which specific claim/condition the data does NOT support]\n'
        '  - "retrieval_action_effect": "positive|none|negative" (did actually executing the action help)\n'
        '  - "coverage_effect": "stable|narrowed|widened" (did revision mainly narrow scope)\n'
        '  - "diagnosis": why (2-3 sentences, cite the evidence numbers)\n'
        '  - "revision_instruction": if REVISE/UNSUPPORTED/REJECT, tell the Expert exactly what to change; '
        "do NOT invent new data patterns, only refine based on the evidence shown.\n"
        "TASK: output only the JSON object.",
    ])
    return _call_llm_list(system, user, "etth1_rf_critic_v1_critic", "critiques", model)


def llm_revise(to_revise, revision_round, model):
    feedbacks = []
    for r in to_revise:
        e = r["expert"]
        crit = r["critic"]
        feedbacks.append({
            "expert_id": e["id"],
            "previous_knowledge": {k: e.get(k) for k in (
                "mechanism", "scope", "activation_condition", "failure_condition",
                "observables", "retrieval_action", "expected_effect", "reason", "condition_spec")},
            "evidence_summary": r["evidence"],
            "critic_diagnosis": crit.get("diagnosis", ""),
            "revision_instruction": crit.get("revision_instruction", ""),
        })
    system = (
        "You are a domain expert revising conditional retrieval knowledge after a critic review. "
        "Revise each failed expert so that its mechanism AND retrieval_action are consistent with the "
        "TRAIN evidence shown. You may change the retrieval_action (including its structured "
        "'retrieval_spec'), the mechanism, the scope, or the activation — but do NOT merely add AND "
        "conditions to escape bad folds unless the retrieval_action itself improves. If the mechanism "
        "cannot be verified on TRAIN (UNSUPPORTED), you MAY switch to a different TRAIN-verifiable "
        "mechanism and MUST set 'replaced': true. Do NOT invent data patterns beyond the evidence shown. "
        "Output strict JSON only."
    )
    user = "\n\n".join([
        _catalog_text(), _expert_schema_text(),
        "REVISION ROUND: " + str(revision_round),
        "FEEDBACK (per expert):\n" + json.dumps(feedbacks, ensure_ascii=False, indent=2),
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"experts": [ ... ]} with the SAME keys as the expert schema PLUS "replaced": true/false. '
        "Keep the same 'id'. Output only the JSON object.",
    ])
    experts, cached = _call_llm_list(system, user, "etth1_rf_critic_v1_revision", "experts", model)
    return experts, cached


# --------------------------------------------------------------------------- #
# 最终 VAL ablation（A/B/C/D，复用 v0 的训练/评估协议）
# --------------------------------------------------------------------------- #
def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


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
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        gamma = float(torch.sigmoid(model.logit_gamma).detach().cpu())
        beta = float(torch.sigmoid(model.logit_beta).detach().cpu())
        print(f"    epoch {ep:02d}/{epochs} loss={np.mean(losses):.6f} gamma={gamma:.4f} beta={beta:.4f}")
    return {"gamma": gamma, "beta": beta}


def act_mats(experts, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats):
    return {
        "bank": build_activation_matrix(experts, X_bank, bank_ts, stats),
        "fit": build_activation_matrix(experts, X_fit, fit_ts, stats),
        "val": build_activation_matrix(experts, X_val, val_ts, stats),
    }


# --------------------------------------------------------------------------- #
# v0 blindspot audit（E1/E5/E7）：raw activation + raw action vs raw activation + revised action
# --------------------------------------------------------------------------- #
def blindspot_audit(v0_raw, v0_r1, train_starts, series, dates, stats, X_train_all, train_ts, ref):
    """对 v0 的 E1/E5/E7：比较 原 Raw activation + 原 action vs 原 Raw activation + revision action。"""
    rows = []
    v0r1_by_id = {e["id"]: e for e in v0_r1}
    for raw in v0_raw:
        lid = raw["id"]
        if lid not in ("E1", "E5", "E7"):
            continue
        rev = v0r1_by_id.get(lid)
        if rev is None:
            continue
        # arm A: raw activation + raw action
        raw_expert = dict(raw)
        raw_expert["compiled"] = raw.get("compiled")
        raw_expert["retrieval_spec"] = derive_retrieval_spec(raw_expert, FEATURE_REGISTRY)
        # arm B: raw activation + revised action
        rev_expert = dict(raw)                       # keep raw activation/condition_spec/observables
        rev_expert["retrieval_action"] = rev.get("retrieval_action", raw.get("retrieval_action"))
        rev_expert["compiled"] = raw.get("compiled")
        rev_expert["retrieval_spec"] = derive_retrieval_spec(rev_expert, FEATURE_REGISTRY)

        g_raw = _fold_mean_gain(raw_expert, train_starts, series, dates, stats, ref)
        g_rev = _fold_mean_gain(rev_expert, train_starts, series, dates, stats, ref)
        rows.append({
            "expert_id": lid,
            "raw_action_gain": g_raw,
            "revised_action_gain": g_rev,
            "delta_action_gain": (g_rev - g_raw) if (g_raw is not None and g_rev is not None) else None,
            "same_activation": True,
            "v0_reward_changed": False,          # v0 evidence 三轮相同 -> reward 未变
            "v1_reward_changed": (g_raw != g_rev) if (g_raw is not None and g_rev is not None) else None,
        })
    return rows


def _fold_mean_gain(expert, train_starts, series, dates, stats, ref):
    """该 expert 的 ACTION-ON 增益（4-fold mean），用于 blindspot 对比。"""
    ops = expert.get("retrieval_spec", {}).get("ops", [])
    n = len(train_starts)
    edges = np.linspace(0, n, N_FOLD_BLOCKS + 1).astype(int)
    gains = []
    for f in range(1, N_FOLD_BLOCKS):
        bank_idx = train_starts[:edges[f]]
        q_idx = train_starts[edges[f]:edges[f + 1]]
        X_bank, Y_bank = build_windows(series, bank_idx, CONTEXT, HORIZON)
        X_q, Y_q = build_windows(series, q_idx, CONTEXT, HORIZON)
        bank_ts = [dates[s + CONTEXT] for s in bank_idx]
        q_ts = [dates[s + CONTEXT] for s in q_idx]
        act_bank = build_activation_matrix([expert], X_bank, bank_ts, stats)[:, 0]
        act_q = build_activation_matrix([expert], X_q, q_ts, stats)[:, 0]
        in_scope = act_q >= 0.5
        if int(in_scope.sum()) < MIN_IN_SCOPE:
            continue
        scope_bank = act_bank >= 0.5
        shape_pred = _topk_mean_forecast(X_q, X_bank, Y_bank, TOPK)
        action_pred = shape_pred[in_scope].copy()
        if int(scope_bank.sum()) >= TOPK:
            feats_q = compute_window_feats(X_q, q_ts, stats)
            feats_bank = compute_window_feats(X_bank, bank_ts, stats)
            lag_q = _hours_since(q_ts, ref)
            lag_bank = _hours_since(bank_ts, ref)
            action_pred = _topk_forecast_action(
                X_q[in_scope], X_bank, Y_bank, TOPK, ops,
                {k: v[in_scope] for k, v in feats_q.items()}, feats_bank,
                lag_q[in_scope], lag_bank, scope_bank)
        ms = mse(Y_q[in_scope], shape_pred[in_scope])
        ma = mse(Y_q[in_scope], action_pred)
        gains.append((ms - ma) / ms if ms > 0 else 0.0)
    return float(np.mean(gains)) if gains else None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--skip-llm", action="store_true", help="skip LLM critic/revision (debug)")
    args = ap.parse_args()
    epochs = args.epochs
    seeds = list(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = av2._default_model()
    print("device:", device, "| LLM model:", model)

    df = pd.read_csv(DATA_DIR / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(DATA_DIR / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    test_present = "test" in idx.files
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

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
    train_ts = [dates[s + CONTEXT] for s in train_starts]
    stats = compute_training_stats(X_train_all)
    ref = dates[0]

    edges = np.linspace(0, len(train_starts), N_FOLD_BLOCKS + 1).astype(int)
    evidence = {
        "feature_distributions": _raw_feature_quantiles(X_train_all),
        "temporal_coverage": _temporal_coverage(dates.iloc[:int(len(train_starts) + CONTEXT)]),
    }

    print("\n" + "=" * 78)
    print("PRE-RUN CHECK (v1: reward executes retrieval_action)")
    print("=" * 78)
    print(f"data split: bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)}")
    print(f"train fold edges (warmup+4 folds) = {edges.tolist()}")
    print(f"TEST accessed = {False}  (test present in npz: {test_present}, never loaded)")
    print(f"seeds = {seeds}")
    print(f"n_raw_expert target = {N_RAW_TARGET} (reused from v0)")
    print(f"VAL accessed during reward loop = {False}")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Raw Experts（复用 v0，不重新生成）----
    raw_experts = load_raw_experts()
    for e in raw_experts:
        e["compiled"] = compile_condition_v6(e, FEATURE_REGISTRY, evidence, COMPILER_CONFIG)
        e["retrieval_spec"] = derive_retrieval_spec(e, FEATURE_REGISTRY)
    print(f"\n[Round 0] loaded {len(raw_experts)} raw experts from v0")
    for e in raw_experts:
        ops = e["retrieval_spec"].get("ops", [])
        print(f"    {e['id']}: scope={e.get('scope','?')} | retrieval_status={e['retrieval_spec']['status']} | "
              f"ops={[o.get('type')+':'+o.get('feature', o.get('direction','')) for o in ops]}")

    # ---- 2. 闭环（Round 0/1/2）----
    lineage_best = {}
    lineage_coverage = {}          # lineage_id -> 上一轮 coverage（算 coverage_change）
    round_files = []
    reward_history_rows = []
    train_fold_rows = []
    action_on_off_rows = []
    lineage_rows = []

    experts = list(raw_experts)
    revision_round = 0

    for rnd in range(0, MAX_REVISION_ROUNDS + 1):
        print(f"\n{'=' * 70}\n[Round {rnd}] process {len(experts)} experts\n{'=' * 70}")
        round_records = []
        for e in experts:
            e["compiled"] = compile_condition_v6(e, FEATURE_REGISTRY, evidence, COMPILER_CONFIG)
            e["retrieval_spec"] = derive_retrieval_spec(e, FEATURE_REGISTRY)   # 始终以启发式为准（忽略 LLM 可能缺 status 的结构化 spec）
            parent_cov = lineage_coverage.get(e["id"])
            v = verify_one_v1(e, evidence, train_starts, series, dates, stats, X_train_all, train_ts,
                              parent_coverage=parent_cov)
            reward = v["reward"]
            verdict = verdict_from_reward_v1(reward)
            lid = e["id"]
            rec = {"expert": e, "evidence": v, "reward": reward, "verdict": verdict,
                   "round": rnd, "lineage_id": lid}
            round_records.append(rec)
            lineage_best[lid] = rec
            lineage_coverage[lid] = v["activation_coverage"]
            print(f"  {lid:<4} scope={e.get('scope','?'):<16} status={v['evidence_status']:<12} "
                  f"retr={v['retrieval_status']:<20} reward={reward:+d} verdict={verdict:<10} "
                  f"cov={v['activation_coverage']:.3f} ae={v['action_effect_size'] if v['action_effect_size'] is not None else float('nan'):+.4f} "
                  f"median_gain={v['median_gain'] if v['median_gain'] is not None else float('nan'):+.4f}")

        # ---- LLM Critic ----
        if args.skip_llm:
            critiques = []
        else:
            critiques, crit_cached = llm_critic(round_records, model)
        crit_by_id = {}
        for c in critiques:
            crit_by_id[c.get("expert_id")] = c
        for rec in round_records:
            lid = rec["lineage_id"]
            crit = crit_by_id.get(lid, {})
            crit.setdefault("verdict", rec["verdict"])
            crit.setdefault("diagnosis", _fallback_diagnosis(rec))
            crit.setdefault("supported_parts", [])
            crit.setdefault("failed_parts", [])
            crit.setdefault("retrieval_action_effect", "")
            crit.setdefault("coverage_effect", "")
            crit.setdefault("revision_instruction", "")
            rec["critic"] = crit
            rec["critic"]["verdict"] = rec["verdict"]

        round_files.append({
            "round": rnd,
            "experts": [dict(r["expert"]) for r in round_records],
            "evidence": [r["evidence"] for r in round_records],
            "critic": [r["critic"] for r in round_records],
        })

        for rec in round_records:
            v = rec["evidence"]
            e = rec["expert"]
            reward_history_rows.append({
                "lineage_id": rec["lineage_id"], "round": rnd, "reward": rec["reward"],
                "evidence_status": v["evidence_status"], "retrieval_status": v["retrieval_status"],
                "action_effect_size": v.get("action_effect_size"),
                "positive_fold_ratio": v.get("positive_fold_ratio"),
                "median_gain": v.get("median_gain"), "worst_gain": v.get("worst_gain"),
                "mean_gain": v.get("mean_gain"), "activation_coverage": v.get("activation_coverage"),
                "coverage_change_vs_parent": v.get("coverage_change_vs_parent"),
                "verdict": rec["verdict"],
            })
            for g in v.get("fold_gains", []):
                train_fold_rows.append({
                    "lineage_id": rec["lineage_id"], "round": rnd, "fold": g.get("fold"),
                    "gain": g.get("gain_action_on"), "mse_shape": g.get("mse_shape"),
                    "mse_expert": g.get("mse_action_on"), "n_in_scope": g.get("n_in_scope"),
                    "covered": g.get("covered"),
                })
                if g.get("covered"):
                    action_on_off_rows.append({
                        "lineage_id": rec["lineage_id"], "round": rnd, "fold": g.get("fold"),
                        "mse_shape": g.get("mse_shape"), "mse_act_only": g.get("mse_act_only"),
                        "mse_action_on": g.get("mse_action_on"),
                        "gain_act_only": g.get("gain_act_only"), "gain_action_on": g.get("gain_action_on"),
                        "action_effect": g.get("action_effect"),
                        "n_in_scope": g.get("n_in_scope"),
                        "candidate_retention": g.get("candidate_retention"),
                    })
            lineage_rows.append({
                "lineage_id": rec["lineage_id"], "round": rnd,
                "mechanism": e.get("mechanism", ""), "scope": e.get("scope", ""),
                "reward": rec["reward"], "verdict": rec["verdict"],
                "mean_gain": v.get("mean_gain"), "median_gain": v.get("median_gain"),
                "positive_fold_ratio": v.get("positive_fold_ratio"),
                "worst_gain": v.get("worst_gain"), "action_effect_size": v.get("action_effect_size"),
                "activation_coverage": v.get("activation_coverage"),
                "coverage_change_vs_parent": v.get("coverage_change_vs_parent"),
                "retrieval_status": v["retrieval_status"],
                "evidence_status": v["evidence_status"],
                "replaced": bool(e.get("replaced", False)),
            })

        to_revise = [r for r in round_records if r["verdict"] in ("REVISE", "UNSUPPORTED", "REJECT")]
        if rnd >= MAX_REVISION_ROUNDS or not to_revise:
            break
        print(f"\n  -> revise {len(to_revise)} experts (verdict REVISE/UNSUPPORTED/REJECT)")
        if args.skip_llm:
            # debug：跳过 LLM，用原样（保证离线可跑，但不会产生真实 revision）
            revised_experts = [dict(r["expert"]) for r in to_revise]
        else:
            revised_experts, rev_cached = llm_revise(to_revise, rnd + 1, model)
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
            ne["retrieval_spec"] = derive_retrieval_spec(ne, FEATURE_REGISTRY)   # 始终重派生（忽略 LLM 结构化 spec 缺 status 的情况）
            next_experts.append(ne)
        experts = next_experts

    # ---- 3. Final pool = 每 lineage 最新 reward>0 的版本 ----
    final_pool = []
    for lid, rec in lineage_best.items():
        if rec["reward"] > 0:
            final_pool.append(rec["expert"])
    final_pool = _sorted_experts(final_pool)
    print(f"\nFinal expert pool: {len(final_pool)} experts (reward>0) -> "
          f"{[e['id'] for e in final_pool]}")

    # ---- 4. blindspot audit（v0 E1/E5/E7）----
    v0_raw = json.loads((V0_DIR / "raw_experts.json").read_text(encoding="utf-8"))
    v0_r1 = json.loads((V0_DIR / "round1_experts.json").read_text(encoding="utf-8"))
    blindspot_rows = blindspot_audit(v0_raw, v0_r1, train_starts, series, dates, stats,
                                     X_train_all, train_ts, ref)
    print(f"\nblindspot audit (v0 E1/E5/E7): {len(blindspot_rows)} rows")
    for r in blindspot_rows:
        print(f"    {r['expert_id']}: raw_action_gain={r['raw_action_gain'] if r['raw_action_gain'] is not None else float('nan'):+.4f} "
              f"revised_action_gain={r['revised_action_gain'] if r['revised_action_gain'] is not None else float('nan'):+.4f} "
              f"delta={r['delta_action_gain'] if r['delta_action_gain'] is not None else float('nan'):+.4f}")

    # ===================================================================== #
    # 5. 最终 VAL ablation A/B/C/D（闭环冻结，第一次访问 VAL）
    # ===================================================================== #
    print("\n" + "=" * 78)
    print("FINAL VAL ABLATION (frozen, 3 seeds)")
    print("=" * 78)

    raw_sorted = _sorted_experts(raw_experts)
    n_raw = len(raw_sorted)
    n_final = len(final_pool)
    n_slots = max(n_raw, n_final)
    print(f"raw={n_raw} final={n_final} slots={n_slots}")

    emb_raw = embed_claims([e["claim"] for e in raw_sorted], device)
    emb_final = embed_claims([e["claim"] for e in final_pool], device) if final_pool else np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    expert_dim = emb_raw.shape[1]

    raw_act = act_mats(raw_sorted, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    final_act = act_mats(final_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats) if final_pool else {
        "bank": np.zeros((len(X_bank), 0), dtype=np.float32),
        "fit": np.zeros((len(X_fit), 0), dtype=np.float32),
        "val": np.zeros((len(X_val), 0), dtype=np.float32)}

    emb_raw, raw_act, raw_mask = pad_to_slots(emb_raw, raw_act, n_slots)
    emb_final, final_act, final_mask = pad_to_slots(emb_final, final_act, n_slots)
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_act = {k: np.zeros_like(raw_act[k]) for k in raw_act}

    configs = {
        "base":                   (emb_zero,  zero_act,  False),   # A
        "shape":                  (emb_zero,  zero_act,  True),    # B
        "raw_expert":             (emb_raw,   raw_act,   True),    # C
        "reward_feedback_expert": (emb_final, final_act, True),    # D
    }

    per_seed = {name: [] for name in configs}
    param_count = None
    for seed in seeds:
        print(f"\n{'=' * 70}\nSEED {seed}\n{'=' * 70}")
        attn.seed_all(seed)
        base_model = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        param_count = n_params(base_model)
        for name, (emb, act, use_shape) in configs.items():
            print(f"\n[{name}] use_shape={use_shape}")
            mdl = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
            mdl.load_state_dict(init_state)
            mdl.set_expert_embeddings(emb)
            train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                      act["bank"], act["fit"], use_shape, device, epochs=epochs)
            mode = "full" if use_shape else "expert"
            pred = retrieve_conditional(mdl, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                        act["val"], act["bank"], mode, TOPK, device)
            per_seed[name].append({"seed": seed, "mse": mse(Y_val, pred), "mae": mae(Y_val, pred)})
            print(f"    [{name}] MSE={per_seed[name][-1]['mse']:.2f} MAE={per_seed[name][-1]['mae']:.4f}")

    def mean_std(vals):
        a = np.array([v["mse"] for v in vals], dtype=np.float64)
        b = np.array([v["mae"] for v in vals], dtype=np.float64)
        return (float(a.mean()), float(a.std()), float(b.mean()), float(b.std()))

    summary = {}
    for name, vals in per_seed.items():
        mm, ms, am, as_ = mean_std(vals)
        summary[name] = {"mse_mean": mm, "mse_std": ms, "mae_mean": am, "mae_std": as_, "per_seed": vals}

    a_m = summary["base"]["mse_mean"]
    b_m = summary["shape"]["mse_mean"]
    c_m = summary["raw_expert"]["mse_mean"]
    d_m = summary["reward_feedback_expert"]["mse_mean"]
    gains = {
        "d_vs_c_pct": (c_m - d_m) / c_m * 100.0 if c_m else None,
        "d_vs_b_pct": (b_m - d_m) / b_m * 100.0 if b_m else None,
        "d_vs_a_pct": (a_m - d_m) / a_m * 100.0 if a_m else None,
        "c_vs_b_pct": (b_m - c_m) / b_m * 100.0 if b_m else None,
        "shape_vs_base_pct": (a_m - b_m) / a_m * 100.0 if a_m else None,
    }

    # ===================================================================== #
    # 6. 写输出
    # ===================================================================== #
    (OUT_DIR / "raw_experts.json").write_text(
        json.dumps(raw_experts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for rf in round_files:
        rnd = rf["round"]
        (OUT_DIR / f"round{rnd}_evidence.json").write_text(
            json.dumps(rf["evidence"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (OUT_DIR / f"round{rnd}_critic.json").write_text(
            json.dumps(rf["critic"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if rnd >= 1:
            (OUT_DIR / f"round{rnd}_experts.json").write_text(
                json.dumps(rf["experts"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "final_expert_pool.json").write_text(
        json.dumps(final_pool, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _csv(fname, rows, cols):
        out = OUT_DIR / fname
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
        return out

    _csv("knowledge_lineage.csv", lineage_rows, ["lineage_id", "round", "mechanism", "scope", "reward", "verdict",
                        "mean_gain", "median_gain", "positive_fold_ratio", "worst_gain", "action_effect_size",
                        "activation_coverage", "coverage_change_vs_parent", "retrieval_status",
                        "evidence_status", "replaced"])
    _csv("train_fold_gain.csv", train_fold_rows, ["lineage_id", "round", "fold", "gain", "mse_shape", "mse_expert",
                           "n_in_scope", "covered"])
    _csv("action_on_off.csv", action_on_off_rows, ["lineage_id", "round", "fold", "mse_shape", "mse_act_only",
                           "mse_action_on", "gain_act_only", "gain_action_on", "action_effect",
                           "n_in_scope", "candidate_retention"])
    _csv("reward_history.csv", reward_history_rows, ["lineage_id", "round", "reward", "evidence_status",
                               "retrieval_status", "action_effect_size", "positive_fold_ratio",
                               "median_gain", "worst_gain", "mean_gain", "activation_coverage",
                               "coverage_change_vs_parent", "verdict"])
    _csv("retrieval_action_blindspot_audit.csv", blindspot_rows, ["expert_id", "raw_action_gain",
                           "revised_action_gain", "delta_action_gain", "same_activation",
                           "v0_reward_changed", "v1_reward_changed"])

    with (OUT_DIR / "val_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "mse", "mae"])
        for name in ("base", "shape", "raw_expert", "reward_feedback_expert"):
            for v in per_seed[name]:
                w.writerow([name, v["seed"], f"{v['mse']:.6f}", f"{v['mae']:.6f}"])
        w.writerow([])
        w.writerow(["arm", "mse_mean", "mse_std", "mae_mean", "mae_std"])
        for name in ("base", "shape", "raw_expert", "reward_feedback_expert"):
            s = summary[name]
            w.writerow([name, f"{s['mse_mean']:.6f}", f"{s['mse_std']:.6f}",
                        f"{s['mae_mean']:.6f}", f"{s['mae_std']:.6f}"])

    report = render_report(raw_experts, round_files, lineage_rows, final_pool, blindspot_rows,
                           summary, gains, per_seed, param_count, n_slots, seeds)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("FINAL VAL ABLATION (mean ± std over seeds)")
    print("=" * 78)
    for name, label in (("base", "A BASE (CNN-only)"), ("shape", "B +SHAPE"),
                        ("raw_expert", "C +RAW EXPERT"), ("reward_feedback_expert", "D +REWARD-FEEDBACK EXPERT v1")):
        s = summary[name]
        print(f"{label:<28} MSE={s['mse_mean']:.2f} ± {s['mse_std']:.2f}   MAE={s['mae_mean']:.4f} ± {s['mae_std']:.4f}")
    print("-" * 78)
    print(f"D vs C = {gains['d_vs_c_pct']:+.2f}%   D vs B = {gains['d_vs_b_pct']:+.2f}%   D vs A = {gains['d_vs_a_pct']:+.2f}%")
    print(f"param_count (A/B/C/D identical) = {param_count}, slots = {n_slots}")
    print(f"Saved to {OUT_DIR}")
    print(f"VAL accessed during reward loop = False; TEST accessed = False")


def _fallback_diagnosis(rec):
    v = rec["evidence"]
    if v["evidence_status"] in ("UNSUPPORTED",):
        return ("TRAIN evidence insufficient to verify this claim: "
                + "; ".join(v["unverifiable_reasons"] or ["no verifiable scope"]))
    return (f"TRAIN fold evidence (action_on): median_gain={v['median_gain']:.4f}, "
            f"positive_fold_ratio={v['positive_fold_ratio']:.2f}, worst_gain={v['worst_gain']:.4f}, "
            f"action_effect_size={v['action_effect_size']:.4f} over {len(v['fold_gains'])} folds.")


def render_report(raw_experts, round_files, lineage_rows, final_pool, blindspot_rows,
                  summary, gains, per_seed, param_count, n_slots, seeds):
    L = []
    L.append("# ETTh1 — Reward-Feedback Critic v1 结果报告")
    L.append("")
    L.append("v1 修复 v0 核心缺陷：reward 真正执行 retrieval_action（ACTION-ON vs ACTION-OFF），"
             "不再只看 activation coverage / 局部 gain。")
    L.append("")

    L.append("## 每轮 verdict 计数")
    L.append("")
    L.append("| round | experts | KEEP | REVISE | REJECT | UNSUPPORTED |")
    L.append("|---|---|---|---|---|---|")
    for rf in round_files:
        rnd = rf["round"]
        n_k = sum(1 for r in rf["critic"] if r.get("verdict") == "KEEP")
        n_rv = sum(1 for r in rf["critic"] if r.get("verdict") == "REVISE")
        n_rj = sum(1 for r in rf["critic"] if r.get("verdict") == "REJECT")
        n_us = sum(1 for r in rf["critic"] if r.get("verdict") == "UNSUPPORTED")
        L.append(f"| {rnd} | {len(rf['evidence'])} | {n_k} | {n_rv} | {n_rj} | {n_us} |")
    L.append("")

    L.append("## 每条知识 lineage（action_effect_size 是关键新指标）")
    L.append("")
    L.append("| lineage_id | round | scope | reward | verdict | action_effect_size | median_gain | positive_fold_ratio | coverage | coverage_change | retrieval_status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in lineage_rows:
        ae = "" if r["action_effect_size"] is None else f"{r['action_effect_size']:+.4f}"
        mg = "" if r["median_gain"] is None else f"{r['median_gain']:+.4f}"
        pfr = "" if r["positive_fold_ratio"] is None else f"{r['positive_fold_ratio']:.2f}"
        cc = "" if r["coverage_change_vs_parent"] is None else f"{r['coverage_change_vs_parent']:+.2f}"
        L.append(f"| {r['lineage_id']} | {r['round']} | {r['scope']} | {r['reward']:+d} | {r['verdict']} "
                 f"| {ae} | {mg} | {pfr} | {r['activation_coverage']:.3f} | {cc} | {r['retrieval_status']} |")
    L.append("")

    L.append("## reward 变化")
    L.append("")
    by_id = {}
    for r in lineage_rows:
        by_id.setdefault(r["lineage_id"], []).append((r["round"], r["reward"], r["verdict"]))
    for lid in sorted(by_id, key=lambda x: int(x.lstrip("E") or 0)):
        seq = " -> ".join(f"r{rnd}:{rw:+d}({vd})" for rnd, rw, vd in by_id[lid])
        L.append(f"- **{lid}**: {seq}")
    L.append("")

    L.append("## v0 blindspot audit（E1/E5/E7：原 activation + 原/改 action）")
    L.append("")
    L.append("| expert_id | raw_action_gain | revised_action_gain | delta_action_gain | same_activation | v0_reward_changed | v1_reward_changed |")
    L.append("|---|---|---|---|---|---|---|")
    for r in blindspot_rows:
        rg = "" if r["raw_action_gain"] is None else f"{r['raw_action_gain']:+.4f}"
        vg = "" if r["revised_action_gain"] is None else f"{r['revised_action_gain']:+.4f}"
        dg = "" if r["delta_action_gain"] is None else f"{r['delta_action_gain']:+.4f}"
        L.append(f"| {r['expert_id']} | {rg} | {vg} | {dg} | {r['same_activation']} | "
                 f"{r['v0_reward_changed']} | {r['v1_reward_changed']} |")
    L.append("")

    L.append("## final pool")
    L.append("")
    L.append(f"共 {len(final_pool)} 条知识（reward>0）：{ [e['id'] for e in final_pool] or '(空)'}")
    for e in final_pool:
        L.append(f"- **{e['id']}** ({e.get('scope','')}): {e.get('mechanism','')}")
    L.append("")

    L.append("## 最终 VAL ablation（闭环冻结后首次访问 VAL，3 seeds）")
    L.append("")
    L.append("| method | seed42 | seed123 | seed2026 | mean MSE | std | mean MAE | std |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, label in (("base", "A BASE"), ("shape", "B +SHAPE"),
                        ("raw_expert", "C +RAW EXPERT"), ("reward_feedback_expert", "D +REWARD-FEEDBACK EXPERT v1")):
        vals = sorted(per_seed[name], key=lambda v: v["seed"])
        cells = " | ".join(f"{v['mse']:.2f}" for v in vals)
        s = summary[name]
        L.append(f"| {label} | {cells} | {s['mse_mean']:.2f} | {s['mse_std']:.2f} | {s['mae_mean']:.4f} | {s['mae_std']:.4f} |")
    L.append("")
    L.append(f"- D vs C (reward-feedback v1 vs raw) = {gains['d_vs_c_pct']:+.2f}%")
    L.append(f"- D vs B (vs shape) = {gains['d_vs_b_pct']:+.2f}%")
    L.append(f"- D vs A (vs base) = {gains['d_vs_a_pct']:+.2f}%")
    L.append(f"- C vs B (raw vs shape) = {gains['c_vs_b_pct']:+.2f}%")
    L.append(f"- Shape vs Base = {gains['shape_vs_base_pct']:+.2f}%")
    L.append(f"- param_count (A/B/C/D 一致) = {param_count}, slots = {n_slots}, seeds = {seeds}")
    L.append("")

    L.append("## 指标解释")
    L.append("")
    L.append("- `action_effect_size > 0`：真正执行 Expert 的“怎么检索”（retrieval_action）后，比只开 activation 更好。")
    L.append("- `coverage_change < -0.4`：适用样本砍掉 40% 以上（只靠收窄刷分）。")
    L.append("- `positive_fold_ratio = 0.75`：4 个 TRAIN 时间段中 3 个提升。")
    L.append("- `gain_* = (MSE_shape - MSE_x) / MSE_shape`：相对 shape baseline 的归一化增益。")
    L.append("")

    d_m, c_m = summary["reward_feedback_expert"]["mse_mean"], summary["raw_expert"]["mse_mean"]
    seeds_same_dir = all(
        per_seed["reward_feedback_expert"][i]["mse"] < per_seed["raw_expert"][i]["mse"]
        for i in range(len(seeds)))
    if d_m < c_m and seeds_same_dir:
        concl = "SUCCESS"
    elif abs(d_m - c_m) / max(c_m, 1e-9) < 0.02:
        concl = "PARTIAL"
    elif d_m > c_m:
        concl = "FAIL"
    else:
        concl = "PARTIAL"
    L.append("## 结论")
    L.append("")
    L.append(f"- Raw Expert 条数：{len(raw_experts)}")
    L.append(f"- final pool 条数：{len(final_pool)}")
    L.append(f"- Raw vs Reward-Feedback v1 的 VAL MSE：{c_m:.2f} vs {d_m:.2f}（{gains['d_vs_c_pct']:+.2f}%）")
    L.append(f"- 3 个 seed 是否同方向（D < C）：{'是' if seeds_same_dir else '否'}")
    L.append(f"- 结论：**{concl}**")
    if concl == "SUCCESS":
        L.append("  （D 稳定优于 C，reward-feedback v1 真正改善 Raw Expert。）")
    elif concl == "PARTIAL":
        L.append("  （D 与 C 接近，但 v1 修复了 v0 的 action blindspot / coverage chasing。）")
    else:
        L.append("  （D 仍明显差于 C，即使 action 真正被执行，reward 仍不能代表泛化价值。）")
    L.append("")
    L.append("```")
    L.append("test_used = false")
    L.append("val_used_during_reward_loop = false")
    L.append("critic_network_trained = false")
    L.append("chronos_backbone_modified = false")
    L.append("retrieval_action_executed_in_reward = true")
    L.append("```")
    return "\n".join(L)


if __name__ == "__main__":
    main()
