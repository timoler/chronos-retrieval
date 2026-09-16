#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reward-Feedback Critic v0 (ETTh1) — minimal closed-loop experiment.

验证 “Expert → TRAIN 数据验证 → Critic 奖励/诊断 → Expert 重写 → 再验证”
是否比 Raw Expert 更可靠。

- 只用 ETTh1；固定原有划分；TRAIN 生成证据/反馈；VAL 仅在闭环完全冻结后评估一次；TEST=False。
- 不训练新 Critic 网络；不改 Chronos 主干；不加 Attention/Gate。
- 复用 v6 的 Compiler / Retriever / Shape baseline / 训练评估代码。
- 3 seeds {42,123,2026}；同结构/参数量/初始化。

输出：results/reward_feedback_critic_v0_etth1/
"""

from __future__ import annotations

import argparse
import json
import os
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

import run_expert_attention_retrieval_v2 as attn            # seed_all / QueryEncoder
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
    build_activation_matrix,
)
from run_expert_agent_closed_loop_v6 import (
    compile_condition_v6,
    COMPILER_CONFIG,
    audit_observability_for_expert,
)
import run_agent_v2_electricity as av2                      # _call_llm / _list_from / _default_model
from priors.schema import _extract_json
from sentence_transformers import SentenceTransformer

REPO_ROOT = HERE.parent
av2.CACHE_DIR = str(REPO_ROOT / "cache/llm_priors_reward_feedback_critic_v0")   # 隔离的 v0 LLM 缓存

OUT_DIR = REPO_ROOT / "results/reward_feedback_critic_v0_etth1"
DATA_DIR = REPO_ROOT / "results/multidataset/ETTh1"

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32
N_RAW_TARGET = 8

# ---- Data Verifier / Reward 常量（绝不根据 VAL 调参）----
N_FOLD_BLOCKS = 5                 # warmup + 4 个 chronological folds
MIN_IN_SCOPE = 20                 # 每个 fold 至少多少 in-scope query 才算有效
REWARD_POS_HIGH = 0.75
REWARD_POS_LOW = 0.50
REWARD_NEG_LOW = 0.25
REWARD_WORST_TOL = -0.02
MAX_REVISION_ROUNDS = 2           # 最多 2 轮 revision（round 1 / round 2）

# --------------------------------------------------------------------------- #
# ETTh1 feature registry + metadata（与 v6 完全一致，仅输入在此）
# --------------------------------------------------------------------------- #
ETTH1_METADATA = {
    "target_variable": "OT",
    "target_description": "transformer temperature (target series)",
    "domain": "transformer_temperature",
    "sampling_frequency": "hourly",
    "n_population": 1,
}

FEATURE_REGISTRY = {
    "hour_of_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "mechanism_family": "calendar_cycle", "state_variable_family": "hour_of_day",
                    "categories": {"morning": [6, 7, 8, 9, 10, 11], "midday": [12, 13, 14, 15],
                                   "evening": [16, 17, 18, 19, 20, 21],
                                   "night": [22, 23, 0, 1, 2, 3, 4, 5]}},
    "day_of_week": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "mechanism_family": "calendar_cycle", "state_variable_family": "day_of_week",
                    "categories": {"weekday": [0, 1, 2, 3, 4], "weekend": [5, 6]}},
    "working_day": {"kind": "calendar", "source_type": "timestamp_known",
                    "dependencies": ["known_timestamp"],
                    "mechanism_family": "calendar_cycle", "state_variable_family": "working_day",
                    "categories": {"workday": [0, 1, 2, 3, 4], "non_working": [5, 6]}},
    "seasonal_level": {"kind": "calendar", "source_type": "timestamp_known",
                       "dependencies": ["known_timestamp"],
                       "mechanism_family": "calendar_cycle", "state_variable_family": "season",
                       "categories": {"winter": [12, 1, 2], "spring": [3, 4, 5],
                                      "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
    "recent_level": {"kind": "continuous", "source_type": "context_derived",
                     "dependencies": ["historical_context", "train_derived_constant"],
                     "mechanism_family": "recent_state_regime", "state_variable_family": "level"},
    "recent_volatility": {"kind": "continuous", "source_type": "context_derived",
                          "dependencies": ["historical_context", "train_derived_constant"],
                          "mechanism_family": "recent_state_regime", "state_variable_family": "volatility"},
    "recent_slope_magnitude": {"kind": "continuous", "source_type": "context_derived",
                               "dependencies": ["historical_context", "train_derived_constant"],
                               "mechanism_family": "recent_state_regime", "state_variable_family": "trend_slope"},
    "recent_diff_magnitude": {"kind": "continuous", "source_type": "context_derived",
                              "dependencies": ["historical_context", "train_derived_constant"],
                              "mechanism_family": "recent_state_regime", "state_variable_family": "change_magnitude"},
    "recent_autocorr": {"kind": "continuous", "source_type": "context_derived",
                        "dependencies": ["historical_context", "train_derived_constant"],
                        "mechanism_family": "recent_state_regime", "state_variable_family": "persistence"},
}

# --------------------------------------------------------------------------- #
# 通用工具
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


def _raw_feature_quantiles(X):
    """TRAIN 窗口上的 raw recent_* 分位数（供 compile_condition_v6 解析连续阈值）。"""
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
    months_present = sorted(int(m) for m in set(months.tolist()))
    hours_present = sorted(int(h) for h in set(hours.tolist()))
    dows_present = sorted(int(d) for d in set(dows.tolist()))
    return {
        "months_present": months_present, "months_missing": [m for m in range(1, 13) if m not in months_present],
        "month_fraction": round(len(months_present) / 12, 4),
        "hours_present": hours_present, "hours_missing": [h for h in range(24) if h not in hours_present],
        "hour_fraction": round(len(hours_present) / 24, 4),
        "dows_present": dows_present, "dow_fraction": round(len(dows_present) / 7, 4),
    }


def _sorted_experts(pool):
    items = list(pool.values()) if isinstance(pool, dict) else list(pool)
    return sorted(items, key=lambda e: int(str(e.get("id", e.get("expert_id", "0"))).lstrip("E") or 0))


# --------------------------------------------------------------------------- #
# Reward（固定常量，绝不根据 VAL 调参）
# --------------------------------------------------------------------------- #
def compute_reward(evidence_status, positive_fold_ratio, median_gain, worst_gain):
    if evidence_status == "UNSUPPORTED":
        return 0
    if positive_fold_ratio >= REWARD_POS_HIGH and median_gain > 0 and worst_gain >= REWARD_WORST_TOL:
        return 2
    if positive_fold_ratio >= REWARD_POS_LOW and median_gain > 0:
        return 1
    if positive_fold_ratio <= REWARD_NEG_LOW and median_gain < 0:
        return -2
    return -1


def verdict_from_reward(r):
    if r == 2:
        return "KEEP"
    if r == 1:
        return "KEEP"
    if r == 0:
        return "UNSUPPORTED"
    if r == -1:
        return "REVISE"
    return "REJECT"


# --------------------------------------------------------------------------- #
# TRAIN-only Data Verifier
# --------------------------------------------------------------------------- #
def verify_one(expert, evidence, train_starts, series, dates, stats, X_train_all, train_ts):
    """对单条 Expert knowledge 生成 evidence report（TRAIN-only）。"""
    lid = expert.get("id", expert.get("expert_id"))

    # ---- A. 可验证性 ----
    observables = list(expert.get("observables", []))
    obs_exist = all(f in FEATURE_REGISTRY for f in observables)
    obs_audit = audit_observability_for_expert(expert, FEATURE_REGISTRY)
    all_observable = bool(obs_audit["all_observable"])
    compiled = expert.get("compiled", {})
    compilable = compiled.get("status") == "COMPILED"
    unverifiable_reasons = []
    if not obs_exist:
        unverifiable_reasons.append("observable feature not in registry: "
                                    + str([f for f in observables if f not in FEATURE_REGISTRY]))
    if not all_observable:
        unverifiable_reasons.append("observable depends on unavailable/future input")
    if not compilable:
        unverifiable_reasons.append("activation condition uncompilable: "
                                    + str(compiled.get("uncompilable", [])))

    # activation coverage over FULL TRAIN（scope 是否有足够覆盖）
    act_all = build_activation_matrix([expert], X_train_all, train_ts, stats)[:, 0]
    activation_coverage = float((act_all >= 0.5).mean())
    n_supported_queries = int((act_all >= 0.5).sum())

    if unverifiable_reasons:
        return {
            "expert_id": lid,
            "evidence_status": "UNSUPPORTED",
            "unverifiable_reasons": unverifiable_reasons,
            "activation_coverage": activation_coverage,
            "n_supported_queries": n_supported_queries,
            "fold_gains": [],
            "mean_gain": None, "median_gain": None, "positive_fold_ratio": None,
            "worst_gain": None, "std_gain": None,
            "reward": 0,
        }

    # ---- B/C. 4 个 chronological folds：memory-only forecast 对比 ----
    n = len(train_starts)
    edges = np.linspace(0, n, N_FOLD_BLOCKS + 1).astype(int)
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
        shape_pred = _topk_mean_forecast(X_q, X_bank, Y_bank, TOPK)
        expert_pred = shape_pred.copy()
        rest_bank = act_bank >= 0.5
        if int(rest_bank.sum()) >= TOPK:
            expert_pred[in_scope] = _topk_mean_forecast(
                X_q[in_scope], X_bank[rest_bank], Y_bank[rest_bank], TOPK)
        ms = mse(Y_q[in_scope], shape_pred[in_scope])
        me = mse(Y_q[in_scope], expert_pred[in_scope])
        row.update({"mse_shape": ms, "mse_expert": me,
                    "gain": (ms - me) / ms if ms > 0 else 0.0})
        fold_gains.append(row)

    covered = [g for g in fold_gains if g.get("covered")]
    n_in_total = sum(g["n_in_scope"] for g in fold_gains)

    if len(covered) == 0 or n_in_total == 0:
        return {
            "expert_id": lid, "evidence_status": "UNSUPPORTED",
            "unverifiable_reasons": ["insufficient TRAIN activation coverage (scope not verifiable)"],
            "activation_coverage": activation_coverage, "n_supported_queries": n_supported_queries,
            "fold_gains": fold_gains,
            "mean_gain": None, "median_gain": None, "positive_fold_ratio": None,
            "worst_gain": None, "std_gain": None,
            "reward": 0,
        }

    gains = np.array([g["gain"] for g in covered], float)
    mean_gain = float(gains.mean())
    median_gain = float(np.median(gains))
    positive_fold_ratio = float((gains > 0).mean())
    worst_gain = float(gains.min())
    std_gain = float(gains.std())
    reward = compute_reward("SUPPORTED", positive_fold_ratio, median_gain, worst_gain)

    return {
        "expert_id": lid, "evidence_status": "SUPPORTED",
        "unverifiable_reasons": [],
        "activation_coverage": activation_coverage, "n_supported_queries": n_supported_queries,
        "fold_gains": fold_gains,
        "mean_gain": mean_gain, "median_gain": median_gain,
        "positive_fold_ratio": positive_fold_ratio, "worst_gain": worst_gain, "std_gain": std_gain,
        "reward": reward,
    }


# --------------------------------------------------------------------------- #
# LLM 提示词：Expert 生成 / Critic / Revision
# --------------------------------------------------------------------------- #
def _metadata_text():
    return (
        "DATASET METADATA:\n"
        "- target variable: OT (transformer oil temperature)\n"
        "- domain: electricity transformer / power system\n"
        "- sampling frequency: hourly\n"
        "- available variables: HUFL, HULL, MUFL, MULL, LUFL, LULL (power load covariates), OT (target)\n"
    )


def _expert_hint_text():
    return (
        "EXPERT HINT (physical mechanism, hypotheses NOT verified facts):\n"
        "Transformer oil temperature is governed by a thermal energy balance. Electrical "
        "current through the windings produces resistive losses (heat source); the oil has "
        "substantial thermal mass, so its temperature is an accumulated, time-integrated "
        "response to net heat input — it reacts gradually and smooths short-lived fluctuations. "
        "The transformer can sit in different thermal conditions: approaching equilibrium under "
        "steady load, or in a transient heating/cooling phase after a load/ambient change."
    )


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
        "\n"
        "For condition_spec.qualifier: continuous features use \"high\" or \"low\"; "
        "calendar features use the category name (e.g. \"summer\", \"night\", \"workday\").\n"
        "Do NOT invent features outside the catalog. Do NOT use future/target values as observables. "
        "Output only the JSON object."
    )


def _call_llm_list(system, user, stage, key, model, max_retries=3):
    """LLM call -> parse list under `key`, retry on truncated/unparseable JSON.

    Invalidate the cache entry on failure so a retry makes a fresh API call (the
    cache key also changes via the appended completion nudge).
    """
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


def generate_raw_experts(model):
    system = (
        "You are a domain expert on forecasting a univariate hourly time series (transformer "
        "oil temperature OT). Propose CONDITIONAL expert knowledge for a retrieval-based "
        "forecaster: each piece states a mechanism about WHEN historical segments are more "
        "predictive and HOW retrieval should change in that regime. These are hypotheses to be "
        "checked against data, not established facts. Output strict JSON only."
    )
    user = "\n\n".join([
        _metadata_text(), _expert_hint_text(), _catalog_text(), _expert_schema_text(),
        f"TASK: Propose exactly {N_RAW_TARGET} conditional expert knowledge entries. "
        "Output only the JSON object.",
    ])
    experts, cached = _call_llm_list(system, user, "etth1_rf_critic_v0_raw_expert", "experts", model)
    # stamp id order
    for i, e in enumerate(experts, 1):
        e["id"] = e.get("id") or f"E{i}"
    return experts, cached


def _evidence_summary_text(rec):
    ev = rec["evidence"]
    e = rec["expert"]
    lines = [
        f"expert {e['id']}: mechanism={e.get('mechanism','')}",
        f"  scope={e.get('scope','')} | activation_condition={e.get('activation_condition','')}",
        f"  observables={e.get('observables',[])} | retrieval_action={e.get('retrieval_action','')}",
        f"  evidence_status={ev['evidence_status']} | auto_reward={rec['reward']}",
        f"  activation_coverage={ev['activation_coverage']:.4f} | n_supported_queries={ev['n_supported_queries']}",
    ]
    if ev["evidence_status"] == "SUPPORTED":
        lines += [
            f"  mean_gain={ev['mean_gain']:.4f} | median_gain={ev['median_gain']:.4f} | "
            f"positive_fold_ratio={ev['positive_fold_ratio']:.4f} | worst_gain={ev['worst_gain']:.4f} | "
            f"std_gain={ev['std_gain']:.4f}",
            f"  fold_gains={ev['fold_gains']}",
        ]
    else:
        lines.append(f"  unverifiable_reasons={ev['unverifiable_reasons']}")
    return "\n".join(lines)


def llm_critic(round_records, model):
    """LLM Critic：给定 evidence + auto reward，输出 diagnosis / supported / failed / revision。"""
    auto_verdicts = {r["expert"]["id"]: r["verdict"] for r in round_records}
    body = "\n\n".join(_evidence_summary_text(r) for r in round_records)
    system = (
        "You are a rigorous critic for a retrieval-based forecaster. Given (a) an expert's "
        "structured knowledge, (b) deterministic TRAIN-only evidence, and (c) an automatic "
        "reward computed from that evidence, produce a per-expert diagnosis. You must NOT "
        "recompute data and must NOT invent evidence. Compare what the expert CLAIMED against "
        "what the data ACTUALLY returned. If evidence is UNSUPPORTED you must say the evidence "
        "is insufficient — NOT that the mechanism is wrong. Output strict JSON only."
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
        '  - "diagnosis": why (2-3 sentences, cite the evidence numbers)\n'
        '  - "revision_instruction": if REVISE/UNSUPPORTED/REJECT, tell the Expert exactly what to '
        "change; do NOT invent new data patterns, only refine based on the evidence shown.\n"
        "TASK: output only the JSON object.",
    ])
    return _call_llm_list(system, user, "etth1_rf_critic_v0_critic", "critiques", model)


def llm_revise(to_revise, revision_round, model):
    """Expert Revision：仅把 REVISE/UNSUPPORTED/REJECT 反馈给 Expert 重写。"""
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
        "Revise each failed expert so that its mechanism is consistent with the TRAIN evidence shown. "
        "If the mechanism cannot be verified on TRAIN (UNSUPPORTED), you MAY switch to a different "
        "TRAIN-verifiable mechanism and MUST set 'replaced': true (do not pretend the original was "
        "proven). Otherwise keep the same mechanism and refine scope/condition/retrieval_action. "
        "Do NOT invent data patterns beyond the evidence shown. Output strict JSON only."
    )
    user = "\n\n".join([
        _catalog_text(), _expert_schema_text(),
        "REVISION ROUND: " + str(revision_round),
        "FEEDBACK (per expert):\n" + json.dumps(feedbacks, ensure_ascii=False, indent=2),
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"experts": [ ... ]} with the SAME keys as the expert schema PLUS "replaced": true/false. '
        "Keep the same 'id'. Output only the JSON object.",
    ])
    experts, cached = _call_llm_list(system, user, "etth1_rf_critic_v0_revision", "experts", model)
    return experts, cached


# --------------------------------------------------------------------------- #
# 最终 VAL ablation（A/B/C/D，复用 v6 ablation 的训练/评估协议）
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
    bank = build_activation_matrix(experts, X_bank, bank_ts, stats)
    fit = build_activation_matrix(experts, X_fit, fit_ts, stats)
    val = build_activation_matrix(experts, X_val, val_ts, stats)
    return {"bank": bank, "fit": fit, "val": val}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--skip-llm", action="store_true", help="skip LLM (offline debug)")
    args = ap.parse_args()
    epochs = args.epochs
    seeds = list(args.seeds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = av2._default_model()
    print("device:", device, "| LLM model:", model)

    # ---- 数据：固定划分（TRAIN bank/fit + VAL；TEST 永不读取）----
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

    # TRAIN fold 边界（chronological，warmup + 4 folds）
    edges = np.linspace(0, len(train_starts), N_FOLD_BLOCKS + 1).astype(int)

    # compile 用 evidence（TRAIN-only raw 分位数 + 时间覆盖）
    evidence = {
        "feature_distributions": _raw_feature_quantiles(X_train_all),
        "temporal_coverage": _temporal_coverage(dates.iloc[:int(len(train_starts) + CONTEXT)]),
    }

    # ===================================================================== #
    # 0. 预运行打印（确认无误后直接跑）
    # ===================================================================== #
    print("\n" + "=" * 78)
    print("PRE-RUN CHECK")
    print("=" * 78)
    print(f"data split: bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)}")
    print(f"train fold edges (warmup+4 folds) = {edges.tolist()}")
    print(f"TEST accessed = {False}  (test present in npz: {test_present}, never loaded)")
    print(f"seeds = {seeds}")
    print(f"n_raw_expert target = {N_RAW_TARGET}")
    print(f"VAL accessed during reward loop = {False}  (only TRAIN for evidence/reward/critic/revision)")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Raw Expert（Round 0）----
    print("\n[Round 0] generate raw experts ...")
    if args.skip_llm:
        raise SystemExit("--skip-llm is debug-only; no offline experts available")
    raw_experts, cached = generate_raw_experts(model)
    raw_experts = raw_experts[:N_RAW_TARGET]
    # 统一字段：claim 供 embedding，observable_features 供审计
    for e in raw_experts:
        e["claim"] = e.get("mechanism", e.get("reason", ""))
        e["observable_features"] = list(e.get("observables", []))
    n_raw = len(raw_experts)
    print(f"  raw experts (LLM cached={cached}) = {n_raw}")
    for e in raw_experts:
        print(f"    {e['id']}: {e.get('scope','?')} | obs={e.get('observables',[])} | "
              f"cond={e.get('condition_spec',{}).get('conditions',[])}")

    # ---- 2. 闭环（Round 0 / 1 / 2）----
    lineage_best = {}                      # lineage_id -> 最新 record
    round_files = []
    reward_history_rows = []
    train_fold_rows = []
    lineage_rows = []

    experts = list(raw_experts)
    revision_round = 0

    for rnd in range(0, MAX_REVISION_ROUNDS + 1):     # rnd = 0,1,2
        print(f"\n{'=' * 70}\n[Round {rnd}] process {len(experts)} experts\n{'=' * 70}")
        round_records = []
        for e in experts:
            e["compiled"] = compile_condition_v6(e, FEATURE_REGISTRY, evidence, COMPILER_CONFIG)
            v = verify_one(e, evidence, train_starts, series, dates, stats, X_train_all, train_ts)
            reward = v["reward"]
            verdict = verdict_from_reward(reward)
            lid = e["id"]
            rec = {"expert": e, "evidence": v, "reward": reward, "verdict": verdict,
                   "round": rnd, "lineage_id": lid}
            round_records.append(rec)
            lineage_best[lid] = rec
            print(f"  {lid:<4} scope={e.get('scope','?'):<16} status={v['evidence_status']:<12} "
                  f"reward={reward:+d} verdict={verdict:<10} "
                  f"cov={v['activation_coverage']:.3f} "
                  f"median_gain={v['median_gain'] if v['median_gain'] is not None else float('nan'):+.4f}")

        # ---- LLM Critic（diagnosis / supported / failed / revision_instruction）----
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
            crit.setdefault("revision_instruction", "")
            rec["critic"] = crit
            # 记录（verdict 以 reward 为准，LLM 只补充诊断）
            rec["critic"]["verdict"] = rec["verdict"]

        # 保存 round 文件
        round_files.append({
            "round": rnd,
            "experts": [dict(r["expert"]) for r in round_records],
            "evidence": [r["evidence"] for r in round_records],
            "critic": [r["critic"] for r in round_records],
        })

        # ---- 记录 CSV 行 ----
        for rec in round_records:
            v = rec["evidence"]
            e = rec["expert"]
            reward_history_rows.append({
                "lineage_id": rec["lineage_id"], "round": rnd, "reward": rec["reward"],
                "evidence_status": v["evidence_status"],
                "positive_fold_ratio": v.get("positive_fold_ratio"),
                "median_gain": v.get("median_gain"), "worst_gain": v.get("worst_gain"),
                "mean_gain": v.get("mean_gain"), "activation_coverage": v.get("activation_coverage"),
                "verdict": rec["verdict"],
            })
            for g in v.get("fold_gains", []):
                train_fold_rows.append({
                    "lineage_id": rec["lineage_id"], "round": rnd, "fold": g.get("fold"),
                    "gain": g.get("gain"), "mse_shape": g.get("mse_shape"),
                    "mse_expert": g.get("mse_expert"), "n_in_scope": g.get("n_in_scope"),
                    "covered": g.get("covered"),
                })
            lineage_rows.append({
                "lineage_id": rec["lineage_id"], "round": rnd,
                "mechanism": e.get("mechanism", ""), "scope": e.get("scope", ""),
                "reward": rec["reward"], "verdict": rec["verdict"],
                "mean_gain": v.get("mean_gain"), "median_gain": v.get("median_gain"),
                "positive_fold_ratio": v.get("positive_fold_ratio"),
                "worst_gain": v.get("worst_gain"),
                "activation_coverage": v.get("activation_coverage"),
                "evidence_status": v["evidence_status"],
                "replaced": bool(e.get("replaced", False)),
            })

        # ---- 决定下一轮 ----
        to_revise = [r for r in round_records if r["verdict"] in ("REVISE", "UNSUPPORTED", "REJECT")]
        if rnd >= MAX_REVISION_ROUNDS or not to_revise:
            break
        print(f"\n  -> revise {len(to_revise)} experts (verdict REVISE/UNSUPPORTED/REJECT)")
        revised_experts, rev_cached = llm_revise(to_revise, rnd + 1, model)
        # stamp lineage
        revised_by_id = {e.get("id"): e for e in revised_experts}
        next_experts = []
        for rec in to_revise:
            lid = rec["lineage_id"]
            ne = revised_by_id.get(lid)
            if ne is None:
                ne = dict(rec["expert"])     # 若 LLM 漏掉，保持原样（不改动）
            ne["id"] = lid
            ne["claim"] = ne.get("mechanism", ne.get("reason", ""))
            ne["observable_features"] = list(ne.get("observables", []))
            ne["revision_round"] = rnd + 1
            ne["replaced"] = bool(ne.get("replaced", False))
            next_experts.append(ne)
        experts = next_experts

    # ---- 3. 最终 pool = 每 lineage 最新 verdict==KEEP 的版本 ----
    final_pool = []
    for lid, rec in lineage_best.items():
        if rec["verdict"] == "KEEP":
            final_pool.append(rec["expert"])
    final_pool = _sorted_experts(final_pool)
    print(f"\nFinal expert pool: {len(final_pool)} KEEP experts -> "
          f"{[e['id'] for e in final_pool]}")

    # ===================================================================== #
    # 4. 最终 VAL ablation A/B/C/D（闭环已冻结，第一次访问 VAL）
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
    # 5. 写输出
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

    # CSV 输出
    def _csv(fname, rows, cols):
        import csv
        out = OUT_DIR / fname
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
        return out

    _csv("knowledge_lineage.csv", lineage_rows, ["lineage_id", "round", "mechanism", "scope", "reward", "verdict",
                        "mean_gain", "median_gain", "positive_fold_ratio", "worst_gain",
                        "activation_coverage", "evidence_status", "replaced"])
    _csv("train_fold_gain.csv", train_fold_rows, ["lineage_id", "round", "fold", "gain", "mse_shape", "mse_expert",
                           "n_in_scope", "covered"])
    _csv("reward_history.csv", reward_history_rows, ["lineage_id", "round", "reward", "evidence_status",
                               "positive_fold_ratio", "median_gain", "worst_gain",
                               "mean_gain", "activation_coverage", "verdict"])

    # val_ablation.csv
    import csv as _csv_mod
    with (OUT_DIR / "val_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv_mod.writer(f)
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

    # ---- 6. report.md ----
    report = render_report(raw_experts, round_files, lineage_rows, final_pool,
                           summary, gains, per_seed, param_count, n_slots, seeds)
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    # 打印 summary
    print("\n" + "=" * 78)
    print("FINAL VAL ABLATION (mean ± std over seeds)")
    print("=" * 78)
    for name, label in (("base", "A BASE (CNN-only)"), ("shape", "B +SHAPE"),
                        ("raw_expert", "C +RAW EXPERT"), ("reward_feedback_expert", "D +REWARD-FEEDBACK EXPERT")):
        s = summary[name]
        print(f"{label:<28} MSE={s['mse_mean']:.2f} ± {s['mse_std']:.2f}   MAE={s['mae_mean']:.4f} ± {s['mae_std']:.4f}")
    print("-" * 78)
    print(f"D vs C = {gains['d_vs_c_pct']:+.2f}%   D vs B = {gains['d_vs_b_pct']:+.2f}%   D vs A = {gains['d_vs_a_pct']:+.2f}%")
    print(f"param_count (A/B/C/D identical) = {param_count}, slots = {n_slots}")
    print(f"Saved to {OUT_DIR}")
    print(f"VAL accessed during reward loop = False; TEST accessed = False")


def _fallback_diagnosis(rec):
    v = rec["evidence"]
    if v["evidence_status"] == "UNSUPPORTED":
        return ("TRAIN evidence insufficient to verify this claim: "
                + "; ".join(v["unverifiable_reasons"] or ["no verifiable scope"]))
    return (f"TRAIN fold evidence: median_gain={v['median_gain']:.4f}, "
            f"positive_fold_ratio={v['positive_fold_ratio']:.2f}, "
            f"worst_gain={v['worst_gain']:.4f} over {len(v['fold_gains'])} folds.")


def render_report(raw_experts, round_files, lineage_rows, final_pool,
                  summary, gains, per_seed, param_count, n_slots, seeds):
    L = []
    L.append("# ETTh1 — Reward-Feedback Critic v0 结果报告")
    L.append("")
    L.append("验证 “Expert → TRAIN 数据验证 → Critic 奖励/诊断 → Expert 重写 → 再验证” 是否比 Raw Expert 更可靠。")
    L.append("")

    # 每轮 KEEP/REVISE/REJECT/UNSUPPORTED
    L.append("## 每轮 verdict 计数")
    L.append("")
    L.append("| round | experts | KEEP | REVISE | REJECT | UNSUPPORTED |")
    L.append("|---|---|---|---|---|---|")
    for rf in round_files:
        rnd = rf["round"]
        evs = rf["evidence"]
        n_k = sum(1 for r in rf["critic"] if r.get("verdict") == "KEEP")
        n_rv = sum(1 for r in rf["critic"] if r.get("verdict") == "REVISE")
        n_rj = sum(1 for r in rf["critic"] if r.get("verdict") == "REJECT")
        n_us = sum(1 for r in rf["critic"] if r.get("verdict") == "UNSUPPORTED")
        L.append(f"| {rnd} | {len(evs)} | {n_k} | {n_rv} | {n_rj} | {n_us} |")
    L.append("")

    # lineage 表
    L.append("## 每条知识 lineage")
    L.append("")
    L.append("| lineage_id | round | scope | reward | verdict | median_gain | positive_fold_ratio | worst_gain | activation_coverage | evidence_status | replaced |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in lineage_rows:
        mg = "" if r["median_gain"] is None else f"{r['median_gain']:+.4f}"
        pfr = "" if r["positive_fold_ratio"] is None else f"{r['positive_fold_ratio']:.2f}"
        wg = "" if r["worst_gain"] is None else f"{r['worst_gain']:+.4f}"
        L.append(f"| {r['lineage_id']} | {r['round']} | {r['scope']} | {r['reward']:+d} | {r['verdict']} "
                 f"| {mg} | {pfr} | {wg} | {r['activation_coverage']:.3f} | {r['evidence_status']} | {r['replaced']} |")
    L.append("")

    # reward 变化
    L.append("## reward 变化")
    L.append("")
    by_id = {}
    for r in lineage_rows:
        by_id.setdefault(r["lineage_id"], []).append((r["round"], r["reward"], r["verdict"]))
    for lid in sorted(by_id, key=lambda x: int(x.lstrip("E") or 0)):
        seq = " -> ".join(f"r{rnd}:{rw:+d}({vd})" for rnd, rw, vd in by_id[lid])
        L.append(f"- **{lid}**: {seq}")
    L.append("")

    # final pool
    L.append("## final pool")
    L.append("")
    L.append(f"共 {len(final_pool)} 条 KEEP 知识：{ [e['id'] for e in final_pool] or '(空)'}")
    for e in final_pool:
        L.append(f"- **{e['id']}** ({e.get('scope','')}): {e.get('mechanism','')}")
    L.append("")

    # VAL ablation
    L.append("## 最终 VAL ablation（闭环冻结后首次访问 VAL，3 seeds）")
    L.append("")
    L.append("| method | seed42 | seed123 | seed2026 | mean MSE | std | mean MAE | std |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, label in (("base", "A BASE"), ("shape", "B +SHAPE"),
                        ("raw_expert", "C +RAW EXPERT"), ("reward_feedback_expert", "D +REWARD-FEEDBACK EXPERT")):
        vals = sorted(per_seed[name], key=lambda v: v["seed"])
        cells = " | ".join(f"{v['mse']:.2f}" for v in vals)
        s = summary[name]
        L.append(f"| {label} | {cells} | {s['mse_mean']:.2f} | {s['mse_std']:.2f} | {s['mae_mean']:.4f} | {s['mae_std']:.4f} |")
    L.append("")
    L.append(f"- D vs C (reward-feedback vs raw) = {gains['d_vs_c_pct']:+.2f}%")
    L.append(f"- D vs B (vs shape) = {gains['d_vs_b_pct']:+.2f}%")
    L.append(f"- D vs A (vs base) = {gains['d_vs_a_pct']:+.2f}%")
    L.append(f"- C vs B (raw vs shape) = {gains['c_vs_b_pct']:+.2f}%")
    L.append(f"- Shape vs Base = {gains['shape_vs_base_pct']:+.2f}%")
    L.append(f"- param_count (A/B/C/D 一致) = {param_count}, slots = {n_slots}, seeds = {seeds}")
    L.append("")

    # 指标解释
    L.append("## 指标解释")
    L.append("")
    L.append("- `positive_fold_ratio = 0.75`：4 个 TRAIN 时间段里有 3 个提升。")
    L.append("- `median_gain = 0.08`：典型时间段约提升 8%。")
    L.append("- `worst_gain = -0.01`：最差时间段退化约 1%。")
    L.append("- `activation_coverage`：TRAIN 中该知识激活的 query 占比（scope 覆盖）。")
    L.append("")

    # 结论
    d_m, c_m = summary["reward_feedback_expert"]["mse_mean"], summary["raw_expert"]["mse_mean"]
    seeds_same_dir = all(
        per_seed["reward_feedback_expert"][i]["mse"] < per_seed["raw_expert"][i]["mse"]
        for i in range(len(seeds)))
    if d_m < c_m and seeds_same_dir:
        concl = "SUCCESS"
    elif abs(d_m - c_m) / c_m < 0.01:
        concl = "PARTIAL"
    elif d_m > c_m:
        concl = "FAIL"
    else:
        concl = "PARTIAL"
    L.append("## 结论")
    L.append("")
    L.append(f"- Raw Expert 条数：{len(raw_experts)}")
    L.append(f"- final pool 条数：{len(final_pool)}")
    L.append(f"- Raw vs Reward-Feedback 的 VAL MSE：{c_m:.2f} vs {d_m:.2f}（{gains['d_vs_c_pct']:+.2f}%）")
    L.append(f"- 3 个 seed 是否同方向（D < C）：{'是' if seeds_same_dir else '否'}")
    L.append(f"- 结论：**{concl}**")
    if concl == "FAIL":
        L.append("  （D 明显差于 C，reward-feedback 尚未形成有效闭环，不美化。）")
    elif concl == "PARTIAL":
        L.append("  （D 与 C 相近，reward-feedback 主要价值可能是减少有害/不稳定知识，而非显著提升。）")
    else:
        L.append("  （D 稳定优于 C，且不是单 seed 偶然。）")
    L.append("")
    L.append("```")
    L.append("test_used = false")
    L.append("val_used_during_reward_loop = false")
    L.append("critic_network_trained = false")
    L.append("chronos_backbone_modified = false")
    L.append("```")
    return "\n".join(L)


if __name__ == "__main__":
    main()
