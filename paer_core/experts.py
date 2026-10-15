# -*- coding: utf-8 -*-

"""Shared unified-protocol component; source mapping is in DERIVATION.json."""

from __future__ import annotations

import os
import re
import numpy as np
from .model import _linear_slope_vec, _lag1_autocorr_vec
from .activation import build_activation_matrix
from .compiler import compile_condition_v6, COMPILER_CONFIG
from . import llm_transport as av2

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:76-76
CONTEXT, HORIZON, TOPK = 512, 96, 5

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:88-88
PREF_SIM_WEIGHT = 0.5             # prefer_similar / prefer_same / lag 的默认权重

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:89-89
PREF_SAME_WEIGHT = 0.3

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:90-90
LAG_CAP_HOURS = 24.0 * 30.0       # lag "short" 偏好的归一化上限（30 天）

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:100-117
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

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:119-119
CONTINUOUS_FEATURES = [f for f, m in FEATURE_REGISTRY.items() if m["kind"] == "continuous"]

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:120-120
CALENDAR_FEATURES = [f for f, m in FEATURE_REGISTRY.items() if m["kind"] == "calendar"]

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:224-239
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

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:242-250
def _temporal_coverage(dt_train):
    months = dt_train.dt.month.to_numpy()
    hours = dt_train.dt.hour.to_numpy()
    dows = dt_train.dt.dayofweek.to_numpy()
    return {
        "months_present": sorted(int(m) for m in set(months.tolist())),
        "hours_present": sorted(int(h) for h in set(hours.tolist())),
        "dows_present": sorted(int(d) for d in set(dows.tolist())),
    }

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:253-255
def _sorted_experts(pool):
    items = list(pool.values()) if isinstance(pool, dict) else list(pool)
    return sorted(items, key=lambda e: int(str(e.get("id", e.get("expert_id", "0"))).lstrip("E") or 0))

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:265-346
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

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:521-529
def _catalog_text():
    lines = ["AVAILABLE OBSERVABLES (choose 'observables' and 'condition_spec.feature' ONLY from this list):"]
    for f, meta in FEATURE_REGISTRY.items():
        if meta["kind"] == "calendar":
            cats = ", ".join(sorted(meta["categories"].keys()))
            lines.append(f"  - {f}: calendar (categories: {cats})")
        else:
            lines.append(f"  - {f}: continuous (qualifier: high or low)")
    return "\n".join(lines)

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:532-559
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

# Source: experiments/run_etth1_reward_feedback_critic_v1.py:562-577
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
