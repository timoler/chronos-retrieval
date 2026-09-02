#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solar Expert→Critic v3 — 重构 Critic 核心逻辑。

相比 frozen v2 的唯一改动：把单一的 Data-Support Critic 拆成三个环节：

    Expert Knowledge
      → Semantic Critic                (复用 v2，只判机制自洽，不动)
      → Evidence Planner               (新 LLM 阶段：规划该算什么证据)
      → TRAIN-only Evidence Executor   (新 确定性：真正计算富目录数值)
      → Data Critic                    (新 LLM 阶段：拿真实数值再判 ACCEPT/REJECT/UNSUPPORTED)
      → Compiler                       (复用 v2)
      → Proxy Validator                (复用 v2)

关键规则：
  1. Evidence Planner 允许使用 date/time 构造 hour / time_of_day / month /
     day_of_year / season / 周期位置；允许从 OT/X 构造 lag / difference /
     rolling variance / autocorrelation / correlation / co-movement。
  2. 「某个统计量没有预先计算」绝不能作为 UNSUPPORTED 的理由——必须先走
     Planner → Executor 真正算出数值。
  3. 人口/子集代表性规则原样保留：即使 corr(S6,Y)=0.9884，也不能推出
     P001-P006 代表 137 站总体。
  4. 禁止错误 proxy：高频波动(rolling volatility)可描述「波动 regime」，
     但 TRAIN 无 cloud ground truth 时不得声称「高频波动=云层」。

只用 Solar 的 TRAIN + VAL；TEST 绝不读取。
输出：results/solar_agent_v3_audit.json + results/expert_critic_v3_audit.md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_agent_v2_electricity as av2  # frozen v2 的 prompt builders / _call_llm / 编译器
import run_agent_v2_solar as av2s       # solar 常量

from priors.executable import ExecutablePrior, compute_feature, feature_similarity
from priors.validator import CONTEXT, PRED, _mem_forecast
from retrieval.decompose import seasonal_decompose
from retrieval.similarity import pearson_similarity_matrix

# --------------------------------------------------------------------------- #
# dataset-specific constants (same as solar v2)
# --------------------------------------------------------------------------- #
PERIOD = av2s.PERIOD                 # 144
TREND_WINDOW = av2s.TREND_WINDOW     # 288
VAL_STRIDE = av2s.VAL_STRIDE         # 6
LOAD_VARS = av2s.LOAD_VARS           # ["P001".."P006"]
N_POPULATION = av2s.N_POPULATION     # 137
BLIND_TO_REAL = av2s.BLIND_TO_REAL
SOLAR_METADATA_V2 = av2s.SOLAR_METADATA_V2
SOLAR_EXPERT_MECHANISM = av2s.SOLAR_EXPERT_MECHANISM

CACHE_DIR = "cache/llm_priors_v3_solar"


def _seed_v3_cache():
    """复用 v2 已缓存的 proposal + semantic 结果（prompt 完全一致 -> 相同 H1..H6）。"""
    src = Path("cache/llm_priors_v2_solar")
    dst = Path(CACHE_DIR)
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("stage") in ("proposal", "semantic_critic"):
            target = dst / f.name
            if not target.exists():
                target.write_text(f.read_text(encoding="utf-8"), encoding="utf-8")


def _corr(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# TRAIN-only Evidence Executor (v3) — 富目录确定性统计
# --------------------------------------------------------------------------- #
def solar_compute_evidence_catalog_v3() -> dict:
    df = pd.read_csv("data/solar.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    dt = dates.iloc[:train_len]

    y = df["OT"].to_numpy(np.float64)[:train_len]
    s6 = np.sum([df[v].to_numpy(np.float64)[:train_len] for v in LOAD_VARS], axis=0)

    hours = dt.dt.hour.to_numpy()
    months = dt.dt.month.to_numpy()

    # ---- 昼夜 / 季节包络（H2 需要）----
    hourly_mean = np.array([float(y[hours == h].mean()) for h in range(24)])
    hourly_count = np.array([int((hours == h).sum()) for h in range(24)])
    monthly_mean = {int(m): round(float(y[months == m].mean()), 2)
                    for m in sorted(set(months.tolist()))}
    tod_mean = {}
    for lo, hi, name in ((0, 6, "00-06h"), (6, 12, "06-12h"), (12, 18, "12-18h"), (18, 24, "18-24h")):
        mask = (hours >= lo) & (hours < hi)
        tod_mean[name] = round(float(y[mask].mean()), 2)

    night_mask = (hours >= 19) | (hours < 6)
    night_zero_frac = float((y[night_mask] == 0).mean()) if night_mask.sum() > 0 else float("nan")

    peak_hour = int(np.argmax(hourly_mean))
    peak_mean = float(hourly_mean.max())
    morning_rise = float(hourly_mean[12] - hourly_mean[6])
    afternoon_fall = float(hourly_mean[18] - hourly_mean[12])

    monthly_daily_peak = {}
    for m in sorted(set(months.tolist())):
        mm = months == m
        hm = np.array([y[mm & (hours == h)].mean() for h in range(24)])
        monthly_daily_peak[int(m)] = round(float(np.nanmax(hm)), 2) if hm.size else float("nan")

    thr = 0.05 * peak_mean if peak_mean > 0 else 1.0
    day_length_hours = float((hourly_mean > thr).sum())

    # ---- 持续性 / 波动（H3 / H4 需要）----
    lag1 = _corr(y[1:], y[:-1])
    lag144 = _corr(y[144:], y[:-144])
    diff = np.diff(y)
    diff_stats = {
        "mean": round(float(diff.mean()), 4),
        "std": round(float(diff.std()), 4),
        "abs_mean": round(float(np.abs(diff).mean()), 4),
    }
    rw = 144  # 1 day window
    roll_std = np.array([y[i:i + rw].std() for i in range(0, len(y) - rw + 1, rw)])
    rolling_std = {
        "mean": round(float(roll_std.mean()), 2),
        "std": round(float(roll_std.std()), 2),
        "max": round(float(roll_std.max()), 2),
        "min": round(float(roll_std.min()), 2),
    }

    # ---- 跨变量 / 代表性（H1 / H5 / H6 需要）----
    corr_s6y = _corr(s6, y)
    diff_corr_s6y = _corr(np.diff(s6), np.diff(y))
    monthly_corr = {int(m): round(_corr(s6[months == m], y[months == m]), 4)
                    for m in sorted(set(months.tolist()))}
    tod_corr = {}
    for lo, hi, name in ((0, 6, "00-06h"), (6, 12, "06-12h"), (12, 18, "12-18h"), (18, 24, "18-24h")):
        mask = (hours >= lo) & (hours < hi)
        tod_corr[name] = round(_corr(s6[mask], y[mask]), 4)

    per_plant_std = {v: round(float(df[v].to_numpy(np.float64)[:train_len].std()), 4)
                     for v in LOAD_VARS}
    pairwise = {}
    for i in range(len(LOAD_VARS)):
        for j in range(i + 1, len(LOAD_VARS)):
            key = f"{LOAD_VARS[i]}-{LOAD_VARS[j]}"
            pairwise[key] = round(_corr(df[LOAD_VARS[i]].to_numpy(np.float64)[:train_len],
                                        df[LOAD_VARS[j]].to_numpy(np.float64)[:train_len]), 4)

    return {
        "train_len": int(train_len),
        "N": int(N),
        "diurnal_seasonal": {
            "hourly_mean_OT": [round(float(v), 2) for v in hourly_mean],
            "hourly_count": [int(c) for c in hourly_count],
            "monthly_mean_OT": monthly_mean,
            "time_of_day_mean_OT": tod_mean,
            "night_zero_fraction": round(night_zero_frac, 4),
            "diurnal_peak_hour": peak_hour,
            "diurnal_peak_mean_OT": round(peak_mean, 2),
            "morning_rise_06_to_12": round(morning_rise, 2),
            "afternoon_fall_12_to_18": round(afternoon_fall, 2),
            "monthly_daily_peak": monthly_daily_peak,
            "day_length_hours_above_5pct_peak": round(day_length_hours, 2),
        },
        "persistence_variability": {
            "lag1_autocorr": round(lag1, 4),
            "lag144_autocorr": round(lag144, 4),
            "lag1_diff_stats": diff_stats,
            "rolling_std_OT_window144": rolling_std,
        },
        "cross_variable": {
            "corr_S6_Y": round(corr_s6y, 4),
            "diff_corr_S6_Y": round(diff_corr_s6y, 4),
            "monthly_corr_S6_Y": monthly_corr,
            "time_of_day_corr_S6_Y": tod_corr,
            "coverage": {
                "mechanism_scope": f"{N_POPULATION} plants (population-level aggregate)",
                "proxy_scope": f"{len(LOAD_VARS)} plants",
                "fraction": round(len(LOAD_VARS) / N_POPULATION, 4),
            },
            "OT_mean": round(float(y.mean()), 2),
            "OT_std": round(float(y.std()), 2),
            "per_plant_std": per_plant_std,
            "pairwise_corr_P00x": pairwise,
        },
    }


# --------------------------------------------------------------------------- #
# Stage: Evidence Planner (新 LLM)
# --------------------------------------------------------------------------- #
_EVIDENCE_PLANNER_SYSTEM = (
    "You are an EVIDENCE PLANNER for a time-series retrieval model. Given a "
    "mechanism hypothesis that already passed semantic review, you plan HOW to "
    "verify it using TRAIN data only. You do NOT judge whether the data supports "
    "it yet — you only specify what evidence SHOULD be computed.\n\n"
    "The dataset has a `date` timestamp, so you may safely construct time features: "
    "hour, minute, time_of_day, day_of_week, month, day_of_year, season, and "
    "within-day / within-year period position. These are NOT external data.\n"
    "You may also derive statistics from the target OT and covariates X1..X6: lag, "
    "difference, rolling variance/std, autocorrelation, correlation, co-movement, "
    "variance, aggregation (sum).\n\n"
    "CRITICAL: never plan to reject a hypothesis merely because a statistic is not "
    "pre-computed. Your job is to write a concrete computation plan so a "
    "deterministic executor can compute real numbers.\n\n"
    "Output strict JSON only."
)


def build_evidence_planner_prompt(entries):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"plans": [ ... ]}, one entry per hypothesis. Each entry:\n'
        '  - "hypothesis_id": short id (copy exactly)\n'
        '  - "name": hypothesis name (copy exactly)\n'
        '  - "hypothesis": what this knowledge actually needs to verify (1-2 sentences)\n'
        '  - "required_information": list of strings — information theoretically needed\n'
        '  - "directly_available": list of strings — what current data directly provides '
        "(include date/time-derived items like hour/month/day_of_year)\n"
        '  - "derived_features": list of strings — features derivable from TRAIN '
        '(e.g. "OT -> lag1", "date -> hour", "X -> correlation")\n'
        '  - "evidence_plan": list of strings — the concrete statistics to compute on '
        "TRAIN to verify this hypothesis\n"
    )
    user = "\n\n".join([
        av2._metadata_text(),
        "HYPOTHESES TO PLAN EVIDENCE FOR:\n" + json.dumps(entries, ensure_ascii=False, indent=2),
        schema,
        "TASK: plan evidence for every hypothesis. Output only the JSON object.",
    ])
    return _EVIDENCE_PLANNER_SYSTEM, user


# --------------------------------------------------------------------------- #
# Stage: Data Critic (新 LLM) — 拿到真实 evidence 再判
# --------------------------------------------------------------------------- #
_DATA_CRITIC_SYSTEM = (
    "You are a DATA critic. You decide whether a mechanism hypothesis can be turned "
    "into an executable retrieval prior, given (a) the hypothesis, (b) an evidence "
    "plan, and (c) the ACTUAL TRAIN evidence that a deterministic executor computed.\n\n"
    "Definitions:\n"
    "  ACCEPT      = mechanism plausible AND the required information can be reliably "
    "observed/constructed from current data AND the TRAIN evidence supports it AND it "
    "can form an executable retrieval signal.\n"
    "  UNSUPPORTED = the knowledge may be correct, but current data cannot reliably "
    "observe, verify, or execute it.\n"
    "  REJECT      = the knowledge itself, its assumption, or its proxy logic is "
    "clearly wrong.\n\n"
    "CRITICAL: \"a statistic was not pre-computed\" is NEVER a valid reason for "
    "UNSUPPORTED. The executor has already computed the planned evidence; base your "
    "decision on the actual numbers.\n"
    "If a mechanism refers to many/population/aggregate/system-wide/common-factor and "
    "the available proxy only covers a small arbitrary subset with no "
    "representativeness evidence, mark UNSUPPORTED.\n"
    "A high-frequency proxy (e.g. rolling volatility) may describe the current "
    "variability regime but must NOT be equated to an unobserved variable (e.g. "
    "cloud) unless the data validates it.\n\n"
    "Output strict JSON only."
)


def build_data_critic_prompt(entries_with_plan, evidence):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"decisions": [ ... ]}, one entry per hypothesis. Each entry:\n'
        '  - "hypothesis_id": short id (copy exactly)\n'
        '  - "name": hypothesis name (copy exactly)\n'
        '  - "decision": "ACCEPT" | "REJECT" | "UNSUPPORTED"\n'
        '  - "reason": 1-3 sentences grounded in the actual evidence values\n'
    )
    user = "\n\n".join([
        av2._metadata_text(),
        av2.RULES_BLOCK,
        "HYPOTHESES WITH EVIDENCE PLANS:\n" + json.dumps(entries_with_plan, ensure_ascii=False, indent=2),
        "ACTUAL TRAIN EVIDENCE (deterministic, computed by executor):\n"
        + json.dumps(evidence, ensure_ascii=False, indent=2),
        schema,
        "TASK: decide every hypothesis using the actual evidence. Output only the JSON object.",
    ])
    return _DATA_CRITIC_SYSTEM, user


# --------------------------------------------------------------------------- #
# pipeline driver
# --------------------------------------------------------------------------- #
def _spec(p):
    v = p.input_variables
    if set(v) == set(LOAD_VARS):
        inp = "X1..X6"
    elif v == ["OT"]:
        inp = "Y"
    else:
        inp = "+".join(v)
    return f"{p.transform}/{p.temporal_aggregation}/{p.similarity} on {inp}"


def run_pipeline_v3(model):
    evidence = solar_compute_evidence_catalog_v3()

    # ---- 1. proposal (复用 v2) ----
    prop_sys, prop_user = av2.build_proposal_prompt()
    prop_raw, prop_path, prop_cached = av2._call_llm(prop_sys, prop_user, "proposal", model)
    hypotheses = av2._list_from(prop_raw, "hypotheses")
    for i, h in enumerate(hypotheses, 1):
        h["hypothesis_id"] = f"H{i}"

    # ---- 2. semantic critic (复用 v2) ----
    sem_sys, sem_user = av2.build_semantic_prompt(hypotheses)
    sem_raw, sem_path, sem_cached = av2._call_llm(sem_sys, sem_user, "semantic_critic", model)
    critiques = av2._list_from(sem_raw, "critiques")
    by_name = {str(c.get("name", "")).strip().lower(): c for c in critiques}
    for h in hypotheses:
        c = by_name.get(str(h.get("name", "")).strip().lower())
        h["_semantic"] = c if c else {"verdict": "accept (no review)"}
    sem_accepted = [h for h in hypotheses
                    if str(h["_semantic"].get("verdict", "accept")).strip().lower() == "accept"]

    # ---- 3. evidence planner (新) ----
    ep_entries = [{"hypothesis_id": h["hypothesis_id"], "name": h["name"],
                   "domain_fact": h.get("domain_fact", ""),
                   "observable_proxy": h.get("observable_proxy", ""),
                   "expected_future_relationship": h.get("expected_future_relationship", "")}
                  for h in sem_accepted]
    ep_sys, ep_user = build_evidence_planner_prompt(ep_entries)
    ep_raw, ep_path, ep_cached = av2._call_llm(ep_sys, ep_user, "evidence_planner", model)
    plans = av2._list_from(ep_raw, "plans")
    plan_by_id = {str(p.get("hypothesis_id", "")).strip().upper(): p for p in plans}
    for h in sem_accepted:
        p = plan_by_id.get(str(h.get("hypothesis_id", "")).strip().upper())
        if p is None:
            p = plan_by_id.get(str(h.get("name", "")).strip().lower())
        h["_evidence_plan"] = p if p else {
            "hypothesis": "", "required_information": [], "directly_available": [],
            "derived_features": [], "evidence_plan": []}

    # ---- 4. TRAIN evidence executor (确定性；evidence 已在上面算好) ----

    # ---- 5. data critic (新) ----
    dc_entries = [{"hypothesis_id": h["hypothesis_id"], "name": h["name"],
                   "domain_fact": h.get("domain_fact", ""),
                   "required_information": h["_evidence_plan"].get("required_information", []),
                   "directly_available": h["_evidence_plan"].get("directly_available", []),
                   "derived_features": h["_evidence_plan"].get("derived_features", []),
                   "evidence_plan": h["_evidence_plan"].get("evidence_plan", [])}
                  for h in sem_accepted]
    dc_sys, dc_user = build_data_critic_prompt(dc_entries, evidence)
    dc_raw, dc_path, dc_cached = av2._call_llm(dc_sys, dc_user, "data_critic", model)
    decisions = av2._list_from(dc_raw, "decisions")
    dc_by_id = {str(d.get("hypothesis_id", "")).strip().upper(): d for d in decisions}
    for h in sem_accepted:
        d = dc_by_id.get(str(h.get("hypothesis_id", "")).strip().upper())
        if d is None:
            d = dc_by_id.get(str(h.get("name", "")).strip().lower())
        h["_data_support"] = d if d else {"decision": "UNSUPPORTED", "reason": "no decision"}

    data_accepted = [h for h in sem_accepted
                     if str(h["_data_support"].get("decision", "")).strip().upper() == "ACCEPT"]

    # ---- 6. compiler (复用 v2) ----
    compiled = []
    comp_sys = comp_user = comp_raw = comp_path = comp_cached = None
    if data_accepted:
        comp_sys, comp_user = av2.build_compile_prompt(data_accepted)
        comp_raw, comp_path, comp_cached = av2._call_llm(comp_sys, comp_user, "compiler", model)
        compiled = av2._list_from(comp_raw, "priors")

    exec_by_name = {}
    unsupported_compile = []
    for d in compiled:
        supported = str(d.get("supported", "true")).strip().lower() in ("true", "1", "yes")
        e = av2.compiled_to_exec(d) if supported else None
        if supported and e is not None:
            exec_by_name[str(d.get("name", "")).strip().lower()] = (d, e)
        else:
            unsupported_compile.append({"name": d.get("name"),
                                        "reason": d.get("unsupported_reason", "")})

    for h in hypotheses:
        h["_compiled"] = None
        if h in data_accepted:
            dd = exec_by_name.get(str(h.get("name", "")).strip().lower())
            h["_compiled"] = dd[0] if dd else None

    compiled_hyps = [h for h in data_accepted if h["_compiled"] is not None]

    # ---- 7. proxy validator (复用 v2) ----
    pv_by_id = {}
    pv_sys = pv_user = pv_raw = pv_path = pv_cached = None
    if compiled_hyps:
        pv_entries = [{"hypothesis_id": h["hypothesis_id"], "name": h["name"],
                       "original_mechanism": h.get("domain_fact", ""),
                       "compiled_proxy": h["_compiled"]} for h in compiled_hyps]
        pv_sys, pv_user = av2.build_proxy_validator_prompt(pv_entries)
        pv_raw, pv_path, pv_cached = av2._call_llm(pv_sys, pv_user, "proxy_validator", model)
        pv_validations = av2._list_from(pv_raw, "validations")
        pv_by_id = {str(v.get("hypothesis_id", "")).strip().upper(): v for v in pv_validations}

    for h in compiled_hyps:
        v = pv_by_id.get(str(h.get("hypothesis_id", "")).strip().upper())
        if v is None:
            v = pv_by_id.get(str(h.get("name", "")).strip().lower())
        h["_proxy_validator"] = v if v else {"decision": "FAIL", "reason": "no validation"}

    supported_hyps = [h for h in compiled_hyps
                      if str(h["_proxy_validator"].get("decision", "")).strip().upper() == "PASS"]
    exec_priors = [exec_by_name[str(h.get("name", "")).strip().lower()][1] for h in supported_hyps]

    # ---- 最终状态（对 neural 编码路径而言：Data Critic ACCEPT 即进入 encoder）----
    def final_state(h):
        if str(h["_semantic"].get("verdict", "accept")).strip().lower() != "accept":
            return "FINAL_REJECT", False
        d = str(h["_data_support"].get("decision", "")).strip().upper()
        if d == "REJECT":
            return "FINAL_REJECT", False
        if d == "UNSUPPORTED":
            return "FINAL_UNSUPPORTED", False
        # d == ACCEPT -> 进入 Expert Encoder（neural path 用其文本）
        return "FINAL_ACCEPT", True

    rows = []
    accepted_expert_texts = []
    for h in hypotheses:
        state, enters = final_state(h)
        name = h.get("name", "")
        if enters:
            accepted_expert_texts.append(h.get("domain_fact", ""))
        rows.append({
            "id": h["hypothesis_id"],
            "name": name,
            "hypothesis": h.get("domain_fact", ""),
            "semantic_critic": str(h["_semantic"].get("verdict", "")).strip().lower(),
            "data_critic": str(h["_data_support"].get("decision", "")).strip().upper()
            if h.get("_data_support") else "n/a",
            "compiler": _spec(exec_by_name[str(name).strip().lower()][1])
            if str(name).strip().lower() in exec_by_name else "not compiled",
            "proxy_validator": str(h["_proxy_validator"].get("decision", "")).strip().upper()
            if h.get("_proxy_validator") else "n/a",
            "final_state": state,
            "enters_expert_encoder": enters,
        })

    return {
        "model": model,
        "metadata": SOLAR_METADATA_V2,
        "expert_mechanism": SOLAR_EXPERT_MECHANISM,
        "evidence": evidence,
        "proposal": {"system": prop_sys, "user": prop_user, "raw_response": prop_raw,
                     "parsed": hypotheses, "cache_path": prop_path, "from_cache": prop_cached},
        "semantic_critic": {"system": sem_sys, "user": sem_user, "raw_response": sem_raw,
                            "parsed": critiques, "cache_path": sem_path, "from_cache": sem_cached},
        "evidence_planner": {"system": ep_sys, "user": ep_user, "raw_response": ep_raw,
                             "parsed": plans, "cache_path": ep_path, "from_cache": ep_cached},
        "evidence_executor": evidence,
        "data_critic": {"system": dc_sys, "user": dc_user, "raw_response": dc_raw,
                        "parsed": decisions, "cache_path": dc_path, "from_cache": dc_cached},
        "compiler": {"system": comp_sys, "user": comp_user, "raw_response": comp_raw,
                     "parsed": compiled, "cache_path": comp_path, "from_cache": comp_cached},
        "proxy_validator": {"system": pv_sys if compiled_hyps else None,
                            "user": pv_user if compiled_hyps else None,
                            "raw_response": pv_raw, "parsed": list(pv_by_id.values()),
                            "cache_path": pv_path, "from_cache": pv_cached},
        "unsupported_compile": unsupported_compile,
        "supported_priors": [p.to_dict() for p in exec_priors],
        "accepted_expert_texts": accepted_expert_texts,
        "rows": rows,
        "n_proposed": len(hypotheses),
        "n_semantic_accepted": len(sem_accepted),
        "n_data_accepted": len(data_accepted),
        "n_data_unsupported": len([h for h in sem_accepted
                                   if str(h["_data_support"].get("decision", "")).strip().upper() == "UNSUPPORTED"]),
        "n_data_rejected": len([h for h in sem_accepted
                                if str(h["_data_support"].get("decision", "")).strip().upper() == "REJECT"]),
        "n_compiled": len(compiled_hyps),
        "n_proxy_passed": len(supported_hyps),
    }


# --------------------------------------------------------------------------- #
# 人工审计 markdown
# --------------------------------------------------------------------------- #
def _fmt(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, indent=2)
    return str(v)


def render_audit_md(result):
    L = []
    L.append("# Solar Expert → Critic v3 审计报告（人工检查用）")
    L.append("")
    L.append(f"> 管线：Expert facts → Hypothesis Generator → Semantic Critic → Evidence Planner → "
             f"TRAIN Evidence Executor → Data Critic → Compiler → Proxy Validator")
    L.append(f"> 模型：{result['model']}；仅 Solar TRAIN + VAL；TEST 未访问。")
    L.append("")
    L.append("## 输入 Expert Facts（机制，未修改）")
    L.append("")
    L.append("```")
    L.append(result["expert_mechanism"])
    L.append("```")
    L.append("")

    hyps = result["proposal"]["parsed"]
    for h in hyps:
        eid = h.get("hypothesis_id", "?")
        name = h.get("name", "")
        sem = h.get("_semantic", {})
        plan = h.get("_evidence_plan", {})
        dc = h.get("_data_support", {})
        row = next((r for r in result["rows"] if r["id"] == eid), {})
        L.append("---")
        L.append("")
        L.append(f"## Expert {eid} — {name}")
        L.append("")
        L.append("**【Source】** LLM")
        L.append("")
        L.append("**【专家原始建议】**（全文，未改写）")
        L.append("")
        L.append(f"- domain_fact: {h.get('domain_fact', '')}")
        L.append(f"- latent_state: {h.get('latent_state', '')}")
        L.append(f"- observable_proxy: {h.get('observable_proxy', '')}")
        L.append(f"- expected_future_relationship: {h.get('expected_future_relationship', '')}")
        L.append(f"- confidence: {h.get('confidence', '')}")
        L.append("")
        L.append("**【Semantic Critic】**")
        L.append(f"Decision: {str(sem.get('verdict','')).upper()}")
        L.append(f"Reason: {sem.get('summary','')}")
        for k in ("consistency", "contradiction", "leakage", "duplication",
                  "variable_existence", "theoretical_expressibility"):
            if sem.get(k):
                L.append(f"- {k}: {sem[k]}")
        L.append("")
        L.append("**【Evidence Planner】**")
        L.append(f"Hypothesis: {plan.get('hypothesis','')}")
        L.append(f"Required Information: {_fmt(plan.get('required_information', []))}")
        L.append(f"Directly Available: {_fmt(plan.get('directly_available', []))}")
        L.append(f"Derived Features: {_fmt(plan.get('derived_features', []))}")
        L.append(f"Evidence Plan: {_fmt(plan.get('evidence_plan', []))}")
        L.append("")
        L.append("**【TRAIN Evidence Executor】**（实际计算的具体数值）")
        L.append("```")
        L.append(_fmt(result["evidence"]))
        L.append("```")
        L.append("")
        L.append("**【Data Critic】**")
        L.append(f"Decision: {str(dc.get('decision','')).upper()}")
        L.append(f"Reason: {dc.get('reason','')}")
        L.append("")
        L.append(f"**【最终状态】** {row.get('final_state','')}")
        L.append("")
        L.append(f"**【是否进入 Expert Encoder】** {'YES' if row.get('enters_expert_encoder') else 'NO'}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## 最终进入模型的 Expert Knowledge（Data Critic ACCEPT）")
    L.append("")
    if result["accepted_expert_texts"]:
        for i, t in enumerate(result["accepted_expert_texts"], 1):
            L.append(f"{i}. {t}")
    else:
        L.append("（无）")
    L.append("")
    return "\n".join(L)


def main():
    _seed_v3_cache()

    av2.ELECTRICITY_METADATA_V2 = SOLAR_METADATA_V2
    av2.ELECTRICITY_EXPERT_MECHANISM = SOLAR_EXPERT_MECHANISM
    av2.BLIND_TO_REAL = BLIND_TO_REAL
    av2.LOAD_VARS = LOAD_VARS
    av2.N_POPULATION = N_POPULATION
    av2.CACHE_DIR = CACHE_DIR

    model = av2._default_model()
    result = run_pipeline_v3(model)

    print("=" * 100)
    print("SOLAR Expert->Critic v3 (Evidence Planner -> Evidence Executor -> Data Critic)")
    print("=" * 100)
    print(f"model = {model}")
    print(f"proposed          : {result['n_proposed']}")
    print(f"semantic accepted : {result['n_semantic_accepted']}")
    print(f"data ACCEPT       : {result['n_data_accepted']}")
    print(f"data UNSUPPORTED  : {result['n_data_unsupported']}")
    print(f"data REJECT       : {result['n_data_rejected']}")
    print(f"compiled          : {result['n_compiled']}")
    print(f"proxy passed      : {result['n_proxy_passed']}")
    print("\n" + "-" * 100)
    print(f"{'ID':<4}{'hypothesis':<30}{'sem':<8}{'data':<12}{'final'}")
    print("-" * 100)
    for r in result["rows"]:
        print(f"{r['id']:<4}{r['name']:<30}{r['semantic_critic']:<8}{r['data_critic']:<12}{r['final_state']}")

    print("\nAccepted expert texts (enter Expert Encoder):")
    for i, t in enumerate(result["accepted_expert_texts"], 1):
        print(f"  [{i}] {t}")

    ts = datetime.now().isoformat(timespec="seconds")
    audit = {
        "dataset": "solar",
        "agent_version": "v3",
        "model": model,
        "timestamp": ts,
        "metadata_sent_to_llm": result["metadata"],
        "expert_facts": result["expert_mechanism"],
        "train_evidence": result["evidence"],
        "proposal": {k: v for k, v in result["proposal"].items() if k != "parsed"},
        "hypotheses": result["proposal"]["parsed"],
        "semantic_critic": result["semantic_critic"],
        "evidence_planner": result["evidence_planner"],
        "evidence_executor": result["evidence_executor"],
        "data_critic": result["data_critic"],
        "compiler": result["compiler"],
        "proxy_validator": result["proxy_validator"],
        "unsupported_compile": result["unsupported_compile"],
        "supported_priors": result["supported_priors"],
        "accepted_expert_texts": result["accepted_expert_texts"],
        "summary": {
            "n_proposed": result["n_proposed"],
            "n_semantic_accepted": result["n_semantic_accepted"],
            "n_data_accepted": result["n_data_accepted"],
            "n_data_unsupported": result["n_data_unsupported"],
            "n_data_rejected": result["n_data_rejected"],
            "n_compiled": result["n_compiled"],
            "n_proxy_passed": result["n_proxy_passed"],
        },
        "rows": result["rows"],
        "test_access": "NO (TRAIN/VAL only; TEST untouched)",
        "leakage_check": "TRAIN evidence + VAL only; TEST never read",
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/solar_agent_v3_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    Path("results/expert_critic_v3_audit.md").write_text(
        render_audit_md(result), encoding="utf-8")
    print("\nSaved results/solar_agent_v3_audit.json")
    print("Saved results/expert_critic_v3_audit.md")
    return audit


if __name__ == "__main__":
    main()
