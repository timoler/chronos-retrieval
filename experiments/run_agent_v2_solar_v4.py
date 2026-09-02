#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solar Conditional Expert → Critic v3 → Activation Compiler (v4)。

核心改动（相对 v3）：
  1. 新增「阶段 A」Conditional Expert 生成：LLM Expert 查看 TRAIN 摘要 + 时间戳 +
     可观测变量，生成**带激活条件**的知识，每条包含
     {name, knowledge, activation_condition, observable_features, expected_relationship, reason,
      domain_fact, latent_state, observable_proxy, expected_future_relationship, confidence}。
     其中 observable_features 必须从固定 catalog 中选取（保证可确定性计算）。
  2. Critic v3 **完全不变**：Semantic Critic → Evidence Planner → TRAIN Evidence
     Executor → Data Critic（复用 run_agent_v2_solar_v3 的 prompt builders）。
  3. 新增 Activation Compiler（确定性）：把 ACCEPT 专家的 observable_features 映射到
     一个可对 query/candidate 计算的 activation feature a_i(q)。
  4. 新增 Activation Validator（LLM）：检查编译出的 activation feature 是否忠实实现
     activation_condition（语义不一致 → FAIL）。

禁止：Attention、softmax 选择 Expert、mean-pool 固定向量。只用 Solar TRAIN+VAL。
输出：results/solar_agent_v4_audit.json + results/expert_critic_v4_audit.md
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

import run_agent_v2_electricity as av2          # _call_llm / _metadata_text / RULES_BLOCK / build_semantic_prompt
import run_agent_v2_solar as av2s                # solar 常量
import run_agent_v2_solar_v3 as v3               # 复用 Critic v3（Semantic/Evidence Planner/Data Critic/Evidence Executor）

PERIOD = av2s.PERIOD
TREND_WINDOW = av2s.TREND_WINDOW
VAL_STRIDE = av2s.VAL_STRIDE
LOAD_VARS = av2s.LOAD_VARS
N_POPULATION = av2s.N_POPULATION
BLIND_TO_REAL = av2s.BLIND_TO_REAL
SOLAR_METADATA_V2 = av2s.SOLAR_METADATA_V2
SOLAR_EXPERT_MECHANISM = av2s.SOLAR_EXPERT_MECHANISM

CACHE_DIR = "cache/llm_priors_v4_solar"

# --------------------------------------------------------------------------- #
# activation feature catalog（每项可对 query context + timestamp 确定性计算）
# --------------------------------------------------------------------------- #
ACTIVATION_CATALOG = [
    "daylight_level",        # TRAIN hourly_mean_OT[hour] / peak  -> 日内阶段（0~1）
    "seasonal_level",        # TRAIN monthly_mean_OT[month] / max  -> 季节阶段（0~1）
    "recent_level",          # zscore(context mean) -> sigmoid      -> 高/低出力状态
    "recent_slope_magnitude",  # zscore(|linear slope|) -> sigmoid   -> 近期趋势强度
    "recent_diff_magnitude",   # zscore(|last diff|) -> sigmoid      -> 近期点间变化幅度
    "recent_volatility",       # zscore(rolling std) -> sigmoid      -> 近期波动状态
    "recent_autocorr",         # lag-1 autocorr -> [0,1]             -> 近期持续性强度
]

ACTIVATION_CATALOG_TEXT = (
    "AVAILABLE ACTIVATION FEATURES (choose observable_features ONLY from this list, 1-2 items):\n"
    "  - daylight_level: current time-of-day output level (0=night, 1=peak), from TRAIN hourly means\n"
    "  - seasonal_level: current season output level, from TRAIN monthly means\n"
    "  - recent_level: current output level relative to typical (z-scored context mean)\n"
    "  - recent_slope_magnitude: strength of recent trend (z-scored |linear slope| of context)\n"
    "  - recent_diff_magnitude: magnitude of the most recent point-to-point change\n"
    "  - recent_volatility: recent variability regime (z-scored rolling std of context)\n"
    "  - recent_autocorr: strength of recent persistence (lag-1 autocorr of context)"
)


def _train_summary_text() -> str:
    ev = v3.solar_compute_evidence_catalog_v3()
    ds = ev["diurnal_seasonal"]
    pv = ev["persistence_variability"]
    cv = ev["cross_variable"]
    return (
        "TRAIN-ONLY DATA SUMMARY (deterministic):\n"
        f"- hourly mean OT: peak hour {ds['diurnal_peak_hour']} = {ds['diurnal_peak_mean_OT']}, "
        f"night zero fraction {ds['night_zero_fraction']}\n"
        f"- hourly mean OT (24h): {ds['hourly_mean_OT']}\n"
        f"- monthly mean OT: {ds['monthly_mean_OT']}\n"
        f"- monthly daily peak: {ds['monthly_daily_peak']}\n"
        f"- lag1 autocorr {pv['lag1_autocorr']}, lag144 autocorr {pv['lag144_autocorr']}\n"
        f"- rolling std (144): mean {pv['rolling_std_OT_window144']['mean']}, std {pv['rolling_std_OT_window144']['std']}\n"
        f"- corr(S6,Y) {cv['corr_S6_Y']}, diff corr {cv['diff_corr_S6_Y']}, coverage {cv['coverage']['fraction']}\n"
        f"- OT mean {cv['OT_mean']}, OT std {cv['OT_std']}\n"
    )


# --------------------------------------------------------------------------- #
# Stage A: Conditional Expert 生成（新）
# --------------------------------------------------------------------------- #
_COND_SYSTEM = (
    "You are a domain expert on solar photovoltaic generation forecasting. A retrieval "
    "model forecasts a univariate solar generation series (OT, 10-min resolution) by "
    "retrieving similar historical windows from a memory bank and averaging their "
    "futures.\n\n"
    "You propose CONDITIONAL expert knowledge: each statement must say WHEN it applies "
    "(based on the current query's observable state), WHAT historical segments should "
    "then be preferred, and WHY. Do NOT propose unconditional statements like 'solar "
    "output persists' — always attach an activation condition.\n\n"
    "The activation condition must be observable from the query's own OT history and its "
    "timestamp ONLY (no external covariates).\n\n"
    "Output strict JSON only."
)


def build_conditional_proposal_prompt():
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"expert_knowledge": [ ... ]} with 4 to 8 entries. Each entry:\n'
        '  - "name": short label\n'
        '  - "knowledge": one conditional sentence: "When [observable condition], historical '
        "segments satisfying [relationship] should be preferred because [reason]\"\n"
        '  - "activation_condition": the "when [observable state]" part, in natural language\n'
        '  - "observable_features": list of 1-2 features from the AVAILABLE ACTIVATION FEATURES list\n'
        '  - "expected_relationship": the "prefer [relationship]" part\n'
        '  - "reason": why this relationship helps retrieval\n'
        '  - "domain_fact": a 1-sentence mechanism (same idea as knowledge, for semantic review)\n'
        '  - "latent_state": the hidden predictive state\n'
        '  - "observable_proxy": the observable quantity proxying the state\n'
        '  - "expected_future_relationship": expected future behavior\n'
        '  - "confidence": number in [0,1]\n'
    )
    user = "\n\n".join([
        av2._metadata_text(),
        _train_summary_text(),
        ACTIVATION_CATALOG_TEXT,
        "DOMAIN FACTS (physical mechanisms, for grounding):\n" + SOLAR_EXPERT_MECHANISM,
        schema,
        "TASK: propose 4-8 conditional expert-knowledge statements. Output only the JSON object.",
    ])
    return _COND_SYSTEM, user


# --------------------------------------------------------------------------- #
# Activation Compiler（确定性：observable_features -> catalog feature）
# --------------------------------------------------------------------------- #
def compile_activation(h):
    """把 observable_features 映射到 catalog feature（取第一个命中的）。"""
    feats = [str(f).strip() for f in h.get("observable_features", [])]
    for f in feats:
        if f in ACTIVATION_CATALOG:
            return f
    return None


# --------------------------------------------------------------------------- #
# Activation Validator（LLM：编译出的 feature 是否忠实实现 activation_condition）
# --------------------------------------------------------------------------- #
_ACT_VALIDATOR_SYSTEM = (
    "You validate that a compiled ACTIVATION FEATURE faithfully realizes an expert "
    "knowledge's activation condition. If the activation condition is about the diurnal "
    "envelope but the compiled feature is just the current output level, that is a "
    "semantic mismatch and you must FAIL. Output strict JSON only."
)


def build_activation_validator_prompt(entries):
    schema = (
        "OUTPUT SCHEMA (strict JSON):\n"
        'Return {"validations": [ ... ]}, one entry per expert. Each entry:\n'
        '  - "name": name (copy exactly)\n'
        '  - "activation_condition": the condition text\n'
        '  - "compiled_feature": the feature name\n'
        '  - "semantic_fidelity": "HIGH"|"MEDIUM"|"LOW"\n'
        '  - "decision": "PASS"|"FAIL"\n'
        '  - "reason": 1-2 sentences\n'
        "RULE: if the compiled_feature does NOT capture the activation_condition, "
        "semantic_fidelity=LOW, decision=FAIL."
    )
    user = "\n\n".join([
        av2._metadata_text(),
        ACTIVATION_CATALOG_TEXT,
        "EXPERTS TO VALIDATE:\n" + json.dumps(entries, ensure_ascii=False, indent=2),
        schema,
        "TASK: validate every compiled activation feature. Output only the JSON object.",
    ])
    return _ACT_VALIDATOR_SYSTEM, user


# --------------------------------------------------------------------------- #
# pipeline driver
# --------------------------------------------------------------------------- #
def run_pipeline_v4(model):
    evidence = v3.solar_compute_evidence_catalog_v3()

    # ---- A. conditional expert 生成 ----
    prop_sys, prop_user = build_conditional_proposal_prompt()
    prop_raw, prop_path, prop_cached = av2._call_llm(prop_sys, prop_user, "conditional_expert", model)
    experts = av2._list_from(prop_raw, "expert_knowledge")
    for i, h in enumerate(experts, 1):
        h["hypothesis_id"] = f"E{i}"

    # ---- B. semantic critic（复用 v3）----
    sem_sys, sem_user = av2.build_semantic_prompt(experts)
    sem_raw, sem_path, sem_cached = av2._call_llm(sem_sys, sem_user, "semantic_critic", model)
    critiques = av2._list_from(sem_raw, "critiques")
    by_name = {str(c.get("name", "")).strip().lower(): c for c in critiques}
    for h in experts:
        c = by_name.get(str(h.get("name", "")).strip().lower())
        h["_semantic"] = c if c else {"verdict": "accept (no review)"}
    sem_accepted = [h for h in experts
                    if str(h["_semantic"].get("verdict", "accept")).strip().lower() == "accept"]

    # ---- C. evidence planner（复用 v3）----
    ep_entries = [{"hypothesis_id": h["hypothesis_id"], "name": h["name"],
                   "domain_fact": h.get("domain_fact", ""),
                   "observable_proxy": h.get("observable_proxy", ""),
                   "expected_future_relationship": h.get("expected_future_relationship", "")}
                  for h in sem_accepted]
    ep_sys, ep_user = v3.build_evidence_planner_prompt(ep_entries)
    ep_raw, ep_path, ep_cached = av2._call_llm(ep_sys, ep_user, "evidence_planner", model)
    plans = av2._list_from(ep_raw, "plans")
    plan_by_id = {str(p.get("hypothesis_id", "")).strip().upper(): p for p in plans}
    for h in sem_accepted:
        p = plan_by_id.get(str(h.get("hypothesis_id", "")).strip().upper())
        if p is None:
            p = plan_by_id.get(str(h.get("name", "")).strip().lower())
        h["_evidence_plan"] = p if p else {"hypothesis": "", "required_information": [],
                                          "directly_available": [], "derived_features": [],
                                          "evidence_plan": []}

    # ---- D. evidence executor（复用 v3，evidence 已算好）----

    # ---- E. data critic（复用 v3）----
    dc_entries = [{"hypothesis_id": h["hypothesis_id"], "name": h["name"],
                   "domain_fact": h.get("domain_fact", ""),
                   "required_information": h["_evidence_plan"].get("required_information", []),
                   "directly_available": h["_evidence_plan"].get("directly_available", []),
                   "derived_features": h["_evidence_plan"].get("derived_features", []),
                   "evidence_plan": h["_evidence_plan"].get("evidence_plan", [])}
                  for h in sem_accepted]
    dc_sys, dc_user = v3.build_data_critic_prompt(dc_entries, evidence)
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

    # ---- F. activation compiler（确定性）----
    for h in experts:
        h["_activation_feature"] = compile_activation(h)

    act_valid_entries = [{"name": h.get("name", ""),
                          "activation_condition": h.get("activation_condition", ""),
                          "compiled_feature": h["_activation_feature"]}
                         for h in data_accepted]
    act_val_by_name = {}
    act_val_sys = act_val_user = act_val_raw = act_val_path = act_val_cached = None
    if act_valid_entries:
        act_val_sys, act_val_user = build_activation_validator_prompt(act_valid_entries)
        act_val_raw, act_val_path, act_val_cached = av2._call_llm(
            act_val_sys, act_val_user, "activation_validator", model)
        act_vals = av2._list_from(act_val_raw, "validations")
        act_val_by_name = {str(v.get("name", "")).strip().lower(): v for v in act_vals}

    for h in experts:
        v = act_val_by_name.get(str(h.get("name", "")).strip().lower())
        h["_activation_validator"] = v if v else {"decision": "FAIL", "reason": "no validation"}

    # ---- final：进入模型的 expert = Data ACCEPT 且 activation 编译有效 且 validator PASS ----
    final_accepted = []
    for h in data_accepted:
        if h["_activation_feature"] is None:
            continue
        if str(h["_activation_validator"].get("decision", "")).strip().upper() != "PASS":
            continue
        final_accepted.append(h)

    rows = []
    for h in experts:
        sem = str(h["_semantic"].get("verdict", "")).strip().lower()
        dc = str(h["_data_support"].get("decision", "")).strip().upper() if h.get("_data_support") else "n/a"
        act = h["_activation_feature"] or "n/a"
        av = str(h["_activation_validator"].get("decision", "")).strip().upper() if h.get("_activation_validator") else "n/a"
        if sem != "accept":
            final = "FINAL_REJECT"
        elif dc == "REJECT":
            final = "FINAL_REJECT"
        elif dc == "UNSUPPORTED":
            final = "FINAL_UNSUPPORTED"
        elif h in final_accepted:
            final = "FINAL_ACCEPT"
        else:
            final = "FINAL_REJECT (activation)"
        rows.append({
            "id": h["hypothesis_id"], "name": h.get("name", ""),
            "semantic": sem, "data_critic": dc,
            "activation_feature": act, "activation_validator": av,
            "final_state": final,
        })

    accepted_experts = []
    for h in final_accepted:
        accepted_experts.append({
            "name": h.get("name", ""),
            "knowledge": h.get("knowledge", ""),
            "activation_condition": h.get("activation_condition", ""),
            "observable_features": h.get("observable_features", []),
            "activation_feature": h["_activation_feature"],
            "expected_relationship": h.get("expected_relationship", ""),
            "reason": h.get("reason", ""),
        })

    return {
        "model": model,
        "metadata": SOLAR_METADATA_V2,
        "expert_mechanism": SOLAR_EXPERT_MECHANISM,
        "evidence": evidence,
        "conditional_expert": {"system": prop_sys, "user": prop_user, "raw_response": prop_raw,
                               "parsed": experts, "cache_path": prop_path, "from_cache": prop_cached},
        "semantic_critic": {"system": sem_sys, "user": sem_user, "raw_response": sem_raw,
                            "parsed": critiques, "cache_path": sem_path, "from_cache": sem_cached},
        "evidence_planner": {"system": ep_sys, "user": ep_user, "raw_response": ep_raw,
                             "parsed": plans, "cache_path": ep_path, "from_cache": ep_cached},
        "evidence_executor": evidence,
        "data_critic": {"system": dc_sys, "user": dc_user, "raw_response": dc_raw,
                        "parsed": decisions, "cache_path": dc_path, "from_cache": dc_cached},
        "activation_validator": {"system": act_val_sys, "user": act_val_user, "raw_response": act_val_raw,
                                 "parsed": list(act_val_by_name.values()),
                                 "cache_path": act_val_path, "from_cache": act_val_cached},
        "accepted_experts": accepted_experts,
        "rows": rows,
        "n_proposed": len(experts),
        "n_semantic_accepted": len(sem_accepted),
        "n_data_accepted": len(data_accepted),
        "n_final_accepted": len(final_accepted),
    }


# --------------------------------------------------------------------------- #
# 人工审计 markdown
# --------------------------------------------------------------------------- #
def render_audit_md(result):
    L = []
    L.append("# Solar Conditional Expert → Critic v3 → Activation 审计报告（人工检查用）")
    L.append("")
    L.append(f"> 管线：Conditional Expert 生成 → Semantic Critic → Evidence Planner → TRAIN Evidence Executor "
             f"→ Data Critic → Activation Compiler → Activation Validator")
    L.append(f"> 模型：{result['model']}；仅 Solar TRAIN+VAL；TEST 未访问；无 Attention；无 mean-pool。")
    L.append("")

    experts = result["conditional_expert"]["parsed"]
    for h in experts:
        eid = h.get("hypothesis_id", "?")
        name = h.get("name", "")
        row = next((r for r in result["rows"] if r["id"] == eid), {})
        sem = h.get("_semantic", {})
        plan = h.get("_evidence_plan", {})
        dc = h.get("_data_support", {})
        L.append("---")
        L.append("")
        L.append(f"## Expert {eid} — {name}")
        L.append("")
        L.append("**【Source】** LLM")
        L.append("")
        L.append("**【Expert 原始建议】**（全文，未改写）")
        L.append(f"- knowledge: {h.get('knowledge','')}")
        L.append(f"- reason: {h.get('reason','')}")
        L.append("")
        L.append("**【activation_condition】**")
        L.append(h.get("activation_condition", ""))
        L.append("")
        L.append("**【observable_features】**")
        L.append(json.dumps(h.get("observable_features", []), ensure_ascii=False))
        L.append("")
        L.append("**【Semantic Critic】**")
        L.append(f"Decision: {str(sem.get('verdict','')).upper()}")
        L.append(f"Reason: {sem.get('summary','')}")
        L.append("")
        L.append("**【Evidence Planner】**")
        L.append(f"Hypothesis: {plan.get('hypothesis','')}")
        L.append(f"Evidence Plan: {json.dumps(plan.get('evidence_plan', []), ensure_ascii=False, indent=2)}")
        L.append("")
        L.append("**【TRAIN Evidence】**（关键数值）")
        ev = result["evidence"]
        L.append(f"- hourly_mean_OT: {ev['diurnal_seasonal']['hourly_mean_OT']}")
        L.append(f"- monthly_mean_OT: {ev['diurnal_seasonal']['monthly_mean_OT']}")
        L.append(f"- lag1_autocorr: {ev['persistence_variability']['lag1_autocorr']}")
        L.append("")
        L.append("**【Data Critic】**")
        L.append(f"Decision: {str(dc.get('decision','')).upper()}")
        L.append(f"Reason: {dc.get('reason','')}")
        L.append("")
        L.append("**【Compiler（activation function）】**")
        L.append(f"compiled_feature: {row.get('activation_feature','')}")
        L.append("")
        L.append("**【最终 activation function】**")
        L.append(f"a_i(query) = normalize({row.get('activation_feature','')})  (deterministic, TRAIN-z-scored)")
        L.append("")
        L.append(f"**【最终状态】** {row.get('final_state','')}")
        L.append("")
        L.append(f"**【是否进入模型】** {'YES' if row.get('final_state') == 'FINAL_ACCEPT' else 'NO'}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## 最终进入模型的 Expert（含 activation function）")
    L.append("")
    if result["accepted_experts"]:
        for i, e in enumerate(result["accepted_experts"], 1):
            L.append(f"{i}. {e['name']}  |  activation_feature = {e['activation_feature']}")
            L.append(f"   knowledge: {e['knowledge']}")
    else:
        L.append("（无）")
    L.append("")
    return "\n".join(L)


def main():
    av2.ELECTRICITY_METADATA_V2 = SOLAR_METADATA_V2
    av2.ELECTRICITY_EXPERT_MECHANISM = SOLAR_EXPERT_MECHANISM
    av2.BLIND_TO_REAL = BLIND_TO_REAL
    av2.LOAD_VARS = LOAD_VARS
    av2.N_POPULATION = N_POPULATION
    av2.CACHE_DIR = CACHE_DIR

    model = av2._default_model()
    result = run_pipeline_v4(model)

    print("=" * 100)
    print("SOLAR Conditional Expert -> Critic v3 -> Activation (v4)")
    print("=" * 100)
    print(f"model = {model}")
    print(f"proposed          : {result['n_proposed']}")
    print(f"semantic accepted : {result['n_semantic_accepted']}")
    print(f"data ACCEPT       : {result['n_data_accepted']}")
    print(f"final accepted    : {result['n_final_accepted']}")
    print("\n" + "-" * 100)
    print(f"{'ID':<5}{'name':<34}{'sem':<8}{'data':<12}{'act_feature':<24}{'final'}")
    print("-" * 100)
    for r in result["rows"]:
        print(f"{r['id']:<5}{r['name']:<34}{r['semantic']:<8}{r['data_critic']:<12}"
              f"{r['activation_feature']:<24}{r['final_state']}")

    print("\nAccepted experts (enter model):")
    for i, e in enumerate(result["accepted_experts"], 1):
        print(f"  [{i}] {e['name']}  activation={e['activation_feature']}")
        print(f"      knowledge: {e['knowledge']}")

    ts = datetime.now().isoformat(timespec="seconds")
    audit = {
        "dataset": "solar",
        "agent_version": "v4",
        "model": model,
        "timestamp": ts,
        "metadata_sent_to_llm": result["metadata"],
        "expert_facts": result["expert_mechanism"],
        "train_evidence": result["evidence"],
        "conditional_expert": {k: v for k, v in result["conditional_expert"].items() if k != "parsed"},
        "experts": result["conditional_expert"]["parsed"],
        "semantic_critic": result["semantic_critic"],
        "evidence_planner": result["evidence_planner"],
        "evidence_executor": result["evidence_executor"],
        "data_critic": result["data_critic"],
        "activation_validator": result["activation_validator"],
        "accepted_experts": result["accepted_experts"],
        "summary": {
            "n_proposed": result["n_proposed"],
            "n_semantic_accepted": result["n_semantic_accepted"],
            "n_data_accepted": result["n_data_accepted"],
            "n_final_accepted": result["n_final_accepted"],
        },
        "rows": result["rows"],
        "test_access": "NO (TRAIN/VAL only; TEST untouched)",
        "leakage_check": "TRAIN evidence + VAL only; TEST never read",
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/solar_agent_v4_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    Path("results/expert_critic_v4_audit.md").write_text(
        render_audit_md(result), encoding="utf-8")
    print("\nSaved results/solar_agent_v4_audit.json")
    print("Saved results/expert_critic_v4_audit.md")
    return audit


if __name__ == "__main__":
    main()
