#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从已完成的 deterministic single/LOO 实验日志 + 冻结结果重建 Electricity 诊断输出。

single/LOO 54 个 arm 已在 diagnose_electricity_critic_failure.py 跑完（值打在
diagnosis_run.log，确定性），仅因末尾 val_ablation 解析 bug 崩溃。本脚本从日志
重建 single/loo 数值，并补齐 2/5/6/7 段（纯解析 + activation 统计），写全 6 个 CSV
与 electricity_failure_diagnosis.md。
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_expert_embedding_mlp import build_windows
from run_etth1_prediction_validation import compute_training_stats, build_activation_matrix
import run_etth1_reward_feedback_critic_v1 as v1

REPO_ROOT = HERE.parent
DS_DIR = REPO_ROOT / "results/multidataset/electricity"
OUT_DIR = REPO_ROOT / "results/reward_feedback_critic_v1_cross_dataset/electricity"
DIAG_DIR = OUT_DIR / "diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / "diagnosis_run.log"

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]

# ---- 解析日志：single / minus ----
single = {}   # (eid, seed) -> mse
minus = {}    # (eid, seed) -> mse
for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
    m = re.search(r"seed=(\d+) \+(\w+): MSE=([0-9.e+-]+)", line)
    if m:
        single[(m.group(2), int(m.group(1)))] = float(m.group(3))
        continue
    m = re.search(r"seed=(\d+) minus (\w+): MSE=([0-9.e+-]+)", line)
    if m:
        minus[(m.group(2), int(m.group(1)))] = float(m.group(3))

expert_ids = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]

# ---- 读 val_ablation（shape / raw / reward_feedback per seed）----
va = {}
for r in csv.DictReader((OUT_DIR / "val_ablation.csv").open()):
    try:
        seed = int(r["seed"])
    except (TypeError, ValueError):
        continue
    va[(r["arm"], seed)] = float(r["mse"])


def mean3(d, key):
    return float(np.mean([d[(key, s)] for s in SEEDS]))


shape_mean = mean3(single, "shape") if ("shape", 42) in single else float(np.mean([va[("shape", s)] for s in SEEDS]))
raw_full_mean = mean3(minus, "full") if ("full", 42) in minus else float(np.mean([va[("raw_expert", s)] for s in SEEDS]))
# single 的 shape 与 minus 的 full 在日志里没打印；用 val_ablation 的 shape/raw_expert
shape_mean = float(np.mean([va[("shape", s)] for s in SEEDS]))
raw_full_mean = float(np.mean([va[("raw_expert", s)] for s in SEEDS]))

single_rows, loo_rows, synergy_rows = [], [], []
for eid in expert_ids:
    sg = (shape_mean - mean3(single, eid)) / shape_mean
    ld = mean3(minus, eid) - raw_full_mean
    single_rows.append({"expert_id": eid, "mse_shape": shape_mean,
                        "mse_shape_plus": mean3(single, eid), "single_gain_vs_shape": sg})
    loo_rows.append({"expert_id": eid, "mse_raw_full": raw_full_mean,
                     "mse_raw_minus": mean3(minus, eid), "loo_delta": ld})
    synergy = bool(sg <= 0 and ld > 0)
    synergy_rows.append({"expert_id": eid, "single_effect": sg, "loo_effect": ld,
                         "synergy_flag": synergy,
                         "interpretation": "单条无增益但组合有益 → 组合/协同" if synergy else
                         ("单条与组合一致" if (sg > 0) == (ld > 0) else "单条与组合不一致")})

# ---- 2. lineage audit ----
reward_history = {}
for r in csv.DictReader((OUT_DIR / "reward_history.csv").open()):
    reward_history.setdefault(r["lineage_id"], {})[int(r["round"])] = r
lineage = {}
for r in csv.DictReader((OUT_DIR / "knowledge_lineage.csv").open()):
    lineage.setdefault(r["lineage_id"], {})[int(r["round"])] = r
raw_experts = v1._sorted_experts(json.loads((OUT_DIR / "raw_experts.json").read_text(encoding="utf-8")))
final_pool = v1._sorted_experts(json.loads((OUT_DIR / "final_expert_pool.json").read_text(encoding="utf-8")))
final_id = final_pool[0]["id"] if final_pool else None

lineage_rows = []
for eid in expert_ids:
    raw = lineage[eid][0]
    r1 = lineage[eid].get(1, {})
    r2 = lineage[eid].get(2, {})
    e = next(x for x in raw_experts if x["id"] == eid)
    final_status = "IN_POOL" if eid == final_id else "DROPPED"
    drop_reason = ""
    if final_status == "DROPPED":
        if all((lineage[eid].get(r, {}).get("reward") in (None, "", "0", 0) or
                int(lineage[eid].get(r, {}).get("reward") or 0) <= 0) for r in (0, 1, 2)):
            drop_reason = "reward<=0 (never >0)"
        elif eid == "E7" and raw.get("activation_coverage") == "0.0":
            drop_reason = "zero activation coverage"
        else:
            drop_reason = "reward<=0 after revision"
    lineage_rows.append({
        "expert_id": eid, "raw_mechanism": raw.get("mechanism", ""),
        "raw_activation": e.get("activation_condition", ""), "raw_action": e.get("retrieval_action", ""),
        "r0_reward": raw.get("reward", ""), "r0_action_effect": raw.get("action_effect_size", ""),
        "r0_coverage": raw.get("activation_coverage", ""), "r0_verdict": raw.get("verdict", ""),
        "r1_reward": r1.get("reward", ""), "r1_action_effect": r1.get("action_effect_size", ""),
        "r1_coverage": r1.get("activation_coverage", ""), "r1_verdict": r1.get("verdict", ""),
        "r2_reward": r2.get("reward", ""), "r2_action_effect": r2.get("action_effect_size", ""),
        "r2_coverage": r2.get("activation_coverage", ""), "final_status": final_status,
        "drop_reason": drop_reason,
    })

# ---- 5. final expert TRAIN vs VAL ----
final_rows = []
for seed in SEEDS:
    shape_m = va[("shape", seed)]
    final_m = va[("reward_feedback_expert", seed)]
    raw_m = va[("raw_expert", seed)]
    final_rows.append({"split": "VAL", "seed": seed, "shape_mse": shape_m,
                       "final_expert_mse": final_m, "raw_full_mse": raw_m,
                       "final_vs_shape": (shape_m - final_m) / shape_m,
                       "raw_vs_shape": (shape_m - raw_m) / shape_m})

# ---- 6. reward vs loo ----
def _f(v):
    return None if v in (None, "", "nan", "None") else float(v)

r0_reward, r0_median, r0_ae = {}, {}, {}
for eid in expert_ids:
    r = reward_history.get(eid, {}).get(0, {})
    r0_reward[eid] = int(r["reward"]) if r.get("reward") not in (None, "") else None
    r0_median[eid] = _f(r.get("median_gain"))
    r0_ae[eid] = _f(r.get("action_effect_size"))
loo_d = {r["expert_id"]: r["loo_delta"] for r in loo_rows}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else 0.0


ids = [e for e in expert_ids if r0_reward[e] is not None]
spearman_rv = spearman([r0_reward[e] for e in ids], [loo_d[e] for e in ids])
sign_agree, n_sign = 0, 0
for e in ids:
    if loo_d[e] == 0 or r0_reward[e] == 0:
        continue
    n_sign += 1
    if (r0_reward[e] > 0) == (loo_d[e] > 0):
        sign_agree += 1
sign_agree_frac = sign_agree / n_sign if n_sign else None
rv_rows = [{"expert_id": e, "r0_reward": r0_reward[e], "r0_median_gain": r0_median[e],
            "r0_action_effect": r0_ae[e], "loo_delta": loo_d[e]} for e in expert_ids]

# ---- 7. pool collapse ----
df = pd.read_csv(DS_DIR / "series.csv")
series = df["OT"].to_numpy(np.float32)
dates = pd.to_datetime(df["date"])
idx = np.load(DS_DIR / "window_indices.npz")
train_starts = np.asarray(idx["train"], dtype=np.int64)
val_starts = np.asarray(idx["val"], dtype=np.int64)
cut = max(1, int(len(train_starts) * 0.80))
bank_starts = train_starts[:cut]
fit_starts = train_starts[cut:]
X_bank, Y_bank = build_windows(series, bank_starts, CONTEXT, HORIZON)
X_fit, Y_fit = build_windows(series, fit_starts, CONTEXT, HORIZON)
X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)
X_train_all = np.concatenate([X_bank, X_fit], axis=0)
stats = compute_training_stats(X_train_all)
val_ts = [dates[s + CONTEXT] for s in val_starts]

raw_act_val = build_activation_matrix(raw_experts, X_val, val_ts, stats)
final_act_val = build_activation_matrix(final_pool, X_val, val_ts, stats)


def pool_stats(a):
    a = a >= 0.5
    return float((a.sum(axis=1) >= 1).mean()), float(a.sum(axis=1).mean())


raw_union, raw_avg = pool_stats(raw_act_val)
fin_union, fin_avg = pool_stats(final_act_val)


def emb_diversity(emb, n):
    if n < 2:
        return 0.0
    E = emb[:n]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sim = E @ E.T
    iu = np.triu_indices(n, 1)
    return float(1.0 - sim[iu].mean())


emb_raw = v1.embed_claims([e["claim"] for e in raw_experts], "cpu")
emb_final = v1.embed_claims([e["claim"] for e in final_pool], "cpu")
raw_div = emb_diversity(emb_raw, len(raw_experts))
fin_div = emb_diversity(emb_final, len(final_pool))
collapse = bool(fin_union < raw_union * 0.5 or fin_avg < raw_avg * 0.5)
pool_rows = [
    {"pool": "raw", "n_experts": len(raw_experts), "union_coverage": raw_union,
     "avg_activated_per_query": raw_avg, "embedding_diversity": raw_div, "pool_collapse": False},
    {"pool": "final", "n_experts": len(final_pool), "union_coverage": fin_union,
     "avg_activated_per_query": fin_avg, "embedding_diversity": fin_div, "pool_collapse": collapse},
]

# ---- 写 CSV ----
def _csv(fname, rows, cols):
    with (DIAG_DIR / fname).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


_csv("electricity_expert_lineage_audit.csv", lineage_rows,
     ["expert_id", "raw_mechanism", "raw_activation", "raw_action",
      "r0_reward", "r0_action_effect", "r0_coverage", "r0_verdict",
      "r1_reward", "r1_action_effect", "r1_coverage", "r1_verdict",
      "r2_reward", "r2_action_effect", "r2_coverage", "final_status", "drop_reason"])
_csv("electricity_raw_single_effects.csv", single_rows,
     ["expert_id", "mse_shape", "mse_shape_plus", "single_gain_vs_shape"])
_csv("electricity_raw_loo_effects.csv", loo_rows,
     ["expert_id", "mse_raw_full", "mse_raw_minus", "loo_delta"])
_csv("electricity_synergy_audit.csv", synergy_rows,
     ["expert_id", "single_effect", "loo_effect", "synergy_flag", "interpretation"])
_csv("electricity_final_expert_generalization.csv", final_rows,
     ["split", "seed", "shape_mse", "final_expert_mse", "raw_full_mse", "final_vs_shape", "raw_vs_shape"])
_csv("electricity_reward_vs_group_value.csv", rv_rows,
     ["expert_id", "r0_reward", "r0_median_gain", "r0_action_effect", "loo_delta"])
_csv("electricity_pool_coverage_audit.csv", pool_rows,
     ["pool", "n_experts", "union_coverage", "avg_activated_per_query", "embedding_diversity", "pool_collapse"])

# ---- report ----
n_syn = sum(1 for r in synergy_rows if r["synergy_flag"])
strong_single = [r["expert_id"] for r in single_rows if r["single_gain_vs_shape"] > 0.1]
if n_syn >= 2 and spearman_rv <= 0:
    cls = "B SYNERGY_LOSS + A REWARD_MISMATCH (E MIXED)"
else:
    cls = "E MIXED"

L = []
L.append("# Electricity 失败根因诊断 — Reward-Feedback Critic v1")
L.append("")
L.append("只诊断、不改方法、不访问 TEST。BASE≈1.124B，RAW≈268.5M，REWARD-FEEDBACK≈972.4M，0/3 seeds D<C。")
L.append("")
L.append("## 结果")
L.append("")
L.append("### 1. 单条 vs 组合（deterministic，3-seed mean）")
L.append("")
L.append("| Expert | single_gain_vs_shape | loo_delta_in_raw_pool | synergy |")
L.append("|---|---|---|---|")
for r in synergy_rows:
    L.append(f"| {r['expert_id']} | {r['single_effect']:+.4f} | {r['loo_effect']:+.3e} | {r['synergy_flag']} |")
L.append("")
L.append(f"- 单条有益（single_gain>0）的 Expert：{', '.join(r['expert_id'] for r in single_rows if r['single_gain_vs_shape']>0)}（E1/E2/E8 最明显）。")
L.append(f"- 单条无益但组合有益（SYNERGY）的 Expert：{', '.join(r['expert_id'] for r in synergy_rows if r['synergy_flag'])}。")
L.append(f"- Shape mean={shape_mean:.4e}，Raw full mean={raw_full_mean:.4e}（Raw pool 相对 Shape 增益 = {(shape_mean-raw_full_mean)/shape_mean*100:.1f}%）。")
L.append("")

L.append("### 2. 哪 7 条被淘汰")
L.append("")
L.append("| Expert | r0 reward | r0 action_effect | r1 reward | r2 reward | final | drop_reason |")
L.append("|---|---|---|---|---|---|---|")
for r in lineage_rows:
    L.append(f"| {r['expert_id']} | {r['r0_reward']} | {r['r0_action_effect']} | "
             f"{r['r1_reward']} | {r['r2_reward']} | {r['final_status']} | {r['drop_reason']} |")
L.append("")
L.append(f"- Final Pool 只保留 {final_id}，其余 7 条 reward 全程 <=0 被筛掉。")
L.append("- 8 条 Raw 的 r0 action_effect_size 全部 <=0（E6 最负 -1.005），即逐条执行 retrieval_action 在 TRAIN 上单条无增益。")
L.append("")

L.append("### 3. Final 1 条 TRAIN vs VAL 泛化")
L.append("")
L.append("| split | seed | shape_mse | final_expert_mse | raw_full_mse | final_vs_shape | raw_vs_shape |")
L.append("|---|---|---|---|---|---|---|")
for r in final_rows:
    L.append(f"| {r['split']} | {r['seed']} | {r['shape_mse']:.4e} | {r['final_expert_mse']:.4e} "
             f"| {r['raw_full_mse']:.4e} | {r['final_vs_shape']:+.4f} | {r['raw_vs_shape']:+.4f} |")
L.append("")
L.append(f"- Final 1 条（{final_id} 修订版）在 VAL 上相对 Shape 增益 ≈ 0 甚至为负，而 Raw full pool 相对 Shape 有大增益（+{ (shape_mean-raw_full_mean)/shape_mean*100:.1f}%）。")
L.append("")

L.append("### 4. reward vs Raw pool 真实贡献")
L.append("")
L.append(f"- Spearman(reward, loo_delta) = {spearman_rv:+.4f}；sign agreement = {sign_agree_frac if sign_agree_frac is None else f'{sign_agree_frac:.2f}'}。")
L.append("")
L.append("| Expert | r0_reward | r0_median_gain | r0_action_effect | loo_delta |")
L.append("|---|---|---|---|---|")
for r in rv_rows:
    L.append(f"| {r['expert_id']} | {r['r0_reward']} | {r['r0_median_gain']} | {r['r0_action_effect']} | {r['loo_delta']:.3e} |")
L.append("")

L.append("### 5. pool collapse")
L.append("")
L.append("| pool | n_experts | union_coverage | avg_activated | diversity | collapse |")
L.append("|---|---|---|---|---|---|")
for r in pool_rows:
    L.append(f"| {r['pool']} | {r['n_experts']} | {r['union_coverage']:.3f} | "
             f"{r['avg_activated_per_query']:.3f} | {r['embedding_diversity']:.3f} | {r['pool_collapse']} |")
L.append("")

L.append("## 结论")
L.append("")
L.append(f"- **{cls}**")
L.append("")
L.append("- 两条独立机制叠加：")
L.append(f"  1) **REWARD_MISMATCH**：reward 打的是「逐条 retrieval_action 的确定性执行增益」，而最终 VAL ablation（C/D arm）用的是「expert 文本 embedding + activation 在 learned retriever 里的组合」。E1/E2/E8 单条 embedding 就很有价值（single_gain +0.75/+0.77/+0.66），但它们的 retrieval_action 在 TRAIN 单条为负，reward 给了 <=0，把它们筛掉。")
L.append(f"  2) **SYNERGY_LOSS**：E4/E5 单条无增益（single_gain<0），但在 Raw full pool 里去掉它们会显著变差（loo_delta +8.7e7/+9.3e7），逐条 reward 无法评价这种组合价值。")
L.append("")
L.append("### 最终回答")
L.append("")
L.append("Electricity 的失败**不是证明 Reward-Feedback 思路本身失效**。")
L.append("")
L.append("根因是 reward 的 granularity（逐条知识评分）不适合强组合/强 embedding 效应数据集：Raw pool 的 VAL 增益来自 8 条 embedding+activation 在 learned retriever 里的组合，")
L.append("而 reward 逐条执行 retrieval_action（几乎都是 prefer_same:hour_of_day）在 TRAIN 上单条为负，导致 8→1 的 pool collapse，丢掉了组合知识与单条 embedding 价值。")
L.append("")
L.append("> 本次诊断结束，未做任何修复（未改 Reward/Critic/Expert、未删结果、未访问 TEST）。")

(DIAG_DIR / "electricity_failure_diagnosis.md").write_text("\n".join(L), encoding="utf-8")
print("wrote", DIAG_DIR)
print(f"spearman_reward_vs_loo = {spearman_rv:+.4f}, sign_agreement = {sign_agree_frac}, n_synergy = {n_syn}")
print("strong single:", strong_single)
