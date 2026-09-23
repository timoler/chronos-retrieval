#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper completion — offline aggregation (P1/P3/P4/P7).

只读取已有 results（冻结 v2 final TEST / full VAL / v1 reward / v2 pool reward /
failure diagnosis），不重跑任何实验、不访问新 TEST、不改 v2。

产出（results/paper_completion/）：
  P1  paper_main_ablation.csv + main_ablation.md
  P3  reward_mechanism_comparison.csv + reward_mechanism_summary.md
  P4  paper_statistics.csv + statistics_report.md
  P7  paper_pool_analysis.csv + expert_pool_structure.csv + pool_coverage_report.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/paper_completion"
OUT.mkdir(parents=True, exist_ok=True)

TEST_ABL = REPO / "results/reward_feedback_critic_v2_final_test/final_test_ablation.csv"
TEST_SUM = REPO / "results/reward_feedback_critic_v2_final_test/final_test_summary.csv"
VAL_ABL = REPO / "results/reward_feedback_critic_v2_full_cross_dataset/full_cross_dataset_ablation.csv"
VAL_SUM = REPO / "results/reward_feedback_critic_v2_full_cross_dataset/full_cross_dataset_summary.csv"
POOL_STATS = REPO / "results/reward_feedback_critic_v2_full_cross_dataset/full_cross_dataset_pool_stats.csv"
V2_POOL_ROOT = REPO / "results/reward_feedback_critic_v2_full_cross_dataset"
V1_CROSS = REPO / "results/reward_feedback_critic_v1_cross_dataset"
V1_ETTH1 = REPO / "results/reward_feedback_critic_v1_etth1"
V1_POOL_STATS = V1_CROSS / "cross_dataset_expert_pool_stats.csv"
REW_ALIGN = REPO / "results/reward_feedback_critic_v2_failure_diagnosis/reward_alignment_train_val.csv"
FAIL_SUM = REPO / "results/reward_feedback_critic_v2_failure_diagnosis/failure_diagnosis_summary.csv"
POOL_COMP = REPO / "results/reward_feedback_critic_v2_failure_diagnosis/pool_comparison.csv"

DATASETS = ["etth1", "etth2", "weather", "exchange_rate", "solar", "electricity", "traffic"]
SEEDS = [42, 123, 2026]
ARM_MAP = {"base": "BASE", "shape": "SHAPE", "raw": "RAW_EXPERT",
           "v1": "REWARD_V1", "v2": "POOL_AWARE_V2"}
ARMS = ["BASE", "SHAPE", "RAW_EXPERT", "REWARD_V1", "POOL_AWARE_V2"]


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def num(x, default=None):
    if x is None or x == "" or x == "nan":
        return default
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# P1 主消融
# --------------------------------------------------------------------------- #
def build_main_ablation():
    test_rows = read_csv(TEST_ABL)
    val_rows = read_csv(VAL_ABL)
    pool_by_ds = {r["dataset"]: r for r in read_csv(POOL_STATS)}

    def pool_size(ds, method):
        if method in ("BASE", "SHAPE"):
            return ""
        if method == "RAW_EXPERT":
            return pool_by_ds.get(ds, {}).get("raw_pool_size", "8")
        if method == "REWARD_V1":
            return pool_by_ds.get(ds, {}).get("v1_pool_size", "")
        return pool_by_ds.get(ds, {}).get("v2_pool_size", "")

    # TEST 主表（mse + mae）
    out = []
    for ds in DATASETS:
        for s in SEEDS:
            for method in ARMS:
                arm = {v: k for k, v in ARM_MAP.items()}[method]
                row = next((r for r in test_rows
                            if r["dataset"] == ds and r["arm"] == arm and int(r["seed"]) == s), None)
                if row is None:
                    continue
                out.append({
                    "dataset": ds, "seed": s, "method": method,
                    "mse": num(row["test_mse"]), "mae": num(row["test_mae"]),
                    "pool_size": pool_size(ds, method), "coverage": "",
                    "runtime_sec": "", "split": "test",
                })

    with open(OUT / "paper_main_ablation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "method", "mse", "mae",
                                          "pool_size", "coverage", "runtime_sec", "split"])
        w.writeheader()
        for r in out:
            w.writerow(r)

    # 汇总（TEST）
    summary = {}
    for ds in DATASETS:
        summary[ds] = {}
        for method in ARMS:
            ms = [r["mse"] for r in out if r["dataset"] == ds and r["method"] == method]
            summary[ds][method] = {
                "mean": float(np.mean(ms)) if ms else float("nan"),
                "std": float(np.std(ms)) if ms else float("nan"),
            }

    def rel(a, b):
        if not (a > 0) or np.isnan(b):
            return float("nan")
        return (a - b) / a * 100.0

    L = ["# P1 主消融表（A/B/C/D/E，TEST，frozen v2）", ""]
    L.append("方法：BASE=CNN-only，SHAPE=+shape 先验，RAW_EXPERT=+raw 检索专家，"
             "REWARD_V1=+reward-feedback v1，POOL_AWARE_V2=+pool-aware v2。")
    L.append("seeds=[42,123,2026]，正式数据集 7 个。指标为 TEST MSE（3-seed mean）。")
    L.append("coverage / runtime 未在已有 artifact 中，留空（不填 0）。")
    L.append("")
    L.append("| Dataset | BASE | SHAPE | RAW | v1 | v2 | v2 vs BASE | v2 vs RAW | v2 vs v1 | wins raw | wins v1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        s = summary[ds]
        v2b = rel(s["BASE"]["mean"], s["POOL_AWARE_V2"]["mean"])
        v2r = rel(s["RAW_EXPERT"]["mean"], s["POOL_AWARE_V2"]["mean"])
        v2v = rel(s["REWARD_V1"]["mean"], s["POOL_AWARE_V2"]["mean"])
        wins_raw = sum(1 for r in out if r["dataset"] == ds and r["method"] == "POOL_AWARE_V2"
                       and r["mse"] < next(x["mse"] for x in out if x["dataset"] == ds
                                           and x["method"] == "RAW_EXPERT" and x["seed"] == r["seed"]))
        wins_v1 = sum(1 for r in out if r["dataset"] == ds and r["method"] == "POOL_AWARE_V2"
                      and r["mse"] < next(x["mse"] for x in out if x["dataset"] == ds
                                          and x["method"] == "REWARD_V1" and x["seed"] == r["seed"]))
        L.append(f"| {ds} | {s['BASE']['mean']:.6g} | {s['SHAPE']['mean']:.6g} "
                 f"| {s['RAW_EXPERT']['mean']:.6g} | {s['REWARD_V1']['mean']:.6g} "
                 f"| {s['POOL_AWARE_V2']['mean']:.6g} | {v2b:+.1f}% | {v2r:+.1f}% "
                 f"| {v2v:+.1f}% | {wins_raw}/3 | {wins_v1}/3 |")
    L.append("")
    L.append("## 结果")
    ri_raw = [rel(summary[ds]["RAW_EXPERT"]["mean"], summary[ds]["POOL_AWARE_V2"]["mean"]) for ds in DATASETS]
    ri_v1 = [rel(summary[ds]["REWARD_V1"]["mean"], summary[ds]["POOL_AWARE_V2"]["mean"]) for ds in DATASETS]
    better_raw = sum(1 for x in ri_raw if x > 0)
    better_v1 = sum(1 for x in ri_v1 if x > 0)
    L.append(f"- v2 vs Raw：{better_raw}/7 数据集胜，mean RI = {np.mean(ri_raw):+.2f}%，median RI = {np.median(ri_raw):+.2f}%")
    L.append(f"- v2 vs v1：{better_v1}/7 数据集胜，mean RI = {np.mean(ri_v1):+.2f}%，median RI = {np.median(ri_v1):+.2f}%")
    L.append("- 失败数据集（v2 不优于 Raw）：electricity（-15.5%）、traffic（-0.4%）。")
    L.append("")
    L.append("## 结论")
    L.append("- 主消融表直接复用 frozen v2 final TEST（`final_test_ablation.csv`），未重跑。")
    L.append("- coverage/runtime 缺失：见 P5（效率）与 P7（pool/coverage）分别补齐。")
    with open(OUT / "main_ablation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    return out, summary


# --------------------------------------------------------------------------- #
# P3 Reward 机制对照
# --------------------------------------------------------------------------- #
def build_reward_mechanism():
    # v1 individual reward（final-round reward）
    def v1_hist(ds):
        src = (V1_ETTH1 / "reward_history.csv") if ds == "etth1" else (V1_CROSS / ds / "reward_history.csv")
        if not src.exists():
            return {}
        best = {}
        for r in read_csv(src):
            lid = r["lineage_id"]
            if lid not in best or int(r["round"]) >= int(best[lid]["round"]):
                best[lid] = r
        return best

    def v1_pool_ids(ds):
        src = (V1_ETTH1 / "final_expert_pool.json") if ds == "etth1" else (V1_CROSS / ds / "final_expert_pool.json")
        if not src.exists():
            return set()
        d = json.load(open(src, encoding="utf-8"))
        if isinstance(d, dict):
            d = list(d.values())
        return {e.get("id") for e in d}

    def v2_round0(ds):
        src = V2_POOL_ROOT / ds / "pool_reward_history.csv"
        r0 = {}
        for r in read_csv(src):
            if r["round"] == "0":
                r0[r["expert_id"]] = r
        return r0

    def v2_pool_ids(ds):
        src = V2_POOL_ROOT / ds / "final_expert_pool.json"
        d = json.load(open(src, encoding="utf-8"))
        if isinstance(d, dict):
            d = list(d.values())
        return {e.get("id") for e in d}

    # val_contribution（已有 reward_alignment，只覆盖 4 数据集）
    val_contrib = {}
    for r in read_csv(REW_ALIGN):
        if r["level"] == "expert":
            val_contrib[(r["dataset"], r["expert_id"])] = num(r["val_contribution"])

    rows = []
    for ds in DATASETS:
        v1h = v1_hist(ds)
        v2r0 = v2_round0(ds)
        v1p = v1_pool_ids(ds)
        v2p = v2_pool_ids(ds)
        for i in range(1, 9):
            eid = f"E{i}"
            v1r = v1h.get(eid, {})
            v2r = v2r0.get(eid, {})
            rows.append({
                "dataset": ds, "seed": 42, "expert_id": eid,
                "individual_reward_v1": num(v1r.get("reward")),
                "individual_action_effect_v1": num(v1r.get("action_effect_size")),
                "pool_reward_v2": num(v2r.get("mean_pool_reward")),
                "median_pool_reward_v2": num(v2r.get("median_pool_reward")),
                "selected_v1": int(eid in v1p),
                "selected_v2": int(eid in v2p),
                "val_contribution_raw_pool": val_contrib.get((ds, eid)),
            })

    with open(OUT / "reward_mechanism_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "seed", "expert_id",
                                          "individual_reward_v1", "individual_action_effect_v1",
                                          "pool_reward_v2", "median_pool_reward_v2",
                                          "selected_v1", "selected_v2", "val_contribution_raw_pool"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # dataset-level：train pool_reward vs val_contribution 对齐（已有 reward_alignment 的 dataset 行）
    align = {}
    for r in read_csv(REW_ALIGN):
        if r["level"] == "dataset":
            align[r["dataset"]] = r

    L = ["# P3 Reward 机制对照（individual v1 vs pool-aware v2）", ""]
    L.append("individual_reward_v1 = v1 逐条孤立 reward（final round 的离散 reward -2..2）；")
    L.append("pool_reward_v2 = v2 的 TRAIN-only 边际贡献 reward（round-0 mean）。")
    L.append("val_contribution = (MSE[raw∖Ei]-MSE[raw])/MSE[raw∖Ei]（seed=42，仅 4 数据集已有）。")
    L.append("")
    L.append("## train pool_reward(v2) 与 VAL contribution 对齐（已有 `reward_alignment_train_val.csv`）")
    L.append("")
    L.append("| Dataset | Spearman | sign-agreement | top3 overlap | top4 overlap | 解读 |")
    L.append("|---|---|---|---|---|---|")
    for ds in ["etth1", "solar", "electricity", "traffic"]:
        a = align.get(ds, {})
        L.append(f"| {ds} | {a.get('spearman_rho', '—')} | {a.get('sign_agreement_ratio', '—')} "
                 f"| {a.get('top3_overlap', '—')} | {a.get('top4_overlap', '—')} | "
                 f"{'对齐' if num(a.get('spearman_rho'), 0) > 0 else '未对齐/负'} |")
    L.append("")
    L.append("> etth2 / weather / exchange_rate 无 val_contribution artifact，"
             "train pool_reward 与 val contribution 的 Spearman 无法在已有结果上计算（见「仍缺实验」）。")
    L.append("")
    L.append("## 结果")
    L.append("- v2 对齐并非全面更强：etth1 正对齐（Spearman=+0.286），solar/traffic 负对齐"
             "（-0.810 / -0.548），electricity 接近 0（-0.190）。")
    L.append("- 说明 pool_reward 在 TRAIN 与 VAL contribution 之间并不总是方向一致；"
             "v2 的正收益不能简单解释为「reward 更准」。")
    L.append("")
    L.append("## 结论")
    L.append("- 如实报告：v2 reward 对齐在 4 个可算数据集上 1 正 / 2 负 / 1 近 0，未发现全面更强的对齐。")
    L.append("- 不因此修改 reward（冻结 v2）。")
    with open(OUT / "reward_mechanism_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return rows


# --------------------------------------------------------------------------- #
# P4 统计可靠性（TEST，3-seed；不重跑 TEST）
# --------------------------------------------------------------------------- #
def build_statistics():
    test_rows = read_csv(TEST_ABL)
    L = ["# P4 统计可靠性（frozen TEST，只读已有结果）", ""]
    L.append("数据粒度：正式 TEST 只有 3-seed aggregate（每数据集每臂 3 个 seed MSE），"
             "无逐窗口 TEST prediction artifact（除 electricity/traffic 的 failure diagnosis）。")
    L.append("因此按协议：n=3，只报 mean±std、seed wins、relative improvement，"
             "不声称 seed-level significance。")
    L.append("")

    out = []
    L.append("| Dataset | raw mean±std | v2 mean±std | v1 mean±std | RI vs raw | RI vs v1 | wins raw | wins v1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        def get(arm):
            return [num(r["test_mse"]) for r in test_rows
                    if r["dataset"] == ds and r["arm"] == arm]
        raw = get("raw"); v1 = get("v1"); v2 = get("v2")
        raw_m, raw_s = float(np.mean(raw)), float(np.std(raw))
        v1_m, v1_s = float(np.mean(v1)), float(np.std(v1))
        v2_m, v2_s = float(np.mean(v2)), float(np.std(v2))
        ri_raw = (raw_m - v2_m) / raw_m * 100
        ri_v1 = (v1_m - v2_m) / v1_m * 100
        wins_raw = sum(1 for a, b in zip(v2, raw) if a < b)
        wins_v1 = sum(1 for a, b in zip(v2, v1) if a < b)
        out.append({"dataset": ds, "comparison": "v2_vs_raw", "ri_pct": ri_raw,
                    "wins": wins_raw, "n_seeds": 3, "mean_mse_a": raw_m, "std_mse_a": raw_s,
                    "mean_mse_b": v2_m, "std_mse_b": v2_s,
                    "effect": (raw_m - v2_m), "ci_low": "", "ci_high": "",
                    "significance": "n=3, no seed-level significance claimed"})
        out.append({"dataset": ds, "comparison": "v2_vs_v1", "ri_pct": ri_v1,
                    "wins": wins_v1, "n_seeds": 3, "mean_mse_a": v1_m, "std_mse_a": v1_s,
                    "mean_mse_b": v2_m, "std_mse_b": v2_s,
                    "effect": (v1_m - v2_m), "ci_low": "", "ci_high": "",
                    "significance": "n=3, no seed-level significance claimed"})
        L.append(f"| {ds} | {raw_m:.6g}±{raw_s:.6g} | {v2_m:.6g}±{v2_s:.6g} "
                 f"| {v1_m:.6g}±{v1_s:.6g} | {ri_raw:+.1f}% | {ri_v1:+.1f}% "
                 f"| {wins_raw}/3 | {wins_v1}/3 |")

    # 逐窗口 bootstrap（仅 electricity/traffic，已有 failure diagnosis 结论，标记 REUSE）
    fail = {r["dataset"]: r for r in read_csv(FAIL_SUM)}
    L.append("")
    L.append("## 逐窗口 paired block bootstrap（仅 electricity / traffic，REUSE failure diagnosis）")
    L.append("")
    L.append("- traffic：pooled rel_diff = +0.42%（|rel_diff|<1%），pooled bootstrap 95% CI 跨 0；"
             "seed42 显著更差（p≈1e-13）、seed123 显著更好（p=0.001）、seed2026 不显著 → COMPARABLE。")
    L.append("- electricity：总体 -15.5%，幅度大、1/3 seeds 赢 Raw，不能归因于纯随机噪声"
             "（真实退化 + 部分 seed 波动）。")
    for ds in ("electricity", "traffic"):
        r = fail.get(ds, {})
        if r:
            L.append(f"- {ds} primary label: {r.get('primary_label', '')} | "
                     f"evidence: {r.get('evidence', '')[:200]}")

    L.append("")
    L.append("## 结果")
    L.append("- 跨数据集（n=7）v2 vs Raw：5/7 胜，mean RI +13.40%，median RI +17.71%。")
    L.append("- 跨数据集（n=7）v2 vs v1：6/7 胜，mean RI +17.77%，median RI +18.34%。")
    L.append("- 因每数据集仅 3 seeds，不计算跨 seed 显著性；不声称 p<0.05。")
    L.append("")
    L.append("## 结论")
    L.append("- 统计证据为「描述性 + 方向一致性（VAL→TEST 7/7）」级别；")
    L.append("- 逐窗口 bootstrap 仅对 electricity/traffic 可做（已有），其余 5 数据集无逐窗口 TEST artifact。")
    with open(OUT / "statistics_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    with open(OUT / "paper_statistics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "comparison", "ri_pct", "wins", "n_seeds",
                                          "mean_mse_a", "std_mse_a", "mean_mse_b", "std_mse_b",
                                          "effect", "ci_low", "ci_high", "significance"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out


# --------------------------------------------------------------------------- #
# P7 Expert Pool / Coverage
# --------------------------------------------------------------------------- #
def build_pool_analysis():
    pool = {r["dataset"]: r for r in read_csv(POOL_STATS)}
    v1stats = {r["dataset"]: r for r in read_csv(V1_POOL_STATS)}
    # coverage（union coverage raw/v1/v2，仅 failure diagnosis 覆盖 electricity/traffic）
    cov = {}
    for r in read_csv(FAIL_SUM):
        pass  # 文本形式，手填下面
    # 手填已知 union coverage（来自 failure_diagnosis_report.md）
    cov = {
        "electricity": {"raw": 1.000, "v2": 0.692},
        "traffic": {"raw": 1.000, "v2": 0.974},
    }

    rows = []
    for ds in DATASETS:
        p = pool[ds]
        raw = int(p["raw_pool_size"]); v1 = int(p["v1_pool_size"]); v2 = int(p["v2_pool_size"])
        c = cov.get(ds, {})
        rows.append({
            "dataset": ds,
            "raw_pool_size": raw, "v1_pool_size": v1, "v2_pool_size": v2,
            "union_coverage_raw": c.get("raw", ""),
            "union_coverage_v1": c.get("v1", ""),
            "union_coverage_v2": c.get("v2", ""),
            "pool_retention_ratio": round(v2 / raw, 4) if raw else "",
            "coverage_retention_ratio": round(c.get("v2", 0) / c["raw"], 4) if c.get("raw") else "",
            "v1_pool_collapse": v1 <= 1,
        })

    with open(OUT / "paper_pool_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # expert_pool_structure.csv（同内容，别名）
    with open(OUT / "expert_pool_structure.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    L = ["# P7 Expert Pool / Coverage 分析", ""]
    L.append("| Dataset | raw | v1 | v2 | pool retention | coverage raw→v2 | v1 collapse |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['dataset']} | {r['raw_pool_size']} | {r['v1_pool_size']} | {r['v2_pool_size']} "
                 f"| {r['pool_retention_ratio']} | "
                 f"{r['coverage_retention_ratio'] if r['coverage_retention_ratio'] != '' else '—'} | "
                 f"{'是' if r['v1_pool_collapse'] else '否'} |")
    L.append("")
    L.append("## 结果")
    L.append("- **Electricity v1 collapse**：v1 pool 8→1（唯一 KEEP=E6），v2 恢复到 5。")
    L.append("- **Electricity coverage loss**：union coverage 1.000→0.692（30.8% TEST 窗口无激活 expert），"
             "v2 删除了 raw 覆盖率最高的 E2(82.7%) 与 E8(58.6%)，属过度剪枝。")
    L.append("- **Traffic**：coverage 1.000→0.974（接近无损），相对差异 <1%，comparable 而非明确失败。")
    L.append("- 其余 5 数据集：raw 恒为 8，v2 保留 5-7，无 v1 塌缩现象。")
    L.append("")
    L.append("## 结论")
    L.append("- v2 相比 v1 的主要机制收益 = 避免 pool collapse（尤其 electricity），")
    L.append("  但 v2 在 electricity 上仍因 coverage loss 输给 Raw，说明 pool-aware reward 并不保证覆盖保持。")
    with open(OUT / "pool_coverage_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return rows


if __name__ == "__main__":
    build_main_ablation()
    build_reward_mechanism()
    build_statistics()
    build_pool_analysis()
    print("P1/P3/P4/P7 offline outputs written to", OUT)
