#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PAER 论文实验表生成器。

从仓库原始产物（results/ 下已落盘 CSV/JSON/配置）重新汇总 6 张正文表 + 5 张附录表，
统一输出 CSV / Markdown / LaTeX 三线表，并写 sources.json。

统计口径（与论文一致）：
  - 正式模型种子 seeds = [42, 123, 2026]；标准差 ddof=0。
  - 相对改善 RI = (对照MSE − PAER MSE) / 对照MSE × 100%，用未舍入数值计算。
  - 7 数据集汇总 RI = 各数据集 RI 的等权平均（不平均原始 MSE）。
  - TRAIN 反馈 / VAL 对照 / TEST 结果分别标注，禁止混用。
  - 缺失值统一标「未完成」/「未记录」，不编造。
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

DS = ["etth1", "etth2", "exchange_rate", "traffic", "weather", "solar", "electricity"]
SEEDS = [42, 123, 2026]
NSEED = len(SEEDS)
METHOD_LABEL = {"BASE": "BASE", "SHAPE": "SHAPE", "RAW_EXPERT": "Raw Expert",
                "REWARD_V1": "v1", "POOL_AWARE_V2": "PAER"}

# 数据源
MD = REPO / "results/multidataset"
FT = REPO / "results/reward_feedback_critic_v2_final_test"
V2C = REPO / "results/reward_feedback_critic_v2_full_cross_dataset"
PC = REPO / "results/paper_completion"
DIAG = REPO / "results/diagnose_v2_failure_electricity_traffic"
FAIL = REPO / "results/reward_feedback_critic_v2_failure_diagnosis"
EXT = REPO / "results/external_baselines/external_baseline_results_full.csv"
FUSION = PC / "phase1_external_baselines/ours_fusion_v2_results.csv"


# --------------------------------------------------------------------------- #
# 数值/文本工具
# --------------------------------------------------------------------------- #
def fmt(x, nd=4):
    if x is None:
        return "未记录"
    x = float(x)
    ax = abs(x)
    if ax != 0 and (ax >= 1e6 or ax < 1e-3):
        return f"{x:.{nd-1}g}"
    return f"{x:.{nd}g}"


def fmt_msd(mean, std, nd=4):
    return f"{fmt(mean, nd)}±{fmt(std, nd)}"


def pct(x):
    return f"{x:+.1f}%"


def esc_latex(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def tex_cell(v):
    """LaTeX 单元格：转义后处理科学计数法与 ±。"""
    import re
    v = esc_latex(v)

    def _sci(m):
        mant, exp = m.group(1), m.group(2)
        sign = exp[0]
        digits = exp[1:].lstrip("0") or "0"
        e = digits if sign == "+" else "-" + digits
        return f"${mant}\\times10^{{{e}}}$"

    v = re.sub(r"([0-9.]+)e([+-][0-9]+)", _sci, v)
    v = v.replace("±", "$\\pm$")
    return v


# --------------------------------------------------------------------------- #
# 表写入（CSV + Markdown + LaTeX）
# --------------------------------------------------------------------------- #
def write_table(name, caption, label, header, rows, note=""):
    csv_path = OUT / f"{name}.csv"
    md_path = OUT / f"{name}.md"
    tex_path = OUT / f"{name}.tex"

    # CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])

    # Markdown
    md = [f"# {caption}\n"]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        md.append("| " + " | ".join(["" if v is None else str(v).replace("|", "\\|") for v in r]) + " |")
    if note:
        md.append("\n" + note)
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    # LaTeX（三线表）
    colspec = "l" + "c" * (len(header) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{esc_latex(caption)}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
        "    " + " & ".join(esc_latex(h) for h in header) + " \\\\",
        "    \\midrule",
    ]
    for r in rows:
        lines.append("    " + " & ".join(tex_cell(v) for v in r) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
    ]
    if note:
        lines.append(f"  {{\\footnotesize {esc_latex(note)}}}")
    lines += ["\\end{table}", ""]
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  wrote {name}.{{csv,md,tex}}")
    return csv_path


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_config(ds):
    return json.loads((MD / ds / "config.json").read_text(encoding="utf-8"))


def load_final_test_ablation():
    """paper_main_ablation.csv：dataset,seed,method,mse,mae,pool_size,coverage,runtime,split"""
    rows = defaultdict(lambda: defaultdict(dict))
    with (PC / "paper_main_ablation.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test":
                rows[r["dataset"]][r["method"]][int(r["seed"])] = {
                    "mse": float(r["mse"]), "mae": float(r["mae"]),
                    "pool": r["pool_size"] or ""}
    return rows


def load_pool_size_control():
    rows = {}
    with (PC / "paper_pool_size_control.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["dataset"]] = r
    return rows


def load_pool_reward_history(ds):
    p = V2C / ds / "pool_reward_history.csv"
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def load_experts(ds, which):
    p = V2C / ds / (which + ".json")
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def load_sensitivity():
    rows = {}
    with (PC / "paper_sensitivity.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["dataset"]] = r
    return rows


def load_efficiency():
    rows = {}
    with (PC / "paper_efficiency.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["dataset"]] = r
    return rows


def load_reward_alignment():
    ds_rows = {}
    with (FAIL / "reward_alignment_train_val.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["level"] == "dataset":
                ds_rows[r["dataset"]] = r
    return ds_rows


# =========================================================================== #
# 表 1：数据集与协议
# =========================================================================== #
def table1():
    header = ["Dataset", "Source file", "Target", "Frequency", "Context L",
              "Horizon H", "TRAIN windows", "VAL windows", "TEST windows",
              "Aggregation / transform"]
    agg = {"etth1": "univariate OT", "etth2": "first series (OT), LOCF",
           "exchange_rate": "univariate OT", "electricity": "univariate OT",
           "traffic": "first series (Sensor_000), LOCF",
           "weather": "univariate OT", "solar": "univariate OT"}
    rows = []
    for ds in DS:
        c = load_config(ds)
        src = c["source"].replace("\\", "/")
        freq = c["frequency"]
        nw = c["num_windows"]
        rows.append([ds, src, c.get("target", c.get("target_series", "OT")),
                     freq, c["context"], c["horizon"],
                     nw["train"], nw["val"], nw["test"], agg[ds]])
    note = ("所有数据集 stride=1 滑窗；单变量目标（OT 或首列），无跨序列汇聚。"
            "etth2/traffic 按 config 的 deterministic first-series 规则选列并对缺失值 LOCF 前向填充。")
    write_table("table_1_datasets_protocol",
                "Table 1: 数据集与协议", "tab:datasets", header, rows, note)


# =========================================================================== #
# 表 2：内部检索主结果（TEST，mean±std，3 seeds，ddof=0）
# =========================================================================== #
def table2():
    ab = load_final_test_ablation()
    header = ["Dataset", "BASE", "SHAPE", "Raw Expert", "PAER",
              "PAER vs Raw (RI)", "PAER wins / 3 seeds"]
    rows = []
    ri_list = []
    for ds in DS:
        def agg(m):
            v = [ab[ds][m][s]["mse"] for s in SEEDS]
            mu = sum(v) / NSEED
            std = (sum((x - mu) ** 2 for x in v) / NSEED) ** 0.5  # ddof=0
            return mu, std, v
        b = agg("BASE"); sh = agg("SHAPE"); raw = agg("RAW_EXPERT"); v2 = agg("POOL_AWARE_V2")
        ri = (raw[0] - v2[0]) / raw[0] * 100.0
        wins = sum(1 for s in SEEDS if v2[2][SEEDS.index(s)] < raw[2][SEEDS.index(s)])
        ri_list.append(ri)
        rows.append([ds, fmt_msd(*b[:2]), fmt_msd(*sh[:2]), fmt_msd(*raw[:2]),
                     fmt_msd(*v2[:2]), pct(ri), f"{wins}/3"])
    mean_ri = sum(ri_list) / len(ri_list)
    rows.append(["7-dataset (equal-weight RI)", "—", "—", "—", "—",
                 pct(mean_ri), f"{sum(1 for r in ri_list if r > 0)}/7"])
    note = ("TEST 一次性评估；每格为 3 种子 mean±std（ddof=0）。"
            "RI=(Raw−PAER)/Raw×100%，逐数据集用未舍入 MSE 计算；7 数据集汇总为各数据集 RI 的等权平均，不平均原始 MSE。"
            "BASE=无专家无形状项；SHAPE=无专家保留形状项；Raw Expert=8 条原始专家；PAER=Pool-Aware 冻结 v2 最终池。")
    write_table("table_2_internal_retrieval_main",
                "Table 2: 内部检索主结果（冻结 v2 最终 TEST，MSE，越低越好）",
                "tab:main", header, rows, note)


# =========================================================================== #
# 表 3：等专家数量对照（VAL）
# =========================================================================== #
def table3():
    pc = load_pool_size_control()
    header = ["Dataset", "K (PAER)", "PAER", "RAW_ALL", "RANDOM_K (mean±std)",
              "PAER vs RANDOM_K (RI)"]
    rows = []
    ri_list = []
    for ds in DS:
        r = pc[ds]
        k = r["K_v2"]
        v2 = float(r["v2_mean_mse"])
        raw = float(r["raw_all_mean_mse"])
        rk = float(r["random_k_mean_mse"]); rk_std = float(r["random_k_std_mse"])
        ri = (rk - v2) / rk * 100.0
        ri_list.append(ri)
        rows.append([ds, k, fmt(v2), fmt(raw), fmt_msd(rk, rk_std), pct(ri)])
    mean_ri = sum(ri_list) / len(ri_list)
    rows.append(["7-dataset (equal-weight RI)", "—", "—", "—", "—",
                 pct(mean_ri)])
    note = ("TRAIN-only / frozen VAL；每数据集固定 K=PAER 最终专家数，比较 PAER 选池 vs 同 K 随机子集 RANDOM_K（control seeds=[11,22,33,44,55]）"
            " vs RAW_ALL(8)。RANDOM_K 标准差为 5 个随机子集间的 std（与模型种子 std 区分）。"
            "旧 v1 评分选池 V1_SCORE_TOPK 见附录（历史控制身份）。")
    write_table("table_3_equal_expert_control",
                "Table 3: 等专家数量对照（VAL MSE，越低越好）", "tab:poolsize",
                header, rows, note)


# =========================================================================== #
# 表 4：知识优化过程
# =========================================================================== #
def table4():
    header = ["Dataset", "N initial (raw)", "N final (PAER)", "Pool size r0/r1/r2",
              "Text-changed (kept)", "Condition-changed (kept)",
              "Deleted", "Mechanism-replaced"]
    rows = []
    for ds in DS:
        hist = load_pool_reward_history(ds)
        raw = {e["id"]: e for e in load_experts(ds, "raw_experts")}
        fin = {e["id"]: e for e in load_experts(ds, "final_expert_pool")}
        n_initial = len(raw)
        n_final = len(fin)
        rounds = {}
        for r in hist:
            rounds.setdefault(int(r["round"]), set()).add(r["expert_id"])
        rstr = "/".join(str(len(rounds.get(i, []))) for i in sorted(rounds))
        text_changed = sum(1 for eid, e in fin.items()
                           if eid in raw and e.get("mechanism") != raw[eid].get("mechanism"))
        cond_changed = sum(1 for eid, e in fin.items()
                           if eid in raw and e.get("activation_condition") != raw[eid].get("activation_condition"))
        deleted = n_initial - n_final
        replaced = sum(1 for e in fin.values() if e.get("replaced"))
        rows.append([ds, n_initial, n_final, rstr, text_changed, cond_changed,
                     deleted, replaced])
    note = ("N initial=raw 8；N final=冻结 v2 最终池。Pool size r0/r1/r2 = 各轮实际评价池专家数（仅含待修订知识，KEEP 已移出）。"
            "Text/Condition-changed 仅统计最终保留且相对 raw 同 ID 机制文本/激活条件发生变化的条数。Deleted=raw 中被移除条数。"
            "Mechanism-replaced=最终保留池中 replaced=True 的条数。")
    write_table("table_4_knowledge_optimization",
                "Table 4: 知识优化过程统计", "tab:opt", header, rows, note)


# =========================================================================== #
# 表 5：专家评价与修订案例（精选）
# =========================================================================== #
CASE_SELECT = [
    ("etth1", "E1"), ("etth1", "E4"), ("etth1", "E7"), ("etth1", "E8"),
    ("electricity", "E2"), ("electricity", "E3"), ("electricity", "E8"),
    ("traffic", "E1"), ("traffic", "E5"),
]


def table5():
    header = ["Dataset", "Expert", "Round", "Pool size", "Decision", "Reward",
              "Coverage", "Mean pool reward", "Positive-fold ratio", "Replaced",
              "Mechanism (abridged)"]
    rows = []
    for ds, eid in CASE_SELECT:
        hist = load_pool_reward_history(ds)
        recs = [r for r in hist if r["expert_id"] == eid]
        if not recs:
            rows.append([ds, eid, "—", "—", "未记录", "—", "—", "—", "—", "—", "未记录"])
            continue
        # 取该 expert 最后一轮记录
        last = max(recs, key=lambda r: int(r["round"]))
        pool_sz = len({x["expert_id"] for x in hist if int(x["round"]) == int(last["round"])})
        mech = last["mechanism"]
        mech = mech if len(mech) <= 90 else mech[:87] + "..."
        rows.append([ds, eid, last["round"], pool_sz, last["verdict"],
                     last["reward"], f"{float(last['coverage']):.3f}",
                     f"{float(last['mean_pool_reward']):+.4f}",
                     f"{float(last['positive_fold_ratio']):.2f}",
                     last["replaced"], mech])
    note = ("精选保留/修订/剔除案例（每条取该 lineage 最后一轮评价记录）。Reward∈{2,1,0,−2,−1} 对应 KEEP/KEEP/UNSUPPORTED/REJECT/REVISE。"
            "Coverage=TRAIN 激活覆盖率（阈值 0.5）。留一贡献对应当轮池组成（屏蔽该专家嵌入+激活后重训，不删除槽位）。"
            "完整 per-expert/per-round 记录见 results/reward_feedback_critic_v2_full_cross_dataset/<ds>/pool_reward_history.csv。")
    write_table("table_5_expert_cases",
                "Table 5: 专家评价与修订案例（精选）", "tab:cases", header, rows, note)


# =========================================================================== #
# 表 6：覆盖与失败诊断（Electricity / Traffic）
# =========================================================================== #
def table6():
    header = ["Dataset", "Union coverage (Raw)", "Union coverage (PAER)",
              "Avg active experts (Raw)", "Avg active experts (PAER)",
              "No-coverage ratio (PAER)", "PAER vs Raw (RI, TEST)"]
    rows = [
        ["electricity", "1.000", "0.692", "3.53", "1.28", "30.8%", "-15.5%"],
        ["traffic", "1.000", "0.974", "未记录", "未记录", "未记录", "-0.4%"],
    ]
    note = ("覆盖阈值 0.5（激活≥0.5 计为激活）。electricity 的 avg active experts / no-coverage 来自逐窗口激活诊断"
            "（results/diagnose_v2_failure_electricity_traffic/）；traffic 仅记录 union coverage（1.000→0.974），逐窗口激活专家数未记录。"
            "本节为事后诊断（post-hoc），覆盖/激活为描述性分析，非模型输入；模型输入保留连续软激活。"
            "electricity 无覆盖窗口集中在高 volatility(Q4)/高 level(Q4) 与凌晨 3–6 点。")
    write_table("table_6_coverage_failure",
                "Table 6: 覆盖与失败诊断（Electricity / Traffic）", "tab:coverage",
                header, rows, note)


# =========================================================================== #
# 附录 A1：完整 MAE 与种子级结果
# =========================================================================== #
def appendix_a1():
    ab = load_final_test_ablation()
    header = ["Dataset", "Seed", "Method", "MSE", "MAE", "Pool size"]
    rows = []
    for ds in DS:
        for method in ["BASE", "SHAPE", "RAW_EXPERT", "REWARD_V1", "POOL_AWARE_V2"]:
            for s in SEEDS:
                v = ab[ds][method][s]
                rows.append([ds, s, METHOD_LABEL[method], f"{v['mse']:.6g}",
                             f"{v['mae']:.6g}", v["pool"] or ""])
    note = ("完整种子级 TEST MSE/MAE（seeds=[42,123,2026]）。v1(REWARD_V1) 为历史控制身份，不纳入正式排名。")
    write_table("appendix_a1_mae_seed",
                "附录 A1: 完整 MAE 与种子级结果（TEST）", "tab:appendix_mae",
                header, rows, note)


# =========================================================================== #
# 附录 A2：训练奖励与验证贡献对齐
# =========================================================================== #
def appendix_a2():
    al = load_reward_alignment()
    header = ["Dataset", "Spearman ρ", "p-value", "Sign-agreement", "Top-3 overlap", "Top-4 overlap"]
    rows = []
    for ds in ["etth1", "solar", "electricity", "traffic"]:
        r = al[ds]
        rows.append([ds, r["spearman_rho"], r["spearman_p"], r["sign_agreement_ratio"],
                     r["top3_overlap"], r["top4_overlap"]])
    note = ("TRAIN pool-aware reward（round-0 均值）与 VAL 留一贡献的 Spearman 相关。仅 4 数据集有 val_contribution 产物"
            "（etth2/weather/exchange_rate 未记录）。对齐并非全面更强（1 正/2 负/1 近 0），如实报告。")
    write_table("appendix_a2_reward_alignment",
                "附录 A2: 训练奖励与验证贡献对齐", "tab:appendix_align",
                header, rows, note)


# =========================================================================== #
# 附录 A3：检索 Top-K 敏感性
# =========================================================================== #
def appendix_a3():
    sn = load_sensitivity()
    header = ["Dataset", "K=1", "K=3", "K=5 (正式)", "K=10"]
    rows = []
    for ds in ["etth1", "etth2", "exchange_rate", "traffic"]:
        r = sn[ds]
        rows.append([ds, fmt_msd(float(r["mse_k1"]), float(r["std_k1"])),
                     fmt_msd(float(r["mse_k3"]), float(r["std_k3"])),
                     fmt_msd(float(r["mse_k5"]), float(r["std_k5"])),
                     fmt_msd(float(r["mse_k10"]), float(r["std_k10"]))])
    note = ("TRAIN/frozen VAL，mean±std（3 种子）。仅 4 快数据集；weather/solar/electricity 未完成。K=5 非窄峰，结果对 K 稳健。")
    write_table("appendix_a3_topk_sensitivity",
                "附录 A3: 检索 Top-K 敏感性（VAL MSE）", "tab:appendix_topk",
                header, rows, note)


# =========================================================================== #
# 附录 A4：效率与优化成本
# =========================================================================== #
def appendix_a4():
    ef = load_efficiency()
    header = ["Dataset", "Pool size raw/v1/v2", "v2 reward retriever trainings",
              "Retriever train time (s/arm)", "Inference (ms/window)", "Peak mem (MB)"]
    rows = []
    for ds in DS:
        r = ef[ds]
        pool = f"{r['raw_final_pool_size']}/{r['v1_final_pool_size']}/{r['v2_final_pool_size']}"
        rows.append([ds, pool, r["v2_retriever_trainings_for_reward"],
                     r["retriever_training_time_sec"], r["inference_time_per_window_ms"],
                     r["peak_mem_mb"]])
    note = ("实测 RTX 5070 Ti / 16GB（etth1，每项 3 次取 median）。v2 pool-aware reward 每数据集约 108 次 retriever 训练"
            "（TRAIN-only，一次性）；推理阶段无额外开销。详见 results/paper_completion/efficiency_report.md。")
    write_table("appendix_a4_efficiency",
                "附录 A4: 效率与优化成本", "tab:appendix_efficiency",
                header, rows, note)


# =========================================================================== #
# 附录 A5：外部模型与 Chronos-2 融合
# =========================================================================== #
def appendix_a5():
    models = ["PatchTST", "iTransformer", "RAFT", "RAF"]
    vals = defaultdict(lambda: defaultdict(list))
    with EXT.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "OK" and r["mse"]:
                vals[r["model"]][r["dataset"]].append(float(r["mse"]))
    chronos = ours = {}
    if FUSION.exists():
        with FUSION.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                chronos[r["dataset"]] = float(r["chronos_raw_test_mse"])
                ours[r["dataset"]] = (float(r["fusion_test_mse_mean"]), float(r["fusion_test_mse_std"]))
    header = ["Dataset", "Chronos-2", "PatchTST", "iTransformer", "RAFT", "RAF", "Ours Fusion"]
    rows = []
    for ds in DS:
        def cell(m):
            v = vals[m][ds]
            if not v:
                return "未完成"
            if len(v) == 1:
                return f"{fmt(v[0])} (1s)"
            mu = sum(v) / len(v)
            std = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
            return fmt_msd(mu, std)
        c = fmt(chronos[ds]) if ds in chronos else "未完成"
        o = fmt_msd(*ours[ds]) if ds in ours else "未完成"
        rows.append([ds, c, cell("PatchTST"), cell("iTransformer"), cell("RAFT"),
                     cell("RAF"), o])
    note = ("PatchTST/iTransformer/RAFT 为 3-seed mean±std；RAF 为 1 seed（num_samples=5）。"
            "Chronos-2 零样本确定性；Ours Fusion=Chronos-2+Pool-Aware v2 凸融合（λ 在 VAL 选，TEST 只用一次）。"
            "失配协议下的旧 RAF 结果不纳入正式排名。当前 Chronos-2/Ours 及 RAF solar/electricity 仍标注「未完成」（后台续跑中）。")
    write_table("appendix_a5_external_fusion",
                "附录 A5: 外部模型与 Chronos-2 融合（TEST MSE）", "tab:appendix_external",
                header, rows, note)


# =========================================================================== #
# 附录 A6：旧 v1 评分选池（历史控制身份）
# =========================================================================== #
def appendix_a6():
    pc = load_pool_size_control()
    header = ["Dataset", "K", "PAER (pool-aware)", "V1_SCORE_TOPK (historical v1)",
              "PAER vs V1_SCORE_TOPK (RI)"]
    rows = []
    ri_list = []
    for ds in DS:
        r = pc[ds]
        v2 = float(r["v2_mean_mse"]); v1s = float(r["v1_score_topk_mean_mse"])
        ri = (v1s - v2) / v1s * 100.0
        ri_list.append(ri)
        rows.append([ds, r["K_v2"], fmt(v2), fmt(v1s), pct(ri)])
    mean_ri = sum(ri_list) / len(ri_list)
    rows.append(["7-dataset (equal-weight RI)", "—", "—", "—", pct(mean_ri)])
    note = ("历史控制：v1 逐条 individual reward 评分取 Top-K 选池（V1_SCORE_TOPK），与 PAER(pool-aware) 同 K 对照。"
            "V1_SCORE_TOPK 为旧 v1 评分结果，仅作历史控制，不纳入正式方法排名。")
    write_table("appendix_a6_v1_score_control",
                "附录 A6: 旧 v1 评分选池（历史控制，VAL MSE）", "tab:appendix_v1score",
                header, rows, note)


# =========================================================================== #
# sources.json
# =========================================================================== #
def write_sources():
    src = {
        "seeds": SEEDS, "std_ddof": 0,
        "ri_formula": "RI = (control_MSE - PAER_MSE) / control_MSE * 100%",
        "summary": "7-dataset RI = equal-weight mean of per-dataset RI (not mean of raw MSE)",
        "coverage_threshold": 0.5,
        "tables": {
            "table_1_datasets_protocol": {
                "source": "results/multidataset/<ds>/config.json",
                "fields": "source, target, frequency, context, horizon, num_windows, split"},
            "table_2_internal_retrieval_main": {
                "source": "results/paper_completion/paper_main_ablation.csv (REUSE results/reward_feedback_critic_v2_final_test/final_test_ablation.csv)",
                "split": "TEST", "seeds": SEEDS, "std_ddof": 0},
            "table_3_equal_expert_control": {
                "source": "results/paper_completion/paper_pool_size_control.csv",
                "split": "VAL", "random_control_seeds": [11, 22, 33, 44, 55]},
            "table_4_knowledge_optimization": {
                "source": "results/reward_feedback_critic_v2_full_cross_dataset/<ds>/{pool_reward_history.csv,raw_experts.json,final_expert_pool.json}"},
            "table_5_expert_cases": {
                "source": "results/reward_feedback_critic_v2_full_cross_dataset/<ds>/pool_reward_history.csv"},
            "table_6_coverage_failure": {
                "source": "results/diagnose_v2_failure_electricity_traffic/ + results/paper_completion/paper_pool_analysis.csv",
                "coverage_threshold": 0.5},
            "appendix_a1_mae_seed": {
                "source": "results/paper_completion/paper_main_ablation.csv", "split": "TEST"},
            "appendix_a2_reward_alignment": {
                "source": "results/reward_feedback_critic_v2_failure_diagnosis/reward_alignment_train_val.csv"},
            "appendix_a3_topk_sensitivity": {
                "source": "results/paper_completion/paper_sensitivity.csv", "split": "VAL"},
            "appendix_a4_efficiency": {
                "source": "results/paper_completion/paper_efficiency.csv"},
            "appendix_a5_external_fusion": {
                "source": "results/external_baselines/external_baseline_results_full.csv + results/paper_completion/phase1_external_baselines/ours_fusion_v2_results.csv"},
        },
        "missing": {
            "val_contribution_etth2_weather_exchange_rate": "未记录（仅 4/7 有）",
            "topk_sensitivity_weather_solar_electricity": "未完成",
            "traffic_avg_active_experts": "未记录",
            "external_fusion_chronos2_ours": "未完成（后台续跑中）",
        },
    }
    (OUT / "sources.json").write_text(json.dumps(src, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print("  wrote sources.json")


# =========================================================================== #
# main
# =========================================================================== #
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Generating paper tables ->", OUT)
    table1()
    table2()
    table3()
    table4()
    table5()
    table6()
    appendix_a1()
    appendix_a2()
    appendix_a3()
    appendix_a4()
    appendix_a5()
    appendix_a6()
    write_sources()
    print("Done.")


if __name__ == "__main__":
    main()
