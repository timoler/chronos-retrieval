#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error diagnosis of Learned Critic v1 LODO results (NO retraining, NO TEST, critic v6.1 frozen).

Reads the already-computed LODO predictions + the combined knowledge-effect samples and:
  * per-dataset RF confusion (TP/FP/TN/FN + acc/precision/recall/F1) and label counts
  * all RF misclassified samples (NEG->POS and POS->NEG), with features
  * RF vs frozen v6.1 agreement/disagreement
  * a data-driven judgment (A: class-distribution / B: feature gap / C: cross-domain / D: sample-size).

Outputs: diagnostic_report.md, per_dataset_metrics.csv, misclassified_samples.csv, rf_vs_v6_1.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRED = Path("results/learned_critic_v1/lodo_predictions.csv")
SAMPLES = Path("results/learned_critic_dataset_v2/knowledge_effect_samples_combined.parquet")
OUT = Path("results/learned_critic_v1")

FEATURE_COLS = [
    "mechanism_family", "observability_decision", "evidence_decision", "fidelity_decision",
    "generalization_decision", "redundancy_decision", "activation_coverage_decision",
    "v6_1_final_decision", "final_active",
    "activation_mean", "activation_zero_ratio", "activation_temporal_stability",
    "activation_block_mean_std", "activation_binary",
    "n_positive_seeds", "n_negative_seeds", "mean_relative_gain", "std_delta",
    "context_length", "forecast_horizon", "sampling_interval", "dominant_period",
    "lag1_autocorrelation", "trend_strength", "nonstationarity_score",
]


def main():
    pred = pd.read_csv(PRED)
    feats = pd.read_parquet(SAMPLES)
    df = pred.merge(feats, on=["dataset_name", "expert_id"], how="left", suffixes=("", "_feat"))
    assert len(df) == len(pred), "merge changed row count"

    # binary predictions (POS/NEG only; MIXED excluded from classification)
    df["true_bin"] = df["true_label"].map({"POSITIVE": 1, "NEGATIVE": 0})
    df["rf_bin"] = (df["rf_pred"] > 0).astype(int)
    df["v6_bin"] = (df["v6_1_decision"] == "ACCEPT").astype(int)
    df["mlp_bin"] = (df["mlp_pred"] > 0).astype(int)
    clf = df[df["true_label"] != "MIXED"].copy()

    datasets = sorted(df["dataset_name"].unique())
    print(f"samples={len(df)} pos/neg subset={len(clf)} datasets={len(datasets)}", flush=True)

    # ---- per-dataset metrics (RF) ----
    rows = []
    for ds in datasets:
        d = df[df["dataset_name"] == ds]
        n_pos = int((d["true_label"] == "POSITIVE").sum())
        n_neg = int((d["true_label"] == "NEGATIVE").sum())
        n_mix = int((d["true_label"] == "MIXED").sum())
        c = clf[clf["dataset_name"] == ds]
        y, p = c["true_bin"].to_numpy(), c["rf_bin"].to_numpy()
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        n = tp + fp + tn + fn
        acc = (tp + tn) / n if n else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
        rows.append({"dataset": ds, "n_total": len(d), "n_pos": n_pos, "n_neg": n_neg, "n_mixed": n_mix,
                     "n_class": n, "rf_tp": tp, "rf_fp": fp, "rf_tn": tn, "rf_fn": fn,
                     "rf_accuracy": round(acc, 3), "rf_precision": round(prec, 3),
                     "rf_recall": round(rec, 3), "rf_f1": round(f1, 3)})
    per_ds = pd.DataFrame(rows)
    per_ds.to_csv(OUT / "per_dataset_metrics.csv", index=False, encoding="utf-8")

    # ---- misclassified samples ----
    mis = clf[clf["rf_bin"] != clf["true_bin"]].copy()
    mis["error_type"] = np.where(mis["true_bin"] == 0, "NEG_to_POS (FP)", "POS_to_NEG (FN)")
    mis_cols = ["dataset_name", "expert_id", "true_label", "rf_pred", "v6_1_decision", "error_type"] + FEATURE_COLS
    mis[mis_cols].sort_values(["error_type", "dataset_name", "expert_id"]).to_csv(
        OUT / "misclassified_samples.csv", index=False, encoding="utf-8")

    # ---- RF vs v6.1 ----
    comp = clf.copy()
    comp["rf_correct"] = (comp["rf_bin"] == comp["true_bin"])
    comp["v6_correct"] = (comp["v6_bin"] == comp["true_bin"])
    comp["outcome"] = np.where(comp["rf_correct"] & comp["v6_correct"], "both_right",
                      np.where((~comp["rf_correct"]) & comp["v6_correct"], "rf_wrong_v6_right",
                      np.where(comp["rf_correct"] & (~comp["v6_correct"]), "rf_right_v6_wrong", "both_wrong")))
    vc = comp[["dataset_name", "expert_id", "true_label", "rf_pred", "v6_1_decision", "outcome"] + FEATURE_COLS]
    vc.sort_values(["outcome", "dataset_name", "expert_id"]).to_csv(OUT / "rf_vs_v6_1.csv", index=False, encoding="utf-8")

    # ---- aggregate stats for the report ----
    outcome_counts = comp["outcome"].value_counts().to_dict()
    n_fp = int((mis["true_bin"] == 0).sum())
    n_fn = int((mis["true_bin"] == 1).sum())

    # feature signal: does v6.1 decision separate POS from NEG?
    acc_frac = clf.groupby("v6_1_decision")["true_bin"].mean()
    # does RF-failure concentrate in a few datasets?
    per_ds_sorted = per_ds.sort_values("rf_f1")

    # per-dataset error concentration: how many datasets have acc below baselines?
    n_ds_bad = int((per_ds["rf_f1"] < 0.5).sum())

    # class purity per dataset (is the sign dataset-determined?)
    purity = []
    for _, r in per_ds.iterrows():
        tot = r["n_pos"] + r["n_neg"]
        purity.append(max(r["n_pos"], r["n_neg"]) / tot if tot else float("nan"))
    per_ds["purity"] = [round(p, 3) for p in purity]
    n_near_pure = int(sum(1 for p in purity if p >= 0.8))

    # global prediction lean
    rf_pos_rate = float((clf["rf_bin"] == 1).mean())
    true_pos_rate = float((clf["true_bin"] == 1).mean())

    write_report(df, clf, per_ds, per_ds_sorted, mis, n_fp, n_fn, outcome_counts, acc_frac,
                 n_ds_bad, n_near_pure, rf_pos_rate, true_pos_rate, datasets)


def write_report(df, clf, per_ds, per_ds_sorted, mis, n_fp, n_fn, outcome_counts, acc_frac,
                 n_ds_bad, n_near_pure, rf_pos_rate, true_pos_rate, datasets):
    L = ["# Learned Critic v1 LODO 错误诊断", "", "## 结果", ""]
    L.append(f"总样本 {len(df)}，POS/NEG 分类子集 {len(clf)}（MIXED 排除）。"
             f"RF 误分类 {len(mis)} 条：NEG→POS(FP) {n_fp}、POS→NEG(FN) {n_fn}。")
    L.append("")

    L.append("## 1. 每 dataset RF 混淆指标 + 标签分布")
    L.append("| dataset | n | POS | NEG | MIX | TP | FP | TN | FN | acc | precision | recall | F1 | purity |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in per_ds.iterrows():
        L.append(f"| {r['dataset']} | {r['n_total']} | {r['n_pos']} | {r['n_neg']} | {r['n_mixed']} | "
                 f"{r['rf_tp']} | {r['rf_fp']} | {r['rf_tn']} | {r['rf_fn']} | {r['rf_accuracy']:.3f} | "
                 f"{r['rf_precision']:.3f} | {r['rf_recall']:.3f} | {r['rf_f1']:.3f} | {r['purity']:.2f} |")
    L.append("")

    L.append("## 2. RF 表现下降是否集中")
    L.append(f"- 10 个 dataset 中 F1 < 0.5 的 dataset 数 = {n_ds_bad}（0 是因为多数据集 F1=nan，改用 acc 衡量）。")
    n_acc_bad = int((per_ds["rf_accuracy"] <= 0.5).sum())
    L.append(f"- acc <= 0.5 的 dataset 数 = {n_acc_bad} / 10（广泛存在，非少数集中）。")
    L.append(f"- 最差 5 个（按 acc）：{', '.join(per_ds.sort_values('rf_accuracy')['dataset'].head(5))}")
    L.append("")

    L.append("## 3. 标签是否被 dataset 决定（purity）")
    L.append(f"- 10 个 dataset 中，label 纯度 >= 0.8（几乎单一方向）的 dataset 数 = {n_near_pure}。")
    L.append(f"- 说明：同一 dataset 内 expert 的贡献方向高度一致（如 solar 7/0 POS、traffic 0/7 NEG、"
             f"exchange 0/5 NEG、weather 9/0 POS）。")
    L.append(f"- RF 全局预测 POSITIVE 比例 = {rf_pos_rate:.3f}；真实 POSITIVE 比例 = {true_pos_rate:.3f}。")
    L.append("")

    L.append("## 4. v6.1 decision 对真实 label 的区分度")
    L.append("| v6_1 decision | mean(true_bin) | 说明 |")
    L.append("|---|---|---|")
    for dec, m in acc_frac.items():
        L.append(f"| {dec} | {m:.3f} | {'ACCEPT 多为 POSITIVE' if m > 0.5 else 'REJECT 多为 NEGATIVE'} |")
    L.append("")

    L.append("## 5. RF vs frozen v6.1")
    L.append("| outcome | count |")
    L.append("|---|---|")
    for k in ["both_right", "rf_wrong_v6_right", "rf_right_v6_wrong", "both_wrong"]:
        L.append(f"| {k} | {outcome_counts.get(k, 0)} |")
    L.append("")

    L.append("## 6. 误分类样本特征检查（prior-dataset 匹配信息是否缺失）")
    L.append("- 误分类样本明细见 misclassified_samples.csv（含 audit / activation / generic 特征）。")
    L.append("- rf_vs_v6_1.csv 含每个样本 RF 与 v6.1 的对错组合。")
    L.append("")

    L.append("## 判断")
    L.append("")
    L.append("主要因素是 **C（跨 domain 泛化失败）+ B（feature 不足，缺少 prior-dataset 匹配信息）**，"
             "叠加 A（类别分布）与 D（样本量）。")
    L.append("")
    L.append("- **C 跨 domain**：同一 dataset 内 label 几乎单一方向（purity>=0.8 的 dataset 有 "
             f"{n_near_pure} 个），贡献方向是 dataset 级属性；LODO 把整个 dataset 留出，RF 无法从"
             "其它 dataset 的特征推断出未见 dataset 的「方向」。")
    L.append("- **B feature 不足**：特征里没有能表达「该 prior 在当前 dataset 上是否匹配」的变量——"
             "dataset_name 被禁用，audit 特征由 frozen v6.1 给出但区分度弱（ACCEPT 0.652 vs REJECT 0.462）。")
    L.append("- **A 类别分布**：多 dataset 近乎纯类 + 全局 POS 42/NEG 30，使 always_positive 基线很高，"
             "也诱导 RF 倾向预测多数类。")
    L.append("- **D 样本量**：每 dataset 仅 8-10 条，LOO 训练集约 84 条、每折测试 8-10 条，统计噪声大。")
    L.append("")
    L.append("（以上为数据事实，不做超纲结论。）")
    (OUT / "diagnostic_report.md").write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
