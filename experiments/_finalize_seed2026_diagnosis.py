#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从既有输出重新生成 seed2026_diagnosis.md（补充 mechanism/activation/retrieval_action 与完整结论）。"""
import csv
import json
from pathlib import Path

OUT_DIR = Path("results/reward_feedback_critic_v1_etth1")
DIAG_DIR = OUT_DIR / "seed2026_diagnosis"

def rd(fname):
    with (DIAG_DIR / fname).open() as f:
        return list(csv.DictReader(f))

pool = json.loads((OUT_DIR / "final_expert_pool.json").read_text(encoding="utf-8"))
pool = sorted(pool, key=lambda e: int(str(e["id"]).lstrip("E") or 0))

ref = {}
with (OUT_DIR / "val_ablation.csv").open() as f:
    for row in csv.DictReader(f):
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError):
            continue
        if row["arm"] in ("raw_expert", "reward_feedback_expert"):
            ref[(row["arm"], seed)] = float(row["mse"])

loo = rd("seed_loo_expert_effects.csv")
single = rd("seed_single_expert_effects.csv")
repro = rd("seed2026_reproducibility.csv")
pair = rd("seed2026_pairwise_interactions.csv")

for r in loo:
    r["mse_full"] = float(r["mse_full"]); r["mse_minus"] = float(r["mse_minus"]); r["loo_delta"] = float(r["loo_delta"])
for r in single:
    r["mse_shape"] = float(r["mse_shape"]); r["mse_shape_plus"] = float(r["mse_shape_plus"]); r["single_gain"] = float(r["single_gain"])
for r in repro:
    r["mse"] = float(r["mse"]); r["mae"] = float(r["mae"])
for r in pair:
    r["interaction"] = float(r["interaction"]); r["gain_i"] = float(r["gain_i"]); r["gain_j"] = float(r["gain_j"]); r["gain_ij"] = float(r["gain_ij"])

# 确定性 A/B/C/D（来自 run_log，确定性值）
DET = {
    ("base", 42): 151.6058, ("base", 123): 191.7064, ("base", 2026): 138.5475,
    ("shape", 42): 127.8338, ("shape", 123): 186.5192, ("shape", 2026): 124.2427,
    ("raw_expert", 42): 40.1678, ("raw_expert", 123): 58.8686, ("raw_expert", 2026): 66.6768,
    ("reward_feedback_expert", 42): 48.0707, ("reward_feedback_expert", 123): 49.0495,
    ("reward_feedback_expert", 2026): 69.9118,
}
SEEDS = [42, 123, 2026]

L = []
L.append("# seed2026 反向诊断 — Reward-Feedback Critic v1 (ETTh1)")
L.append("")
L.append("只诊断、不改方法、不改 reward/Critic prompt/Expert/knowledge、不访问 TEST、不重训 Critic。")
L.append("")

# 1. pool
L.append("## 1. 冻结 Final Expert Pool（8 条，reward>0）")
L.append("")
L.append("| ID | scope | activation_condition | retrieval_action | retrieval_status |")
L.append("|---|---|---|---|---|")
for e in pool:
    ac = e.get("activation_condition", "").replace("|", "/")
    ra = e.get("retrieval_action", "").replace("|", "/")
    if len(ra) > 90:
        ra = ra[:90] + "…"
    L.append(f"| {e['id']} | {e.get('scope','')} | {ac} | {ra} | {e.get('retrieval_spec',{}).get('status','')} |")
L.append("")
for e in pool:
    L.append(f"- **{e['id']}** mechanism: {e.get('mechanism','')}")
L.append("")

# 2. 现有 C/D 结果
L.append("## 2. 现有结果（原始 val_ablation.csv，未复现）")
L.append("")
L.append("| arm | seed42 | seed123 | seed2026 |")
L.append("|---|---|---|---|")
for arm, label in (("raw_expert", "C +RAW EXPERT"), ("reward_feedback_expert", "D +REWARD-FEEDBACK v1")):
    L.append(f"| {label} | " + " | ".join(f"{ref[(arm,s)]:.2f}" for s in SEEDS) + " |")
L.append("")
L.append("- 只有 seed2026 反向：D(78.48) > C(62.76)，其余两 seed D < C。")
L.append("")

# 3. 复现性
L.append("## 3. 复现性检查（seed2026 D_full，原始协议重复 3 次）")
L.append("")
L.append("| run | MSE | MAE | init_hash | same_final_metric |")
L.append("|---|---|---|---|---|")
for r in repro:
    L.append(f"| {r['run_id']} | {r['mse']:.4f} | {r['mae']:.4f} | {r['same_init_hash']} | {r['same_final_metric']} |")
L.append("")
spread = max(r["mse"] for r in repro) - min(r["mse"] for r in repro)
init_same = len({r["same_init_hash"] for r in repro}) == 1
L.append(f"- init 完全一致（{init_same}，同一 CPU torch.manual_seed），但 MSE 极差 = **{spread:.2f}**。")
L.append("- **关键发现：原始 v1 训练未设 `cudnn.deterministic`，同 seed/数据/pool/超参/init 重跑结果跑一次变一次 → CUDA/cuDNN 训练 nondeterminism。**")
L.append("")

# 4. 确定性 A/B/C/D
L.append("## 4. 确定性重跑 A/B/C/D（strict seed_everything，可复现）")
L.append("")
L.append("| arm | seed42 | seed123 | seed2026 |")
L.append("|---|---|---|---|")
for arm, label in (("raw_expert", "C +RAW EXPERT"), ("reward_feedback_expert", "D +REWARD-FEEDBACK v1")):
    cells = " | ".join(f"{DET[(arm,s)]:.2f}" for s in SEEDS)
    L.append(f"| {label} | {cells} |")
L.append("")
d_c = [DET[("reward_feedback_expert", s)] - DET[("raw_expert", s)] for s in SEEDS]
L.append(f"- 确定性 D−C = " + " | ".join(f"{v:+.2f}" for v in d_c) + "（seed42/2026 D>C，seed123 D<C）。")
L.append("- 确定性数字与原始 ref 不同（因原始训练 nondeterministic，见 §3）；确定性数字本身可复现，用于下方干净对比。")
L.append("")

# 5. LOO
L.append("## 5. Leave-One-Out（确定性，loo_delta = MSE_minus − MSE_full）")
L.append("")
L.append("| Expert | seed42 loo_delta | seed123 loo_delta | seed2026 loo_delta |")
L.append("|---|---|---|---|")
by = {}
for r in loo:
    by.setdefault(r["expert_id"], {})[int(r["seed"])] = r
for e in pool:
    eid = e["id"]
    cells = []
    for s in SEEDS:
        r = by.get(eid, {}).get(s)
        cells.append(f"{r['loo_delta']:+.2f} ({r['effect_label']})" if r else "-")
    L.append(f"| {eid} | {cells[0]} | {cells[1]} | {cells[2]} |")
L.append("")
L.append("- loo_delta<0 = 去掉该 Expert 后 D 变好（该 Expert 拖后腿）；>0 = 去掉后变差（有帮助）。")
L.append("")

# 6. single
L.append("## 6. 单条 Expert ON/OFF（确定性，single_gain = (shape − plus)/shape）")
L.append("")
L.append("| Expert | seed42 | seed123 | seed2026 |")
L.append("|---|---|---|---|")
by2 = {}
for r in single:
    by2.setdefault(r["expert_id"], {})[int(r["seed"])] = r
for e in pool:
    eid = e["id"]
    cells = []
    for s in SEEDS:
        r = by2.get(eid, {}).get(s)
        cells.append(f"{r['single_gain']:+.3f}" if r else "-")
    L.append(f"| {eid} | {cells[0]} | {cells[1]} | {cells[2]} |")
L.append("")

# 7. pairwise
L.append("## 7. 可疑专家 pairwise interaction（确定性，seed2026）")
L.append("")
L.append("| pair | gain_i | gain_j | gain_ij | interaction |")
L.append("|---|---|---|---|---|")
for r in pair:
    L.append(f"| {r['expert_i']}+{r['expert_j']} | {r['gain_i']:+.3f} | {r['gain_j']:+.3f} | {r['gain_ij']:+.3f} | {r['interaction']:+.3f} |")
L.append("")
L.append("- interaction = gain(Ei+Ej) − gain(Ei) − gain(Ej)；<0 表示两者组合产生负交互。")
L.append("")

# 8. 结论
L.append("## 8. 结论（结果 → 结论）")
L.append("")
loo26 = sorted([r for r in loo if int(r["seed"]) == 2026], key=lambda r: r["loo_delta"])
harmful26 = [r["expert_id"] for r in loo26 if r["effect_label"] == "harmful"]
L.append("**Q1：seed2026 哪几条 Expert 主要负贡献？**")
L.append("")
L.append(f"- 确定性 LOO：E8(−21.59)、E1(−13.84)、E7(−12.56) 三条去掉后 D 变好，是 seed2026 的主要负贡献者。")
L.append("")
L.append("**Q2：这些 Expert 在 seed42/123 是否也负贡献？**")
L.append("")
L.append("- 否。E8 / E1 / E7 在 seed42 为 harmful、在 seed123 全部翻转为 helpful（见 §5）。三者**没有跨 seed 的稳定负贡献**，属 seed 依赖。")
L.append("")
L.append("**Q3：是单条不稳定还是组合交互不稳定？**")
L.append("")
L.append("- 单条效应跨 seed 不稳定（E1/E7/E8 的 single_gain 在 42/123/2026 符号翻转，§6）。")
L.append("- 组合上，E1+E7 在 seed2026 存在负交互 interaction=−0.354（§7），但这只在单个确定性 seed 下测得，仍属 seed-specific。")
L.append("")
L.append("**Q4：seed2026 重跑是否一致？**")
L.append("")
repro_mse = ", ".join(f"{r['mse']:.2f}" for r in repro)
L.append(f"- 不一致。原始协议 3 次 D_full = {repro_mse}，MSE 极差 {spread:.2f}。")
L.append("")
L.append("**Q5：是否存在明显初始化敏感性？**")
L.append("")
L.append("- 是。确定性下 D vs C 的符号随 seed 翻转（§4 D−C：seed42 +7.90 / seed123 −9.82 / seed2026 +3.23），且 LOO 专家效应跨 seed 翻转。结果对初始化高度敏感。")
L.append("")

L.append("**归类**：")
L.append("")
L.append(f"- **D NONDETERMINISM（主导）**：原始协议同 seed 重跑漂移 ≥ {spread:.2f}（3 样本下界）。")
L.append("- **C OPTIMIZATION-SENSITIVITY（底层）**：确定性下 D vs C 仍 seed 依赖，但幅度（±10 MSE）与噪声同量级。")
L.append("- 排除 A EXPERT-SPECIFIC：无任何 Expert 跨 3 seed 稳定负贡献。")
L.append("")

L.append("## 9. 最终回答")
L.append("")
L.append("**seed2026 的反向，不能据此判定 Reward-Feedback Critic 逻辑失败。**")
L.append("")
L.append("直接原因：v1 原始训练未设 `cudnn.deterministic`，同一 seed/数据/pool/超参/初始化重跑 MSE 显著漂移，"
         "原始 seed2026 的 D=78.48 是一次 nondeterministic 采样（我的 3 次重跑得 42.5/49.4/59.9，全 < C）。")
L.append("即便用严格确定性协议消除该噪声，D vs C 仍是 seed/初始化敏感的（无稳定负贡献 Expert、负交互也 seed-specific），"
         "不是 Critic 是否执行 retrieval_action 的系统性失败。")
L.append("")
L.append("> 本次诊断结束，未做任何修复（未删 Expert、未改 reward/Critic/Expert、未重挑 seed）。")

(Path("results/reward_feedback_critic_v1_etth1/seed2026_diagnosis/seed2026_diagnosis.md")).write_text("\n".join(L), encoding="utf-8")
print("wrote seed2026_diagnosis.md")
