# 第一阶段最终对比实验 — 总结（中文）

- 日期：2026-10-09
- 状态：**BLOCKED（存在关键协议冲突，未进入执行）**
- 协议：冻结 Pool-Aware v2；不开发 v3；不因 TEST 改方法；seeds=[42,123,2026]；正式 7 数据集 = etth1 / etth2 / weather / exchange_rate / solar / electricity / traffic。

---

## 1. 已完成的实验

| 模型 | 完成度 | 状态 |
|---|---|---|
| PatchTST | **7/7（21/21 seed）** | ✅ 复用已有结果 |
| iTransformer | **7/7（21/21 seed）** | ✅ 复用已有结果 |
| RAFT | **7/7（21/21 seed）** | ✅ 复用已有结果 |
| RAF | **4/7**（缺 weather/solar/electricity，seed42） | ⚠️ 需补 3 个 |
| Chronos-2 | **0/7**（正式协议下无任何预测） | ❌ 需 7/7 重跑 |
| Ours Fusion | **0/7** | ❌ **BLOCKED（方法身份冲突）** |

## 2. 最终 Ours 的真实结果

**尚无。** 原因是关键方法身份冲突（详见 `OURS_FUSION_AUDIT.md`）：

- 任务要求的「Ours」= **Chronos-2 + Frozen Pool-Aware v2 Expert Memory/Fusion**。
- 仓库内 `experiments/external_baselines/run_ours_fusion.py` 实现的是**另一个更早的方法**「Chronos-2 + 时间对齐形状检索（residual-Pearson + circular-LUT + mean-anchor）融合」，**没有**加载 v2 专家池、没有 Pool-Aware Reward、没有专家指导检索。
- 「Chronos-2 + Pool-Aware v2 专家记忆融合」**从未被实现**。

因此无法给出「Ours vs Chronos-2」的相对改善。在方法身份确定前，任何数字都是不可信/不可并表的。

## 3. 与外部 baseline 的比较（当前可算部分）

以下是**已完成的 4 个外部 baseline** 的原始值 MSE（mean±std over seeds，越低越好）：

| Dataset | PatchTST | iTransformer | RAFT | RAF | Chronos-2 | Ours |
|---|---|---|---|---|---|---|
| etth1 | 7.987±0.0717 | 9.067±0.212 | 6.819±0.00824 | 8.939 | 待跑 | 待定 |
| etth2 | 14.50±0.0459 | 16.78±0.345 | 16.48±0.128 | 18.02 | 待跑 | 待定 |
| exchange_rate | 0.002213±0.00134 | 0.002021±0.000149 | 0.000955±2.45e-06 | 0.001215 | 待跑 | 待定 |
| traffic | 0.001499±9.43e-07 | 0.001346±2.94e-06 | 0.002308±5.93e-05 | 0.002084 | 待跑 | 待定 |
| weather | 157.6±7.73 | 168.4±5.96 | 146.0±0.0867 | 待补 | 待跑 | 待定 |
| solar | 1.763e5±967 | 2.195e5±4.55e3 | 2.010e5±447 | 待补 | 待跑 | 待定 |
| electricity | 1.019e8±4.42e5 | 1.020e8±2.22e5 | 1.375e8±1.74e5 | 待补 | 待跑 | 待定 |

**Ours vs 外部 baseline 的胜负无法计算**（Ours 未跑且方法未定）。

## 4. 与之前内部实验的关系（关键区分）

- **CNN-only v2 内部消融**（`results/reward_feedback_critic_v2_final_test/`，冻结）：回答「机制是否有效」。结果 = v2 vs Raw **5/7 胜、mean RI +13.40%**；v2 vs v1 6/7 胜。这是**机制证据链**，其 backbone 是 CNN/Expert-MLP（`ConditionalExpertMLPRetriever`），**不是** Chronos-2。
- **最终 Ours Fusion**（本次目标）：回答「最终预测系统是否有竞争力」。应是 **Chronos-2 + Pool-Aware v2 记忆融合**，与外部 baseline 在同口径下对比。**这是两套不同结论，不能混用。**
- **不能混用的原因**：CNN-only v2 的 etth1 MSE=42.42，明显劣于 PatchTST(7.99)/iTransformer(9.07)/RAFT(6.82)/RAF(8.94)；但这不否定机制，因为机制结论来自「v2 vs 内部 BASE/SHAPE/RAW/v1」的消融，而非「vs 外部深度 baseline」的 SOTA 对比。

## 5. 是否还需要补实验

投稿前**真正必要**的缺口（精确到 Dataset×Seed×Model）：

1. **RAF / weather / solar / electricity / seed42**（3 个，可独立补跑，约 6.5h）—— 无方法争议。
2. **Chronos-2 / 7 数据集**（正式协议下重跑）—— 需先解决 chronos 版本与 `amazon/chronos-2` 兼容问题（见 `CHRONOS2_REUSE_AUDIT.md` §3）。
3. **Ours Fusion / 7 数据集** —— **必须先决策方法身份**（见 `OURS_FUSION_AUDIT.md` §4 的 Q1/Q2/Q3），不决策则无法启动。

## 6. 论文是否可以开始正式写作

**核心机制证据链已闭环，可以开始写「方法 + 内部消融」部分**（P1 主消融 / P2 等 pool size 控制 / P3 reward 对齐 / P5 效率 / P7 coverage，均有 7/7 冻结结果）。

但**「与外部 baseline 的最终对比表」尚未完成**，且存在一个必须先决策的方法身份问题（Ours Fusion 到底是「Chronos-2 + 时间对齐检索」还是「Chronos-2 + Pool-Aware v2 专家记忆」）。在决策前：

- 不应声称「outperforms Chronos-2 / deep forecasting baselines」。
- 不应把 CNN-only v2 数字填进外部对比表冒充最终 Ours。

**结论：论文可进入「方法 + 机制消融」的正式写作；「外部对比表」一节需在 Ours Fusion 方法身份决策 + 补跑后补齐。**

---

## 附：本次交付文件清单

- `CHRONOS2_REUSE_AUDIT.md` — Chronos-2 复用审计（0/7 复用，7/7 重跑）
- `OURS_FUSION_AUDIT.md` — Ours Fusion 方法身份冲突审计（BLOCKED）
- `RAF_COMPLETION_REPORT.md` — RAF 补全报告（4/7，缺 3）
- `PHASE1_FINAL_SUMMARY_CN.md` — 本文件
- `EXTERNAL_BASELINE_FINAL_TABLE.md` — **未生成**（Ours 与 Chronos-2 缺失，无法生成可信主表）
