# 机制控制实验存在性检查

针对「专家知识的数值评价、反馈修订与组合筛选机制」这一论文重点，逐项检查三类机制控制实验是否存在。**不编造结果**；缺失实验只列缺失清单与所需控制条件。

## 1. 仅筛选不修订（filter-only，无 revision loop）

**状态：缺失（MISSING）。**

- 现有 v2 闭环固定执行 `round 0 → round 1 → round 2`，其中 REVISE/REJECT/UNSUPPORTED 专家会进入下一轮被 Expert 修订（`llm_revise_v2`）。没有「只用 round-0 pool-aware reward 直接筛选、跳过全部修订」的对照臂。
- 所需控制条件：与正式 PAER 完全相同的 round-0 池评价与 reward 规则，但**不执行任何修订**，直接把 round-0 中 reward>0 的专家作为最终池，其余流程（retriever 架构/训练/种子/split/Top-K）不变。
- 作用：分离「知识筛选」与「知识修订」的贡献。当前无法把 PAER 的收益归因于「筛选」还是「筛选+修订」二者之和。

## 2. 修订但保留全部专家（revise-keep-all）

**状态：缺失（MISSING）。**

- 现有闭环在每轮后把 KEEP 专家移出、仅修订 REVISE/REJECT/UNSUPPORTED，最终按 reward>0 删除部分专家。没有「保留全部 8 条专家、只做修订、不删除」的对照臂。
- 所需控制条件：与正式 PAER 相同的修订循环，但最终池保留全部 raw 8 条（不按 reward 删除），其余流程不变。
- 作用：分离「知识文本修订」的贡献（若 revise-keep-all 与 raw 无差异，说明修订本身不产生收益，收益来自筛选）。

## 3. 独立评分 vs 池感知评分的受控比较

**状态：部分存在（PARTIAL，已有替代对照 + 诊断）。**

- **替代对照（端到端，VAL）**：`table_3` 与 `appendix_a6` 已含同 K 选池对照——PAER（pool-aware reward 选池）vs `V1_SCORE_TOPK`（v1 逐条 individual reward 取 Top-K 选池），二者唯一差异是「选池所用评分机制」，retriever/初始化/训练预算/split 均相同。这构成「独立评分 vs 池感知评分」的**选池层面**受控比较。
- **诊断对照（数值层）**：`results/paper_completion/reward_mechanism_comparison.csv` 逐专家列出 `individual_reward_v1` 与 `pool_reward_v2` 及 `val_contribution`，用于对齐分析（附录 A2）。
- **限制**：这是「v1 独立评分」与「v2 池感知评分」两种**历史机制**的对照，不是在同一池感知框架内关闭/开启「池感知」开关的消融。因此只能支持「pool-aware 选池优于 v1 独立选池（5/7）」的结论，**不能**解释为「池感知奖励相对任一独立评分基线的独立贡献证明」。
- **明确禁止**：不把 Table 2 的 BASE/SHAPE/Raw/PAER 四臂比较解释成 pool-aware reward 的独立贡献证明——该四臂差异同时包含「是否用专家」「是否用形状项」「是否用 reward 筛选」，不是单一变量。

## 结论

| 机制控制 | 状态 | 现有替代 |
|---|---|---|
| 仅筛选不修订 | 缺失 | 无 |
| 修订但保留全部 | 缺失 | 无 |
| 独立评分 vs 池感知评分 | 部分 | Table 3 / A6（V1_SCORE_TOPK）+ reward_mechanism_comparison.csv |

论文撰写时应：① 把「筛选 vs 修订」「删除 vs 修订」的归因表述为**待补实验**；② 仅把 Table 3/A6 作为「选池质量」证据，不越界声明。
