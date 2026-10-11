# Ours Fusion 审计（OURS_FUSION_AUDIT）

- 生成日期：2026-10-09
- 任务目标：完成「**Chronos-2 + Frozen Pool-Aware v2 Expert Memory/Fusion**」
- 审计结论：**关键协议冲突，BLOCKED。** `experiments/external_baselines/run_ours_fusion.py` **没有**实现「Chronos-2 + Frozen Pool-Aware v2 Expert Memory/Fusion」，它实现的是另一个更早的方法「Chronos-2 + 时间对齐形状检索（temporal-weights）融合」。仓库内**不存在**「Chronos-2 + Pool-Aware v2 专家记忆融合」的任何实现。

---

## 0. 一句话结论

`run_ours_fusion.py` 的 fusion 记忆来源是 **residual-Pearson 形状相似 + 时间 circular-LUT（hour/dow/month）→ top-K=5 → mean-anchor**，这是旧「连续时间权重」方法（源自 `archive/old_experiments/run_continuous_temporal_weights_all_ett.py`）。它**没有**加载 v2 专家池，**没有**执行专家指导检索，**没有**用 Pool-Aware Reward 选出的 expert 记忆。而论文最终方法（Pool-Aware v2）的记忆来源是 `ConditionalExpertMLPRetriever`（CNN/Expert-MLP）在 v2 expert pool 下的条件检索，两者是**不同的方法**。

## 1. 仓库内实际上存在三套「记忆/检索」系统（不能混用）

| 系统 | 脚本 | 记忆来源 | backbone | 状态 |
|---|---|---|---|---|
| A. 时间对齐形状检索（temporal-weights） | `run_temporal_ablation_all_ett.py` / `run_continuous_temporal_weights_all_ett.py` → `external_baselines/run_ours_fusion.py` | residual-Pearson 形状 + circular-LUT 时间对齐 → mean-anchor | **Chronos-2** | 旧「Ours」，已审计未跑 |
| B. Pool-Aware v2 专家记忆（论文核心） | `run_reward_feedback_critic_v2_*.py` → `run_reward_feedback_critic_v2_final_test.py` | LLM expert 知识 + Pool-Aware Reward 选 pool → `ConditionalExpertMLPRetriever` | **CNN/Expert-MLP（非 Chronos-2）** | 冻结，已跑 7/7 TEST |
| C. Chronos-2 融合（目标） | **不存在** | Pool-Aware v2 专家记忆 ⊕ Chronos-2 预测 | **Chronos-2** | 未实现 |

## 2. `run_ours_fusion.py` 逐项核对（对照任务 Task 2 检查清单）

| 检查项 | 结果 | 说明 |
|---|---|---|
| 1. Chronos-2 作为 backbone | ✅ 是 | `Chronos2Pipeline.from_pretrained("amazon/chronos-2")` |
| 2. 加载正确的 v2 专家池 | ❌ 否 | 完全没有读 `final_expert_pool.json`，无任何 expert 概念 |
| 3. 执行专家指导的历史检索 | ❌ 否 | 用 residual-Pearson + circular-LUT，非专家知识 |
| 4. 用历史未来片段构建 Memory Prediction | ✅ 是（但非 v2 记忆） | `mean_anchor` 用 bank.futures，是形状检索记忆，不是 expert 记忆 |
| 5. 按已确定的融合公式/门控 | ✅ 公式对，来源错 | `fused=(1-λ)·chronos+λ·memory`，但 memory 是 A 系统不是 B 系统 |
| 6. 未来信息泄漏 | ✅ 无 | 记忆库只由 TRAIN 段构建，test 只评一次 |
| 7. 参数均来自 TRAIN/VAL | ✅ 是 | λ/α/w/period 都在 VAL 上选 |
| 8. 与 Chronos-2 baseline 同窗口 | ✅ 是 | 同一 `window_indices.npz` test 窗口 |

**结论：`run_ours_fusion.py` 是「Chronos-2 + A 系统」，不是任务要求的「Chronos-2 + B 系统（Pool-Aware v2）」。**

## 3. 为什么这是关键冲突（不能擅自处理）

1. 任务约束 #5：**「不得把 CNN-only v2 误标为最终 Ours Fusion」** —— 即论文的最终 Ours 必须是「Chronos-2 + Pool-Aware v2 记忆」，而不是 CNN-only v2。
2. 但「Chronos-2 + Pool-Aware v2 记忆融合」**从未被实现**：Pool-Aware v2 的冻结产物只有 `final_expert_pool.json`（每个数据集 5–7 个 expert 的 claim/embedding），其记忆预测由 `ConditionalExpertMLPRetriever` 在推理时**重新训练**（`train_arm`，30 epoch）+ `retrieve_conditional` 得到，且**只算 MSE 不落预测数组**（`run_reward_feedback_critic_v2_final_test.py` 未保存 per-window `pred`）。
3. 要把 v2 记忆接入 Chronos-2 融合，需要：a) 在正式协议下重跑 v2 retriever 拿到 TEST 上的 v2 记忆预测数组；b) 在正式协议下拿到 Chronos-2 TEST 预测数组；c) 用 `fused=(1-λ)·chronos+λ·v2_memory`（λ 在 VAL 选）。这一步**既是新接线，也可能被判定为「方法开发」**，任务明文禁止「本次是实验收尾，不是方法开发」。

## 4. 需要用户决策的关键问题

**Q1：最终「Ours Fusion」到底指哪一个？**
- (a) `run_ours_fusion.py` 已实现的「Chronos-2 + 时间对齐形状检索」融合（A 系统）—— 若选它，直接跑即可，但它是**旧方法**，与论文「Pool-Aware v2」核心贡献不一致；
- (b) 「Chronos-2 + Pool-Aware v2 专家记忆」融合（B 系统接入 Chronos-2）—— 符合论文方向，但**需要新接线**（且需确认是否算方法开发）。

**Q2：融合公式用哪个？** 仓库内已有两种候选：
- 固定凸融合 `fused=(1-λ)·chronos+λ·memory`，λ 在 VAL 网格选（A 系统、solar 旧融合均用此法）；
- 自适应门控 `run_adaptive_fusion_gate.py`（学一个 query-specific λ 的 MLP gate）。

**Q3：v2 记忆预测是否需要落盘重跑？** 冻结的 `final_test` 只有 MSE，没有 per-window 预测数组；要融合必须重跑 v2 retriever 得到 TEST 预测数组（约每数据集每 seed 65s 训练 + 单窗推理，TRAIN 阶段 pool-aware reward 已冻结，不重算）。

## 5. 当前状态

- `run_ours_fusion.py`：**未执行**（无 `results/external_baselines/ours_fusion_results.csv`）。
- 正确的「Chronos-2 + Pool-Aware v2」：**未实现**。
- 结论：**BLOCKED，等待 Q1/Q2/Q3 决策后再执行**，不擅自以 A 系统冒充最终 Ours，也不擅自开发新融合变体。
