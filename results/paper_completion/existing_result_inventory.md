# P0 已有结果清单（existing_result_inventory）

扫描日期：2026-09-20。列出与本论文（LLM Expert → Executable Retrieval → Reward Feedback → Pool-Aware v2）相关的已有结果。
标记：`REUSE_EXISTING_RESULT` 表示本次补实验直接复用，不重跑。

## 核心正式结果（冻结 v2 链路）

### TEST（frozen v2 final TEST，one-time，只读）
目录 `results/reward_feedback_critic_v2_final_test/`
- `final_test_ablation.csv` — 7 数据集 × 5 臂 × 3 seeds，含 test_mse / test_mae。**REUSE**
- `final_test_summary.csv` — 每数据集 mean mse/mae、v2 vs raw/v1 %、seed wins。**REUSE**
- `final_test_stats.json` — mean/median RI vs raw/v1（+13.40% / +17.71% / +17.77% / +18.34%）。**REUSE**
- `final_test_report.md` / `FINAL_RESULT_zh.md` — 结论报告。**REUSE**
- `val_test_generalization.csv` — VAL→TEST 方向 7/7。**REUSE**
- 各数据集 `result.json` / `test_ablation.csv` — per-seed 明细。**REUSE**

### VAL（frozen v2 full cross dataset）
目录 `results/reward_feedback_critic_v2_full_cross_dataset/`
- `full_cross_dataset_ablation.csv` — 7 数据集 × 5 臂 × 3 seeds MSE。**REUSE**
- `full_cross_dataset_summary.csv` — mean/RI/seed wins/pool size。**REUSE**
- `full_cross_dataset_pool_stats.csv` — raw/v1/v2 pool size。**REUSE**
- `full_cross_dataset_stats.json` / `report.md`。**REUSE**
- 各数据集 `final_expert_pool.json`（v2 pool）、`raw_experts.json`、`pool_reward_history.csv`（v2 pool-aware reward round-0..N）。**REUSE**

### v1（reward-feedback critic v1，冻结 pool 来源）
目录 `results/reward_feedback_critic_v1_cross_dataset/`（etth1 在 `..._v1_etth1/`）
- 各数据集 `final_expert_pool.json`、`reward_history.csv`（v1 逐条 individual reward）、`val_ablation.csv`。**REUSE**
- `cross_dataset_summary.csv`、`cross_dataset_reward_stats.csv`、`cross_dataset_expert_pool_stats.csv`（n_keep/n_revise/n_reject/n_unsupported）。**REUSE**

### v2 pool-aware pilot（3 数据集早期）
目录 `results/reward_feedback_critic_v2_poolaware_pilot/`
- `pilot_summary.csv`、`pilot_val_ablation.csv`、`pilot_report.md`。已被 full cross dataset 取代，仅作溯源。**REUSE（溯源）**

## 失败诊断（Electricity / Traffic）
目录 `results/reward_feedback_critic_v2_failure_diagnosis/`
- `failure_diagnosis_report.md` / `failure_diagnosis_summary.csv` — Electricity=COVERAGE_LOSS（coverage 1.0→0.692）、Traffic=COMPARABLE。**REUSE**
- `reward_alignment_train_val.csv` — 4 数据集 train pool_reward vs val_contribution 的 Spearman/sign/topk。**REUSE（P3 核心）**
- `pool_context_dependence/` — 子分析（coverage-performance link、expert context contribution）。**REUSE**
- `pairwise_synergy_trainval.csv` — 两两交互。**REUSE**

## 相关但非本论文核心（不纳入正式表）

- `results/multidataset/` — 7 数据集 series.csv / config.json / window_indices.npz（数据准备，**REUSE**）。
- `results/learned_critic_*` — Learned Critic 数据集/训练（另一条线，v1 LODO 未超 v6.1，**不包装为成功**）。
- `results/etth1_critic_v6_1_final` / `v7_final` / `electricity_critic_v6_1_cross_dataset` — Critic v6.1/v7 链（早期 Agent 闭环）。
- `results/reward_feedback_critic_v2_1_robust_reward_pilot` — v2.1 robust reward（**方向 1 已证伪**，不纳入成功方法）。
- `results/expert_mlp_retrieval_v0/v1`、`expert_embedding_mlp`、`expert_main_shape_aux_v3`、`v3_feature_ablation` — Expert-MLP / 特征消融（早期）。
- `results/cross_dataset_validation/`、`results/cross_season_*`、`results/multiagent/` — 早期跨数据集/季节/多智能体。

## 缺口（本补实验要填）

1. **等 Pool Size 控制**（P2）— 无 RANDOM_K / V1_SCORE_TOPK 对照。→ 需新跑。
2. **Top-K 敏感性**（P6）— 无 K∈{1,3,10} 对照。→ 需新跑。
3. **效率开销**（P5）— 日志无 wall-clock，需重测时间。
4. **val_contribution**（P3）— 仅 4/7 数据集有；etth2/weather/exchange_rate 缺。
5. **逐窗口 TEST prediction**（P4）— 仅 electricity/traffic 有；其余 5 数据集只有 3-seed aggregate。
6. **外部 baseline**（P8）— 无 PatchTST/iTransformer/RAFT 实现（见 baseline_audit.md）。

## 结论

正式 A/B/C/D/E 消融的 TEST 与 VAL 均已存在且冻结，可直接复用（P1/P3/P4/P7 大部分为 REUSE）。
真正需要新跑的只有 P2（等 pool size 控制）与 P6（Top-K 敏感性），以及 P5 的效率重测。
