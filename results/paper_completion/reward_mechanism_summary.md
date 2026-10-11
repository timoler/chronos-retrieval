# P3 Reward 机制对照（individual v1 vs pool-aware v2）

individual_reward_v1 = v1 逐条孤立 reward（final round 的离散 reward -2..2）；
pool_reward_v2 = v2 的 TRAIN-only 边际贡献 reward（round-0 mean）。
val_contribution = (MSE[raw∖Ei]-MSE[raw])/MSE[raw∖Ei]（seed=42，仅 4 数据集已有）。

## train pool_reward(v2) 与 VAL contribution 对齐（已有 `reward_alignment_train_val.csv`）

| Dataset | Spearman | sign-agreement | top3 overlap | top4 overlap | 解读 |
|---|---|---|---|---|---|
| etth1 | 0.286 | 0.625 | 0.667 | 0.75 | 对齐 |
| solar | -0.81 | 0.375 | 0.0 | 0.25 | 未对齐/负 |
| electricity | -0.19 | 0.375 | 0.333 | 0.5 | 未对齐/负 |
| traffic | -0.548 | 0.375 | 0.333 | 0.25 | 未对齐/负 |

> etth2 / weather / exchange_rate 无 val_contribution artifact，train pool_reward 与 val contribution 的 Spearman 无法在已有结果上计算（见「仍缺实验」）。

## 结果
- v2 对齐并非全面更强：etth1 正对齐（Spearman=+0.286），solar/traffic 负对齐（-0.810 / -0.548），electricity 接近 0（-0.190）。
- 说明 pool_reward 在 TRAIN 与 VAL contribution 之间并不总是方向一致；v2 的正收益不能简单解释为「reward 更准」。

## 结论
- 如实报告：v2 reward 对齐在 4 个可算数据集上 1 正 / 2 负 / 1 近 0，未发现全面更强的对齐。
- 不因此修改 reward（冻结 v2）。