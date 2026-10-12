> 版本提示：本目录为旧论文表格；最新中文稿采用 [2026-10-12 表格与作图包](../kbs_draft_20261012/README.md)。旧外部融合表、效率表和机制归因说明的限制见新包作者说明。

# PAER 论文实验表（paper/tables/）

本目录由 `regenerate_tables.py` 从仓库原始产物重新汇总生成，每张表输出三种格式：`.csv`、`.md`（Markdown）、`.tex`（LaTeX 三线表）。数据来源路径、配置、种子、split 与统计方式见 `sources.json`。

## 一、正文 6 张核心表

| 编号 | 文件前缀 | 内容 | split | 状态 |
|---|---|---|---|---|
| Table 1 | `table_1_datasets_protocol` | 数据集与协议（来源/目标/频率/context/horizon/窗口数） | — | 完整 7/7 |
| Table 2 | `table_2_internal_retrieval_main` | 内部检索主结果：BASE/SHAPE/Raw Expert/PAER 的 TEST MSE（mean±std）+ RI + 胜出种子 | TEST | 完整 7/7 |
| Table 3 | `table_3_equal_expert_control` | 等专家数量对照：PAER vs RAW_ALL vs RANDOM_K（VAL MSE） | VAL | 完整 7/7 |
| Table 4 | `table_4_knowledge_optimization` | 知识优化过程：初始/最终专家数、各轮池大小、文本/条件变化、删除、机制替换 | TRAIN | 完整 7/7 |
| Table 5 | `table_5_expert_cases` | 专家评价与修订案例（精选保留/修订/剔除） | TRAIN | 完整（精选 9 例） |
| Table 6 | `table_6_coverage_failure` | 覆盖与失败诊断（重点 Electricity/Traffic） | TEST(事后) | 部分（traffic 平均激活专家数未记录） |

## 二、附录表

| 编号 | 文件前缀 | 内容 | 状态 |
|---|---|---|---|
| A1 | `appendix_a1_mae_seed` | 完整 MAE 与种子级结果（5 臂 × 3 种子 × 7 数据集） | 完整 |
| A2 | `appendix_a2_reward_alignment` | 训练 pool-aware reward 与 VAL 留一贡献对齐 | 4/7（etth2/weather/exchange_rate 缺 val_contribution） |
| A3 | `appendix_a3_topk_sensitivity` | 检索 Top-K∈{1,3,5,10} 敏感性 | 4/7（weather/solar/electricity 未完成） |
| A4 | `appendix_a4_efficiency` | 效率与优化成本 | 完整 |
| A5 | `appendix_a5_external_fusion` | 外部模型 + Chronos-2 融合 | 部分（Chronos-2/Ours、RAF solar/electricity 未完成） |
| A6 | `appendix_a6_v1_score_control` | 旧 v1 评分选池（历史控制身份） | 完整 |

## 三、统一统计口径

- **TRAIN 反馈 / VAL 对照 / TEST 结果分别标注**，本目录表注已逐表标明，禁止混用。
- 正式模型种子 `seeds=[42, 123, 2026]`（已核对日志与 result.json）。
- 标准差 `ddof=0`（总体标准差）。
- 相对改善 `RI = (对照MSE − PAER MSE) / 对照MSE × 100%`，用未舍入 MSE 计算。
- 7 数据集汇总 RI = 各数据集 RI 的**等权平均**，不平均原始 MSE。
- 区分「模型种子标准差」（3 种子间 std）与「随机子集标准差」（RANDOM_K 5 个控制子集间 std）。
- 专家数量与检索历史片段数量（Top-K=5）是不同字段，不混用。
- 留一贡献对应当轮池组成（屏蔽该专家嵌入+激活后**重训**，不删除槽位）；**不**宣称最终合并池已逐专家重验。
- 保留全部负结果；缺失值标「未完成」/「未记录」；n=3 不写「显著提升」。

## 四、核验结论（相对论文摘要数字）

| 摘要数字 | 正式产物核验 | 一致 |
|---|---|---|
| v2 vs Raw 胜 **5/7** | 5/7 | ✅ |
| mean RI vs Raw **+13.40%** | +13.4002% | ✅ |
| median RI vs Raw **+17.71%** | +17.7068% | ✅ |
| v2 vs v1 胜 **6/7** | 6/7 | ✅ |
| mean RI vs v1 **+17.77%** | +17.7736% | ✅ |

## 五、支持的结论（仅以下可声称）

- Pool-Aware PAER 相对 Raw Expert 检索在 **5/7** 数据集、mean RI **+13.40%**、median **+17.71%** 提升（TEST，冻结一次性）。
- PAER 相对 v1 在 **6/7** 数据集提升；VAL→TEST 方向 **7/7** 一致。
- PAER 修复 v1 在 electricity 的 pool collapse（8→1 修复为 5）。
- 等 K 下 PAER 选池优于随机子集 RANDOM_K（6/7）；优于 v1-score 选池 V1_SCORE_TOPK（5/7）。
- Electricity/Traffic 上 PAER 不优于 Raw（−15.5% / −0.4%），如实报告。

## 六、不可声称的结论

- 不声称 SOTA（外部深度模型 baseline 见 A5，Chronos-2 融合未完成）。
- 不声称 PAER 在**所有**数据集一致优于 Raw/v1/随机选池。
- 不声称 seed 级统计显著（n=3）。
- 不把现有四臂（BASE/SHAPE/Raw/PAER）比较解释成 pool-aware reward 的**独立贡献**证明（见 `mechanism_control_check.md`）。
- 不把 v1 评分（A6）或 Learned Critic 包装为成功方法。

## 七、待补实验（缺失清单）

见 `mechanism_control_check.md`。数据层缺口：
1. `val_contribution` 仅 4/7（缺 etth2/weather/exchange_rate）。
2. Top-K 敏感性仅 4/7（缺 weather/solar/electricity）。
3. traffic 逐窗口平均激活专家数未记录。
4. Chronos-2/Ours Fusion 及 RAF solar/electricity（后台续跑中，跑完回填 A5）。

## 八、重新生成

```bash
E:/Miniconda3/envs/myenv/python paper/tables/regenerate_tables.py
```
