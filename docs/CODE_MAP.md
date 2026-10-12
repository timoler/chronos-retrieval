# 代码导航

| 用途 | 入口 | 注意事项 |
|---|---|---|
| 独立结果复算 | scripts/audit_results.py | 读取缓存；显式--repo与--out；无GPU前向 |
| 最新插件矩阵融合 | experiments/external_baselines/run_paer_backbone_matrix.py | PAER-Aligned；部分记忆依赖历史Codex工作区 |
| 最新底座前向 | experiments/external_baselines/infer_patchtst_matrix.py、infer_itransformer_matrix.py、infer_raft_matrix.py | 本地外部仓库和checkpoint；前向会写实验目录 |
| 双聚合检索 | experiments/external_baselines/etth2_agg_alignment.py | retrieve_dual输出同Top-K下原始/均值对齐预测 |
| 原始平均融合 | experiments/external_baselines/run_ours_fusion_v3.py | 原聚合、逐seed λ；不是新版对齐插件 |
| 原始平均融合核验 | experiments/external_baselines/verify_fusion_results.py | 默认写历史目录，审计时请用独立--out |
| 奖励和专家修订 | experiments/run_reward_feedback_critic_v2_poolaware_pilot.py | 原始平均奖励、内部时间折限制、最终池未重验 |
| 条件检索模型 | experiments/run_conditional_expert_mlp_v4.py | 固定文本embedding×激活→可训练投影 |
| 等数量对照 | experiments/run_pool_size_control.py | Raw子集与修订PAER混合比较，不是评分单因素 |
| RAF适配 | experiments/external_baselines/raf_rolling.py | 已知目标错位，待独立修正 |
| 旧数据准备 | experiments/prepare_multidataset_experiments.py | Traffic为传感器均值；不匹配冻结Sensor_000 |

实验路径本次保留，避免破坏硬编码导入、缓存依赖和可追溯性。历史代码继续置于既有archive；新增导航不改变运行语义。大型缓存不得用git clean清除。
