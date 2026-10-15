# 当前代码结构

- experiments/unified/common.py：运行身份、输入及严格时间划分。
- experiments/unified/backbone.py：四底座拟合与预测。
- experiments/unified/reward_core.py、pools.py：TRAIN 折内反馈、独立融合奖励、Critic 和专家池维护。
- experiments/unified/final.py：完整 VAL 选择与冻结后的 TEST 读出。
- experiments/unified/unified_retrieval.py：Top5 等权均值平移及检索索引保存。
- experiments/unified/runner.py：可恢复单 GPU 队列及所有 VAL 选择的统一冻结。
- experiments/unified/verify.py：从保存的预测与标签重算指标和融合选择。
- experiments/unified/initialize.py：为全新输出目录建立运行清单，不启动训练。
- paer_core/：从已审查实现提取的实际复用组件；不包含旧实验命令入口。

数据在 data/processed/；原始 Raw8 在 data/experts/；输入迁移相对整理前 Git 快照保持字节一致；与现有运行输入的文本换行可能不同，数值、索引和专家内容已独立核对一致，新运行须重建字节清单。data/ 下的原始 CSV 和其他既存数据输入保留。它们不代表当前协议扩大到更多数据集。

来源行号、文件哈希和组件等价检查见 docs/unified/。正在执行的实验快照另见 execution_snapshot/IDENTITY.json。
