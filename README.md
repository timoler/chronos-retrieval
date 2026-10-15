# Chronos-2 + PAER

当前唯一协议为 **PAER 统一协议 v1：七个数据集、四个底座、三个 seed、八个核心臂**。新实验正在运行；最终结果尚未产生。

- [统一协议](docs/unified/PROTOCOL.md)：时间划分、折内奖励、逐 seed 融合选择及测试冻结。
- [进度快照](docs/unified/STATUS.json)：请检查 updated_utc，文件时间不代表实时在线服务。
- [运行与复现](docs/REPRODUCIBILITY.md)、[代码结构](docs/CODE_MAP.md)。
- [结果登记](docs/RESULTS_INDEX.md)、[整理记录](docs/unified/CLEANUP.md)。

底座为 Chronos-2、PatchTST、iTransformer、RAFT。主方法预先指定为 FusionReward；各底座独立计算融合奖励。所有 λ 只用对应 VAL 选择，保留负结果、退化和空专家池。

当前分支已移除旧协议运行脚本、旧消融汇总、旧表图和旧论文稿件。旧成果可从 Git 历史追溯，未作为本轮统一实验的结果。原始数据、固定窗口和 Raw8 输入继续保留。

本次整理没有切换正在执行的代码：既有运行继续使用独立冻结快照。整理后代码的来源及等价检查单独记录，不能把该运行误标为由整理后的提交训练。
