# PAER 统一协议 v1：四底座完整核心对照

状态：新运行进行中。这里不继承旧实验的完成状态或数值。

- 数据：ETTh1、ETTh2、Exchange、Traffic、Weather、Solar、Electricity，冻结单变量、context/horizon及完整外层索引。
- 底座：Chronos-2、PatchTST、iTransformer、RAFT。RAF不进入本协议。
- 最终seed：42、123、2026；选池seed固定42，LLM路径不因TEST结果重选。
- 核心臂：Backbone、Shape、NoExpertMLP、Raw、MemoryReward、FusionReward、FusionReward_NoCritic、FusionReward_Independent。主PAER臂预先指定为FusionReward；其他臂用于对照，不按TEST择优改名。
- 覆盖：7×4×3×8=672条最终逐seed指标。共享底座/记忆/空池不计为额外独立训练。

## 时间隔离与训练

统一使用TRAIN前50%作为初始历史前缀，后50%分成4个扩展时间折；每个当前块前半CAL、后半EVAL。历史前缀再分bank/fit，bank→fit→CAL→EVAL各边界删除未来标签尚不可用的窗口，CAL/EVAL步长H。原20%前缀在Exchange上不足以满足RAFT自检索非重叠掩码；50%规则在新实验前冻结，并同样用于全部底座。

折内PatchTST/iTransformer/RAFT只在bank训练，在fit选epoch；CAL选融合λ，EVAL算奖励。禁止拿完整TRAIN checkpoint评价TRAIN奖励折。Chronos-2按固定本地预训练snapshot前向。外层监督底座重新在完整TRAIN训练，用官方VAL选checkpoint。每个监督底座100epoch，原架构/Adam/lr规则；不使用TEST训练或选epoch。

检索器保持30epoch、AdamW lr.001/wd.0001、batch64、候选256、固定8槽位、相同seed初始权重与随机抽样。阈值/激活统计只使用当前历史bank/fit输入。所有记忆与奖励使用Top5等权均值平移，训练保留softmax松弛；不声称消除了训练/推理差异。

## 奖励与机制控制

MemoryReward基于纯记忆误差，一个共享池应用于四底座。FusionReward分别依据各底座独立折内预测优化；FULL/MINUS各自在CAL选λ，再比较EVAL融合误差。KEEP留在后续整池，最多两次修订；最后逐轮重评最终池、按固定规则删除未通过者，允许空池，并明确与NoExpertMLP对应。

NoCritic臂只移除单独Critic调用，保持数字奖励、LLM修订、预算及最终筛选。Independent臂用单个专家对无专家检索器的改善替代整池留一奖励，固定架构/8槽位/其余流程；不将其通过解释为整池正边际贡献。

所有底座/臂/seed分别在完整VAL用固定网格[0,.05,.1,.15,.2,.3,.5,.7,1]选λ。全部选择、专家池和checkpoint冻结后，再统一TEST读出。保留MSE、MAE、ddof0及ddof1标准差，正文默认ddof1，负结果与空池全部报告。

## 当前入口与历史边界

代码：`experiments/unified/`；机器可读协议：`configs/paer_unified_v1.json`；进度：`docs/unified/STATUS.json`。

旧共用λ、原值聚合、无purge奖励、旧池参考、RAF及旧Gate/Switch/loss试验不进入此次运行队列或主表。历史文件由 Git 历史追溯，已退出当前分支，不能当作新协议结果。此前TEST已被观察；本轮是统一的探索性重跑，不包装成新的盲测。未做统计显著性检验时只报告描述性变化。

大模型权重、NPZ、逐epoch日志保留独立运行目录；仓库更新协议、代码hash、进度、最终逐seed表和独立核验表。不提交凭据。每个完成阶段更新状态，未完成部分明确pending。
