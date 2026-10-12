> 本报告为论文写作前的审计快照。后续中文稿与图表交接见 [论文包](../../../paper/kbs_draft_20261012/README.md)。路径显示已归一化，哈希值未改。

# PAER 结果审计 2026年10月12日

## 审计结论

当前最新证据是本机的 PAER-Aligned 四底座矩阵，不是 GitHub 中截至10月11日的原始平均融合表。缓存复算确认 84 个 dataset×backbone×seed 条目、28 个组合，25 个组合均值改善。这个结果支持固定插件的经验加装收益；不能单独证明专家优化或池感知奖励的独立贡献，也不构成统计显著性结论。

本轮没有修改实验模型、专家池、标签、预测、权重或历史表格；没有重训或重跑 GPU 前向。仓库组织只更新文档入口并添加审计脚本、结果注册表和审计产物。论文安排尚未展开。

## 范围与证据强度

- Git HEAD：3b30a1a414d06ab085a55b29efb16bfeb5cc6153。最新10月12日实验脚本及若干报告为本机未跟踪文件/被忽略结果，不能把 HEAD 当成这些产物的完整版本号。
- 独立编写的脚本不导入实验模块，不执行其副作用；从预测数组、series.csv、window_indices.npz 重新计算。
- 已核验交付包232个文件哈希，以及包中已有的7份专家池和3个示例checkpoint来源哈希；另建立63份当前矩阵checkpoint的完整路径和SHA256清单。后者是当前盘点，不能反向证明生成缓存时的checkpoint身份。
- 七个数据集的 TRAIN/VAL/TEST 索引均与配置连续分区一致；VAL/TEST缓存标签逐元素等于原序列从 s+L 提取的目标，缓存索引一致，外层完整窗口隔离通过。
- 84组矩阵的形状、有限值、底座MSE、记忆MSE、VAL最优λ、融合MSE、相对改善及28组均值/样本标准差复算一致。
- 21个旧记忆TEST MSE与冻结逐种子结果在严格数值容差内一致。原始平均融合旧表的小差异来自float32乘法先于float64求和；保留原操作顺序可复现，不属于结果错误。
- 三个数据集×三个seed×Raw/PAER×VAL/TEST的36份检索记录，已从原始序列和Top-K索引重建原始/对齐预测，核对查询索引及候选未来可用性。
- 只完成缓存与源码层面的审计；未证明所有训练运行历史、协议决策时间、跨设备复现、LLM生成复现或统计推断有效性。自动检查全通过不表示方法学问题全部通过。

## 最新插件矩阵

指标为每组合三个seed的均值MSE之比所算RI；跨数据集等权平均。标准差采用ddof=1。Chronos-2底座三个seed共享同一预测，不能视作三次独立底座实验。

| 底座 | 改善组合 | 七数据集平均RI |
|---|---:|---:|
| Chronos-2 | 6/7 | +6.2716% |
| PatchTST | 7/7 | +4.1393% |
| iTransformer | 6/7 | +1.9338% |
| RAFT | 6/7 | +7.0357% |

负结果保留：Weather＋Chronos-2约−8.80%；Exchange Rate＋iTransformer约−4.74%；ETTh1＋RAFT约−0.82%。25/28组合并非28个独立数据集，四底座共享相同数据、记忆和测试区间。不得以25/28直接推导显著性。

## 需要处理的问题

### F01 结果协议并存且文档过期

至少三个不同对象应分别命名：原始平均内部检索（Raw对PAER，平均RI 13.4002%）；原始平均Chronos融合（10月11日逐seed λ，平均RI约4.86%，ddof=0）；均值对齐四底座插件（10月12日矩阵，Chronos平均RI 6.2716%，ddof=1）。它们不能合成一列。旧共用seed42 λ表另有5.56%的平均改善，只能作历史协议参考。

paper/tables/appendix_a5_external_fusion.md仍写未完成，EXTERNAL_BASELINE_FINAL_TABLE.md仍使用旧融合及部分旧底座指标。HANDOFF原文也落后于本机矩阵。本次通过新入口与结果注册表明确版本；历史文件保留。

### F02 RAF仍存在目标时刻错位

experiments/external_baselines/raf_rolling.py直接用正式窗口起点s，历史series[:s]、标签series[s:s+H]；正式协议标签为series[s+L:s+L+H]。其泄漏断言并不检查跨方法标签一致。RAF还使用扩展历史，与固定TRAIN bank的信息范围不同。

处置：已有RAF数字不得进入同协议排名；标为待修正重跑。RAF与本次矩阵中的RAFT是不同方法，不能混淆。未在这次整理中改变RAF计算代码或覆盖历史结果。

### F03 旧基线评价与完整窗口结果不同

最新PatchTST脚本明确使用shuffle=False、drop_last=False，并从原checkpoint重新前向。ETTh2旧约14.5019（3200窗）与完整3293窗的14.7618不能混用；Traffic旧约0.00149867与完整3318窗的0.00146160也不同。当前矩阵缓存形状与完整目标窗一致。

RAFT矩阵入口加载checkpoint.pth并按index恢复VAL顺序。源码支持这些修正已被实现；本轮未重跑前向来独立证明每个缓存均由这些checkpoint生成。当前63份checkpoint已全部登记，外部数据adapter哈希亦记录。

### F04 加装收益不等于专家优化收益

对齐融合下，同协议Raw Expert对PAER三数据集配对复算：

| 数据集 | Raw融合MSE | PAER融合MSE | PAER相对Raw |
|---|---:|---:|---:|
| ETTh1 | 7.8369134 | 7.8927134 | −0.7120% |
| Weather | 126.9269942 | 140.5641691 | −10.7441% |
| Traffic | 0.0012960443 | 0.0012906066 | +0.4196% |

Raw指初始专家池，不是未经均值对齐的预测。矩阵仅对底座与底座＋PAER-Aligned；缺乏完整七数据集、四底座的Raw与简单检索对照，因此不能独立归因于专家优化。

### F05 奖励评价与部署聚合不一致

train_arm训练使用softmax候选、均值平移；奖励train_eval仍调用原始未来Top-K均值；新插件部署使用均值平移Top-K。均值平移在新插件的推理侧已经补上，因此不能笼统说“新版推理仍未对齐”。但奖励评价与插件部署仍不一致，softmax随机候选训练与Top-K推理也仍有差别。

三个配对数据集的对齐记忆MSE均为PAER高于Raw。这个现象支持协议敏感性，不能证明“聚合错位就是主要原因”；需控制协议重新评价才能归因。旧PAER_CORE_DIAGNOSIS.md中的“主因已证实”表述应降级为待检验解释。

### F06 池感知评分对照未固定知识版本

run_pool_size_control.py的eval_subset使用raw_emb/raw_act，PAER使用eval_fresh(v2_pool)。V1_SCORE_TOPK与PAER既改变筛选来源，也改变知识版本。mechanism_control_check.md称“唯一差异是评分机制”不成立。已有5/7胜出可作为组合流程比较，不能用作奖励单因素消融。filter-only和revise-keep-all仍缺。

### F07 内部时间反馈和最终池缺口仍在

奖励折按相邻窗口起点切分、使用全TRAIN统计；最终bank/fit边界在最早fit预测时有H−1个bank候选的目标末端尚不可用（H=96时95个、H=24时23个）。这是内部代理评价/训练时间可用性的限制；外层TEST隔离已通过，不能混称TEST泄漏。

KEEP专家退出后续轮次；最终池为各谱系最近正奖励版本合并，没有最终组合池留一复评。因此不能声称所有最终专家均在最终池中有正边际贡献。

### F08 统计文字比实际算法更强

statistics_report.md写“paired block bootstrap”，但diagnose_v2_failure_electricity_traffic.py使用delta[rng.integers(...)]逐窗口独立抽样，并直接调用Wilcoxon；并未保留时间块。步长1的重叠目标高度相关。旧p值、COMPARABLE等效性说法、显著性和“覆盖下降导致退化”的因果表述均不能直接沿用。

本轮报告描述性指标，不补做事后显著性来提升结论等级。后续统计方案需处理时间依赖、种子层次、多重比较，并明确哪些测试区间已参与探索。

### F09 数据重建入口与冻结数据不一致

冻结Traffic的series.csv逐元素等于data/traffic.csv的Sensor_000；配置也声明Sensor_000。但prepare_multidataset_experiments.py仍生成862传感器均值（与冻结目标最大绝对差约0.37045）。直接重跑此旧准备脚本会改变任务。

处置：冻结series/config/window_indices作为当前协议输入；准备脚本标为历史入口。后续应另建版本化重建脚本，并核验数据来源与目标身份；Weather、Exchange、Solar、Electricity更上游物理变量/汇聚来源仍需补足，当前OT列一致并不解决上游来源问题。

### F10 源码和可复现材料未完整纳入版本控制

当前Git HEAD缺少10月12日矩阵生成脚本，新结果目录被.gitignore忽略。交付包原provenance只有3个示例checkpoint，并非全部63个。部分诊断和记忆依赖另一个Codex工作区的绝对路径。

本轮新增完整checkpoint清单、输入哈希、可传repo/out参数的审计入口，保留大型缓存路径，选择性开放小型结果文档的版本控制资格。没有commit、push或删除缓存；待用户复核后再决定发布哪个快照。

## 审计后的使用边界

可以引用数值事实：最新矩阵25/28均值改善、三个负结果、原始平均内部13.4002%、三数据集Raw配对。应注明同协议、完整窗口、逐seed VAL选择及探索性质。

暂不支持的结论：显著提升；池感知奖励独立有效；文本语义不可替代；插件普适；25/28独立重复；所有未来实验均为盲测；RAF同协议排名；聚合错位已经被证明是失败主因。

优先处理顺序：统一版本和数据重建入口；修正RAF并独立重跑；补齐对齐协议下Raw/简单检索对照；再设计奖励/修订/筛选的控制实验与时间隔离评价。此处为审计处置顺序，不是论文结构或投稿承诺。

## 复核材料

- audit_summary.json：自动核验计数及汇总。
- checks.csv：每项判定、容差与差值。
- matrix_per_seed.csv / matrix_summary.csv：从数组独立复算的新矩阵。
- protocol_checks.csv：窗口数量、长度与内部边界诊断。
- raw_paer_pairs.csv：三数据集Raw/PAER、两种聚合的配对指标。
- checkpoint_inventory.csv：63份当前checkpoint路径及哈希。
- input_hashes.json：本轮读取的输入哈希。
- scripts/audit_results.py：不训练、不前向、不修改来源的复算入口。
