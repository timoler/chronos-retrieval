# PAER 论文绘图交接要求

版本：2026-10-12。计划 **正文 5 张图、补充材料 2 张图，共 7 张**。这是一份作图任务书，不表示图已经完成。先完成图 1、图 2 的结构确认，再画图 3、图 4、图 5，最后制作补充图。初稿正文与图注见 [中文稿](manuscript_zh.md)，数据从本目录 `figure_data/` 读取。

文章要讲清楚：专家知识提出可观测的状态假设，PAER 用预测反馈评价和修订这些假设，两者形成知识增强检索流程；冻结记忆分支可以补充不同预测底座。现有数据支持完整插件的经验收益，不支持把每个环节都画成必然正向增益。图中保留负结果和实际实现边界。

## 一 共用规范与交付方式

以下尺寸是本项目的绘图工作尺寸，不宣称已经核验为 KBS 的强制格式。投稿前由通讯作者再核对当期官方要求。

- 双栏宽图按 180 mm 制作，单栏图按 88 mm；图 1、2、3、4、5 均建议双栏。各图建议高度见后文，允许为阅读性微调。
- 图内使用英文，交接说明用中文。面板标号 a、b、c 放在左上角，位置一致。最终尺寸下正文标签建议 8–9 pt，任何文字不得低于 5 pt；轴标题与图例可用 8 pt，面板标号 10 pt。
- 白底、黑色文字，不用渐变阴影、3D、背景装饰或拟人化大脑图标。数值分支用蓝色 `#0072B2`，知识分支用青绿 `#009E73`，反馈/修订用橙色 `#E69F00`。颜色之外使用文字或线型编码。
- 结果热图采用以零为中心的蓝白橙发散色：蓝表示改善，橙表示退化；全图同一对称色阶。不要只靠红绿区分正负。正负定义必须写在图例。
- 机制图用矢量形状和可编辑文字；统计图由数据文件驱动。不得手工改数、平滑三种子点、补齐未完成数据或添加显著性星号。
- 主矩阵及配对表采用三种子样本标准差 ddof=1；原始内部实验、旧 Top-K 表沿用 ddof=0。不要混成同一种误差条；如果没有逐点不确定性，就不画误差条。
- 同一横排的同等面板使用相同绘图区宽度、轴高与间距，图例不能挤压某一面板。数值标注不能压在点、线或误差条上。
- 交付：`fig01_framework.pdf` 至 `fig05_expert_lineage.pdf`，`figS01_lambda.pdf`、`figS02_topk.pdf`；每张另交可编辑源文件（SVG/AI/PPTX 或绘图源代码）、PNG 预览、所用 CSV、字体信息和 README。文字必须保持可编辑，PDF 嵌入字体。
- 代码绘图的语言由作图者选定并保持一致。本次只交需求和数据，未替同门选定 Python/R 或生成统计图。
- 验收时检查最终物理尺寸、全部文字、跨面板对齐、裁切、遮挡、图例、正负号、来源与脚本复现。交图时提供最终 PDF 的文字和碰撞检查记录；不能用一张屏幕截图代替源文件。

## 二 图 1 整体框架

**要回答的问题：** 专家知识、PAER 反馈、历史检索和预测底座如何协同工作？

**类型与尺寸：** 两阶段机制图，180 × 110 mm 左右；上方为 Offline knowledge refinement，下方为 Frozen forecasting。

上半部分按左到右排列：`TRAIN observations` 与 `Initial expert pool` → `Executable conditions + frozen text embeddings` → `Pool-aware evaluation` → `Critic / Expert revision`。评价模块与修订模块之间画反馈环，旁边注明 `up to 2 revisions`。输出接到 `Frozen expert pool`。训练数据仅连到离线评价与训练模块，不要画 TEST 标签进入任何训练或修订框。

下半部分分两条平行支路：

1. `Query context` → `Forecasting backbone` → `Base forecast`。底座框下标出 `Chronos-2 / PatchTST / iTransformer / RAFT`，斜杠表示可选底座，不是四模型同时集成。
2. `Query context` 与 `Frozen expert pool` → `Conditional retriever`；`TRAIN memory bank (context, future)` → 检索器 → `Top-5 histories` → `Mean-aligned aggregation` → `Memory forecast`。

两条输出连到同一融合结点：`(1−λ) × base + λ × memory` → `Final forecast`。独立的小块 `VAL only` 以虚线连到 λ，标明 `grid-selected, then frozen`。角落写 `No LLM calls during forecasting`，不要写 `zero overhead`。

**必须准确：** 原始专家池为 8 条，最终池为 5–7 条；不要把最终池画成固定 8 条。均值对齐发生在被选历史未来上，不能画成对 TEST 真值对齐。查询输入使用历史上下文，不包括未知未来。

**来源：** 初稿第 3 节；`PROTOCOL.json`。图中不需要虚构时间序列曲线；若使用示意波形，明确标 `illustrative`。

**图注：** 使用初稿“图 1”图注。该图表达协同设计，不承担模块独立增益的实证结论。

## 三 图 2 池感知反馈与版本归档

**要回答的问题：** 奖励如何由当前池的预测误差产生，保留专家之后如何进入最终池？

**类型与尺寸：** 两面板机制图，180 × 100 mm。a、b 左右排列，建议宽比 1.1:1，但绘图前记录这是有意的非等宽布局。

**a 条件边际贡献。** 上方放一个 `Current evaluation pool P^(r)`，下分两路：完整池与 `Mask expert i`。遮罩只将某专家槽位置零，槽位保留；不要画成删除一个网络分支并减少参数。两路都接 `Retrain retriever`，再接四个查询折误差 `M_f(P)` 与 `M_f(P without i)`。汇合处写公式 `R_i,f = [M_f(P without i) − M_f(P)] / M_f(P without i)`。底部列 `mean / median / positive-fold fraction / coverage`，指向 `Deterministic reward`。不要把贡献叫成精确因果效应或 Shapley value。

**b 修订与归档。** `Current pool` → `Evaluate` 后分两路：KEEP → `Lineage archive`；其他判定 → `Revision` → `Next evaluation pool`。档案不接回下一轮评价池。循环终点从档案和最近评价记录合并到 `Final pool: latest positive record per lineage`。下方明确小字 `Final merged pool is not re-evaluated by the pool reward`。

**附属标注：** 四折属于 `TRAIN proxy evaluation`；不要标 `leakage-free rolling validation`。允许把奖励阈值放在图注而非挤入图内。奖励详情按正文 3.4，不得把中位数条件误写为均值条件。

**来源：** 初稿算法 1；仓库 `experiments/run_reward_feedback_critic_v2_poolaware_pilot.py`。使用初稿“图 2”图注。

## 四 图 3 七数据集四底座的主结果热图

**要回答的问题：** 同一类完整插件能否补充不同预测底座，收益和退化分布在哪些组合？

**尺寸：** 180 × 90 mm 左右，一张热图即可，不加重复柱状图。

**数据：** `figure_data/matrix_summary.csv`，28 行。使用 `ri_pct`，不是将 `matrix_per_seed.csv` 的 RI 直接平均。公式为种子均值 MSE 的相对变化。矩阵行顺序固定为 ETTh1、ETTh2、Exchange Rate、Traffic、Weather、Solar、Electricity；列为 Chronos-2、PatchTST、iTransformer、RAFT。

每格标带符号的百分比，保留两位小数，字号不低于 8 pt。对称色阶建议 −35% 至 +35%，零为白色，全部 28 格使用同一色阶，不截断负结果。色条标题 `Relative MSE improvement (%)`；图下注 `Positive = lower MSE`。列名下可加 `6/7 improved`、`7/7 improved`、`6/7 improved`、`6/7 improved`；跨数据集平均 RI 用一行文字标 6.27%、4.14%、1.93%、7.04%，不要当第八个数据集混入同一热图。

**关键复核值：** Traffic–RAFT +34.60；Traffic–Chronos-2 +23.34；Weather–Chronos-2 −8.80；Exchange Rate–iTransformer −4.74；ETTh1–RAFT −0.82。主结果为 25/28 改善。

**不能表达：** 不能画显著性星号；不能标成“PAER 专家优化贡献”；不能把 28 格当独立实验；不能据此给四底座作跨数据集绝对 MSE 排名。使用初稿“图 3”图注。

## 五 图 4 预测误差互补与失败条件

**要回答的问题：** 记忆分支何时能降低融合误差，为什么误差方向不同仍可能退化？

**类型与尺寸：** 定量三面板，180 × 115 mm。a 在左占约一半宽度，b、c 在右上下排列；右两面板等宽、等轴高。

**a 全矩阵误差方向。** 读取 `figure_data/fig04_error_decomposition_summary.csv`，按图 3 的行列次序画 7×4 热图，字段为 `error_cosine`。定义为三个种子分别计算 C/√(BM) 后的平均。C=mean[(base−truth)(memory−truth)]，B 和 M 为对应 MSE。色阶固定 −1 至 1，中心零。标题 `Normalized error cross-moment`，不得称 Pearson correlation；它未经中心化，不是相关系数。标两位小数，不加显著性。

**b Traffic–Chronos-2。** 在对应 summary 行读取 `base_term_normalized`、`memory_term_normalized`、`cross_term_normalized`，画三个相加分量形成的堆叠柱，并用一条水平虚线标 `Base MSE = 1`。再以细黑点标总值 `fused_over_base`，标数值 0.7666。三项约为 0.2500、0.2269、0.2897。标 `λ = 0.5 for all seeds`。

**c Weather–Chronos-2。** 采用相同做法，三项约为 0.5400、0.2321、0.3159，总值 1.0880，仍以 1 为参照。不得因图需要而改 λ；实际种子 λ 见 `figS01_lambda.csv`。b、c 的 y 轴必须相同，建议 0 至 1.2。分量颜色一致，图例只画一次。

图下写恒等式 `MSE_fused/B = (1−λ)² + λ²M/B + 2λ(1−λ)C/B`。数据文件是先按每个种子的 B 归一化，再平均各项；不要重新用平均 λ 拼凑分量。两个案例底座为共享 Chronos-2，因此总值与主表 RI 精确对应。84 组误差恒等式均已核验。

**样本与误差条：** summary 已经是三种子均值，不附带置信区间；主图不画误差条。若要显示种子差异，可从 per_seed 文件叠加三个总值点，但不要连成时间线。

**解释限制：** 两案例是看过结果后选择的正/负示例，不称随机或预注册样本。图 4 解释底座与记忆输出的组合，不能用来宣称专家知识和 PAER 奖励已被独立因果验证。使用初稿“图 4”图注。

## 六 图 5 知识修订谱系

**要回答的问题：** 专家知识如何在反馈后保留、修订或移除，最终池与各轮评价池有何区别？

**类型与尺寸：** 知识流程记录图，180 × 120 mm；a、b 上下排列，a 占约 35%，b 占约 65%。

**a 七数据集池大小。** 从 `figure_data/table_4_knowledge_optimization.csv` 读取 `Pool size r0/r1/r2` 和 `N final (PAER)`。横轴四个离散位置：`Evaluated r0`、`Evaluated r1`、`Evaluated r2`、`Final merged`；纵轴专家数量 0–8。使用每数据集一行的小型数字矩阵或点阵，不建议七条彩线叠在一起。最终列与前三列之间留间距并加分隔标签，避免读者把最终池当成第四轮评价。ETTh1 应为 8、5、5、7；Weather 为 8、3、2、7。

**b ETTh1 八条完整谱系。** 从 `figure_data/fig05_expert_lineage.csv` 筛选 `dataset=etth1`，纵轴 E1 至 E8，横轴 r0、r1、r2、final。所有八条都展示，不能只画成功修订者。实际有评价记录的位置画形状标记，并以文字/形状区分 KEEP、REVISE、REJECT、UNSUPPORTED。KEEP 后未参与下一轮的位置留空，以虚线直接连到 final 的保留标记，不要补造后续评分。最后一列根据 `final_retained` 显示 retained/removed。

可在旁边以简短文本说明 E1：初始“低波动与高持续性下的热惯性假设”经修订变为“高近期水平条件”；E8 最终覆盖率为 0，标记 UNSUPPORTED。完整文本及条件从 `figure_data/expert_history/etth1/raw_experts.json` 和 `final_expert_pool.json` 读取，若某专家最终被移除则最终文本从 history 的最后记录读取。说明为中文意译，英文图内应忠实翻译，不加入“已证明机制正确”。

**不能表达：** REJECT 不是立即永远删除，修订预算内仍可能被修订；某轮奖励增加不是 TEST MSE 增益；不得将 final 的合并表示为第四轮重新验证。使用初稿“图 5”图注。

## 七 补充图 S1 验证权重

**尺寸与数据：** 180 × 90 mm，读取 `figure_data/figS01_lambda.csv` 的全部 84 行。三块并排的 7×4 热图分别对应 seed 42、123、2026，共享 0–1 的色阶和行列顺序，单元标具体 λ，精度至 0.01。

标签 `Validation-selected memory weight λ`。0 表示底座，1 表示记忆，不是专家激活值。Chronos-2 底座预测虽共享，三个插件种子的记忆与 λ 仍需分别展示。不能将种子作为训练轮次，不画递增趋势箭头。使用初稿“图 S1”图注。

## 八 补充图 S2 原始聚合的 Top-K 敏感性

**尺寸与数据：** 180 × 110 mm，2×2 等宽等高面板，顺序 ETTh1、ETTh2、Exchange Rate、Traffic。使用 `figure_data/figS02_topk_long.csv` 的 16 行；横轴真实数值 K=1、3、5、10，纵轴 VAL MSE。不同任务单位不同，允许不同 y 轴尺度，但标明数值，不跨面板比较绝对高度。

均值连线配 ±1 总体标准差误差条，n=3；不要称 95% CI。K=5 用空心圈或竖虚线标 `fixed main setting`，不能标 `optimal`。数据来自已经排版的旧表，保留原有有效位数，不声称为全精度导出；逐种子原值未随该敏感性表保存到本包。

标题与图注必须出现 `Original aggregation / frozen VAL`。没有 Weather、Solar、Electricity 的已完成结果，不能补画，不能借用新主矩阵的 TEST 数据。使用初稿“图 S2”图注。

## 九 数据字典与复核顺序

| 文件 | 行数或覆盖 | 用途 | 注意事项 |
|---|---|---|---|
| matrix_summary.csv | 28 | 图 3 / 表 2–3 | 均值 MSE 后计算 RI，ddof=1 |
| matrix_per_seed.csv | 84 | 数字追踪 | λ 来自 VAL，TEST 不重选 |
| fig04_error_decomposition_summary.csv | 28 | 图 4 | 归一化分量的种子均值 |
| fig04_error_decomposition_per_seed.csv | 84 | 图 4 追踪 | 交叉矩非 Pearson；全部恒等式通过 |
| fig05_expert_lineage.csv | 116 | 图 5 | 全七集所有已记录轮次；每谱系不一定三轮 |
| figS01_lambda.csv | 84 | 图 S1 | 三种子分别呈现 |
| figS02_topk_long.csv | 16 | 图 S2 | 原始聚合 VAL，旧表精度 |
| raw_paer_pairs.csv | 36 | 表 6 及补充核对 | 包含 original/aligned，两种协议不可混合 |
| expert_history/ | 七集 | 图 5 文本与谱系 | 原始专家、历史评价、最终池 |

数据集代码 `exchange_rate` 显示为 Exchange Rate，其余按正文大小写。`seed` 是模型训练种子；`lambda_val` 是固定融合权重。`error_cosine` 为未中心化误差内积除以范数，`fused_over_base` 为按种子归一化后求均值。

交图前按顺序检查：先确认所用协议与 split，再核对 CSV 数值与图中文字，然后检查完整样本覆盖，最后检查版式、导出和可编辑性。发现数据不一致时，保留原 CSV 并反馈具体行列，不自行修正实验结果。

## 十 暂不制作的图

不制作“各模块贡献占比”饼图、奖励机制独立贡献图、伪造显著性柱状图、未经选样规则说明的预测曲线图，也不制作七数据集统一延迟图。现有证据不足以支撑这些表达。若后续补做同协议的固定知识版本消融或新增独立测试，先更新数据与图注，再确定新增图号。
