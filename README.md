# chronos-retrieval

本仓库研究专家知识优化与历史检索预测。当前最新实验为 **PAER-Aligned 四底座矩阵**；原始平均内部检索和旧Chronos融合保留为独立历史协议。

## 从这里开始

- [中文论文初稿与同门绘图交接](paper/kbs_draft_20261012/README.md)：完整中文稿、最新版表格、5 张主图与 2 张补充图任务书。

- [结果审计与限制](docs/audits/2026-10-12/REPORT.md)：2026-10-12缓存复算、协议检查和待处理项。
- [结果版本注册表](docs/RESULTS_INDEX.md)：每套结果的用途、版本、源目录和可支持结论。
- [复核与复现](docs/REPRODUCIBILITY.md)：只读复算入口与本地缓存需求。
- [代码导航](docs/CODE_MAP.md)：当前入口、历史脚本与外部依赖。
- [原项目README](docs/history/README_before_20261012.md)：早期研究流程。

## 当前证据

本机最新矩阵覆盖4底座×7数据集×3seed，共84项，28组中25组平均MSE降低。其数值已从缓存复算；不等于统计显著，不独立证明池感知奖励/专家优化的贡献。负结果与Raw对照见审计报告。

方法名称请区分：Raw Expert初始专家池、PAER优化专家池、PAER-Aligned均值对齐记忆、底座＋PAER-Aligned融合。`v2/v3`是脚本版本，不作为统一方法名。

## 目录

| 目录 | 内容 |
|---|---|
| experiments/ | 实验脚本；部分为历史诊断，按代码导航选择 |
| retrieval/ | 通用检索组件 |
| results/ | 冻结结果与本地预测缓存；见结果索引，保留原路径 |
| docs/ | 当前审计、结果注册表、复现指南与文档历史 |
| scripts/audit_results.py | 独立缓存审计，不训练或运行GPU前向 |
| paper/tables/ | 历史论文表，使用前与结果注册表核对 |
| archive/ | 既有历史代码 |

冻结输入为results/multidataset中的series.csv、config.json、window_indices.npz。旧prepare_multidataset_experiments.py的Traffic聚合规则不匹配当前Sensor_000任务，勿将其当成当前数据重建入口。现有RAF目标窗口错位，暂不用于同协议排名。

大型预测缓存仍为本地材料；仅clone仓库不保证可直接复算最新矩阵。审计命令、环境与材料边界见复现指南。仓库整理不改变历史实验产物。
