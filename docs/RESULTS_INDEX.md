# 结果版本注册表

更新时间：2026-10-12。按评价对象与聚合规则选结果，不按文件名里的final或v编号推断权威性。

| ID | 评价对象 | 源目录 | 状态与用途 |
|---|---|---|---|
| internal-original | Raw vs PAER原始未来Top-5均值 | results/reward_feedback_critic_v2_final_test | 冻结内部实验，7集；5/7改善，平均13.4002%；非融合结果 |
| chronos-original-shared-lambda | Chronos＋原始平均记忆，共用seed42 λ | results/paper_completion/phase1_external_baselines/EXTERNAL_BASELINE_FINAL_TABLE.md | 历史协议，约5.56%；含旧基线/RAF不可直接排名 |
| chronos-original-per-seed | Chronos＋原始平均记忆，逐seed λ | results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011 | 已核验旧融合，7集4改善2持平1退化，约4.86%，ddof=0 |
| aligned-backbone-matrix | 4底座＋均值对齐PAER，逐seed λ | results/paer_aligned_backbone_matrix | 当前最新经验矩阵，84项28组，25组改善；ddof=1；本地预测 |
| aligned-package | 上一矩阵的可复算材料快照 | results/deepseek_matrix_delivery | 232文件，数组/汇总哈希通过；外部checkpoint不在包中 |
| raw-paer-paired | 3集Raw vs PAER，两种聚合 | results/paer_core_contribution_audit | 数字已独立复算；原文“主因已证实”因果解释不采纳 |
| plugin-four-combos | ETTh2/Traffic×Chronos/PatchTST | results/paer_plugin_evidence_audit | 四组早期插件证据，已被完整28组矩阵扩展 |
| raf-legacy | RAF历史滚动适配 | results/external_baselines/raf* | 目标时刻未对齐，待修复重跑，非RAFT |
| old-paper-tables | 原稿表格 | paper/tables | 快照未更新；appendix_a5等不得直接作为最新表 |

最新可复核审计快照位于docs/audits/2026-10-12。统计默认报告描述性变化；25/28不代表独立重复的显著性。每次修改聚合、目标变量、checkpoint或窗口范围，需新版本和新的结果目录。
