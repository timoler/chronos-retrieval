# 当前版本整理记录

旧实验入口、旧奖励/门控试验、旧结果表与日志、旧论文和图、历史文件副本已从当前分支移除。它们仍可在 Git 历史中追溯。保留原始数据；52个输入文件迁移到 data/processed/ 和 data/experts/，另一个混在输入目录中的旧实验日志已剔除。

实际复用的组件提取为 paer_core/，删除旧 priors/、retrieval/ 和旧 experiments 脚本。50段定义保留原AST及计算顺序，来源行号和哈希见 paer_core/DERIVATION.json。CPU固定输入314项检查通过，覆盖模型初始化、专家编译和激活、编码与打分、微型优化、Top5和预测、奖励规则与底座参数；这不是完整GPU复现实验。

现有实验继续使用独立运行快照。清理期间发现Exchange日志原子替换的Windows拒绝访问，单独加入有限重试并记录 IO_RECOVERY.json；25项重试控制流检查通过。已完成26项及28个原始输入保持原哈希，失败的未完成训练产物另存。执行源版本及清单见 execution_snapshot/IDENTITY.json，禁止将其改归属到整理后的源码。

清理后仓库的源码清单见 CURRENT_CODE_MANIFEST.json。初始化验证在全新临时输出目录使用真实七数据集和116个外部Python源码完成，没有启动额外GPU训练或声明LLM就绪。

输入迁移相对整理前Git快照字节一致；与现有E盘运行输入有文本换行差异，数值、索引和Raw8语义已核对一致。新运行必须重新冻结字节。INPUT_MIGRATION_CHECKS.json 是清理过程中53项的只读检查，最终保留52项见 INPUT_MIGRATION.json。

2026-10-15恢复时，原生Git推送预检和真实LLM文本预检均已通过。状态与时间见 BLOCKERS.json、LLM_PREFLIGHT.json 和 STATUS.json；已有31项底座产物通过独立完整性核验。恢复传输不代表实验已完成。

公开执行源码可由 execution_snapshot/RECONSTRUCT_EXECUTION.patch 基于公开提交 5ce382b 重建（路径相对 docs/unified/）；随后运行同目录 restore_source_bytes.py --source-root <重建目录> 恢复已记录的LF/CRLF差异。118项源码哈希已逐项验证。原本地执行提交号用于本地追溯，不作为已发布远端分支；机器绝对路径已替换为逻辑根目录。
