> 用户于2026-10-15明确要求继续跑。已核验并恢复执行快照，完成标记自动跳过；以 STATUS.json 的实际状态为准。

# 统一协议交接

用户已授权：七数据集 × 四底座的完整核心对照；融合奖励按各底座独立计算；统一协议；清理旧版本；更新仓库。主表目标为 672 条逐 seed 指标。

唯一入口为 experiments/unified/runner.py，协议为 configs/paer_unified_v1.json。组件库为 paer_core/。旧实验脚本、旧表格和旧稿件已退出当前分支，Git 历史仍可追溯。

现有运行前26项使用提交 3f45a01c8e1c4003221c82bed00f5ce0e69df2cb 的独立快照。Exchange首项在日志原子替换时发生 Windows PermissionError，失败产物已保存。只对 JSON 写入增加有限重试，代码提交为 e53736ad11545075c537318ee5aacb8266e74b5f，控制器已从磁盘重启并加载完整冻结检查。训练计算、协议和输入未改变，已完成26项保持原样。执行清单及分段来源见 docs/unified/execution_snapshot/；不能拿这些清单验证整理后的目录。

2026-10-15恢复：31项已完成产物独立核验全部通过，队列从未完成任务继续。LLM真实文本预检已通过，工具使用数为0，就绪证据见 docs/unified/LLM_PREFLIGHT.json。原生Git推送预检通过；最新远端同步以运行发布记录为准。连接器旧403记录不代表原生Git无法写入。

旧代码清单漏列间接导入的 priors 文件，属于冻结范围缺口。清理后的依赖直接纳入 paer_core/ 并完整建清单；这不能追溯性地宣称旧快照已完整冻结。

仓库整理不改写现有运行或原始输入；日志重试修复已独立记录。新目录和新源码必须建立独立运行清单，不能覆盖原预测或改写已完成结果的源码身份。

公开执行源码可由 execution_snapshot/RECONSTRUCT_EXECUTION.patch 基于公开提交 5ce382b 重建（路径相对 docs/unified/）；随后运行同目录 restore_source_bytes.py --source-root <重建目录> 恢复已记录的LF/CRLF差异。118项源码哈希已逐项验证。原本地执行提交号用于本地追溯，不作为已发布远端分支；机器绝对路径已替换为逻辑根目录。
