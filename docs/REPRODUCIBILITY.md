# 统一协议复现

唯一协议为 configs/paer_unified_v1.json。新运行需使用独立空目录，输入包括 data/processed/ 下七数据集的 series.csv、config.json、window_indices.npz，以及 data/experts/ 下的 Raw8。

复制 configs/run_paths.example.json 到机器本地，填写协议与实际路径。需要 CUDA、Chronos-2 和 MiniLM 本地模型，以及 PatchTST、iTransformer、RAFT 的外部架构源码。大型权重和 checkpoint 不随 Git 发布；外部源码身份以运行中的 SHA256 清单为准。

先运行 initialize.py --config <本地配置路径> 建立新清单，再把 PAER_RUN_CONFIG 指向初始化生成的 run_config.json，以 runner.py --phase backbones 启动底座阶段。完整运行使用 --phase all；专家池阶段必须先通过真实 LLM 调用预检。初始化不会生成 LLM 就绪标记，也不会训练或覆盖任何旧结果。

代码、输入与外部架构清单在每个任务启动前校验。所有 VAL 选择、预测和 checkpoint 冻结后才读取 TEST。verify.py 从原始数组重算指标和 Top5 预测，通过后才登记正式结果。重算缓存指标与独立重新训练须分别报告。

现有运行使用整理前的独立快照，见 unified/execution_snapshot/IDENTITY.json。清理后的代码是组件提取与路径整理；现有产物必须标记真实执行快照，不能改归属。Git 文本换行可能影响字节哈希，新机器应建立新清单并保存输入差异证据。

三 seed 表示监督底座与检索器的随机性；Chronos、确定性 Shape 和相同专家池预测缓存允许共享。专家池不是三个独立 LLM 样本。此前 TEST 已被观察，本轮属于统一协议下的探索性重跑。

组件提取的 CPU 等价检查可用 tests/verify_core_extraction.py --reference-root <整理前3f45a01源码目录> --work-dir <全新临时目录> 重做。它检验固定输入的计算一致性，不等同于完整 GPU 复现实验。

公开执行源码可由 execution_snapshot/RECONSTRUCT_EXECUTION.patch 基于公开提交 5ce382b 重建（路径相对 docs/unified/）；随后运行同目录 restore_source_bytes.py --source-root <重建目录> 恢复已记录的LF/CRLF差异。118项源码哈希已逐项验证。原本地执行提交号用于本地追溯，不作为已发布远端分支；机器绝对路径已替换为逻辑根目录。
