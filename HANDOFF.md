> 2026-10-12 论文交接：[中文初稿、表格与绘图要求](paper/kbs_draft_20261012/README.md)。本页其余部分为 10 月 11 日历史记录。

> 2026-10-12 更新：以下为10月11日历史交接。当前最新结果和限制请先读[结果注册表](docs/RESULTS_INDEX.md)与[审计报告](docs/audits/2026-10-12/REPORT.md)。本机已有均值对齐四底座完整矩阵；旧RAF仍存在目标时刻错位。

# HANDOFF — chronos-retrieval 项目交接（2026-10-11）

本文档面向接手者（Codex），说明方法的统一名称、最终代码入口、环境、数据/模型/结果位置、实验完成度、核查结论与已知问题。最终实验口径以本文为准。

---

## 1. 方法说明与统一名称

| 统一名称 | 含义 |
|---|---|
| **Chronos-2** | 单独用 `amazon/chronos-2` 零样本预测，不加入记忆 |
| **原始专家记忆（Raw Expert）** | 未经优化（无 reward/pool-aware）的专家知识，检索历史片段出记忆预测（`raw_experts.json`） |
| **PAER 专家记忆** | 经 Pool-Aware + reward 优化后的专家知识，检索历史片段出记忆预测（`final_expert_pool.json`，即 Pool-Aware v2） |
| **融合模型** | Chronos-2 预测 + PAER 专家记忆预测，按 λ 加权：`fused = (1-λ)·chronos + λ·paer` |

- 「PAER 专家记忆」是方法模块；「融合模型」是接入 Chronos-2 后的完整预测方案。**不要只写「Ours」，必须指明指哪个对象。**
- **λ=0 表示「未采用记忆」**：该数据集最终预测等于 Chronos-2，不计入融合改善。
- 融合 λ 的两种**规则**（不是两套模型）：
  - **旧规则** = seed42 在 VAL 选一个 λ，三个种子共用；
  - **新规则** = 三个种子各自在 VAL 选自己的 λ（**正式采用**）。
- 代码里的 `v2`/`v3` 只是版本号，不当方法名。

## 2. 最终代码入口与目录导航

正式融合（新规则，逐 seed λ）：

- `experiments/external_baselines/run_ours_fusion_v3.py` — 融合模型·新规则的正式实现（逐 seed λ + 落盘预测缓存 + 逐 seed 复现检查 + seed42 共用 λ 参考列）。
- `experiments/external_baselines/verify_fusion_results.py` — 从预测缓存**独立重算** MSE/MAE/λ/融合/复现并生成最终表（不读 v3 的 CSV，供独立核对）。
- `experiments/external_baselines/check_frozen_paer_reproduction.py` — 逐 seed 对逐 seed 复现审计（相对容差 1e-5）。
- `experiments/external_baselines/run_ours_fusion_v2.py` — 融合模型·旧规则（seed42 共用 λ）参考实现。
- `experiments/external_baselines/aggregate_final_table.py` — 外部模型最终主表聚合。
- `experiments/external_baselines/run_batch.py` / `raf_rolling.py` — 外部基线（PatchTST/iTransformer/RAFT/RAF）编排与 RAF 滚动适配。
- `experiments/run_conditional_expert_mlp_v4.py` / `run_reward_feedback_critic_v2_poolaware_pilot.py` — PAER 专家记忆核心模型与冻结协议。
- `paper/tables/` — 论文表格（6 正文表 + 6 附录表，`regenerate_tables.py` 一键重生成）。

## 3. 环境、安装与运行命令

- **Python 环境**：`E:/Miniconda3/envs/myenv/python`（torch 2.12.0+cu128，numpy 2.4，pandas 3.0，chronos 2.3.1 editable 于 `E:\GitHub_Project\chronos-forecasting-main\src`）。GPU = RTX 5070 Ti（16GB）。
- **注意**：`base` 环境是 CPU-only，必须用 `myenv`。

```bash
cd /e/GitHub_Project/chronos-retrieval

# 正式融合（新规则，逐 seed λ；断点续跑：--resume 跳过已有 preds/<ds>.npz）
E:/Miniconda3/envs/myenv/python experiments/external_baselines/run_ours_fusion_v3.py --resume

# 独立重算 + 最终表
E:/Miniconda3/envs/myenv/python experiments/external_baselines/verify_fusion_results.py

# 逐 seed 复现审计
E:/Miniconda3/envs/myenv/python experiments/external_baselines/check_frozen_paer_reproduction.py
```

## 4. 数据、模型、缓存、结果位置

| 类别 | 位置 | 是否 git 跟踪 |
|---|---|---|
| 原始数据 | `data/*.csv`（9 数据集） | ✅ |
| 正式协议数据 | `results/multidataset/<ds>/{series.csv,config.json,window_indices.npz,expert_knowledge.txt}` | ✅（本次纳入） |
| 冻结专家池 | `results/reward_feedback_critic_v2_full_cross_dataset/<ds>/{final_expert_pool.json,raw_experts.json}` | ✅（本次纳入） |
| 冻结逐 seed 结果 | `results/reward_feedback_critic_v2_final_test/<ds>/result.json`（`summary.v2.seeds_mse`） | ✅（本次纳入） |
| 融合结果 | `results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011/{fusion_per_seed_results.csv,fusion_summary.csv,repro_check.csv,val_lambda_search.csv,fusion_final_table_md.txt}` | ✅（本次纳入） |
| 预测缓存 | `.../fusion_reaudit_20261011/preds/<ds>.npz`（105MB，electricity 单集 69MB） | ❌ 可再生，未跟踪 |

**PAER 专家记忆无落盘模型权重**：`ConditionalExpertMLPRetriever` 每次按 seed 确定性重训（`_init_state` + `train_arm`），冻结的是专家池 + 协议 + seed，不是 checkpoint。复现靠同一代码 + 同一专家池 + 同一 seed。

## 5. 已完成 / 未完成实验

| 实验 | 状态 |
|---|---|
| 内部消融（PAER 专家记忆 vs Raw Expert） | ✅ 7/7，5/7 胜，mean RI +13.4%（见 `results/reward_feedback_critic_v2_final_test/`） |
| 外部基线 PatchTST / iTransformer / RAFT | ✅ 21/21 |
| 外部基线 RAF | ⚠️ 6/7（electricity 未完成，暂停；有断点可续） |
| 融合模型（Chronos-2 + PAER，新规则逐 seed λ） | ✅ 7/7 |
| 融合模型（旧规则 seed42 共用 λ） | ✅ 7/7（参考，非正式） |

## 6. 最终实验规则与核查结论

- 融合 λ 规则：**各随机种子分别在 VAL 选 λ**（网格 `[0, .05, .1, .15, .2, .3, .5, .7, 1.0]`），固定到各自 TEST。λ 只依据 VAL，不据 TEST 调整、不挑种子。
- 复现核查：**21/21 PASS**（3 seed × 7 数据集，逐 seed 对逐 seed，相对容差 1e-5，实际相对误差 ~1e-7）。
- 最终结果（原始值 MSE，越低越好；`±` = 3 seed 标准差 ddof=0）：

| Dataset | Chronos-2 | 融合模型(新规则) | RI vs Chronos-2 | λ(42/123/2026) |
|---|---|---|---|---|
| etth1 | 7.961 | 7.961 | +0.0%（未采用记忆） | 0/0/0 |
| etth2 | 18.085 | 17.809 | +1.5% | .05/0/.05 |
| exchange_rate | 0.000957 | 0.000957 | +0.0%（未采用记忆） | 0/0/0 |
| traffic | 0.001684 | 0.001291 | +23.3% | .5/.5/.5 |
| weather | 129.19 | 140.27 | −8.6% | .2/.2/.3 |
| solar | 210843 | 176077 | +16.5% | .3/.3/.3 |
| electricity | 9.808e7 | 9.686e7 | +1.2% | .1/.05/.1 |

- **汇总（等权 RI）≈ +4.86%，4/7 数据集正改善。** 未发现一致的大幅提升；weather 明确退化（新规则下 seed2026 选 λ=0.3 掺入更多劣质记忆，比旧规则 −3.9% 更差）。**结论：当前证据对「PAER 增强 Chronos-2」支持偏弱。**

## 7. 已知问题

1. **目录大小写**：`results/multidataset/` 下 ETT 数据集目录名为大写 `ETTh1`/`ETTh2`，而正式协议代码用 `etth1`/`etth2`（小写）。Windows 文件系统不区分大小写所以能跑，**Linux 上会找不到目录**。接手若在 Linux 复现需先统一目录名大小写。
2. **`results/` 大部分被 `.gitignore` 忽略**：本次仅将冻结关键产物、正式协议数据、融合结果纳入版本控制；预测缓存 `preds/*.npz`（105MB）与大量历史结果目录仍本地-only。若缺，按第 3 节命令重跑融合即可再生成。
3. **RAF electricity 未完成**：暂停中，`raf_rolling.py` 已有断点续跑（每 200 窗口存 `raf_ckpt_electricity.json`）。
4. 旧 `*_chronos.npz`（`results/ETTh*_chronos.npz` 等）是**日历 split** 的 Chronos-2 预测，与正式协议 test 窗口不对齐，不可用于正式表。

## 8. Codex 接手检查顺序

1. 审查代码与数据口径：确认 `window_indices.npz` 的 train/val/test 起始下标是权威划分；确认融合公式 `(1-λ)·chronos + λ·paer` 与 λ 网格。
2. 独立重算缓存指标：跑 `verify_fusion_results.py`，核对 `repro_check.csv` 21/21 PASS、`fusion_summary.csv` 数值。
3. 抽查推理：对 exchange_rate（最小集）与 etth1 单独重跑一个 seed，核对 MSE 与冻结 `result.json` 一致。
4. 仅重跑受影响部分：若发现不一致，按第 3 节命令补跑对应数据集（`--resume` 跳过已有缓存）。
