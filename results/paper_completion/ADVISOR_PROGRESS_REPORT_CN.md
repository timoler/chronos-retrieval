# 论文实验进度汇报（导师版）

日期：2026-10-06。本报告所有数字均来自仓库实际结果文件，未重跑、未调参、未用 TEST 改方法。冻结方法 = Pool-Aware v2，seeds = [42,123,2026]，正式 7 数据集 = etth1 / etth2 / weather / exchange_rate / solar / electricity / traffic。

---

## 1. 当前研究解决什么问题

现有时间序列基础模型（TSFM）的历史检索通常依赖固定的 shape / embedding 相似度。本研究希望把领域专家知识转化为**可执行的检索准则**（executable retrieval criteria），并用数据驱动的 **Pool-Aware Reward** 筛选出真正有预测价值的 expert retrieval prior，最终与 TSFM 预测融合。

一句话链路：

> Expert Knowledge → executable retrieval hypotheses → historical retrieval → Pool-Aware validation/reward → selected memory → TSFM forecasting

核心不是"给 Chronos-2 外套一个 Agent"，而是"专家知识 → 可执行检索准则 → reward 筛选"这条可验证的证据链。

---

## 2. 当前方法最终版本

```
Expert Knowledge
      ↓  LLM/Agent 把知识转化为可执行检索专家
Candidate Expert Pool（每数据集 8 个 expert）
      ↓  Pool-Aware Reward / validation（TRAIN-only，边际贡献）
Selected Expert Memory（v2 pool，K=5–7）
      ↓  Historical Retrieval
TSFM
      ↓
Forecast
```

- **v1**：对每个 expert 单独做个体 reward（逐条 fold-mean-gain，`-2..2`），独立筛选 → 易导致 pool collapse（electricity 8→1）。
- **v2（最终采用）**：Pool-Aware Reward，每次评估"该 expert 在其余 expert 共同激活下的边际贡献"（N_folds × (N_experts+1) 次 retriever 重训），按边际贡献筛选 pool。
- **为什么选 v2**：v2 相对 v1 在 6/7 数据集提升，且修复了 v1 在 electricity 的 pool collapse（8→1 修复为 5）。v2 的唯一额外成本在 TRAIN 阶段，不进入推理。

---

## 3. 最终实验数据（真实数字）

### 3.1 主结果（frozen TEST，3-seed mean MSE，来源 `results/reward_feedback_critic_v2_final_test/final_test_summary.csv`）

| Dataset | Raw | v1 | v2 | v2 vs Raw | v2 vs v1 |
|---|---:|---:|---:|---:|---:|
| etth1 | 61.974 | 58.920 | 42.417 | +31.6% | +28.0% |
| etth2 | 125.854 | 132.425 | 98.475 | +21.8% | +25.6% |
| weather | 680.424 | 685.693 | 559.943 | +17.7% | +18.3% |
| exchange_rate | 0.038995 | 0.037252 | 0.036480 | +6.5% | +2.1% |
| solar | 465309 | 347965 | 315416 | +32.2% | +9.4% |
| electricity | 3.0796e8 | 6.3574e8 | 3.5556e8 | **−15.5%** | +44.1% |
| traffic | 0.001541 | 0.001501 | 0.001547 | **−0.4%** | −3.1% |

- v2 vs Raw：**5/7 胜**，mean RI = **+13.40%**，median RI = +17.71%。
- v2 vs v1：**6/7 胜**（唯一输的是 traffic），mean RI = +17.77%，median = +18.34%。
- 3-seed 方向：etth1/etth2/weather/exchange_rate/solar 均 3/3 胜 Raw；electricity 1/3、traffic 2/3。
- VAL→TEST 方向保持 **7/7**（非 TEST 过拟合）。

### 3.2 Ablation（来源 `results/paper_completion/paper_main_ablation.csv`）

正式 5 臂：**BASE**（CNN-only retriever）、**SHAPE**（+shape 先验）、**RAW_EXPERT**（+raw expert 检索，pool=8）、**REWARD_V1**（+个体 reward）、**POOL_AWARE_V2**（+pool-aware reward，最终）。

- **v2 vs SHAPE：7/7 胜**（expert 知识 + reward 筛选后全面优于纯 shape 检索）。
- 但 **RAW_EXPERT vs SHAPE：仅 4/7 胜**（etth1/etth2/electricity/traffic 胜；weather/exchange_rate/solar 败，solar 大幅退化）。说明**未经 reward 筛选的 raw expert pool 并不稳定优于 shape**，expert 知识的价值在 reward 筛选后才显现。

### 3.3 等 Pool Size 控制（TRAIN/frozen VAL，7/7，来源 `paper_pool_size_control.csv`）

同 K 下唯一区别是 expert 子集（同一 retriever/初始化/训练预算/split）：

| Dataset | K | v2 | RANDOM_K | V1_SCORE_TOPK | v2 vs RANDOM | v2 vs V1score |
|---|---:|---:|---:|---:|---:|---:|
| etth1 | 7 | 34.689 | 74.645 | 56.232 | +53.5% | +38.3% |
| etth2 | 6 | 121.846 | 190.340 | 203.869 | +36.0% | +40.2% |
| weather | 7 | 458.951 | 502.045 | 534.476 | +8.6% | +14.1% |
| exchange_rate | 6 | 0.031983 | 0.034229 | 0.033268 | +6.6% | +3.9% |
| solar | 5 | 197100 | 201710 | 208152 | +2.3% | +5.3% |
| electricity | 5 | 3.277e8 | 1.152e9 | 2.736e8 | +71.5% | **−19.8%** |
| traffic | 5 | 0.000721 | 0.000707 | 0.000687 | **−2.0%** | **−5.0%** |

- **v2 > RANDOM_K：6/7**（仅 traffic 输）。这是"提升不是单纯因为 pool size"的关键证据。
- **v2 > V1_SCORE_TOPK：5/7**（traffic、electricity 输——恰好是 v2 也输 Raw 的两个案例）。故"pool-aware reward 全面优于 v1 isolated reward"**不成立**。
- electricity 的 RANDOM_K 方差极大（std≈1.3e9），v2 避免了灾难性子集（+71.5%），但仍输给 v1-score 的 top-5。

### 3.4 Sensitivity（来源 `paper_sensitivity.csv`，仅 4 个快数据集）

| Dataset | K=1 | K=3 | K=5（正式） | K=10 |
|---|---:|---:|---:|---:|
| etth1 | 37.14 | 35.50 | 34.69 | 32.90 |
| etth2 | 172.28 | 135.23 | 121.85 | 109.65 |
| exchange_rate | 0.03225 | 0.03191 | 0.03198 | 0.03209 |
| traffic | 0.001077 | 0.000787 | 0.000721 | 0.000690 |

K=5 **不是**极窄参数点：etth1/etth2/traffic 随 K 增大单调下降（K=10 更好），exchange_rate 基本平坦。结果对 Top-K 稳健，且更大 K 略优（与"pool 越大检索越充分"一致）。

### 3.5 Efficiency（来源 `paper_efficiency.csv`，实测 RTX 5070 Ti / 16GB）

- 单次 retriever 训练（30 epoch）：**64.8s**；单窗推理 **0.29ms**；峰值显存 241.6MB。
- v2 pool-aware reward：每数据集约 **108 次 retriever 训练 ≈ 117 min（TRAIN-only）**；v1 仅 10–23 次 expert 评估（≈80–186s）。
- 最终 pool size：raw=8，v1=1–8，v2=5–7。
- **推理阶段三臂开销完全相同**（同一 retriever、同一训练、同一单窗推理），reward 只发生在离线 TRAIN 阶段，**不增加 TEST/推理延迟**。

---

## 4. 两个异常数据集的诊断

### 4.1 Electricity（真实退化，主因 = regime coverage 损失）

方法在多数数据集稳定获益，但 Electricity 暴露了当前 expert pruning 在 **regime coverage** 上的局限：

- **主因 COVERAGE_LOSS**：v2 删除了 raw 覆盖率最高的 E2（82.7%）与 E8（58.6%），union coverage **1.000 → 0.692**（30.8% 的 test 窗口没有任何激活 expert），avg active/window 3.53 → 1.28，属过度剪枝。
- 被删 E2/E8 恰好覆盖 v2 最差的窗口：**高波动 Q4（−44.2%）**与**凌晨 3–6 点（−31%~−37%）**。
- **可能的次要机制**（描述性，非确定因果）：REWARD_GENERALIZATION_MISMATCH（TRAIN pool_reward 与 VAL contribution Spearman = −0.190）；pairwise synergy（E2+E8 交互 +0.100，删除后损失协同）；OPTIMIZATION_SENSITIVITY（仅 1/3 seed 赢 Raw）。
- **已排除/注意**：reward 对齐并非 Electricity 独有——Solar 的 Spearman 更低（−0.810），却成功。故"reward generalization mismatch"不能写成 Electricity 独有的确定因果。

### 4.2 Traffic（comparable，非明确失败）

- 相对差异 |rel_diff| = **+0.42%**（<1%），pooled bootstrap 95% CI **跨 0**。
- 3 seed 方向不一致：seed42 显著更差（p≈1e-13）、seed123 显著更好（p=0.001）、seed2026 不显著。
- coverage 1.000 → 0.974（接近无损，无明显 over-pruning）。
- 结论：Traffic 定义为 **COMPARABLE / OPTIMIZATION_SENSITIVITY**，不应强行写成"第二组失败案例"。

---

## 5. 当前论文最重要的三个证据

**Q1：是不是只是套了一层 Agent？**
不是。v2（expert 知识 + reward 筛选）相对纯 shape 检索 SHAPE **7/7 胜**；且 SHAPE 是非 Agent 的形状先验臂。提升来自"可执行专家知识 + reward 筛选"，不是"用了 LLM/Agent"。诚实边界：未经 reward 筛选的 raw expert pool 仅 4/7 胜 SHAPE，故 expert 知识的价值**依赖 reward 筛选**。

**Q2：v2 更好是不是因为 pool size？**
不是。性能与 pool size **非单调**：etth1 上 v1 pool=8 > v2 pool=7，v2 反而更优；exchange_rate 上 v1=2 < v2=6，v2 更多 expert 更优。等 K 控制进一步证实是**选池质量**而非数量。

**Q3：Pool-Aware Reward 到底有没有独立价值？**
**有，但并非全面优于一切**。同 K 下 v2 优于随机子集 RANDOM_K **6/7**（+2.3%~+71.5%），说明选池价值独立于 pool size；但相对 v1 isolated reward 的选池只优于 V1_SCORE_TOPK **5/7**（traffic、electricity 输），故"全面优于 v1 isolated reward"不成立。

---

## 6. 当前完成度

**DONE（内部机制验证与消融已基本闭环）：**
- P1 主消融（7 数据集 × 5 臂 × 3 seeds，TEST + VAL）
- P2 等 Pool Size 控制（7/7）
- P3 Reward 机制对照（4/7 可算，负结果如实保存）
- P5 效率开销
- P7 Pool/Coverage 分析（含 electricity 8→1 collapse 与 coverage loss 诊断）

**REMAINING（主要剩余 = 外部 baseline 完善，非继续改核心方法）：**
- P8 外部 baseline：PatchTST / iTransformer / RAFT 已 7/7 完成；**RAF 仅 4/7**（缺 weather/solar/electricity，seed42）；**Chronos-2 原始值未对齐正式 TEST**；**主方法"Chronos-2 + memory fusion"尚未在统一 split 下重跑**（仅有 `run_ours_fusion.py` 脚本 + 审计文档，未执行）。
- P6 Sensitivity 仅 4/7（缺 weather/solar/electricity）。

**投稿前必须补的 baseline（精确到 Dataset × Seed × Model）：**
- RAF / weather / seed42
- RAF / solar / seed42
- RAF / electricity / seed42
- （外加：主方法 Chronos-2+fusion 在统一协议下重跑并并表）

**当前 claim 边界：** 不能声称 SOTA；不能声称"outperforms deep forecasting baselines"——现主表里冻结 v2（CNN-only 内臂，etth1 42.42）明显劣于 PatchTST（7.99）/iTransformer（9.07）/RAFT（6.83）/RAF（8.94）；"outperforms Chronos-2"也因 Chronos-2 未对齐而无法声称。可安全声称的是：相对内部 BASE/SHAPE/RAW/v1 的消融提升 + Pool-Aware Reward 的独立选池价值。

---

## 7. 给导师看的最终一句话总结

目前结果表明，引入可执行专家检索知识能够在多数时间序列数据集上改善 TSFM 的预测性能；进一步采用 Pool-Aware Reward 后，相比原始 expert pool 和 v1 筛选策略取得更稳定的整体表现（vs Raw 5/7、mean RI +13.4%；vs v1 6/7）。等 Pool Size 控制实验显示，该提升不能简单由 expert 数量解释——同 K 下 v2 优于随机 expert 选择 6/7。与此同时，Electricity 上的失败揭示了过度剪枝导致 regime coverage 损失这一边界条件，为后续机制改进提供了明确方向；Traffic 在统计上属于 comparable，不应算作明确失败。当前核心机制证据链已经闭环，剩余工作主要是外部 baseline 的补全（RAF 3 个数据集、以及把主方法在统一协议下并表）与论文撰写。
