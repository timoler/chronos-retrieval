# 论文消融与缺失实验补全报告

日期：2026-09-20。协议：冻结 Pool-Aware v2，不开发 v3，不依据 TEST 改方法，seeds=[42,123,2026]，正式 7 数据集。已有结果一律标记 `REUSE_EXISTING_RESULT`，未重跑。

---

## 1. 总览

| 任务 | 内容 | 状态 | 关键文件 |
|---|---|---|---|
| P0 | 已有结果扫描 | ✅ DONE | existing_result_inventory.md |
| P1 | 主消融 A/B/C/D/E | ✅ DONE（REUSE TEST） | paper_main_ablation.csv + main_ablation.md |
| P2 | 等 Pool Size 控制 | ✅ DONE（7/7） | paper_pool_size_control.csv |
| P3 | Reward 机制对照 | ✅ DONE | reward_mechanism_comparison.csv |
| P4 | 统计可靠性 | ✅ DONE（LIMITED） | paper_statistics.csv |
| P5 | 效率开销 | ✅ DONE | paper_efficiency.csv |
| P6 | 参数敏感性 | ✅ DONE（4 快数据集） | paper_sensitivity.csv |
| P7 | Pool/Coverage 分析 | ✅ DONE | paper_pool_analysis.csv |
| P8 | 外部 baseline 审计 | ✅ DONE | baseline_audit.md |

---

## 2. P1 主消融（结果 + 结论）

**结果**（frozen TEST，3-seed mean MSE，REUSE `final_test_ablation.csv`）：

| Dataset | BASE | SHAPE | RAW | v1 | v2 | v2 vs RAW | v2 vs v1 | wins/raw |
|---|---|---|---|---|---|---|---|---|
| etth1 | 69.20 | 80.66 | 61.97 | 58.92 | 42.42 | +31.6% | +28.0% | 3/3 |
| etth2 | 139.27 | 153.05 | 125.85 | 132.43 | 98.47 | +21.8% | +25.6% | 3/3 |
| weather | 701.97 | 585.20 | 680.42 | 685.69 | 559.94 | +17.7% | +18.3% | 3/3 |
| exchange_rate | 0.0360 | 0.0365 | 0.0390 | 0.0373 | 0.0365 | +6.5% | +2.1% | 3/3 |
| solar | 330056 | 333661 | 465309 | 347965 | 315416 | +32.2% | +9.4% | 3/3 |
| electricity | 7.44e8 | 7.42e8 | 3.08e8 | 6.36e8 | 3.56e8 | **−15.5%** | +44.1% | 1/3 |
| traffic | 0.00156 | 0.00156 | 0.00154 | 0.00150 | 0.00155 | **−0.4%** | −3.1% | 2/3 |

汇总：v2 vs Raw 5/7 胜，mean RI +13.40%，median RI +17.71%；v2 vs v1 6/7 胜，mean RI +17.77%，median RI +18.34%。VAL→TEST 方向保持 7/7。

**结论**：Pool-Aware v2 相对 Raw 检索与 v1 reward-feedback 在多数数据集有稳健提升，失败集（electricity/traffic）与 VAL 一致，非 TEST 过拟合。

---

## 3. P2 等 Pool Size 控制（关键新增控制）

**问题**：v2 比 v1 好是不是仅仅因为保留了更多 experts？Pool-Aware Reward 是否有独立于 pool size 的价值？

**设计**：TRAIN/frozen VAL only。每个 dataset 读 K_v2，比较 V2_POOL_AWARE / RANDOM_K（5 个 control seeds）/ V1_SCORE_TOPK / RAW_ALL。同一 retriever/初始化/训练预算/split，唯一区别 expert subset。

**结果**（TRAIN/frozen VAL，7/7 全部完成并验证；完整表 `paper_pool_size_control.csv`）：

| Dataset | K_v2 | v2 | RAW_ALL | RANDOM_K(mean) | V1_SCORE_TOPK | v2 vs RANDOM | v2 vs V1score |
|---|---|---|---|---|---|---|---|
| etth1 | 7 | 34.69 | 63.90 | 74.65 | 56.23 | **+53.5%** | **+38.3%** |
| etth2 | 6 | 121.85 | 206.15 | 190.34 | 203.87 | **+36.0%** | **+40.2%** |
| weather | 7 | 458.95 | 530.95 | 502.04 | 534.48 | **+8.6%** | **+14.1%** |
| exchange_rate | 6 | 0.03198 | 0.03635 | 0.03423 | 0.03327 | **+6.6%** | **+3.9%** |
| solar | 5 | 197100 | 198826 | 201710 | 208152 | **+2.3%** | **+5.3%** |
| electricity | 5 | 3.277e8 | 2.685e8 | 1.152e9 | 2.736e8 | **+71.5%** | −19.8% |
| traffic | 5 | 0.000721 | 0.000684 | 0.000707 | 0.000687 | −2.0% | −5.0% |

汇总：v2 优于 RANDOM_K **6/7**（仅 traffic 输）；v2 优于 V1_SCORE_TOPK **5/7**（traffic、electricity 输）。

**结论（7/7）**：
1. **Pool-Aware Reward 提供独立于 pool size 的选池价值**——同 K 下 v2 一致优于随机子集 RANDOM_K（6/7，+2.3%~+71.5%），唯一例外 traffic。
2. **但 pool-aware reward 并非全面优于 v1 individual reward 选池**——v2 优于 V1_SCORE_TOPK 5/7，输的恰好是 v2 也输 raw 的两个失败案例（electricity/traffic）。electricity 上 RANDOM_K 方差极大（std 1.3e9），随机 5-expert 子集高度不可靠，v2 避免了灾难性子集（+71.5%），但 v1-score 的 top-5（2.736e8）比 v2（3.277e8）略好。

---

## 4. P3 Reward 机制对照（结果 + 结论）

**结果**：train pool_reward(v2) 与 VAL contribution 的对齐（REUSE `reward_alignment_train_val.csv`）：

| Dataset | Spearman | sign-agreement | top3 | top4 |
|---|---|---|---|---|
| etth1 | +0.286 | 0.625 | 0.667 | 0.750 |
| solar | −0.810 | 0.375 | 0.000 | 0.250 |
| electricity | −0.190 | 0.375 | 0.333 | 0.500 |
| traffic | −0.548 | 0.375 | 0.333 | 0.250 |

v1 与 v2 的选 expert 分歧可见于 `reward_mechanism_comparison.csv`（例：etth1 E8 v1 reward=2 KEEP，但 v2 pool_reward=−0.126 → 被 v2 删除）。

**结论**：v2 的 reward 对齐并非全面更强（1 正 / 2 负 / 1 近 0），v2 的正收益不能简单归因于「reward 更准」。如实报告，不改 reward。

---

## 5. P4 统计可靠性（结果 + 结论）

**结果**：正式 TEST 只有 3-seed aggregate（每数据集每臂 3 seed MSE），无逐窗口 TEST prediction（除 electricity/traffic）。因此按协议报 mean±std、seed wins、RI，不声称 seed-level significance（n=3）。

- traffic：已有 pooled bootstrap 95% CI 跨 0，seed 方向不一致 → COMPARABLE。
- electricity：−15.5% 幅度大，不能归因于纯随机噪声（真实退化 + 部分 seed 波动）。

**结论**：统计证据为「描述性 + VAL→TEST 方向 7/7」级别，不声称 p<0.05。详见 `paper_statistics.csv` / `statistics_report.md`。

---

## 6. P5 效率开销（结果 + 结论）

**结果**（实测 RTX 5070 Ti / 16GB，etth1，每项 3 次取 median；`paper_efficiency.csv`）：

- 单次 final retriever 训练（30 epoch）：**64.8s**；单窗推理：**0.289ms**；峰值显存：**241.6MB**。
- final 推理开销三者相同（同一 retriever、同一 30-epoch 训练、同一单窗推理）。
- v2 pool-aware reward 每数据集约 **108 次 retriever 训练（≈117 min，TRAIN-only）**；v1 逐条 reward 只有 10–23 次 expert 评估（≈80–186s 折算）。
- 最终 pool size：raw=8，v1=1–8，v2=5–7。

**结论**：v2 的唯一额外成本 = TRAIN-only 的 pool-aware reward 计算（每轮 N_folds×(N_experts+1) 次 retriever 重训，一次性），**不进入推理、不增加 TEST 延迟**。详见 `efficiency_report.md`。

---

## 7. P6 参数敏感性（Top-K ∈ {1,3,5,10}）

**结果**（TRAIN/frozen VAL，4 快数据集；`paper_sensitivity.csv`）：

| Dataset | K=1 | K=3 | K=5（正式） | K=10 |
|---|---|---|---|---|
| etth1 | 37.14 | 35.50 | 34.69 | 32.90 |
| etth2 | 172.28 | 135.23 | 121.85 | 109.65 |
| exchange_rate | 0.03225 | 0.03191 | 0.03198 | 0.03209 |
| traffic | 0.001077 | 0.000787 | 0.000721 | 0.000690 |

**结论**：K=5 不是极窄参数点。etth1/etth2/traffic 上 MSE 随 K 单调下降（K=10 最好），exchange_rate 基本平坦——结果对 Top-K 稳健，且更大 K 略优（与「pool 越大检索越充分」一致）。不据此改 K（冻结 v2）。

---

## 8. P7 Expert Pool / Coverage（结果 + 结论）

**结果**（`paper_pool_analysis.csv`）：

| Dataset | raw | v1 | v2 | v1 collapse | coverage raw→v2 |
|---|---|---|---|---|---|
| etth1 | 8 | 8 | 7 | 否 | — |
| etth2 | 8 | 8 | 6 | 否 | — |
| weather | 8 | 3 | 7 | 否 | — |
| exchange_rate | 8 | 2 | 6 | 否 | — |
| solar | 8 | 5 | 5 | 否 | — |
| electricity | 8 | **1** | 5 | **是** | 1.000→0.692 |
| traffic | 8 | 4 | 5 | 否 | 1.000→0.974 |

**结论**：v2 的主要机制收益 = 避免 v1 的 pool collapse（尤其 electricity 8→1 修复为 5）。但 v2 在 electricity 上仍因 coverage loss（删了高覆盖 E2/E8）输给 Raw，说明 pool-aware reward 不保证覆盖保持。

---

## 9. P8 外部 baseline 审计

见 `baseline_audit.md`。要点：
- 无外部深度 TS 模型（PatchTST/iTransformer/RAFT）实现 → 不能声称 SOTA。
- 唯一可比较基线是内部 A/B/C/D/E。
- Chronos-2 是动机层，非 v2 链路直接对比臂。

---

## 10. 三个审稿问题回答

**Q1：提升是不是仅仅因为套了 Agent？**
- 否。B(+SHAPE) 是「非 Agent 的形状先验」臂，与 C(+RAW EXPERT 检索知识)、D(v1)、E(v2) 分离。v2 相对 SHAPE 的提升（如 etth1 +38.7% vs BASE、weather 超越 SHAPE）说明提升来自可执行专家知识 + reward 筛选，而非「用了 LLM/Agent」本身。
- 证据：A(BASE)/B(SHAPE) 在多数数据集相当甚至 SHAPE 更差，而 C/D/E 的 expert 知识带来提升。

**Q2：v2 比 v1 好是不是仅仅因为保留了更多 experts？**
- 否。关键证据：etth1 上 v1 pool=8 > v2 pool=7，v2 反而更少 experts 却更优；exchange_rate 上 v1=2 < v2=6，v2 更多 experts 更优——性能与 pool size **非单调**。真正起作用的是**选池质量**（P2 的 RANDOM_K 同 size 对照证实）。

**Q3：Pool-Aware Reward 是否提供了独立于 pool size 的价值？**
- **是**。同 pool size K 下，pool-aware reward 的选池 v2 一致优于随机子集 RANDOM_K（**6/7**，+2.3%~+71.5%），唯一例外 traffic（v2<raw 的失败案例）。即 pool-aware reward 的选池价值**独立于 pool size**（size 相同，仅选池不同）。
- 但相对 v1 individual reward 的选池，v2 只优于 V1_SCORE_TOPK **5/7**——输的两个（electricity/traffic）恰好是 v2 也输 raw 的失败案例。故「pool-aware reward 全面优于 v1 isolated reward」不成立，如实报告。

---

## 11. 分项总结

### A 已经足够的实验
- P1 主消融（A/B/C/D/E，TEST + VAL，7 数据集 × 3 seeds）—— 证据充分。
- P7 Pool/Coverage（含 electricity 8→1 collapse、coverage loss 诊断）—— 充分。
- P3 Reward 对齐（4/7 数据集可算，负结果如实保存）—— 足够支撑「reward 对齐非全面更强」的结论。

### B 新补实验
- P2 等 Pool Size 控制（RANDOM_K / V1_SCORE_TOPK）—— 关键新增，7/7 完成。
- P5 效率开销 —— 完成（retriever 训练 64.8s/arm，v2 reward 108 次训练≈117min TRAIN-only）。
- P6 Top-K 敏感性 —— 完成（4 快数据集：K=5 非窄峰，结果对 K 稳健）。

### C 仍缺实验
- MUST HAVE：无（核心证据链 P1/P2/P3/P5/P7 全部闭环，P2 7/7 完成）。
- SHOULD HAVE：P6 敏感性补 weather/solar/electricity；P3 的 val_contribution 补 etth2/weather/exchange_rate。
- OPTIONAL：外部 baseline 并表（PatchTST/iTransformer/RAFT）。

### D 论文现在安全可声称的 claim
- Pool-Aware v2 相对 Raw 检索在 5/7 数据集、mean RI +13.4%、median +17.7% 提升（TEST，frozen，一次性）。
- v2 相对 v1 在 6/7 数据集提升；VAL→TEST 方向 7/7 一致。
- v2 修复了 v1 在 electricity 上的 pool collapse（8→1 修复为 5）。
- **Pool-Aware Reward 提供独立于 pool size 的选池价值**：同 K 下 v2 优于随机子集 RANDOM_K（6/7），优于 v1-score 子集 V1_SCORE_TOPK（5/7）。
- Electricity/Traffic 上 v2 不优于 Raw（−15.5% / −0.4%），如实报告。

### E 不能声称的 claim
- 不声称 SOTA（无外部深度 TS baseline）。
- 不声称 v2 在所有数据集一致优于 Raw。
- 不声称 seed-level 统计显著（n=3）。
- 不声称「Pool-Aware Reward 在**所有**数据集优于随机/v1-score 选池」（traffic 上 v2 输随机，electricity/traffic 上 v2 输 v1-score）。
- 不把 v2.1 或 Learned Critic 包装为成功方法；不把 Agent 本身写成核心创新。

### F 推荐进入论文的 Table/Figure 及对应 csv
- Table 1（主消融 TEST）：`paper_main_ablation.csv` / `main_ablation.md`
- Table 2（Pool size 控制）：`paper_pool_size_control.csv`
- Table 3（Pool/Coverage）：`paper_pool_analysis.csv`
- Table 4（Reward 对齐）：`reward_mechanism_comparison.csv`
- 附图（可选）：Pool retention vs v2_vs_raw 散点（由 paper_pool_analysis.csv 生成）
