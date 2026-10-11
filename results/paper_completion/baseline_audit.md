# P8 外部 Baseline 审计

范围：扫描仓库内已实现 / 可比较的外部 baseline，按四类标记状态。
状态：EXISTS（已实现且有结果）/ MISSING_BUT_IMPORTANT（重要且缺失）/ OPTIONAL（可选）/ NOT_COMPARABLE（不公平或不可比）。

## 结论速览

- 本方法（LLM Expert → Executable Retrieval → Reward Feedback → Pool-Aware v2）在冻结 v2 实验里的**唯一可比较基线是内部臂 A/B/C/D/E**，即 BASE / SHAPE / RAW_EXPERT / REWARD_V1 / POOL_AWARE_V2。
- 仓库内**没有实现任何外部深度 TS 模型**（PatchTST/iTransformer/TimesNet/TSMixer/N-BEATS 等）。
- README 提到的 Chronos-2 是「检索融合」早期动机，不是本 v2 链路的直接对比臂（当前 BASE 臂是 CNN-only 的 learned retriever，非 Chronos-2 pipeline）。
- 因此论文**目前不能声称 SOTA**；只能声称「相对内部 CNN-only / raw-retrieval / v1 的消融提升」。

## A. Backbone / TSFM（时间序列基础模型）

| Baseline | 状态 | 说明 |
|---|---|---|
| Chronos / Chronos-2 | OPTIONAL（动机层存在） | README 早期将检索预测与 Chronos-2 做 λ 融合；但冻结 v2 链路未把 Chronos-2 作为 arm 对比，BASE 是 CNN-only retriever。若要对比需固定 λ 融合协议，当前未做。 |
| TimesFM / Lag-Llama / Moirai | NOT_COMPARABLE | 未实现、未下载权重；与「executable expert retrieval」范式不可直接比。 |

## B. 深度 TS 模型

| Baseline | 状态 | 说明 |
|---|---|---|
| PatchTST | MISSING_BUT_IMPORTANT | 检索 vs 长程 transformer 的经典对照，未实现。 |
| iTransformer | MISSING_BUT_IMPORTANT | 同上。 |
| TimesNet | OPTIONAL | 多周期分解模型，与 expert 先验有主题相关，但非必需。 |
| TSMixer / N-BEATS / DLinear | OPTIONAL | 简单强基线；未实现。 |

> 说明：这些缺失不致命——本文核心贡献是「可执行 LLM 专家知识 + 检索 reward」的证据链，而非与 SOTA 深度模型比精度。但审稿人若要求「检索是否只是绕路」，PatchTST/iTransformer 是关键对照（MISSING_BUT_IMPORTANT）。

## C. Retrieval Forecasting（最接近的方法）

| Baseline | 状态 | 说明 |
|---|---|---|
| RAFT（Retrieval-Augmented Forecasting） | MISSING_BUT_IMPORTANT | 最接近的方法：检索历史片段做预测。未实现。 |
| N-BEATS / 其他 retrieval 变体 | OPTIONAL | 未实现。 |

> RAFT 是最直接的可比工作，缺失是本文检索贡献的重要缺口。建议列入 SHOULD-HAVE。

## D. LLM / Agent Forecasting（最接近的公开方法）

| Baseline | 状态 | 说明 |
|---|---|---|
| Time-LLM / GPT4TS | OPTIONAL | LLM 时序预测；未实现。 |
| Agent 类闭环（自建 v5/v6/v7） | EXISTS（内部） | 仓库内有 Expert-Agent 闭环 v5、Critic v4/v5/v6/v7、multiagent 等，但按协议**不把 Agent 本身写成核心创新**，不作为外部 baseline。 |
| TEMPO（references/tempo_extracted.txt） | OPTIONAL（文献） | ICLR 2024 prompt-based TS 预训练；仅作为引用背景，未实现。 |

## 公平性与缺失原因

1. **范式不匹配**：本方法输出是「检索融合预测」，与端到端深度模型的输入/训练/评估协议不同；直接并表需统一 horizon/split/seed/重训练，成本高。
2. **数据协议固定**：正式 7 数据集、seeds=[42,123,2026]、deterministic protocol；外部模型需在同一协议下重跑才公平，当前未做。
3. **冻结 v2 约束**：本次补实验禁止改 v2/超参/seeds，外部 baseline 训练属于新协议，需单独建立。

## 建议

- MUST HAVE：无（内部 A/B/C/D/E 消融已足够支撑当前 claim）。
- SHOULD HAVE：PatchTST + iTransformer（或至少 RAFT）在同一 split/seed 下重跑，用于回答「检索是否只是绕路」。
- OPTIONAL：Chronos-2 作为 backbone 的 λ 融合臂。

## 结论

- 论文当前可安全声称：**相对内部 BASE/SHAPE/RAW/v1 的消融提升**。
- 不可声称：SOTA、超越外部深度 TS 模型、或「检索优于 PatchTST/iTransformer」——除非补齐 B/C 类并表实验。
