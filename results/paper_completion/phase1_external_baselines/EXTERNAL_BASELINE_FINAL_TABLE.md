# 外部模型最终主表（原始值 MSE，越低越好）

> 生成：2026-10-09。Ours = Chronos-2 + Frozen Pool-Aware v2 Expert Memory/Fusion。
> 口径：统一 split/context/horizon/target/原始值 MSE。
> 种子说明：PatchTST/iTransformer/RAFT/Ours 为 3-seed mean±std；
> RAF 无训练参数（seed 只影响 Chronos 采样，num_samples=5，1 seed）；
> Chronos-2 为零样本确定性（seed 无关，单值）。

| Dataset | Chronos-2 | PatchTST | iTransformer | RAFT | RAF | Ours |
|---|---|---|---|---|---|---|
| etth1 | 7.961 | 7.987±0.0717 | 9.067±0.212 | 6.819±0.00824 | 8.939 (1 seed) | 7.961±0 |
| etth2 | 18.08 | 14.5±0.0459 | 16.78±0.345 | 16.48±0.128 | 18.02 (1 seed) | 17.7±0.0475 |
| exchange_rate | 0.000957 | 0.002213±0.00134 | 0.002021±0.000149 | 0.000955±2.45e-06 | 0.001215 (1 seed) | 0.000957±0 |
| traffic | 0.001684 | 0.001499±9.43e-07 | 0.001346±2.94e-06 | 0.002308±5.93e-05 | 0.002084 (1 seed) | 0.001291±4e-06 |
| weather | 129.2 | 157.6±7.73 | 168.4±5.96 | 146±0.0867 | 197.5 (1 seed) | 134.3±3.19 |
| solar | 2.108e+05 | 1.763e+05±967 | 2.195e+05±4.55e+03 | 2.01e+05±447 | 6.115e+05 (1 seed) | 1.761e+05±2.24e+03 |
| electricity | 9.808e+07 | 1.019e+08±4.42e+05 | 1.02e+08±2.22e+05 | 1.375e+08±1.74e+05 | N/A | 9.722e+07±8.87e+05 |

## 完成度

- **PatchTST**: 7/7 数据集完成
- **iTransformer**: 7/7 数据集完成
- **RAFT**: 7/7 数据集完成
- **RAF**: 6/7 数据集完成
- **Chronos-2**: 7/7
- **Ours (Chronos-2 + Pool-Aware v2)**: 7/7

## A. Ours vs Chronos-2（RI = (MSE_Chronos2 − MSE_Ours) / MSE_Chronos2）

| Dataset | Chronos-2 | Ours | RI |
|---|---|---|---|
| etth1 | 7.961 | 7.961 | +0.0% |
| etth2 | 18.08 | 17.7 | +2.1% |
| exchange_rate | 0.000957 | 0.000957 | +0.0% |
| traffic | 0.001684 | 0.001291 | +23.3% |
| weather | 129.2 | 134.3 | -3.9% |
| solar | 2.108e+05 | 1.761e+05 | +16.5% |
| electricity | 9.808e+07 | 9.722e+07 | +0.9% |

- 胜/负：**4/7** 数据集 RI > 0
- mean RI = **+5.56%**；median RI = **+0.87%**

## B. Ours vs 外部 baseline（胜/负，Ours MSE < baseline MSE 记胜）

- **vs PatchTST**: 6/7 胜
- **vs iTransformer**: 6/7 胜
- **vs RAFT**: 4/7 胜
- **vs RAF**: 6/6 胜

## C. 结果解释

- **机制是否有效**：由冻结 CNN-only v2 内部消融回答（v2 vs Raw 5/7、mean RI +13.4%，见 `results/reward_feedback_critic_v2_final_test/`）。
- **最终预测系统是否有竞争力**：由本表 Chronos-2 + Ours Fusion 与外部 baseline 的统一实验回答。
- 两者是不同结论，不混用。

