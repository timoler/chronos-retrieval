# P1 主消融表（A/B/C/D/E，TEST，frozen v2）

方法：BASE=CNN-only，SHAPE=+shape 先验，RAW_EXPERT=+raw 检索专家，REWARD_V1=+reward-feedback v1，POOL_AWARE_V2=+pool-aware v2。
seeds=[42,123,2026]，正式数据集 7 个。指标为 TEST MSE（3-seed mean）。
coverage / runtime 未在已有 artifact 中，留空（不填 0）。

| Dataset | BASE | SHAPE | RAW | v1 | v2 | v2 vs BASE | v2 vs RAW | v2 vs v1 | wins raw | wins v1 |
|---|---|---|---|---|---|---|---|---|---|---|
| etth1 | 69.2042 | 80.6598 | 61.974 | 58.9196 | 42.4171 | +38.7% | +31.6% | +28.0% | 3/3 | 2/3 |
| etth2 | 139.265 | 153.052 | 125.854 | 132.425 | 98.4748 | +29.3% | +21.8% | +25.6% | 3/3 | 3/3 |
| weather | 701.974 | 585.202 | 680.424 | 685.693 | 559.943 | +20.2% | +17.7% | +18.3% | 3/3 | 3/3 |
| exchange_rate | 0.0359643 | 0.036519 | 0.038995 | 0.037252 | 0.0364797 | -1.4% | +6.5% | +2.1% | 3/3 | 2/3 |
| solar | 330056 | 333661 | 465309 | 347965 | 315416 | +4.4% | +32.2% | +9.4% | 3/3 | 3/3 |
| electricity | 7.44485e+08 | 7.42462e+08 | 3.0796e+08 | 6.35739e+08 | 3.55565e+08 | +52.2% | -15.5% | +44.1% | 1/3 | 3/3 |
| traffic | 0.00156033 | 0.00155833 | 0.00154067 | 0.001501 | 0.001547 | +0.9% | -0.4% | -3.1% | 2/3 | 0/3 |

## 结果
- v2 vs Raw：5/7 数据集胜，mean RI = +13.40%，median RI = +17.71%
- v2 vs v1：6/7 数据集胜，mean RI = +17.77%，median RI = +18.34%
- 失败数据集（v2 不优于 Raw）：electricity（-15.5%）、traffic（-0.4%）。

## 结论
- 主消融表直接复用 frozen v2 final TEST（`final_test_ablation.csv`），未重跑。
- coverage/runtime 缺失：见 P5（效率）与 P7（pool/coverage）分别补齐。