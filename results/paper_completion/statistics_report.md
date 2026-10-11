# P4 统计可靠性（frozen TEST，只读已有结果）

数据粒度：正式 TEST 只有 3-seed aggregate（每数据集每臂 3 个 seed MSE），无逐窗口 TEST prediction artifact（除 electricity/traffic 的 failure diagnosis）。
因此按协议：n=3，只报 mean±std、seed wins、relative improvement，不声称 seed-level significance。

| Dataset | raw mean±std | v2 mean±std | v1 mean±std | RI vs raw | RI vs v1 | wins raw | wins v1 |
|---|---|---|---|---|---|---|---|---|
| etth1 | 61.974±7.21746 | 42.4171±3.5244 | 58.9196±13.1702 | +31.6% | +28.0% | 3/3 | 2/3 |
| etth2 | 125.854±3.14119 | 98.4748±10.9504 | 132.425±7.17625 | +21.8% | +25.6% | 3/3 | 3/3 |
| weather | 680.424±7.4922 | 559.943±64.3018 | 685.693±61.0011 | +17.7% | +18.3% | 3/3 | 3/3 |
| exchange_rate | 0.038995±0.0027494 | 0.0364797±0.00454129 | 0.037252±0.000938659 | +6.5% | +2.1% | 3/3 | 2/3 |
| solar | 465309±32830.9 | 315416±8672.02 | 347965±17127.6 | +32.2% | +9.4% | 3/3 | 3/3 |
| electricity | 3.0796e+08±7.16879e+06 | 3.55565e+08±5.32215e+07 | 6.35739e+08±2.19161e+08 | -15.5% | +44.1% | 1/3 | 3/3 |
| traffic | 0.00154067±6.06429e-05 | 0.001547±9.27362e-06 | 0.001501±3.20312e-05 | -0.4% | -3.1% | 2/3 | 0/3 |

## 逐窗口 paired block bootstrap（仅 electricity / traffic，REUSE failure diagnosis）

- traffic：pooled rel_diff = +0.42%（|rel_diff|<1%），pooled bootstrap 95% CI 跨 0；seed42 显著更差（p≈1e-13）、seed123 显著更好（p=0.001）、seed2026 不显著 → COMPARABLE。
- electricity：总体 -15.5%，幅度大、1/3 seeds 赢 Raw，不能归因于纯随机噪声（真实退化 + 部分 seed 波动）。
- electricity primary label: COVERAGE_LOSS | evidence: union coverage 1.000→0.692（30.8% test 窗口无激活 expert），avg active/window 3.53→1.28。 | v2 删除了 raw 覆盖率最高的 E2（82.7%）与 E8（58.6%），属于过度剪枝。 | electricity Spearman=-0.190<0，TRAIN pool_reward 与 VAL contribution 方
- traffic primary label: COMPARABLE | evidence: traffic 相对差异 |rel_diff|<1%（+0.42%），pooled bootstrap CI 跨 0，3 seed 方向不一致（seed42 显著更差 p≈1e-13、seed123 显著更好 p=0.001、seed2026 不显著）=> 均值打平，非明确失败。 | traffic seed42 vs seed123/2026 方向相反、pooled 差异接近 0，属于 seed

## 结果
- 跨数据集（n=7）v2 vs Raw：5/7 胜，mean RI +13.40%，median RI +17.71%。
- 跨数据集（n=7）v2 vs v1：6/7 胜，mean RI +17.77%，median RI +18.34%。
- 因每数据集仅 3 seeds，不计算跨 seed 显著性；不声称 p<0.05。

## 结论
- 统计证据为「描述性 + 方向一致性（VAL→TEST 7/7）」级别；
- 逐窗口 bootstrap 仅对 electricity/traffic 可做（已有），其余 5 数据集无逐窗口 TEST artifact。