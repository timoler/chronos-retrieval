# Frozen v2 Final TEST Evaluation（Pool-Aware Reward）

完全冻结 v2，不调用 LLM / Critic / revision / 重新筛 pool，直接读 VAL 冻结 pool 换 TEST split。
TEST 只评估一次，禁止回头改 v2。

## 结果

- TEST 上 v2 比 Raw 好：5/7
- TEST 上 v2 比 v1 好：6/7
- mean RI vs Raw = +13.40%
- median RI vs Raw = +17.71%
- mean RI vs v1 = +17.77%
- median RI vs v1 = +18.34%
- 3/3 seeds 赢 Raw：5/7
- 2/3 或 3/3 seeds 赢 Raw：6/7
- v2 std < Raw std：3/7
- VAL→TEST 方向保持：7/7
- TEST 上 v2 比 Raw 差的数据集：['electricity', 'traffic']

主表：

| Dataset | Raw | v1 | v2 | v2 vs Raw | v2 vs v1 | wins/raw |
|---|---|---|---|---|---|---|
| etth1 | 61.974 | 58.9196 | 42.4171 | +31.6% | +28.0% | 3/3 |
| etth2 | 125.854 | 132.425 | 98.4748 | +21.8% | +25.6% | 3/3 |
| weather | 680.424 | 685.693 | 559.943 | +17.7% | +18.3% | 3/3 |
| exchange_rate | 0.0389951 | 0.037252 | 0.0364798 | +6.5% | +2.1% | 3/3 |
| solar | 465309 | 347965 | 315416 | +32.2% | +9.4% | 3/3 |
| electricity | 3.0796e+08 | 6.35739e+08 | 3.55565e+08 | -15.5% | +44.1% | 1/3 |
| traffic | 0.00154063 | 0.00150109 | 0.00154713 | -0.4% | -3.1% | 2/3 |

VAL → TEST 对照：

| Dataset | VAL v2 vs Raw | TEST v2 vs Raw | Direction preserved |
|---|---|---|---|
| etth1 | +45.7% | +31.6% | yes |
| etth2 | +40.9% | +21.8% | yes |
| weather | +13.6% | +17.7% | yes |
| exchange_rate | +12.0% | +6.5% | yes |
| solar | +0.9% | +32.2% | yes |
| electricity | -22.0% | -15.5% | yes |
| traffic | -5.4% | -0.4% | yes |

### FAIL_DATASET: electricity
- TEST v2 vs Raw = -15.5%

### FAIL_DATASET: traffic
- TEST v2 vs Raw = -0.4%

## 结论

- **STRONG GENERALIZATION**
  （多数 TEST 数据集优于 Raw，mean/median RI vs Raw > 0。）

```
test_used = true
test_tuning = false
expert_critic_calls = false
pool_revision = false
method_frozen = v2_pool_aware
```