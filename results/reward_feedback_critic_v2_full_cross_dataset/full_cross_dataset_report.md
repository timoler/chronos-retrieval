# Reward-Feedback Critic v2 全量跨数据集验证（Pool-Aware Reward）

冻结 v2（Pool-Aware Reward），跑剩余 4 个数据集，合并已有 3 个，共 7 数据集。
不访问 TEST、不改方法、不挑 seed、失败数据集保留并标记。

## 结果

- v2 比 Raw 好：5/7
- v2 比 v1 好：4/7
- mean RI vs Raw = +12.24%
- median RI vs Raw = +12.02%
- mean RI vs v1 = +11.39%
- median RI vs v1 = +0.68%
- 3/3 seeds 赢 Raw：3/7
- 2/3 或 3/3 seeds 赢 Raw：4/7
- v2 std < Raw std：4/7
- v2 比 Raw 差的数据集：['electricity', 'traffic']

主表：

| Dataset | Raw | v1 | v2 | v2 vs Raw | v2 vs v1 | wins/raw | Pool v1→v2 |
|---|---|---|---|---|---|---|---|
| etth1 | 63.9024 | 41.2939 | 34.6889 | +45.7% | +16.0% | 3/3 | 8→7 |
| electricity | 2.68541e+08 | 9.72416e+08 | 3.27719e+08 | -22.0% | +66.3% | 1/3 | 1→5 |
| weather | 530.951 | 462.081 | 458.951 | +13.6% | +0.7% | 3/3 | 3→7 |
| etth2 | 206.149 | 214.941 | 121.846 | +40.9% | +43.3% | 3/3 | 8→6 |
| solar | 198826 | 192707 | 197100 | +0.9% | -2.3% | 1/3 | 5→5 |
| exchange_rate | 0.0363535 | 0.0222776 | 0.0319826 | +12.0% | -43.6% | 2/3 | 2→6 |
| traffic | 0.000684264 | 0.000715604 | 0.00072093 | -5.4% | -0.7% | 0/3 | 4→5 |

Pool size 是否仍有明显 collapse：

| Dataset | Raw pool | v1 pool | v2 pool |
|---|---|---|---|
| etth1 | 8 | 8 | 7 |
| electricity | 8 | 1 | 5 |
| weather | 8 | 3 | 7 |
| etth2 | 8 | 8 | 6 |
| solar | 8 | 5 | 5 |
| exchange_rate | 8 | 2 | 6 |
| traffic | 8 | 4 | 5 |

### FAIL_DATASET: electricity
- v2 vs Raw = -22.0%（v2=3.27719e+08，Raw=2.68541e+08）
- possible causes: marginal reward mismatch / pool over-pruning / weak expert signal / optimization sensitivity / combination effect

### FAIL_DATASET: traffic
- v2 vs Raw = -5.4%（v2=0.00072093，Raw=0.000684264）
- possible causes: marginal reward mismatch / pool over-pruning / weak expert signal / optimization sensitivity / combination effect

## 结论

- **STRONG SUCCESS**
  （v2 多数数据集优于 Raw，mean_RI_vs_raw>0，median_RI_vs_raw>0，且多数优于 v1。）

```
test_used = false
deterministic_protocol = true
method_frozen = v2_pool_aware_reward
```