# P7 Expert Pool / Coverage 分析

| Dataset | raw | v1 | v2 | pool retention | coverage raw→v2 | v1 collapse |
|---|---|---|---|---|---|---|
| etth1 | 8 | 8 | 7 | 0.875 | — | 否 |
| etth2 | 8 | 8 | 6 | 0.75 | — | 否 |
| weather | 8 | 3 | 7 | 0.875 | — | 否 |
| exchange_rate | 8 | 2 | 6 | 0.75 | — | 否 |
| solar | 8 | 5 | 5 | 0.625 | — | 否 |
| electricity | 8 | 1 | 5 | 0.625 | 0.692 | 是 |
| traffic | 8 | 4 | 5 | 0.625 | 0.974 | 否 |

## 结果
- **Electricity v1 collapse**：v1 pool 8→1（唯一 KEEP=E6），v2 恢复到 5。
- **Electricity coverage loss**：union coverage 1.000→0.692（30.8% TEST 窗口无激活 expert），v2 删除了 raw 覆盖率最高的 E2(82.7%) 与 E8(58.6%)，属过度剪枝。
- **Traffic**：coverage 1.000→0.974（接近无损），相对差异 <1%，comparable 而非明确失败。
- 其余 5 数据集：raw 恒为 8，v2 保留 5-7，无 v1 塌缩现象。

## 结论
- v2 相比 v1 的主要机制收益 = 避免 pool collapse（尤其 electricity），
  但 v2 在 electricity 上仍因 coverage loss 输给 Raw，说明 pool-aware reward 并不保证覆盖保持。