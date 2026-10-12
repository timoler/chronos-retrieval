# 附录 A6: 旧 v1 评分选池（历史控制，VAL MSE）

| Dataset | K | PAER (pool-aware) | V1_SCORE_TOPK (historical v1) | PAER vs V1_SCORE_TOPK (RI) |
|---|---|---|---|---|
| etth1 | 7 | 34.69 | 56.23 | +38.3% |
| etth2 | 6 | 121.8 | 203.9 | +40.2% |
| exchange_rate | 6 | 0.03198 | 0.03327 | +3.9% |
| traffic | 5 | 0.000721 | 0.000687 | -5.0% |
| weather | 7 | 459 | 534.5 | +14.1% |
| solar | 5 | 1.971e+05 | 2.082e+05 | +5.3% |
| electricity | 5 | 3.28e+08 | 2.74e+08 | -19.8% |
| 7-dataset (equal-weight RI) | — | — | — | +11.0% |

历史控制：v1 逐条 individual reward 评分取 Top-K 选池（V1_SCORE_TOPK），与 PAER(pool-aware) 同 K 对照。V1_SCORE_TOPK 为旧 v1 评分结果，仅作历史控制，不纳入正式方法排名。
