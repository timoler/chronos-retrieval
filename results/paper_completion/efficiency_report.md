# P5 效率开销（RAW / REWARD_V1 / POOL_AWARE_V2）

实测环境：RTX 5070 Ti / 16GB，代表性数据集 etth1，每项 3 次取 median。
单次 final retriever 训练（30 epoch）：64.764s；单窗推理：0.289ms；峰值显存：241.6MB。

| Dataset | raw pool | v1 pool | v2 pool | v1 eval 次数 | v2 reward 训练次数 | v1 val 时间(估) | v2 val 时间(估) |
|---|---|---|---|---|---|---|---|---|
| etth1 | 8 | 8 | 7 | 10 | 108 | 80.95s | 6994.47s |
| etth2 | 8 | 8 | 6 | 16 | 108 | 129.53s | 6994.47s |
| weather | 8 | 3 | 7 | 18 | 108 | 145.72s | 6994.47s |
| exchange_rate | 8 | 2 | 6 | 22 | 108 | 178.1s | 6994.47s |
| solar | 8 | 5 | 5 | 22 | 108 | 178.1s | 6994.47s |
| electricity | 8 | 1 | 5 | 23 | 108 | 186.2s | 6994.47s |
| traffic | 8 | 4 | 5 | 20 | 108 | 161.91s | 6994.47s |

## 结果
- 最终推理开销三者相同（同一 retriever，同一 30-epoch 训练，同一单窗推理）。
- v2 pool-aware reward 比 v1 逐条 reward 贵：每轮要 N_folds × (N_experts+1) 次 retriever 训练（v1 是逐条 fold-mean-gain，不做 full-minus 重训）。
- 但 reward 只发生在 TRAIN 阶段，不进入推理；最终 TEST/推理的 wall-clock 三者无差异。

## 结论
- v2 的额外成本 = TRAIN-only 的 pool-aware reward 计算（一次性），不增加推理延迟。
- 时间估算基于解析次数 × 实测单次训练时间；v1 val 时间为保守折算（未重跑 v1 全流程）。