# Table 4: 知识优化过程统计

| Dataset | N initial (raw) | N final (PAER) | Pool size r0/r1/r2 | Text-changed (kept) | Condition-changed (kept) | Deleted | Mechanism-replaced |
|---|---|---|---|---|---|---|---|
| etth1 | 8 | 7 | 8/5/5 | 4 | 4 | 1 | 4 |
| etth2 | 8 | 6 | 8/5/2 | 3 | 3 | 2 | 3 |
| exchange_rate | 8 | 6 | 8/7/5 | 5 | 5 | 2 | 5 |
| traffic | 8 | 5 | 8/5/3 | 2 | 0 | 3 | 2 |
| weather | 8 | 7 | 8/3/2 | 2 | 2 | 1 | 1 |
| solar | 8 | 5 | 8/7/3 | 4 | 3 | 3 | 4 |
| electricity | 8 | 5 | 8/5/3 | 2 | 2 | 3 | 2 |

N initial=raw 8；N final=冻结 v2 最终池。Pool size r0/r1/r2 = 各轮实际评价池专家数（仅含待修订知识，KEEP 已移出）。Text/Condition-changed 仅统计最终保留且相对 raw 同 ID 机制文本/激活条件发生变化的条数。Deleted=raw 中被移除条数。Mechanism-replaced=最终保留池中 replaced=True 的条数。
