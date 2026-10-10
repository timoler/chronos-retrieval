# Table 6: 覆盖与失败诊断（Electricity / Traffic）

| Dataset | Union coverage (Raw) | Union coverage (PAER) | Avg active experts (Raw) | Avg active experts (PAER) | No-coverage ratio (PAER) | PAER vs Raw (RI, TEST) |
|---|---|---|---|---|---|---|
| electricity | 1.000 | 0.692 | 3.53 | 1.28 | 30.8% | -15.5% |
| traffic | 1.000 | 0.974 | 未记录 | 未记录 | 未记录 | -0.4% |

覆盖阈值 0.5（激活≥0.5 计为激活）。electricity 的 avg active experts / no-coverage 来自逐窗口激活诊断（results/diagnose_v2_failure_electricity_traffic/）；traffic 仅记录 union coverage（1.000→0.974），逐窗口激活专家数未记录。本节为事后诊断（post-hoc），覆盖/激活为描述性分析，非模型输入；模型输入保留连续软激活。electricity 无覆盖窗口集中在高 volatility(Q4)/高 level(Q4) 与凌晨 3–6 点。
