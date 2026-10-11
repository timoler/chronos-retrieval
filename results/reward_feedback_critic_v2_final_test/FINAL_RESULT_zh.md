# 冻结 v2（Pool-Aware Reward）最终 TEST 结果

- 日期：2026-09-16
- 方法：冻结 v2（Pool-Aware Reward）
- 协议：完全冻结 v2 —— 不调 LLM / Critic / revision / 重新筛 pool，直接读 VAL 冻结的 pool、换 TEST split；TEST 只评一次，不改方法。
- seeds：[42, 123, 2026]，全部通过 determinism 校验（`det_ok=True` / `determinism_pass=True`）。
- 结果目录：`results/reward_feedback_critic_v2_final_test/`（自动报告 `final_test_report.md`）

## 主表（TEST，MSE，3 seed mean）

| 数据集 | Raw | v1 | v2 | v2 vs Raw | v2 vs v1 | seeds 赢 Raw |
|---|---|---|---|---|---|---|
| etth1 | 61.974 | 58.920 | 42.417 | +31.6% | +28.0% | 3/3 |
| etth2 | 125.854 | 132.425 | 98.475 | +21.8% | +25.6% | 3/3 |
| weather | 680.424 | 685.693 | 559.943 | +17.7% | +18.3% | 3/3 |
| exchange_rate | 0.038995 | 0.037252 | 0.036480 | +6.5% | +2.1% | 3/3 |
| solar | 465309 | 347965 | 315416 | +32.2% | +9.4% | 3/3 |
| electricity | 3.080e8 | 6.357e8 | 3.556e8 | −15.5% | +44.1% | 1/3 |
| traffic | 0.001541 | 0.001501 | 0.001547 | −0.4% | −3.1% | 2/3 |

## 汇总（n=7）

| 指标 | 值 |
|---|---|
| v2 vs Raw 胜 | 5/7 |
| v2 vs v1 胜 | 6/7 |
| mean RI vs Raw | +13.40% |
| median RI vs Raw | +17.71% |
| mean RI vs v1 | +17.77% |
| median RI vs v1 | +18.34% |
| 3/3 seeds 全胜 Raw | 5/7 |
| 2/3 或 3/3 seeds 赢 Raw | 6/7 |
| v2 std < Raw std | 3/7 |

## VAL → TEST 对照（v2 vs Raw）

| 数据集 | VAL | TEST | 方向 |
|---|---|---|---|
| etth1 | +45.7% | +31.6% | 保持 |
| etth2 | +40.9% | +21.8% | 保持 |
| weather | +13.6% | +17.7% | 保持 |
| exchange_rate | +12.0% | +6.5% | 保持 |
| solar | +0.9% | +32.2% | 保持 |
| electricity | −22.0% | −15.5% | 保持 |
| traffic | −5.4% | −0.4% | 保持 |

方向保持 7/7。

## 现象记录（只报实验事实）

1. TEST 上 v2 输 Raw 的数据集，正好是 VAL 上输 Raw 的同一对：electricity、traffic。VAL→TEST 方向 7/7 一致。
2. electricity 的差距从 VAL −22.0% 收窄到 TEST −15.5%；traffic 的 −0.4% 接近打平（可视为无差异）。
3. electricity 的 v1 pool 只有 1 个 expert（`v1=1`），故其 "vs v1 +44.1%" 无参考意义（v1 自身退化，v1 MSE 6.357e8 反而高于 Raw 的 3.080e8）；真正要看的 vs Raw 仍差 15.5%。
4. electricity 上 Raw 本身已较强：base 7.445e8 → raw 3.080e8（检索本身 −59%），v2（3.556e8）未在此基础上再赢。
5. v2 std < Raw std 仅 3/7，即跨 seed 波动未普遍压下来。
6. 各数据集 pool 规模：raw/v1/v2 见各 `result.json`（electricity 为 raw=8 v1=1 v2=5，solar 为 raw=8 v1=5 v2=5）。

## 复现环境

- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`
- Python：`E:/Miniconda3/envs/myenv/python.exe`（GPU，CUDA，16GB）
- 关键脚本：`experiments/run_reward_feedback_critic_v2_final_test.py`
