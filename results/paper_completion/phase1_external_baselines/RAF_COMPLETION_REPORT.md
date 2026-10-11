# RAF 补全报告（RAF_COMPLETION_REPORT）

- 生成日期：2026-10-09
- 结论：**RAF 4/7 完成，缺 3 个数据集（weather / solar / electricity，seed42）。** 与上次审计（memory: external baseline repro state）一致，缺失情况未变化。

---

## 1. 当前实际状态（以仓库文件为准）

| 数据集 | seed | status | 结果来源 |
|---|---|---|---|
| etth1 | 42 | ✅ OK | `external_baseline_results_full.csv` + `raf_result_etth1.json` |
| etth2 | 42 | ✅ OK | `external_baseline_results_full.csv` + `raf_result_etth2.json` |
| exchange_rate | 42 | ✅ OK | `external_baseline_results_full.csv` + `raf_result_exchange_rate.json` |
| traffic | 42 | ✅ OK | `external_baseline_results_full.csv` + `raf_result_traffic.json` |
| weather | 42 | ❌ 缺失 | `logs/RAF/weather_s42.log` 存在但**未完成**（无 `raf_result_weather.json`，日志尾部只有 Chronos「prediction length ≤ 64」提示刷屏，被中途杀死） |
| solar | 42 | ❌ 缺失 | 无日志、无结果 |
| electricity | 42 | ❌ 缺失 | 无日志、无结果 |

RAF 无训练参数（zero-shot，backbone `amazon/chronos-t5-small`），seed 只影响 Chronos 采样；官方协议下只需 seed42（`--raf-seeds 42`）。已在 `run_batch.py` / `raf_search_space.json` 记录 `num_samples=5`（官方默认 20，为可行性降采样，点预测取中位数不变）。

## 2. 缺失组合（需补跑）

- `RAF / weather / seed42`
- `RAF / solar / seed42`
- `RAF / electricity / seed42`

## 3. 补跑命令（单并发，训练 baseline 已全部完成，不与训练争抢 GPU）

```bash
cd /e/GitHub_Project/chronos-retrieval
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True E:/Miniconda3/envs/myenv/python \
  experiments/external_baselines/run_batch.py --baselines RAF --max-workers 1 \
  > results/external_baselines/batch_raf.log 2>&1
```

- 断点=CSV（`status=OK` 自动跳过，已完成的 4 个不会重跑）。
- 每个数据集产出 `raf_result_<ds>.json`（含 lambda）+ `raf_<ds>_leak_audit.json`。
- 预计时长（参考 PROGRESS.md）：weather/solar 各约数小时，electricity 约 1h 检索 + Chronos forward 主导，合计约 6.5h（`num_samples=5`）。

## 4. 协议合规要点（已在 `raf_rolling.py` 内实现 + 泄漏断言）

- 相同 TEST 时间范围：读 `window_indices.npz` 的 test 起点。
- 相同预测长度 / 目标变量：`series.csv` 的 OT，horizon 由 `config.json`。
- 相同 MSE 口径：原始值 MSE。
- TRAIN/VAL 不接触 TEST：`raf_rolling.py` 带 `raf_<ds>_leak_audit.json` 泄漏断言。

## 5. 备注

- RAF 是官方 `chronos-raf.py` 的滚动适配（backbone=chronos-t5-small），非 Chronos-2。它与 Ours Fusion 的方法身份冲突**无关**，可独立补跑。
