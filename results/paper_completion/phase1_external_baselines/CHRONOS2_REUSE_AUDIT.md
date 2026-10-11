# Chronos-2 复用审计（CHRONOS2_REUSE_AUDIT）

- 生成日期：2026-10-09
- 审计对象：仓库内已有 Chronos-2 预测文件 vs 正式协议（`results/multidataset/<ds>/config.json` + `window_indices.npz`）
- 结论：**0/7 可复用，7/7 需补跑**。无任何数据集在正式协议下已有 Chronos-2 预测。
- 原则：只审计，不重跑，不猜测。

---

## 1. 正式协议（权威口径）

| 数据集 | context | horizon | test 窗口数 | val 窗口数 | 备注 |
|---|---|---|---|---|---|
| etth1 | 512 | 96 | 2877 | 2877 | — |
| etth2 | 168 | 24 | 3293 | 3293 | horizon=24（非 96） |
| exchange_rate | 512 | 96 | 911 | 911 | — |
| electricity | 512 | 96 | 27445 | 27444 | 15min、140256 行 |
| weather | 512 | 96 | 9933 | 9932 | 10min |
| traffic | 168 | 24 | 3318 | 3318 | horizon=24（非 96） |
| solar | 512 | 96 | 9905 | 9905 | 10min |

## 2. 已有 Chronos-2 预测文件核对

仓库内 `results/*_chronos.npz` 是**旧「日历 date split」**（train=[2016-07-01,2017-07-01) 等）生成的缓存，**不是**正式协议的行区间 split（`[0,10452]/[10452,13936]/[13936,17420]` 等）。

| 数据集 | 已有 npz | 内容 | 与正式协议对比 | 判定 |
|---|---|---|---|---|
| etth1 | `ETTh1_chronos.npz` | val(2857,96) test(5613,96) | test 5613 ≠ 2877，split 不同 | **RERUN_REQUIRED** |
| etth2 | `ETTh2_chronos.npz` | val(2857,96) test(5613,96) | horizon 96 ≠ 24 且 test 5613 ≠ 3293 | **RERUN_REQUIRED** |
| exchange_rate | 无 | — | — | **RERUN_REQUIRED** |
| electricity | `electricity_chronos.npz` | 仅 val(6989,96) | 无 test；val 6989 ≠ 27444 | **RERUN_REQUIRED** |
| weather | 无 | — | — | **RERUN_REQUIRED** |
| traffic | 无 | — | — | **RERUN_REQUIRED** |
| solar | `solar_chronos.npz` | 仅 val(1737,96) | 无 test；val 1737 ≠ 9905 | **RERUN_REQUIRED** |

## 3. 环境风险（补跑前必须先验证）

`results/external_baselines/PROGRESS.md` §3/§7.2 记录：

- 当前 `chronos` 2.3.1（editable 装于 `E:\GitHub_Project\chronos-forecasting-main\src`）与缓存的 `amazon/chronos-2` config **不兼容**（`input_patch_size` 字段报错）。
- `Chronos2Pipeline` 可 import，但 `from_pretrained("amazon/chronos-2")` 是否真能加载需**第一步验证**（`experiments/external_baselines/smoke_chronos2.py` 就是为此而写，但当前结果未落地）。
- `amazon/chronos-t5-small` 已能加载（RAF 官方 backbone 用它，不受影响）。

→ Chronos-2 补跑是否可行，取决于能否解决 chronos 版本与 `amazon/chronos-2` config 的兼容问题。**若无法加载 chronos-2，则 Chronos-2 补跑判 BLOCKED。**

## 4. 逐数据集判定汇总

| 数据集 | 判定 | 原因 |
|---|---|---|
| etth1 | RERUN_REQUIRED | 旧日历 split，test 窗口数不符 |
| etth2 | RERUN_REQUIRED | 旧日历 split + horizon 96≠24 |
| exchange_rate | RERUN_REQUIRED | 无任何 Chronos-2 预测 |
| electricity | RERUN_REQUIRED | 仅 val 预测，无 test |
| weather | RERUN_REQUIRED | 无任何 Chronos-2 预测 |
| traffic | RERUN_REQUIRED | 无任何 Chronos-2 预测 |
| solar | RERUN_REQUIRED | 仅 val 预测，无 test |

**REUSE = 0/7；RERUN_REQUIRED = 7/7；BLOCKED（协议层）= 0/7**（BLOCKED 风险仅在环境层：chronos-2 能否加载）。

## 5. 补跑方式

正式协议下的 Chronos-2 raw 预测，复用 `experiments/external_baselines/run_ours_fusion.py` 内的 `load_formal()` + `run_chronos()`（读 `config.json` + `window_indices.npz`，对 test 起点逐窗预测，取 mean）。建议拆出**只跑 Chronos-2 raw** 的独立脚本（不掺入 fusion），避免与 Ours Fusion 的方法身份问题耦合（见 `OURS_FUSION_AUDIT.md`）。
