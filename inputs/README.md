# 实验输入

每个目录包含实际建模使用的单变量 `series.csv`（列为 `date,OT`）、原始滑动窗口起点 `window_indices.npz`、半开行区间与任务配置 `protocol.json`。其他输入为八条初始知识 `initial.json`、每轮实际提出的候选修订 `proposals.json`，以及最终冻结的知识 `experts.json`。

| 任务 | 目标 | 历史长度 / 预测长度 |
|---|---|---:|
| ETTh1 | 变压器油温 OT | 512 / 96 |
| ETTh2 | 变压器油温 OT | 168 / 24 |
| Exchange Rate | 第七条汇率序列，关闭日历条件 | 512 / 96 |
| Traffic | Sensor_000 占用率 | 168 / 24 |
| Weather | 耶拿屋顶站 CO₂ 浓度，单位 ppm | 512 / 96 |
| Solar | 137 路发电量之和 | 512 / 96 |
| Electricity | 370 个用户负荷之和，保留原始零值 | 512 / 96 |

每个任务按原始行划分 TRAIN、VAL、TEST，并只保留输入和目标完全在所属区间内的窗口。Traffic 与 Solar 使用重建的等间隔时间索引。Weather 用历史值前向填补 50 条缺测观测。检索器在每一折用当时可用的候选上下文估计缩放统计量。各序列保留原始目标尺度。

`embeddings/` 保存实际用到的冻结 384 维 all-MiniLM-L6-v2 文本向量。文件名来自机制文本的 JSON 编码哈希。知识记录中的检索建议用于说明机制，数值计算读取 `condition_spec` 和 `mechanism`。输入文件完整性可用 `python tools/verify_inputs.py` 核对。

ETTh 数据来自 ETT，Exchange Rate、Traffic 和 Solar 来自公开的多变量时序基准数据。Weather 数据归 Max Planck Institute for Biogeochemistry, Jena 所有，许可为 CC BY 4.0。Electricity 来源为 A. Trindade 的 ElectricityLoadDiagrams20112014，许可为 CC BY 4.0。[ETT](https://github.com/zhouhaoyi/ETDataset) · [多变量时序数据](https://github.com/laiguokun/multivariate-time-series-data) · [屋顶站数据](https://www.bgc-jena.mpg.de/wetter/weather_data.html) · [UCI 电力数据](https://doi.org/10.24432/C58C86) · [文本编码器模型卡](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
