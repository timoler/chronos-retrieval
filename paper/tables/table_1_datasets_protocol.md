# Table 1: 数据集与协议

| Dataset | Source file | Target | Frequency | Context L | Horizon H | TRAIN windows | VAL windows | TEST windows | Aggregation / transform |
|---|---|---|---|---|---|---|---|---|---|
| etth1 | data/ETTh1.csv | OT | 1h | 512 | 96 | 9845 | 2877 | 2877 | univariate OT |
| etth2 | data/ETTh2.csv | OT | hourly | 168 | 24 | 10261 | 3293 | 3293 | first series (OT), LOCF |
| exchange_rate | data/exchange_rate.csv | OT | daily | 512 | 96 | 3945 | 911 | 911 | univariate OT |
| traffic | data/traffic.csv | Sensor_000 | hourly | 168 | 24 | 10335 | 3318 | 3318 | first series (Sensor_000), LOCF |
| weather | data/weather.csv | OT | 10min | 512 | 96 | 31010 | 9932 | 9933 | univariate OT |
| solar | data/solar.csv | OT | 10min | 512 | 96 | 30929 | 9905 | 9905 | univariate OT |
| electricity | data/electricity.csv | OT | 15min | 512 | 96 | 83546 | 27444 | 27445 | univariate OT |

所有数据集 stride=1 滑窗；单变量目标（OT 或首列），无跨序列汇聚。etth2/traffic 按 config 的 deterministic first-series 规则选列并对缺失值 LOCF 前向填充。
