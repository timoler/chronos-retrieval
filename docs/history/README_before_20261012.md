# chronos-retrieval

验证「历史模式检索」(historical-pattern retrieval) 能否提升 [Chronos-2](https://github.com/amazon-science/chronos-forecasting) 的时间序列预测精度。

## 思路

从训练序列构建一个**记忆库**(滑动窗口切出的 `(上下文, 未来)` 对),预测时用相似度检索出与当前上下文最相似的历史窗口,softmax 加权得到「记忆预测」,再与 Chronos-2 的预测做线性融合:

```
y = (1 - λ) · y_chronos + λ · y_memory
```

- `λ = 0` 即纯 Chronos-2 baseline
- 在 `TopK × λ` 网格上扫描,比较 MSE / MAE

## 目录结构

```
chronos-retrieval/
├── retrieval/                    # 核心包(与 Chronos-2 源码解耦)
│   ├── memory_bank.py            # 记忆库:滑动窗口 (context, future) 对
│   ├── similarity.py             # Pearson / 余弦 相似度
│   ├── retriever.py              # top-k 检索
│   ├── decompose.py              # 去趋势 / 结构分解 (T+S+R) 工具
│   └── fusion.py                 # softmax 加权 + λ 融合
├── run_retrieval_experiment.py   # 主实验:TopK × λ 网格
├── verify_pipeline.py            # 管线正确性校验
├── compare_*.py / ablate_*.py / diagnose_*.py / scan_*.py   # 消融与诊断脚本
├── data/                         # ETT 数据集(已纳入版本控制,clone 即可用)
└── results/                      # 实验输出(不入库,运行后生成)
```

## 依赖

主实验需要:

- Python 3.11+
- `torch`(本机验证环境为 2.11.0+cu128,GPU)
- `numpy` / `pandas` / `matplotlib` / `scikit-learn`
- [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)(提供 `chronos.Chronos2Pipeline`)

`chronos` 的安装(本机是从源码 editable 安装的 2.3.1):

```bash
git clone https://github.com/amazon-science/chronos-forecasting.git
cd chronos-forecasting
pip install -e .
```

> 国内访问 HuggingFace 会超时,脚本里已通过 `HF_ENDPOINT=https://hf-mirror.com` 走镜像下载 `amazon/chronos-2` 权重。

## 数据

`data/` 目录包含 ETT 数据集:

- `ETTh1.csv` / `ETTh2.csv` — 小时级
- `ETTm1.csv` / `ETTm2.csv` — 15 分钟级

## 运行

```bash
# 主实验(Chronos-2 baseline + 检索融合, TopK × λ 网格)
python run_retrieval_experiment.py

# 只看 baseline
python run_retrieval_experiment.py --baseline-only

# 加大测试窗口步长(减少窗口数、加速)
python run_retrieval_experiment.py --stride 8
```

输出写到 `results/`:`results.csv`(网格结果)、`topk_scores.csv`、`cases_summary.csv` 以及 `cases/*.png`(样本对比图)。

## 主要配置

`run_retrieval_experiment.py` 顶部常量:

- `MODEL = "amazon/chronos-2"`
- `CONTEXT_LENGTH = 512`、`PREDICTION_LENGTH = 96`
- `TOP_K_LIST = [1, 3, 5, 10]`、`LAMBDA_LIST = [0.0, 0.1, ..., 0.5]`
