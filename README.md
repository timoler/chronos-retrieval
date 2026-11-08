# PAER

**Pool-Aware Expert Refinement** 将结构化专家知识用于时间序列历史案例检索。每个窗口根据自身的趋势、波动、持续性或日历条件激活相应知识。检索器结合窗口的数值特征、激活后的文本向量与残差形状，为历史案例排序。入选案例的后续轨迹经均值对齐后形成记忆预测，再按验证集选定的比例与基础预测模型融合。

知识修订包含两种比较。在当前池中逐条屏蔽知识并重训，可以评价该条知识的条件贡献。保持其他条目固定、比较同一条知识的新旧版本，可以判断修改是否降低原始记忆预测误差。训练期决策使用四个按时间扩展的折，部署使用冻结后的知识池。

## 快速开始

核对已发布的七任务结果、五任务组件对照与 Weather 新月份汇总，只需 Python 标准库。

```bash
python tools/verify_results.py
```

安装核心数值包，然后运行实现自检。已有文本向量随项目提供。

```bash
python -m pip install -e .
python -m paer check
python tools/verify_inputs.py
```

以下命令在 ETTh2 上训练冻结知识池对应的检索器，读取随仓库提供的 Chronos-2 验证和测试预测，在 VAL 选取融合权重，输出逐窗口预测及 MSE、MAE。运行输出写入新目录。CPU 或不同 CUDA 环境可能产生数值差异。

```bash
python -m paer deploy --series inputs/etth2/series.csv --protocol inputs/etth2/protocol.json --pool inputs/etth2/experts.json --embeddings inputs/embeddings --base-val inputs/etth2/backbone/chronos2_val.npz --base-test inputs/etth2/backbone/chronos2_test.npz --seed 42 --out runs/etth2
```

原运行环境下，这个示例在 VAL 选得 λ = 0.2，在 TEST 得到 MSE 17.14149828、MAE 2.88340658。项目还提供旧验证窗口上的数值一致性检查记录。[实现核验](docs/validation.md)

若要从官方观测重建 Weather 新月份的标签，使用保存的预测核对完整结果，可运行 `python heldout/verify.py`。

## 材料

| 内容 | 位置 |
|---|---|
| 数值编码、条件激活、案例评分及记忆预测 | [paer/](paer/) |
| 成对替换、四折贡献筛选和最终归档 | [refine.py](paer/refine.py) |
| 七个任务的目标序列、切分、初始知识、修订提议和冻结池 | [inputs/](inputs/README.md) |
| 知识反馈与修订的提示及可执行条件 | [知识处理说明](docs/knowledge.md) |
| 四个底座、七项任务、三个种子的已完成结果 | [实验与核验说明](evidence/README.md) |
| 耶拿屋顶站后续完整月份的预测与复核 | [独立月份材料](heldout/) |

## 实验结果

在已查看过的七项任务测试区间，PAER 相对单独底座的三种子平均 MSE 在 28 个组合中下降 25 个、持平 1 个、上升 2 个。相对初始知识分支，下降 13 个、持平 1 个、上升 14 个。五任务的组件实验涵盖无知识检索、只筛选、只修订、关闭条件激活及打乱文本等对照。[逐种子主表](evidence/main/per_seed.csv) · [组件对照](evidence/controls/per_seed.csv)

Weather 的后续评价固定使用耶拿屋顶站 2021 年 1 月的 3,857 个预测窗口。相对底座，PAER 在 iTransformer 和 RAFT 上分别降低 MSE 3.30% 和 1.72%，在 PatchTST 上持平，在 Chronos-2 上使 MSE 上升 5.33%。各分支使用原验证集选定的权重。[后续月份逐种子结果](evidence/weather_2021_01/per_seed.csv)
