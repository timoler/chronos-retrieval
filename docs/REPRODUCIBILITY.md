# 复核与复现

## 缓存层审计

依赖Python、NumPy、pandas。本轮使用E:/Miniconda3/envs/myenv/python.exe及NumPy 2.4.4。

从仓库根目录执行，输出选用新的目录：

```powershell
& 'E:/Miniconda3/envs/myenv/python.exe' scripts/audit_results.py --repo . --out work/audit_recheck --paired-workspace 'C:/Users/陈佳政/Documents/Codex/2026-10-11/https-github-com-timoler-chronos-retrieval'
```

paired-workspace只用于Raw/PAER三个数据集的检索级重建；省略它将省略该部分。主审计需要results/deepseek_matrix_delivery、冻结数据/结果及相邻_external_repos中的checkpoint。输入只读；--out内同名审计产物会被覆盖，请选新的目录。

本次固定审计结果在docs/audits/2026-10-12，包含输入哈希、63份checkpoint索引、逐seed指标和自动检查。包装哈希通过仅证明内容一致；不证明原模型执行历史、checkpoint最佳性或数据选择独立性。

## 模型级复现边界

本轮没有重新训练或GPU推理。矩阵脚本加载的63份checkpoint均存在，已记录当前哈希；交付包原先仅记录3份示例。外部仓库adapter与checkpoint仍需随正式复现实验固定版本。PAER优化阶段通常冻结的是专家池、协议与seed，未普遍保存每次训练模型权重。

## 数据与评价

当前权威输入：results/multidataset/<dataset>/{series.csv,config.json,window_indices.npz}。ETT目录大小写应跨平台核对。窗口起点s的目标是s+L开始的H步；所有方法须对同一索引顺序输出完整窗口。

新矩阵使用完整VAL和TEST、逐底座逐seed的VAL λ选择、ddof=1标准差；原内部结果/旧融合使用ddof=0，混合报告时必须标明或统一重算。训练奖励仍是代理评价；不得写严格滚动验证。

## 数据安全与版本

results预测缓存和交付包很大，继续保留本地路径。新增.gitignore例外仅开放小型报告和清单，未自动提交、上传数组或推送。克隆仓库缺缓存时应报告缺失，不能把缺失视作已复现。
