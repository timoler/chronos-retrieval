# PAER-Aligned 插件 × 底座矩阵

固定插件 `PAER_original_training_mean_aligned_v1` 接入 4 底座 × 7 数据集 × 3 seed 的加装前后对照。
目标：同一固定完整插件接到四个底座上，相比各自底座改善多少。不拆内部贡献、不改机制、不临时加方法。

## 运行

```bash
# 盘点缓存
E:/Miniconda3/envs/myenv/python experiments/external_baselines/run_paer_backbone_matrix.py --inventory-only

# 生成缺失的 PAER-Aligned 记忆（base-agnostic）
E:/Miniconda3/envs/myenv/python experiments/external_baselines/run_paer_backbone_matrix.py --generate-memory --dataset <ds>
```

## 当前进度（诚实状态）

- **记忆（base-agnostic）**：etth1/etth2/traffic/weather OK（复用 Codex 对齐 + 本地 phase2/3 + paer_align_*）；exchange_rate/solar/electricity 由 `--generate-memory` 生成。
- **底座预测**：
  - Chronos-2：7 数据集完整（reaudit 缓存，seed 无关）。
  - PatchTST：完整窗口 VAL/TEST 仅 etth2/traffic 已补（drop_last 尾部丢弃已修）；etth1/exchange_rate/weather/solar/electricity 待补前向。
  - iTransformer：test pred.npy 存在（完整窗口），但缺 VAL 预测，需补 VAL 前向。
  - RAFT：test pred.npy 存在（完整窗口），但 checkpoint 选择差异（F1）需核对，且缺 VAL。

## 交付文件

| 文件 | 状态 |
|---|---|
| `inventory.csv` | ✅ 已生成 |
| `PROTOCOL.json` | ✅ |
| `progress.csv` | 待融合执行后生成 |
| `per_seed_results.csv` / `summary.csv` / `RESULT_TABLE.md` | 待完成 |
| `MANIFEST.json` | 待完成 |
| `predictions/` | 记忆已落盘 `mem/` 子目录 |
