#!/usr/bin/env bash
# 串行续跑 Phase1 外部 baseline 对比（2026-10-09）。
# 顺序：RAF 补跑 3 缺失集 -> Ours Fusion + Chronos-2 -> 生成最终主表。
# 每步独立记日志 + echo 退出码；断点=CSV，可重复执行（已 OK 自动跳过）。
set -u
cd /e/GitHub_Project/chronos-retrieval

# Windows 重定向 stdout 默认 GBK，aggregate 里的 '−'(U+2212) 会崩，强制 UTF-8
export PYTHONIOENCODING=utf-8

PY=("E:/Miniconda3/envs/myenv/python")

echo "=========================================================="
echo "[STEP 1/3] RAF 补跑 weather/solar/electricity (~6.5h)"
echo "=========================================================="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY[@]}" \
  experiments/external_baselines/run_batch.py --baselines RAF --max-workers 1 \
  > results/external_baselines/batch_raf.log 2>&1
echo "[STEP 1] exit=$?"

echo "=========================================================="
echo "[STEP 2/3] Ours Fusion + Chronos-2 (7 数据集, ~3-6h)"
echo "=========================================================="
"${PY[@]}" experiments/external_baselines/run_ours_fusion_v2.py --resume \
  > results/paper_completion/phase1_external_baselines/ours_fusion_v2_run.log 2>&1
echo "[STEP 2] exit=$?"

echo "=========================================================="
echo "[STEP 3/3] 生成最终主表"
echo "=========================================================="
"${PY[@]}" experiments/external_baselines/aggregate_final_table.py \
  > results/paper_completion/phase1_external_baselines/aggregate_final_table.log 2>&1
echo "[STEP 3] exit=$?"

echo "=========================================================="
echo "ALL DONE"
echo "=========================================================="
