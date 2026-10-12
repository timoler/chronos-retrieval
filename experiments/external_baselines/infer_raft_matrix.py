#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAFT 全窗口 VAL/TEST 前向，使用 VAL-best checkpoint（修正 F1 的 checkpoint 选择差异）。

官方 test 用 train() 结束时的 last-epoch 模型（run.py 未接收 train 返回的 best_model）。
本脚本重建模型（含 prepare_dataset 检索）+ 加载 checkpoint.pth（VAL-best）+ 前向。
输出原始值空间预测到 results/paer_aligned_backbone_matrix/backbone/<ds>/。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

RAFT_DIR = Path("E:/GitHub_Project/_external_repos/RAFT")
sys.path.insert(0, str(RAFT_DIR))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # noqa: E402

OUT_DIR = Path("E:/GitHub_Project/chronos-retrieval/results/paer_aligned_backbone_matrix/backbone")
SEEDS = [42, 123, 2026]
BATCH = 32


def build_args(seq_len, pred_len):
    a = argparse.Namespace()
    a.task_name = "long_term_forecast"
    a.model = "RAFT"
    a.data = "chronos_retrieval"
    a.root_path = ""   # 运行时填
    a.data_path = "series.csv"
    a.features = "S"; a.target = "OT"; a.freq = "h"
    a.seq_len = seq_len; a.label_len = 48; a.pred_len = pred_len
    a.enc_in = 1; a.dec_in = 1; a.c_out = 1
    a.n_period = 3; a.topm = 20
    a.d_model = 512; a.n_heads = 8; a.e_layers = 2; a.d_layers = 1; a.d_ff = 2048
    a.moving_avg = 25; a.factor = 1; a.embed = "timeF"; a.dropout = 0.1
    a.activation = "gelu"; a.output_attention = False; a.distil = True
    a.batch_size = BATCH; a.num_workers = 0; a.gpu = 0; a.use_gpu = True; a.use_multi_gpu = False
    a.inverse = True; a.use_amp = False; a.seasonal_patterns = "M"; a.timeenc = 0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset
    root = Path("E:/GitHub_Project/chronos-retrieval/results/multidataset") / ds
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    n_val, n_test = int(cfg["num_windows"]["val"]), int(cfg["num_windows"]["test"])
    out = OUT_DIR / ds
    out.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        a = build_args(context, horizon)
        a.root_path = str(root)
        exp = Exp_Long_Term_Forecast(a)   # 建模型 + prepare_dataset 检索
        model = exp.model
        setting = (f"long_term_forecast_batch_{ds}_s{seed}_RAFT_chronos_retrieval_ftS_sl{context}_ll48_pl{horizon}_"
                   f"dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0")
        ckpt = RAFT_DIR / "checkpoints" / setting / "checkpoint.pth"
        model.load_state_dict(torch.load(ckpt, map_location=exp.device))

        def forward(flag, mode):
            dset, loader = exp._get_data(flag=flag)
            all_preds = []
            all_indices = []
            model.eval()
            with torch.no_grad():
                for index, batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                    batch_x = batch_x.float().to(exp.device)
                    outputs = model(batch_x, index, mode=mode)
                    outputs = outputs[:, -horizon:, :]
                    all_preds.append(outputs.detach().cpu().numpy())
                    all_indices.append(index.cpu().numpy())
            preds = np.concatenate(all_preds, axis=0)   # [n, horizon, 1]
            indices = np.concatenate(all_indices, axis=0)  # [n]（val loader shuffle=True，需按 index 复原顺序）
            preds = preds[np.argsort(indices)]
            preds = dset.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
            return preds[:, :, 0]

        vp = forward("val", "valid")
        tp = forward("test", "test")
        assert vp.shape == (n_val, horizon) and tp.shape == (n_test, horizon), f"{ds}/seed{seed} {vp.shape}/{tp.shape}"
        np.save(out / f"raft_{ds}_s{seed}_val.npy", vp.astype(np.float32))
        np.save(out / f"raft_{ds}_s{seed}_test.npy", tp.astype(np.float32))
        print(f"  [{ds}] seed{seed}: val={vp.shape} test={tp.shape}", flush=True)


if __name__ == "__main__":
    main()
