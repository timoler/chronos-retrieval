#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iTransformer 全窗口 VAL/TEST 前向（原 VAL-selected checkpoint），通用版。

iTransformer test pred.npy 已存在（完整窗口），但缺 VAL 预测用于选 λ。本脚本补 VAL + TEST
前向，输出原始值空间预测到 results/paer_aligned_backbone_matrix/backbone/<ds>/。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ITR_DIR = Path("E:/GitHub_Project/_external_repos/iTransformer")
sys.path.insert(0, str(ITR_DIR))

from data_provider.chronos_loader import Dataset_ChronosRetrieval  # noqa: E402
from model.iTransformer import Model  # noqa: E402

OUT_DIR = Path("E:/GitHub_Project/chronos-retrieval/results/paer_aligned_backbone_matrix/backbone")
SEEDS = [42, 123, 2026]
BATCH = 32   # iTransformer 官方 batch


def build_args(seq_len, pred_len):
    a = argparse.Namespace()
    a.seq_len = seq_len; a.label_len = 48; a.pred_len = pred_len
    a.enc_in = 1; a.dec_in = 1; a.c_out = 1
    a.d_model = 512; a.n_heads = 8; a.e_layers = 3; a.d_layers = 1; a.d_ff = 512
    a.factor = 1; a.embed = "timeF"; a.freq = "h"; a.dropout = 0.1
    a.activation = "gelu"; a.output_attention = False
    a.features = "S"; a.target = "OT"; a.data = "chronos_retrieval"
    a.distil = True; a.use_amp = False; a.inverse = True
    a.class_strategy = "projection"; a.use_norm = True
    a.gpu = 0; a.use_gpu = True; a.use_multi_gpu = False
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT_DIR / ds
    out.mkdir(parents=True, exist_ok=True)

    def forward(model, flag):
        dset = Dataset_ChronosRetrieval(root_path=str(root), flag=flag,
                                        size=[context, 48, horizon], features="S",
                                        data_path="series.csv", target="OT", scale=True, timeenc=0, freq="h")
        loader = torch.utils.data.DataLoader(dset, batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False)
        preds = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                dec_inp = torch.zeros_like(batch_y[:, -horizon:, :]).float()
                dec_inp = torch.cat([batch_y[:, :48, :], dec_inp], dim=1).float().to(device)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = 0
                outputs = outputs[:, -horizon:, f_dim:]
                preds.append(outputs.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)          # [n, horizon, 1]
        preds = dset.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
        return preds[:, :, 0]

    for seed in SEEDS:
        a = build_args(context, horizon)
        model = Model(a).float().to(device)
        setting = (f"batch_{ds}_s{seed}_iTransformer_chronos_retrieval_S_ft{context}_sl48_ll{horizon}_"
                   f"pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dttest_projection_0")
        ckpt = ITR_DIR / "checkpoints" / setting / "checkpoint.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        vp = forward(model, "val")
        tp = forward(model, "test")
        assert vp.shape == (n_val, horizon) and tp.shape == (n_test, horizon), f"{ds}/seed{seed} {vp.shape}/{tp.shape}"
        np.save(out / f"itransformer_{ds}_s{seed}_val.npy", vp.astype(np.float32))
        np.save(out / f"itransformer_{ds}_s{seed}_test.npy", tp.astype(np.float32))
        print(f"  [{ds}] seed{seed}: val={vp.shape} test={tp.shape}", flush=True)


if __name__ == "__main__":
    main()
