#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PatchTST 全窗口前向（修复尾部丢弃），通用版，覆盖 7 数据集 × 3 seed。

用原协议 VAL-selected checkpoint，VAL/TEST shuffle=False + drop_last=False 全窗口前向，
输出原始值空间预测到 results/paer_aligned_backbone_matrix/backbone/<ds>/。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PATCHTST_DIR = Path("E:/GitHub_Project/_external_repos/PatchTST/PatchTST_supervised")
sys.path.insert(0, str(PATCHTST_DIR))

from data_provider.chronos_loader import Dataset_ChronosRetrieval  # noqa: E402
from models import PatchTST  # noqa: E402

OUT_DIR = Path("E:/GitHub_Project/chronos-retrieval/results/paer_aligned_backbone_matrix/backbone")
SEEDS = [42, 123, 2026]
BATCH = 128


def build_args(seq_len, pred_len):
    a = argparse.Namespace()
    a.enc_in = 1; a.dec_in = 1; a.c_out = 1
    a.seq_len = seq_len; a.label_len = 48; a.pred_len = pred_len
    a.e_layers = 3; a.d_model = 128; a.n_heads = 16; a.d_ff = 256
    a.dropout = 0.2; a.fc_dropout = 0.2; a.head_dropout = 0.0
    a.patch_len = 16; a.stride = 8; a.padding_patch = "end"; a.individual = 0
    a.revin = 1; a.affine = 0; a.subtract_last = 0
    a.decomposition = 0; a.kernel_size = 25
    a.features = "S"; a.target = "OT"; a.freq = "h"; a.data = "chronos_retrieval"
    a.batch_size = BATCH; a.num_workers = 0; a.gpu = 0; a.use_gpu = True
    a.use_multi_gpu = False; a.inverse = True
    return a


def forward(model, ds, device):
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False,
                                         num_workers=0, drop_last=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)
            outputs = outputs[:, -ds.pred_len:, 0:]
            preds.append(outputs.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    preds = ds.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
    return preds[:, :, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    ds = args.dataset
    cfg = json.loads((Path("E:/GitHub_Project/chronos-retrieval/results/multidataset") / ds / "config.json").read_text(encoding="utf-8"))
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    n_val, n_test = int(cfg["num_windows"]["val"]), int(cfg["num_windows"]["test"])
    root = Path("E:/GitHub_Project/chronos-retrieval/results/multidataset") / ds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = OUT_DIR / ds
    out.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        a = build_args(context, horizon)
        model = PatchTST.Model(a).float().to(device)
        setting = (f"batch_{ds}_s{seed}_PatchTST_chronos_retrieval_ftS_sl{context}_ll48_pl{horizon}_"
                   f"dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_0")
        ckpt = PATCHTST_DIR / "checkpoints" / setting / "checkpoint.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        kw = dict(size=[context, 48, horizon], features="S", data_path="series.csv",
                  target="OT", scale=True, timeenc=0, freq="h")
        val_ds = Dataset_ChronosRetrieval(root_path=str(root), flag="val", **kw)
        test_ds = Dataset_ChronosRetrieval(root_path=str(root), flag="test", **kw)
        vp = forward(model, val_ds, device)
        tp = forward(model, test_ds, device)
        assert vp.shape == (n_val, horizon) and tp.shape == (n_test, horizon), f"{ds}/seed{seed} shape {vp.shape}/{tp.shape}"
        np.save(out / f"patchtst_{ds}_s{seed}_val.npy", vp.astype(np.float32))
        np.save(out / f"patchtst_{ds}_s{seed}_test.npy", tp.astype(np.float32))
        print(f"  [{ds}] seed{seed}: val={vp.shape} test={tp.shape} (期望 {n_val}/{n_test})", flush=True)


if __name__ == "__main__":
    main()
