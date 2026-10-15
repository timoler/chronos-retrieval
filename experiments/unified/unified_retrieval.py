# -*- coding: utf-8 -*-
import torch
import numpy as np
@torch.no_grad()
def retrieve_capture(model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                         val_act, bank_act, mode, topk, device, q_chunk=128, bank_chunk=4096):
    L = X_val.shape[1]
    model.eval()
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    # 预计算 bank 张量 + 编码（逐样本、eval 模式，位一致）。
    # 原实现在每个 query 块内都重新编码整个 bank（O(test*bank) 冗余），
    # 这里只在 query 循环外编码一次，按相同的 bank_chunk 分块保证逐位一致。
    X_bank_gpu = torch.tensor(X_bank, dtype=torch.float32, device=device)
    X_bank_res_gpu = torch.tensor(X_bank_res, dtype=torch.float32, device=device)
    z_bank = None
    if mode != "shape":
        z_chunks = []
        for bs in range(0, len(X_bank), bank_chunk):
            be = min(bs + bank_chunk, len(X_bank))
            z_chunks.append(model.encode(X_bank_gpu[bs:be], bank_act_t[bs:be]))
        z_bank = torch.cat(z_chunks, dim=0)

    preds = []
    transferred_preds = []
    selected = []
    for qs in range(0, len(X_val), q_chunk):
        qe = min(qs + q_chunk, len(X_val))
        nq = qe - qs
        best_scores = np.full((nq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nq, topk), -1, dtype=np.int64)

        x_q = torch.tensor(X_val[qs:qe], dtype=torch.float32, device=device)
        act_q = torch.tensor(val_act[qs:qe], dtype=torch.float32, device=device)
        res_q = torch.tensor(X_val_res[qs:qe], dtype=torch.float32, device=device)
        z_q = model.encode(x_q, act_q) if mode != "shape" else None

        for bs in range(0, len(X_bank), bank_chunk):
            be = min(bs + bank_chunk, len(X_bank))
            res_c = X_bank_res_gpu[bs:be]
            shape = (res_q @ res_c.T) / L
            if mode == "shape":
                s = shape
            else:
                z_c = z_bank[bs:be]
                delta = model.delta(z_q, z_c)
                s = model.combine(delta, shape, mode)

            s_np = s.cpu().numpy()
            kk = min(topk, be - bs)
            local_part = np.argpartition(s_np, -kk, axis=1)[:, -kk:]
            local_scores = np.take_along_axis(s_np, local_part, axis=1)
            local_idx = local_part + bs
            comb_scores = np.concatenate([best_scores, local_scores], axis=1)
            comb_idx = np.concatenate([best_idx, local_idx], axis=1)
            part = np.argpartition(comb_scores, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(comb_scores, part, axis=1)
            best_idx = np.take_along_axis(comb_idx, part, axis=1)

        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_idx = np.take_along_axis(best_idx, order, axis=1)
        pred = Y_bank[best_idx].mean(axis=1)
        transfer = pred + X_val[qs:qe].mean(axis=1)[:, None] - X_bank[best_idx].mean(axis=2).mean(axis=1)[:, None]
        transferred_preds.append(transfer)
        selected.append(best_idx)
        preds.append(pred)
    return np.concatenate(preds, axis=0), np.concatenate(transferred_preds, axis=0), np.concatenate(selected, axis=0)