"""Diagnose which signal predicts "memory helps": chronos error vs top1_sim.

Per test window, define memory advantage = chronos_err - mem_err (positive =>
memory beats Chronos).  Then check whether advantage tracks Chronos's own
error better than it tracks retrieval confidence (top1_sim).  Saves forecasts
so follow-ups do not reload Chronos.
"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from retrieval.decompose import seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
PERIOD, TREND_WINDOW = 24, 48
MODEL = "amazon/chronos-2"
BATCH = 256
LAM = 0.1


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return dates, ot, train, int(test_idx[0]), int(test_idx[-1]) + 1


def topk(scores, k):
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def spearman(a, b):
    def rank(x):
        return np.argsort(np.argsort(x)).astype(np.float64)
    return pearson(rank(a), rank(b))


def bins_by(signal, n_bins=5):
    edges = np.quantile(signal, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(signal, edges) - 1


def main():
    dates, ot, train, test_start, test_end = load("data/ETTh1.csv")
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    # memory forecast (Residual retrieval + Mean-anchor transfer)
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    sim_R = pearson_similarity_matrix(R_q, R_m)
    idx_res = topk(sim_R, K)
    scores_res = np.take_along_axis(sim_R, idx_res, axis=-1)
    top1_sim = scores_res[:, 0]

    w = softmax(scores_res)
    fut = bank.futures[idx_res]
    ctx = bank.contexts[idx_res]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    y_mem = contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)

    # Chronos (also capture quantiles for uncertainty)
    print(f"Loading {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    quantiles, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    y_chronos = np.stack([m.squeeze(0).numpy() for m in means])

    # Chronos's reported uncertainty = mean over steps of (max quantile - min quantile)
    spreads = []
    for q in quantiles:
        qn = q.numpy()  # (n_quantiles, pred_len)
        spreads.append(float((qn.max(axis=0) - qn.min(axis=0)).mean()))
    spread = np.array(spreads)

    # per-window errors
    chronos_err = ((y_chronos - futures) ** 2).mean(axis=1)
    mem_err = ((y_mem - futures) ** 2).mean(axis=1)
    fused = (1 - LAM) * y_chronos + LAM * y_mem
    fused_err = ((fused - futures) ** 2).mean(axis=1)
    advantage = chronos_err - mem_err  # >0 => memory beats chronos

    # save for reuse
    np.savez(
        "results/forecasts.npz",
        y_chronos=y_chronos, y_mem=y_mem, futures=futures,
        top1_sim=top1_sim, spread=spread,
        chronos_err=chronos_err, mem_err=mem_err, fused_err=fused_err,
    )
    print("saved results/forecasts.npz")

    # --- correlations ---
    print("\n=== correlations (advantage = chronos_err - mem_err) ===")
    print(f"  pearson (advantage, chronos_err)  = {pearson(advantage, chronos_err):+.4f}")
    print(f"  pearson (advantage, top1_sim)     = {pearson(advantage, top1_sim):+.4f}")
    print(f"  pearson (advantage, chronos_spread) = {pearson(advantage, spread):+.4f}")
    print(f"  pearson (spread, chronos_err)     = {pearson(spread, chronos_err):+.4f}  (is uncertainty a proxy for error?)")

    # --- binning by chronos_err ---
    print("\n=== bins by CHRONOS ERROR (quintiles) ===")
    print(f"{'bin':>3} {'n':>5} {'chronos_err':>11} {'mem_err':>9} {'advantage':>10} {'fused@0.1':>10} {'improve%':>9}")
    b = bins_by(chronos_err)
    for i in range(5):
        m = b == i
        c = chronos_err[m].mean()
        me = mem_err[m].mean()
        f = fused_err[m].mean()
        adv = c - me
        imp = (c - f) / c * 100
        print(f"{i:>3} {int(m.sum()):>5} {c:11.4f} {me:9.4f} {adv:+10.4f} {f:10.4f} {imp:+9.2f}%")

    # --- binning by top1_sim (contrast) ---
    print("\n=== bins by TOP1_SIM (quintiles, contrast) ===")
    print(f"{'bin':>3} {'n':>5} {'top1_sim':>8} {'chronos_err':>11} {'advantage':>10} {'improve%':>9}")
    b2 = bins_by(top1_sim)
    for i in range(5):
        m = b2 == i
        c = chronos_err[m].mean()
        f = fused_err[m].mean()
        adv = c - mem_err[m].mean()
        imp = (c - f) / c * 100
        print(f"{i:>3} {int(m.sum()):>5} {top1_sim[m].mean():8.3f} {c:11.4f} {adv:+10.4f} {imp:+9.2f}%")


if __name__ == "__main__":
    main()
