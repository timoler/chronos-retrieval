"""Step 1 diagnostic: does memory's value depend on retrieval confidence?

Bins test windows by top1 residual-Pearson similarity (quintiles), with the
fixed best memory forecast (Residual retrieval + Mean-anchor transfer), and
reports per-bin Chronos / memory / λ=0.1-fusion MSE plus the fusion's
improvement over Chronos.  No gate is fitted here.
"""

import os

# HuggingFace is blocked in this environment; route through the mirror.
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


def load_and_split(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return ot, train, int(test_idx[0]), int(test_idx[-1]) + 1


def topk(scores, k):
    k = min(k, scores.shape[-1])
    idx = np.argpartition(scores, -k, axis=-1)[..., -k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def main():
    ot, train, test_start, test_end = load_and_split("data/ETTh1.csv")
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)
    print(f"test windows: {len(starts)}")

    # --- residual retrieval (for confidence signal + memory selection) ---
    _, _, R_q, _, _ = seasonal_decompose(contexts, PRED, PERIOD, TREND_WINDOW)
    _, _, R_m, _, _ = seasonal_decompose(bank.contexts, PRED, PERIOD, TREND_WINDOW)
    sim_R = pearson_similarity_matrix(R_q, R_m)
    idx_res = topk(sim_R, K)
    scores_res = np.take_along_axis(sim_R, idx_res, axis=-1)
    top1_sim = scores_res[:, 0]

    # --- mean-anchor memory forecast from residual-selected memories ---
    w = softmax(scores_res)
    fut = bank.futures[idx_res]
    ctx = bank.contexts[idx_res]
    shape = fut - ctx.mean(axis=2)[:, :, None]
    y_mem = contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)

    # --- Chronos-2 forecast ---
    print(f"Loading {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    y_chronos = np.stack([m.squeeze(0).numpy() for m in means])

    # --- per-window MSE ---
    chronos_mse = ((y_chronos - futures) ** 2).mean(axis=1)
    mem_mse = ((y_mem - futures) ** 2).mean(axis=1)
    fused = (1 - LAM) * y_chronos + LAM * y_mem
    fused_mse = ((fused - futures) ** 2).mean(axis=1)

    # --- quintile bins ---
    edges = np.quantile(top1_sim, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_idx = np.digitize(top1_sim, edges) - 1

    header = (
        f"{'bin':>3} {'n':>5} {'sim_range':>17} {'mean_sim':>8} "
        f"{'chronos':>9} {'memory':>9} {'fused@0.1':>10} {'improve%':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for b in range(5):
        m = bin_idx == b
        n = int(m.sum())
        lo, hi = top1_sim[m].min(), top1_sim[m].max()
        c = chronos_mse[m].mean()
        mm = mem_mse[m].mean()
        f = fused_mse[m].mean()
        imp = (c - f) / c * 100.0
        print(
            f"{b:>3} {n:>5} [{lo:7.3f},{hi:7.3f}] {top1_sim[m].mean():8.3f} "
            f"{c:9.4f} {mm:9.4f} {f:10.4f} {imp:+9.2f}%"
        )

    # overall reference
    n_all = len(futures)
    print("-" * len(header))
    print(
        f"{'all':>3} {n_all:>5} [{top1_sim.min():7.3f},{top1_sim.max():7.3f}] "
        f"{top1_sim.mean():8.3f} {chronos_mse.mean():9.4f} {mem_mse.mean():9.4f} "
        f"{fused_mse.mean():10.4f} {(chronos_mse.mean() - fused_mse.mean()) / chronos_mse.mean() * 100:+9.2f}%"
    )


if __name__ == "__main__":
    main()
