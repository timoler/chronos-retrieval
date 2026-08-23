"""Verify Chronos-dependent checks + three small-sample hand-cases.

  7. fusion: 0.9*Chronos + 0.1*Memory == program's fused prediction
  8. metrics: independently recompute chronos/memory/fused MSE, confirm
     6.1968 / 18.0965 / 6.0821 reproduce.

Also prints three full cases (low / median / high residual top1_sim):
  query index (ETTh1 row), query date, top1 memory index + date, sim,
  mean-anchor prediction, Chronos prediction, fusion prediction, true future.
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


def pvec(name, a):
    print(f"  {name} (mean={a.mean():.4f}):")
    for r in range(0, len(a), 16):
        print("    " + "  ".join(f"{v:.2f}" for v in a[r : r + 16]))


def main():
    dates, ot, train, test_start, test_end = load("data/ETTh1.csv")
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    # memory forecast: Residual retrieval + Mean-anchor transfer
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

    # Chronos
    print(f"Loading {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    y_chronos = np.stack([m.squeeze(0).numpy() for m in means])

    # ---------- CHECK 8: metrics (independent direct computation) ----------
    chronos_mse = float(np.mean((y_chronos - futures) ** 2))
    mem_mse = float(np.mean((y_mem - futures) ** 2))
    fused = (1 - LAM) * y_chronos + LAM * y_mem
    fused_mse = float(np.mean((fused - futures) ** 2))
    print("\n=== CHECK 8: metrics (independent recompute) ===")
    print(f"  chronos MSE = {chronos_mse:.4f}  (expect 6.1968)")
    print(f"  memory MSE  = {mem_mse:.4f}  (expect 18.0965)")
    print(f"  fused@{LAM} MSE = {fused_mse:.4f}  (expect 6.0821)")
    ok8 = (abs(chronos_mse - 6.1968) < 0.001 and abs(mem_mse - 18.0965) < 0.001 and abs(fused_mse - 6.0821) < 0.001)
    print(f"  -> {'PASS' if ok8 else 'FAIL'}")

    # ---------- CHECK 7: fusion hand-check on one random sample ----------
    rng = np.random.default_rng(3)
    i = int(rng.choice(len(contexts), size=1)[0])
    hand_fused = 0.9 * y_chronos[i] + 0.1 * y_mem[i]
    diff = np.abs(hand_fused - fused[i]).max()
    ok7 = diff < 1e-5
    print("\n=== CHECK 7: fusion hand-check ===")
    print(f"  sample {i}: max|0.9*chronos + 0.1*memory - program_fused| = {diff:.2e} -> {'PASS' if ok7 else 'FAIL'}")

    # ---------- three small-sample cases ----------
    order = np.argsort(top1_sim)
    picks = {
        "LOW top1_sim": int(order[0]),
        "MEDIAN top1_sim": int(order[len(order) // 2]),
        "HIGH top1_sim": int(order[-1]),
    }
    for label, i in picks.items():
        s = starts[i]
        mi = int(idx_res[i, 0])
        sim = float(scores_res[i, 0])
        print(f"\n{'=' * 70}")
        print(f"{label}")
        print(f"  query index (ETTh1 row) = {s}   date = {dates.iloc[s]}")
        print(f"  top1 memory index = {mi}   memory date = {dates.iloc[mi]} (train)")
        print(f"  top1_sim = {sim:.6f}")
        pvec("true future (96)", futures[i])
        pvec("mean-anchor memory (96)", y_mem[i])
        pvec("Chronos (96)", y_chronos[i])
        pvec("fusion 0.9C+0.1M (96)", fused[i])


if __name__ == "__main__":
    main()
