"""Compare memory-forecast anchoring methods, standalone and fused with Chronos-2.

Memory forecast variants:
  naive       — copy the raw retrieved future (baseline, wrong level)
  delta       — re-anchor the retrieved shape to the query's last value
  mean        — re-anchor the retrieved shape to the query's mean level
  detrended   — linear-detrend, retrieve on residual, add back the query's
                own extrapolated trend

Each is reported standalone (MSE/MAE) and then fused with Chronos-2 at a few
lambdas, so we can see whether a corrected memory forecast helps at all.
"""

import os

# HuggingFace is blocked in this environment; route through the mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from retrieval.decompose import linear_detrend, residual_future
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.retriever import retrieve_batch

CONTEXT, PRED, K = 512, 96, 5
MODEL = "amazon/chronos-2"
BATCH = 256
LAMBDAS = [0.0, 0.1, 0.2, 0.3]


def load_and_split(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return ot, train, int(test_idx[0]), int(test_idx[-1]) + 1


def mse_mae(y, futures):
    return float(np.mean((y - futures) ** 2)), float(np.mean(np.abs(y - futures)))


def main():
    ot, train, test_start, test_end = load_and_split("data/ETTh1.csv")
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)
    print(f"test windows: {len(starts)}")

    # --- raw retrieval (shared by naive / delta / mean) ---
    idx_raw, scores_raw, fut_raw = retrieve_batch(contexts, bank, K)
    w = softmax(scores_raw)
    mem_ctx = bank.contexts[idx_raw]

    naive = np.einsum("nk,nkh->nh", w, fut_raw)

    delta_shape = fut_raw - mem_ctx[:, :, -1:].astype(np.float64)
    delta = contexts[:, -1:] + np.einsum("nk,nkh->nh", w, delta_shape)

    mean_shape = fut_raw - mem_ctx.mean(axis=2)[:, :, None].astype(np.float64)
    mean_c = contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, mean_shape)

    # --- detrended retrieval ---
    resid_ctx_m, trend_fut_m = linear_detrend(bank.contexts, PRED)
    resid_fut_m = residual_future(bank.futures, trend_fut_m)
    resid_bank = MemoryBank(resid_ctx_m, resid_fut_m)

    resid_ctx_q, trend_fut_q = linear_detrend(contexts, PRED)
    _, scores_res, resid_fut_topk = retrieve_batch(resid_ctx_q, resid_bank, K)
    w_res = softmax(scores_res)
    detrended = trend_fut_q + np.einsum("nk,nkh->nh", w_res, resid_fut_topk)

    # --- standalone comparison ---
    methods = {
        "naive (raw level)": naive,
        "delta (last-value)": delta,
        "mean (query-mean)": mean_c,
        "detrended (linear)": detrended,
    }
    print("\nstandalone memory forecast:")
    for name, y in methods.items():
        mse, mae = mse_mae(y, futures)
        print(f"  {name:22s}  MSE={mse:9.4f}  MAE={mae:8.4f}")

    # --- Chronos-2 baseline + fusion ---
    print(f"\nLoading {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    y_chronos = np.stack([m.squeeze(0).numpy() for m in means])
    chronos_mse, chronos_mae = mse_mae(y_chronos, futures)
    print(f"Chronos-2 standalone: MSE={chronos_mse:.4f}  MAE={chronos_mae:.4f}")

    print("\nfusion MSE  (cols = anchoring method, rows = lambda):")
    header = f"{'lambda':>6}  " + "  ".join(f"{k.split(' (')[0]:>18}" for k in methods)
    print(header)
    for lam in LAMBDAS:
        row = []
        for y in methods.values():
            yf = y_chronos if lam == 0.0 else (1 - lam) * y_chronos + lam * y
            row.append(mse_mae(yf, futures)[0])
        print(f"{lam:6.2f}  " + "  ".join(f"{v:18.4f}" for v in row))


if __name__ == "__main__":
    main()
