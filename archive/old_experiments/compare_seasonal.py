"""Step 5: structure-aware retrieval over trend + daily-seasonal + residual.

Decompose x = T + S + R, compute a separate Pearson similarity for each
component, combine them with weights (sim = a*sim_T + b*sim_S + g*sim_R),
retrieve top-k, and forecast as
    y = T_fut_q + S_fut_q + softmax-weighted retrieved R_fut.

Reference baselines (from step 4): mean-anchor 19.30, Chronos-2 6.1968.
"""

import os

# HuggingFace is blocked in this environment; route through the mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from retrieval.decompose import decompose_future, seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.retriever import retrieve_batch
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
MODEL = "amazon/chronos-2"
BATCH = 256
PERIOD = 24
TREND_WINDOW = 48
FUSION_LAMBDAS = [0.1, 0.2]


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


def mse_mae(y, futures):
    return float(np.mean((y - futures) ** 2)), float(np.mean(np.abs(y - futures)))


def main():
    ot, train, test_start, test_end = load_and_split("data/ETTh1.csv")
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)
    print(f"test windows: {len(starts)}")

    # --- reference: mean-anchor from step 4 (raw retrieval) ---
    idx_raw, scores_raw, fut_raw = retrieve_batch(contexts, bank, K)
    w_raw = softmax(scores_raw)
    mem_ctx = bank.contexts[idx_raw]
    mean_shape = fut_raw - mem_ctx.mean(axis=2)[:, :, None].astype(np.float64)
    mean_anchor = contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w_raw, mean_shape)

    # --- decompose memory + query ---
    T_m, S_m, R_m, T_fut_m, S_fut_m = seasonal_decompose(
        bank.contexts, PRED, PERIOD, TREND_WINDOW
    )
    R_fut_m = decompose_future(bank.futures, T_fut_m, S_fut_m)

    T_q, S_q, R_q, T_fut_q, S_fut_q = seasonal_decompose(
        contexts, PRED, PERIOD, TREND_WINDOW
    )

    # --- component similarities (float32 to bound memory) ---
    print("computing component similarity matrices ...")
    sim_T = pearson_similarity_matrix(T_q, T_m).astype(np.float32)
    sim_S = pearson_similarity_matrix(S_q, S_m).astype(np.float32)
    sim_R = pearson_similarity_matrix(R_q, R_m).astype(np.float32)

    combos = {
        "trend-only": (1.0, 0.0, 0.0),
        "seasonal-only": (0.0, 1.0, 0.0),
        "residual-only": (0.0, 0.0, 1.0),
        "seasonal+residual": (0.0, 1.0, 1.0),
        "equal (T+S+R)": (1.0, 1.0, 1.0),
    }

    print("\nstandalone memory forecast:")
    mse, mae = mse_mae(mean_anchor, futures)
    print(f"  {'mean-anchor (ref)':22s}  MSE={mse:9.4f}  MAE={mae:8.4f}")

    results = {}
    for name, (a, b, g) in combos.items():
        sim = a * sim_T + b * sim_S + g * sim_R
        idx = topk(sim, K)
        scores_k = np.take_along_axis(sim, idx, axis=-1)
        w = softmax(scores_k)
        y = T_fut_q + S_fut_q + np.einsum("nk,nkh->nh", w, R_fut_m[idx])
        results[name] = y
        mse, mae = mse_mae(y, futures)
        print(f"  {name:22s}  MSE={mse:9.4f}  MAE={mae:8.4f}")

    # --- Chronos-2 fusion ---
    print(f"\nLoading {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, means = pipeline.predict_quantiles(inputs, prediction_length=PRED, batch_size=BATCH)
    y_chronos = np.stack([m.squeeze(0).numpy() for m in means])
    chronos_mse, chronos_mae = mse_mae(y_chronos, futures)
    print(f"Chronos-2 standalone: MSE={chronos_mse:.4f}  MAE={chronos_mae:.4f}")

    print(f"\nfusion MSE (rows=lambda, cols=method):")
    all_methods = {"mean-anchor": mean_anchor, **results}
    header = f"{'lambda':>6}  " + "  ".join(f"{k:>20}" for k in all_methods)
    print(header)
    for lam in FUSION_LAMBDAS:
        row = []
        for y in all_methods.values():
            yf = (1 - lam) * y_chronos + lam * y
            row.append(mse_mae(yf, futures)[0])
        print(f"{lam:6.2f}  " + "  ".join(f"{v:20.4f}" for v in row))


if __name__ == "__main__":
    main()
