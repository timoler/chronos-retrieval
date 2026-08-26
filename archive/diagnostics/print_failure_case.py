"""Print the signal groups for one failing retrieval case, incl. Chronos-2.

Groups:
  1. query context        — the last 512 observed points before the forecast start
  2. retrieved context    — the top-1 memory window's 512 points
  3. retrieved future     — that memory window's next 96 points (its "answer")
  4. ground-truth future  — the actual next 96 points of the query
  5. Chronos-2 forecast   — what Chronos-2 predicts for those same 96 steps
"""

import os

# HuggingFace is blocked in this environment; route through the mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from retrieval.memory_bank import MemoryBank
from retrieval.retriever import retrieve_batch

CONTEXT, PRED = 512, 96
MODEL = "amazon/chronos-2"
SAMPLE = 12194  # from cases_summary.csv: top1_sim=0.767, memory_mse=268 vs chronos 0.6


def print_arr(label, a, per_line=16):
    print(f"\n=== {label} ===")
    print(f"(len={len(a)}  mean={a.mean():.3f}  min={a.min():.3f}  max={a.max():.3f}  "
          f"first={a[0]:.2f}  last={a[-1]:.2f})")
    for i in range(0, len(a), per_line):
        chunk = a[i : i + per_line]
        print("  " + "  ".join(f"{v:.2f}" for v in chunk))


def chronos_forecast(pipeline, context, prediction_length):
    inputs = [torch.tensor(context, dtype=torch.float32)]
    _, mean = pipeline.predict_quantiles(inputs, prediction_length=prediction_length)
    return mean[0].squeeze(0).numpy()


def main():
    df = pd.read_csv("data/ETTh1.csv")
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)

    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    train = ot[train_mask]
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    s = SAMPLE
    query = ot[s - CONTEXT : s].astype(np.float32)   # 1. current 512
    true = ot[s : s + PRED].astype(np.float32)       # 4. ground-truth 96

    idx, scores, mem_futures = retrieve_batch(query[None], bank, 1)
    i = idx[0, 0]
    score = scores[0, 0]
    mem_ctx = bank.contexts[i].astype(np.float32)    # 2. retrieved 512
    mem_fut = bank.futures[i].astype(np.float32)     # 3. retrieved 96

    print(f"sample_index={s}  date={dates.iloc[s]}  top1_sim={score:.4f}")
    print(f"retrieved memory index={i}  (train row {i + CONTEXT}..{i + CONTEXT + PRED})")

    print("\nLoading Chronos-2 ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    yc = chronos_forecast(pipeline, query, PRED)     # 5. chronos forecast

    print_arr("1. query context (current 512)", query)
    print_arr("2. retrieved context (history 512)", mem_ctx)
    print_arr("3. retrieved future (history 96)", mem_fut)
    print_arr("4. ground-truth future (real 96)", true)
    print_arr("5. Chronos-2 forecast (96)", yc)

    # per-window comparison
    chronos_mse = float(np.mean((yc - true) ** 2))
    mem_mse = float(np.mean((mem_fut - true) ** 2))
    print("\n=== per-window MSE vs ground truth ===")
    print(f"Chronos-2      : {chronos_mse:.3f}")
    print(f"memory (naive) : {mem_mse:.3f}")
    print(f"level means    : query_ctx={query.mean():.2f}  mem_ctx={mem_ctx.mean():.2f}  "
          f"chronos={yc.mean():.2f}  true={true.mean():.2f}")


if __name__ == "__main__":
    main()
