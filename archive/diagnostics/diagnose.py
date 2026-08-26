"""Isolate why naive retrieval fails: level mismatch vs. shape mismatch.

Compares the raw (naive) memory forecast against level-corrected variants
that re-anchor the retrieved "shape" to the query's own level.
"""

import numpy as np
import pandas as pd

from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.retriever import retrieve_batch

CONTEXT, PRED, K = 512, 96, 5


def load_and_split(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    return ot, train, int(test_idx[0]), int(test_idx[-1]) + 1


def main():
    ot, train, test_start, test_end = load_and_split("data/ETTh1.csv")
    bank = MemoryBank.from_series(train, CONTEXT, PRED)

    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    idx, scores, mem_futures = retrieve_batch(contexts, bank, K)
    mem_contexts = bank.contexts[idx]
    w = softmax(scores)

    naive = np.einsum("nk,nkh->nh", w, mem_futures)

    # re-anchor to query's last observed value
    delta_shape = mem_futures - mem_contexts[:, :, -1:].astype(np.float64)
    delta = contexts[:, -1:] + np.einsum("nk,nkh->nh", w, delta_shape)

    # re-anchor to query's mean level
    mean_shape = mem_futures - mem_contexts.mean(axis=2)[:, :, None].astype(np.float64)
    mean_c = contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, mean_shape)

    def report(name, y):
        mse = float(np.mean((y - futures) ** 2))
        mae = float(np.mean(np.abs(y - futures)))
        print(f"{name:24s}  MSE={mse:9.4f}  MAE={mae:8.4f}")

    report("naive (raw level)", naive)
    report("delta (last-value anchor)", delta)
    report("mean (query-mean anchor)", mean_c)


if __name__ == "__main__":
    main()
