"""2x2 ablation: retriever x transfer.

Factors (2 levels each):
  retriever  — which Pearson similarity selects the top-K memories:
               Raw Pearson vs Residual Pearson
  transfer   — how the query future is reconstructed from the selected
               memories' futures:
               Mean-anchor (mean(x_q) + w*(x_fut_m - mean(x_ctx_m)))
               T+S+R       (T_fut_q + S_fut_q + w*R_fut_m)

Fixed: TopK=5, softmax weights, same test set. Reports MSE/MAE of the four
combinations to disentangle whether the retriever or the transfer dominates.
"""

import numpy as np
import pandas as pd

from retrieval.decompose import decompose_future, seasonal_decompose
from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.retriever import retrieve_batch
from retrieval.similarity import pearson_similarity_matrix

CONTEXT, PRED, K = 512, 96, 5
PERIOD, TREND_WINDOW = 24, 48


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

    # --- raw retrieval (retriever = Raw Pearson) ---
    idx_raw, scores_raw, _ = retrieve_batch(contexts, bank, K)

    # --- decomposition (for residual retrieval + T+S+R transfer) ---
    T_q, S_q, R_q, T_fut_q, S_fut_q = seasonal_decompose(
        contexts, PRED, PERIOD, TREND_WINDOW
    )
    T_m, S_m, R_m, T_fut_m, S_fut_m = seasonal_decompose(
        bank.contexts, PRED, PERIOD, TREND_WINDOW
    )
    R_fut_m = decompose_future(bank.futures, T_fut_m, S_fut_m)

    # --- residual retrieval (retriever = Residual Pearson) ---
    sim_R = pearson_similarity_matrix(R_q, R_m)
    idx_res = topk(sim_R, K)
    scores_res = np.take_along_axis(sim_R, idx_res, axis=-1)

    def mean_anchor_transfer(idx, scores):
        w = softmax(scores)
        fut = bank.futures[idx]              # (n, K, 96)
        ctx = bank.contexts[idx]             # (n, K, 512)
        shape = fut - ctx.mean(axis=2)[:, :, None]
        return contexts.mean(axis=1)[:, None] + np.einsum("nk,nkh->nh", w, shape)

    def tsr_transfer(idx, scores):
        w = softmax(scores)
        return T_fut_q + S_fut_q + np.einsum("nk,nkh->nh", w, R_fut_m[idx])

    combos = {
        "Raw + Mean-anchor": mean_anchor_transfer(idx_raw, scores_raw),
        "Residual + Mean-anchor": mean_anchor_transfer(idx_res, scores_res),
        "Raw + T+S+R": tsr_transfer(idx_raw, scores_raw),
        "Residual + T+S+R": tsr_transfer(idx_res, scores_res),
    }

    print("\n2x2 ablation (TopK=5, softmax weights, fixed test set):")
    print(f"{'retriever':>18} {'transfer':>13} | {'MSE':>9} | {'MAE':>8}")
    print("-" * 52)
    for ret, tr in [
        ("Raw Pearson", "Mean-anchor"),
        ("Residual Pearson", "Mean-anchor"),
        ("Raw Pearson", "T+S+R"),
        ("Residual Pearson", "T+S+R"),
    ]:
        key = f"{ret.split()[0]} + {tr}"
        mse, mae = mse_mae(combos[key], futures)
        print(f"{ret:>18} {tr:>13} | {mse:9.4f} | {mae:8.4f}")


if __name__ == "__main__":
    main()
