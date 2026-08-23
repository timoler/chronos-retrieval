"""Validate whether historical-pattern retrieval improves Chronos-2.

Runs the Chronos-2 baseline alone (lambda = 0) and a fixed-lambda fusion
with softmax-weighted memory forecasts, over a Top-k x lambda grid.
Chronos-2 source is untouched; retrieval lives in retrieval/.
"""

import argparse
import os
import random
from pathlib import Path

# HuggingFace is blocked in this environment; route through the mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from retrieval.fusion import softmax
from retrieval.memory_bank import MemoryBank
from retrieval.retriever import retrieve_batch

# ---- config ----
MODEL = "amazon/chronos-2"
DATA_PATH = Path("data/ETTh1.csv")
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
TOP_K_LIST = [1, 3, 5, 10]
LAMBDA_LIST = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEED = 42
BATCH_SIZE = 256
N_CASES = 8
CASE_K = 5
CASE_LAMBDA = 0.3
OUTPUT_DIR = Path("results")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_series(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    return dates, df["OT"].to_numpy(dtype=np.float32)


def split(dates, ot):
    # 12/4/8 split: train 2016-07..2017-06, val 2017-07..2017-10, test 2017-11..2018-06.
    train_mask = ((dates >= "2016-07-01") & (dates < "2017-07-01")).to_numpy()
    test_mask = (dates >= "2017-11-01").to_numpy()
    train = ot[train_mask]
    test_idx = np.flatnonzero(test_mask)
    test_start = int(test_idx[0])
    test_end = int(test_idx[-1]) + 1  # exclusive
    return train, test_start, test_end


def build_test_windows(ot, test_start, test_end, context_length, prediction_length, stride=1):
    # future must lie entirely within [test_start, test_end); context may use
    # any already-observable history before the window start.
    starts = np.arange(test_start, test_end - prediction_length + 1, stride)
    contexts = np.stack([ot[s - context_length : s] for s in starts])
    futures = np.stack([ot[s : s + prediction_length] for s in starts])
    return contexts.astype(np.float32), futures.astype(np.float32), starts


def chronos_forecasts(pipeline, contexts, prediction_length, batch_size):
    inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    _, mean = pipeline.predict_quantiles(
        inputs, prediction_length=prediction_length, batch_size=batch_size
    )
    # predict_quantiles returns the 0.5 quantile as its "mean"; use it as the point forecast.
    return np.stack([m.squeeze(0).numpy() for m in mean])


def memory_forecasts(scores, futures):
    weights = softmax(scores)  # (n_test, k)
    return np.einsum("nk,nkh->nh", weights, futures)


def evaluate(y, futures):
    mse = float(np.mean((y - futures) ** 2))
    mae = float(np.mean(np.abs(y - futures)))
    return mse, mae


def save_topk_scores(starts, topk_scores):
    cols = {f"top{i + 1}_sim": topk_scores[:, i] for i in range(topk_scores.shape[1])}
    df = pd.DataFrame({"sample_index": starts, **cols})
    out = OUTPUT_DIR / "topk_scores.csv"
    df.to_csv(out, index=False)
    print(f"Saved top-k Pearson scores ({df.shape[0]} samples x {topk_scores.shape[1]} cols) to {out}")
    print(df.head().to_string(index=False))


def save_cases(contexts, futures, starts, y_chronos, topk_scores, topk_futures):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_dir = OUTPUT_DIR / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    y_mem = memory_forecasts(topk_scores[:, :CASE_K], topk_futures[:, :CASE_K])
    y_final = (1 - CASE_LAMBDA) * y_chronos + CASE_LAMBDA * y_mem

    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(futures), size=min(N_CASES, len(futures)), replace=False)

    rows = []
    for idx in indices:
        gt = futures[idx]
        yc = y_chronos[idx]
        ym = y_mem[idx]
        yf = y_final[idx]

        chronos_mse = float(np.mean((yc - gt) ** 2))
        mem_mse = float(np.mean((ym - gt) ** 2))
        final_mse = float(np.mean((yf - gt) ** 2))
        top1_sim = float(topk_scores[idx, 0])

        rows.append(
            {
                "sample_index": int(starts[idx]),
                "top1_similarity": top1_sim,
                "chronos_mse": chronos_mse,
                "memory_mse": mem_mse,
                "final_mse": final_mse,
                "retrieval_helped": bool(mem_mse < chronos_mse),
            }
        )

        x = np.arange(PREDICTION_LENGTH)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, gt, label="Ground Truth", color="k", linewidth=2)
        ax.plot(x, yc, label="Chronos-2", color="tab:blue")
        ax.plot(x, ym, label="Memory Forecast", color="tab:orange")
        ax.plot(x, yf, label=f"Final (lambda={CASE_LAMBDA})", color="tab:red", linestyle="--")
        ax.set_title(
            f"sample@{starts[idx]}  top1_sim={top1_sim:.3f}  "
            f"chronos_mse={chronos_mse:.3f}  mem_mse={mem_mse:.3f}"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(case_dir / f"case_{starts[idx]}.png", dpi=110)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "cases_summary.csv", index=False)
    print(f"Saved {len(indices)} case plots to {case_dir}")


def main():
    args = parse_args()
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates, ot = load_series(DATA_PATH)
    train, test_start, test_end = split(dates, ot)

    print("Dataset: ETTh1 (target=OT)")
    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Prediction length: {PREDICTION_LENGTH}")
    print(f"Train points: {len(train)}, test range: [{test_start}, {test_end})")

    contexts, futures, starts = build_test_windows(
        ot, test_start, test_end, CONTEXT_LENGTH, PREDICTION_LENGTH, stride=args.stride
    )
    n_test = len(contexts)
    print(f"Test windows: {n_test} (stride={args.stride})")

    print(f"\nLoading model: {MODEL} ...")
    pipeline = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    print(
        f"Loaded. model_context_length={pipeline.model_context_length}, "
        f"model_prediction_length={pipeline.model_prediction_length}"
    )

    print("\nRunning Chronos-2 baseline ...")
    y_chronos = chronos_forecasts(pipeline, contexts, PREDICTION_LENGTH, BATCH_SIZE)
    base_mse, base_mae = evaluate(y_chronos, futures)

    print("\nChronos-2")
    print(f"MSE: {base_mse:.4f}")
    print(f"MAE: {base_mae:.4f}")

    if args.baseline_only:
        return

    print("\nBuilding memory bank (train only) ...")
    bank = MemoryBank.from_series(train, CONTEXT_LENGTH, PREDICTION_LENGTH)
    print(f"Memory bank: {bank.n_memories} windows")

    max_k = max(TOP_K_LIST)
    print(f"Retrieving top-{max_k} for all queries ...")
    _, topk_scores, topk_futures = retrieve_batch(contexts, bank, max_k)

    save_topk_scores(starts, topk_scores)

    rows = []
    for k in TOP_K_LIST:
        y_mem = memory_forecasts(topk_scores[:, :k], topk_futures[:, :k])
        for lam in LAMBDA_LIST:
            y = y_chronos if lam == 0.0 else (1 - lam) * y_chronos + lam * y_mem
            mse, mae = evaluate(y, futures)
            rows.append(
                {
                    "Method": "Chronos-2" if lam == 0.0 else "Retrieval Fusion",
                    "TopK": k,
                    "Lambda": lam,
                    "MSE": mse,
                    "MAE": mae,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print("\n" + results.to_string(index=False))

    save_cases(contexts, futures, starts, y_chronos, topk_scores, topk_futures)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-only", action="store_true", help="run Chronos-2 baseline only")
    p.add_argument("--stride", type=int, default=1, help="stride between consecutive test windows")
    return p.parse_args()


if __name__ == "__main__":
    main()
