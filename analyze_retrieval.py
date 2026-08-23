"""Analyze topk_scores.csv: how good is the retrieval?

Reads the per-sample top-k Pearson similarities written by
run_retrieval_experiment.py and reports how similar the retrieved
historical fragments are to each query.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("results")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def main():
    path = OUTPUT_DIR / "topk_scores.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run run_retrieval_experiment.py first.")

    df = pd.read_csv(path)
    sim_cols = [c for c in df.columns if c.startswith("top") and c.endswith("_sim")]
    scores = df[sim_cols].to_numpy(dtype=np.float64)
    n_samples, n_k = scores.shape
    print(f"Loaded {path}: {n_samples} samples x {n_k} ranks\n")

    # 1. per-rank statistics
    print("Per-rank Pearson similarity:")
    stats = pd.DataFrame(
        {
            "rank": [f"top{i + 1}" for i in range(n_k)],
            "mean": scores.mean(axis=0),
            "median": np.median(scores, axis=0),
            "std": scores.std(axis=0),
            "min": scores.min(axis=0),
            "max": scores.max(axis=0),
        }
    )
    print(stats.round(4).to_string(index=False))

    # 2. top1 distribution + threshold hit rate
    top1 = scores[:, 0]
    print("\ntop1_sim distribution:")
    for label, v in zip(["min", "p25", "p50", "p75", "max"], np.percentile(top1, [0, 25, 50, 75, 100])):
        print(f"  {label:4s} = {v:.4f}")

    print("\nshare of samples with top1_sim >= threshold:")
    for t in THRESHOLDS:
        n = int((top1 >= t).sum())
        print(f"  >= {t:.1f}: {n / n_samples * 100:5.2f}%  ({n}/{n_samples})")

    # 3. rank decay
    print("\nmean similarity by rank (decay):")
    for i in range(n_k):
        print(f"  top{i + 1:2d}: {scores[:, i].mean():.4f}")

    save_histogram(top1, OUTPUT_DIR)


def save_histogram(top1, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(top1, bins=50, color="tab:blue", alpha=0.8)
    ax.axvline(top1.mean(), color="r", linestyle="--", label=f"mean={top1.mean():.3f}")
    ax.set_xlabel("top1 Pearson similarity")
    ax.set_ylabel("count")
    ax.set_title(f"Retrieval top1 similarity (n={len(top1)})")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "top1_sim_hist.png"
    fig.savefig(out, dpi=110)
    print(f"\nSaved histogram to {out}")


if __name__ == "__main__":
    main()
