"""Control: does memory's fusion gain equal a pure-mean (constant) baseline?

y_mean = [mean(x_q), ..., mean(x_q)] (all 96 points = query-context mean).
Compare, fused with Chronos at lambda=0.1:
  0.9*chronos + 0.1*mean    vs    0.9*chronos + 0.1*memory
If they match, the memory gain is just regression-to-the-mean.
"""

import numpy as np
import pandas as pd

CONTEXT, PRED = 512, 96
LAM = 0.1


def load(path):
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(dtype=np.float32)
    test_mask = (dates >= "2017-11-01").to_numpy()
    test_idx = np.flatnonzero(test_mask)
    return ot, int(test_idx[0]), int(test_idx[-1]) + 1


def mse(y, futures):
    return float(np.mean((y - futures) ** 2))


def main():
    ot, test_start, test_end = load("data/ETTh1.csv")
    starts = np.arange(test_start, test_end - PRED + 1)
    contexts = np.stack([ot[s - CONTEXT : s] for s in starts]).astype(np.float32)
    futures = np.stack([ot[s : s + PRED] for s in starts]).astype(np.float32)

    d = np.load("results/forecasts.npz")
    y_chronos = d["y_chronos"]
    y_mem = d["y_mem"]
    chronos_err = d["chronos_err"]

    y_mean = np.broadcast_to(contexts.mean(axis=1)[:, None], futures.shape)

    chronos_mse = mse(y_chronos, futures)
    mem_mse = mse(y_mem, futures)
    mean_mse = mse(y_mean, futures)

    fused_mem = (1 - LAM) * y_chronos + LAM * y_mem
    fused_mean = (1 - LAM) * y_chronos + LAM * y_mean
    fused_mem_mse = mse(fused_mem, futures)
    fused_mean_mse = mse(fused_mean, futures)

    print(f"=== standalone MSE ===")
    print(f"  Chronos-2          : {chronos_mse:.4f}")
    print(f"  memory (resid+anch): {mem_mse:.4f}")
    print(f"  pure mean (const)  : {mean_mse:.4f}")
    print(f"\n=== fusion @ lambda=0.1 ===")
    print(f"  0.9C + 0.1 memory   : {fused_mem_mse:.4f}  (improve {(chronos_mse - fused_mem_mse) / chronos_mse * 100:+.2f}%)")
    print(f"  0.9C + 0.1 mean     : {fused_mean_mse:.4f}  (improve {(chronos_mse - fused_mean_mse) / chronos_mse * 100:+.2f}%)")

    # per-bin improvement (chronos-error quintiles), mean vs memory
    print(f"\n=== per Chronos-error bin: fusion improvement% (mean vs memory) ===")
    print(f"{'bin':>24} {'memory':>9} {'pure-mean':>10}")
    edges = np.quantile(chronos_err, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins = np.digitize(chronos_err, edges) - 1
    names = ["低误差", "较低误差", "中误差", "较高误差", "高误差"]
    for b in range(5):
        m = bins == b
        c = chronos_err[m].mean()
        fm = ((1 - LAM) * y_chronos[m] + LAM * y_mem[m] - futures[m]) ** 2
        fmean = ((1 - LAM) * y_chronos[m] + LAM * y_mean[m] - futures[m]) ** 2
        imp_mem = (c - fm.mean()) / c * 100
        imp_mean = (c - fmean.mean()) / c * 100
        print(f"{'Chronos ' + names[b] + '误差组':>24} {imp_mem:+8.2f}% {imp_mean:+9.2f}%")


if __name__ == "__main__":
    main()
