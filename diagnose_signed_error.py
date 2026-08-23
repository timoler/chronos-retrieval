"""Signed-error diagnostics: is fusion benefit from error-direction complementarity?

Loads saved forecasts, computes signed errors e_c = y_chronos - y_true and
e_m = y_mem - y_true, then reports their correlation and the rate of
opposite-sign (one over-predicts, the other under-predicts), overall and
within each Chronos-error quintile.
"""

import numpy as np


def main():
    d = np.load("results/forecasts.npz")
    y_chronos = d["y_chronos"]
    y_mem = d["y_mem"]
    futures = d["futures"]
    chronos_err = d["chronos_err"]

    e_c = (y_chronos - futures).ravel()
    e_m = (y_mem - futures).ravel()
    n_points = e_c.size
    print(f"total (sample x step) points = {n_points}")

    # 1. correlation of signed errors
    r = float(np.corrcoef(e_c, e_m)[0, 1])
    print(f"\ncorr(e_c, e_m) = {r:+.4f}")

    # 2. overall opposite-sign rate
    opp = float((e_c * e_m < 0).mean())
    same = float((e_c * e_m > 0).mean())
    zero = float((e_c * e_m == 0).mean())
    print(f"\noverall: opposite-sign = {opp * 100:.2f}%   same-sign = {same * 100:.2f}%   zero = {zero * 100:.2f}%")

    # 3. per-bin opposite-sign rate (bins by Chronos error)
    print("\nper Chronos-error bin:")
    print(f"{'bin':>24} {'chronos_MSE':>11} {'opposite%':>10} {'corr(e_c,e_m)':>13}")
    edges = np.quantile(chronos_err, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins = np.digitize(chronos_err, edges) - 1
    names = ["低误差", "较低误差", "中误差", "较高误差", "高误差"]
    for b in range(5):
        m = bins == b
        ec = (y_chronos[m] - futures[m]).ravel()
        em = (y_mem[m] - futures[m]).ravel()
        opp_b = float((ec * em < 0).mean())
        r_b = float(np.corrcoef(ec, em)[0, 1]) if ec.size > 1 else float("nan")
        print(f"{'Chronos ' + names[b] + '误差组':>24} {chronos_err[m].mean():11.4f} {opp_b * 100:9.2f}% {r_b:+13.4f}")


if __name__ == "__main__":
    main()
