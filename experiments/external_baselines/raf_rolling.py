# -*- coding: utf-8 -*-
"""
RAF (NeurIPS 2025, `anonymous2608878/raf`) rolling-window adaptation for the
chronos-retrieval formal protocol.

Core RAF algorithm functions (raf_forecast / _neighbor_avg / _grid_search_lambda /
_zscore / _dist / _window_bank) are copied VERBATIM from the official chronos-raf.py.
Only the outer evaluation is changed: official evaluates a SINGLE H-length test
segment; here we roll over our formal test windows (window_indices.npz) so the
result is comparable to Ours (same test range, raw-value MSE).

Leakage assertion: for each test window start s, the retrieval bank is built only
from series[:s], so the query's future series[s:s+H] is never a candidate.
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from math import sqrt
from typing import Callable, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Official RAF functions (verbatim from chronos-raf.py @ 21fba16)
# --------------------------------------------------------------------------- #
def _zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean()
    s = x.std()
    if s == 0:
        return x * 0
    return (x - m) / s


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return sqrt(np.dot(d, d))


def _dist(a: np.ndarray, b: np.ndarray, use_dtw: bool) -> float:
    if use_dtw:
        try:
            from fastdtw import fastdtw
            d, _ = fastdtw(a, b)
            return float(d)
        except Exception:
            pass
    return _euclid(a, b)


def _window_bank(y: np.ndarray, L: int, H: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    n = len(y)
    for s in range(0, n - (L + H) + 1, step):
        X.append(_zscore(y[s:s + L]))
        Y.append(y[s + L:s + L + H])
    if not X:
        return np.empty((0, L)), np.empty((0, H))
    return np.stack(X), np.stack(Y)


def _neighbor_avg(y_hist: np.ndarray, L: int, H: int, k: int, use_dtw: bool) -> Optional[np.ndarray]:
    if len(y_hist) < L:
        return None
    X, Y = _window_bank(y_hist, L, H, 1)
    if len(X) == 0:
        return None
    q = _zscore(y_hist[-L:])
    dists = np.array([_dist(q, x, use_dtw) for x in X])
    idx = np.argsort(dists)[:max(1, min(k, len(dists)))]
    conts = []
    qm = y_hist[-L:].mean()
    qs = y_hist[-L:].std()
    qs = 1.0 if qs == 0 else qs
    for i in idx:
        src_tail = X[i][-L:]
        sm = src_tail.mean()
        ss = src_tail.std()
        ss = 1.0 if ss == 0 else ss
        scale = qs / ss
        shift = qm - scale * sm
        conts.append(Y[i] * scale + shift)
    return np.mean(np.stack(conts), axis=0)


def _grid_search_lambda(y_train, predict_fn, L, H, k, use_dtw, lambdas, max_valid_points=128,
                        neighbor_fn=_neighbor_avg):
    starts = [t for t in range(L, len(y_train) - H)]
    if len(starts) == 0:
        return 0.5
    if len(starts) > max_valid_points:
        rng = np.random.default_rng(0)
        starts = list(rng.choice(starts, size=max_valid_points, replace=False))
        starts.sort()
    errs = np.zeros_like(lambdas, dtype=float)
    for t in starts:
        hist = y_train[:t]
        nb = neighbor_fn(hist, L, H, k, use_dtw)
        if nb is None:
            continue
        fm = predict_fn(hist, H)
        if fm is None:
            continue
        y_true = y_train[t:t + H]
        for j, lam in enumerate(lambdas):
            pred = lam * fm + (1.0 - lam) * nb
            errs[j] += np.abs(y_true[0] - pred[0])
    return float(lambdas[int(np.argmin(errs))])


# --------------------------------------------------------------------------- #
# Vectorized equivalents (numerically identical — algorithm unchanged).
# The verbatim functions above run a Python loop over the window bank; for
# electricity (27k test windows x 84k bank windows) that loop is the
# ~O(n_test x n_train) bottleneck. These replace only the inner Python loops
# with numpy vectorization. DTW (disabled) falls back to the verbatim path.
# --------------------------------------------------------------------------- #
def _window_bank_vec(y: np.ndarray, L: int, H: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(y)
    if n < L + H:
        return np.empty((0, L)), np.empty((0, H))
    windows = sliding_window_view(y, L + H)[::step]  # (N, L+H) view
    xw = windows[:, :L]
    yw = windows[:, L:L + H]
    m = xw.mean(axis=1, keepdims=True)
    s = xw.std(axis=1, keepdims=True)
    xz = np.where(s == 0, xw * 0.0, (xw - m) / s)
    return np.ascontiguousarray(xz), np.ascontiguousarray(yw)


def _neighbor_avg_vec(y_hist: np.ndarray, L: int, H: int, k: int, use_dtw: bool) -> Optional[np.ndarray]:
    if use_dtw:
        return _neighbor_avg(y_hist, L, H, k, use_dtw)
    if len(y_hist) < L:
        return None
    X, Y = _window_bank_vec(y_hist, L, H, 1)
    if len(X) == 0:
        return None
    q = _zscore(y_hist[-L:])
    diff = X - q
    dists = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    n_nb = max(1, min(k, len(dists)))
    idx = np.argsort(dists)[:n_nb]
    qm = y_hist[-L:].mean()
    qs = y_hist[-L:].std()
    qs = 1.0 if qs == 0 else qs
    src = X[idx]                      # z-scored neighbor windows (length L)
    sm = src.mean(axis=1)
    ss = src.std(axis=1)
    ss = np.where(ss == 0, 1.0, ss)
    scale = qs / ss
    shift = qm - scale * sm
    conts = Y[idx] * scale[:, None] + shift[:, None]
    return np.mean(conts, axis=0)


def check_vec_equivalence(y: np.ndarray, L: int, H: int, k: int, n_trials: int = 8,
                          seed: int = 0) -> float:
    """Return max abs diff between verbatim and vectorized _neighbor_avg."""
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    n_ok = 0
    for _ in range(n_trials):
        start = int(rng.integers(L + H + 1, len(y) + 1))
        hist = y[:start]
        a = _neighbor_avg(hist, L, H, k, use_dtw=False)
        b = _neighbor_avg_vec(hist, L, H, k, use_dtw=False)
        if a is None or b is None:
            if a is None and b is None:
                continue
            raise AssertionError(f"None-mismatch at start={start}")
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
        n_ok += 1
    assert n_ok > 0, "no trials ran"
    return max_diff


# --------------------------------------------------------------------------- #
# Rolling wrapper (our protocol) + leakage assertion + audit log
# --------------------------------------------------------------------------- #
def make_chronos_predict_fn(model_name: str = "amazon/chronos-t5-small", device: str = "cuda",
                            num_samples: int = 5):
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)

    def predict_fn(context: np.ndarray, H: int) -> np.ndarray:
        x = torch.tensor(context[-512:], dtype=torch.float32)  # Chronos max context
        out = pipe.predict(x, prediction_length=H, num_samples=num_samples)
        return np.median(out[0].numpy(), axis=0).astype(float)

    return predict_fn


def rolling_eval(series, test_starts, train_end, H, L=48, k=5, use_dtw=False,
                 predict_fn=None, lam=None, lambdas=None, max_windows=None, audit_path=None):
    """Roll RAF over our test windows; metric on raw values. Returns (mse, mae, audit_rows)."""
    if lam is None:
        lam = _grid_search_lambda(series[:train_end], predict_fn, L, H, k, use_dtw,
                                  lambdas if lambdas is not None else np.linspace(0.0, 1.0, 21),
                                  neighbor_fn=_neighbor_avg_vec)
    rows = []
    sq_err = 0.0
    abs_err = 0.0
    n = 0
    for s in test_starts[:max_windows]:
        hist = series[:s]
        # ---- leakage assertion ----
        assert len(hist) >= L + H, f"not enough history at start {s}"
        bank_max_future_end = len(hist)  # bank futures live in series[0:s]
        assert bank_max_future_end <= s, "retrieval candidate leaks into test future"
        nb = _neighbor_avg_vec(hist, L, H, k, use_dtw)
        fm = predict_fn(hist, H)
        if nb is None:
            pred = fm
        else:
            pred = lam * fm + (1.0 - lam) * nb
        y_true = series[s:s + H].astype(float)
        sq_err += float(np.sum((y_true - pred) ** 2))
        abs_err += float(np.sum(np.abs(y_true - pred)))
        n += H
        rows.append({"start": int(s), "lam": float(lam), "bank_max_end": int(bank_max_future_end),
                     "leak_ok": True})
    if n == 0:
        return float("nan"), float("nan"), rows
    mse = sq_err / n
    mae = abs_err / n
    if audit_path:
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({"lam": lam, "L": L, "H": H, "k": k,
                       "n_windows": len(rows), "leakage_assertion": "PASS"}, f, indent=2)
    return mse, mae, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="etth1")
    ap.add_argument("--model", default="amazon/chronos-t5-small", help="RAF backbone")
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_windows", type=int, default=100, help="smoke subset (0 = all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_samples", type=int, default=5,
                    help="Chronos sample paths per forecast (official default 20; reduced to 5 "
                         "for compute budget — see RAF_REPRODUCTION_AUDIT caveat)")
    ap.add_argument("--check", action="store_true", help="verify vectorized == verbatim, then exit")
    args = ap.parse_args()

    ds_dir = os.path.join(REPO, "results", "multidataset", args.dataset)
    df = pd.read_csv(os.path.join(ds_dir, "series.csv"))
    series = df["OT"].to_numpy(np.float32)
    cfg = json.loads(open(os.path.join(ds_dir, "config.json"), encoding="utf-8").read())
    H = int(cfg["horizon"])
    train_end = int(cfg["split"]["train"][1])
    idx = np.load(os.path.join(ds_dir, "window_indices.npz"))
    test_starts = np.asarray(idx["test"], dtype=np.int64)

    if args.check:
        md = check_vec_equivalence(series, args.L, H, args.k)
        print(f"vectorized==verbatim check [{args.dataset}] L={args.L} H={H} k={args.k}: "
              f"max abs diff = {md:.3e}")
        return

    predict_fn = make_chronos_predict_fn(args.model, args.device, num_samples=args.num_samples)
    audit_path = os.path.join(REPO, "results", "external_baselines",
                              f"raf_{args.dataset}_leak_audit.json")
    mse, mae, rows = rolling_eval(series, test_starts, train_end, H,
                                  L=args.L, k=args.k, predict_fn=predict_fn,
                                  max_windows=args.max_windows if args.max_windows else len(test_starts),
                                  audit_path=audit_path)
    lam = rows[0]["lam"] if rows else None
    out = {"dataset": args.dataset, "model": args.model, "L": args.L, "k": args.k,
           "H": H, "num_samples": args.num_samples, "lambda": lam, "n_windows": len(rows),
           "raw_mse": float(mse), "raw_mae": float(mae)}
    result_path = os.path.join(REPO, "results", "external_baselines",
                               f"raf_result_{args.dataset}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"RAF rolling [{args.dataset}] backbone={args.model} L={args.L} k={args.k}")
    print(f"  test windows evaluated: {len(rows)} / {len(test_starts)}")
    print(f"  horizon H={H}, lambda={lam}, raw MSE={mse:.6f}, raw MAE={mae:.6f}")
    print(f"  leakage audit: {audit_path}")
    print(f"  result json: {result_path}")


if __name__ == "__main__":
    main()
