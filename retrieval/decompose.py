"""Decomposition helpers for de-leveling the series before retrieval.

The retrieval failure mode we found is a *level* mismatch: Pearson matches
shape only, so a winter query can retrieve a summer memory whose absolute
level is way off, and the naive forecast copies that wrong level.

Two levels of correction are offered:

Step 4 -- linear detrend (``linear_detrend``):
    x = T + R with T a least-squares line, retrieve on R, add back the query's
    own extrapolated T.  (Result: no gain over mean-anchoring -- the linear
    slope itself is transferable signal, so removing it loses information.)

Step 5 -- structure decomposition (``seasonal_decompose``):
    x = T + S + R with T a slow moving average, S the per-window daily pattern
    (mean by hour-of-day, tiled), R the rest.  The caller computes a separate
    similarity for each component and combines them (``sim = a*sim_T +
    b*sim_S + g*sim_R``) instead of only retrieving on R.
"""

import numpy as np


def linear_detrend(contexts, prediction_length):
    """Linear-detrend each row of `contexts` and extrapolate the trend.

    The trend is fit on the context only (never the future), so the returned
    future trend is leak-free: it is the query's own projected level.

    Args:
        contexts: (n, L) array of context windows.
        prediction_length: H, the forecast horizon.

    Returns:
        resid_context: (n, L) float32, contexts minus the fitted line.
        trend_future:  (n, H) float32, the fitted line extrapolated H steps.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    n, L = contexts.shape

    t = np.arange(L, dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T  # (L, 2): [time, 1]
    coef, *_ = np.linalg.lstsq(A, contexts.T, rcond=None)  # (2, n)

    trend_ctx = (A @ coef).T          # (n, L)
    resid_ctx = contexts - trend_ctx

    tf = np.arange(L, L + prediction_length, dtype=np.float64)
    Af = np.vstack([tf, np.ones_like(tf)]).T  # (H, 2)
    trend_fut = (Af @ coef).T                # (n, H)

    return resid_ctx.astype(np.float32), trend_fut.astype(np.float32)


def residual_future(futures, trend_future):
    """Subtract the (extrapolated) trend from a window's future.

    Used on memory windows so their stored "answer" is stored as a shape
    (deviation from its own trend) rather than an absolute level.
    """
    futures = np.asarray(futures, dtype=np.float64)
    trend_future = np.asarray(trend_future, dtype=np.float64)
    return (futures - trend_future).astype(np.float32)


def _moving_average(x, window):
    """Centered moving average along the last axis, edge-padded."""
    n, L = x.shape
    pad = window // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    cs = np.cumsum(xp, axis=1)
    return (cs[:, window:] - cs[:, :-window]) / window  # (n, L)


def seasonal_decompose(contexts, prediction_length, period=24, trend_window=48):
    """Decompose each window into trend (T), daily seasonal (S), residual (R).

    x = T + S + R.  T is a slow centered moving average (daily cycle removed),
    S is the per-window daily pattern (mean by hour-of-day, tiled), R the rest.

    Args:
        contexts: (n, L) context windows.
        prediction_length: H, the forecast horizon.
        period: seasonal period in samples (24 = daily for hourly data).
        trend_window: moving-average window for T (>= period to remove the
            daily cycle).

    Returns:
        T_ctx, S_ctx, R_ctx: (n, L) float32 components.
        T_fut, S_fut:        (n, H) float32; trend extrapolated linearly from
            its tail, seasonal tiled forward from the window end.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    n, L = contexts.shape
    H = prediction_length

    T_ctx = _moving_average(contexts, trend_window)      # (n, L)
    det = contexts - T_ctx

    # per-window daily pattern: mean of detrended by hour-of-day
    seasonal = np.zeros((n, period), dtype=np.float64)
    for h in range(period):
        seasonal[:, h] = det[:, h::period].mean(axis=1)
    S_ctx = seasonal[:, np.arange(L) % period]           # (n, L)

    R_ctx = contexts - T_ctx - S_ctx

    # future trend: linear extrapolation of the trend tail
    m = trend_window
    tail = T_ctx[:, -m:]                                 # (n, m)
    tt = np.arange(m, dtype=np.float64)
    A = np.vstack([tt, np.ones(m)]).T
    coef = np.linalg.lstsq(A, tail.T, rcond=None)[0]     # (2, n)
    tf = np.arange(m, m + H, dtype=np.float64)
    Af = np.vstack([tf, np.ones(H)]).T
    T_fut = (Af @ coef).T                                 # (n, H)

    # future seasonal: tile the daily pattern, continuing from the window end
    S_fut = seasonal[:, np.arange(L, L + H) % period]     # (n, H)

    return (
        T_ctx.astype(np.float32),
        S_ctx.astype(np.float32),
        R_ctx.astype(np.float32),
        T_fut.astype(np.float32),
        S_fut.astype(np.float32),
    )


def decompose_future(futures, T_fut, S_fut):
    """R_fut = futures - T_fut - S_fut (memory windows only)."""
    futures = np.asarray(futures, dtype=np.float64)
    T_fut = np.asarray(T_fut, dtype=np.float64)
    S_fut = np.asarray(S_fut, dtype=np.float64)
    return (futures - T_fut - S_fut).astype(np.float32)
