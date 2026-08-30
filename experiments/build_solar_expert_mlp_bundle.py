"""Build the Solar NPZ bundle for experiments/expert_mlp_retrieval_v0.py.

This is a NEW, separate retriever experiment (Expert-Conditioned MLP). It does
NOT touch the frozen Agent v2 / Shape / Expert / Gate scripts. It only prepares
a leak-safe input bundle over Solar-Energy, then hands off to the MLP script.

Split (chronological, leak-safe):
    TRAIN_A = memory bank   : forecast-starts in [L, train_a_end - H], stride 1
    TRAIN_B = MLP training  : forecast-starts in [train_a_end, train_len - H], stride 6
    VAL     = model select  : forecast-starts in [val_start, val_end - H], stride 6
    TEST    = NEVER touched.

With  L = CONTEXT = 512,  H = PRED = 96  (matching the frozen pipeline and the
cached Chronos VAL forecast, so chronos_val_pred aligns to VAL windows).

Context is the UNIVARIATE target OT only (no P001..P006). This keeps the shape
representation free of any "P001..P006 represents the population" assumption.

Expert features (M = 6), each a per-window scalar computed at forecast-start s,
using ONLY past info (timestamp + the context window) -> no future leakage:

    col 0  hour_sin         (diurnal position, circular)
    col 1  hour_cos         (diurnal position, circular)
    col 2  doy_sin          (seasonal position, circular, day-of-year)
    col 3  doy_cos          (seasonal position, circular, day-of-year)
    col 4  recent_level     (mean of the context window -> high/low output state)
    col 5  recent_change    (OT[s-1] - OT[s-L] -> recent momentum / direction)

These map to the established Solar mechanisms (diurnal cycle, seasonal cycle,
short-term persistence) WITHOUT asserting P001..P006 representativeness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# frozen global window constants (same as priors.validator / run_solar_agent_v2_val)
CONTEXT = 512     # L
PRED = 96         # H
VAL_STRIDE = 6    # hourly query stride, matches the cached Chronos VAL windows
TRAIN_B_FRAC = 0.20

OUT_DIR = Path("results/expert_mlp_retrieval_v0")
BUNDLE = OUT_DIR / "solar_bundle.npz"
CHRONOS_CACHE = Path("results/solar_chronos.npz")

EXPERT_COLS = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "recent_level", "recent_change",
]


def _expert_at(s: int, dates, ot) -> np.ndarray:
    ts = dates.iloc[s]
    hour_frac = ts.hour + ts.minute / 60.0
    doy = float(ts.dayofyear)

    hour_sin = np.sin(2.0 * np.pi * hour_frac / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hour_frac / 24.0)
    doy_sin = np.sin(2.0 * np.pi * doy / 365.0)
    doy_cos = np.cos(2.0 * np.pi * doy / 365.0)

    recent_level = float(ot[s - CONTEXT:s].mean())
    recent_change = float(ot[s - 1] - ot[s - CONTEXT])
    return np.array([hour_sin, hour_cos, doy_sin, doy_cos,
                     recent_level, recent_change], dtype=np.float32)


def build_windows(s_indices, dates, ot):
    ctx = np.stack([ot[s - CONTEXT:s] for s in s_indices]).astype(np.float32)
    fut = np.stack([ot[s:s + PRED] for s in s_indices]).astype(np.float32)
    exp = np.stack([_expert_at(s, dates, ot) for s in s_indices]).astype(np.float32)
    return ctx, fut, exp


def main():
    df = pd.read_csv("data/solar.csv")
    df["date"] = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(np.float32)
    dates = df["date"]

    N = len(df)
    train_len = int(0.6 * N)
    val_start = train_len
    val_end = train_len + int(0.2 * N)
    train_b_len = int(TRAIN_B_FRAC * train_len)
    train_a_end = train_len - train_b_len

    # forecast-start indices per split
    s_mem = np.arange(CONTEXT, train_a_end - PRED + 1)                       # TRAIN_A, stride 1
    s_trainb = np.arange(train_a_end, train_len - PRED + 1, VAL_STRIDE)      # TRAIN_B
    s_val = np.arange(val_start, val_end - PRED + 1, VAL_STRIDE)             # VAL

    # ---- leak-safety verification ----
    mem_future_end = int(s_mem.max() + PRED) if len(s_mem) else -1
    assert mem_future_end <= train_a_end, f"TRAIN_A future leaks: {mem_future_end} > {train_a_end}"
    assert train_a_end <= int(s_trainb.min()), "TRAIN_B query starts before TRAIN_A ends"
    assert (int(s_trainb.max()) + PRED) <= val_start, "TRAIN_B future leaks into VAL"
    assert (int(s_val.max()) + PRED) <= val_end, "VAL future exceeds val_end"

    a_ctx, a_fut, a_exp = build_windows(s_mem, dates, ot)
    b_ctx, b_fut, b_exp = build_windows(s_trainb, dates, ot)
    v_ctx, v_fut, v_exp = build_windows(s_val, dates, ot)

    # optional aligned Chronos VAL forecast
    extra = {}
    if CHRONOS_CACHE.exists():
        yc = np.load(CHRONOS_CACHE)["y_chronos_val"].astype(np.float32)
        assert yc.shape == v_fut.shape, f"chronos {yc.shape} vs val_future {v_fut.shape}"
        extra["chronos_val_pred"] = yc

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BUNDLE,
        train_a_context=a_ctx, train_a_future=a_fut, train_a_expert=a_exp,
        train_b_context=b_ctx, train_b_future=b_fut, train_b_expert=b_exp,
        val_context=v_ctx, val_future=v_fut, val_expert=v_exp,
        **extra,
    )

    meta = {
        "dataset": "solar",
        "context_len": CONTEXT,
        "pred_len": PRED,
        "val_stride": VAL_STRIDE,
        "N": N,
        "train_len": train_len,
        "train_a_end": train_a_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": val_end,
        "split": {
            "train_a_memory": {"n": int(len(s_mem)), "forecast_start_range": [int(s_mem[0]), int(s_mem[-1])]},
            "train_b_queries": {"n": int(len(s_trainb)), "forecast_start_range": [int(s_trainb[0]), int(s_trainb[-1])]},
            "val_queries": {"n": int(len(s_val)), "forecast_start_range": [int(s_val[0]), int(s_val[-1])]},
        },
        "expert_cols": EXPERT_COLS,
        "expert_col_semantics": {
            "hour_sin/hour_cos": "diurnal position (circular hour-of-day)",
            "doy_sin/doy_cos": "seasonal position (circular day-of-year)",
            "recent_level": "mean of context window (high/low output state)",
            "recent_change": "OT[s-1] - OT[s-L] (recent momentum/direction)",
        },
        "context_is": "univariate OT only (no P001..P006, no representativeness assumption)",
        "chronos_val_pred": "chronos_val_pred" in extra,
        "test_accessed": False,
    }
    (OUT_DIR / "bundle_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Solar Expert-MLP bundle built")
    print("=" * 72)
    for k in ["train_a_memory", "train_b_queries", "val_queries"]:
        print(f"  {k}: n={meta['split'][k]['n']}  "
              f"forecast_start_range={meta['split'][k]['forecast_start_range']}")
    print(f"  context_len={CONTEXT}  pred_len={PRED}  expert_dim={len(EXPERT_COLS)}")
    print(f"  chronos_val_pred included: {meta['chronos_val_pred']}")
    print(f"  bundle: {BUNDLE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
