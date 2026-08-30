"""Build the Solar NPZ bundle for experiments/expert_mlp_fair_ablation_v1.py.

This bundle adds, on top of the V0 bundle, the two ORIGINAL frozen-retriever
shape similarity score matrices that V1 requires:

    train_b_shape_scores : [Nb, Na]      TRAIN_B query vs TRAIN_A memory
    val_shape_scores     : [Nv, Na+Nb]   VAL query vs TRAIN_A+TRAIN_B memory

The "shape score" is exported from the FROZEN original retriever exactly as it
computes it (NOT re-implemented differently):

    S_shape = pearson_similarity_matrix(R_query, R_memory)
    R = seasonal_decompose(context, PRED, PERIOD=144, TREND_WINDOW=288).residual

i.e. the Pearson correlation of the seasonal-deseasonalized residual, the same
S_shape the frozen Agent v2 pipeline uses. We do NOT use the raw standardized
context (that is V0's different "shape" definition).

Same leak-safe split as V0:
    TRAIN_A memory  : forecast-starts [L, train_a_end - H], stride 1
    TRAIN_B queries : forecast-starts [train_a_end, train_len - H], stride 6
    VAL queries     : forecast-starts [val_start, val_end - H], stride 6
    TEST            : never accessed.

Context = univariate OT (L=512), future = OT (H=96), expert dim M=6.
chronos_val_pred is included from the existing cache (aligned to VAL windows).
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

from retrieval.decompose import seasonal_decompose          # noqa: E402
from retrieval.similarity import pearson_similarity_matrix  # noqa: E402

CONTEXT = 512     # L
PRED = 96         # H
PERIOD = 144      # daily period for 10-min data
TREND_WINDOW = 288  # 2 * period
VAL_STRIDE = 6
TRAIN_B_FRAC = 0.20

OUT_DIR = Path("results/expert_mlp_retrieval_v1")
BUNDLE = OUT_DIR / "solar_fair_bundle.npz"
CHRONOS_CACHE = Path("results/solar_chronos.npz")

EXPERT_COLS = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "recent_level", "recent_change",
]


def _expert_at(s, dates, ot):
    ts = dates.iloc[s]
    hour_frac = ts.hour + ts.minute / 60.0
    doy = float(ts.dayofyear)
    return np.array([
        np.sin(2 * np.pi * hour_frac / 24.0),
        np.cos(2 * np.pi * hour_frac / 24.0),
        np.sin(2 * np.pi * doy / 365.0),
        np.cos(2 * np.pi * doy / 365.0),
        float(ot[s - CONTEXT:s].mean()),
        float(ot[s - 1] - ot[s - CONTEXT]),
    ], dtype=np.float32)


def build_windows(s_indices, dates, ot):
    ctx = np.stack([ot[s - CONTEXT:s] for s in s_indices]).astype(np.float32)
    fut = np.stack([ot[s:s + PRED] for s in s_indices]).astype(np.float32)
    exp = np.stack([_expert_at(s, dates, ot) for s in s_indices]).astype(np.float32)
    return ctx, fut, exp


def residual_of(ctx):
    """Frozen original-retriever 'shape' = residual of seasonal_decompose."""
    _, _, R, _, _ = seasonal_decompose(ctx, PRED, PERIOD, TREND_WINDOW)
    return R  # (n, L) float32


def shape_scores(query_ctx, mem_ctx):
    """pearson_similarity_matrix(residual_q, residual_m) -> [n_query, n_mem]."""
    Rq = residual_of(query_ctx)
    Rm = residual_of(mem_ctx)
    return pearson_similarity_matrix(Rq, Rm)  # float64


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

    s_mem = np.arange(CONTEXT, train_a_end - PRED + 1)                   # TRAIN_A
    s_trainb = np.arange(train_a_end, train_len - PRED + 1, VAL_STRIDE)  # TRAIN_B
    s_val = np.arange(val_start, val_end - PRED + 1, VAL_STRIDE)         # VAL

    mem_future_end = int(s_mem.max() + PRED) if len(s_mem) else -1
    assert mem_future_end <= train_a_end
    assert train_a_end <= int(s_trainb.min())
    assert (int(s_trainb.max()) + PRED) <= val_start
    assert (int(s_val.max()) + PRED) <= val_end

    a_ctx, a_fut, a_exp = build_windows(s_mem, dates, ot)
    b_ctx, b_fut, b_exp = build_windows(s_trainb, dates, ot)
    v_ctx, v_fut, v_exp = build_windows(s_val, dates, ot)

    # frozen original-retriever shape score matrices
    print("computing train_b_shape_scores [Nb, Na] ...")
    train_b_shape_scores = shape_scores(b_ctx, a_ctx).astype(np.float32)  # [Nb, Na]

    print("computing val_shape_scores [Nv, Na+Nb] ...")
    mem_ctx = np.concatenate([a_ctx, b_ctx], axis=0)                      # TRAIN_A + TRAIN_B
    val_shape_scores = shape_scores(v_ctx, mem_ctx).astype(np.float32)    # [Nv, Na+Nb]

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
        train_b_shape_scores=train_b_shape_scores,
        val_shape_scores=val_shape_scores,
        **extra,
    )

    meta = {
        "dataset": "solar",
        "shape_score_definition": "pearson_similarity_matrix(residual, residual) where "
                                  "residual = seasonal_decompose(ctx, PRED=96, PERIOD=144, TREND_WINDOW=288).R "
                                  "(frozen original retriever S_shape)",
        "context_len": CONTEXT, "pred_len": PRED, "val_stride": VAL_STRIDE,
        "N": N, "train_len": train_len, "train_a_end": train_a_end,
        "val_start": val_start, "val_end": val_end, "test_start": val_end,
        "split": {
            "train_a_memory": int(len(s_mem)),
            "train_b_queries": int(len(s_trainb)),
            "val_queries": int(len(s_val)),
            "val_memory": int(len(s_mem)) + int(len(s_trainb)),
        },
        "shapes": {
            "train_b_shape_scores": list(train_b_shape_scores.shape),
            "val_shape_scores": list(val_shape_scores.shape),
        },
        "expert_cols": EXPERT_COLS,
        "context_is": "univariate OT only",
        "chronos_val_pred": "chronos_val_pred" in extra,
        "test_accessed": False,
    }
    (OUT_DIR / "bundle_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Solar fair-ablation V1 bundle built")
    print("=" * 72)
    print(f"  train_a_memory={meta['split']['train_a_memory']}  "
          f"train_b_queries={meta['split']['train_b_queries']}  "
          f"val_queries={meta['split']['val_queries']}  "
          f"val_memory={meta['split']['val_memory']}")
    print(f"  train_b_shape_scores shape={meta['shapes']['train_b_shape_scores']}")
    print(f"  val_shape_scores shape={meta['shapes']['val_shape_scores']}")
    print(f"  chronos_val_pred included: {meta['chronos_val_pred']}")
    print(f"  bundle: {BUNDLE}")
    print("=" * 72)


if __name__ == "__main__":
    main()
