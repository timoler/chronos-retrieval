#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-season pseudo-validation driver for V3.

Goal
----
Use TRAIN only. Never touch TEST.
Build two continuous pseudo-validation blocks:

Spring:
    past      < 2006-03-01
    pseudoVAL = 2006-03-01 .. 2006-04-30
Summer:
    past      < 2006-06-01
    pseudoVAL = 2006-06-01 .. 2006-07-31

For each block run four feature groups:
    hour                 0,1
    hour+doy             0,1,2,3
    hour+level/change    0,1,4,5
    all6                 0,1,2,3,4,5

IMPORTANT FOR CLAUDE
--------------------
Do NOT re-implement the frozen Shape retriever differently.
Wire export_frozen_shape_scores() to the exact functions already used in
build_solar_expert_mlp_fair_bundle.py:
    residual = seasonal_decompose(ctx, PRED=96, PERIOD=144, TREND_WINDOW=288)
    score    = Pearson(residual_query, residual_memory)
No raw cosine, no new temperature, no pseudo-VAL future leakage.

This file intentionally raises NotImplementedError at that adapter until the
existing frozen-score implementation is connected.
"""

from __future__ import annotations
import argparse, csv, json, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "recent_level", "recent_change"
]

CONFIGS = [
    ("hour", "0,1"),
    ("hour_doy", "0,1,2,3"),
    ("hour_level_change", "0,1,4,5"),
    ("all6", "0,1,2,3,4,5"),
]

@dataclass
class Block:
    name: str
    val_start: str
    val_end: str

BLOCKS = [
    Block("spring", "2006-03-01 00:00:00", "2006-04-30 23:59:59"),
    Block("summer", "2006-06-01 00:00:00", "2006-07-31 23:59:59"),
]


def _find(d, names):
    for n in names:
        if n in d:
            return n
    return None


def load_full_train_source(path: Path):
    """
    Expected full-TRAIN window source. Claude may replace this function by
    directly importing the existing builder's raw-data/window-generation code.

    Required logical arrays:
      context       [N,L] or [N,L,C]
      future        [N,H]
      expert        [N,6]
      forecast_time [N]  (forecast-start timestamps)
    """
    if "test" in str(path).lower():
        raise RuntimeError("Refusing source path containing 'test'.")

    # Build full-TRAIN windows directly from data/solar.csv, reusing the exact
    # window/expert-feature generation from build_solar_expert_mlp_fair_bundle.py
    # (univariate OT, L=512, H=96, 6 expert features). ``path`` is only a TEST guard.
    from build_solar_expert_mlp_fair_bundle import CONTEXT, PRED, build_windows

    df = pd.read_csv("data/solar.csv")
    df["date"] = pd.to_datetime(df["date"])
    ot = df["OT"].to_numpy(np.float32)
    dates = df["date"]
    N = len(df)
    train_len = int(0.6 * N)

    # full-TRAIN forecast-start indices (stride 1); future ends at train_len (no leak)
    s = np.arange(CONTEXT, train_len - PRED + 1)
    ctx, fut, exp = build_windows(s, dates, ot)
    ts = pd.DatetimeIndex(dates.iloc[s].to_numpy())

    # Hard safety: this experiment must be TRAIN-only (nothing on/after 2006-10-20).
    if ts.max() >= pd.Timestamp("2006-10-20 00:00:00"):
        raise RuntimeError(
            "TRAIN source exceeds expected range (>= 2006-10-20)."
        )

    return ctx, fut.astype(np.float32), exp[:, :6].astype(np.float32), ts


def split_block(ts, block: Block, train_b_days=14, train_b_stride=6):
    """
    Continuous chronological split:
      TRAIN_A = earlier past
      TRAIN_B = last train_b_days immediately before pseudo-VAL, stride sampled
      VAL     = continuous pseudo-validation block

    Assignment uses forecast-start timestamp.
    """
    v0 = pd.Timestamp(block.val_start)
    v1 = pd.Timestamp(block.val_end)
    b0 = v0 - pd.Timedelta(days=train_b_days)

    a = np.flatnonzero(ts < b0)
    b_full = np.flatnonzero((ts >= b0) & (ts < v0))
    b = b_full[::train_b_stride]
    v = np.flatnonzero((ts >= v0) & (ts <= v1))

    if min(len(a), len(b), len(v)) == 0:
        raise ValueError(
            f"{block.name}: empty split A={len(a)}, B={len(b)}, V={len(v)}"
        )

    assert ts[a].max() < ts[b_full].min()
    assert ts[b_full].max() < ts[v].min()
    return a, b, v


def export_frozen_shape_scores(train_a_context, train_b_context, val_context):
    """
    CLAUDE: replace ONLY this function body by importing/reusing the exact
    frozen residual-Pearson score implementation from
    build_solar_expert_mlp_fair_bundle.py.

    Required return:
      train_b_shape_scores : [Nb, Na]
      val_shape_scores     : [Nv, Na+Nb]

    Exact intended logic from current fair pipeline:
      R_a = seasonal residual(train_a_context, PRED=96, PERIOD=144, TREND_WINDOW=288)
      R_b = seasonal residual(train_b_context, PRED=96, PERIOD=144, TREND_WINDOW=288)
      R_v = seasonal residual(val_context,     PRED=96, PERIOD=144, TREND_WINDOW=288)

      train_b_shape_scores = pearson(R_b, R_a)
      val_shape_scores = pearson(R_v, concat(R_a, R_b))

    Do not use val future here.
    """
    # Reuse the EXACT frozen residual-Pearson implementation from the fair bundle
    # builder (seasonal_decompose PRED=96/PERIOD=144/TREND_WINDOW=288 -> pearson).
    from build_solar_expert_mlp_fair_bundle import shape_scores

    train_b_shape_scores = shape_scores(train_b_context, train_a_context)
    mem_context = np.concatenate([train_a_context, train_b_context], axis=0)
    val_shape_scores = shape_scores(val_context, mem_context)
    return train_b_shape_scores, val_shape_scores


def build_bundle(ctx, fut, exp, ts, block, out_path, train_b_days, train_b_stride):
    ia, ib, iv = split_block(ts, block, train_b_days, train_b_stride)

    a_ctx, a_fut, a_exp = ctx[ia], fut[ia], exp[ia]
    b_ctx, b_fut, b_exp = ctx[ib], fut[ib], exp[ib]
    v_ctx, v_fut, v_exp = ctx[iv], fut[iv], exp[iv]

    bs, vs = export_frozen_shape_scores(a_ctx, b_ctx, v_ctx)

    if bs.shape != (len(ib), len(ia)):
        raise ValueError(f"bad train_b_shape_scores shape {bs.shape}")
    if vs.shape != (len(iv), len(ia) + len(ib)):
        raise ValueError(f"bad val_shape_scores shape {vs.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        train_a_context=a_ctx,
        train_a_future=a_fut,
        train_a_expert=a_exp,
        train_b_context=b_ctx,
        train_b_future=b_fut,
        train_b_expert=b_exp,
        val_context=v_ctx,
        val_future=v_fut,
        val_expert=v_exp,
        train_b_shape_scores=bs.astype(np.float32),
        val_shape_scores=vs.astype(np.float32),
    )

    meta = {
        "block": block.name,
        "pseudo_val": [block.val_start, block.val_end],
        "train_b_days": train_b_days,
        "train_b_stride": train_b_stride,
        "counts": {
            "train_a": int(len(ia)),
            "train_b": int(len(ib)),
            "val": int(len(iv)),
            "val_memory": int(len(ia) + len(ib)),
        },
        "ranges": {
            "train_a": [str(ts[ia].min()), str(ts[ia].max())],
            "train_b": [str(ts[ib].min()), str(ts[ib].max())],
            "pseudo_val": [str(ts[iv].min()), str(ts[iv].max())],
        },
        "test_accessed": False,
        "chronos_used": False,
    }
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def run_v3(python_exec, v3_script, bundle, out_dir, cols,
           epochs, temp, topk, seed, device):
    cmd = [
        python_exec, str(v3_script),
        "--bundle", str(bundle),
        "--out-dir", str(out_dir),
        "--expert-cols", cols,
        "--epochs", str(epochs),
        "--softmax-temperature", str(temp),
        "--topk", str(topk),
        "--seed", str(seed),
        "--device", str(device),
    ]
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    rp = out_dir / "report.json"
    return json.loads(rp.read_text(encoding="utf-8"))


def g(d, *path):
    x = d
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return None
        x = x[k]
    return x


def summary_row(block, config, report):
    sm = g(report, "memory_only", "shape", "mse")
    em = g(report, "memory_only", "expert_only", "mse")
    hm = g(report, "memory_only", "expert_main_shape_aux", "mse")
    return {
        "block": block,
        "config": config,
        "shape_memory_mse": sm,
        "expert_memory_mse": em,
        "hybrid_memory_mse": hm,
        "expert_vs_shape_pct": (sm-em)/sm*100 if sm else None,
        "hybrid_vs_expert_pct": (em-hm)/em*100 if em else None,
        "expert_gamma": g(report, "memory_only", "expert_only", "gamma"),
        "shape_beta": g(report, "memory_only", "expert_main_shape_aux", "beta"),
        "expert_hybrid_topk_overlap_pct": (
            g(report, "topk_overlap", "expert_vs_hybrid") * 100
            if g(report, "topk_overlap", "expert_vs_hybrid") is not None else None
        ),
    }


def save_summary(rows, root):
    (root / "cross_season_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (root / "cross_season_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    lines = [
        "# Cross-season pseudo-validation summary",
        "",
        "| Block | Config | Shape Mem | Expert Mem | Hybrid Mem | Expert vs Shape | Hybrid vs Expert | beta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['block']} | {r['config']} "
            f"| {r['shape_memory_mse']:.2f} "
            f"| {r['expert_memory_mse']:.2f} "
            f"| {r['hybrid_memory_mse']:.2f} "
            f"| {r['expert_vs_shape_pct']:+.2f}% "
            f"| {r['hybrid_vs_expert_pct']:+.2f}% "
            f"| {r['shape_beta']:.4f} |"
        )
    lines += [
        "",
        "重点比较：",
        "- hour vs hour_doy: doy 独立增量",
        "- hour vs hour_level_change: level/change 独立增量",
        "- hour vs all6: 联合增量",
        "- hour_level_change vs all6: 在状态上下文下加入 doy 的增量",
        "- hour_doy vs all6: 在季节上下文下加入 level/change 的增量",
    ]
    (root / "cross_season_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="TRAIN-only full-window source NPZ; Claude may adapt loader to existing builder")
    ap.add_argument("--v3-script", type=Path,
                    default=Path("experiments/expert_main_shape_aux_v3.py"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("results/cross_season_v3"))
    ap.add_argument("--train-b-days", type=int, default=14)
    ap.add_argument("--train-b-stride", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--softmax-temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    ctx, fut, exp, ts = load_full_train_source(args.source)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for block in BLOCKS:
        root = args.out_root / block.name
        bundle = root / f"{block.name}_fair_bundle.npz"
        print("\n" + "="*90)
        print("BUILD", block.name, block.val_start, "->", block.val_end)
        print("="*90)
        meta = build_bundle(
            ctx, fut, exp, ts, block, bundle,
            args.train_b_days, args.train_b_stride
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))

        if args.build_only:
            continue

        for config, cols in CONFIGS:
            report = run_v3(
                args.python, args.v3_script, bundle, root/config, cols,
                args.epochs, args.softmax_temperature, args.topk,
                args.seed, args.device
            )
            rows.append(summary_row(block.name, config, report))

    if rows:
        save_summary(rows, args.out_root)
        print("\nsummary:", args.out_root / "cross_season_summary.md")


if __name__ == "__main__":
    main()
