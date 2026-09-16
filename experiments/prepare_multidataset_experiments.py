#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_multidataset_experiments.py

统一准备 7 个数据集：
ETTh1 / ETTh2 / ETTm1 / ETTm2 / electricity / solar / traffic

原则
----
1. 统一输出 date + OT，先做单变量预测，保证跨数据集比较干净。
2. Traffic: OT = 862 个传感器占用率的均值。
3. 不构造 hour/doy/level/change 等人工 Expert 特征。
4. 不把专家知识硬编码成数值特征。
5. 为后续统一的：
   Expert Text -> Agent审核 -> Text Embedding
   -> Query + Expert Attention -> Retrieval
   准备数据、专家文本和统一配置。
6. 默认只准备数据，不运行 TEST。
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = {
    "ETTh1": {
        "candidates": ["data/ETTh1.csv", "data/ETT-small/ETTh1.csv"],
        "freq": "1h",
        "domain": "transformer_temperature",
        "expert_text": (
            "Transformer oil temperature has thermal inertia. "
            "Recent operating state and load-related conditions can affect future oil temperature. "
            "Historical periods with similar thermal and operating dynamics may be more useful for forecasting."
        ),
    },
    "ETTh2": {
        "candidates": ["data/ETTh2.csv", "data/ETT-small/ETTh2.csv"],
        "freq": "1h",
        "domain": "transformer_temperature",
        "expert_text": (
            "Transformer oil temperature has thermal inertia. "
            "Recent operating state and load-related conditions can affect future oil temperature. "
            "Historical periods with similar thermal and operating dynamics may be more useful for forecasting."
        ),
    },
    "ETTm1": {
        "candidates": ["data/ETTm1.csv", "data/ETT-small/ETTm1.csv"],
        "freq": "15min",
        "domain": "transformer_temperature",
        "expert_text": (
            "Transformer oil temperature has thermal inertia. "
            "Recent operating state and load-related conditions can affect future oil temperature. "
            "Historical periods with similar thermal and operating dynamics may be more useful for forecasting."
        ),
    },
    "ETTm2": {
        "candidates": ["data/ETTm2.csv", "data/ETT-small/ETTm2.csv"],
        "freq": "15min",
        "domain": "transformer_temperature",
        "expert_text": (
            "Transformer oil temperature has thermal inertia. "
            "Recent operating state and load-related conditions can affect future oil temperature. "
            "Historical periods with similar thermal and operating dynamics may be more useful for forecasting."
        ),
    },
    "electricity": {
        "candidates": ["data/electricity.csv"],
        "freq": "15min",
        "domain": "aggregate_electricity_load",
        "expert_text": (
            "Aggregate electricity demand often shows recurring usage regimes and short-term persistence. "
            "Historical periods reflecting similar demand dynamics can be useful, while a small subset of "
            "individual users should not be assumed to represent the total population."
        ),
    },
    "solar": {
        "candidates": ["data/solar.csv"],
        "freq": "10min",
        "domain": "aggregate_solar_generation",
        "expert_text": (
            "Solar generation is strongly affected by the current generation regime and short-term irradiance persistence. "
            "Historical periods with similar generation dynamics can provide useful forecasting analogues."
        ),
    },
    "traffic": {
        "candidates": ["data/traffic.csv"],
        "freq": "1h",
        "domain": "aggregate_traffic_occupancy",
        "expert_text": (
            "Traffic occupancy exhibits recurring congestion regimes and short-term persistence. "
            "Historical periods with similar network-level traffic dynamics can provide useful forecasting analogues."
        ),
    },
    "weather": {
        "candidates": ["data/weather.csv"],
        "freq": "10min",
        "domain": "weather_meteorological",
        "expert_text": (
            "Weather-station measurements evolve through diurnal and seasonal cycles "
            "with short-term persistence. Historical periods with similar recent "
            "meteorological dynamics may provide useful forecasting analogues."
        ),
    },
    "exchange_rate": {
        "candidates": ["data/exchange_rate.csv"],
        "freq": "daily",
        "domain": "daily_exchange_rate",
        "expert_text": (
            "Daily exchange rates show short-term persistence with weak weekly and "
            "annual structure. Historical periods with similar recent rate dynamics "
            "may provide useful forecasting analogues."
        ),
    },
}

def resolve_path(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None

def load_dataset(name, path):
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{name}: 缺少 date 列")

    df["date"] = pd.to_datetime(df["date"])

    if name == "traffic":
        sensor_cols = [c for c in df.columns if c.startswith("Sensor_")]
        if len(sensor_cols) != 862:
            raise ValueError(f"traffic: 预期 862 个 Sensor 列，实际 {len(sensor_cols)}")
        # 网络平均占用率
        ot = df[sensor_cols].mean(axis=1)
        out = pd.DataFrame({"date": df["date"], "OT": ot.astype("float32")})
    else:
        if "OT" not in df.columns:
            raise ValueError(f"{name}: 缺少 OT 列")
        out = df[["date", "OT"]].copy()
        out["OT"] = pd.to_numeric(out["OT"], errors="coerce").astype("float32")

    if out["OT"].isna().any():
        raise ValueError(f"{name}: OT 存在缺失值")

    if not out["date"].is_monotonic_increasing:
        out = out.sort_values("date").reset_index(drop=True)

    return out

def make_splits(n, train_ratio=0.6, val_ratio=0.2):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train": [0, train_end],
        "val": [train_end, val_end],
        "test": [val_end, n],
    }

def make_windows(start, end, context, horizon, stride):
    last_start = end - context - horizon
    if last_start < start:
        return np.empty((0,), dtype=np.int64)
    return np.arange(start, last_start + 1, stride, dtype=np.int64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=1)
    ap.add_argument("--test-stride", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=Path("results/multidataset"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for name in args.datasets:
        if name not in DATASETS:
            raise ValueError(f"未知数据集: {name}")

        cfg = DATASETS[name]
        src = resolve_path(cfg["candidates"])
        if src is None:
            print(f"[MISS] {name}: 未找到 {cfg['candidates']}")
            continue

        print("\n" + "=" * 72)
        print(f"[DATASET] {name} <- {src}")
        df = load_dataset(name, src)

        ds_dir = args.out_dir / name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # 统一后的单变量文件
        prepared_path = ds_dir / "series.csv"
        df.to_csv(prepared_path, index=False)

        splits = make_splits(len(df), args.train_ratio, args.val_ratio)

        train_idx = make_windows(
            *splits["train"], args.context, args.horizon, args.train_stride
        )
        val_idx = make_windows(
            *splits["val"], args.context, args.horizon, args.val_stride
        )
        test_idx = make_windows(
            *splits["test"], args.context, args.horizon, args.test_stride
        )

        # 这里只保存 TEST 索引，不运行 TEST。
        np.savez_compressed(
            ds_dir / "window_indices.npz",
            train=train_idx,
            val=val_idx,
            test=test_idx,
        )

        expert_path = ds_dir / "expert_knowledge.txt"
        expert_path.write_text(cfg["expert_text"] + "\n", encoding="utf-8")

        metadata = {
            "dataset": name,
            "source": str(src),
            "domain": cfg["domain"],
            "frequency": cfg["freq"],
            "rows": len(df),
            "context": args.context,
            "horizon": args.horizon,
            "split": splits,
            "num_windows": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "target": (
                "mean occupancy across 862 sensors"
                if name == "traffic"
                else "OT"
            ),
            "expert_pipeline": (
                "expert_text -> agent_review -> text_embedding "
                "-> query_expert_attention -> retrieval"
            ),
            "manual_expert_features": False,
            "test_policy": "indices prepared only; do not evaluate until model is frozen",
        }

        (ds_dir / "config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary.append(metadata)

        print(f"rows       : {len(df)}")
        print(f"time       : {df.date.iloc[0]} -> {df.date.iloc[-1]}")
        print(f"OT range   : {df.OT.min():.6f} -> {df.OT.max():.6f}")
        print(f"windows    : train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        print(f"prepared   : {prepared_path}")
        print(f"expert txt : {expert_path}")

    (args.out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for x in summary:
        rows.append({
            "dataset": x["dataset"],
            "domain": x["domain"],
            "frequency": x["frequency"],
            "rows": x["rows"],
            "train_windows": x["num_windows"]["train"],
            "val_windows": x["num_windows"]["val"],
            "test_windows": x["num_windows"]["test"],
            "target": x["target"],
        })

    if rows:
        pd.DataFrame(rows).to_csv(
            args.out_dir / "dataset_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n" + "=" * 72)
    print("数据准备完成。")
    print("注意：这个脚本没有实现/训练 Attention，也没有运行 TEST。")
    print("下一步把统一 Expert-Attention Retrieval 模型接到这些目录即可。")
    print("=" * 72)

if __name__ == "__main__":
    main()
