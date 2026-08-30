#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_solar_same_val_baselines.py

目的：
在当前 results/multidataset/solar/window_indices.npz 的同一批 VAL 窗口上，
补跑：
1) Shape Retrieval
2) Chronos
并与已有 Expert-Attention Memory 结果放到同一张表。

严格约束：
- context=512, horizon=96
- 使用当前统一 VAL 索引（默认应为 9905）
- TEST 完全不读取、不评估
- Shape 不使用专家文本
- Chronos 直接预测相同 VAL query
- 不做 lambda 融合，不调 Gate

依赖：
    pip install numpy pandas torch
Chronos 部分需要你当前项目已有的 chronos/autogluon 环境。
脚本会优先尝试 transformers 风格的 ChronosPipeline，
失败时给出明确提示，不影响 Shape 结果保存。

运行：
    python run_solar_same_val_baselines.py --root results/multidataset

如果你的 Chronos 模型接口不同，可先只跑 Shape：
    python run_solar_same_val_baselines.py --root results/multidataset --skip-chronos
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def build_windows(series, starts, context, horizon):
    X = np.stack([series[s:s+context] for s in starts]).astype(np.float32)
    Y = np.stack([series[s+context:s+context+horizon] for s in starts]).astype(np.float32)
    return X, Y


def residualize(x):
    """
    轻量 residual shape：
    每个窗口去线性趋势，再标准化。
    用于 Pearson/余弦 shape matching。
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    t = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    tc = t - t.mean()
    denom = np.sum(tc * tc)

    xm = x.mean(axis=-1, keepdims=True)
    slope = np.sum((x - xm) * tc[None, :], axis=-1, keepdims=True) / denom
    trend = xm + slope * tc[None, :]
    r = x - trend

    r = r - r.mean(axis=-1, keepdims=True)
    sd = r.std(axis=-1, keepdims=True)
    r = r / np.maximum(sd, 1e-6)
    return r


def shape_retrieval(Xq, Xbank, Ybank, topk=5, q_chunk=128, bank_chunk=4096):
    """
    Residual Pearson Top-K。
    residual 标准化后，点积/context 等价于 Pearson。
    为避免内存爆炸，query/bank 都分块。
    """
    Q = residualize(Xq)
    B = residualize(Xbank)
    L = Q.shape[1]

    all_pred = []
    for qs in range(0, len(Q), q_chunk):
        q = Q[qs:qs+q_chunk]
        nq = len(q)

        best_scores = np.full((nq, topk), -np.inf, dtype=np.float32)
        best_idx = np.full((nq, topk), -1, dtype=np.int64)

        for bs in range(0, len(B), bank_chunk):
            b = B[bs:bs+bank_chunk]
            scores = (q @ b.T) / L

            kk = min(topk, scores.shape[1])
            local_part = np.argpartition(scores, -kk, axis=1)[:, -kk:]
            local_scores = np.take_along_axis(scores, local_part, axis=1)
            local_idx = local_part + bs

            comb_scores = np.concatenate([best_scores, local_scores], axis=1)
            comb_idx = np.concatenate([best_idx, local_idx], axis=1)

            part = np.argpartition(comb_scores, -topk, axis=1)[:, -topk:]
            best_scores = np.take_along_axis(comb_scores, part, axis=1)
            best_idx = np.take_along_axis(comb_idx, part, axis=1)

        order = np.argsort(best_scores, axis=1)[:, ::-1]
        best_idx = np.take_along_axis(best_idx, order, axis=1)
        pred = Ybank[best_idx].mean(axis=1)
        all_pred.append(pred)

        print(f"Shape: {min(qs+q_chunk, len(Q))}/{len(Q)}")

    return np.concatenate(all_pred, axis=0)


def run_chronos(Xval, horizon, model_name, batch_size):
    """
    尝试官方 chronos 包常见接口。
    如果你项目中的 Chronos-2 调用方式不同，报错信息会明确显示。
    """
    import torch

    try:
        from chronos import ChronosPipeline
    except Exception as e:
        raise RuntimeError(
            "无法 import `from chronos import ChronosPipeline`。\n"
            "请在你之前能跑 Chronos 的环境执行，或把你当前 Chronos 调用代码发给我，我按你的接口改。\n"
            f"原始错误: {e}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("Loading Chronos:", model_name)
    print("device:", device)

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=dtype,
    )

    preds = []
    for i in range(0, len(Xval), batch_size):
        xb = Xval[i:i+batch_size]
        contexts = [torch.tensor(row, dtype=torch.float32) for row in xb]

        forecast = pipeline.predict(
            contexts,
            prediction_length=horizon,
        )

        # 常见输出 [B, num_samples, H]；取样本中位数
        if isinstance(forecast, list):
            batch_pred = []
            for f in forecast:
                f = f.detach().float().cpu().numpy()
                if f.ndim == 2:
                    batch_pred.append(np.median(f, axis=0))
                else:
                    batch_pred.append(np.asarray(f).reshape(-1)[-horizon:])
            batch_pred = np.stack(batch_pred)
        else:
            f = forecast.detach().float().cpu().numpy()
            if f.ndim == 3:
                batch_pred = np.median(f, axis=1)
            elif f.ndim == 2:
                batch_pred = f
            else:
                raise ValueError(f"未知 Chronos 输出 shape: {f.shape}")

        preds.append(batch_pred.astype(np.float32))
        print(f"Chronos: {min(i+batch_size, len(Xval))}/{len(Xval)}")

    return np.concatenate(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/multidataset"))
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-bank", type=int, default=25000)
    ap.add_argument("--skip-chronos", action="store_true")
    ap.add_argument("--chronos-model",
                    default="amazon/chronos-t5-small",
                    help="改成你项目实际使用的 Chronos checkpoint，必须和论文 baseline 保持一致")
    ap.add_argument("--chronos-batch", type=int, default=32)
    args = ap.parse_args()

    ds = args.root / args.dataset
    out = ds / "same_val_baselines"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)

    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

    print("=" * 70)
    print("SAME-VAL FAIR COMPARISON")
    print("dataset:", args.dataset)
    print("VAL windows:", len(val_starts))
    print("context/horizon:", args.context, args.horizon)
    print("TEST: NOT USED")
    print("=" * 70)

    # 与 Expert-Attention 脚本一致：前 80% TRAIN 作为 memory bank
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]

    if len(bank_starts) > args.max_bank:
        take = np.linspace(0, len(bank_starts)-1, args.max_bank).astype(int)
        bank_starts = bank_starts[take]

    Xbank, Ybank = build_windows(
        series, bank_starts, args.context, args.horizon
    )
    Xval, Yval = build_windows(
        series, val_starts, args.context, args.horizon
    )

    results = {
        "dataset": args.dataset,
        "val_windows": int(len(val_starts)),
        "context": args.context,
        "horizon": args.horizon,
        "topk": args.topk,
        "test_used": False,
    }

    # 1. Shape
    print("\n[1/2] Running Shape Retrieval...")
    shape_pred = shape_retrieval(
        Xval, Xbank, Ybank, topk=args.topk
    )
    results["shape_mse"] = mse(Yval, shape_pred)
    results["shape_mae"] = mae(Yval, shape_pred)
    print(
        f"Shape MSE={results['shape_mse']:.6f} "
        f"MAE={results['shape_mae']:.6f}"
    )

    # 读取已有 Expert-Attention，只做同 VAL 检查
    ea_file = ds / "expert_attention" / "val_predictions.npz"
    if ea_file.exists():
        ea = np.load(ea_file)
        if "y_true" in ea and "expert_attention_memory_pred" in ea:
            ea_y = ea["y_true"]
            ea_p = ea["expert_attention_memory_pred"]
            if ea_y.shape == Yval.shape and np.allclose(ea_y, Yval):
                results["expert_attention_mse"] = mse(Yval, ea_p)
                results["expert_attention_mae"] = mae(Yval, ea_p)
                print(
                    f"Expert-Attention MSE={results['expert_attention_mse']:.6f}"
                )
            else:
                print("[WARN] 已有 Expert-Attention VAL 与当前 VAL 不完全一致，不纳入表格。")

    # 2. Chronos
    chronos_pred = None
    if not args.skip_chronos:
        print("\n[2/2] Running Chronos...")
        try:
            chronos_pred = run_chronos(
                Xval,
                args.horizon,
                args.chronos_model,
                args.chronos_batch,
            )
            results["chronos_mse"] = mse(Yval, chronos_pred)
            results["chronos_mae"] = mae(Yval, chronos_pred)
            print(
                f"Chronos MSE={results['chronos_mse']:.6f} "
                f"MAE={results['chronos_mae']:.6f}"
            )
        except Exception as e:
            results["chronos_error"] = str(e)
            print("\n[Chronos 未完成]")
            print(e)

    np.savez_compressed(
        out / "val_predictions.npz",
        y_true=Yval,
        val_starts=val_starts,
        shape_pred=shape_pred,
        **({"chronos_pred": chronos_pred} if chronos_pred is not None else {}),
    )

    (out / "result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("RESULT")
    if "chronos_mse" in results:
        print(f"Chronos          : {results['chronos_mse']:.6f}")
    print(f"Shape Retrieval  : {results['shape_mse']:.6f}")
    if "expert_attention_mse" in results:
        print(f"Expert-Attention : {results['expert_attention_mse']:.6f}")
    print("=" * 70)
    print("Saved to:", out)
    print("TEST was NOT evaluated.")


if __name__ == "__main__":
    main()
