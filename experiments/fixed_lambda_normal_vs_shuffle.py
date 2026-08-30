#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed-Lambda Normal-vs-Shuffle Expert Comparison
=================================================

目的：
严格控制 lambda，逐个比较：

    Normal Expert Memory
    vs
    Shuffled Expert Memory

在完全相同的 lambda 下与 Chronos 融合后的 MSE/MAE。

这不是“各自找最优 lambda”的实验。
核心问题只有一个：

    当 lambda 固定时，只把 Expert 对应关系打乱，
    最终预测误差是否持续恶化？

需要两个已经跑好的 V3 输出目录：
1) normal_dir  : 正常 all6 Expert
2) shuffle_dir : all6_shuffle Expert

两个目录都应包含：
    val_predictions.npz

脚本会自动寻找常见字段名；若你的 V3 字段名不同，
Claude 只需修改 load_predictions() 中的字段映射即可。

输出：
    fixed_lambda_comparison.csv
    fixed_lambda_comparison.json
    fixed_lambda_comparison.md
    fixed_lambda_curve.png

注意：
- 不重新训练
- 不重新检索
- 不碰 TEST
- 不为 Normal / Shuffle 分别挑最优 lambda
- 同一个 lambda 一一对应比较
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_LAMBDAS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]

# V3 的 val_predictions.npz 不存 chronos；chronos 在 bundle 里，且 normal/shuffle 共用同一个。
CHRONOS_NPZ = Path("results/expert_mlp_retrieval_v1/solar_fair_bundle.npz")


def first_key(d, names):
    for name in names:
        if name in d:
            return name
    return None


def load_predictions(path: Path):
    if "test" in str(path).lower():
        raise RuntimeError("拒绝运行：输入路径包含 test。")

    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}

    y_key = first_key(d, [
        "y_true", "val_y_true", "val_future", "target", "targets"
    ])

    chronos_key = first_key(d, [
        "chronos_pred", "chronos_val_pred", "val_chronos_pred"
    ])

    # V3 中我们要比较的是 Expert-main + Shape-aux 的最终 Memory
    hybrid_key = first_key(d, [
        "hybrid_memory_pred",
        "expert_main_shape_aux_pred",
        "expert_main_shape_aux_memory_pred",
        "hybrid_pred",
        "memory_hybrid_pred",
    ])

    # 如果想顺便核对 Expert-only，也读出来，但不是必须
    expert_key = first_key(d, [
        "expert_memory_pred",
        "expert_only_pred",
        "expert_only_memory_pred",
    ])

    # chronos 不在 val_predictions.npz 里（V3 只存 y/memory），从共享 bundle 读。
    chronos_arr = None
    if chronos_key is not None:
        chronos_arr = np.asarray(d[chronos_key], dtype=np.float64)
    elif CHRONOS_NPZ.exists():
        with np.load(CHRONOS_NPZ, allow_pickle=False) as g:
            chronos_arr = np.asarray(g["chronos_val_pred"], dtype=np.float64)

    missing = []
    if y_key is None:
        missing.append("y_true")
    if chronos_arr is None:
        missing.append("chronos_pred (bundle 中也找不到)")
    if hybrid_key is None:
        missing.append("hybrid/expert_main_shape_aux memory pred")

    if missing:
        raise KeyError(
            f"{path} 缺少字段: {missing}\n"
            f"现有字段: {list(d.keys())}\n"
            "请让 Claude 只修改 load_predictions() 的字段映射，"
            "不要改实验逻辑。"
        )

    out = {
        "y_true": np.asarray(d[y_key], dtype=np.float64),
        "chronos": chronos_arr,
        "hybrid": np.asarray(d[hybrid_key], dtype=np.float64),
        "expert": None if expert_key is None else np.asarray(d[expert_key], dtype=np.float64),
    }

    return out


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def mae(y, pred):
    return float(np.mean(np.abs(y - pred)))


def fuse(chronos, memory, lam):
    return (1.0 - lam) * chronos + lam * memory


def parse_lambdas(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("lambda 列表不能为空")
    for x in vals:
        if x < 0 or x > 1:
            raise ValueError(f"lambda 必须在 [0,1]，实际 {x}")
    return vals


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--normal-dir",
        type=Path,
        required=True,
        help="正常 all6 V3 输出目录",
    )
    ap.add_argument(
        "--shuffle-dir",
        type=Path,
        required=True,
        help="all6_shuffle V3 输出目录",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/fixed_lambda_normal_vs_shuffle"),
    )
    ap.add_argument(
        "--lambdas",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAMBDAS),
    )

    args = ap.parse_args()

    normal_file = args.normal_dir / "val_predictions.npz"
    shuffle_file = args.shuffle_dir / "val_predictions.npz"

    if not normal_file.exists():
        raise FileNotFoundError(normal_file)
    if not shuffle_file.exists():
        raise FileNotFoundError(shuffle_file)

    normal = load_predictions(normal_file)
    shuffle = load_predictions(shuffle_file)

    # ---------------------------
    # 公平性检查
    # ---------------------------
    if normal["y_true"].shape != shuffle["y_true"].shape:
        raise ValueError("Normal / Shuffle y_true shape 不一致")

    if normal["chronos"].shape != shuffle["chronos"].shape:
        raise ValueError("Normal / Shuffle Chronos shape 不一致")

    if not np.allclose(normal["y_true"], shuffle["y_true"], rtol=0, atol=1e-8):
        raise ValueError("Normal / Shuffle 的 y_true 不完全一致，不能公平比较")

    if not np.allclose(normal["chronos"], shuffle["chronos"], rtol=0, atol=1e-8):
        raise ValueError("Normal / Shuffle 的 Chronos prediction 不完全一致")

    y = normal["y_true"]
    chronos = normal["chronos"]

    lambdas = parse_lambdas(args.lambdas)

    chronos_mse = mse(y, chronos)
    chronos_mae = mae(y, chronos)

    rows = []

    for lam in lambdas:
        p_normal = fuse(chronos, normal["hybrid"], lam)
        p_shuffle = fuse(chronos, shuffle["hybrid"], lam)

        n_mse = mse(y, p_normal)
        s_mse = mse(y, p_shuffle)

        n_mae = mae(y, p_normal)
        s_mae = mae(y, p_shuffle)

        # 正值 = Shuffle 比 Normal 更差
        mse_gap = s_mse - n_mse
        mae_gap = s_mae - n_mae

        gap_pct = None if n_mse == 0 else mse_gap / n_mse * 100.0

        rows.append({
            "lambda": lam,
            "normal_mse": n_mse,
            "shuffle_mse": s_mse,
            "shuffle_minus_normal_mse": mse_gap,
            "shuffle_worse_pct": gap_pct,
            "normal_mae": n_mae,
            "shuffle_mae": s_mae,
            "shuffle_minus_normal_mae": mae_gap,
            "normal_vs_chronos_pct": (chronos_mse - n_mse) / chronos_mse * 100.0,
            "shuffle_vs_chronos_pct": (chronos_mse - s_mse) / chronos_mse * 100.0,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = args.out_dir / "fixed_lambda_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # JSON
    payload = {
        "experiment": "fixed_lambda_normal_vs_shuffle",
        "chronos_mse": chronos_mse,
        "chronos_mae": chronos_mae,
        "normal_memory_mse": mse(y, normal["hybrid"]),
        "shuffle_memory_mse": mse(y, shuffle["hybrid"]),
        "rows": rows,
        "interpretation": (
            "Normal and Shuffle use exactly the same lambda at every comparison. "
            "The only intended difference is whether expert information is correctly aligned."
        ),
    }

    json_path = args.out_dir / "fixed_lambda_comparison.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown
    md = []
    md.append("# Fixed-λ: Normal Expert vs Shuffled Expert")
    md.append("")
    md.append(f"Chronos baseline MSE = **{chronos_mse:.2f}**")
    md.append("")
    md.append(
        "本实验不分别寻找最优 λ。Normal 与 Shuffle 在每一行使用完全相同的 λ。"
    )
    md.append("")
    md.append("| λ | Normal MSE | Shuffle MSE | Shuffle-Normal | Shuffle 更差 | Normal vs Chronos | Shuffle vs Chronos |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        md.append(
            f"| {r['lambda']:.3f} "
            f"| {r['normal_mse']:.2f} "
            f"| {r['shuffle_mse']:.2f} "
            f"| {r['shuffle_minus_normal_mse']:+.2f} "
            f"| {r['shuffle_worse_pct']:+.2f}% "
            f"| {r['normal_vs_chronos_pct']:+.2f}% "
            f"| {r['shuffle_vs_chronos_pct']:+.2f}% |"
        )

    md.append("")
    md.append("## 判读")
    md.append("")
    md.append(
        "- λ=0 时两者必须完全相同，因为此时 Memory 权重为 0。"
    )
    md.append(
        "- λ>0 后，如果 Normal MSE 持续低于 Shuffle MSE，说明在相同融合强度下，正确 Expert 信息比打乱 Expert 信息更有效。"
    )
    md.append(
        "- 这个实验用于验证 Expert 信息本身的贡献，不把“各自最优 λ”作为主要结论。"
    )

    md_path = args.out_dir / "fixed_lambda_comparison.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    # Plot
    try:
        import matplotlib.pyplot as plt

        xs = [r["lambda"] for r in rows]
        yn = [r["normal_mse"] for r in rows]
        ys = [r["shuffle_mse"] for r in rows]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, yn, marker="o", label="Normal Expert")
        plt.plot(xs, ys, marker="o", label="Shuffled Expert")
        plt.axhline(chronos_mse, linestyle="--", label="Chronos")
        plt.xlabel("Lambda")
        plt.ylabel("MSE")
        plt.title("Fixed Lambda: Normal vs Shuffled Expert")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_dir / "fixed_lambda_curve.png", dpi=180)
        plt.close()
    except Exception as e:
        print(f"[WARN] 曲线图生成失败，但表格结果不受影响: {e}")

    print("\n" + "=" * 100)
    print("FIXED-LAMBDA COMPARISON")
    print("=" * 100)
    print(f"Chronos MSE: {chronos_mse:.2f}")
    print()
    print(f"{'lambda':>8} {'Normal':>14} {'Shuffle':>14} {'S-N':>14} {'worse%':>10}")
    print("-" * 65)

    for r in rows:
        print(
            f"{r['lambda']:>8.3f} "
            f"{r['normal_mse']:>14.2f} "
            f"{r['shuffle_mse']:>14.2f} "
            f"{r['shuffle_minus_normal_mse']:>+14.2f} "
            f"{r['shuffle_worse_pct']:>+9.2f}%"
        )

    print("-" * 65)
    print("summary:", md_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
