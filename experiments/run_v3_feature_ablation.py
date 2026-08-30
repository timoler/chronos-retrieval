#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V3 Expert Feature Ablation Batch Runner
=======================================

一次性跑 4 组专家特征消融：

1) hour only
   --expert-cols 0,1

2) doy only
   --expert-cols 2,3

3) level + change
   --expert-cols 4,5

4) hour + doy
   --expert-cols 0,1,2,3

默认直接调用：
    experiments/expert_main_shape_aux_v3.py

并自动汇总每组：
- Shape Memory MSE
- Expert-only Memory MSE
- Expert+Shape Memory MSE
- Expert gamma
- Hybrid gamma
- Shape beta
- Top-K Expert/Hybrid overlap
- Chronos
- Best Chronos+Shape
- Best Chronos+Expert
- Best Chronos+Expert+Shape
- 相对 Chronos 提升
- Hybrid vs Expert 提升

不会访问 TEST；是否访问 TEST 仍由 V3 自身的保护逻辑负责。

运行示例：
python experiments/run_v3_feature_ablation.py \
  --bundle results/expert_mlp_retrieval_v1_t1/solar_fair_bundle.npz \
  --v3-script experiments/expert_main_shape_aux_v3.py \
  --root-out results/v3_feature_ablation

如果想额外跑 all-6：
  --include-all

如果想额外跑 shuffled all-6：
  --include-shuffle
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


FEATURE_NAMES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "recent_level", "recent_change"]

ABLATIONS = [
    ("hour_only", "0,1", ["hour_sin", "hour_cos"]),
    ("doy_only", "2,3", ["doy_sin", "doy_cos"]),
    ("level_change", "4,5", ["recent_level", "recent_change"]),
    ("hour_doy", "0,1,2,3", ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]),
]


def run_one(
    python_exec: str,
    v3_script: Path,
    bundle: Path,
    out_dir: Path,
    expert_cols: str | None,
    epochs: int,
    hidden: int,
    batch_size: int,
    candidate_size: int,
    lr: float,
    topk: int,
    temperature: float,
    beta_init: float,
    seed: int,
    device: str,
    shuffle: bool = False,
):
    cmd = [
        python_exec,
        str(v3_script),
        "--bundle", str(bundle),
        "--out-dir", str(out_dir),
        "--epochs", str(epochs),
        "--hidden", str(hidden),
        "--batch-size", str(batch_size),
        "--candidate-size", str(candidate_size),
        "--lr", str(lr),
        "--topk", str(topk),
        "--softmax-temperature", str(temperature),
        "--beta-init", str(beta_init),
        "--seed", str(seed),
        "--device", str(device),
    ]

    if expert_cols is not None:
        cmd += ["--expert-cols", expert_cols]

    if shuffle:
        cmd += ["--shuffle-expert"]

    print()
    print("=" * 90)
    print("RUN:")
    print(" ".join(cmd))
    print("=" * 90)

    subprocess.run(cmd, check=True)

    report_path = out_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"未找到输出报告: {report_path}")

    return json.loads(report_path.read_text(encoding="utf-8"))


def safe_get(d, path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pct(old, new):
    if old in (None, 0) or new is None:
        return None
    return (old - new) / old * 100.0


def row_from_report(name, cols, feature_names, report):
    shape_mse = safe_get(report, ["memory_only", "shape", "mse"])
    expert_mse = safe_get(report, ["memory_only", "expert_only", "mse"])
    hybrid_mse = safe_get(report, ["memory_only", "expert_main_shape_aux", "mse"])

    expert_gamma = safe_get(report, ["memory_only", "expert_only", "gamma"])
    hybrid_gamma = safe_get(report, ["memory_only", "expert_main_shape_aux", "gamma"])
    beta = safe_get(report, ["memory_only", "expert_main_shape_aux", "beta"])

    overlap_eh = safe_get(report, ["topk_overlap", "expert_vs_hybrid"])

    chronos = safe_get(report, ["chronos_fusion", "chronos", "mse"])
    best_shape = safe_get(report, ["chronos_fusion", "shape", "best", "mse"])
    best_shape_lam = safe_get(report, ["chronos_fusion", "shape", "best", "lambda"])

    best_expert = safe_get(report, ["chronos_fusion", "expert_only", "best", "mse"])
    best_expert_lam = safe_get(report, ["chronos_fusion", "expert_only", "best", "lambda"])

    best_hybrid = safe_get(report, ["chronos_fusion", "expert_main_shape_aux", "best", "mse"])
    best_hybrid_lam = safe_get(report, ["chronos_fusion", "expert_main_shape_aux", "best", "lambda"])

    return {
        "name": name,
        "expert_cols": cols if cols is not None else "all",
        "features": ",".join(feature_names),
        "shuffle": bool(report.get("shuffle_expert", False)),

        "shape_memory_mse": shape_mse,
        "expert_memory_mse": expert_mse,
        "hybrid_memory_mse": hybrid_mse,
        "expert_vs_shape_memory_pct": pct(shape_mse, expert_mse),
        "hybrid_vs_expert_memory_pct": pct(expert_mse, hybrid_mse),

        "expert_gamma": expert_gamma,
        "hybrid_gamma": hybrid_gamma,
        "shape_beta": beta,
        "topk_expert_hybrid_overlap_pct": overlap_eh * 100.0 if overlap_eh is not None else None,

        "chronos_mse": chronos,
        "best_shape_fusion_mse": best_shape,
        "best_shape_lambda": best_shape_lam,
        "best_expert_fusion_mse": best_expert,
        "best_expert_lambda": best_expert_lam,
        "best_hybrid_fusion_mse": best_hybrid,
        "best_hybrid_lambda": best_hybrid_lam,

        "expert_vs_chronos_pct": pct(chronos, best_expert),
        "hybrid_vs_chronos_pct": pct(chronos, best_hybrid),
        "hybrid_vs_expert_fusion_pct": pct(best_expert, best_hybrid),
    }


def fmt(x, digits=2):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def write_summary(root_out: Path, rows: list[dict]):
    root_out.mkdir(parents=True, exist_ok=True)

    # JSON
    (root_out / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV
    fields = list(rows[0].keys())
    with (root_out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Markdown
    md = []
    md.append("# V3 Expert Feature Ablation Summary")
    md.append("")
    md.append("| Ablation | Expert Memory | Hybrid Memory | Best + Expert | Best + Hybrid | Hybrid vs Chronos | beta | E/H Top-K overlap |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        md.append(
            f"| {r['name']} "
            f"| {fmt(r['expert_memory_mse'])} "
            f"| {fmt(r['hybrid_memory_mse'])} "
            f"| {fmt(r['best_expert_fusion_mse'])} "
            f"| {fmt(r['best_hybrid_fusion_mse'])} "
            f"| {fmt(r['hybrid_vs_chronos_pct'])}% "
            f"| {fmt(r['shape_beta'], 4)} "
            f"| {fmt(r['topk_expert_hybrid_overlap_pct'])}% |"
        )

    md.append("")
    md.append("## 诊断")
    md.append("")
    md.append("| Ablation | Expert vs Shape (Memory) | Hybrid vs Expert (Memory) | Hybrid vs Expert (Fusion) | Expert λ | Hybrid λ |")
    md.append("|---|---:|---:|---:|---:|---:|")

    for r in rows:
        md.append(
            f"| {r['name']} "
            f"| {fmt(r['expert_vs_shape_memory_pct'])}% "
            f"| {fmt(r['hybrid_vs_expert_memory_pct'])}% "
            f"| {fmt(r['hybrid_vs_expert_fusion_pct'])}% "
            f"| {fmt(r['best_expert_lambda'], 2)} "
            f"| {fmt(r['best_hybrid_lambda'], 2)} |"
        )

    (root_out / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument(
        "--v3-script",
        type=Path,
        default=Path("experiments/expert_main_shape_aux_v3.py"),
    )
    p.add_argument(
        "--root-out",
        type=Path,
        default=Path("results/v3_feature_ablation"),
    )

    p.add_argument("--python", type=str, default=sys.executable)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--candidate-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--softmax-temperature", type=float, default=1.0)
    p.add_argument("--beta-init", type=float, default=0.2)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--include-all", action="store_true")
    p.add_argument("--include-shuffle", action="store_true")

    args = p.parse_args()

    if "test" in str(args.bundle).lower():
        raise RuntimeError("拒绝运行：当前批处理实验禁止使用 TEST bundle。")

    if not args.v3_script.exists():
        raise FileNotFoundError(f"找不到 V3 脚本: {args.v3_script}")

    jobs = list(ABLATIONS)

    if args.include_all:
        jobs.append(
            ("all6", "0,1,2,3,4,5", FEATURE_NAMES)
        )

    rows = []

    for name, cols, names in jobs:
        out_dir = args.root_out / name

        report = run_one(
            python_exec=args.python,
            v3_script=args.v3_script,
            bundle=args.bundle,
            out_dir=out_dir,
            expert_cols=cols,
            epochs=args.epochs,
            hidden=args.hidden,
            batch_size=args.batch_size,
            candidate_size=args.candidate_size,
            lr=args.lr,
            topk=args.topk,
            temperature=args.softmax_temperature,
            beta_init=args.beta_init,
            seed=args.seed,
            device=args.device,
            shuffle=False,
        )

        rows.append(
            row_from_report(name, cols, names, report)
        )

    if args.include_shuffle:
        name = "all6_shuffle"
        out_dir = args.root_out / name

        report = run_one(
            python_exec=args.python,
            v3_script=args.v3_script,
            bundle=args.bundle,
            out_dir=out_dir,
            expert_cols="0,1,2,3,4,5",
            epochs=args.epochs,
            hidden=args.hidden,
            batch_size=args.batch_size,
            candidate_size=args.candidate_size,
            lr=args.lr,
            topk=args.topk,
            temperature=args.softmax_temperature,
            beta_init=args.beta_init,
            seed=args.seed,
            device=args.device,
            shuffle=True,
        )

        rows.append(
            row_from_report(
                name,
                "0,1,2,3,4,5",
                FEATURE_NAMES,
                report,
            )
        )

    write_summary(args.root_out, rows)

    print()
    print("=" * 100)
    print("V3 FEATURE ABLATION SUMMARY")
    print("=" * 100)

    header = (
        f"{'name':<18}"
        f"{'ExpertMem':>14}"
        f"{'HybridMem':>14}"
        f"{'BestExpert':>14}"
        f"{'BestHybrid':>14}"
        f"{'vsChronos':>12}"
        f"{'beta':>10}"
    )
    print(header)
    print("-" * 100)

    for r in rows:
        print(
            f"{r['name']:<18}"
            f"{fmt(r['expert_memory_mse']):>14}"
            f"{fmt(r['hybrid_memory_mse']):>14}"
            f"{fmt(r['best_expert_fusion_mse']):>14}"
            f"{fmt(r['best_hybrid_fusion_mse']):>14}"
            f"{(fmt(r['hybrid_vs_chronos_pct']) + '%'):>12}"
            f"{fmt(r['shape_beta'],4):>10}"
        )

    print("-" * 100)
    print("summary:", args.root_out / "summary.md")
    print("=" * 100)


if __name__ == "__main__":
    main()
