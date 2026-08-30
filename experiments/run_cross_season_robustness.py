#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-Season Robustness Runner (TRAIN_B length ablation)
========================================================

目的
----
在不触碰 TEST 的前提下，对 spring / summer 两个 cross-season 伪验证块，
把 TRAIN_B 从原来的 14 天扩展到 30 天，检查：

1) hour / hour+doy / hour+level_change / all6 的结论是否稳定；
2) summer 中 doy 的增益是否仍存在；
3) spring 中 all6 的异常是否因 TRAIN_B 太短而减弱；
4) 比较 14d vs 30d 的 MLP 稳定性。

本脚本假定你已经有：
    experiments/run_cross_season_v3.py

并且其中两个 adapter 已经接好冻结实现：
- full TRAIN source adapter
- frozen residual-Pearson shape score adapter

本脚本不重新实现 builder，只调用 run_cross_season_v3.py 两次：
- train_b_days=14
- train_b_days=30

最后自动汇总：
    results/cross_season_robustness/
        run_14d/
        run_30d/
        robustness_summary.csv
        robustness_summary.json
        robustness_summary.md

不会读取 TEST。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


TRAIN_B_DAYS = [14, 30]

BLOCKS = ["spring", "summer"]
CONFIGS = ["hour", "hour_doy", "hour_level_change", "all6"]


def run_cross_season(
    python_exec: str,
    driver_script: Path,
    source: Path,
    v3_script: Path,
    out_root: Path,
    train_b_days: int,
    train_b_stride: int,
    epochs: int,
    temperature: float,
    topk: int,
    seed: int,
    device: str,
):
    cmd = [
        python_exec,
        str(driver_script),
        "--source", str(source),
        "--v3-script", str(v3_script),
        "--out-root", str(out_root),
        "--train-b-days", str(train_b_days),
        "--train-b-stride", str(train_b_stride),
        "--epochs", str(epochs),
        "--softmax-temperature", str(temperature),
        "--topk", str(topk),
        "--seed", str(seed),
        "--device", str(device),
    ]

    print("\n" + "=" * 100)
    print(f"RUN TRAIN_B={train_b_days} days")
    print(" ".join(cmd))
    print("=" * 100)

    subprocess.run(cmd, check=True)


def load_summary(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到 summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def key_rows(rows):
    return {(r["block"], r["config"]): r for r in rows}


def pct_change(old, new):
    if old in (None, 0) or new is None:
        return None
    return (old - new) / old * 100.0


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_robustness_rows(summary_14, summary_30):
    d14 = key_rows(summary_14)
    d30 = key_rows(summary_30)

    out = []

    for block in BLOCKS:
        for config in CONFIGS:
            r14 = d14.get((block, config))
            r30 = d30.get((block, config))

            if r14 is None or r30 is None:
                raise KeyError(f"缺少结果: {block}/{config}")

            h14 = safe_float(r14.get("hybrid_memory_mse"))
            h30 = safe_float(r30.get("hybrid_memory_mse"))

            e14 = safe_float(r14.get("expert_memory_mse"))
            e30 = safe_float(r30.get("expert_memory_mse"))

            beta14 = safe_float(r14.get("shape_beta"))
            beta30 = safe_float(r30.get("shape_beta"))

            gamma14 = safe_float(r14.get("expert_gamma"))
            gamma30 = safe_float(r30.get("expert_gamma"))

            out.append({
                "block": block,
                "config": config,

                "expert_mse_14d": e14,
                "expert_mse_30d": e30,
                "expert_30d_vs_14d_pct": pct_change(e14, e30),

                "hybrid_mse_14d": h14,
                "hybrid_mse_30d": h30,
                "hybrid_30d_vs_14d_pct": pct_change(h14, h30),

                "beta_14d": beta14,
                "beta_30d": beta30,
                "gamma_14d": gamma14,
                "gamma_30d": gamma30,
            })

    return out


def find_row(rows, block, config):
    for r in rows:
        if r["block"] == block and r["config"] == config:
            return r
    raise KeyError((block, config))


def compute_mechanism_diagnostics(summary_rows):
    """
    对每个 block 计算：
    - hour -> hour_doy 的 doy 增量
    - hour -> hour_level_change 的 state 增量
    - hour -> all6 的联合增量
    - hour_level_change -> all6 的“加 doy”增量
    - hour_doy -> all6 的“加 state”增量
    """
    diagnostics = []

    for block in BLOCKS:
        hour = safe_float(find_row(summary_rows, block, "hour")["hybrid_memory_mse"])
        hd = safe_float(find_row(summary_rows, block, "hour_doy")["hybrid_memory_mse"])
        hs = safe_float(find_row(summary_rows, block, "hour_level_change")["hybrid_memory_mse"])
        all6 = safe_float(find_row(summary_rows, block, "all6")["hybrid_memory_mse"])

        diagnostics.append({
            "block": block,
            "doy_increment_from_hour_pct": pct_change(hour, hd),
            "state_increment_from_hour_pct": pct_change(hour, hs),
            "all6_increment_from_hour_pct": pct_change(hour, all6),
            "doy_increment_given_state_pct": pct_change(hs, all6),
            "state_increment_given_doy_pct": pct_change(hd, all6),
        })

    return diagnostics


def fmt(x, digits=2):
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def save_outputs(out_root: Path, robustness_rows, diag14, diag30):
    out_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "robustness_rows": robustness_rows,
        "diagnostics_14d": diag14,
        "diagnostics_30d": diag30,
    }

    (out_root / "robustness_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out_root / "robustness_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(robustness_rows[0].keys()))
        w.writeheader()
        w.writerows(robustness_rows)

    lines = []
    lines.append("# Cross-Season Robustness: TRAIN_B 14d vs 30d")
    lines.append("")
    lines.append("## 结果对比")
    lines.append("")
    lines.append(
        "| Block | Config | Hybrid 14d | Hybrid 30d | 30d vs 14d | Expert 14d | Expert 30d | beta 14d | beta 30d |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in robustness_rows:
        lines.append(
            f"| {r['block']} | {r['config']} "
            f"| {fmt(r['hybrid_mse_14d'])} "
            f"| {fmt(r['hybrid_mse_30d'])} "
            f"| {fmt(r['hybrid_30d_vs_14d_pct'])}% "
            f"| {fmt(r['expert_mse_14d'])} "
            f"| {fmt(r['expert_mse_30d'])} "
            f"| {fmt(r['beta_14d'],4)} "
            f"| {fmt(r['beta_30d'],4)} |"
        )

    lines.append("")
    lines.append("## 机制增量：14d")
    lines.append("")
    lines.append(
        "| Block | +doy from hour | +state from hour | all6 from hour | +doy given state | +state given doy |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    for r in diag14:
        lines.append(
            f"| {r['block']} "
            f"| {fmt(r['doy_increment_from_hour_pct'])}% "
            f"| {fmt(r['state_increment_from_hour_pct'])}% "
            f"| {fmt(r['all6_increment_from_hour_pct'])}% "
            f"| {fmt(r['doy_increment_given_state_pct'])}% "
            f"| {fmt(r['state_increment_given_doy_pct'])}% |"
        )

    lines.append("")
    lines.append("## 机制增量：30d")
    lines.append("")
    lines.append(
        "| Block | +doy from hour | +state from hour | all6 from hour | +doy given state | +state given doy |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")

    for r in diag30:
        lines.append(
            f"| {r['block']} "
            f"| {fmt(r['doy_increment_from_hour_pct'])}% "
            f"| {fmt(r['state_increment_from_hour_pct'])}% "
            f"| {fmt(r['all6_increment_from_hour_pct'])}% "
            f"| {fmt(r['doy_increment_given_state_pct'])}% "
            f"| {fmt(r['state_increment_given_doy_pct'])}% |"
        )

    lines.append("")
    lines.append("## 判读规则")
    lines.append("")
    lines.append(
        "- 若 summer 的 `hour -> hour_doy` 在 14d 和 30d 都明显为正，说明 doy 的跨季节增益较稳。"
    )
    lines.append(
        "- 若 spring 的 all6 在 30d 明显恢复，说明 14d 的异常主要与 TRAIN_B 太短/训练不稳定有关。"
    )
    lines.append(
        "- 若 spring 在 30d 仍然表现为 doy 或 all6 有害，则更支持“目标季节覆盖不足时，季节机制会失效或误导”。"
    )
    lines.append(
        "- beta/gamma 只做稳定性辅助观察，不把数值直接解释成机制重要性百分比。"
    )

    (out_root / "robustness_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="只包含原 TRAIN 的 source；禁止 TEST。",
    )
    ap.add_argument(
        "--driver-script",
        type=Path,
        default=Path("experiments/run_cross_season_v3.py"),
    )
    ap.add_argument(
        "--v3-script",
        type=Path,
        default=Path("experiments/expert_main_shape_aux_v3.py"),
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/cross_season_robustness"),
    )

    ap.add_argument("--train-b-stride", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--softmax-temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--python", type=str, default=sys.executable)

    ap.add_argument(
        "--reuse-14d",
        type=Path,
        default=None,
        help=(
            "可选：若已有 14d 结果，可直接指向现有 cross_season_v3 根目录，"
            "脚本只重跑 30d。"
        ),
    )

    args = ap.parse_args()

    if "test" in str(args.source).lower():
        raise RuntimeError("拒绝运行：source 路径包含 test。")

    if not args.driver_script.exists():
        raise FileNotFoundError(args.driver_script)

    if not args.v3_script.exists():
        raise FileNotFoundError(args.v3_script)

    # ---------- 14d ----------
    if args.reuse_14d is not None:
        run14_root = args.reuse_14d
        print(f"\nReuse existing 14d results: {run14_root}")
    else:
        run14_root = args.out_root / "run_14d"
        run_cross_season(
            python_exec=args.python,
            driver_script=args.driver_script,
            source=args.source,
            v3_script=args.v3_script,
            out_root=run14_root,
            train_b_days=14,
            train_b_stride=args.train_b_stride,
            epochs=args.epochs,
            temperature=args.softmax_temperature,
            topk=args.topk,
            seed=args.seed,
            device=args.device,
        )

    # ---------- 30d ----------
    run30_root = args.out_root / "run_30d"
    run_cross_season(
        python_exec=args.python,
        driver_script=args.driver_script,
        source=args.source,
        v3_script=args.v3_script,
        out_root=run30_root,
        train_b_days=30,
        train_b_stride=args.train_b_stride,
        epochs=args.epochs,
        temperature=args.softmax_temperature,
        topk=args.topk,
        seed=args.seed,
        device=args.device,
    )

    summary14 = load_summary(run14_root / "cross_season_summary.json")
    summary30 = load_summary(run30_root / "cross_season_summary.json")

    robustness_rows = build_robustness_rows(summary14, summary30)
    diag14 = compute_mechanism_diagnostics(summary14)
    diag30 = compute_mechanism_diagnostics(summary30)

    save_outputs(
        args.out_root,
        robustness_rows,
        diag14,
        diag30,
    )

    print("\n" + "=" * 100)
    print("ROBUSTNESS DONE")
    print("14d:", run14_root)
    print("30d:", run30_root)
    print("summary:", args.out_root / "robustness_summary.md")
    print("=" * 100)


if __name__ == "__main__":
    main()
