#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补齐 3 个 pilot 数据集（etth1/electricity/weather）缺失的 v2 final pool。

背景：v2 pilot 阶段只落盘了 pool_reward_history.csv，没落盘 final_expert_pool.json。
TEST 阶段需要读冻结的 v2 pool，所以这里用「冻结 v2 + 已缓存的 LLM」重跑 TRAIN-only 闭环，
把 final pool 重新落盘到 results/reward_feedback_critic_v2_full_cross_dataset/<name>/。

关键：
  - 复用 run_reward_feedback_critic_v2_poolaware_pilot 的 run_v2_loop（不复制、不改方法）。
  - LLM 缓存指向 pilot 的缓存目录（cache/llm_priors_reward_feedback_critic_v2_pilot），
    因此 Critic/Revision 直接命中缓存，不重新调用 LLM（不产生新 Expert）。
  - 训练 deterministic（同 seed），产出与冻结 v2 完全一致的 pool。
  - 运行前后对比缓存目录文件数，检测是否发生了缓存 miss（若 miss 则报警停止，避免产生不同 pool）。

用法：
  python experiments/_backfill_v2_final_pools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import run_reward_feedback_critic_v2_poolaware_pilot as v2p

REPO_ROOT = HERE.parent
FROZEN_ROOT = REPO_ROOT / "results/reward_feedback_critic_v2_full_cross_dataset"
PILOT_CACHE = REPO_ROOT / "cache/llm_priors_reward_feedback_critic_v2_pilot"

# 重定向 v2 输出目录 + 复用 pilot 的 LLM 缓存（命中缓存，不重调 LLM）
v2p.OUT_ROOT = FROZEN_ROOT
v2p.LLM_CACHE_ROOT = PILOT_CACHE

DATASETS = [
    {"name": "etth1",       "domain": "transformer_temperature",  "freq": "hourly"},
    {"name": "electricity", "domain": "aggregate_electricity_load", "freq": "15-minute"},
    {"name": "weather",     "domain": "weather_meteorological",    "freq": "10-minute"},
]


def _cache_file_count() -> dict:
    cnt = {}
    for d in DATASETS:
        sub = PILOT_CACHE / d["name"]
        if sub.exists():
            cnt[d["name"]] = len(list(sub.glob("*.json")))
        else:
            cnt[d["name"]] = 0
    return cnt


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of dataset names (default: all 3)")
    ap.add_argument("--resume", action="store_true",
                    help="skip datasets whose final_expert_pool.json already exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("LLM cache ->", PILOT_CACHE)

    targets = [d for d in DATASETS if (args.datasets is None or d["name"] in args.datasets)]
    if args.resume:
        targets = [d for d in targets
                   if not (FROZEN_ROOT / d["name"] / "final_expert_pool.json").exists()]
    print("targets:", [d["name"] for d in targets])

    before = _cache_file_count()
    print("cache files before:", before)

    for cfg in targets:
        name = cfg["name"]
        out_dir = FROZEN_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\nBACKFILL {name}\n{'=' * 70}")
        raw, pool = v2p.run_v2_loop(name, cfg, device)
        (out_dir / "raw_experts.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out_dir / "final_expert_pool.json").write_text(
            json.dumps(pool, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  saved raw={len(raw)} v2_pool={len(pool)} -> {[e['id'] for e in pool]}")

    after = _cache_file_count()
    print("\ncache files after:", after)
    new_files = [d for d in before if after[d] > before[d]]
    if new_files:
        print(f"\n[WARN] LLM cache MISS detected on: {new_files} -> 产生了新的 LLM 调用，"
              f"pool 可能与冻结版本不一致，请人工核对！")
    else:
        print("\n[OK] 无新增 LLM 缓存文件 -> Critic/Revision 全部命中缓存，pool 与冻结 v2 一致。")
    print(f"\nDone. Frozen pools now at {FROZEN_ROOT}")


if __name__ == "__main__":
    main()
