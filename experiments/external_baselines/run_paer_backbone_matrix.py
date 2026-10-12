#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PAER-Aligned 插件 × 底座矩阵批量脚本（固定插件版本，加装前后对照）。

插件：PAER_original_training_mean_aligned_v1（原冻结专家池 + 原 softmax-256 训练 + 原 Top-5 评分
+ 推理均值对齐）。同一数据集、同一 seed 的记忆预测供四个底座共同复用。

底座：PatchTST / iTransformer / RAFT / Chronos-2；数据集 7 个；seed [42,123,2026]。

用法：
  --inventory-only                      仅盘点缓存，写 inventory.csv
  --dataset X [--backbone Y [--seed S]]  生成缺失记忆 / 计算融合
  --resume                              跳过已完成组合
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import run_ours_fusion_v2 as v2  # noqa: E402
from run_conditional_expert_mlp_v4 import ConditionalExpertMLPRetriever  # noqa: E402
from etth2_agg_alignment import retrieve_dual  # noqa: E402

REPO = v2.REPO_ROOT
OUT = REPO / "results/paer_aligned_backbone_matrix"
PRED = OUT / "predictions"
(OUT / "logs").mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)

VERSION = "PAER_original_training_mean_aligned_v1"
DATASETS = ["etth1", "etth2", "exchange_rate", "traffic", "weather", "solar", "electricity"]
BACKBONES = ["PatchTST", "iTransformer", "RAFT", "Chronos-2"]
SEEDS = [42, 123, 2026]
LAMBDA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

REAUDIT = REPO / "results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011/preds"
CODEX = Path("C:/Users/陈佳政/Documents/Codex/2026-10-11/https-github-com-timoler-chronos-retrieval")
EXT = Path("E:/GitHub_Project/_external_repos")
POOL_ROOT = REPO / "results/reward_feedback_critic_v2_full_cross_dataset"


def mse(y, p):
    return float(((y.astype(np.float64) - p.astype(np.float64)) ** 2).mean())


def sha256(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return "MISSING"


# --------------------------------------------------------------------------- #
# 记忆来源（对齐版，base-agnostic）
# --------------------------------------------------------------------------- #
def memory_paths(ds, seed):
    """返回 (val, test) 对齐记忆路径；不存在则 None。"""
    if ds == "etth2":
        v = REPO / f"results/etth2_agg_alignment_phase2/seed{seed}.npz"
        t = REPO / f"results/etth2_agg_alignment_phase3/seed{seed}_test.npz"
        return (v, "align_val"), (t, "align_test")
    if ds == "traffic":
        return ((REPO / f"results/patchtst_paer_fusion_traffic/paer_align_val_s{seed}.npy"), None), \
               ((REPO / f"results/patchtst_paer_fusion_traffic/paer_align_test_s{seed}.npy"), None)
    if ds in ("weather", "etth1"):
        d = CODEX / "work" / ("pilot_artifacts" if seed == 42 else f"replicate_artifacts/seed{seed}") / ds
        return ((d / "PAER_val_retrieval.npz"), "transferred"), ((d / "PAER_test_retrieval.npz"), "transferred")
    # exchange_rate / solar / electricity：本地生成
    return ((OUT / f"mem/{ds}/paer_align_val_s{seed}.npy"), None), \
           ((OUT / f"mem/{ds}/paer_align_test_s{seed}.npy"), None)


def load_array(p, key):
    if p is None or not Path(p).exists():
        return None
    a = np.load(p)
    return a[key].astype(np.float64) if key else a.astype(np.float64)


def generate_memory(ds):
    """按冻结协议重建 PAER-Aligned 记忆（VAL+TEST，3 seed），不调参。"""
    ds_dir = REPO / "results/multidataset" / ds
    cfg = json.loads((ds_dir / "config.json").read_text(encoding="utf-8"))
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    v2.v1.CONTEXT, v2.v1.HORIZON = context, horizon
    df = pd.read_csv(ds_dir / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds_dir / "window_indices.npz")
    train = np.asarray(idx["train"], np.int64); val = np.asarray(idx["val"], np.int64); test = np.asarray(idx["test"], np.int64)
    cut = max(1, int(len(train) * 0.80))
    bank_s, fit_s = train[:cut], train[cut:]
    X_bank, Y_bank = v2.build_windows(series, bank_s, context, horizon)
    X_fit, Y_fit = v2.build_windows(series, fit_s, context, horizon)
    X_val, Y_val = v2.build_windows(series, val, context, horizon)
    X_test, Y_test = v2.build_windows(series, test, context, horizon)
    X_bank_res, X_fit_res = v2.residualize(X_bank), v2.residualize(X_fit)
    X_val_res, X_test_res = v2.residualize(X_val), v2.residualize(X_test)
    bank_ts = [dates[s + context] for s in bank_s]; fit_ts = [dates[s + context] for s in fit_s]
    val_ts = [dates[s + context] for s in val]; test_ts = [dates[s + context] for s in test]
    stats = v2.compute_training_stats(np.concatenate([X_bank, X_fit], axis=0))
    pool = v2.load_v2_pool(ds)
    n_slots = max(len(v2.load_raw_pool(ds)), len(v2.load_v1_pool(ds)), len(pool))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb_orig = v2.v1.embed_claims([e["claim"] for e in pool], device)
    expert_dim = emb_orig.shape[1]
    act_val = v2.v1.act_mats(pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    act_test = v2.v1.act_mats(pool, X_bank, X_fit, X_test, bank_ts, fit_ts, test_ts, stats)
    a_val = {"bank": act_val["bank"], "fit": act_val["fit"], "val": act_val["val"]}
    a_test = {"bank": act_test["bank"], "fit": act_test["fit"], "test": act_test["val"]}
    emb, a_val, _ = v2.v1.pad_to_slots(emb_orig, a_val, n_slots)
    _, a_test, _ = v2.v1.pad_to_slots(emb_orig, a_test, n_slots)
    (OUT / f"mem/{ds}").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        init = v2._init_state(seed, context, expert_dim, n_slots, device)
        mdl = ConditionalExpertMLPRetriever(context=context, expert_dim=expert_dim, n_experts=n_slots,
                                            d_enc=v2.D_ENC, d_exp=v2.D_EXP, d_z=v2.D_Z, hidden=v2.HIDDEN).to(device)
        mdl.load_state_dict(init); mdl.set_expert_embeddings(emb)
        v2.v2p.train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                         a_val["bank"], a_val["fit"], True, device, seed)
        _, alv, _ = retrieve_dual(mdl, X_val, X_bank, Y_bank, X_val_res, X_bank_res, a_val["val"], a_val["bank"], v2.TOPK, device)
        _, alt, _ = retrieve_dual(mdl, X_test, X_bank, Y_bank, X_test_res, X_bank_res, a_test["test"], a_test["bank"], v2.TOPK, device)
        np.save(OUT / f"mem/{ds}/paer_align_val_s{seed}.npy", alv.astype(np.float32))
        np.save(OUT / f"mem/{ds}/paer_align_test_s{seed}.npy", alt.astype(np.float32))
        np.save(OUT / f"mem/{ds}/Y_val_s{seed}.npy", Y_val.astype(np.float32))
        np.save(OUT / f"mem/{ds}/Y_test_s{seed}.npy", Y_test.astype(np.float32))
        print(f"  [{ds}] seed{seed}: memory generated val={alv.shape} test={alt.shape}", flush=True)


# --------------------------------------------------------------------------- #
# 底座预测来源
# --------------------------------------------------------------------------- #
def backbone_paths(ds, backbone, seed):
    """返回 (val, test) 底座预测路径；不存在/不完整则 None。"""
    if backbone == "Chronos-2":
        p = REAUDIT / f"{ds}.npz"
        return (p, "yc_val"), (p, "yc_test")
    if backbone == "PatchTST":
        # 完整窗口预测（infer_patchtst_matrix.py 产出，全部 7 数据集）
        fv = OUT / f"backbone/{ds}/patchtst_{ds}_s{seed}_val.npy"
        ft = OUT / f"backbone/{ds}/patchtst_{ds}_s{seed}_test.npy"
        return (fv, None), (ft, None)
    if backbone == "iTransformer":
        fv = OUT / f"backbone/{ds}/itransformer_{ds}_s{seed}_val.npy"
        ft = OUT / f"backbone/{ds}/itransformer_{ds}_s{seed}_test.npy"
        return (fv, None), (ft, None)
    if backbone == "RAFT":
        fv = OUT / f"backbone/{ds}/raft_{ds}_s{seed}_val.npy"
        ft = OUT / f"backbone/{ds}/raft_{ds}_s{seed}_test.npy"
        return (fv, None), (ft, None)
    return (None, None), (None, None)


# --------------------------------------------------------------------------- #
# 盘点
# --------------------------------------------------------------------------- #
def inventory():
    rows = []
    for ds in DATASETS:
        for bb in BACKBONES:
            for seed in SEEDS:
                (mv, mk), (mt, mtk) = memory_paths(ds, seed)
                mem_ok = load_array(mv, mk) is not None and load_array(mt, mtk) is not None
                (bv, bk), (bt, btk) = backbone_paths(ds, bb, seed)
                base_ok = (bb == "Chronos-2" and (REAUDIT / f"{ds}.npz").exists()) or \
                          (load_array(bv, bk) is not None and load_array(bt, btk) is not None)
                status = "可复用" if (mem_ok and base_ok) else \
                         ("需补记忆" if (not mem_ok and base_ok) else \
                          ("需补底座前向" if (mem_ok and not base_ok) else "需补记忆+底座"))
                rows.append({
                    "dataset": ds, "backbone": bb, "seed": seed,
                    "memory_cache": "OK" if mem_ok else "MISSING",
                    "backbone_cache": "OK" if base_ok else "MISSING",
                    "status": status,
                    "memory_path": str(mv), "backbone_path": str(bv if bb != "Chronos-2" else REAUDIT / f"{ds}.npz"),
                })
    with open(OUT / "inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print("inventory.csv written. 组合状态（84=4底座×7集×3seed）：", dict(c))
    print("（记忆 base-agnostic：同数据集同 seed 四底座共用）")
    for ds in DATASETS:
        mem_ok = any(r["memory_cache"] == "OK" for r in rows if r["dataset"] == ds)
        bb = Counter(r["backbone_cache"] for r in rows if r["dataset"] == ds)
        print(f"  {ds:14s} 记忆={'OK' if mem_ok else 'MISSING'}  底座[{bb['OK']}/12]")
    return rows


def select_lambda(y, base_val, mem_val):
    best_lam, best_mse = None, None
    for lam in LAMBDA_GRID:
        m = mse(y, (1 - lam) * base_val + lam * mem_val)
        if best_mse is None or m < best_mse:
            best_mse, best_lam = m, lam
    return best_lam


def compute_fusion(ds, backbone):
    """对一个 dataset×backbone 计算融合（3 seed），产出 per_seed + summary。"""
    rows = []
    for seed in SEEDS:
        (mv, mk), (mt, mtk) = memory_paths(ds, seed)
        mem_val = load_array(mv, mk); mem_test = load_array(mt, mtk)
        (bv, bk), (bt, btk) = backbone_paths(ds, backbone, seed)
        base_val = load_array(bv, bk); base_test = load_array(bt, btk)
        if mem_val is None or mem_test is None or base_val is None or base_test is None:
            print(f"  [{ds}/{backbone}/seed{seed}] 缺缓存，跳过（mem={mem_val is not None} base={base_test is not None}）", flush=True)
            continue
        # 标签：Chronos-2 用 reaudit Y；PatchTST 用 mem 目录 Y（若生成）或 reaudit
        lab = REAUDIT / f"{ds}.npz"
        Y_val = np.load(lab)["Y_val"].astype(np.float64)
        Y_test = np.load(lab)["Y_test"].astype(np.float64)
        if Y_val.shape != mem_val.shape or Y_test.shape != mem_test.shape:
            # 用本地生成的 Y（mem 目录）
            Y_val = np.load(OUT / f"mem/{ds}/Y_val_s{seed}.npy").astype(np.float64)
            Y_test = np.load(OUT / f"mem/{ds}/Y_test_s{seed}.npy").astype(np.float64)
        lam = select_lambda(Y_val, base_val, mem_val)
        fused = (1 - lam) * base_test + lam * mem_test
        rows.append({
            "dataset": ds, "backbone": backbone, "seed": seed,
            "base_val_mse": mse(Y_val, base_val), "base_test_mse": mse(Y_test, base_test),
            "mem_test_mse": mse(Y_test, mem_test),
            "lambda": lam, "fused_test_mse": mse(Y_test, fused),
            "ri_pct": (mse(Y_test, base_test) - mse(Y_test, fused)) / mse(Y_test, base_test) * 100,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--generate-memory", action="store_true")
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.inventory_only:
        inventory()
        return

    if args.generate_memory and args.dataset:
        generate_memory(args.dataset)
        return

    if args.fuse and args.dataset and args.backbone:
        rows = compute_fusion(args.dataset, args.backbone)
        if rows:
            import csv as _csv
            out_csv = OUT / f"per_seed_results_{args.dataset}_{args.backbone}.csv"
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            bm = float(np.mean([r["base_test_mse"] for r in rows]))
            fm = float(np.mean([r["fused_test_mse"] for r in rows]))
            lam = "/".join(f"{r['lambda']:g}" for r in rows)
            ri = (bm - fm) / bm * 100
            print(f"[{args.dataset}/{args.backbone}] base={bm:.6g} fused={fm:.6g} RI={ri:+.4f}% λ=({lam})")
        return


if __name__ == "__main__":
    main()
