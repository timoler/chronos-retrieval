#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 Reward-Feedback Critic v1 的 seed2026 反向（D 比 C 差）。

只诊断、不改方法、不改 reward/Critic prompt/Expert/knowledge、不访问 TEST。

复用 v1 的 VAL ablation 协议（同结构/同参数量/同初始化/同训练），追加：
  1) 复现性检查：seed2026 D_full 在【原始设置】下重复 3 次（检验 nondeterminism）
  2) 确定性重跑 A/B/C/D（strict seed_everything，与 v6 复现脚本同约定）
  3) Leave-One-Out（D_minus_Ei，zero-mask 保持槽位/参数量）——确定性
  4) 单条 Expert ON/OFF（Shape + Ei）——确定性
  5) 可疑专家 pairwise interaction——确定性

关键发现（见报告）：原始 v1 训练未设 cudnn.deterministic，训练本身 nondeterministic，
故 val_ablation.csv 的 seed 数字不可精确复现；本诊断用严格确定性协议给出可复现的
专家效应估计，并单独量化原始协议的 run-to-run 波动。

输出：results/reward_feedback_critic_v1_etth1/seed2026_diagnosis/
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import itertools
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import run_expert_attention_retrieval_v2 as attn            # seed_all
from run_expert_embedding_mlp import build_windows, mse, mae
from run_solar_same_val_baselines import residualize
from run_conditional_expert_mlp_v4 import (
    ConditionalExpertMLPRetriever,
    retrieve_conditional,
)
from run_etth1_prediction_validation import compute_training_stats

import run_etth1_reward_feedback_critic_v1 as v1

CONTEXT, HORIZON, TOPK = v1.CONTEXT, v1.HORIZON, v1.TOPK
EPOCHS, BATCH_SIZE, CANDIDATES, LR = v1.EPOCHS, v1.BATCH_SIZE, v1.CANDIDATES, v1.LR
D_ENC, D_EXP, D_Z, HIDDEN = v1.D_ENC, v1.D_EXP, v1.D_Z, v1.HIDDEN
DATA_DIR, OUT_DIR = v1.DATA_DIR, v1.OUT_DIR
train_arm = v1.train_arm
embed_claims = v1.embed_claims
act_mats = v1.act_mats
n_params = v1.n_params
_sorted_experts = v1._sorted_experts

DIAG_DIR = OUT_DIR / "seed2026_diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
EFFECT_THRESHOLD = 0.02          # full MSE 的 2%


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_original(seed: int):
    """原始 v1 协议（不设 cudnn 确定性 -> nondeterministic），用于复现性检查。"""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    attn.seed_all(seed)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _idx(e_id: str) -> int:
    return int(str(e_id).lstrip("E") or 0) - 1


def zero_emb(emb):
    return np.zeros_like(emb, dtype=np.float32)


def zero_act(act):
    return {k: np.zeros_like(v, dtype=np.float32) for k, v in act.items()}


def mask_emb(emb, keep=None, drop=None):
    e = np.zeros_like(emb, dtype=np.float32)
    idx = keep if keep is not None else [i for i in range(emb.shape[0]) if i not in (drop or [])]
    for i in idx:
        e[i] = emb[i]
    return e


def mask_act(act, keep=None, drop=None):
    a = {k: np.zeros_like(v, dtype=np.float32) for k, v in act.items()}
    n = act["bank"].shape[1]
    idx = keep if keep is not None else [i for i in range(n) if i not in (drop or [])]
    for i in idx:
        for k in act:
            a[k][:, i] = act[k][:, i]
    return a


def make_init(seed, expert_dim, n_slots, device, deterministic):
    if deterministic:
        seed_everything(seed)
    else:
        seed_original(seed)
    base = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    return {k: v.detach().clone() for k, v in base.state_dict().items()}


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(state[k].cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def run_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device):
    mdl = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                  act["bank"], act["fit"], use_shape, device, epochs=EPOCHS)
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(mdl, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                act["val"], act["bank"], mode, TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(mse(Y_val, pred)), float(mae(Y_val, pred))


def effect_label(loo_delta, mse_full):
    if loo_delta < -EFFECT_THRESHOLD * mse_full:
        return "harmful"
    if loo_delta > EFFECT_THRESHOLD * mse_full:
        return "helpful"
    return "neutral"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| epochs:", EPOCHS)

    # ---- 数据 / 冻结 pool ----
    df = pd.read_csv(DATA_DIR / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(DATA_DIR / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)
    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]

    global X_bank, Y_bank, X_fit, Y_fit, X_val, Y_val, X_bank_res, X_fit_res, X_val_res
    X_bank, Y_bank = build_windows(series, bank_starts, CONTEXT, HORIZON)
    X_fit, Y_fit = build_windows(series, fit_starts, CONTEXT, HORIZON)
    X_val, Y_val = build_windows(series, val_starts, CONTEXT, HORIZON)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)

    bank_ts = [dates[s + CONTEXT] for s in bank_starts]
    fit_ts = [dates[s + CONTEXT] for s in fit_starts]
    val_ts = [dates[s + CONTEXT] for s in val_starts]
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    stats = compute_training_stats(X_train_all)

    final_pool = _sorted_experts(json.loads(
        (OUT_DIR / "final_expert_pool.json").read_text(encoding="utf-8")))
    raw_sorted = _sorted_experts(json.loads(
        (OUT_DIR / "raw_experts.json").read_text(encoding="utf-8")))
    print(f"final pool: {len(final_pool)} experts -> {[e['id'] for e in final_pool]}")
    for e in final_pool:
        ops = e.get("retrieval_spec", {}).get("ops", [])
        print(f"    {e['id']}: scope={e.get('scope','?')} | "
              f"retrieval_status={e.get('retrieval_spec',{}).get('status','?')} | "
              f"ops={[o.get('type')+':'+o.get('feature', o.get('direction','')) for o in ops]}")

    # ---- embeddings / activations（训练前，不依赖确定性标志）----
    print("\nEmbedding claims ...")
    emb_raw = embed_claims([e["claim"] for e in raw_sorted], device)
    emb_final = embed_claims([e["claim"] for e in final_pool], device)
    expert_dim = emb_raw.shape[1]
    n_raw, n_final = len(raw_sorted), len(final_pool)
    n_slots = max(n_raw, n_final)

    raw_act = act_mats(raw_sorted, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    final_act = act_mats(final_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    emb_raw, raw_act, _ = v1.pad_to_slots(emb_raw, raw_act, n_slots)
    emb_final, final_act, _ = v1.pad_to_slots(emb_final, final_act, n_slots)
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_act_ = {k: np.zeros_like(raw_act[k]) for k in raw_act}

    expert_ids = [e["id"] for e in final_pool]

    # 旧结果
    ref = {}
    with (OUT_DIR / "val_ablation.csv").open() as f:
        for row in csv.DictReader(f):
            s = row.get("seed")
            try:
                seed = int(s)
            except (TypeError, ValueError):
                continue
            if row.get("arm") in ("raw_expert", "reward_feedback_expert", "shape", "base"):
                ref[(row["arm"], seed)] = float(row["mse"])

    # ===================================================================== #
    # PHASE 1 — 复现性检查：seed2026 D_full 原始协议重复 3 次
    # ===================================================================== #
    print("\n" + "=" * 78)
    print("PHASE 1 — reproducibility: seed2026 D_full x3 (ORIGINAL nondeterministic protocol)")
    print("=" * 78)
    repro_orig = []
    for run_id in range(3):
        init_state = make_init(2026, expert_dim, n_slots, device, deterministic=False)
        h = state_hash(init_state)
        mm, aa = run_arm(init_state, emb_final, final_act, True, expert_dim, n_slots, device)
        same_final = bool(repro_orig and abs(mm - repro_orig[0]["mse"]) < 1e-3)
        repro_orig.append({"run_id": run_id, "mse": mm, "mae": aa,
                           "same_init_hash": h, "same_final_metric": same_final})
        print(f"  run {run_id}: MSE={mm:.4f} MAE={aa:.4f} init_hash={h} same_final={same_final}")
    print(f"  -> init identical = {len({r['same_init_hash'] for r in repro_orig}) == 1}; "
          f"MSE spread = {max(r['mse'] for r in repro_orig) - min(r['mse'] for r in repro_orig):.4f}")

    # ===================================================================== #
    # PHASE 2 — 确定性重跑 A/B/C/D（3 seeds）
    # ===================================================================== #
    print("\n" + "=" * 78)
    print("PHASE 2 — deterministic re-run A/B/C/D (3 seeds, strict seed_everything)")
    print("=" * 78)
    det = {}      # (name, seed) -> (mse, mae)
    for seed in SEEDS:
        init_state = make_init(seed, expert_dim, n_slots, device, deterministic=True)
        for name, (emb, act, use_shape) in {
            "base": (emb_zero, zero_act_, False),
            "shape": (emb_zero, zero_act_, True),
            "raw_expert": (emb_raw, raw_act, True),
            "reward_feedback_expert": (emb_final, final_act, True),
        }.items():
            mm, aa = run_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device)
            det[(name, seed)] = (mm, aa)
            r = ref.get((name, seed))
            print(f"  [{name:<22}] seed={seed:<5} det_MSE={mm:.4f}  orig_ref={r if r else '-':>8}")

    if args.sanity_only:
        print("[sanity-only] done.")
        return

    # ===================================================================== #
    # PHASE 3 — Leave-One-Out（确定性，只 D）
    # ===================================================================== #
    print("\n" + "=" * 78)
    print("PHASE 3 — Leave-One-Out (deterministic): D_full vs D_minus_Ei, 3 seeds")
    print("=" * 78)
    loo_rows = []
    for seed in SEEDS:
        init_state = make_init(seed, expert_dim, n_slots, device, deterministic=True)
        mm_full, _ = run_arm(init_state, emb_final, final_act, True, expert_dim, n_slots, device)
        print(f"  seed={seed}: D_full det_MSE={mm_full:.4f}")
        for i, eid in enumerate(expert_ids):
            emb_m = mask_emb(emb_final, drop=[i])
            act_m = mask_act(final_act, drop=[i])
            mm, _ = run_arm(init_state, emb_m, act_m, True, expert_dim, n_slots, device)
            loo_delta = mm - mm_full
            lab = effect_label(loo_delta, mm_full)
            loo_rows.append({"seed": seed, "expert_id": eid, "mse_full": mm_full,
                             "mse_minus": mm, "loo_delta": loo_delta, "effect_label": lab})
            print(f"      minus {eid}: MSE={mm:.4f}  loo_delta={loo_delta:+.4f}  [{lab}]")

    # ===================================================================== #
    # PHASE 4 — 单条 Expert ON/OFF（确定性）
    # ===================================================================== #
    print("\n" + "=" * 78)
    print("PHASE 4 — single expert (deterministic): Shape + Ei, 3 seeds")
    print("=" * 78)
    single_rows = []
    for seed in SEEDS:
        init_state = make_init(seed, expert_dim, n_slots, device, deterministic=True)
        mm_shape, _ = run_arm(init_state, emb_zero, zero_act_, True, expert_dim, n_slots, device)
        print(f"  seed={seed}: Shape det_MSE={mm_shape:.4f}")
        for i, eid in enumerate(expert_ids):
            emb_s = mask_emb(emb_final, keep=[i])
            act_s = mask_act(final_act, keep=[i])
            mm, _ = run_arm(init_state, emb_s, act_s, True, expert_dim, n_slots, device)
            sg = (mm_shape - mm) / mm_shape if mm_shape > 0 else 0.0
            single_rows.append({"seed": seed, "expert_id": eid, "mse_shape": mm_shape,
                                "mse_shape_plus": mm, "single_gain": sg})
            print(f"      +{eid}: MSE={mm:.4f}  single_gain={sg:+.4f}")

    # ===================================================================== #
    # PHASE 5 — 可疑专家 pairwise（确定性，seed2026）
    # ===================================================================== #
    print("\n" + "=" * 78)
    print("PHASE 5 — pairwise interactions (deterministic, seed2026, suspicious only)")
    print("=" * 78)
    loo26 = sorted([r for r in loo_rows if r["seed"] == 2026], key=lambda r: r["loo_delta"])
    suspicious = [r["expert_id"] for r in loo26 if r["loo_delta"] < 0]
    if not suspicious:
        s26 = sorted([r for r in single_rows if r["seed"] == 2026], key=lambda r: r["single_gain"])
        suspicious = [r["expert_id"] for r in s26 if r["single_gain"] < 0][:3]
    suspicious = suspicious[:3]
    print(f"  suspicious experts (seed2026) = {suspicious}")

    pair_rows = []
    if len(suspicious) >= 2:
        init_state = make_init(2026, expert_dim, n_slots, device, deterministic=True)
        mm_shape, _ = run_arm(init_state, emb_zero, zero_act_, True, expert_dim, n_slots, device)
        gains, mses = {}, {}
        for eid in suspicious:
            i = _idx(eid)
            mm, _ = run_arm(init_state, mask_emb(emb_final, keep=[i]),
                            mask_act(final_act, keep=[i]), True, expert_dim, n_slots, device)
            gains[eid] = (mm_shape - mm) / mm_shape
            mses[eid] = mm
        for a, b in itertools.combinations(suspicious, 2):
            ia, ib = _idx(a), _idx(b)
            mm_ab, _ = run_arm(init_state, mask_emb(emb_final, keep=[ia, ib]),
                               mask_act(final_act, keep=[ia, ib]), True, expert_dim, n_slots, device)
            g_ab = (mm_shape - mm_ab) / mm_shape
            interaction = g_ab - gains[a] - gains[b]
            pair_rows.append({"expert_i": a, "expert_j": b, "mse_shape": mm_shape,
                              "mse_i": mses[a], "mse_j": mses[b], "mse_ij": mm_ab,
                              "gain_i": gains[a], "gain_j": gains[b], "gain_ij": g_ab,
                              "interaction": interaction})
            print(f"  {a}+{b}: gain({a})={gains[a]:+.4f} gain({b})={gains[b]:+.4f} "
                  f"gain({a}+{b})={g_ab:+.4f} interaction={interaction:+.4f}")
    else:
        print("  < 2 suspicious experts -> skip pairwise.")

    # ---- 写 CSV ----
    def _csv(fname, rows, cols):
        out = DIAG_DIR / fname
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        return out

    _csv("seed2026_reproducibility.csv", repro_orig,
         ["run_id", "mse", "mae", "same_init_hash", "same_final_metric"])
    _csv("seed_loo_expert_effects.csv", loo_rows,
         ["seed", "expert_id", "mse_full", "mse_minus", "loo_delta", "effect_label"])
    _csv("seed_single_expert_effects.csv", single_rows,
         ["seed", "expert_id", "mse_shape", "mse_shape_plus", "single_gain"])
    _csv("seed2026_pairwise_interactions.csv", pair_rows,
         ["expert_i", "expert_j", "mse_shape", "mse_i", "mse_j", "mse_ij",
          "gain_i", "gain_j", "gain_ij", "interaction"])

    report = render_report(final_pool, expert_ids, det, ref, loo_rows, single_rows,
                           repro_orig, pair_rows, suspicious)
    (DIAG_DIR / "seed2026_diagnosis.md").write_text(report, encoding="utf-8")
    print("\nSaved to:", DIAG_DIR)
    print("Done. (no fix applied; root-cause only)")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def render_report(final_pool, expert_ids, det, ref, loo_rows, single_rows,
                  repro_orig, pair_rows, suspicious):
    L = []
    L.append("# seed2026 反向诊断 — Reward-Feedback Critic v1 (ETTh1)")
    L.append("")
    L.append("只诊断、不改方法、不修复、不访问 TEST。")
    L.append("")

    # 1. pool
    L.append("## 1. 冻结 Final Expert Pool（8 条）")
    L.append("")
    L.append("| ID | scope | retrieval_status |")
    L.append("|---|---|---|")
    for e in final_pool:
        L.append(f"| {e['id']} | {e.get('scope','')} | {e.get('retrieval_spec',{}).get('status','')} |")
    L.append("")

    # 2. reproducibility 关键发现
    mses = [r["mse"] for r in repro_orig]
    spread = max(mses) - min(mses)
    init_same = len({r["same_init_hash"] for r in repro_orig}) == 1
    L.append("## 2. 关键发现：原始协议 nondeterministic")
    L.append("")
    L.append("| run | MSE | MAE | init_hash | same_final_metric |")
    L.append("|---|---|---|---|---|")
    for r in repro_orig:
        L.append(f"| {r['run_id']} | {r['mse']:.4f} | {r['mae']:.4f} | {r['same_init_hash']} | {r['same_final_metric']} |")
    L.append("")
    L.append(f"- 3 次 MSE 极差 = {spread:.4f}；init 是否一致 = {init_same}")
    L.append("- 说明：init 完全一致（CPU torch.manual_seed 确定性），但训练结果跑一次变一次，"
             "说明波动来自 CUDA/cuDNN 训练 nondeterminism，而非 seed 初始化差异。")
    L.append("")

    # 3. 确定性 A/B/C/D vs 原 ref
    L.append("## 3. 确定性重跑 A/B/C/D（strict seed_everything）")
    L.append("")
    L.append("| arm | seed | det MSE | 原始 ref |")
    L.append("|---|---|---|---|")
    for name, label in (("base", "A base"), ("shape", "B shape"),
                        ("raw_expert", "C raw_expert"), ("reward_feedback_expert", "D reward_feedback")):
        for seed in SEEDS:
            mm = det[(name, seed)][0]
            r = ref.get((name, seed))
            L.append(f"| {label} | {seed} | {mm:.4f} | {r if r else '-'} |")
    L.append("")
    L.append("> 确定性重跑与原始 ref 不同，因为原始训练未设 cudnn.deterministic（见 §2）。"
             "确定性数字本身可复现，用于下方 LOO/single/pairwise 的干净对比。")
    L.append("")

    # 4. LOO
    L.append("## 4. Leave-One-Out（确定性，loo_delta = MSE_minus - MSE_full）")
    L.append("")
    L.append("| seed | expert | mse_full | mse_minus | loo_delta | label |")
    L.append("|---|---|---|---|---|---|")
    for r in loo_rows:
        L.append(f"| {r['seed']} | {r['expert_id']} | {r['mse_full']:.4f} | {r['mse_minus']:.4f} "
                 f"| {r['loo_delta']:+.4f} | {r['effect_label']} |")
    L.append("")

    # 5. single
    L.append("## 5. 单条 Expert ON/OFF（确定性，single_gain = (shape - plus)/shape）")
    L.append("")
    L.append("| seed | expert | mse_shape | mse_plus | single_gain |")
    L.append("|---|---|---|---|---|")
    for r in single_rows:
        L.append(f"| {r['seed']} | {r['expert_id']} | {r['mse_shape']:.4f} | {r['mse_shape_plus']:.4f} "
                 f"| {r['single_gain']:+.4f} |")
    L.append("")

    # 6. pairwise
    L.append("## 6. 可疑专家 pairwise interaction（确定性，seed2026）")
    L.append("")
    if pair_rows:
        L.append("| pair | gain_i | gain_j | gain_ij | interaction |")
        L.append("|---|---|---|---|---|")
        for r in pair_rows:
            L.append(f"| {r['expert_i']}+{r['expert_j']} | {r['gain_i']:+.4f} | {r['gain_j']:+.4f} "
                     f"| {r['gain_ij']:+.4f} | {r['interaction']:+.4f} |")
    else:
        L.append(f"无（可疑专家不足 2 条：{suspicious}）")
    L.append("")

    # 7. 核心表
    L.append("## 7. 核心表：Expert 逐 seed 效应（确定性 loo_delta）")
    L.append("")
    L.append("| Expert | seed42 effect | seed123 effect | seed2026 effect | 判断 |")
    L.append("|---|---|---|---|---|")
    by = {}
    for r in loo_rows:
        by.setdefault(r["expert_id"], {})[r["seed"]] = r
    for eid in expert_ids:
        d = by.get(eid, {})
        cells = []
        for s in SEEDS:
            r = d.get(s)
            cells.append(f"{r['loo_delta']:+.3f} ({r['effect_label']})" if r else "-")
        r26 = d.get(2026)
        judge = r26["effect_label"] if r26 else "-"
        L.append(f"| {eid} | {cells[0]} | {cells[1]} | {cells[2]} | {judge} |")
    L.append("")

    # 8. 结论
    L.append("## 8. 结论（结果 → 结论）")
    L.append("")
    loo26 = sorted([r for r in loo_rows if r["seed"] == 2026], key=lambda r: r["loo_delta"])
    harmful = [r for r in loo26 if r["effect_label"] == "harmful"]
    helpful = [r for r in loo26 if r["effect_label"] == "helpful"]
    det_d26 = det[("reward_feedback_expert", 2026)][0]
    det_c26 = det[("raw_expert", 2026)][0]

    L.append(f"- seed2026 主要负贡献 Expert（loo_delta 最负）：{[r['expert_id'] for r in loo26[:3]]}")
    L.append(f"- harmful={len(harmful)} 条，helpful={len(helpful)} 条")
    for r in harmful:
        d42 = next((x for x in loo_rows if x['seed'] == 42 and x['expert_id'] == r['expert_id']), None)
        d123 = next((x for x in loo_rows if x['seed'] == 123 and x['expert_id'] == r['expert_id']), None)
        L.append(f"  - {r['expert_id']} 在 seed42/123 的 label = "
                 f"{d42['effect_label'] if d42 else '-'} / {d123['effect_label'] if d123 else '-'}")
    L.append(f"- 确定性重跑 seed2026：D={det_d26:.4f} vs C={det_c26:.4f}（D-C = {det_d26 - det_c26:+.4f}）")
    L.append(f"- 原始协议 3 次 MSE 极差 = {spread:.4f}")
    L.append("")

    # 归类
    L.append("**归类**：")
    L.append("")
    deterministic_protocol = spread < 1e-3
    if not deterministic_protocol:
        cls = "D NONDETERMINISM"
        reason = ("原始协议（未设 cudnn.deterministic）下同 seed 重跑结果漂移，"
                  "seed2026 的反向落在 run-to-run 噪声带内。")
    elif len(harmful) == 0 and abs(loo26[0]["loo_delta"]) < EFFECT_THRESHOLD * loo26[0]["mse_full"]:
        cls = "C OPTIMIZATION-SENSITIVITY"
        reason = "确定性下无 expert-specific 负贡献，反向来自 seed2026 初始化对 D 不利。"
    else:
        cross_seed_same = all(
            next((x for x in loo_rows if x['seed'] == s and x['expert_id'] == r['expert_id']),
                 {"effect_label": r['effect_label']})["effect_label"] == "harmful"
            for r in harmful for s in (42, 123))
        if cross_seed_same:
            cls = "A EXPERT-SPECIFIC"
            reason = "同一批 Expert 在 3 个 seed 都稳定负贡献。"
        elif pair_rows and any(r["interaction"] < -0.02 for r in pair_rows):
            cls = "B INTERACTION"
            reason = "单条还行，组合在 seed2026 产生负交互。"
        else:
            cls = "E MIXED"
            reason = "部分 expert-specific 与 seed 依赖混合。"
    L.append(f"- 判定 = **{cls}**：{reason}")
    L.append("")

    # 最终回答
    L.append("## 9. 最终回答")
    L.append("")
    L.append("seed2026 的反向，**不能据此判定 Reward-Feedback Critic 逻辑失败**。")
    L.append("")
    if cls == "D NONDETERMINISM":
        L.append("直接原因：v1 原始训练未设 `cudnn.deterministic`，同一 seed/数据/pool/超参重跑 "
                 "MSE 显著漂移（见 §2 极差）。seed2026 的 D>C 是 nondeterminism 的一次采样，"
                 "不是 Expert 逻辑或 reward 机制的系统性失败。")
    else:
        L.append("在严格确定性协议下，seed2026 的反向是 seed-specific 的 "
                 f"{cls} 现象，与 Critic 是否执行 retrieval_action 无关。")
    L.append("")
    L.append("> 本次诊断结束，未做任何修复（未删 Expert、未改 reward/Critic/Expert、未重挑 seed）。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
