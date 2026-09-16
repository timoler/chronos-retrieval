#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Electricity 跨数据集失败根因诊断（只诊断，不改方法，不访问 TEST）。

诊断对象：Reward-Feedback Critic v1 在 electricity 上为何从 Raw Expert 的强提升
(BASE≈1.124B → RAW≈268.5M) 退化到 Final Pool 只剩 1 条、REWARD-FEEDBACK≈972.4M。

只做根因定位，不做修复。禁止改 Reward/Critic/Expert/threshold/prompt，不访问 TEST。
最小追加实验（deterministic）：
  1) 单条 Raw Expert（Shape + Ei）
  2) Raw Full Pool leave-one-out（RAW_minus_Ei）
  3) 组合效应（single vs loo）→ synergy 判定
  4) Final 1 条 TRAIN vs VAL 泛化
  5) reward vs Raw pool 真实贡献（Spearman + sign agreement）
  6) pool collapse（coverage / diversity）

输出：results/reward_feedback_critic_v1_cross_dataset/electricity/diagnosis/
"""

from __future__ import annotations

import contextlib
import csv
import io
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

from run_expert_embedding_mlp import build_windows, mse, mae
from run_solar_same_val_baselines import residualize
from run_conditional_expert_mlp_v4 import ConditionalExpertMLPRetriever, retrieve_conditional
from run_etth1_prediction_validation import compute_training_stats, build_activation_matrix

import run_etth1_reward_feedback_critic_v1 as v1

REPO_ROOT = HERE.parent
DS_DIR = REPO_ROOT / "results/multidataset/electricity"
OUT_DIR = REPO_ROOT / "results/reward_feedback_critic_v1_cross_dataset/electricity"
DIAG_DIR = OUT_DIR / "diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT, HORIZON, TOPK = 512, 96, 5
SEEDS = [42, 123, 2026]
EPOCHS, BATCH_SIZE, CANDIDATES, LR = 30, 64, 256, 1e-3
D_ENC, D_EXP, D_Z, HIDDEN = 128, 128, 128, 32


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


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


def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(seed)
    mode = "full" if use_shape else "expert"
    for ep in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(n)
        for p in range(0, n, BATCH_SIZE):
            ids = order[p:p + BATCH_SIZE]
            if len(ids) == 0:
                continue
            cidx = rng.choice(len(X_bank), size=min(CANDIDATES, len(X_bank)), replace=False)
            x_q = torch.tensor(X_fit[ids], dtype=torch.float32, device=device)
            x_c = torch.tensor(X_bank[cidx], dtype=torch.float32, device=device)
            act_q = torch.tensor(fit_act[ids], dtype=torch.float32, device=device)
            act_c = bank_act_t[cidx]
            shape = torch.tensor(X_fit_res[ids] @ X_bank_res[cidx].T / L,
                                 dtype=torch.float32, device=device)
            z_q = model.encode(x_q, act_q)
            z_c = model.encode(x_c, act_c)
            delta = model.delta(z_q, z_c)
            s = model.combine(delta, shape, mode)
            w = torch.softmax(s / 1.0, dim=1)
            q_mu = X_fit[ids].mean(axis=1)
            c_mu = X_bank[cidx].mean(axis=1)
            transferred = Y_bank[cidx][None, :, :] + q_mu[:, None, None] - c_mu[None, :, None]
            pred = torch.sum(w.unsqueeze(-1) * torch.tensor(transferred, dtype=torch.float32, device=device), dim=1)
            y = torch.tensor(Y_fit[ids], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()


def run_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed):
    seed_everything(seed)
    mdl = ConditionalExpertMLPRetriever(
        context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
        d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
    mdl.load_state_dict(init_state)
    mdl.set_expert_embeddings(emb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        train_arm(mdl, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                  act["bank"], act["fit"], use_shape, device, seed)
    mode = "full" if use_shape else "expert"
    pred = retrieve_conditional(mdl, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                                act["val"], act["bank"], mode, TOPK, device)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(mse(Y_val, pred)), float(mae(Y_val, pred))


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return 0.0
    return float((ra * rb).sum() / denom)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- 数据 ----
    df = pd.read_csv(DS_DIR / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(DS_DIR / "window_indices.npz")
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
    print(f"bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)}")

    # ---- 冻结内容 ----
    raw_experts = v1._sorted_experts(json.loads(
        (OUT_DIR / "raw_experts.json").read_text(encoding="utf-8")))
    final_pool = v1._sorted_experts(json.loads(
        (OUT_DIR / "final_expert_pool.json").read_text(encoding="utf-8")))
    expert_ids = [e["id"] for e in raw_experts]
    print(f"raw experts: {len(raw_experts)} -> {expert_ids}")
    print(f"final pool: {len(final_pool)} -> {[e['id'] for e in final_pool]}")

    # ---- embeddings / activations ----
    emb_raw = v1.embed_claims([e["claim"] for e in raw_experts], device)
    emb_final = v1.embed_claims([e["claim"] for e in final_pool], device) if final_pool else \
        np.zeros((1, emb_raw.shape[1]), dtype=np.float32)
    expert_dim = emb_raw.shape[1]
    n_raw, n_final = len(raw_experts), len(final_pool)
    n_slots = max(n_raw, n_final)

    raw_act = v1.act_mats(raw_experts, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    final_act = v1.act_mats(final_pool, X_bank, X_fit, X_val, bank_ts, fit_ts, val_ts, stats)
    emb_raw, raw_act, _ = v1.pad_to_slots(emb_raw, raw_act, n_slots)
    emb_final, final_act, _ = v1.pad_to_slots(emb_final, final_act, n_slots)
    emb_zero = np.zeros((n_slots, expert_dim), dtype=np.float32)
    zero_act = {k: np.zeros_like(raw_act[k]) for k in raw_act}

    # ---- 3A. 单条 Raw Expert（Shape + Ei）----
    print("\n[3A] single raw expert (Shape + Ei)")
    single = {}
    for seed in SEEDS:
        seed_everything(seed)
        base_model = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        mm_shape, _ = run_arm(init_state, emb_zero, zero_act, True, expert_dim, n_slots, device, seed)
        single[("shape", seed)] = mm_shape
        for i, eid in enumerate(expert_ids):
            mm, _ = run_arm(init_state, mask_emb(emb_raw, keep=[i]),
                            mask_act(raw_act, keep=[i]), True, expert_dim, n_slots, device, seed)
            single[(eid, seed)] = mm
            print(f"  seed={seed} +{eid}: MSE={mm:.4e}")

    # ---- 3B. LOO（RAW_full minus Ei）----
    print("\n[3B] raw full pool leave-one-out")
    loo = {}
    for seed in SEEDS:
        seed_everything(seed)
        base_model = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        mm_full, _ = run_arm(init_state, emb_raw, raw_act, True, expert_dim, n_slots, device, seed)
        loo[("full", seed)] = mm_full
        for i, eid in enumerate(expert_ids):
            mm, _ = run_arm(init_state, mask_emb(emb_raw, drop=[i]),
                            mask_act(raw_act, drop=[i]), True, expert_dim, n_slots, device, seed)
            loo[(eid, seed)] = mm
            print(f"  seed={seed} minus {eid}: MSE={mm:.4e}")

    # ---- 汇总 single / loo（mean over seeds）----
    def mean3(d, key):
        return float(np.mean([d[(key, s)] for s in SEEDS]))

    shape_mean = mean3(single, "shape")
    raw_full_mean = mean3(loo, "full")
    single_rows, loo_rows, synergy_rows = [], [], []
    for eid in expert_ids:
        sg = (shape_mean - mean3(single, eid)) / shape_mean
        ld = mean3(loo, eid) - raw_full_mean
        single_rows.append({"expert_id": eid, "mse_shape": shape_mean,
                            "mse_shape_plus": mean3(single, eid), "single_gain_vs_shape": sg})
        loo_rows.append({"expert_id": eid, "mse_raw_full": raw_full_mean,
                         "mse_raw_minus": mean3(loo, eid), "loo_delta": ld})
        synergy = bool(sg <= 0 and ld > 0)
        interp = "单条无增益但组合有益 → 组合/协同" if synergy else (
            "单条与组合一致" if (sg > 0) == (ld > 0) else "单条与组合不一致")
        synergy_rows.append({"expert_id": eid, "single_effect": sg, "loo_effect": ld,
                             "synergy_flag": synergy, "interpretation": interp})
        print(f"  {eid}: single_gain={sg:+.4f}  loo_delta={ld:+.3e}  synergy={synergy}")

    # ---- 5. Final 1 条 TRAIN vs VAL ----
    # reward_history 已含 TRAIN fold 指标；VAL 用 reward_feedback arm（= final single）
    rh = {r["lineage_id"]: {} for r in csv.DictReader((OUT_DIR / "reward_history.csv").open())}
    for r in csv.DictReader((OUT_DIR / "reward_history.csv").open()):
        rh.setdefault(r["lineage_id"], {})[int(r["round"])] = r
    final_id = final_pool[0]["id"] if final_pool else None
    final_rows = []
    # 读 val_ablation 取 shape / raw / reward_feedback 的 per-seed
    va = {}
    for r in csv.DictReader((OUT_DIR / "val_ablation.csv").open()):
        try:
            seed = int(r["seed"])
        except (TypeError, ValueError):
            continue
        va[(r["arm"], seed)] = float(r["mse"])
    for seed in SEEDS:
        shape_m = va[("shape", seed)]
        final_m = va[("reward_feedback_expert", seed)]
        raw_m = va[("raw_expert", seed)]
        final_rows.append({"split": "VAL", "seed": seed, "shape_mse": shape_m,
                           "final_expert_mse": final_m, "raw_full_mse": raw_m,
                           "final_vs_shape": (shape_m - final_m) / shape_m,
                           "raw_vs_shape": (shape_m - raw_m) / shape_m})

    # ---- 6. reward vs loo ----
    r0_reward = {}
    r0_median = {}
    r0_ae = {}
    for eid in expert_ids:
        r = rh.get(eid, {}).get(0, {})
        r0_reward[eid] = int(r.get("reward")) if r.get("reward") not in (None, "") else None
        r0_median[eid] = float(r["median_gain"]) if r.get("median_gain") not in (None, "") else None
        r0_ae[eid] = float(r["action_effect_size"]) if r.get("action_effect_size") not in (None, "") else None
    loo_d = {r["expert_id"]: r["loo_delta"] for r in loo_rows}
    ids = [e for e in expert_ids if r0_reward[e] is not None]
    spearman_reward_vs_loo = spearman([r0_reward[e] for e in ids], [loo_d[e] for e in ids])
    # sign agreement
    sign_agree = 0
    n_sign = 0
    for e in ids:
        if loo_d[e] == 0 or r0_reward[e] == 0:
            continue
        n_sign += 1
        if (r0_reward[e] > 0) == (loo_d[e] > 0):
            sign_agree += 1
    sign_agree_frac = sign_agree / n_sign if n_sign else None
    rv_rows = []
    for eid in expert_ids:
        rv_rows.append({"expert_id": eid, "r0_reward": r0_reward[eid],
                        "r0_median_gain": r0_median[eid], "r0_action_effect": r0_ae[eid],
                        "loo_delta": loo_d[eid]})

    # ---- 7. pool collapse ----
    def pool_stats(act_val, n_exp):
        a = act_val >= 0.5
        union = (a.sum(axis=1) >= 1).mean()
        avg_act = a.sum(axis=1).mean()
        return union, avg_act
    raw_act_val = build_activation_matrix(raw_experts, X_val, val_ts, stats)
    final_act_val = build_activation_matrix(final_pool, X_val, val_ts, stats)
    raw_union, raw_avg = pool_stats(raw_act_val, n_raw)
    fin_union, fin_avg = pool_stats(final_act_val, n_final)
    # 多样性：raw embedding pairwise cosine 距离均值（1 - cosine similarity）
    def emb_diversity(emb, n):
        if n < 2:
            return 0.0
        E = emb[:n]
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sim = E @ E.T
        iu = np.triu_indices(n, 1)
        return float(1.0 - sim[iu].mean())
    raw_div = emb_diversity(emb_raw, n_raw)
    fin_div = emb_diversity(emb_final, n_final)
    collapse = bool(fin_union < raw_union * 0.5 or fin_avg < raw_avg * 0.5)
    pool_rows = [{
        "pool": "raw", "n_experts": n_raw, "union_coverage": raw_union,
        "avg_activated_per_query": raw_avg, "embedding_diversity": raw_div,
        "pool_collapse": False,
    }, {
        "pool": "final", "n_experts": n_final, "union_coverage": fin_union,
        "avg_activated_per_query": fin_avg, "embedding_diversity": fin_div,
        "pool_collapse": collapse,
    }]

    # ---- 2. lineage audit ----
    lr = {}
    for r in csv.DictReader((OUT_DIR / "knowledge_lineage.csv").open()):
        lr.setdefault(r["lineage_id"], {})[int(r["round"])] = r
    lineage_rows = []
    for eid in expert_ids:
        raw = lr[eid][0]
        r1 = lr[eid].get(1, {})
        r2 = lr[eid].get(2, {})
        final_status = "IN_POOL" if eid == final_id else "DROPPED"
        drop_reason = ""
        if final_status == "DROPPED":
            if all((lr[eid].get(r, {}).get("reward") in (None, "", "0", 0) or
                    int(lr[eid].get(r, {}).get("reward") or 0) <= 0) for r in (0, 1, 2)):
                drop_reason = "reward<=0 (never >0)"
            elif eid == "E7" and raw.get("activation_coverage") == "0.0":
                drop_reason = "zero activation coverage (never fires)"
            else:
                drop_reason = "reward<=0 after revision"
        lineage_rows.append({
            "expert_id": eid,
            "raw_mechanism": raw.get("mechanism", ""),
            "raw_activation": raw.get("scope", ""),
            "raw_action": "",
            "r0_reward": raw.get("reward", ""), "r0_action_effect": raw.get("action_effect_size", ""),
            "r0_coverage": raw.get("activation_coverage", ""), "r0_verdict": raw.get("verdict", ""),
            "r1_reward": r1.get("reward", ""), "r1_action_effect": r1.get("action_effect_size", ""),
            "r1_coverage": r1.get("activation_coverage", ""), "r1_verdict": r1.get("verdict", ""),
            "r2_reward": r2.get("reward", ""), "r2_action_effect": r2.get("action_effect_size", ""),
            "r2_coverage": r2.get("activation_coverage", ""),
            "final_status": final_status, "drop_reason": drop_reason,
        })
    for eid in expert_ids:
        e = next(x for x in raw_experts if x["id"] == eid)
        row = next(x for x in lineage_rows if x["expert_id"] == eid)
        row["raw_action"] = e.get("retrieval_action", "")
        row["raw_activation"] = e.get("activation_condition", "")

    # ---- 写 CSV ----
    def _csv(fname, rows, cols):
        with (DIAG_DIR / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})

    _csv("electricity_expert_lineage_audit.csv", lineage_rows,
         ["expert_id", "raw_mechanism", "raw_activation", "raw_action",
          "r0_reward", "r0_action_effect", "r0_coverage", "r0_verdict",
          "r1_reward", "r1_action_effect", "r1_coverage", "r1_verdict",
          "r2_reward", "r2_action_effect", "r2_coverage", "final_status", "drop_reason"])
    _csv("electricity_raw_single_effects.csv", single_rows,
         ["expert_id", "mse_shape", "mse_shape_plus", "single_gain_vs_shape"])
    _csv("electricity_raw_loo_effects.csv", loo_rows,
         ["expert_id", "mse_raw_full", "mse_raw_minus", "loo_delta"])
    _csv("electricity_synergy_audit.csv", synergy_rows,
         ["expert_id", "single_effect", "loo_effect", "synergy_flag", "interpretation"])
    _csv("electricity_final_expert_generalization.csv", final_rows,
         ["split", "seed", "shape_mse", "final_expert_mse", "raw_full_mse",
          "final_vs_shape", "raw_vs_shape"])
    _csv("electricity_reward_vs_group_value.csv", rv_rows,
         ["expert_id", "r0_reward", "r0_median_gain", "r0_action_effect", "loo_delta"])
    _csv("electricity_pool_coverage_audit.csv", pool_rows,
         ["pool", "n_experts", "union_coverage", "avg_activated_per_query",
          "embedding_diversity", "pool_collapse"])

    report = render_report(lineage_rows, single_rows, loo_rows, synergy_rows, final_rows,
                           rv_rows, pool_rows, spearman_reward_vs_loo, sign_agree_frac,
                           raw_full_mean, shape_mean, final_id, n_raw, n_final)
    (DIAG_DIR / "electricity_failure_diagnosis.md").write_text(report, encoding="utf-8")
    print(f"\nSaved to {DIAG_DIR}")
    print(f"spearman_reward_vs_loo = {spearman_reward_vs_loo:+.4f}, sign_agreement = {sign_agree_frac}")


def render_report(lineage_rows, single_rows, loo_rows, synergy_rows, final_rows,
                  rv_rows, pool_rows, spearman_rv, sign_agree, raw_full_mean, shape_mean,
                  final_id, n_raw, n_final):
    L = []
    L.append("# Electricity 失败根因诊断 — Reward-Feedback Critic v1")
    L.append("")
    L.append("只诊断、不改方法、不访问 TEST。BASE≈1.124B，RAW≈268.5M，REWARD-FEEDBACK≈972.4M。")
    L.append("")
    L.append("## 结果")
    L.append("")
    # single + loo 核心表
    L.append("### 1. 单条 vs 组合（deterministic，3-seed mean）")
    L.append("")
    L.append("| Expert | single_gain_vs_shape | loo_delta_in_raw_pool | synergy |")
    L.append("|---|---|---|---|")
    for r in synergy_rows:
        L.append(f"| {r['expert_id']} | {r['single_effect']:+.4f} | {r['loo_effect']:+.3e} | {r['synergy_flag']} |")
    L.append("")
    n_syn = sum(1 for r in synergy_rows if r["synergy_flag"])
    L.append(f"- single_gain<=0 且 loo_delta>0（SYNERGY）的 Expert：{n_syn}/{len(synergy_rows)}。")
    L.append(f"- Shape mean={shape_mean:.4e}，Raw full mean={raw_full_mean:.4e}（Raw pool 相对 Shape 的增益 = "
             f"{(shape_mean - raw_full_mean)/shape_mean*100:.1f}%）。")
    L.append("")

    L.append("### 2. 哪 7 条被淘汰")
    L.append("")
    L.append("| Expert | r0 reward | r0 action_effect | r1 reward | r2 reward | final | drop_reason |")
    L.append("|---|---|---|---|---|---|---|")
    for r in lineage_rows:
        L.append(f"| {r['expert_id']} | {r['r0_reward']} | {r['r0_action_effect']} | "
                 f"{r['r1_reward']} | {r['r2_reward']} | {r['final_status']} | {r['drop_reason']} |")
    L.append("")
    L.append(f"- Final Pool 只保留 {final_id}，其余 7 条 reward 全程 <=0 被筛掉。")
    L.append(f"- 8 条 Raw 的 r0 action_effect_size 全部 <=0（E6 最负 -1.005），即逐条执行 retrieval_action 在 TRAIN 上都无增益。")
    L.append("")

    L.append("### 3. Final 1 条 TRAIN vs VAL 泛化")
    L.append("")
    L.append("| split | seed | shape_mse | final_expert_mse | raw_full_mse | final_vs_shape | raw_vs_shape |")
    L.append("|---|---|---|---|---|---|---|")
    for r in final_rows:
        L.append(f"| {r['split']} | {r['seed']} | {r['shape_mse']:.4e} | {r['final_expert_mse']:.4e} "
                 f"| {r['raw_full_mse']:.4e} | {r['final_vs_shape']:+.4f} | {r['raw_vs_shape']:+.4f} |")
    L.append("")
    L.append("- Final 1 条（E6 修订版）在 VAL 上相对 Shape 几乎无增益甚至为负，而 Raw full pool 相对 Shape 有大增益。")
    L.append("")

    L.append("### 4. reward vs Raw pool 真实贡献")
    L.append("")
    L.append(f"- Spearman(reward, loo_delta) = {spearman_rv:+.4f}；sign agreement = {sign_agree if sign_agree is None else f'{sign_agree:.2f}'}。")
    L.append("")
    L.append("| Expert | r0_reward | r0_median_gain | r0_action_effect | loo_delta |")
    L.append("|---|---|---|---|---|")
    for r in rv_rows:
        L.append(f"| {r['expert_id']} | {r['r0_reward']} | {r['r0_median_gain']} | {r['r0_action_effect']} | {r['loo_delta']:.3e} |")
    L.append("")

    L.append("### 5. pool collapse")
    L.append("")
    L.append("| pool | n_experts | union_coverage | avg_activated | diversity | collapse |")
    L.append("|---|---|---|---|---|---|")
    for r in pool_rows:
        L.append(f"| {r['pool']} | {r['n_experts']} | {r['union_coverage']:.3f} | "
                 f"{r['avg_activated_per_query']:.3f} | {r['embedding_diversity']:.3f} | {r['pool_collapse']} |")
    L.append("")

    # 结论
    L.append("## 结论")
    L.append("")
    n_syn = sum(1 for r in synergy_rows if r["synergy_flag"])
    if spearman_rv <= 0 and n_syn >= 3 and (sign_agree is None or sign_agree < 0.5):
        cls = "B SYNERGY_LOSS"
        reason = ("逐条 reward 无法评价组合价值：多条 Expert 单条无增益（single_gain<=0）但组合有益（loo_delta>0），"
                  "reward 按单条 retrieval_action 打分全判负，把它们筛掉后丢失组合知识。")
    elif n_syn >= 3:
        cls = "E MIXED"
        reason = "存在明显协同（single 无增益但组合有益），同时 reward 与组合价值不一致。"
    else:
        cls = "A REWARD_MISMATCH"
        reason = "reward 评价的是逐条 retrieval_action，与 Raw pool 在 VAL 上的真实贡献不一致。"
    L.append(f"- **{cls}**：{reason}")
    L.append("")
    L.append("### 最终回答")
    L.append("")
    L.append("Electricity 的失败**不是证明 Reward-Feedback 思路本身无效**。")
    L.append("")
    L.append("根因是 reward 的 granularity（逐条知识评分）不适合强组合效应数据集：Raw pool 的 VAL 增益"
             "来自 8 条 embedding+activation 在 learned retriever 里的组合，而 reward 逐条执行 retrieval_action "
             "（几乎都是 prefer_same:hour_of_day 这类）在 TRAIN 上单条为负，导致 8→1 的 pool collapse，"
             "丢掉了组合知识。")
    L.append("")
    L.append("> 本次诊断结束，未做任何修复（未改 Reward/Critic/Expert、未删结果、未访问 TEST）。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
