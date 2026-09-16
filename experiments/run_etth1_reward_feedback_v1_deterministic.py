#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETTh1 Reward-Feedback Critic v1 — 严格 deterministic 正式 3-seed ablation。

只修复实验复现协议（随机种子 + deterministic 配置），不改方法本身：
  - 不改 Expert/Final Pool / Reward / Critic prompt / Compiler / retrieval_action
  - 不改 Retriever 架构 / 数据划分 / epochs / optimizer / batch / Top-K / 任何超参数
  - 不访问 TEST

协议改动（唯一允许的）：
  1) seed_everything(seed) 在每个 method、每个 seed 训练前都调用，
     设 CUBLAS_WORKSPACE_CONFIG / cudnn.deterministic / benchmark=False /
     use_deterministic_algorithms(True)。
  2) train_arm 的 numpy batch/candidate 采样 rng 由 seed 参数化（原硬编码 12345）。
  3) 记录每个 (method, seed) 的 init_hash。

输出：results/reward_feedback_critic_v1_etth1_deterministic/
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import csv
import hashlib
import io
import json
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
from run_conditional_expert_mlp_v4 import (
    ConditionalExpertMLPRetriever,
    retrieve_conditional,
)
from run_etth1_prediction_validation import compute_training_stats

import run_etth1_reward_feedback_critic_v1 as v1

CONTEXT, HORIZON, TOPK = v1.CONTEXT, v1.HORIZON, v1.TOPK
EPOCHS, BATCH_SIZE, CANDIDATES, LR = v1.EPOCHS, v1.BATCH_SIZE, v1.CANDIDATES, v1.LR
D_ENC, D_EXP, D_Z, HIDDEN = v1.D_ENC, v1.D_EXP, v1.D_Z, v1.HIDDEN
DATA_DIR = v1.DATA_DIR
embed_claims = v1.embed_claims
act_mats = v1.act_mats
n_params = v1.n_params
_sorted_experts = v1._sorted_experts

REPO_ROOT = HERE.parent
OUT_DIR = REPO_ROOT / "results/reward_feedback_critic_v1_etth1_deterministic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
ARMS = [("base", "A +BASE"), ("shape", "B +SHAPE"),
        ("raw_expert", "C +RAW EXPERT"), ("reward_feedback_expert", "D +REWARD-FEEDBACK v1")]


# --------------------------------------------------------------------------- #
# deterministic seeding
# --------------------------------------------------------------------------- #
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


def model_init_hash(state_dict):
    h = hashlib.sha256()
    for p in state_dict.values():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# train_arm（与 v1.train_arm 完全一致，唯一差异：rng 由 seed 参数化）
# --------------------------------------------------------------------------- #
def train_arm(model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
              bank_act, fit_act, use_shape, device, seed):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    L = X_bank.shape[1]
    Yb = torch.tensor(Y_bank, dtype=torch.float32, device=device)
    bank_act_t = torch.tensor(bank_act, dtype=torch.float32, device=device)
    n = len(X_fit)
    rng = np.random.default_rng(seed)          # 原 12345 -> 现由 seed 控制
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
    seed_everything(seed)                      # 每个 method、每个 seed 训练前重新调用
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


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-determinism-check", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| epochs:", EPOCHS, "| seeds:", SEEDS)

    # ---- 数据（与 v1 main() 一致；TEST 不加载）----
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

    # ---- 冻结 pool（不修改）----
    final_pool = _sorted_experts(json.loads(
        (v1.OUT_DIR / "final_expert_pool.json").read_text(encoding="utf-8")))
    raw_sorted = _sorted_experts(json.loads(
        (v1.OUT_DIR / "raw_experts.json").read_text(encoding="utf-8")))
    print(f"final pool: {len(final_pool)} experts -> {[e['id'] for e in final_pool]}")
    print(f"raw experts: {len(raw_sorted)}")

    # ---- embeddings / activations（确定性数据）----
    print("Embedding claims ...")
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
    zero_act = {k: np.zeros_like(raw_act[k]) for k in raw_act}

    configs = {
        "base": (emb_zero, zero_act, False),
        "shape": (emb_zero, zero_act, True),
        "raw_expert": (emb_raw, raw_act, True),
        "reward_feedback_expert": (emb_final, final_act, True),
    }

    # ---- 1. 复现 sanity check：D seed2026 连跑 3 次 ----
    det_rows = []
    if not args.skip_determinism_check:
        print("\n" + "=" * 78)
        print("DETERMINISM CHECK: D (+REWARD-FEEDBACK v1) seed2026 x3")
        print("=" * 78)
        emb_d, act_d, _ = configs["reward_feedback_expert"]
        for run_id in range(3):
            seed_everything(2026)
            base_model = ConditionalExpertMLPRetriever(
                context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
                d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
            init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
            ih = model_init_hash(init_state)
            mm, aa = run_arm(init_state, emb_d, act_d, True, expert_dim, n_slots, device, 2026)
            det_rows.append({"run_id": run_id, "seed": 2026, "method": "reward_feedback_expert",
                             "init_hash": ih, "mse": mm, "mae": aa})
            print(f"  run {run_id}: init_hash={ih} MSE={mm:.6f} MAE={aa:.6f}")

        ih_same = len({r["init_hash"] for r in det_rows}) == 1
        mse_span = max(r["mse"] for r in det_rows) - min(r["mse"] for r in det_rows)
        mae_span = max(r["mae"] for r in det_rows) - min(r["mae"] for r in det_rows)
        print(f"  -> init_hash identical = {ih_same}; MSE span = {mse_span:.2e}; MAE span = {mae_span:.2e}")
        if not (ih_same and mse_span < 1e-4 and mae_span < 1e-4):
            print("  [STOP] determinism check FAILED: 结果不一致，停止正式实验。")
            _write_csv(OUT_DIR / "determinism_check.csv", det_rows,
                       ["run_id", "seed", "method", "init_hash", "mse", "mae"])
            return
        print("  [PASS] determinism check passed（三次 init_hash / MSE / MAE 一致）。")

    # ---- 2. 正式 3-seed ablation ----
    print("\n" + "=" * 78)
    print("FORMAL 3-SEED ABLATION (A/B/C/D, deterministic)")
    print("=" * 78)
    rows = []          # formal_val_ablation
    init_rows = []     # init_hashes
    per_seed = {name: {} for name, _ in ARMS}
    param_count = None
    for seed in SEEDS:
        seed_everything(seed)
        base_model = ConditionalExpertMLPRetriever(
            context=CONTEXT, expert_dim=expert_dim, n_experts=n_slots,
            d_enc=D_ENC, d_exp=D_EXP, d_z=D_Z, hidden=HIDDEN).to(device)
        init_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
        ih = model_init_hash(init_state)
        param_count = n_params(base_model)
        print(f"\nSEED {seed}  init_hash={ih}  param_count={param_count}")
        for name, label in ARMS:
            emb, act, use_shape = configs[name]
            mm, aa = run_arm(init_state, emb, act, use_shape, expert_dim, n_slots, device, seed)
            rows.append({"method": name, "seed": seed, "init_hash": ih, "mse": mm, "mae": aa})
            init_rows.append({"method": name, "seed": seed, "init_hash": ih})
            per_seed[name][seed] = mm
            print(f"  [{label:<24}] MSE={mm:.6f} MAE={aa:.6f}")

    # ---- 3. 汇总 ----
    summary_rows = []
    for name, label in ARMS:
        ms = [per_seed[name][s] for s in SEEDS]
        mae_by_seed = {r["seed"]: r["mae"] for r in rows if r["method"] == name}
        summary_rows.append({
            "method": name,
            "mean_mse": float(np.mean(ms)),
            "std_mse": float(np.std(ms)),
            "mean_mae": float(np.mean([mae_by_seed[s] for s in SEEDS])),
            "std_mae": float(np.std([mae_by_seed[s] for s in SEEDS])),
            "seed42_mse": per_seed[name][42],
            "seed123_mse": per_seed[name][123],
            "seed2026_mse": per_seed[name][2026],
        })

    # ---- 4. 写 CSV ----
    _write_csv(OUT_DIR / "determinism_check.csv", det_rows,
               ["run_id", "seed", "method", "init_hash", "mse", "mae"])
    _write_csv(OUT_DIR / "formal_val_ablation.csv", rows,
               ["method", "seed", "init_hash", "mse", "mae"])
    _write_csv(OUT_DIR / "formal_val_summary.csv", summary_rows,
               ["method", "mean_mse", "std_mse", "mean_mae", "std_mae",
                "seed42_mse", "seed123_mse", "seed2026_mse"])
    _write_csv(OUT_DIR / "init_hashes.csv", init_rows, ["method", "seed", "init_hash"])

    # ---- 5. audit + report ----
    report, audit = render(summary_rows, per_seed, rows, det_rows, param_count, n_slots)
    (OUT_DIR / "deterministic_audit.md").write_text(audit, encoding="utf-8")
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 78)
    print("FORMAL ABLATION (mean ± std over seeds)")
    print("=" * 78)
    for name, label in ARMS:
        s = next(r for r in summary_rows if r["method"] == name)
        print(f"{label:<24} MSE={s['mean_mse']:.4f} ± {s['std_mse']:.4f}  MAE={s['mean_mae']:.4f} ± {s['std_mae']:.4f}")
    print(f"\nSaved to: {OUT_DIR}")
    print("TEST accessed = False")


def _write_csv(path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _get(summary_rows, name, key):
    return next(r[key] for r in summary_rows if r["method"] == name)


def render(summary_rows, per_seed, rows, det_rows, param_count, n_slots):
    a_m = _get(summary_rows, "base", "mean_mse")
    b_m = _get(summary_rows, "shape", "mean_mse")
    c_m = _get(summary_rows, "raw_expert", "mean_mse")
    d_m = _get(summary_rows, "reward_feedback_expert", "mean_mse")
    d_vs_c = (c_m - d_m) / c_m * 100.0
    d_vs_b = (b_m - d_m) / b_m * 100.0
    d_vs_a = (a_m - d_m) / a_m * 100.0
    c_vs_b = (b_m - c_m) / b_m * 100.0
    d_std = _get(summary_rows, "reward_feedback_expert", "std_mse")
    c_std = _get(summary_rows, "raw_expert", "std_mse")

    all_d_better = all(per_seed["reward_feedback_expert"][s] < per_seed["raw_expert"][s] for s in SEEDS)
    if d_m < c_m and all_d_better:
        concl = "SUCCESS"
    elif d_m < c_m:
        concl = "PARTIAL"
    else:
        concl = "FAIL"

    # ---- audit ----
    audit = [
        "# deterministic audit — ETTh1 Reward-Feedback Critic v1 正式重跑",
        "",
        "## 协议改动（唯一允许项：随机种子 + deterministic 配置）",
        "",
        "- `seed_everything(seed)` 在每个 method、每个 seed 训练前调用：",
        "  `PYTHONHASHSEED` / `CUBLAS_WORKSPACE_CONFIG=:4096:8` / `random` / `numpy` /",
        "  `torch.manual_seed` / `torch.cuda.manual_seed(_all)` / `cudnn.deterministic=True` /",
        "  `cudnn.benchmark=False` / `use_deterministic_algorithms(True)`。",
        "- `train_arm` 的 batch permutation / candidate sampling 使用 `np.random.default_rng(seed)`",
        "  （原硬编码 `12345`，现由 seed 参数化）。",
        "- **DataLoader N/A**：本实验 train_arm 使用手动 numpy batch 迭代（`rng.permutation` + `rng.choice`），",
        "  不经过 `DataLoader`，故无 `shuffle` / `worker_init_fn` / `torch.Generator` 需要固定。",
        "- 未改 Expert/Final Pool / Reward / Critic prompt / Compiler / retrieval_action / 架构 / 数据划分 /",
        "  epochs / optimizer / batch size / Top-K / 超参数；未访问 TEST。",
        "- 每个 seed 的 4 个 method 共享同一 `init_state`（same initialization policy）。",
        f"- `param_count` = {param_count}，`slots` = {n_slots}，`seeds` = {SEEDS}。",
        "",
        "## determinism check",
        "",
    ]
    if det_rows:
        ih_same = len({r["init_hash"] for r in det_rows}) == 1
        mse_span = max(r["mse"] for r in det_rows) - min(r["mse"] for r in det_rows)
        audit.append(f"- D seed2026 连跑 3 次：init_hash 一致 = {ih_same}，MSE 极差 = {mse_span:.2e}。")
        audit.append(f"- 判定：{'PASS（结果可复现）' if (ih_same and mse_span < 1e-4) else 'FAIL'}。")
    else:
        audit.append("- （跳过 determinism check）")
    audit.append("")

    # ---- report ----
    L = []
    L.append("# ETTh1 — Reward-Feedback Critic v1 正式结果（严格 deterministic）")
    L.append("")
    L.append("旧 nondeterministic 数字只作探索参考；本报告为严格 deterministic 协议下的正式结果。")
    L.append("")

    L.append("## 结果")
    L.append("")
    # determinism
    if det_rows:
        ih_same = len({r["init_hash"] for r in det_rows}) == 1
        mse_span = max(r["mse"] for r in det_rows) - min(r["mse"] for r in det_rows)
        L.append(f"- seed2026 连跑 3 次：init_hash 一致 = {ih_same}，MSE 极差 = {mse_span:.2e} → "
                 f"{'一致（deterministic 问题已修复）' if (ih_same and mse_span < 1e-4) else '不一致'}。")
    L.append("")

    L.append("A/B/C/D 正式 3-seed MSE：")
    L.append("")
    L.append("| method | seed42 | seed123 | seed2026 | mean MSE | std MSE | mean MAE | std MAE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, label in ARMS:
        s = next(r for r in summary_rows if r["method"] == name)
        L.append(f"| {label} | {s['seed42_mse']:.4f} | {s['seed123_mse']:.4f} | {s['seed2026_mse']:.4f} "
                 f"| {s['mean_mse']:.4f} | {s['std_mse']:.4f} | {s['mean_mae']:.4f} | {s['std_mae']:.4f} |")
    L.append("")

    L.append("- D vs C = " + f"{d_vs_c:+.2f}%" + "（正 = D 优于 C）")
    L.append("- D vs B = " + f"{d_vs_b:+.2f}%")
    L.append("- D vs A = " + f"{d_vs_a:+.2f}%")
    L.append("- C vs B = " + f"{c_vs_b:+.2f}%")
    L.append(f"- D 是否 3/3 seeds 都优于 C：{'是' if all_d_better else '否'}")
    L.append(f"- D 的 seed 间 std = {d_std:.4f}，C 的 std = {c_std:.4f}")
    L.append("")

    L.append("区分：")
    L.append("- 同 seed 重跑不一致 = **nondeterminism**（本报告已通过 determinism check 排除）。")
    L.append("- 不同 seed 结果不同 = **initialization sensitivity**（由 seed 间 std 与 3/3 一致性刻画）。")
    L.append("")

    L.append("## 结论")
    L.append("")
    L.append(f"- **{concl}**")
    if concl == "SUCCESS":
        L.append("  （D 平均优于 C，且同 seed 可复现、3/3 seeds 均优于 C。）")
    elif concl == "PARTIAL":
        L.append("  （D 平均优于 C，但存在 seed sensitivity：非 3/3 seeds 均优于 C。）")
    else:
        L.append("  （D 平均不优于 C。）")
    L.append("")
    L.append("```")
    L.append("test_used = false")
    L.append("deterministic_protocol = true")
    L.append(f"param_count = {param_count}")
    L.append("```")

    return "\n".join(L), "\n".join(audit)


if __name__ == "__main__":
    main()
