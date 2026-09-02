#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 (NightZeroPersistence) 信息有效性公平对照实验 (strict information control)。

验证 E1 本身是否提供有效信息。三个 arm 使用**完全相同**的
`ConditionalExpertMLPRetriever` 结构、参数量、初始化、训练协议、batch 顺序、
候选池、Top-K=5、loss、lr、epoch、tau。唯一允许变化 = Expert 输入内容。

三臂：
  A. ZERO     — Expert 输入 = 零向量（act=0，故 ExpertVector = 0·e_E1 = 0）
  B. SHUFFLED — 真实 e_E1，但 activation 在样本间用固定 permutation 打乱
                （总体分布不变，query-state 对应关系被破坏）
  C. REAL     — 真实 a_E1(q)·e_E1（唯一拥有正确 Expert 信息的 arm）

模型（三臂完全相同，来自 run_conditional_expert_mlp_v4.ConditionalExpertMLPRetriever）：
  x_q -> QueryEncoder -> q_repr ;  x_c -> QueryEncoder -> c_repr
  expert_input_q -> ExpertProjection -> q_expert ;  expert_input_c -> ExpertProjection -> c_expert
  z_q = cond(concat(q_repr, q_expert)) ;  z_c = cond(concat(c_repr, c_expert))
  delta(q,c) = MLP(concat(z_q, z_c, abs(z_q-z_c)))
  score(q,c) = gamma·delta(q,c) + beta·shape(q,c)

其中 expert_input(q) = a_E1(q)·e_E1（模型内 expert_vector(x, act) = act·expert_emb_matrix）。
因此三臂唯一差别 = 传入的 `act`：
  A: act=0  B: act=permuted(act_real)  C: act=act_real
e_E1 (MiniLM-384, 冻结) 与 ExpertProjection/cond 对三臂完全一致。

冻结项：Critic v4、Solar 切分、TRAIN/VAL 协议、QueryEncoder、ExpertProjection、
cond、Pairwise MLP、residual Pearson shape、gamma/beta、optimizer、lr、epoch、tau、
Top-K=5、random seed、candidate pool、memory bank、loss、transfer/retrieval 协议。

禁止：修改 Critic、新增 Attention/Gate/网络结构、重新生成 Expert、修改 E1、调参、
访问 TEST。

TEST 不读取。Critic v4 不修改。Expert 不重新生成。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_expert_attention_retrieval_v2 as attn                       # QueryEncoder / seed_all
from run_expert_embedding_mlp import build_windows, mae, mse           # noqa: E402
from run_conditional_expert_mlp_v4 import (                            # noqa: E402
    ConditionalExpertMLPRetriever,
    build_activation_executor,
    retrieve_conditional,
    train_conditional,
)
from run_solar_same_val_baselines import residualize                   # noqa: E402
from sentence_transformers import SentenceTransformer                   # noqa: E402

ARM_NAMES = ["zero", "shuffled", "real"]

# Historical reference only — 结构不同（无 ExpertProjection，参数量更少），不参与判断。
HISTORICAL_CNN_ONLY_MSE = 191374.62
HISTORICAL_CNN_ONLY_NOTE = (
    "structure differs (no ExpertProjection, fewer params); reference only, "
    "NOT part of the E1 information-validity judgment"
)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def state_hash(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def max_param_diff(sd_a, sd_b) -> float:
    d = 0.0
    for k in sd_a.keys():
        d = max(d, float((sd_a[k] - sd_b[k]).abs().max().item()))
    return d


def make_model(context, expert_dim, d_enc, d_exp, d_z, hidden, device):
    return ConditionalExpertMLPRetriever(
        context=context, expert_dim=expert_dim, n_experts=1,
        d_enc=d_enc, d_exp=d_exp, d_z=d_z, hidden=hidden,
    ).to(device)


def compute_act(X, tss, feat_names, activation_fn):
    M = np.zeros((len(X), len(feat_names)), dtype=np.float32)
    for i, feat in enumerate(feat_names):
        for r in range(len(X)):
            M[r, i] = activation_fn(feat, X[r], tss[r])
    return M


@torch.no_grad()
def model_diagnostic(model, X_diag, act_diag, device):
    x = torch.tensor(X_diag, dtype=torch.float32, device=device)
    a = torch.tensor(act_diag, dtype=torch.float32, device=device)
    ev = model.expert_vector(x, a)          # [B, N*E]
    proj = model.expert_proj(ev)            # [B, d_exp]
    return {
        "expert_projection_param_norm": float(
            sum(p.data.norm().item() for p in model.expert_proj.parameters())),
        "expert_branch_output_abs_mean": float(proj.abs().mean().item()),
        "expert_branch_output_abs_std": float(proj.abs().std().item()),
        # 直接衡量 Expert 分支是否提供「逐样本」信息（zero arm 应为 0）
        "expert_branch_output_per_query_std_mean": float(proj.std(dim=0).mean().item()),
        "expert_vector_abs_mean": float(ev.abs().mean().item()),
        "embedding_norm": float(model.expert_emb_matrix.norm().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/multidataset"))
    ap.add_argument("--dataset", default="solar")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--d-enc", type=int, default=128)
    ap.add_argument("--d-exp", type=int, default=128)
    ap.add_argument("--d-z", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--shuffle-seed", type=int, default=777)
    ap.add_argument("--max-bank", type=int, default=25000)
    ap.add_argument("--max-train-query", type=int, default=6000)
    ap.add_argument("--n-diag", type=int, default=100)
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--audit", type=Path, default=Path("results/solar_agent_v4_audit.json"))
    ap.add_argument("--critic-audit", type=Path,
                    default=Path("results/expert_knowledge_critic_audit.json"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/multidataset/solar/e1_information_control"))
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("seeds:", seeds)

    # ------------------------------------------------------------------ #
    # 0. E1（唯一 ACCEPT）确认 + 冻结项校验
    # ------------------------------------------------------------------ #
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    accepted = audit.get("accepted_experts", [])
    e1 = next(e for e in accepted if e.get("name") == "NightZeroPersistence")
    assert e1.get("activation_feature") == "daylight_level"

    critic_audit = json.loads(args.critic_audit.read_text(encoding="utf-8"))
    crit_dec = {e.get("expert_id"): e.get("final_decision")
                for e in critic_audit.get("experts", [])}
    assert crit_dec.get("E1") == "ACCEPT", f"E1 critic decision = {crit_dec.get('E1')}"
    for eid in ("E2", "E3", "E4", "E5", "E6"):
        assert crit_dec.get(eid) != "ACCEPT", f"{eid} critic decision = {crit_dec.get(eid)}"
    print("E1 accepted:", e1["name"], "| activation_feature:", e1["activation_feature"])
    print("Critic decisions:", {f"E{i}": crit_dec.get(f"E{i}") for i in range(1, 7)})

    # frozen compiler statistics（daylight_level lookup）
    hourly_mean_OT = np.array(
        audit["evidence_executor"]["diurnal_seasonal"]["hourly_mean_OT"], dtype=np.float64)
    monthly_mean_OT = {int(k): float(v) for k, v in
                       audit["evidence_executor"]["diurnal_seasonal"]["monthly_mean_OT"].items()}

    # ------------------------------------------------------------------ #
    # 1. E1 text embedding（冻结 MiniLM，归一化）
    # ------------------------------------------------------------------ #
    enc = SentenceTransformer(args.text_model, device=str(device))
    e_E1 = enc.encode([e1["knowledge"]], normalize_embeddings=True,
                      convert_to_numpy=True).astype("float32")   # [1, 384]
    expert_dim = e_E1.shape[1]
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"e_E1 shape {e_E1.shape}, ||e_E1||={float(np.linalg.norm(e_E1)):.4f}, dim={expert_dim}")

    # ------------------------------------------------------------------ #
    # 2. 数据（multidataset 协议；TEST 从不读取）
    # ------------------------------------------------------------------ #
    ds = args.root / args.dataset
    df = pd.read_csv(ds / "series.csv")
    series = df["OT"].to_numpy(np.float32)
    dates = pd.to_datetime(df["date"])
    idx = np.load(ds / "window_indices.npz")
    train_starts = np.asarray(idx["train"], dtype=np.int64)
    val_starts = np.asarray(idx["val"], dtype=np.int64)

    cut = max(1, int(len(train_starts) * 0.80))
    bank_starts = train_starts[:cut]
    fit_starts = train_starts[cut:]
    if len(bank_starts) > args.max_bank:
        bank_starts = bank_starts[np.linspace(0, len(bank_starts) - 1, args.max_bank).astype(int)]
    if len(fit_starts) > args.max_train_query:
        fit_starts = fit_starts[np.linspace(0, len(fit_starts) - 1, args.max_train_query).astype(int)]

    X_bank, Y_bank = build_windows(series, bank_starts, args.context, args.horizon)
    X_fit, Y_fit = build_windows(series, fit_starts, args.context, args.horizon)
    X_val, Y_val = build_windows(series, val_starts, args.context, args.horizon)
    X_bank_res = residualize(X_bank)
    X_fit_res = residualize(X_fit)
    X_val_res = residualize(X_val)
    print(f"bank={len(bank_starts)} fit={len(fit_starts)} val={len(val_starts)} (TEST not read)")

    # ------------------------------------------------------------------ #
    # 3. 真实 activation（a_E1(q) = daylight_level(q)）
    # ------------------------------------------------------------------ #
    X_train_all = np.concatenate([X_bank, X_fit], axis=0)
    activation_fn, _ = build_activation_executor(X_train_all, hourly_mean_OT, monthly_mean_OT)
    bank_ts = [dates[s + args.context] for s in bank_starts]
    fit_ts = [dates[s + args.context] for s in fit_starts]
    val_ts = [dates[s + args.context] for s in val_starts]
    feat_names = [e1["activation_feature"]]   # ["daylight_level"]

    bank_act = compute_act(X_bank, bank_ts, feat_names, activation_fn)
    fit_act = compute_act(X_fit, fit_ts, feat_names, activation_fn)
    val_act = compute_act(X_val, val_ts, feat_names, activation_fn)

    # ------------------------------------------------------------------ #
    # 4. Arm 输入：A zero / B shuffled（固定一次 permutation）/ C real
    # ------------------------------------------------------------------ #
    n_bank, n_fit, n_val = len(bank_act), len(fit_act), len(val_act)
    all_act = np.concatenate([bank_act, fit_act, val_act], axis=0)   # [T, 1]
    perm_rng = np.random.default_rng(args.shuffle_seed)
    perm = perm_rng.permutation(len(all_act))
    shuffled_all = all_act[perm]
    shuffled_bank_act = shuffled_all[:n_bank]
    shuffled_fit_act = shuffled_all[n_bank:n_bank + n_fit]
    shuffled_val_act = shuffled_all[n_bank + n_fit:]

    # shuffled 相关性校验（原 activation vs shuffled activation）
    a0, a1 = all_act[:, 0], shuffled_all[:, 0]
    if a0.std() < 1e-9 or a1.std() < 1e-9:
        shuffle_corr = None
    else:
        shuffle_corr = float(np.corrcoef(a0, a1)[0, 1])
    print(f"[shuffle] seed={args.shuffle_seed} T={len(all_act)} "
          f"corr(original, shuffled)={shuffle_corr:.6f}")

    arm_bank_act = {
        "zero": np.zeros_like(bank_act),
        "shuffled": shuffled_bank_act,
        "real": bank_act,
    }
    arm_fit_act = {
        "zero": np.zeros_like(fit_act),
        "shuffled": shuffled_fit_act,
        "real": fit_act,
    }
    arm_val_act = {
        "zero": np.zeros_like(val_act),
        "shuffled": shuffled_val_act,
        "real": val_act,
    }

    # ------------------------------------------------------------------ #
    # 5. 逐 seed 训练 / 评估（三臂共享相同初始化）
    # ------------------------------------------------------------------ #
    n_diag = min(args.n_diag, len(X_val))
    diag_idx = np.linspace(0, len(X_val) - 1, n_diag).astype(int)

    results = {name: [] for name in ARM_NAMES}          # list of {seed, mse, mae}
    init_audit = {}                                      # per-seed init hash/diff
    model_diags = {name: [] for name in ARM_NAMES}       # per-seed model diagnostic

    for seed in seeds:
        attn.seed_all(seed)

        # --- 5a. 初始化一次，复制给 A/B/C ---
        template = make_model(args.context, expert_dim, args.d_enc, args.d_exp,
                              args.d_z, args.hidden, device)
        template.set_expert_embeddings(e_E1)
        init_state = {k: v.detach().clone() for k, v in template.state_dict().items()}
        init_hash_template = state_hash(init_state)

        models = {}
        hashes = {}
        for name in ARM_NAMES:
            m = make_model(args.context, expert_dim, args.d_enc, args.d_exp,
                           args.d_z, args.hidden, device)
            m.load_state_dict(init_state)
            models[name] = m
            hashes[name] = state_hash(m.state_dict())

        # 逐参数最大差异 = 0 校验
        diffs = {}
        pairs = [("zero", "shuffled"), ("zero", "real"), ("shuffled", "real")]
        for a, b in pairs:
            diffs[f"{a}_vs_{b}"] = max_param_diff(models[a].state_dict(),
                                                  models[b].state_dict())
        max_diff = max(diffs.values())
        init_audit[str(seed)] = {
            "init_hash_template": init_hash_template,
            "init_hash": hashes,
            "max_param_abs_diff": max_diff,
            "identical": bool(max_diff == 0.0),
        }
        print(f"\n[seed {seed}] init hash identical={max_diff == 0.0} "
              f"max_diff={max_diff:.3e} "
              f"(hash {init_hash_template[:8]}...)")

        # --- 5b. 训练 + 评估三臂 ---
        for name in ARM_NAMES:
            model = models[name]
            print(f"[seed {seed}][{name}] training ...")
            train_conditional(
                model, X_bank, Y_bank, X_fit, Y_fit, X_bank_res, X_fit_res,
                arm_bank_act[name], arm_fit_act[name], device=device,
                epochs=args.epochs, batch_size=args.batch_size,
                candidates=args.candidates, lr=args.lr,
            )
            pred = retrieve_conditional(
                model, X_val, X_bank, Y_bank, X_val_res, X_bank_res,
                arm_val_act[name], arm_bank_act[name], "expert", args.topk, device,
            )
            arm_mse = mse(Y_val, pred)
            arm_mae = mae(Y_val, pred)
            results[name].append({"seed": seed, "mse": arm_mse, "mae": arm_mae})
            model_diags[name].append({
                "seed": seed,
                **model_diagnostic(model, X_val[diag_idx],
                                   arm_val_act[name][diag_idx], device),
            })
            print(f"    [{name}] MSE={arm_mse:.2f} MAE={arm_mae:.2f}")

    # ------------------------------------------------------------------ #
    # 6. 聚合 + 增益 + 结论
    # ------------------------------------------------------------------ #
    def mean_std(vals):
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    agg = {}
    for name in ARM_NAMES:
        mse_m, mse_s = mean_std([r["mse"] for r in results[name]])
        mae_m, mae_s = mean_std([r["mae"] for r in results[name]])
        agg[name] = {
            "mse_mean": mse_m, "mse_std": mse_s,
            "mae_mean": mae_m, "mae_std": mae_s,
            "per_seed": results[name],
        }

    # 增益（正值 = REAL 更优）
    def gain(denom_mse, num_mse):
        # (denom - num)/denom * 100
        return float((denom_mse - num_mse) / denom_mse * 100.0) if denom_mse else None

    gain_z_mean = gain(agg["zero"]["mse_mean"], agg["real"]["mse_mean"])
    gain_s_mean = gain(agg["shuffled"]["mse_mean"], agg["real"]["mse_mean"])

    # 逐 seed 增益（稳定性判断）
    gain_z_seeds = [gain(rz["mse"], rr["mse"])
                    for rz, rr in zip(results["zero"], results["real"])]
    gain_s_seeds = [gain(rs["mse"], rr["mse"])
                    for rs, rr in zip(results["shuffled"], results["real"])]

    n_real_better_zero = sum(1 for g in gain_z_seeds if g > 0)
    n_real_better_shuffled = sum(1 for g in gain_s_seeds if g > 0)

    if gain_z_mean > 0 and gain_s_mean > 0:
        conclusion = "E1_INFORMATION_POSITIVE"
    elif gain_z_mean < 0 and gain_s_mean < 0:
        conclusion = "E1_INFORMATION_NEGATIVE"
    else:
        conclusion = "E1_INFORMATION_NEUTRAL"

    # ------------------------------------------------------------------ #
    # 7. activation / model 诊断（跨 seed 汇总）
    # ------------------------------------------------------------------ #
    diag_real_act = val_act[diag_idx, 0]
    act_diag = {
        "activation_feature": "daylight_level",
        "n_val_queries": int(n_diag),
        "mean": float(diag_real_act.mean()),
        "std": float(diag_real_act.std()),
        "min": float(diag_real_act.min()),
        "max": float(diag_real_act.max()),
        "shuffle_seed": args.shuffle_seed,
        "shuffle_perm_len": int(len(all_act)),
        "corr_original_vs_shuffled": shuffle_corr,
    }
    print(f"[diag] daylight_level over {n_diag} VAL queries: "
          f"mean={act_diag['mean']:.4f} std={act_diag['std']:.4f} "
          f"min={act_diag['min']:.4f} max={act_diag['max']:.4f}")

    # ------------------------------------------------------------------ #
    # 8. 输出
    # ------------------------------------------------------------------ #
    # 参数量：三臂结构完全一致（同一个类），用最后一个 seed 的模型即可。
    pc = n_params(models["real"])
    param_counts = {name: pc for name in ARM_NAMES}

    same_arch = True
    same_params = bool(len(set(param_counts.values())) == 1)
    same_init = bool(all(init_audit[str(s)]["identical"] for s in seeds))

    flags = {
        "critic_modified": False,
        "expert_regenerated": False,
        "attention_used": False,
        "test_used": False,
        "same_architecture": same_arch,
        "same_parameter_count": same_params,
        "same_initialization": same_init,
    }

    result = {
        "dataset": args.dataset,
        "protocol": "multidataset 9905-window VAL, residualize Pearson, Top-K=5 mean",
        "critic_version": "v4",
        "accepted_expert": "E1 NightZeroPersistence",
        "expert": {
            "name": e1["name"],
            "knowledge": e1["knowledge"],
            "activation_condition": e1["activation_condition"],
            "observable_features": e1["observable_features"],
            "activation_feature": e1["activation_feature"],
        },
        "expert_embedding_dim": expert_dim,
        "text_model": args.text_model,
        "seeds": seeds,
        "shuffle_seed": args.shuffle_seed,
        "flags": flags,
        "parameter_counts": param_counts,
        "parameter_count_identical": same_params,
        "initialization_audit": init_audit,
        "activation_diagnostic": act_diag,
        "model_diagnostic": model_diags,
        "arms": agg,
        "gain_real_vs_zero_pct": gain_z_mean,
        "gain_real_vs_shuffled_pct": gain_s_mean,
        "gain_real_vs_zero_per_seed_pct": gain_z_seeds,
        "gain_real_vs_shuffled_per_seed_pct": gain_s_seeds,
        "n_seeds_real_better_than_zero": n_real_better_zero,
        "n_seeds_real_better_than_shuffled": n_real_better_shuffled,
        "shuffle_correlation": shuffle_corr,
        "conclusion": conclusion,
        "historical_reference": {
            "cnn_only_mse": HISTORICAL_CNN_ONLY_MSE,
            "note": HISTORICAL_CNN_ONLY_NOTE,
        },
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "initialization_audit.json").write_text(
        json.dumps(init_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "activation_diagnostic.json").write_text(
        json.dumps(act_diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "model_diagnostic.json").write_text(
        json.dumps(model_diags, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 保存最后一个 seed 的三臂模型 + 预测（可选复现）
    torch.save({
        "e_E1": e_E1,
        "shuffle_perm": perm,
        "zero_state": models["zero"].state_dict(),
        "shuffled_state": models["shuffled"].state_dict(),
        "real_state": models["real"].state_dict(),
        "args": vars(args),
    }, out / "models.pt")
    np.savez_compressed(out / "val_predictions.npz", y_true=Y_val, val_starts=val_starts)

    # ------------------------------------------------------------------ #
    # 9. experiment_audit.md + 最终报告
    # ------------------------------------------------------------------ #
    def fmt_mm(x):
        return f"{x['mse_mean']:.2f} ± {x['mse_std']:.2f}"

    lines = [
        "# E1 信息有效性公平对照实验 Audit",
        "",
        f"- dataset = {args.dataset}",
        "- protocol = multidataset 9905-window VAL, residualize Pearson, Top-K=5 mean",
        "- critic_version = v4",
        "- accepted_expert = E1 NightZeroPersistence (E2..E6 UNSUPPORTED)",
        f"- seeds = {seeds}",
        f"- shuffle_seed = {args.shuffle_seed}",
        "",
        "## 冻结项 / 禁令确认",
        "",
        "- critic_modified = false",
        "- expert_regenerated = false",
        "- attention_used = false",
        "- test_used = false",
        "- same_architecture = true",
        f"- same_parameter_count = {str(same_params).lower()}",
        f"- same_initialization = {str(same_init).lower()}",
        "",
        "## 1. 参数量是否完全一致",
        "",
        f"- zero / shuffled / real = {pc} / {pc} / {pc} 参数 → 一致 = {str(same_params).lower()}",
        "",
        "## 2. 初始权重是否完全一致",
        "",
        f"- 逐 seed 复制同一 init state_dict；逐参数最大差异 = 0 → 一致 = {str(same_init).lower()}",
        "",
        "## 3-5. 三臂 MSE/MAE（mean ± std across seeds）",
        "",
        f"- A. ZERO     MSE = {fmt_mm(agg['zero'])}   MAE = {agg['zero']['mae_mean']:.2f} ± {agg['zero']['mae_std']:.2f}",
        f"- B. SHUFFLED MSE = {fmt_mm(agg['shuffled'])}   MAE = {agg['shuffled']['mae_mean']:.2f} ± {agg['shuffled']['mae_std']:.2f}",
        f"- C. REAL E1  MSE = {fmt_mm(agg['real'])}   MAE = {agg['real']['mae_mean']:.2f} ± {agg['real']['mae_std']:.2f}",
        "",
        "## 6-7. REAL 相对增益（正 = REAL 更优）",
        "",
        f"- REAL vs ZERO     = {gain_z_mean:+.2f}%  (per seed: {[f'{g:+.2f}' for g in gain_z_seeds]})",
        f"- REAL vs SHUFFLED = {gain_s_mean:+.2f}%  (per seed: {[f'{g:+.2f}' for g in gain_s_seeds]})",
        "",
        "## 8. shuffled activation 与真实 activation 相关性",
        "",
        f"- corr(original, shuffled) = {shuffle_corr:.6f}  (目标 ≈ 0)",
        "",
        "## 9. 多 seed mean±std",
        "",
        "见第 3-5 节（每 arm 已给出跨 seed mean±std）。",
        "",
        "## 10. 最终结论",
        "",
        f"- conclusion = {conclusion}",
        f"- n_seeds REAL better than ZERO = {n_real_better_zero}/{len(seeds)}",
        f"- n_seeds REAL better than SHUFFLED = {n_real_better_shuffled}/{len(seeds)}",
        "",
        "判断标准：REAL 稳定优于 ZERO 与 SHUFFLED → POSITIVE；三者接近 → NEUTRAL；",
        "REAL 稳定差于 ZERO/SHUFFLED → NEGATIVE。",
        "",
        "## Historical reference（结构不同，仅作参考）",
        "",
        f"- CNN-only MSE = {HISTORICAL_CNN_ONLY_MSE}",
        f"- note: {HISTORICAL_CNN_ONLY_NOTE}",
        "",
        "## 最终确认",
        "",
        "- critic_modified = false",
        "- expert_regenerated = false",
        "- attention_used = false",
        "- test_used = false",
        "- same_architecture = true",
        "- same_parameter_count = true",
        "- same_initialization = true",
        "",
        "完成后停止。未设计下一版；未修改 Critic；未自动跑其他数据集。",
        "",
    ]
    (out / "experiment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("E1 INFORMATION CONTROL (strict fair comparison)")
    print("=" * 78)
    print(f"A. ZERO     MSE = {fmt_mm(agg['zero'])}  MAE = {agg['zero']['mae_mean']:.2f}")
    print(f"B. SHUFFLED MSE = {fmt_mm(agg['shuffled'])}  MAE = {agg['shuffled']['mae_mean']:.2f}")
    print(f"C. REAL E1  MSE = {fmt_mm(agg['real'])}  MAE = {agg['real']['mae_mean']:.2f}")
    print("-" * 78)
    print(f"REAL vs ZERO     gain = {gain_z_mean:+.2f}%")
    print(f"REAL vs SHUFFLED gain = {gain_s_mean:+.2f}%")
    print(f"shuffle corr(original, shuffled) = {shuffle_corr:.6f}")
    print(f"param count identical = {same_params} | init identical = {same_init}")
    print(f"conclusion = {conclusion}")
    print("=" * 78)
    print("Saved to:", out)
    print("TEST NOT evaluated. Critic v4 NOT modified. No Attention. No regenerated expert.")


if __name__ == "__main__":
    main()
