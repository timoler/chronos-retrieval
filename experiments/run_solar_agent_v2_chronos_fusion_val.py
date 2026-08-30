"""Agent v2 -- Solar Chronos + Memory fusion VAL (frozen pipeline inputs).

This is the FINAL fusion stage for Solar. It reuses, unmodified:

  * the frozen Agent v2 result (proposal / semantic / data-support / compiler /
    proxy validator) already persisted in results/solar_agent_v2_audit.json;
  * the single supported expert prior  ``Short-term irradiance persistence``
    compiled as  Y difference/last/abs_diff  (acts on the target OT only, NEVER
    on P001..P006 as a population proxy);
  * the cached Chronos VAL forecast  results/solar_chronos.npz  (median output,
    same VAL windows as the memory validation);
  * the frozen retrieval math (seasonal_decompose / pearson_similarity /
    _mem_forecast / compute_feature / feature_similarity).

NO LLM is called. NO prompt / critic / threshold / expert fact is changed.
alpha is FROZEN at 0.3 (not re-searched). Only lambda is swept on VAL.

It compares three lines on VAL:

    A  Chronos baseline
    B  Chronos + Shape Memory
    C  Chronos + Shape + Expert Memory   (alpha = 0.3)

with  y_hat = (1 - lambda) * y_chronos + lambda * y_memory.

Memory database = TRAIN only. Query = VAL only. TEST is never accessed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# frozen solar inputs + constants (identical to the memory validation run)
import run_solar_agent_v2_val as prev            # noqa: E402
from priors.executable import (                  # noqa: E402
    ExecutablePrior,
    compute_feature,
    feature_similarity,
)
from priors.validator import CONTEXT, PRED, _mem_forecast  # noqa: E402
from retrieval.decompose import seasonal_decompose          # noqa: E402
from retrieval.similarity import pearson_similarity_matrix  # noqa: E402

PERIOD = prev.PERIOD                # 144
TREND_WINDOW = prev.TREND_WINDOW    # 288
VAL_STRIDE = prev.VAL_STRIDE        # 6
LOAD_VARS = prev.LOAD_VARS          # ["P001".."P006"]
CHUNK = prev.CHUNK                  # 256

ALPHA_EXPERT = 0.3                  # FROZEN from the memory validation (not re-searched)
LAMBDA_GRID = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]


def _mse(p, t):
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(t, dtype=np.float64)) ** 2))


def _mae(p, t):
    return float(np.mean(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(t, dtype=np.float64))))


def load_frozen_prior():
    audit = json.loads(Path("results/solar_agent_v2_audit.json").read_text(encoding="utf-8"))
    pdict = audit["supported_priors"][0]
    prior = ExecutablePrior(**pdict)
    assert prior.input_variables == ["OT"], prior.input_variables
    assert prior.transform == "difference" and prior.temporal_aggregation == "last" \
        and prior.similarity == "abs_diff", (prior.transform, prior.temporal_aggregation, prior.similarity)
    return prior, audit


def build_aligned():
    df = pd.read_csv("data/solar.csv")
    N = len(df)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    val_start, val_end = train_len, train_len + val_len

    series = {"OT": df["OT"].to_numpy(np.float32)}
    for v in LOAD_VARS:
        series[v] = df[v].to_numpy(np.float32)
    ot = series["OT"]

    mem_starts = np.arange(0, train_len - CONTEXT - PRED + 1)
    val_starts = np.arange(val_start, val_end - PRED + 1, VAL_STRIDE)
    mem_ctx = np.stack([ot[m:m + CONTEXT] for m in mem_starts]).astype(np.float32)
    mem_fut = np.stack([ot[m + CONTEXT:m + CONTEXT + PRED] for m in mem_starts]).astype(np.float32)
    val_ctx = np.stack([ot[s - CONTEXT:s] for s in val_starts]).astype(np.float32)
    val_fut = np.stack([ot[s:s + PRED] for s in val_starts]).astype(np.float32)

    # cached Chronos VAL forecast (median), same val_starts
    yc = np.load("results/solar_chronos.npz")["y_chronos_val"].astype(np.float32)
    assert yc.shape == val_fut.shape, f"chronos {yc.shape} vs y_true {val_fut.shape}"
    assert yc.shape[0] == len(val_starts)

    return series, mem_starts, mem_ctx, mem_fut, val_starts, val_ctx, val_fut, yc


def compute_memory_preds(prior, series, mem_starts, mem_ctx, mem_fut, val_starts, val_ctx, val_fut):
    """Recompute the two memory forecasts (Shape; Shape+Expert @ alpha=0.3).

    Identical math to the memory validation; returns full (n_val, PRED) arrays
    in float64 so the MSEs reproduce 281798.5357 / 256712.9541.
    """
    _, _, Rm, _, _ = seasonal_decompose(mem_ctx, PRED, PERIOD, TREND_WINDOW)
    mem_ctx_mean = mem_ctx.mean(axis=1)
    val_ctx_mean = val_ctx.mean(axis=1)
    n_val = len(val_ctx)

    mf, qf, _ = compute_feature(prior, series, mem_starts, val_starts, np.zeros(2))

    shape_pred = np.empty((n_val, PRED), dtype=np.float64)
    expert_pred = np.empty((n_val, PRED), dtype=np.float64)

    for q0 in range(0, n_val, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, n_val))
        _, _, Rv, _, _ = seasonal_decompose(val_ctx[sl], PRED, PERIOD, TREND_WINDOW)
        S_shape = pearson_similarity_matrix(Rv, Rm)
        shape_pred[sl] = _mem_forecast(S_shape, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
        S_prior = feature_similarity(prior, qf[sl], mf)
        S = (1.0 - ALPHA_EXPERT) * S_shape + ALPHA_EXPERT * S_prior
        expert_pred[sl] = _mem_forecast(S, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
    return shape_pred, expert_pred


def main():
    prior, audit = load_frozen_prior()
    series, mem_starts, mem_ctx, mem_fut, val_starts, val_ctx, val_fut, yc = build_aligned()

    shape_pred, expert_pred = compute_memory_preds(
        prior, series, mem_starts, mem_ctx, mem_fut, val_starts, val_ctx, val_fut
    )

    y_true = val_fut.astype(np.float64)
    chronos = yc.astype(np.float64)

    # ---- standalone metrics (verify memory MSEs reproduce the frozen run) ----
    chronos_mse, chronos_mae = _mse(chronos, y_true), _mae(chronos, y_true)
    shape_mse, shape_mae = _mse(shape_pred, y_true), _mae(shape_pred, y_true)
    expert_mse, expert_mae = _mse(expert_pred, y_true), _mae(expert_pred, y_true)

    # tolerance check against the frozen memory validation numbers
    shape_ref, expert_ref = 281798.5357, 256712.9541
    assert abs(shape_mse - shape_ref) < 1.0, f"shape_mse {shape_mse:.4f} != {shape_ref} (mismatch)"
    assert abs(expert_mse - expert_ref) < 1.0, f"expert_mse {expert_mse:.4f} != {expert_ref} (mismatch)"

    # ---- lambda sweep (VAL only) ----
    shape_fusion = []   # (lambda, mse, mae)
    expert_fusion = []  # (lambda, mse, mae)
    for lam in LAMBDA_GRID:
        b = (1.0 - lam) * chronos + lam * shape_pred
        c = (1.0 - lam) * chronos + lam * expert_pred
        shape_fusion.append((lam, _mse(b, y_true), _mae(b, y_true)))
        expert_fusion.append((lam, _mse(c, y_true), _mae(c, y_true)))

    best_lambda_shape, best_shape_mse, best_shape_mae = min(shape_fusion, key=lambda r: r[1])
    best_lambda_expert, best_expert_mse, best_expert_mae = min(expert_fusion, key=lambda r: r[1])

    best_shape_fusion_pred = (1.0 - best_lambda_shape) * chronos + best_lambda_shape * shape_pred
    best_expert_fusion_pred = (1.0 - best_lambda_expert) * chronos + best_lambda_expert * expert_pred

    imp_shape_vs_chronos = (chronos_mse - best_shape_mse) / chronos_mse * 100.0
    imp_expert_vs_chronos = (chronos_mse - best_expert_mse) / chronos_mse * 100.0
    imp_expert_vs_shape = (best_shape_mse - best_expert_mse) / best_shape_mse * 100.0

    # ---- same-lambda ablation ----
    same_lambda = []
    for (lam, b_mse, b_mae), (_, c_mse, c_mae) in zip(shape_fusion, expert_fusion):
        same_lambda.append({
            "lambda": lam,
            "shape_mse": b_mse, "expert_mse": c_mse,
            "shape_mae": b_mae, "expert_mae": c_mae,
            "expert_rel_improvement_pct": (b_mse - c_mse) / b_mse * 100.0,
        })

    # ---- positive-case judgments ----
    memory_positive = expert_mse < shape_mse   # already established (alpha=0.3)
    full_positive = (
        best_expert_mse < chronos_mse
        and best_expert_mse < best_shape_mse
        and best_lambda_expert > 0
    )

    # ---- persist predictions ----
    Path("results/solar_agent_v2_chronos_fusion_val_predictions.npz")
    np.savez_compressed(
        "results/solar_agent_v2_chronos_fusion_val_predictions.npz",
        y_true=y_true.astype(np.float32),
        chronos_pred=chronos.astype(np.float32),
        shape_memory_pred=shape_pred.astype(np.float32),
        expert_memory_pred=expert_pred.astype(np.float32),
        best_shape_fusion_pred=best_shape_fusion_pred.astype(np.float32),
        best_expert_fusion_pred=best_expert_fusion_pred.astype(np.float32),
        best_shape_lambda=np.float32(best_lambda_shape),
        best_expert_lambda=np.float32(best_lambda_expert),
        alpha=np.float32(ALPHA_EXPERT),
    )

    out = {
        "dataset": "solar",
        "agent_version": "v2",
        "experiment": "chronos_memory_fusion_val",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "alpha": ALPHA_EXPERT,
        "lambda_grid": LAMBDA_GRID,
        "frozen_prior": {
            "name": prior.name,
            "input_variables": prior.input_variables,
            "transform": prior.transform,
            "temporal_aggregation": prior.temporal_aggregation,
            "similarity": prior.similarity,
        },
        "standalone": {
            "chronos": {"mse": chronos_mse, "mae": chronos_mae},
            "shape_memory": {"mse": shape_mse, "mae": shape_mae,
                             "reference_frozen_mse": shape_ref},
            "expert_memory_alpha_03": {"mse": expert_mse, "mae": expert_mae,
                                       "reference_frozen_mse": expert_ref},
        },
        "chronos_plus_shape": {
            "sweep": [{"lambda": l, "mse": m, "mae": a} for l, m, a in shape_fusion],
            "best_lambda": best_lambda_shape,
            "best_mse": best_shape_mse,
            "best_mae": best_shape_mae,
            "improvement_vs_chronos_pct": imp_shape_vs_chronos,
        },
        "chronos_plus_shape_plus_expert": {
            "alpha": ALPHA_EXPERT,
            "sweep": [{"lambda": l, "mse": m, "mae": a} for l, m, a in expert_fusion],
            "best_lambda": best_lambda_expert,
            "best_mse": best_expert_mse,
            "best_mae": best_expert_mae,
            "improvement_vs_chronos_pct": imp_expert_vs_chronos,
            "improvement_vs_best_shape_pct": imp_expert_vs_shape,
        },
        "same_lambda_ablation": same_lambda,
        "memory_positive_case": memory_positive,
        "full_positive_case": full_positive,
        "selected_val_candidate_for_future_test": {
            "method": "chronos + shape + expert memory",
            "alpha": ALPHA_EXPERT,
            "lambda": best_lambda_expert,
            "val_mse": best_expert_mse,
            "val_mae": best_expert_mae,
        },
        "test_access": "NO (TRAIN memory + VAL query only; TEST never read)",
        "llm_called": "NO (all pipeline stages reused from frozen cache)",
        "agent_modified": "NO (prompt/critic/threshold/expert-fact/flow untouched)",
        "model_trained": "NO",
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/solar_agent_v2_chronos_fusion_val.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    Path("results/solar_agent_v2_chronos_fusion_val.md").write_text(
        build_markdown(out), encoding="utf-8")

    # ---- console report ----
    print("=" * 100)
    print("SOLAR Chronos + Memory fusion VAL (frozen pipeline, alpha=0.3)")
    print("=" * 100)
    print(f"[standalone] Chronos             MSE={chronos_mse:.4f}  MAE={chronos_mae:.4f}")
    print(f"[standalone] Shape Memory        MSE={shape_mse:.4f}  MAE={shape_mae:.4f}   (frozen ref {shape_ref})")
    print(f"[standalone] Expert Mem (a=0.3)  MSE={expert_mse:.4f}  MAE={expert_mae:.4f}   (frozen ref {expert_ref})")
    print()
    print("Chronos + Shape sweep:")
    for l, m, a in shape_fusion:
        print(f"    lambda={l:<6} MSE={m:.4f}  MAE={a:.4f}")
    print(f"  best_lambda_shape={best_lambda_shape}  best_mse={best_shape_mse:.4f}  "
          f"vs_chronos={imp_shape_vs_chronos:+.4f}%")
    print()
    print("Chronos + Shape + Expert sweep (alpha=0.3):")
    for l, m, a in expert_fusion:
        print(f"    lambda={l:<6} MSE={m:.4f}  MAE={a:.4f}")
    print(f"  best_lambda_expert={best_lambda_expert}  best_mse={best_expert_mse:.4f}  "
          f"vs_chronos={imp_expert_vs_chronos:+.4f}%  vs_best_shape={imp_expert_vs_shape:+.4f}%")
    print()
    print("Same-lambda ablation (expert vs shape):")
    for r in same_lambda:
        print(f"    lambda={r['lambda']:<6} shape={r['shape_mse']:.4f}  "
              f"expert={r['expert_mse']:.4f}  expert_imp={r['expert_rel_improvement_pct']:+.4f}%")
    print()
    print(f"SOLAR MEMORY POSITIVE CASE: {'YES' if memory_positive else 'NO'}")
    print(f"SOLAR FULL POSITIVE CASE:   {'YES' if full_positive else 'NO'}")
    print()
    print("Saved results/solar_agent_v2_chronos_fusion_val.json")
    print("Saved results/solar_agent_v2_chronos_fusion_val.md")
    print("Saved results/solar_agent_v2_chronos_fusion_val_predictions.npz")
    return out


def build_markdown(out):
    L = []
    L.append("# Solar Chronos + Memory 融合 VAL 报告（冻结管线）\n")
    L.append(f"> 时间：{out['timestamp']}\n")
    L.append("> alpha 冻结 = 0.3（不重新搜索）。仅在 VAL 上搜索 lambda。Chronos 未重新推理（读缓存）。TEST 未访问。\n")

    fp = out["frozen_prior"]
    L.append("\n## 冻结的 Expert Prior\n")
    L.append(f"- name: `{fp['name']}`")
    L.append(f"- 编译: `{fp['transform']}/{fp['temporal_aggregation']}/{fp['similarity']}` on `{'/'.join(fp['input_variables'])}`（仅作用于目标 OT，未用 P001..P006）")

    L.append("\n## 1. Standalone（无融合）\n")
    s = out["standalone"]
    L.append("| 方法 | MSE | MAE |")
    L.append("|---|---|---|")
    L.append(f"| Chronos baseline | {s['chronos']['mse']:.4f} | {s['chronos']['mae']:.4f} |")
    L.append(f"| Shape Memory | {s['shape_memory']['mse']:.4f} | {s['shape_memory']['mae']:.4f} |")
    L.append(f"| Shape+Expert Memory (alpha=0.3) | {s['expert_memory_alpha_03']['mse']:.4f} | {s['expert_memory_alpha_03']['mae']:.4f} |")
    L.append("\nMemory MSE 复现校验：")
    L.append(f"- Shape Memory: {s['shape_memory']['mse']:.4f} ≈ {s['shape_memory']['reference_frozen_mse']} ✓")
    L.append(f"- Expert Memory: {s['expert_memory_alpha_03']['mse']:.4f} ≈ {s['expert_memory_alpha_03']['reference_frozen_mse']} ✓")

    L.append("\n## 2. Chronos + Shape Memory\n")
    b = out["chronos_plus_shape"]
    L.append("| lambda | MSE | MAE |")
    L.append("|---|---|---|")
    for r in b["sweep"]:
        L.append(f"| {r['lambda']} | {r['mse']:.4f} | {r['mae']:.4f} |")
    L.append(f"\n- best_lambda_shape = **{b['best_lambda']}**")
    L.append(f"- best_mse_shape = **{b['best_mse']:.4f}**")
    L.append(f"- 相对 Chronos 提升 = **{b['improvement_vs_chronos_pct']:+.4f}%**")

    L.append("\n## 3. Chronos + Shape + Expert Memory（alpha=0.3）\n")
    c = out["chronos_plus_shape_plus_expert"]
    L.append("| lambda | MSE | MAE |")
    L.append("|---|---|---|")
    for r in c["sweep"]:
        L.append(f"| {r['lambda']} | {r['mse']:.4f} | {r['mae']:.4f} |")
    L.append(f"\n- best_lambda_expert = **{c['best_lambda']}**")
    L.append(f"- best_mse_expert = **{c['best_mse']:.4f}**")
    L.append(f"- 相对 Chronos 提升 = **{c['improvement_vs_chronos_pct']:+.4f}%**")
    L.append(f"- 相对最佳 Chronos+Shape 提升 = **{c['improvement_vs_best_shape_pct']:+.4f}%**")

    L.append("\n## 4. Same-lambda 对照（Chronos+Shape vs Chronos+Shape+Expert）\n")
    L.append("| lambda | Shape MSE | Expert MSE | Expert 相对 Shape 改善 |")
    L.append("|---|---|---|---|")
    for r in out["same_lambda_ablation"]:
        L.append(f"| {r['lambda']} | {r['shape_mse']:.4f} | {r['expert_mse']:.4f} | {r['expert_rel_improvement_pct']:+.4f}% |")

    L.append("\n## 5. Positive Case 判定\n")
    L.append(f"- **SOLAR MEMORY POSITIVE CASE: {'YES' if out['memory_positive_case'] else 'NO'}**")
    L.append(f"- **SOLAR FULL POSITIVE CASE:   {'YES' if out['full_positive_case'] else 'NO'}**")
    L.append("\n判定条件（FULL）：best(Chronos+Shape+Expert) < Chronos baseline；"
             "best(Chronos+Shape+Expert) < best(Chronos+Shape)；best_lambda_expert > 0。")

    L.append("\n## 6. selected_val_candidate_for_future_test\n")
    sc = out["selected_val_candidate_for_future_test"]
    L.append(f"- method: `{sc['method']}`")
    L.append(f"- alpha: {sc['alpha']}")
    L.append(f"- lambda: {sc['lambda']}")
    L.append(f"- VAL MSE: {sc['val_mse']:.4f}  | VAL MAE: {sc['val_mae']:.4f}")

    L.append("\n## 7. 合规检查\n")
    L.append(f"- 访问 TEST：{out['test_access']}")
    L.append(f"- 重新调用 LLM：{out['llm_called']}")
    L.append(f"- 修改 Agent/Prompt/Critic/阈值：{out['agent_modified']}")
    L.append(f"- 训练模型：{out['model_trained']}")
    L.append("- alpha 未重新搜索（固定 0.3）；lambda 仅在 VAL 上搜索；Chronos 读缓存未重推理。")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
