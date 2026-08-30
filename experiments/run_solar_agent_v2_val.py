"""Agent v2 -- Solar-Energy VAL experiment (FROZEN pipeline, memory-only).

Same FROZEN Agent v2 as ``run_agent_v2_solar.py`` (metadata / 6 expert facts /
prompt builders / RULES_BLOCK / whitelist / stage orchestration unchanged), but
this run answers ONE question only:

    Does the Agent's auto-generated Expert Retrieval Prior add incremental value
    to Shape Memory Retrieval (the shape-residual Pearson baseline)?

Therefore the retrieval validation here compares, on VAL only:

    Shape Memory Retrieval          (alpha = 0)
    vs
    Shape + Expert Memory Retrieval (alpha in {0.1, 0.2, 0.3})

with the combination  S = (1 - alpha) * S_shape + alpha * S_expert  and the
memory bank / Top-K / mean-anchor transfer / softmax aggregation held exactly
as in the frozen pipeline.

Chronos is NOT run and NOT used for fusion in this experiment. TEST is never
accessed (TRAIN memory + VAL query only). No prompt / critic / threshold /
alpha-grid-beyond-the-task / variable selection is modified.
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

# Frozen solar inputs (metadata, 6 expert facts, column mapping, TRAIN evidence)
import run_agent_v2_solar as solar            # noqa: E402
import run_agent_v2_electricity as av2        # frozen Agent v2 module         noqa: E402
from run_agent_v2_electricity import run_pipeline  # noqa: E402

from priors.executable import (               # noqa: E402
    ExecutablePrior,
    compute_feature,
    feature_similarity,
)
from priors.validator import CONTEXT, PRED, _mem_forecast  # noqa: E402
from retrieval.decompose import seasonal_decompose          # noqa: E402
from retrieval.similarity import pearson_similarity_matrix  # noqa: E402

# --------------------------------------------------------------------------- #
# freeze the frozen module's dataset-specific globals (identical to solar.main)
# --------------------------------------------------------------------------- #
av2.ELECTRICITY_METADATA_V2 = solar.SOLAR_METADATA_V2
av2.ELECTRICITY_EXPERT_MECHANISM = solar.SOLAR_EXPERT_MECHANISM
av2.BLIND_TO_REAL = solar.BLIND_TO_REAL
av2.LOAD_VARS = solar.LOAD_VARS
av2.N_POPULATION = solar.N_POPULATION
av2.compute_train_evidence = solar.solar_compute_train_evidence
av2.CACHE_DIR = "cache/llm_priors_v2_solar"

# dataset-specific retrieval constants (10-min data -> daily period 144)
PERIOD = solar.PERIOD                # 144
TREND_WINDOW = solar.TREND_WINDOW    # 288
VAL_STRIDE = solar.VAL_STRIDE        # 6 (~hourly VAL queries)
LOAD_VARS = solar.LOAD_VARS          # ["P001".."P006"]
CHUNK = solar.CHUNK                  # 256

# memory-only alpha grid (narrow, per task; NOT the frozen pipeline's ALPHAS)
ALPHAS_MEM = [0.0, 0.1, 0.2, 0.3]


# --------------------------------------------------------------------------- #
# memory-only retrieval validation (no Chronos)
# --------------------------------------------------------------------------- #
def run_retrieval_validation_memory_only(exec_priors):
    """VAL sweep: Shape Memory (alpha=0) vs Shape+Expert Memory (alpha>0).

    Identical memory/query/decomposition math to the frozen pipeline, minus the
    Chronos fusion. alpha selected to minimize MEMORY MSE (not fused MSE).
    """
    np.random.seed(0)
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

    _, _, Rm, _, _ = seasonal_decompose(mem_ctx, PRED, PERIOD, TREND_WINDOW)
    mem_ctx_mean = mem_ctx.mean(axis=1)
    val_ctx_mean = val_ctx.mean(axis=1)
    n_val = len(val_ctx)
    n_pts = n_val * PRED
    nonzero = [a for a in ALPHAS_MEM if a != 0.0]

    pf = {}
    for p in exec_priors:
        mf, qf, _ = compute_feature(p, series, mem_starts, val_starts, np.zeros(2))
        pf[p.name] = (p, mf, qf)

    shape_mem_sq = 0.0
    prior_mem_sq = {p.name: {a: 0.0 for a in nonzero} for p in exec_priors}

    for q0 in range(0, n_val, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, n_val))
        _, _, Rv, _, _ = seasonal_decompose(val_ctx[sl], PRED, PERIOD, TREND_WINDOW)
        S_shape = pearson_similarity_matrix(Rv, Rm)
        y_mem_shape = _mem_forecast(S_shape, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
        shape_mem_sq += float(((y_mem_shape - val_fut[sl]) ** 2).sum())
        for name, (p, mf, qf) in pf.items():
            S_prior = feature_similarity(p, qf[sl], mf)
            for a in nonzero:
                S = (1 - a) * S_shape + a * S_prior
                y_mem = _mem_forecast(S, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
                prior_mem_sq[name][a] += float(((y_mem - val_fut[sl]) ** 2).sum())

    shape_mem_mse = shape_mem_sq / n_pts

    rows = []
    for p in exec_priors:
        mem = {0.0: shape_mem_mse}
        for a in nonzero:
            mem[a] = prior_mem_sq[p.name][a] / n_pts
        best_a = min(mem, key=mem.get)   # dict order 0.0 first -> tie => alpha=0
        rows.append(dict(
            name=p.name,
            best_alpha=float(best_a),
            shape_mem_mse=float(shape_mem_mse),
            expert_mem_mse=float(mem[best_a]),
            per_alpha_mem_mse={a: float(mem[a]) for a in ALPHAS_MEM},
            improvement=float((shape_mem_mse - mem[best_a]) / shape_mem_mse * 100.0),
        ))
    return dict(
        shape_mem_mse=float(shape_mem_mse),
        n_val=int(n_val),
        n_memory=int(len(mem_starts)),
        alpha_grid=ALPHAS_MEM,
        rows=rows,
    )


def _spec_str(p: ExecutablePrior) -> str:
    v = p.input_variables
    if set(v) == set(LOAD_VARS):
        inp = "X1..X6"
    elif v == ["OT"]:
        inp = "Y"
    else:
        inp = "+".join(v)
    return f"{p.transform}/{p.temporal_aggregation}/{p.similarity} on {inp}"


def main():
    model = av2._default_model()
    result = run_pipeline(model)          # FROZEN stages 1-5 (LLM cache hit)

    exec_priors = []
    if result["supported_priors"]:
        exec_priors = [ExecutablePrior(**d) for d in result["supported_priors"]]

    retrieval = None
    if exec_priors:
        retrieval = run_retrieval_validation_memory_only(exec_priors)

    # ---- per-hypothesis stage trace (map retrieval row back by name) ----
    ret_by_name = {}
    if retrieval:
        for r in retrieval["rows"]:
            ret_by_name[str(r["name"]).strip().lower()] = r

    hypotheses = result["proposal"]["parsed"]
    trace = []
    for h in hypotheses:
        name = h.get("name", "")
        data_support = (h.get("_data_support") or {}).get("decision", "")
        compiled = None
        proxy = None
        if h.get("_compiled") is not None:
            compiled = _spec_str(ExecutablePrior(
                name="", source="", input_variables=h["_compiled"].get("input_variables", []),
                transform=h["_compiled"].get("transform", ""),
                temporal_aggregation=h["_compiled"].get("temporal_aggregation", ""),
                similarity=h["_compiled"].get("similarity", ""),
                expected_mechanism="",
            ))
        if h.get("_proxy_validator") is not None:
            proxy = (h["_proxy_validator"].get("decision", ""))
        r = ret_by_name.get(str(name).strip().lower())
        row = {
            "hypothesis_id": h.get("hypothesis_id", ""),
            "name": name,
            "domain_fact": h.get("domain_fact", ""),
            "semantic": str((h.get("_semantic") or {}).get("verdict", "")).strip().lower(),
            "data_support": str(data_support).strip().upper(),
            "compiled_proxy": compiled,
            "proxy_validation": str(proxy).strip().upper() if proxy else None,
            "final_decision": None,
            "retrieval": None,
        }
        # final decision per frozen pipeline ordering
        d = str(data_support).strip().upper()
        if row["semantic"] != "accept":
            row["final_decision"] = "REJECTED (semantic)"
        elif d == "REJECT":
            row["final_decision"] = "REJECTED (data)"
        elif d == "UNSUPPORTED":
            row["final_decision"] = "UNSUPPORTED"
        elif h.get("_compiled") is None:
            row["final_decision"] = "REJECTED (compiler)"
        elif row["proxy_validation"] != "PASS":
            row["final_decision"] = "REJECTED (proxy validator)"
        else:
            row["final_decision"] = "ACCEPTED (supported prior)"

        if r is not None:
            row["retrieval"] = {
                "best_alpha": r["best_alpha"],
                "shape_mem_mse": r["shape_mem_mse"],
                "expert_mem_mse": r["expert_mem_mse"],
                "per_alpha_mem_mse": r["per_alpha_mem_mse"],
                "improvement_pct": r["improvement"],
                "positive_case": bool(r["best_alpha"] > 0 and r["improvement"] > 0),
            }
        trace.append(row)

    # ---- positive case judgment (strict) ----
    positive_rows = [t for t in trace
                     if t["retrieval"] is not None and t["retrieval"]["positive_case"]]
    positive_case = bool(positive_rows)

    # ---- console report ----
    print("=" * 100)
    print("SOLAR Agent v2 VAL (FROZEN pipeline; memory-only: Shape vs Shape+Expert)")
    print("=" * 100)
    print(f"model = {model}")
    print(f"proposed          : {result['n_proposed']}")
    print(f"semantic accepted : {result['n_semantic_accepted']}")
    print(f"data supported    : {result['n_data_supported']}")
    print(f"data unsupported  : {result['n_data_unsupported']}")
    print(f"data rejected     : {result['n_data_rejected']}")
    print(f"compiled          : {result['n_compiled']}")
    print(f"proxy passed      : {result['n_proxy_passed']}")
    print("\nSTAGE TRACE:")
    for t in trace:
        print(f"  {t['hypothesis_id']:<3} {t['name']:<40} sem={t['semantic']:<7} "
              f"data={t['data_support']:<11} proxy={t['proxy_validation'] or 'n/a':<5} "
              f"final={t['final_decision']}")
        if t["retrieval"]:
            rr = t["retrieval"]
            print(f"        best_alpha={rr['best_alpha']}  shape_mse={rr['shape_mem_mse']:.4f}  "
                  f"shape+expert_mse={rr['expert_mem_mse']:.4f}  imp={rr['improvement_pct']:+.4f}%  "
                  f"positive={rr['positive_case']}")

    print("\n" + "=" * 100)
    if retrieval:
        print(f"Shape Memory MSE (alpha=0) = {retrieval['shape_mem_mse']:.4f}")
        print(f"VAL queries = {retrieval['n_val']}  memory windows = {retrieval['n_memory']}  "
              f"alpha_grid = {retrieval['alpha_grid']}")
    print(f"SOLAR POSITIVE CASE: {'YES' if positive_case else 'NO'}")
    if positive_rows:
        for t in positive_rows:
            rr = t["retrieval"]
            print(f"  -> {t['name']}  best_alpha={rr['best_alpha']}  "
                  f"VAL MSE improvement={rr['improvement_pct']:+.4f}%")

    # ---- persist ----
    out = {
        "dataset": "solar",
        "agent_version": "v2",
        "experiment": "memory_only_shape_vs_shape_plus_expert",
        "model": model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "alpha_grid": ALPHAS_MEM,
        "chronos": "NOT RUN / NOT USED (memory-only comparison)",
        "expert_facts": result["expert_mechanism"],
        "summary": {
            "n_proposed": result["n_proposed"],
            "n_semantic_accepted": result["n_semantic_accepted"],
            "n_data_supported": result["n_data_supported"],
            "n_data_unsupported": result["n_data_unsupported"],
            "n_data_rejected": result["n_data_rejected"],
            "n_compiled": result["n_compiled"],
            "n_proxy_passed": result["n_proxy_passed"],
            "expert_prior_available": len(exec_priors) > 0,
        },
        "stage_trace": trace,
        "retrieval_validation": retrieval,
        "positive_case": positive_case,
        "positive_priors": [
            {"name": t["name"], "best_alpha": t["retrieval"]["best_alpha"],
             "improvement_pct": t["retrieval"]["improvement_pct"]}
            for t in positive_rows
        ],
        "test_access": "NO (TRAIN memory + VAL query only; TEST never read)",
        "leakage_check": "TRAIN evidence + VAL retrieval only; no Chronos; TEST untouched",
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/solar_agent_v2_val.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # ---- markdown report ----
    md = build_markdown(out)
    Path("results/solar_agent_v2_val.md").write_text(md, encoding="utf-8")

    print("\nSaved results/solar_agent_v2_val.json")
    print("Saved results/solar_agent_v2_val.md")
    return out


def build_markdown(out):
    L = []
    L.append("# Solar Agent v2 VAL 实验报告（冻结管线，仅 Memory）\n")
    L.append(f"> 时间：{out['timestamp']}\n")
    L.append("> 目标：仅回答「Agent 自动产生的 Expert Retrieval Prior 是否给历史检索带来增量价值」。\n")
    L.append("> 比较对象：**Shape Memory Retrieval (alpha=0)** vs **Shape+Expert Memory Retrieval (alpha>0)**。\n")
    L.append("> Chronos：**未运行、未使用**。TEST：**未访问**。\n")

    L.append("\n## 1. 管线摘要\n")
    s = out["summary"]
    L.append(f"- 提出假设 proposed: {s['n_proposed']}")
    L.append(f"- 语义通过 semantic accepted: {s['n_semantic_accepted']}")
    L.append(f"- 数据支持 ACCEPT: {s['n_data_supported']}")
    L.append(f"- 数据不支持 UNSUPPORTED: {s['n_data_unsupported']}")
    L.append(f"- 数据拒绝 REJECT: {s['n_data_rejected']}")
    L.append(f"- 编译 compiled: {s['n_compiled']}")
    L.append(f"- 代理验证通过 proxy passed: {s['n_proxy_passed']}")
    L.append(f"- Expert prior 可用: {s['expert_prior_available']}")

    L.append("\n## 2. Stage Trace（每条假设的完整判决）\n")
    L.append("| ID | Hypothesis | Semantic | Data-Support | Compiled Proxy | Proxy Val. | Final |")
    L.append("|---|---|---|---|---|---|---|")
    for t in out["stage_trace"]:
        L.append(f"| {t['hypothesis_id']} | {t['name']} | {t['semantic']} | {t['data_support']} "
                 f"| {t['compiled_proxy'] or '—'} | {t['proxy_validation'] or '—'} | {t['final_decision']} |")

    L.append("\n## 3. Retrieval Validation（仅 Memory，alpha ∈ {0, 0.1, 0.2, 0.3}）\n")
    if out["retrieval_validation"]:
        rv = out["retrieval_validation"]
        L.append(f"- Shape Memory MSE (alpha=0): **{rv['shape_mem_mse']:.4f}**")
        L.append(f"- VAL 查询数: {rv['n_val']}，记忆窗口数: {rv['n_memory']}")
        L.append("\n| Expert Fact (Hypothesis) | Data Support | Proxy | Best Alpha | Shape MSE | Shape+Expert MSE | Improvement | Positive Case |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for t in out["stage_trace"]:
            if t["retrieval"] is None:
                continue
            r = t["retrieval"]
            L.append(f"| {t['name']} | {t['data_support']} | {t['proxy_validation']} "
                     f"| {r['best_alpha']} | {r['shape_mem_mse']:.4f} | {r['expert_mem_mse']:.4f} "
                     f"| {r['improvement_pct']:+.4f}% | {'YES' if r['positive_case'] else 'NO'} |")
        L.append("\n每 alpha 的 Memory MSE：")
        for t in out["stage_trace"]:
            if t["retrieval"] is None:
                continue
            L.append(f"- {t['name']}: { {a: round(v, 4) for a, v in t['retrieval']['per_alpha_mem_mse'].items()} }")
    else:
        L.append("- 无通过 Proxy Validator 的 expert prior（合法结果）。")

    L.append("\n## 4. Positive Case 判定\n")
    L.append(f"**SOLAR POSITIVE CASE: {'YES' if out['positive_case'] else 'NO'}**\n")
    if out["positive_priors"]:
        for p in out["positive_priors"]:
            L.append(f"- {p['name']}: best_alpha={p['best_alpha']}，VAL Memory MSE 提升 {p['improvement_pct']:+.4f}%")
    else:
        L.append("- 无满足全部 5 条件的候选：Agent 接受的 expert fact + Data-Support ACCEPT + "
                 "Proxy PASS + best_alpha>0 + Shape+Expert VAL MSE 明确低于 Shape VAL MSE。")

    L.append("\n## 5. 输入 Expert Facts（未修改）\n")
    for f in out["expert_facts"].split("\n"):
        if f.strip():
            L.append(f"- {f.strip()}")

    L.append("\n## 6. 风险与说明\n")
    L.append("- 本实验**未**运行 Chronos，未做 Chronos+Memory 融合；只比较 Shape 与 Shape+Expert 的 Memory MSE。")
    L.append("- alpha 网格按任务要求限定为 [0, 0.1, 0.2, 0.3]；Top-K、mean-anchor、softmax、分解参数均沿用冻结管线。")
    L.append("- 仅有的 expert prior 作用在 Y 自身（difference/last/abs_diff，即自回归型 prior），未使用 P001..P006 作为代理。")
    L.append("- P001..P006 的代表性假设（H6）被 Data-Support Critic 判为 UNSUPPORTED，未被人工删除。")
    L.append("- TEST 全程未访问；泄漏检查：仅 TRAIN 记忆 + VAL 查询。")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
