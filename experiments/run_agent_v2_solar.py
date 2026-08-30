"""Agent v2 -- Solar-Energy (NREL, non-ETT), FROZEN pipeline.

Runs the FROZEN Agent v2 pipeline on Solar-Energy, reusing the exact prompt
builders / RULES_BLOCK / whitelist / stage orchestration / retrieval math from
``run_agent_v2_electricity``. Only the anonymized metadata, expert mechanism
facts, column mapping, TRAIN evidence, and dataset-specific decomposition params
(period=144 / trend_window=288 for 10-min data) change.

Agent v2 is treated as frozen: none of Hypothesis Generator / Semantic Critic /
Data-Support Critic / Compiler / Proxy Validator / Retrieval Validation logic is
modified here. The prompt builders are reused by monkey-patching the module
globals that hold the dataset-specific inputs.

TEST is never accessed (TRAIN/VAL only).
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

import run_agent_v2_electricity as av2  # frozen Agent v2 module

from priors.executable import ExecutablePrior, compute_feature, feature_similarity
from priors.validator import CONTEXT, PRED, _mem_forecast
from retrieval.decompose import seasonal_decompose
from retrieval.similarity import pearson_similarity_matrix

# --------------------------------------------------------------------------- #
# dataset-specific constants (sampling 10-min -> daily period 144)
# --------------------------------------------------------------------------- #
PERIOD = 144          # 24h * 6 samples/hour
TREND_WINDOW = 288    # 2 * period (ETTm convention: trend_window = 2*period)
VAL_STRIDE = 6        # ~hourly VAL queries
LOAD_VARS = ["P001", "P002", "P003", "P004", "P005", "P006"]  # X1..X6
N_POPULATION = 137

BLIND_TO_REAL = {
    "Y": "OT",
    "X1": "P001", "X2": "P002", "X3": "P003",
    "X4": "P004", "X5": "P005", "X6": "P006",
}

SOLAR_METADATA_V2 = {
    "target_variable": "Y",
    "target_description": "total solar photovoltaic power output, equal to the aggregate output of 137 co-located plants",
    "observed_variables": ["X1", "X2", "X3", "X4", "X5", "X6"],
    "observed_description": "six individual plant power outputs corresponding to the first six plants (P001 through P006)",
    "observed_population_relation": "X1-X6 are only 6 plants among the 137 plants summed into Y",
    "selection_note": "X1-X6 were selected as the first six plant columns, with no representativeness guarantee",
    "sampling_frequency": "10 minutes",
}

SOLAR_EXPERT_MECHANISM = (
    "solar photovoltaic output is driven by incoming solar irradiance, which rises from zero at sunrise to a peak near solar noon and falls back to zero at sunset\n"
    "irradiance is a common external driver shared across co-located plants, so their outputs rise and fall together\n"
    "the sun's elevation varies with the season, so maximum output and day length are larger in summer than in winter\n"
    "cloud cover and atmospheric conditions modulate irradiance on short timescales, causing rapid correlated fluctuations across plants\n"
    "solar output tends to persist over short horizons because irradiance changes gradually between consecutive samples\n"
    "aggregate output smooths plant-level idiosyncratic fluctuations because it averages over many plants"
)

ALPHAS = av2.ALPHAS          # [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
LAMBDA = av2.LAMBDA          # 0.1
CHUNK = av2.CHUNK            # 256


# --------------------------------------------------------------------------- #
# helpers + TRAIN-only evidence (solar-specific daily period = 144)
# --------------------------------------------------------------------------- #
def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _load_train():
    df = pd.read_csv("data/solar.csv")
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    series = {"OT": df["OT"].to_numpy(np.float64)}
    for v in LOAD_VARS:
        series[v] = df[v].to_numpy(np.float64)
    return df, dates, series, N, train_len


def solar_compute_train_evidence():
    df, dates, series, N, train_len = _load_train()
    dt = dates.iloc[:train_len]

    y = series["OT"][:train_len].astype(np.float64)
    s6 = np.sum([series[v][:train_len].astype(np.float64) for v in LOAD_VARS], axis=0)

    corr_s6y = _corr(s6, y)
    diff_corr_s6y = _corr(np.diff(s6), np.diff(y))
    # S6/Y ratio only over daylight (y > 0); night is 0/0
    day = y > 0.0
    ratio = s6[day] / y[day]
    ratio_stats = {
        "mean": float(np.mean(ratio)),
        "std": float(np.std(ratio)),
        "min": float(np.min(ratio)),
        "max": float(np.max(ratio)),
        "note": "computed over daylight timesteps only (y > 0)",
    }

    months = dt.dt.month.to_numpy()
    monthly = {int(m): round(_corr(s6[months == m], y[months == m]), 4)
               for m in sorted(set(months.tolist()))}

    hours = dt.dt.hour.to_numpy()
    tod = {}
    for lo, hi in ((0, 6), (6, 12), (12, 18), (18, 24)):
        mask = (hours >= lo) & (hours < hi)
        tod[f"{lo:02d}-{hi:02d}h"] = round(_corr(s6[mask], y[mask]), 4)

    coverage = {
        "mechanism_scope": f"{N_POPULATION} plants (population-level aggregate)",
        "proxy_scope": f"{len(LOAD_VARS)} plants",
        "fraction": round(len(LOAD_VARS) / N_POPULATION, 4),
    }

    # Y persistence (daily period = 144 for 10-min data)
    lag1 = _corr(y[1:], y[:-1])
    lag144 = _corr(y[144:], y[:-144])  # same clock-time, previous day
    yc = y[:-144]
    yf = np.array([y[i + 1 : i + 1 + 144].mean() for i in range(len(y) - 144)])
    cur_vs_future = _corr(yc, yf)

    return {
        "train_len": int(train_len),
        "N": int(N),
        "s6_vs_y": {
            "description": "sum(X1..X6) vs Y (population total), TRAIN only",
            "pearson_corr": round(corr_s6y, 4),
            "first_difference_corr": round(diff_corr_s6y, 4),
            "relative_magnitude": ratio_stats,
            "coverage": coverage,
            "monthly_corr": monthly,
            "time_of_day_corr": tod,
        },
        "y_persistence": {
            "description": "Y vs its own past/future, TRAIN only",
            "lag1_autocorr": round(lag1, 4),
            "same_clock_prev_day_corr": round(lag144, 4),
            "current_vs_next_day_mean_corr": round(cur_vs_future, 4),
        },
    }


# --------------------------------------------------------------------------- #
# solar retrieval validation (same frozen math, solar data + period/stride)
# --------------------------------------------------------------------------- #
def run_solar_retrieval_validation(exec_priors):
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

    yc = np.load("results/solar_chronos.npz")["y_chronos_val"].astype(np.float32)
    assert yc.shape[0] == len(val_starts), f"cached chronos mismatch: {yc.shape[0]} vs {len(val_starts)}"

    _, _, Rm, _, _ = seasonal_decompose(mem_ctx, PRED, PERIOD, TREND_WINDOW)
    mem_ctx_mean = mem_ctx.mean(axis=1)
    val_ctx_mean = val_ctx.mean(axis=1)
    n_val = len(val_ctx)
    n_pts = n_val * PRED
    nonzero = [a for a in ALPHAS if a != 0.0]

    pf = {}
    for p in exec_priors:
        mf, qf, _ = compute_feature(p, series, mem_starts, val_starts, np.zeros(2))
        pf[p.name] = (p, mf, qf)

    chronos_sq = shape_mem_sq = shape_fus_sq = 0.0
    prior_mem_sq = {p.name: {a: 0.0 for a in nonzero} for p in exec_priors}
    prior_fus_sq = {p.name: {a: 0.0 for a in nonzero} for p in exec_priors}

    for q0 in range(0, n_val, CHUNK):
        sl = slice(q0, min(q0 + CHUNK, n_val))
        _, _, Rv, _, _ = seasonal_decompose(val_ctx[sl], PRED, PERIOD, TREND_WINDOW)
        S_shape = pearson_similarity_matrix(Rv, Rm)
        y_mem_shape = _mem_forecast(S_shape, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
        shape_mem_sq += float(((y_mem_shape - val_fut[sl]) ** 2).sum())
        yc_sl = yc[sl]
        fused_shape = (1 - LAMBDA) * yc_sl + LAMBDA * y_mem_shape
        shape_fus_sq += float(((fused_shape - val_fut[sl]) ** 2).sum())
        chronos_sq += float(((yc_sl - val_fut[sl]) ** 2).sum())
        for name, (p, mf, qf) in pf.items():
            S_prior = feature_similarity(p, qf[sl], mf)
            for a in nonzero:
                S = (1 - a) * S_shape + a * S_prior
                y_mem = _mem_forecast(S, mem_fut, mem_ctx_mean, val_ctx_mean[sl])
                prior_mem_sq[name][a] += float(((y_mem - val_fut[sl]) ** 2).sum())
                fused = (1 - LAMBDA) * yc_sl + LAMBDA * y_mem
                prior_fus_sq[name][a] += float(((fused - val_fut[sl]) ** 2).sum())

    chronos_mse = chronos_sq / n_pts
    shape_mem_mse = shape_mem_sq / n_pts
    shape_fus_mse = shape_fus_sq / n_pts

    rows = []
    for p in exec_priors:
        fus = {0.0: shape_fus_mse}
        mem = {0.0: shape_mem_mse}
        for a in nonzero:
            fus[a] = prior_fus_sq[p.name][a] / n_pts
            mem[a] = prior_mem_sq[p.name][a] / n_pts
        best_a = min(fus, key=fus.get)
        rows.append(dict(
            name=p.name, best_alpha=best_a,
            val_mem_mse=float(mem[best_a]), val_fus_mse=float(fus[best_a]),
            improvement=float((shape_fus_mse - fus[best_a]) / shape_fus_mse * 100.0),
        ))
    return dict(chronos_mse=float(chronos_mse), shape_mem_mse=float(shape_mem_mse),
                shape_fus_mse=float(shape_fus_mse), rows=rows)


# --------------------------------------------------------------------------- #
# Chronos VAL forecasts (cached)
# --------------------------------------------------------------------------- #
def generate_solar_chronos():
    cache = Path("results/solar_chronos.npz")
    if cache.exists():
        print(f"[chronos] cache hit {cache}")
        return

    from run_adaptive_fusion_gate import load_pipeline, run_chronos

    df = pd.read_csv("data/solar.csv")
    N = len(df)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    val_start, val_end = train_len, train_len + val_len
    ot = df["OT"].to_numpy(np.float32)
    val_starts = np.arange(val_start, val_end - PRED + 1, VAL_STRIDE)
    val_ctx = np.stack([ot[s - CONTEXT:s] for s in val_starts]).astype(np.float32)

    print(f"[chronos] running on {len(val_ctx)} VAL contexts ...")
    pipeline = load_pipeline()
    yc = run_chronos(pipeline, [c for c in val_ctx])
    np.savez(cache, y_chronos_val=yc)
    print(f"[chronos] saved {cache}")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    generate_solar_chronos()

    # monkey-patch the FROZEN module's dataset-specific globals (same code path)
    av2.ELECTRICITY_METADATA_V2 = SOLAR_METADATA_V2
    av2.ELECTRICITY_EXPERT_MECHANISM = SOLAR_EXPERT_MECHANISM
    av2.BLIND_TO_REAL = BLIND_TO_REAL
    av2.LOAD_VARS = LOAD_VARS
    av2.N_POPULATION = N_POPULATION
    av2.compute_train_evidence = solar_compute_train_evidence
    av2.CACHE_DIR = "cache/llm_priors_v2_solar"

    model = av2._default_model()
    result = av2.run_pipeline(model)  # frozen stage logic + prompt builders

    retrieval = None
    if result["supported_priors"]:
        exec_priors = [ExecutablePrior(**d) for d in result["supported_priors"]]
        retrieval = run_solar_retrieval_validation(exec_priors)

    # ---- report ----
    print("=" * 100)
    print("SOLAR Agent v2 (FROZEN pipeline)")
    print("=" * 100)
    print(f"model = {model}")
    print(f"proposed          : {result['n_proposed']}")
    print(f"semantic accepted : {result['n_semantic_accepted']}")
    print(f"data supported    : {result['n_data_supported']}")
    print(f"data unsupported  : {result['n_data_unsupported']}")
    print(f"data rejected     : {result['n_data_rejected']}")
    print(f"compiled          : {result['n_compiled']}")
    print(f"proxy passed      : {result['n_proxy_passed']}")
    print("\n" + "-" * 100)
    hdr = f"{'ID':<4}{'hypothesis':<26}{'semantic':<10}{'data':<13}{'compiler':<34}{'proxy':<7}{'final'}"
    print(hdr)
    print("-" * 100)
    for r in result["rows"]:
        print(f"{r['id']:<4}{r['name']:<26}{r['semantic_critic']:<10}{r['data_support']:<13}"
              f"{r['compiler']:<34}{r['proxy_validator']:<7}{r['final_decision']}")

    if retrieval:
        print("\nRETRIEVAL VALIDATION (VAL only):")
        print(f"  Chronos VAL MSE            : {retrieval['chronos_mse']:.4f}")
        print(f"  Shape Memory MSE (standalone): {retrieval['shape_mem_mse']:.4f}")
        print(f"  Chronos+Shape MSE (lambda=0.1): {retrieval['shape_fus_mse']:.4f}")
        for r in retrieval["rows"]:
            print(f"  [{r['name']}] alpha={r['best_alpha']} memMSE={r['val_mem_mse']:.4f} "
                  f"fusMSE={r['val_fus_mse']:.4f} imp={r['improvement']:+.2f}%")
    else:
        print("\nRETRIEVAL VALIDATION: no supported expert prior (legal result).")

    # ---- audit JSON ----
    ts = datetime.now().isoformat(timespec="seconds")
    audit = {
        "dataset": "solar",
        "agent_version": "v2",
        "model": model,
        "timestamp": ts,
        "metadata_sent_to_llm": result["metadata"],
        "expert_facts": result["expert_mechanism"],
        "train_evidence": result["evidence"],
        "proposal": {k: v for k, v in result["proposal"].items() if k != "parsed"},
        "hypotheses": result["proposal"]["parsed"],
        "semantic_critic": result["semantic_critic"],
        "data_support_critic": result["data_support_critic"],
        "compiler": result["compiler"],
        "proxy_validator": result["proxy_validator"],
        "unsupported_compile": result["unsupported_compile"],
        "supported_priors": result["supported_priors"],
        "summary": {
            "n_proposed": result["n_proposed"],
            "n_semantic_accepted": result["n_semantic_accepted"],
            "n_data_supported": result["n_data_supported"],
            "n_data_unsupported": result["n_data_unsupported"],
            "n_data_rejected": result["n_data_rejected"],
            "n_compiled": result["n_compiled"],
            "n_proxy_passed": result["n_proxy_passed"],
            "expert_prior_available": len(result["supported_priors"]) > 0,
        },
        "rows": result["rows"],
        "retrieval_validation": retrieval,
        "test_access": "NO (TRAIN/VAL only; TEST untouched)",
        "leakage_check": "TRAIN evidence + VAL retrieval only; TEST never read",
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/solar_agent_v2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nSaved results/solar_agent_v2_audit.json")
    return audit


if __name__ == "__main__":
    main()
