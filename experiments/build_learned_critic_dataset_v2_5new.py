#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned Critic — Stage 2: extend the Knowledge-Effect dataset to 5 NEW datasets.

Reuses the Stage-1 experiment definition (build_learned_critic_dataset_5datasets.py)
**unchanged**: frozen Critic v6.1, FULL_RAW / Leave-One-Out, 3 deterministic seeds
(42/123/2026), same ConditionalExpertMLPRetriever architecture, fixed expert slots,
TEST locked.

The ONLY thing this stage adds is the same data-generation protocol applied to 5
new cross-domain datasets (ETTh2, Traffic, LondonSmartMeters, WindFarms, Tourism),
using frequency-appropriate context/horizon and a deterministic target-selection
rule (first series in original order with enough history + qualified missing ratio).

TEST is never read. test_used = false.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
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

# frozen reuse (verbatim imports; nothing here modifies the Critic / Stage-1)
import build_learned_critic_dataset_5datasets as v1        # noqa: E402
from run_expert_agent_closed_loop import compile_condition  # noqa: E402  (v5 compiler)
import run_agent_v2_electricity as av2                     # noqa: E402  (_list_from / _default_model)
from priors.llm_prior import _get_client, _default_model   # noqa: E402

v1.OUT_ROOT = Path("results/learned_critic_dataset_v2")    # redirect FULL/LOO checkpoints

# --------------------------------------------------------------------------- #
# constants / flags
# --------------------------------------------------------------------------- #
SEEDS = [42, 123, 2026]
MAX_MISSING_RATIO = 0.25      # target-selection missing threshold (documented)
MIN_LEN_FACTOR = 6.0          # length >= (context+horizon)*FACTOR  -> guarantees val has >= context+horizon windows

LLM_CACHE_DIR = "cache/llm_priors_v2_5new"
LLM_MODEL = _default_model()
MAX_LLM_RETRIES = 3

OUT_ROOT = Path("results/learned_critic_dataset_v2")
MULTI_DIR = Path("results/multidataset")   # reuse Stage-1 data dir (ETTh2/traffic re-prepared)

# --------------------------------------------------------------------------- #
# dataset registry (5 new; target-selection is deterministic, documented)
# --------------------------------------------------------------------------- #
V2_DATASETS = [
    {
        "key": "etth2", "dir": "ETTh2",
        "kind": "csv_ot", "source": "data/ETTh2.csv",
        "sampling_str": "hourly", "sampling_minutes": 60,
        "context": 168, "horizon": 24, "period": 24,
        "domain": "transformer_temperature",
        "target_description": "transformer oil temperature (OT)",
    },
    {
        "key": "traffic", "dir": "traffic",
        "kind": "csv_first_sensor", "source": "data/traffic.csv",
        "sampling_str": "hourly", "sampling_minutes": 60,
        "context": 168, "horizon": 24, "period": 24,
        "domain": "road_traffic_occupancy",
        "target_description": "road occupancy of a single sensor (first sensor column in original order)",
    },
    {
        "key": "london_smart_meters", "dir": "LondonSmartMeters",
        "kind": "tsf_first_series",
        "source": "data/learned_critic_datasets/LondonSmartMeters/london_smart_meters_dataset_with_missing_values.tsf",
        "sampling_str": "half_hourly", "sampling_minutes": 30,
        "context": 336, "horizon": 48, "period": 48,
        "domain": "household_electricity",
        "target_description": "half-hourly electricity consumption of one household (first series in original order)",
    },
    {
        "key": "wind_farms", "dir": "WindFarms",
        "kind": "tsf_first_series",
        "source": "data/learned_critic_datasets/WindFarms/wind_farms_minutely_dataset_with_missing_values.tsf",
        "sampling_str": "minutely", "sampling_minutes": 1,
        "context": 1440, "horizon": 144, "period": 1440, "stride": 144,
        "domain": "wind_power_generation",
        "target_description": "minutely power output of one wind farm (first series in original order)",
    },
    {
        "key": "tourism", "dir": "Tourism",
        "kind": "tsf_first_series",
        "source": "data/learned_critic_datasets/Tourism/tourism_monthly_dataset.tsf",
        "sampling_str": "monthly", "sampling_minutes": 43200,
        "context": 36, "horizon": 12, "period": 12,
        "domain": "tourism_demand",
        "target_description": "monthly tourism demand of one series (first series in original order)",
    },
]


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _eta2(groups, y):
    groups, y = np.asarray(groups), np.asarray(y, float)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    within = sum(float(((y[groups == g] - y[groups == g].mean()) ** 2).sum())
                 for g in np.unique(groups))
    return 1.0 - within / total


# --------------------------------------------------------------------------- #
# target selection (deterministic; documented rule)
# --------------------------------------------------------------------------- #
def _parse_tsf_series(path: Path):
    """Yield (name, start_str, values_list) in original order."""
    in_data = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                if line.startswith("@data"):
                    in_data = True
                continue
            if in_data:
                segs = line.split(":")
                name = segs[0]
                start = ""
                for s in segs[1:-1]:
                    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
                        start = s
                        break
                values = [v.strip() for v in segs[-1].split(",")]
                yield name, start, values


def _tsf_start_to_datetime(start: str):
    # "2012-10-13 00-00-01" -> Timestamp
    date_part = start[:10]
    time_part = start[11:].replace("-", ":")
    return pd.to_datetime(f"{date_part} {time_part}")


def select_target_series(cfg: dict):
    """Return (DataFrame(date, OT), series_name) for the chosen target."""
    series_name = None
    if cfg["kind"] == "csv_ot":
        df = pd.read_csv(cfg["source"])
        out = df[["date", "OT"]].copy()
        out["OT"] = pd.to_numeric(out["OT"], errors="coerce")
        series_name = "OT"
    elif cfg["kind"] == "csv_first_sensor":
        df = pd.read_csv(cfg["source"])
        sensor_cols = [c for c in df.columns if c.startswith("Sensor_")]
        first = sensor_cols[0]
        series_name = first
        out = pd.DataFrame({"date": df["date"], "OT": pd.to_numeric(df[first], errors="coerce")})
    elif cfg["kind"] == "tsf_first_series":
        min_len = int((cfg["context"] + cfg["horizon"]) * MIN_LEN_FACTOR)
        chosen = None
        for name, start, values in _parse_tsf_series(Path(cfg["source"])):
            miss = sum(1 for v in values if v in ("", "?"))
            ratio = miss / max(1, len(values))
            if len(values) >= min_len and ratio <= MAX_MISSING_RATIO:
                chosen = (name, start, values)
                break
        if chosen is None:
            raise RuntimeError(f"{cfg['key']}: no series satisfies the target-selection rule")
        series_name, start, values = chosen
        vals = [np.nan if v in ("", "?") else float(v) for v in values]
        ot = pd.Series(vals, dtype="float64").ffill()  # LOCF (documented)
        dates = pd.date_range(start=_tsf_start_to_datetime(start),
                              periods=len(values), freq=f"{cfg['sampling_minutes']}min")
        out = pd.DataFrame({"date": dates, "OT": ot.astype("float32")})
    else:
        raise ValueError(f"unknown kind: {cfg['kind']}")

    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["OT"]).reset_index(drop=True)
    if not out["date"].is_monotonic_increasing:
        out = out.sort_values("date").reset_index(drop=True)
    return out, series_name


def _make_splits(n, tr=0.6, va=0.2):
    te = int(n * tr)
    ve = int(n * (tr + va))
    return {"train": [0, te], "val": [te, ve], "test": [ve, n]}


def _make_windows(start, end, context, horizon, stride=1):
    last = end - context - horizon
    if last < start:
        return np.empty((0,), dtype=np.int64)
    return np.arange(start, last + 1, stride, dtype=np.int64)


def prepare_dataset(cfg: dict):
    """Write series.csv + window_indices.npz + config.json under results/multidataset/{dir}."""
    out_dir = MULTI_DIR / cfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df, series_name = select_target_series(cfg)
    df.to_csv(out_dir / "series.csv", index=False)

    n = len(df)
    splits = _make_splits(n)
    stride = cfg.get("stride", 1)
    train_idx = _make_windows(*splits["train"], cfg["context"], cfg["horizon"], stride)
    val_idx = _make_windows(*splits["val"], cfg["context"], cfg["horizon"], stride)
    test_idx = _make_windows(*splits["test"], cfg["context"], cfg["horizon"], stride)
    np.savez_compressed(out_dir / "window_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    meta = {
        "dataset": cfg["key"], "source": cfg["source"], "domain": cfg["domain"],
        "frequency": cfg["sampling_str"], "rows": n,
        "context": cfg["context"], "horizon": cfg["horizon"], "stride": stride,
        "split": splits,
        "num_windows": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "target_selection_rule": (
            "deterministic first-series: first series in original order with "
            f"length >= (context+horizon)*{MIN_LEN_FACTOR} and missing_ratio <= {MAX_MISSING_RATIO}; "
            "missing values LOCF-forward-filled"),
        "target_series": series_name,
        "test_policy": "indices prepared only; do not evaluate",
    }
    (out_dir / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{cfg['key']}] prepared: rows={n} windows train={len(train_idx)} val={len(val_idx)} "
          f"target={series_name} context={cfg['context']}/{cfg['horizon']}", flush=True)
    return df


# --------------------------------------------------------------------------- #
# v5-style TRAIN-only evidence (for generation + v5 compilation)
# --------------------------------------------------------------------------- #
def build_evidence_v5(cfg: dict, df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["date"])
    N = len(df)
    train_len = int(0.6 * N)
    dt = dates.iloc[:train_len]
    y = df["OT"].to_numpy(np.float64)[:train_len]
    months = dt.dt.month.to_numpy()
    hours = dt.dt.hour.to_numpy()
    dows = dt.dt.dayofweek.to_numpy()

    months_present = sorted(int(m) for m in set(months.tolist()))
    hours_present = sorted(int(h) for h in set(hours.tolist()))
    dows_present = sorted(int(d) for d in set(dows.tolist()))

    monthly_mean = {int(m): float(y[months == m].mean()) for m in months_present}
    monthly_vals = [monthly_mean[m] for m in months_present]
    hourly_mean = np.array([float(y[hours == h].mean()) if (hours == h).any() else 0.0 for h in range(24)])
    dow_mean = np.array([float(y[dows == d].mean()) if (dows == d).any() else 0.0 for d in range(7)])

    # feature thresholds (TRAIN-derived, for the compiler), deterministic
    rng = np.random.default_rng(0)
    ctx = 64
    train_len_c = max(ctx + 1, train_len)
    starts = rng.choice(train_len_c - ctx, size=min(3000, train_len_c - ctx), replace=False)
    t = np.linspace(-1, 1, ctx)
    tc = t - t.mean()
    denom = float(np.sum(tc * tc))
    win = {"recent_level": [], "recent_volatility": [], "recent_slope_magnitude": [],
           "recent_diff_magnitude": [], "recent_autocorr": []}
    for s in starts:
        x = y[s:s + ctx]
        win["recent_level"].append(x.mean())
        win["recent_volatility"].append(x.std())
        win["recent_slope_magnitude"].append(abs(float(np.sum((x - x.mean()) * tc) / denom)))
        win["recent_diff_magnitude"].append(abs(x[-1] - x[-2]))
        win["recent_autocorr"].append(_corr(x[1:], x[:-1]))
    feature_thresholds = {}
    for feat, arr in win.items():
        arr = np.asarray(arr, float)
        feature_thresholds[feat] = {
            "train_median": float(np.median(arr)), "train_median_low": float(np.median(arr)),
            "train_high": float(arr.mean() + arr.std()), "train_low": float(arr.mean() - arr.std()),
        }

    period = cfg["period"]
    lag_period = _corr(y[period:], y[:-period]) if train_len > period else float("nan")

    return {
        "train_span": {"start": str(dt.min()), "end": str(dt.max()),
                       "n_days": int(dt.dt.date.nunique()), "n_rows": int(train_len),
                       "total_rows": int(N), "sampling": cfg["sampling_str"]},
        "temporal_coverage": {
            "months_present": months_present, "months_missing": [m for m in range(1, 13) if m not in months_present],
            "month_fraction": round(len(months_present) / 12, 4),
            "hours_present": hours_present, "hours_missing": [h for h in range(24) if h not in hours_present],
            "hour_fraction": round(len(hours_present) / 24, 4),
            "dows_present": dows_present, "dow_fraction": round(len(dows_present) / 7, 4),
        },
        "diurnal": {"hourly_mean": [round(v, 2) for v in hourly_mean],
                    "eta2_hour_on_ot": round(_eta2(hours, y), 4)},
        "seasonal": {"monthly_mean": {k: round(v, 2) for k, v in monthly_mean.items()},
                     "monthly_mean_values": [round(v, 2) for v in monthly_vals],
                     "eta2_month_on_ot": round(_eta2(months, y), 4)},
        "weekly": {"dow_mean": [round(v, 2) for v in dow_mean],
                   "eta2_dow_on_ot": round(_eta2(dows, y), 4)},
        "population": {"n_proxy": 1, "n_population": 1, "proxy_fraction": 1.0,
                       "corr_proxy_sum_vs_target": 1.0},
        "persistence": {"lag1_autocorr": round(_corr(y[1:], y[:-1]), 4),
                        "lag_period_autocorr": round(lag_period, 4) if lag_period == lag_period else None},
        "variability": {"ot_mean": round(float(y.mean()), 2), "ot_std": round(float(y.std()), 2)},
        "feature_thresholds": feature_thresholds,
        "x_subset_representativeness": {},
    }


# --------------------------------------------------------------------------- #
# feature catalog (generic, same as frozen v5)
# --------------------------------------------------------------------------- #
FEATURE_RULES = {
    "hour_of_day": {"kind": "calendar", "categories": {
        "morning": [6, 7, 8, 9, 10, 11], "midday": [12, 13, 14, 15],
        "evening": [16, 17, 18, 19, 20, 21], "night": [22, 23, 0, 1, 2, 3, 4, 5]}},
    "day_of_week": {"kind": "calendar", "categories": {"weekday": [0, 1, 2, 3, 4], "weekend": [5, 6]}},
    "working_day": {"kind": "calendar", "categories": {"workday": [0, 1, 2, 3, 4], "non_working": [5, 6]}},
    "seasonal_level": {"kind": "calendar", "categories": {
        "winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8], "autumn": [9, 10, 11]}},
    "recent_level": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low",
        "above_typical": "@train_high", "below_typical": "@train_low"}},
    "recent_slope_magnitude": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
    "recent_diff_magnitude": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
    "recent_volatility": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
    "recent_autocorr": {"kind": "continuous", "qualifiers": {
        "high": "@train_median", "low": "@train_median_low"}},
}


# --------------------------------------------------------------------------- #
# LLM call with token logging (cache-aware)
# --------------------------------------------------------------------------- #
def call_llm_v2(dataset: str, system: str, user: str, stage: str, model: str, api_log: list):
    key = hashlib.sha256(json.dumps({"stage": stage, "system": system, "user": user, "model": model},
                                    sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(LLM_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        api_log.append({"dataset": dataset, "model": model,
                        "prompt_tokens": cached.get("prompt_tokens"),
                        "completion_tokens": cached.get("completion_tokens"),
                        "total_tokens": cached.get("total_tokens"), "retry_count": 0, "cached": True})
        return cached["raw_response"]

    retry = 0
    while True:
        try:
            client = _get_client()
            resp = client.messages.create(model=model, max_tokens=20000, system=system,
                                          messages=[{"role": "user", "content": user}])
            content = resp.content
            raw = content if isinstance(content, str) else "".join(
                getattr(b, "text", "") for b in content if getattr(b, "type", "text") == "text")
            if not raw.strip():
                raise RuntimeError("LLM returned no text")
            usage = getattr(resp, "usage", None)
            pt = getattr(usage, "input_tokens", None)
            ct = getattr(usage, "output_tokens", None)
            api_log.append({"dataset": dataset, "model": model,
                            "prompt_tokens": pt, "completion_tokens": ct,
                            "total_tokens": (pt or 0) + (ct or 0), "retry_count": retry, "cached": False})
            cache_path.write_text(json.dumps({"stage": stage, "model": model, "system": system,
                                              "user": user, "raw_response": raw,
                                              "prompt_tokens": pt, "completion_tokens": ct,
                                              "total_tokens": (pt or 0) + (ct or 0),
                                              "timestamp": datetime.now().isoformat(timespec="seconds")},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
            return raw
        except Exception as e:  # noqa: BLE001
            retry += 1
            if retry > MAX_LLM_RETRIES:
                raise
            print(f"    [LLM retry {retry}] {e!r}", flush=True)
            time.sleep(2.0)


# --------------------------------------------------------------------------- #
# expert generation (propose ~10 + v5 compile)
# --------------------------------------------------------------------------- #
_GEN_SYSTEM = (
    "You are a domain expert on forecasting a univariate {freq} time series (target "
    "variable OT). A retrieval model forecasts the series by retrieving similar "
    "historical windows from a memory bank and averaging their futures. Propose "
    "CONDITIONAL expert knowledge: each statement says WHEN it applies, WHAT historical "
    "segments to prefer, and WHY. The activation condition must be observable from the "
    "query's own history and timestamp. Output strict JSON only."
)


def _metadata_text(cfg):
    return (
        "DATASET:\n"
        f"- target variable: OT ({cfg['target_description']})\n"
        f"- domain: {cfg['domain']}\n"
        f"- sampling frequency: {cfg['sampling_str']}\n"
        "- observed variables: OT only (univariate target series)"
    )


_CATALOG = (
    "AVAILABLE ACTIVATION FEATURES (choose observable_features ONLY from this list):\n"
    "  - hour_of_day: current hour-of-day phase (morning/midday/evening/night)\n"
    "  - day_of_week: weekday vs weekend\n"
    "  - working_day: workday vs non-working day\n"
    "  - seasonal_level: month-of-year phase (winter/spring/summer/autumn)\n"
    "  - recent_level: current level relative to typical\n"
    "  - recent_slope_magnitude: recent trend strength\n"
    "  - recent_diff_magnitude: recent point-to-point change magnitude\n"
    "  - recent_volatility: recent variability regime\n"
    "  - recent_autocorr: recent persistence strength"
)


def _train_summary_text(ev):
    ds, se = ev["diurnal"], ev["seasonal"]
    pv = ev["persistence"]
    return (
        "TRAIN-ONLY DATA SUMMARY (deterministic):\n"
        f"- train span: {ev['train_span']['start']} -> {ev['train_span']['end']} "
        f"({ev['train_span']['n_days']} days, {ev['train_span']['sampling']})\n"
        f"- coverage: {len(ev['temporal_coverage']['months_present'])}/12 months, "
        f"{len(ev['temporal_coverage']['hours_present'])}/24 hours, "
        f"{len(ev['temporal_coverage']['dows_present'])}/7 days\n"
        f"- eta2(hour,OT)={ds['eta2_hour_on_ot']}, eta2(month,OT)={se['eta2_month_on_ot']}, "
        f"eta2(dow,OT)={ev['weekly']['eta2_dow_on_ot']}\n"
        f"- hourly mean OT: {ds['hourly_mean']}\n"
        f"- monthly mean OT: {se['monthly_mean']}\n"
        f"- lag1 autocorr {pv['lag1_autocorr']}, lag_period autocorr {pv.get('lag_period_autocorr')}\n"
        f"- OT mean {ev['variability']['ot_mean']}, OT std {ev['variability']['ot_std']}\n"
    )


_SCHEMA = (
    "OUTPUT SCHEMA (strict JSON):\n"
    'Return {"expert_knowledge": [ ... ]} with 6 to 10 entries. Each entry:\n'
    '  - "expert_id": "E1", "E2", ...\n'
    '  - "name": short label\n'
    '  - "claim": one conditional sentence: "When [observable condition], historical '
    'segments satisfying [relationship] should be preferred because [reason]"\n'
    '  - "activation_condition": the "when [observable state]" part, natural language\n'
    '  - "observable_features": list of features from AVAILABLE ACTIVATION FEATURES\n'
    '  - "condition_spec": {"logic": "AND"|"OR", "conditions": [{"feature": "...", '
    '"qualifier": "..."}]} — exact logical structure, one entry per primitive\n'
    '  - "reason": why this helps retrieval\n'
)


def build_api_usage_from_cache():
    """Read the LLM cache to reconstruct the full token accounting (cached runs included)."""
    log = []
    cache_dir = Path(LLM_CACHE_DIR)
    if not cache_dir.exists():
        return log
    for f in sorted(cache_dir.glob("*.json")):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        stage = c.get("stage", "")
        log.append({"dataset": stage.replace("_expert_r1", ""), "model": c.get("model", LLM_MODEL),
                    "prompt_tokens": c.get("prompt_tokens"), "completion_tokens": c.get("completion_tokens"),
                    "total_tokens": c.get("total_tokens"), "retry_count": 0, "cached": True})
    return log


def generate_experts(cfg: dict, df: pd.DataFrame, api_log: list):
    """Propose ~10 experts via DeepSeek + compile each with the v5 compiler."""
    evidence = build_evidence_v5(cfg, df)
    sys_p = _GEN_SYSTEM.format(freq=cfg["sampling_str"])
    usr_p = "\n\n".join([_metadata_text(cfg), _train_summary_text(evidence), _CATALOG, _SCHEMA,
                         "TASK: propose 6-10 conditional expert-knowledge statements. Output only the JSON object."])
    raw = call_llm_v2(cfg["key"], sys_p, usr_p, f"{cfg['key']}_expert_r1", LLM_MODEL, api_log)
    experts = av2._list_from(raw, "expert_knowledge")

    out = []
    for i, e in enumerate(experts):
        if not e.get("expert_id"):
            e["expert_id"] = f"E{i + 1}"
        e["compiled"] = compile_condition(e, FEATURE_RULES, evidence)
        out.append(e)
    # dedup by name/claim (sequential, keep first)
    seen, dedup = set(), []
    for e in out:
        k = (e.get("name", ""), e.get("claim", "")[:80])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(e)
    print(f"[{cfg['key']}] generated {len(dedup)} experts: "
          f"{[e['expert_id'] for e in dedup]}", flush=True)
    return dedup, evidence


# --------------------------------------------------------------------------- #
# FULL_RAW + LOO (reuse Stage-1 unchanged, per-dataset context/horizon)
# --------------------------------------------------------------------------- #
def run_full_loo(cfg: dict, raw_pool: Path, device):
    v1.CONTEXT = cfg["context"]
    v1.HORIZON = cfg["horizon"]
    v1.OUT_ROOT = OUT_ROOT

    v1cfg = {
        "key": cfg["key"], "dir": cfg["dir"], "raw_pool": str(raw_pool),
        "sampling_str": cfg["sampling_str"], "sampling_minutes": cfg["sampling_minutes"],
        "metadata": {"target_description": cfg["target_description"], "n_population": 1},
        "evidence": "generic",
    }
    data = v1.load_and_prepare(v1cfg, device)
    n_raw = data["n_raw"]
    print(f"[{cfg['key']}] raw experts={n_raw} bank={len(data['X_bank'])} fit={len(data['X_fit'])} "
          f"val={len(data['X_val'])} (context={cfg['context']})", flush=True)

    audit = v1.run_audit(v1cfg, data)
    safety, disabled = v1.run_safety_check(v1cfg, data, audit, device)
    gen_feats = v1.dataset_generic_features(data["series"], int(0.6 * len(data["series"])), v1cfg)
    raw_train_act = np.concatenate([data["raw_bank_act"], data["raw_fit_act"]], axis=0)

    # persist audit + features
    audit_json = []
    for i in range(n_raw):
        r = audit["rows"][i]
        audit_json.append({
            "expert_id": r["expert_id"], "name": r["name"],
            "pre_repair_execution_state": r["pre_state"],
            "post_repair_execution_state": r["post_state"],
            "repair_happened": r["repair_happened"],
            "post_repair_safety_triggered": r["triggered"],
            "post_repair_safety_decision": safety.get(i, {}).get("safety"),
            "v6_1_final_decision": ("ACCEPT" if (i in audit["keep"] and i not in disabled) else "REJECT"),
            "final_active": bool(i in audit["keep"] and i not in disabled),
            "observability_decision": "PASS" if r["observability"]["all_observable"] else "FAIL",
            "evidence_decision": "PASS" if r["ev_pass"] else "FAIL",
            "compiler_decision": r["compiled"]["status"],
            "compiler_repair": r["compiled"].get("repairs", []),
            "fidelity_decision": "PASS" if r["fid_pass"] else "FAIL",
            "generalization_decision": "PASS" if r["gen_pass"] else "FAIL",
            "redundancy_decision": "DUPLICATE_ACTIVATION" if r["redundant"] else None,
            "activation_coverage_decision": ("DEAD" if r["coverage"]["dead"]
                                             else ("CONSTANT_LOW_INFORMATION" if r["coverage"]["constant"]
                                                   else ("ACCEPTED_LOCAL_SCOPE" if r["coverage"]["local_scope"] else "OK"))),
            "mechanism_family": r["lineage"]["mechanism_family"],
            "state_variable_family": r["lineage"]["state_variable_family"],
        })

    tr = v1.run_training(v1cfg, data, device)
    labels = v1.compute_labels(tr["full_mse"], tr["wo_mse"])

    samples = []
    for i in range(n_raw):
        r = audit["rows"][i]
        e = data["raw_experts"][i]
        act_f = v1.activation_features(raw_train_act[:, i])
        lab = labels[i]
        full_vals = [tr["full_mse"][s] for s in SEEDS]
        row = {
            "dataset_name": cfg["key"], "expert_id": e["expert_id"], "expert_index": i,
            "pool_size": n_raw, "expert_text": e.get("claim", ""), "name": e.get("name", ""),
            "observable_features": json.dumps(e.get("observable_features", [])),
            "mechanism_family": r["lineage"]["mechanism_family"],
        }
        row.update(gen_feats)
        row.update(act_f)
        row.update(audit_json[i])
        row["full_raw_mse_mean"] = float(np.mean(full_vals))
        row["full_raw_mse_std"] = float(np.std(full_vals))
        for s in SEEDS:
            row[f"without_i_mse_seed{s}"] = tr["wo_mse"][i][s]
            row[f"delta_seed{s}"] = lab["delta"][f"delta_seed{s}"]
            row[f"relative_gain_seed{s}"] = lab["relative_gain"][f"relative_gain_seed{s}"]
        row["mean_delta"] = lab["mean_delta"]
        row["std_delta"] = lab["std_delta"]
        row["mean_relative_gain"] = lab["mean_relative_gain"]
        row["std_relative_gain"] = lab["std_relative_gain"]
        row["n_positive_seeds"] = lab["n_positive_seeds"]
        row["n_negative_seeds"] = lab["n_negative_seeds"]
        row["contribution_label"] = lab["contribution_label"]
        row["other_active_expert_count"] = sum(1 for j in range(n_raw) if j != i and raw_train_act[:, j].max() > 0.0)
        samples.append(row)

    label_counts = {}
    for lab in labels:
        label_counts[lab["contribution_label"]] = label_counts.get(lab["contribution_label"], 0) + 1
    summary = {
        "dataset": cfg["key"], "n_experts": n_raw,
        "full_raw_mse": {str(s): tr["full_mse"][s] for s in SEEDS},
        "full_raw_mse_mean": float(np.mean([tr["full_mse"][s] for s in SEEDS])),
        "full_raw_mse_std": float(np.std([tr["full_mse"][s] for s in SEEDS])),
        "param_count": tr["param_count"],
        "n_repair_created_activation": audit["n_repair_created_activation"],
        "label_counts": label_counts,
        "context": cfg["context"], "horizon": cfg["horizon"],
    }
    return samples, summary, audit_json


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "experts", "train", "merge", "all"], default="all")
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| deterministic ON | critic frozen v6.1", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    api_log = []
    all_samples = []
    dataset_summaries = []
    all_audits = {}

    for cfg in V2_DATASETS:
        key = cfg["key"]
        if args.only and key != args.only:
            continue
        t0 = time.time()
        print(f"\n===== dataset: {key} =====", flush=True)

        # Stage: prep (target selection + series.csv + window_indices)
        if args.stage in ("prep", "experts", "train", "all"):
            df = prepare_dataset(cfg)

        # Stage: experts (DeepSeek + compile)
        raw_pool = OUT_ROOT / key / "expert_round_1.json"
        if args.stage in ("experts", "train", "all"):
            raw_pool.parent.mkdir(parents=True, exist_ok=True)
            if raw_pool.exists():
                print(f"[{key}] expert pool already exists, skip generation (cached)", flush=True)
            else:
                experts, evidence = generate_experts(cfg, df, api_log)
                raw_pool.write_text(json.dumps(experts, ensure_ascii=False, indent=2), encoding="utf-8")
                (OUT_ROOT / key / "evidence.json").write_text(
                    json.dumps(evidence, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        # Stage: train (FULL_RAW + LOO)
        if args.stage in ("train", "all"):
            if not raw_pool.exists():
                print(f"[{key}] SKIP train: expert pool missing", flush=True)
                continue
            samples, summary, audit_json = run_full_loo(cfg, raw_pool, device)
            all_samples.extend(samples)
            dataset_summaries.append(summary)
            all_audits[key] = audit_json
            # true GPU training runtime = sum of per-run checkpoint runtimes (resume-proof)
            runs_dir = OUT_ROOT / key / "runs"
            if runs_dir.exists():
                summary["runtime_seconds"] = round(sum(
                    json.loads(f.read_text(encoding="utf-8")).get("runtime_seconds", 0.0)
                    for f in runs_dir.glob("*.json")), 2)
            (OUT_ROOT / key / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            (OUT_ROOT / key / "audit.json").write_text(
                json.dumps(audit_json, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{key}] done labels={summary['label_counts']}", flush=True)

    # ---- write v2 new samples ----
    if args.stage in ("train", "all") and all_samples:
        df_new = pd.DataFrame(all_samples)
        df_new.to_csv(OUT_ROOT / "knowledge_effect_samples_new.csv", index=False, encoding="utf-8")
        try:
            df_new.to_parquet(OUT_ROOT / "knowledge_effect_samples_new.parquet", index=False)
        except Exception as e:  # noqa: BLE001
            print("parquet write skipped:", e, flush=True)
        (OUT_ROOT / "dataset_summary.json").write_text(
            json.dumps(dataset_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- merge v1 + v2 ----
    if args.stage in ("merge", "all") and all_samples:
        v1_parquet = Path("results/learned_critic_dataset_v1/knowledge_effect_samples.parquet")
        df_new = pd.DataFrame(all_samples)
        if v1_parquet.exists():
            df_v1 = pd.read_parquet(v1_parquet)
            keys = set(zip(df_v1["dataset_name"], df_v1["expert_id"]))
            df_new_dedup = df_new[~df_new.apply(lambda r: (r["dataset_name"], r["expert_id"]) in keys, axis=1)]
            dup = len(df_new) - len(df_new_dedup)
            combined = pd.concat([df_v1, df_new_dedup], ignore_index=True)
            print(f"[merge] v1={len(df_v1)} v2_new={len(df_new)} duplicate_dropped={dup} combined={len(combined)}", flush=True)
        else:
            combined = df_new
            print(f"[merge] v1 missing; combined={len(combined)}", flush=True)
        combined.to_parquet(OUT_ROOT / "knowledge_effect_samples_combined.parquet", index=False)
        combined.to_csv(OUT_ROOT / "knowledge_effect_samples_combined.csv", index=False, encoding="utf-8")

    # ---- api usage (always read from LLM cache, so cached runs are counted) ----
    api_usage = build_api_usage_from_cache()
    if api_usage:
        (OUT_ROOT / "api_usage.json").write_text(
            json.dumps(api_usage, ensure_ascii=False, indent=2), encoding="utf-8")
        tot = sum(x["total_tokens"] or 0 for x in api_usage)
        print(f"[api] calls={len(api_usage)} total_tokens={tot}", flush=True)
    elif api_log:
        (OUT_ROOT / "api_usage.json").write_text(
            json.dumps(api_log, ensure_ascii=False, indent=2), encoding="utf-8")
    api_usage = api_usage or api_log

    # ---- report ----
    if args.stage in ("train", "all"):
        gpu_runs = sum(len(list((OUT_ROOT / ds["dataset"] / "runs").glob("*.json")))
                       for ds in dataset_summaries if (OUT_ROOT / ds["dataset"] / "runs").exists())
        write_report(dataset_summaries, all_samples, api_usage, gpu_runs)

    # ---- final flags ----
    print_flags()


def write_report(dataset_summaries, samples, api_log, gpu_runs=None):
    L = ["## 结果", ""]
    L.append(f"新增 5 个跨领域数据集；新增知识样本 = {len(samples)} 条。")
    if samples:
        labels = [s["contribution_label"] for s in samples]
        L.append(f"新增样本 label 分布：POSITIVE={labels.count('POSITIVE')}, "
                 f"NEGATIVE={labels.count('NEGATIVE')}, MIXED={labels.count('MIXED')}")
    L += ["", "| Dataset | n Expert | FULL_RAW mean MSE | Positive | Negative | Mixed | 3-seed consistent | runtime |",
          "|---|---|---|---|---|---|---|---|"]
    for ds in dataset_summaries:
        lc = ds["label_counts"]
        ds_samples = [s for s in samples if s["dataset_name"] == ds["dataset"]]
        n_consistent = sum(1 for s in ds_samples
                           if s["n_positive_seeds"] == 3 or s["n_negative_seeds"] == 3)
        L.append(f"| {ds['dataset']} | {ds['n_experts']} | {ds['full_raw_mse_mean']:.4f} | "
                 f"{lc.get('POSITIVE', 0)} | {lc.get('NEGATIVE', 0)} | {lc.get('MIXED', 0)} | "
                 f"{n_consistent} | {ds.get('runtime_seconds', 0)}s |")
    L.append("")

    # merge summary (v1 / v2 new / combined)
    v1_parquet = Path("results/learned_critic_dataset_v1/knowledge_effect_samples.parquet")
    n_v1 = len(pd.read_parquet(v1_parquet)) if v1_parquet.exists() else 0
    comb_parquet = OUT_ROOT / "knowledge_effect_samples_combined.parquet"
    n_combined = len(pd.read_parquet(comb_parquet)) if comb_parquet.exists() else (n_v1 + len(samples))
    L.append(f"- v1 samples = {n_v1}")
    L.append(f"- v2 new samples = {len(samples)}")
    L.append(f"- combined samples = {n_combined}")
    if comb_parquet.exists():
        comb = pd.read_parquet(comb_parquet)
        cl = comb["contribution_label"]
        L.append(f"- combined label 分布：POSITIVE={int((cl == 'POSITIVE').sum())}, "
                 f"NEGATIVE={int((cl == 'NEGATIVE').sum())}, MIXED={int((cl == 'MIXED').sum())}")
    L.append("")

    # tokens / GPU
    if api_log:
        tot = sum(x["total_tokens"] or 0 for x in api_log)
        L.append(f"- DeepSeek total tokens = {tot}；API calls = {len(api_log)}")
    if gpu_runs:
        gpu_secs = sum(ds.get("runtime_seconds", 0) for ds in dataset_summaries)
        L.append(f"- GPU total training runs = {gpu_runs}；GPU runtime = {gpu_secs:.0f}s ({gpu_secs/3600:.2f}h)")
    L.append("")
    L.append("## 结论")
    L.append("（仅汇总数据；未修改方法。）")
    (OUT_ROOT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


def print_flags():
    flags = (
        'stage = "knowledge_effect_dataset_expansion_v2"',
        "new_datasets = 5",
        'critic_version = "v6.1"',
        "critic_frozen = true",
        'knowledge_label_method = "deterministic_leave_one_out"',
        "seeds = [42,123,2026]",
        "dataset_specific_critic_rules = false",
        "expert_cached = true",
        "learned_critic_trained = false",
        "test_used = false",
    )
    print("\n===== FLAGS =====", flush=True)
    for f in flags:
        print(f, flush=True)


if __name__ == "__main__":
    main()
