#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned Critic 第二阶段 — 数据集下载与整理.

只做【下载 + 整理 + 检查】，不做任何训练 / Critic / Expert / forecasting / split。

目标: 在现有 5 个数据集 (ETTh1 / Electricity / Solar / Weather / Exchange Rate)
基础上, 补齐到约 15 个公开真实时间序列数据集。

数据来源 (官方/公开稳定入口):
- ETT 系列: 官方 ETDataset repository (本项目 data/ 已存在, 不重复下载)
- Monash Time Series Forecasting Repository (经 Hugging Face 官方镜像
  `Monash-University/monash_tsf`, 原始 .tsf 结构)
- NYC Taxi: GluonTS 官方多变量 benchmark (mbohlkeschneider/gluon-ts mv_release)

行为:
- 自动建目录; 已存在的文件跳过 (断点/可重跑)
- 单个数据集下载失败只记 FAILED, 不中断整程序
- 保留原始数据文件; 不修改数值; 不做 train/val/test split
- 已存在的 9 个数据集不重复下载, 记 ALREADY_EXISTS 并保留当前路径

输出:
- data/learned_critic_datasets/dataset_registry.json
- data/learned_critic_datasets/download_report.md
"""
import fnmatch
import json
import os
import tarfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import requests

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
PROXY = os.environ.get("LC_PROXY", "http://127.0.0.1:7890")
OUT_ROOT = Path("data/learned_critic_datasets")

HF_BASE = "https://huggingface.co/datasets/Monash-University/monash_tsf/resolve/main/data/"

# 采样间隔字符串 -> 大致天数 (仅用于计算近似 end_time)
FREQ_DAYS = {
    "yearly": 365, "quarterly": 91, "monthly": 30, "weekly": 7, "daily": 1,
    "hourly": 1.0 / 24, "half_hourly": 1.0 / 48, "30min": 1.0 / 48,
    "15min": 1.0 / 96, "10min": 1.0 / 144, "minutely": 1.0 / 1440,
    "1min": 1.0 / 1440, "4sec": 4.0 / 86400,
}


# --------------------------------------------------------------------------- #
# dataset 定义
# --------------------------------------------------------------------------- #
# 每个条目: name, source, source_url, 以及 either local_path(已存在) 或 files(待下载).
DATASETS = [
    # ---- 已存在 (不重复下载) ----
    {"name": "ETTh1", "source": "ETDataset (official ETT repository)",
     "source_url": "https://github.com/zhouhaoyi/ETDataset", "local_path": "data/ETTh1.csv"},
    {"name": "ETTh2", "source": "ETDataset (official ETT repository)",
     "source_url": "https://github.com/zhouhaoyi/ETDataset", "local_path": "data/ETTh2.csv"},
    {"name": "ETTm1", "source": "ETDataset (official ETT repository)",
     "source_url": "https://github.com/zhouhaoyi/ETDataset", "local_path": "data/ETTm1.csv"},
    {"name": "ETTm2", "source": "ETDataset (official ETT repository)",
     "source_url": "https://github.com/zhouhaoyi/ETDataset", "local_path": "data/ETTm2.csv"},
    {"name": "Electricity", "source": "UCI Electricity Load Diagrams",
     "source_url": "https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014",
     "local_path": "data/electricity.csv"},
    {"name": "Solar", "source": "NREL Solar-Energy (Alabama)",
     "source_url": "https://www.nrel.gov/grid/solar-power-data.html",
     "local_path": "data/solar.csv"},
    {"name": "Traffic", "source": "San Francisco Bay Area Traffic (PeMS)",
     "source_url": "https://github.com/laiguokun/multivariate-time-series-data",
     "local_path": "data/traffic.csv"},
    {"name": "Weather", "source": "Max-Planck Biogeochemistry Weather",
     "source_url": "https://www.bgc-jena.mpg.de/wetter/", "local_path": "data/weather.csv"},
    {"name": "ExchangeRate", "source": "Exchange Rate (multivariate)",
     "source_url": "https://github.com/laiguokun/multivariate-time-series-data",
     "local_path": "data/exchange_rate.csv"},

    # ---- 待下载 (Monash via HuggingFace 官方镜像) ----
    {"name": "LondonSmartMeters", "source": "Monash Time Series Forecasting Repository",
     "source_url": HF_BASE + "london_smart_meters_dataset_with_missing_values.zip",
     "files": [{"url": HF_BASE + "london_smart_meters_dataset_with_missing_values.zip",
                "archive": "zip"}]},
    {"name": "AustralianElectricityDemand", "source": "Monash Time Series Forecasting Repository",
     "source_url": HF_BASE + "australian_electricity_demand_dataset.zip",
     "files": [{"url": HF_BASE + "australian_electricity_demand_dataset.zip", "archive": "zip"}]},
    {"name": "WindFarms", "source": "Monash Time Series Forecasting Repository",
     "source_url": HF_BASE + "wind_farms_minutely_dataset_with_missing_values.zip",
     "files": [{"url": HF_BASE + "wind_farms_minutely_dataset_with_missing_values.zip",
                "archive": "zip"}]},
    {"name": "PedestrianCounts", "source": "Monash Time Series Forecasting Repository",
     "source_url": HF_BASE + "pedestrian_counts_dataset.zip",
     "files": [{"url": HF_BASE + "pedestrian_counts_dataset.zip", "archive": "zip"}]},
    {"name": "Tourism", "source": "Monash Time Series Forecasting Repository",
     "source_url": HF_BASE + "tourism_monthly_dataset.zip",
     "files": [
         {"url": HF_BASE + "tourism_monthly_dataset.zip", "archive": "zip"},
         {"url": HF_BASE + "tourism_quarterly_dataset.zip", "archive": "zip"},
         {"url": HF_BASE + "tourism_yearly_dataset.zip", "archive": "zip"},
     ]},

    # ---- 待下载 (NYC Taxi, GluonTS 多变量 benchmark) ----
    {"name": "Taxi", "source": "NYC Taxi (GluonTS multivariate benchmark)",
     "source_url": "https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release/datasets",
     "files": [{"url": "https://raw.githubusercontent.com/mbohlkeschneider/gluon-ts/"
                       "mv_release/datasets/taxi_30min.tar.gz", "archive": "tar.gz"}]},
]


# --------------------------------------------------------------------------- #
# download / extract (with resume & failure isolation)
# --------------------------------------------------------------------------- #
_session = None


def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        if PROXY:
            _session.proxies = {"http": PROXY, "https": PROXY}
    return _session


def download(url: str, dest: Path):
    """下载单个文件; 已存在则跳过. 返回 (status, path)."""
    if dest.exists() and dest.stat().st_size > 0:
        return "SKIPPED", dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with get_session().get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
        return "SUCCESS", dest
    except Exception as e:  # noqa: BLE001
        print(f"    [download FAILED] {url}\n    {e!r}")
        return "FAILED", dest


def extract(path: Path, out_dir: Path, archive: str):
    """解压 zip / tar.gz 到 out_dir; 返回解压出的文件路径列表."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive == "zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            z.extractall(out_dir)
        return [out_dir / n for n in names if not Path(n).name.startswith("._")]
    if archive == "tar.gz":
        with tarfile.open(path, "r:gz") as t:
            t.extractall(out_dir)
            names = [m.name for m in t.getmembers() if m.isfile()]
        return [out_dir / n for n in names if not Path(n).name.startswith("._")]
    raise ValueError(f"unknown archive type: {archive}")


# --------------------------------------------------------------------------- #
# inspect (数据检查)
# --------------------------------------------------------------------------- #
def _shift_days(start_str: str, days: float) -> str:
    try:
        base = datetime.strptime(start_str[:10], "%Y-%m-%d")
        return (base + timedelta(days=days)).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return ""


def _looks_like_timestamp(s: str) -> bool:
    s = s.strip()
    return len(s) >= 10 and s[4] == "-" and s[7] == "-"


def parse_tsf(path: Path) -> dict:
    """解析 Monash .tsf, 返回检查信息. 保留原结构, 只统计."""
    meta = {"attribute": []}
    series = []  # (name, start_str, values_list)
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
                parts = line.split(" ", 1)
                key = parts[0][1:]
                val = parts[1].strip() if len(parts) > 1 else ""
                if key == "attribute":
                    meta["attribute"].append(val)
                else:
                    meta[key] = val
                continue
            if in_data:
                # 数据行形如: series_name[:attr...]:start_timestamp:value1,value2,...
                # 其中 timestamp 字段本身不含冒号 (Monash 用 - 分隔: 2002-01-01 00-00-00).
                segs = line.split(":")
                name = segs[0]
                start = ""
                for s in segs[1:-1]:
                    if _looks_like_timestamp(s):
                        start = s
                        break
                vals = segs[-1]
                series.append((name, start, [v.strip() for v in vals.split(",")]))

    n_series = len(series)
    lengths = [len(v) for _, _, v in series]
    n_values = sum(lengths)
    missing = sum(1 for _, _, v in series for x in v if x in ("", "?"))
    starts = [s for _, s, _ in series if s]
    freq = meta.get("frequency", "")

    start_time = min(starts) if starts else ""
    end_time = ""
    if start_time and lengths and freq in FREQ_DAYS:
        end_time = _shift_days(start_time, FREQ_DAYS[freq] * (max(lengths) - 1))

    return {
        "file_format": "tsf", "n_series": n_series, "n_rows": n_series,
        "total_observations": n_values, "series_length_max": max(lengths) if lengths else 0,
        "n_columns": 2, "time_column": "start_timestamp", "start_time": start_time,
        "end_time": end_time, "sampling_interval": freq, "missing_values": missing,
        "numeric_columns": 1, "equallength": meta.get("equallength", ""),
    }


def parse_csv(path: Path) -> dict:
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    time_col = None
    for c in ("date", "Date", "datetime", "timestamp", "start_timestamp"):
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        for c in df.columns[:5]:
            try:
                pd.to_datetime(df[c])
                time_col = c
                break
            except Exception:  # noqa: BLE001
                continue

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    start_time = end_time = sampling = ""
    if time_col is not None:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        valid = ts.dropna()
        if len(valid):
            start_time = str(valid.min())
            end_time = str(valid.max())
            if len(valid) > 1:
                diffs = valid.diff().dropna()
                if len(diffs):
                    sampling = str(diffs.median())

    return {
        "file_format": "csv", "n_series": len(numeric_cols), "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]), "time_column": time_col or "",
        "start_time": start_time, "end_time": end_time, "sampling_interval": sampling,
        "missing_values": int(df.isna().sum().sum()), "numeric_columns": len(numeric_cols),
    }


def parse_gluonts_jsonl(root: Path) -> dict:
    """解析 GluonTS 格式 (metadata.json + train/test .json JSONL). 递归查找."""
    import pandas as pd

    meta_files = list(root.rglob("metadata.json"))
    freq, target_dim = "", None
    if meta_files:
        with open(meta_files[0], "r", encoding="utf-8") as f:
            m = json.load(f)
        freq = m.get("time_granularity", "")
        target_dim = m.get("target_dim")

    def _count(p: Path):
        n = 0
        first_start = ""
        max_len = 0
        n_missing = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    obj = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not first_start:
                    first_start = obj.get("start", "")
                tgt = obj.get("target", [])
                max_len = max(max_len, len(tgt))
                n_missing += sum(1 for x in tgt
                                 if x is None or (isinstance(x, float) and pd.isna(x)))
        return n, first_start, max_len, n_missing

    train_files = list(root.rglob("train.json"))
    test_files = list(root.rglob("test.json"))
    n_train = n_test = 0
    train_start = test_start = ""
    train_len = 0
    n_missing = 0
    if train_files:
        n_train, train_start, train_len, n_missing = _count(train_files[0])
    if test_files:
        n_test, test_start, _, tm = _count(test_files[0])
        n_missing += tm

    n_series = target_dim if target_dim else n_train
    start_time = train_start
    end_time = ""
    if start_time and train_len and freq in FREQ_DAYS:
        end_time = _shift_days(start_time, FREQ_DAYS[freq] * (train_len - 1))

    note = (f"原始已分 train/test (未做新 split); train={n_train} series"
            f"({train_start} 起), test={n_test} window entries({test_start} 起)")
    return {
        "file_format": "gluonts_jsonl", "n_series": n_series, "n_rows": n_train,
        "total_observations": None, "n_columns": 1, "time_column": "start",
        "start_time": start_time, "end_time": end_time, "sampling_interval": freq,
        "missing_values": n_missing, "numeric_columns": 1, "note": note,
    }


def inspect_dataset(ds_dir: Path) -> tuple:
    """按目录内文件类型分发检查. 返回 (primary_path, info)."""
    tsf_files = sorted(ds_dir.rglob("*.tsf"))
    if tsf_files:
        return tsf_files[0], parse_tsf(tsf_files[0])
    if list(ds_dir.rglob("metadata.json")):
        return ds_dir, parse_gluonts_jsonl(ds_dir)
    csv_files = sorted(ds_dir.rglob("*.csv"))
    if csv_files:
        return csv_files[0], parse_csv(csv_files[0])
    return None, {}


# --------------------------------------------------------------------------- #
# registry / report
# --------------------------------------------------------------------------- #
def build_registry(results):
    out = []
    for r in results:
        info = r.get("info", {})
        out.append({
            "dataset_name": r["name"],
            "source": r["source"],
            "source_url": r["source_url"],
            "local_path": r.get("local_path", ""),
            "file_format": info.get("file_format", ""),
            "n_rows": info.get("n_rows"),
            "n_columns": info.get("n_columns"),
            "numeric_columns": info.get("numeric_columns"),
            "time_column": info.get("time_column", ""),
            "start_time": info.get("start_time", ""),
            "end_time": info.get("end_time", ""),
            "sampling_interval": info.get("sampling_interval", ""),
            "missing_values": info.get("missing_values"),
            "number_of_series": info.get("n_series"),
            "total_observations": info.get("total_observations"),
            "series_length_max": info.get("series_length_max"),
            "file_size_bytes": r.get("size_bytes"),
            "download_status": r["status"],
            "note": info.get("note", ""),
        })
    return out


def build_report(results):
    L = ["# Learned Critic 第二阶段 — 数据集下载与整理报告\n", "## 结果\n",
         "| Dataset | Status | Rows | Series | Frequency | Time Range | Local Path |",
         "|---|---|---|---|---|---|---|"]
    ok = fail = exist = 0
    for r in results:
        info = r.get("info", {})
        tr = f"{info.get('start_time', '')} ~ {info.get('end_time', '')}".strip(" ~")
        L.append(f"| {r['name']} | {r['status']} | {info.get('n_rows', '')} | "
                 f"{info.get('n_series', '')} | {info.get('sampling_interval', '')} | "
                 f"{tr} | {r.get('local_path', '')} |")
        if r["status"] == "SUCCESS":
            ok += 1
        elif r["status"] == "FAILED":
            fail += 1
        elif r["status"] == "ALREADY_EXISTS":
            exist += 1
    total = ok + exist
    L += ["", "## 结论", "",
          f"- 成功下载: {ok} 个", f"- 失败: {fail} 个",
          f"- 已有 (未重复下载): {exist} 个",
          f"- 当前可用真实数据集总数: {total} 个", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for ds in DATASETS:
        name = ds["name"]
        print(f"\n=== [{name}] ===", flush=True)

        # ---- 已存在: 只检查, 不下载 ----
        if "local_path" in ds:
            p = Path(ds["local_path"])
            status = "ALREADY_EXISTS" if p.exists() else "MISSING"
            info = parse_csv(p) if p.exists() else {}
            size = p.stat().st_size if p.exists() else None
            print(f"    {status}: {ds['local_path']} "
                  f"(n_series={info.get('n_series')}, freq={info.get('sampling_interval')})")
            results.append({"name": name, "source": ds["source"],
                            "source_url": ds["source_url"], "local_path": ds["local_path"],
                            "status": status, "info": info, "size_bytes": size})
            continue

        # ---- 待下载 ----
        ds_dir = OUT_ROOT / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        status = "SUCCESS"

        for f in ds["files"]:
            dest = ds_dir / Path(f["url"]).name
            st, path = download(f["url"], dest)
            if st == "FAILED":
                status = "FAILED"
                continue
            try:
                extract(path, ds_dir, f["archive"])
            except Exception as e:  # noqa: BLE001
                print(f"    [extract FAILED] {path}\n    {e!r}")
                status = "FAILED"

        # 检查
        primary, info = None, {}
        size = None
        try:
            primary, info = inspect_dataset(ds_dir)
        except Exception as e:  # noqa: BLE001
            print(f"    [inspect FAILED] {e!r}")
            status = "FAILED"

        if primary is None:
            status = "FAILED"
        else:
            if isinstance(primary, Path) and primary.is_dir():
                size = sum(f.stat().st_size for f in primary.rglob("*") if f.is_file())
            else:
                size = primary.stat().st_size if isinstance(primary, Path) else None
            # "with_missing" 变体在 note 里标注
            if "with_missing" in ds.get("source_url", "") and not info.get("note"):
                info["note"] = "Monash with_missing 变体 (保留缺失, 未填补)"
            print(f"    primary: {primary}")
            print(f"    -> n_series={info.get('n_series')} rows={info.get('n_rows')} "
                  f"freq={info.get('sampling_interval')} missing={info.get('missing_values')}")

        results.append({"name": name, "source": ds["source"],
                        "source_url": ds["source_url"],
                        "local_path": str(primary) if primary else "",
                        "status": status, "info": info, "size_bytes": size})

    # ---- 输出 registry + report ----
    registry = build_registry(results)
    (OUT_ROOT / "dataset_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "download_report.md").write_text(build_report(results), encoding="utf-8")

    print("\n===== 完成 =====", flush=True)
    print(f"registry: {OUT_ROOT / 'dataset_registry.json'}", flush=True)
    print(f"report:   {OUT_ROOT / 'download_report.md'}", flush=True)


if __name__ == "__main__":
    main()
