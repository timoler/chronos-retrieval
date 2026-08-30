#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下载 Traffic benchmark 数据集到 data/traffic.csv

数据来源：
laiguokun/multivariate-time-series-data
原始 Traffic benchmark：862 个传感器、17544 个小时级时间点。

本脚本：
1. 下载 traffic.txt.gz
2. 解压
3. 添加 date 列
4. 保存为 data/traffic.csv

不会修改你已有的 ETT / Electricity / Solar 数据。
"""

from pathlib import Path
import gzip
import shutil
import urllib.request
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/laiguokun/"
    "multivariate-time-series-data/master/traffic/traffic.txt.gz"
)

OUT_DIR = Path("data")
GZ_PATH = OUT_DIR / "_raw" / "traffic.txt.gz"
TXT_PATH = OUT_DIR / "_raw" / "traffic.txt"
CSV_PATH = OUT_DIR / "traffic.csv"

EXPECTED_ROWS = 17544
EXPECTED_SENSORS = 862


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"[skip] 已存在: {path}")
        return

    print(f"[download] {url}")
    urllib.request.urlretrieve(url, path)
    print(f"[ok] 下载完成: {path}")


def extract(gz_path: Path, txt_path: Path):
    if txt_path.exists():
        print(f"[skip] 已解压: {txt_path}")
        return

    print("[extract] 正在解压...")
    with gzip.open(gz_path, "rb") as src, open(txt_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    print(f"[ok] 解压完成: {txt_path}")


def convert(txt_path: Path, csv_path: Path):
    print("[convert] 读取 Traffic 数据...")
    df = pd.read_csv(txt_path, header=None)

    print(f"原始 shape = {df.shape}")

    if df.shape[0] != EXPECTED_ROWS:
        print(
            f"[warning] 行数预期 {EXPECTED_ROWS}，实际 {df.shape[0]}"
        )

    if df.shape[1] != EXPECTED_SENSORS:
        print(
            f"[warning] 传感器数预期 {EXPECTED_SENSORS}，实际 {df.shape[1]}"
        )

    # benchmark 常用小时级时间索引
    dates = pd.date_range(
        start="2015-01-01 00:00:00",
        periods=len(df),
        freq="h",
    )

    df.columns = [f"Sensor_{i:03d}" for i in range(df.shape[1])]
    df.insert(0, "date", dates)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"[ok] CSV 保存完成: {csv_path}")
    print(f"最终 shape = {df.shape}")
    print(f"时间范围 = {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    print(f"传感器数量 = {df.shape[1] - 1}")


def main():
    print("=" * 70)
    print("Traffic Dataset Downloader")
    print("=" * 70)

    download(URL, GZ_PATH)
    extract(GZ_PATH, TXT_PATH)
    convert(TXT_PATH, CSV_PATH)

    print("\n完成。")
    print(f"后续实验直接使用: {CSV_PATH}")


if __name__ == "__main__":
    main()
