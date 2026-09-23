# -*- coding: utf-8 -*-
"""
Batch runner for external baselines under FAIRNESS_PROTOCOL.

  PatchTST / iTransformer / RAFT : 7 datasets x 3 seeds (TRAIN fit, VAL early-stop)
  RAF (rolling, zero-shot)        : 7 datasets x 1 seed (Chronos-t5-small backbone)

Resumable: a (model, dataset, seed) row whose `status` is "OK" in the results CSV
is skipped on re-run. Every run logs full stdout/stderr to logs/<model>/<ds>_s<seed>.log.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "E:/GitHub_Project/chronos-retrieval"
EXTERNAL = "E:/GitHub_Project/_external_repos"
PYTHON = "E:/Miniconda3/envs/myenv/python"
RESULTS_DIR = os.path.join(REPO, "results", "external_baselines")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CSV_PATH = os.path.join(RESULTS_DIR, "external_baseline_results_full.csv")

DATASETS = ["etth1", "etth2", "exchange_rate", "traffic", "weather", "solar", "electricity"]
SEEDS = [42, 123, 2026]

BASELINE_DIRS = {
    "PatchTST": os.path.join(EXTERNAL, "PatchTST", "PatchTST_supervised"),
    "iTransformer": os.path.join(EXTERNAL, "iTransformer"),
    "RAFT": os.path.join(EXTERNAL, "RAFT"),
}

CONFIG_PATHS = {
    "PatchTST": "patchtst_search_space.json",
    "iTransformer": "itransformer_search_space.json",
    "RAFT": "raft_search_space.json",
    "RAF": "raf_search_space.json",
}

CSV_FIELDS = ["dataset", "seed", "model", "mse", "mae", "runtime_sec", "status",
              "implementation", "backbone", "official_or_adapted", "config_path", "result_source"]

METRIC_RE = re.compile(r"mse:([0-9.eE+\-]+), mae:([0-9.eE+\-]+)")
RAF_METRIC_RE = re.compile(r"raw MSE=([0-9.eE+\-]+), raw MAE=([0-9.eE+\-]+)")

_csv_lock = threading.Lock()


def seq_pred(ds: str):
    return (168, 24) if ds in ("etth2", "traffic") else (512, 96)


def root_path(ds: str) -> str:
    # native Windows path (subprocess runs without MSYS path translation)
    return "E:/GitHub_Project/chronos-retrieval/results/multidataset/" + ds


def build_cmd(baseline: str, ds: str, seed: int):
    seq, pred = seq_pred(ds)
    mid = f"batch_{ds}_s{seed}"
    common = ["--root_path", root_path(ds), "--data_path", "series.csv",
              "--features", "S", "--target", "OT", "--freq", "h",
              "--seq_len", str(seq), "--label_len", "48", "--pred_len", str(pred),
              "--enc_in", "1", "--dec_in", "1", "--c_out", "1"]
    if baseline == "PatchTST":
        return [PYTHON, "run_longExp.py", "--is_training", "1", "--model", "PatchTST",
                "--model_id", mid, "--data", "chronos_retrieval", *common,
                "--e_layers", "3", "--d_model", "128", "--n_heads", "16", "--d_ff", "256",
                "--dropout", "0.2", "--fc_dropout", "0.2", "--head_dropout", "0.0",
                "--patch_len", "16", "--stride", "8", "--individual", "0",
                "--revin", "1", "--affine", "0", "--subtract_last", "0",
                "--train_epochs", "100", "--batch_size", "128", "--patience", "100",
                "--learning_rate", "0.0001", "--lradj", "type3", "--itr", "1",
                "--random_seed", str(seed), "--inverse", "--num_workers", "0", "--gpu", "0"]
    if baseline == "iTransformer":
        return [PYTHON, "run.py", "--is_training", "1", "--model", "iTransformer",
                "--model_id", mid, "--data", "chronos_retrieval", *common,
                "--e_layers", "3", "--d_model", "512", "--n_heads", "8", "--d_ff", "512",
                "--dropout", "0.1",
                "--train_epochs", "100", "--batch_size", "32", "--patience", "100",
                "--learning_rate", "0.0001", "--lradj", "type1", "--itr", "1",
                "--seed", str(seed), "--inverse", "--num_workers", "0", "--gpu", "0"]
    if baseline == "RAFT":
        return [PYTHON, "run.py", "--is_training", "1", "--model", "RAFT",
                "--model_id", mid, "--data", "chronos_retrieval", *common,
                "--n_period", "3", "--topm", "20",
                "--train_epochs", "100", "--batch_size", "32", "--patience", "100",
                "--learning_rate", "0.0001", "--lradj", "type1", "--itr", "1",
                "--seed", str(seed), "--inverse", "--num_workers", "0", "--gpu", "0"]
    raise ValueError(baseline)


def parse_metric(text: str, kind: str):
    if kind == "RAF":
        m = RAF_METRIC_RE.findall(text)
        if not m:
            return None, None
        return float(m[-1][0]), float(m[-1][1])
    m = METRIC_RE.findall(text)
    if not m:
        return None, None
    return float(m[-1][0]), float(m[-1][1])


def run_one(baseline: str, ds: str, seed: int) -> dict:
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, baseline, f"{ds}_s{seed}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if baseline == "RAF":
        cmd = [PYTHON, "experiments/external_baselines/raf_rolling.py",
               "--dataset", ds, "--model", "amazon/chronos-t5-small",
               "--max_windows", "0", "--device", "cuda"]
        cwd = REPO
    else:
        cmd = build_cmd(baseline, ds, seed)
        cwd = BASELINE_DIRS[baseline]

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    runtime = time.time() - t0
    text = open(log_path, encoding="utf-8", errors="replace").read()
    mse, mae = parse_metric(text, baseline)
    status = "OK" if (proc.returncode == 0 and mse is not None) else "FAILED"

    row = {
        "dataset": ds, "seed": str(seed), "model": baseline,
        "mse": "" if mse is None else f"{mse:.6f}",
        "mae": "" if mae is None else f"{mae:.6f}",
        "runtime_sec": f"{runtime:.1f}", "status": status,
        "implementation": ("official chronos-raf.py (rolling adaptation)" if baseline == "RAF"
                           else "official repo + chronos_retrieval loader + --inverse"),
        "backbone": "amazon/chronos-t5-small" if baseline == "RAF" else baseline,
        "official_or_adapted": "official",
        "config_path": CONFIG_PATHS[baseline],
        "result_source": os.path.relpath(log_path, RESULTS_DIR).replace("\\", "/"),
    }
    return row


def load_completed():
    completed = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "OK":
                    completed.add((r["model"], r["dataset"], r["seed"]))
    return completed


def upsert_row(row: dict):
    """One row per (model, dataset, seed): drop any prior row with the same key."""
    with _csv_lock:
        rows = []
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    if not (r["model"] == row["model"] and r["dataset"] == row["dataset"]
                            and r["seed"] == row["seed"]):
                        rows.append(r)
        rows.append(row)
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", nargs="+", default=["PatchTST", "iTransformer", "RAFT", "RAF"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--raf-seeds", nargs="+", type=int, default=[42],
                    help="RAF has no trainable params; seed only affects Chronos sampling")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = []
    for b in args.baselines:
        seeds = args.raf_seeds if b == "RAF" else args.seeds
        for ds in args.datasets:
            for s in seeds:
                tasks.append((b, ds, s))

    completed = load_completed()
    todo = [t for t in tasks if (t[0], t[1], str(t[2])) not in completed]
    print(f"tasks={len(tasks)} completed={len(completed)} todo={len(todo)}")

    if args.dry_run:
        for b, ds, s in todo:
            if b == "RAF":
                print(f"[RAF] {ds} s{s}  (rolling, chronos-t5-small)")
            else:
                print(f"[{b}] {ds} s{s}  {' '.join(build_cmd(b, ds, s))}")
        return

    if not todo:
        print("nothing to do.")
        return

    def _run(t):
        b, ds, s = t
        tag = f"[{b}] {ds} s{s}"
        print(f"START {tag}", flush=True)
        try:
            row = run_one(b, ds, s)
            upsert_row(row)
            print(f"DONE  {tag}  status={row['status']} mse={row['mse']} ({row['runtime_sec']}s)",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {tag}: {e!r}", flush=True)
            upsert_row({"dataset": ds, "seed": str(s), "model": b, "mse": "", "mae": "",
                        "runtime_sec": "", "status": "ERROR",
                        "implementation": "", "backbone": b,
                        "official_or_adapted": "", "config_path": CONFIG_PATHS.get(b, ""),
                        "result_source": ""})
        return b, ds, s

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(_run, t) for t in todo]
        for _ in as_completed(futs):
            pass

    print("batch finished.")


if __name__ == "__main__":
    main()
