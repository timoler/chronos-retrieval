"""Recompute the published January 2021 Weather results from frozen forecasts."""
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TABLES = HERE.parent / "evidence" / "weather_2021_01"
MODELS = ("Chronos-2", "PatchTST", "iTransformer", "RAFT")
ARMS = ("Base", "Retrieval without experts", "Raw Expert", "PAER")
METRICS = ("mse", "mae", "filled_mse", "filled_mae")
SEEDS = (42, 123, 2026)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def by_key(rows, columns):
    result = {}
    for row in rows:
        key = tuple(row[c] for c in columns)
        require(key not in result, f"Duplicate table key {key}")
        result[key] = row
    return result


def compare(actual, published):
    np.testing.assert_allclose(actual, float(published), atol=1e-10, rtol=1e-11)


def observations(spec):
    raw = HERE / "observations" / "mpi_roof_2021a.zip"
    with zipfile.ZipFile(raw) as archive:
        with archive.open("mpi_roof_2021a.csv") as source:
            rows = csv.DictReader(io.TextIOWrapper(source, encoding="iso-8859-1", newline=""))
            begin = datetime.fromisoformat(spec["period_start_exclusive"])
            end = datetime.fromisoformat(spec["period_end_inclusive"])
            times, values = [], []
            for row in rows:
                instant = datetime.strptime(row["Date Time"], "%d.%m.%Y %H:%M:%S")
                if begin < instant <= end:
                    times.append(instant)
                    values.append(float(row["CO2 (ppm)"]))
    times = np.asarray(times, dtype="datetime64[ns]")
    series = np.asarray(values, dtype=np.float32)
    require(len(series) == spec["observation_rows"], "Observation count differs")
    require(np.all(np.diff(times) == np.timedelta64(10, "m")), "Irregular measurement times")
    require(np.isfinite(series).all() and not np.any(series == -9999), "Missing measurement")
    context, horizon = spec["L"], spec["H"]
    windows = np.lib.stride_tricks.sliding_window_view(series, context + horizon)
    y = windows[:, context:]
    with np.load(HERE / "observations" / "labels.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["y"], y)
        np.testing.assert_array_equal(saved["starts"], np.arange(len(y)))
        np.testing.assert_array_equal(saved["origins"], times[context:context + len(y)])
        require(saved["observed"].all(), "Observed target mask differs")
    require(y.shape == (spec["windows"], horizon), "Window count or horizon differs")
    return y.astype(np.float64)


def validation_weights():
    groups = defaultdict(list)
    for row in read_rows(HERE / "val_grid.csv"):
        key = row["model"], row["arm"], int(row["seed"])
        groups[key].append((float(row["mse"]), float(row["lambda_value"])))
    stored = read_rows(HERE / "frozen_weights.csv")
    grid = {0, .05, .1, .15, .2, .3, .5, .7, 1}
    require(len(groups) == len(stored) == 36, "Validation configurations differ")
    selected = {}
    for row in stored:
        key = row["model"], row["arm"], int(row["seed"])
        entries = groups[key]
        require(key not in selected, f"Duplicate frozen weight {key}")
        require(len(entries) == len(grid) and {w for _, w in entries} == grid,
                f"Validation grid differs for {key}")
        weight = min(entries)[1]
        require(weight == float(row["lambda_value"]), f"Frozen selection differs for {key}")
        selected[key] = weight
    return selected


def load_predictions(name, spec, shape):
    path = HERE / spec["prediction_files"][name]
    expected = {"shared"} if name == "Chronos-2" else {f"seed{s}" for s in SEEDS}
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected, f"Seed members differ for {name}")
        result = {}
        for seed in SEEDS:
            key = "shared" if name == "Chronos-2" else f"seed{seed}"
            values = archive[key]
            require(values.shape == shape and np.isfinite(values).all(),
                    f"Invalid forecast array for {name}/{seed}")
            result[seed] = values.astype(np.float64)
        return result


def block_interval(values, bootstrap):
    """Sample contiguous origin blocks, using the same draws for each contrast."""
    length, n = bootstrap["block_length"], len(values)
    rng = np.random.default_rng(bootstrap["seed"])
    samples = np.zeros(bootstrap["replicates"])
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    for offset in range(0, n, length):
        size = min(length, n - offset)
        first = rng.integers(0, n - length + 1, size=len(samples))
        samples += cumulative[first + size] - cumulative[first]
    return np.quantile(samples / n, bootstrap["quantiles"])


def verify():
    manifest = json.loads((HERE / "SHA256.json").read_text(encoding="utf8"))
    for relative, expected in manifest.items():
        require(hashlib.sha256((HERE / relative).read_bytes()).hexdigest() == expected,
                f"Digest mismatch for {relative}")
    spec = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    y = observations(spec)
    weights = validation_weights()
    individual = by_key(read_rows(TABLES / "per_seed.csv"), ("model", "arm", "seed"))
    summary = by_key(read_rows(TABLES / "summary.csv"), ("model", "arm"))
    comparisons = by_key(read_rows(TABLES / "comparisons.csv"), ("model", "reference", "metric"))
    require((len(individual), len(summary), len(comparisons)) == (48, 16, 24),
            "Published table dimensions differ")
    memories = {arm: load_predictions(arm, spec, y.shape) for arm in ARMS[1:]}
    for model in MODELS:
        base = load_predictions(model, spec, y.shape)
        means, origin_loss = {}, {}
        for arm in ARMS:
            scores, losses = [], []
            for seed in SEEDS:
                weight = 0.0 if arm == "Base" else weights[model, arm, seed]
                pred = base[seed]
                if arm != "Base":
                    pred = (1 - weight) * pred + weight * memories[arm][seed]
                error = pred - y
                loss = np.column_stack(((error ** 2).mean(1), np.abs(error).mean(1)))
                losses.append(loss)
                scores.append(loss.mean(0))
                reported = individual[model, arm, str(seed)]
                require(reported["dataset"] == "weather" and int(reported["windows"]) == len(y)
                        and int(reported["observed_cells"]) == y.size, "Unexpected sample coverage")
                compare(weight, reported["lambda"])
                for i, metric in enumerate(METRICS):
                    compare(scores[-1][i % 2], reported[metric])
            means[arm] = np.mean(scores, axis=0)
            origin_loss[arm] = np.mean(losses, axis=0)
            sd = np.std(scores, axis=0, ddof=1)
            for i, metric in enumerate(METRICS):
                compare(means[arm][i % 2], summary[model, arm][metric])
                compare(sd[i % 2], summary[model, arm][metric + "_sd"])
        for ref in ARMS[:-1]:
            for i, metric in enumerate(("mse", "mae")):
                record = comparisons[model, ref, metric]
                delta = means["PAER"][i] - means[ref][i]
                compare(delta, record["paer_minus_reference"])
                compare(-100 * delta / means[ref][i], record["ri_pct"])
                low, high = block_interval(origin_loss["PAER"][:, i] - origin_loss[ref][:, i],
                                           spec["bootstrap"])
                compare(low, record["ci_low"])
                compare(high, record["ci_high"])
    print("Verified 4,464 measurements, 3,857 forecast origins and 36 validation weights.")
    print("Reproduced 48 seed rows, 16 summaries and 24 paired time-block intervals.")


if __name__ == "__main__":
    verify()
