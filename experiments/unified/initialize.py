# -*- coding: utf-8 -*-
"""Create an empty, independently frozen run. Does not train or enable LLM calls.

Paths in the supplied JSON are relative to that JSON's directory. Invoke this
file from the repository you intend to execute; its own location fixes the code
root. Subsequent workers must use the printed run_config.json, not the template.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ARCHITECTURES = (
    "PatchTST/PatchTST_supervised/models/PatchTST.py",
    "iTransformer/model/iTransformer.py",
    "RAFT/models/RAFT.py",
)
DATA_FILES = ("series.csv", "window_indices.npz", "config.json")
SEMANTIC_KEYS = (
    "id", "mechanism", "scope", "activation_condition", "failure_condition",
    "observables", "retrieval_action", "expected_effect", "reason",
    "condition_spec", "replaced",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_new(path, value):
    """Exclusive creation makes accidental replacement an error."""
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def overlaps(left, right):
    left, right = left.resolve(), right.resolve()
    return left == right or left in right.parents or right in left.parents


def require(condition, message):
    if not condition:
        raise ValueError(message)


def semantic(pool):
    """Matches the frozen reward_core.semantic projection, without GPU imports."""
    items = list(pool.values()) if isinstance(pool, dict) else list(pool)
    items.sort(key=lambda e: int(str(e.get("id", e.get("expert_id", "0"))).lstrip("E") or 0))
    result = []
    for expert in items:
        row = {key: expert[key] for key in SEMANTIC_KEYS if key in expert}
        row["claim"] = expert.get("mechanism", expert.get("claim", expert.get("reason", "")))
        row["observable_features"] = list(expert.get("observables", expert.get("observable_features", [])))
        result.append(row)
    return result


def inspect_dataset(root, dataset):
    import numpy as np

    directory = root / "data" / "processed" / dataset
    cfg = read_json(directory / "config.json")
    context, horizon = int(cfg["context"]), int(cfg["horizon"])
    require(context >= 48 and horizon > 0, dataset + ": invalid context/horizon")
    count = 0
    with (directory / "series.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require({"date", "OT"} <= set(reader.fieldnames or []), dataset + ": date/OT columns absent")
        for row in reader:
            require(bool(row["date"]) and math.isfinite(float(row["OT"])), dataset + ": invalid date/OT")
            count += 1
    require(count == int(cfg["rows"]), dataset + ": row count differs from config")
    counts, intervals = {}, []
    with np.load(directory / "window_indices.npz", allow_pickle=False) as archive:
        for split in ("train", "val", "test"):
            starts = archive[split]
            require(starts.ndim == 1 and np.issubdtype(starts.dtype, np.integer), dataset + ": invalid index dtype")
            lo, hi = map(int, cfg["split"][split])
            require(0 <= lo < hi <= count, dataset + ": invalid row interval")
            expected = np.arange(lo, hi - context - horizon + 1, int(cfg.get("stride", 1)))
            require(len(expected) > 0 and np.array_equal(starts, expected),
                    dataset + ": indices differ from complete declared row interval: " + split)
            require(len(starts) == int(cfg["num_windows"][split]), dataset + ": window count differs")
            counts[split] = len(starts)
            intervals.append((lo, hi))
        require(all(intervals[i][1] <= intervals[i + 1][0] for i in range(2)), dataset + ": overlapping splits")
        # Validate the smallest expanding fold and RAFT's non-overlap retrieval bank.
        starts = archive["train"].astype(np.int64)
        edges = np.linspace(len(starts) // 2, len(starts), 5).astype(int)
        for fold in range(4):
            prefix, block = starts[:edges[fold]], starts[edges[fold]:edges[fold + 1]]
            mid = len(block) // 2
            evaluation = block[mid::horizon]
            require(len(evaluation) > 0, dataset + ": empty EVAL fold")
            calibration = block[:mid][block[:mid] + horizon <= evaluation[0]][::horizon]
            require(len(calibration) > 0, dataset + ": empty CAL fold")
            prefix = prefix[prefix + horizon <= calibration[0]]
            cut = int(len(prefix) * .8)
            fit = prefix[cut:]
            require(len(fit) > 0, dataset + ": empty fit fold")
            bank = prefix[:cut][prefix[:cut] + horizon <= fit[0]]
            require(len(bank) > 2 * (context + horizon) - 1 + 20,
                    dataset + ": insufficient RAFT non-overlap bank")
    expert_path = root / "data" / "experts" / dataset / "raw_experts.json"
    raw = semantic(read_json(expert_path))
    require(len(raw) == 8 and len({item["id"] for item in raw}) == 8, dataset + ": expected eight unique experts")
    return {
        "input_hashes": {name: file_hash(directory / name) for name in DATA_FILES},
        "raw_sha256": digest(raw),
    }, {"rows": count, "context": context, "horizon": horizon, "windows": counts}


def initialize(config_path, repository=None):
    repository = (repository or Path(__file__).resolve().parents[2]).resolve()
    config_path = Path(config_path).resolve()
    request = read_json(config_path)

    def configured_path(name):
        value = request.get(name)
        require(isinstance(value, str) and bool(value.strip()), "Missing path: " + name)
        path = Path(value).expanduser()
        return (path if path.is_absolute() else config_path.parent / path).resolve()

    canonical_path = repository / "configs" / "paer_unified_v1.json"
    protocol = read_json(canonical_path)
    supplied = request.get("protocol")
    if "protocol_file" in request:
        from_file = read_json(configured_path("protocol_file"))
        require(supplied is None or supplied == from_file, "Inline protocol differs from protocol_file")
        supplied = from_file
    require(supplied == protocol, "Protocol must exactly match configs/paer_unified_v1.json")
    names = ("run_root", "report_root", "input_repo", "external_root", "embedding_snapshot", "chronos_snapshot")
    paths = {name: configured_path(name) for name in names}
    run, reports = paths["run_root"], paths["report_root"]
    protected = {"repository": repository, "configuration": config_path, "protocol": canonical_path}
    protected.update({key: paths[key] for key in names if key not in ("run_root", "report_root")})
    for output in (run, reports):
        require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
                "Output must be new or empty; will not overwrite: " + str(output))
        for ancestor in (output, *output.parents):
            prior_run = ((ancestor / "protocol.json").is_file() and
                         (ancestor / "code_manifest.json").is_file())
            prior_run = prior_run or (ancestor / "runner.lock").is_file()
            require(not prior_run, "Output is inside an existing run: " + str(ancestor))
        for label, source in protected.items():
            require(not overlaps(output, source), "Output overlaps protected " + label + ": " + str(output))
    require(not overlaps(run, reports), "run_root and report_root must be disjoint")
    for name in names[2:]:
        require(paths[name].is_dir(), "Directory absent: " + str(paths[name]))
    for relative in ARCHITECTURES:
        require((paths["external_root"] / relative).is_file(), "Architecture file absent: " + relative)
    for name in ("embedding_snapshot", "chronos_snapshot"):
        require((paths[name] / "config.json").is_file(), name + ": config.json absent")
        require(any(paths[name].rglob("*.safetensors")) or any(paths[name].rglob("pytorch_model*.bin")),
                name + ": model weights absent")

    code = {}
    for relative in ("experiments/unified", "paer_core"):
        directory = repository / relative
        require(directory.is_dir(), "Source directory absent: " + relative)
        for file in sorted(directory.rglob("*.py")):
            code[file.relative_to(repository).as_posix()] = file_hash(file)
    code[canonical_path.relative_to(repository).as_posix()] = file_hash(canonical_path)
    require("experiments/unified/runner.py" in code, "Unified runner absent")
    require(any(name.startswith("paer_core/") for name in code), "PAER core source absent")
    inputs, provenance, checks = {}, {}, {}
    embedding_hash = file_hash(paths["embedding_snapshot"] / "config.json")
    for dataset in protocol["datasets"]:
        provenance[dataset], checks[dataset] = inspect_dataset(paths["input_repo"], dataset)
        provenance[dataset]["embedding_config_sha256"] = embedding_hash
        relatives = ["data/processed/" + dataset + "/" + name for name in DATA_FILES]
        relatives.append("data/experts/" + dataset + "/raw_experts.json")
        for relative in relatives:
            inputs[relative] = file_hash(paths["input_repo"] / relative)
    external = {}
    for directory in ("PatchTST", "iTransformer", "RAFT"):
        for file in sorted((paths["external_root"] / directory).rglob("*.py")):
            relative = file.relative_to(paths["external_root"]).as_posix()
            external[relative] = file_hash(file)

    # No output writes occur until all paths, inputs, and split checks pass.
    run.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    require(not any(run.iterdir()) and not any(reports.iterdir()), "Output changed during preflight")
    write_new(run / "INITIALIZING.json", {
        "state": "INITIALIZING", "config_sha256": file_hash(config_path),
        "completion_marker": "INITIALIZED.json",
        "if_completion_marker_absent": "Incomplete initialization; do not execute this directory.",
    })
    snapshot = run / "sources_external"
    for relative, expected in external.items():
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (paths["external_root"] / relative).open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        require(file_hash(destination) == expected, "External source changed while copying: " + relative)
    require(all(file_hash(repository / name) == value for name, value in code.items()), "Code changed during initialization")
    require(all(file_hash(paths["input_repo"] / name) == value for name, value in inputs.items()), "Input changed during initialization")
    normalized = {name: str(path) for name, path in paths.items()}
    normalized["external_root"] = str(snapshot)
    normalized["protocol"] = protocol
    created = datetime.now(timezone.utc).isoformat()
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "python_executable": sys.executable,
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]},
    }
    origin = {
        "created_utc": created, "repository": str(repository), "source_configuration": str(config_path),
        "source_configuration_sha256": file_hash(config_path), "protocol_source": str(canonical_path),
        "requested_paths": {key: str(value) for key, value in paths.items()},
        "external_snapshot": str(snapshot), "model_config_hashes": {
            name: file_hash(paths[name] / "config.json") for name in ("embedding_snapshot", "chronos_snapshot")},
        "scope": "Initialization only: no training, no TEST evaluation, no LLM readiness declaration.",
    }
    artifacts = {
        "protocol.json": protocol, "code_manifest.json": code,
        "input_bytes_manifest.json": inputs, "external_code_manifest.json": external,
        "data_provenance.json": provenance, "PREFLIGHT_CHECKS.json": checks,
        "ENVIRONMENT.json": environment, "PATH_PROVENANCE.json": origin,
        "run_config.json": normalized,
    }
    for name, value in artifacts.items():
        write_new(run / name, value)
    status = {
        "state": "INITIALIZED_NOT_STARTED", "created_utc": created,
        "protocol_hash": digest(protocol), "run_config": str(run / "run_config.json"),
        "test_readout_started": False, "llm_preflight_verified": False,
        "counts": {"code_files": len(code), "input_files": len(inputs), "external_files": len(external)},
    }
    write_new(run / "INITIALIZED.json", status)
    write_new(reports / "INITIALIZATION.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        status = initialize(arguments.config)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, "Initialization refused: " + str(error) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print("Set PAER_RUN_CONFIG to run_config above before separately starting the runner.")


if __name__ == "__main__":
    main()
