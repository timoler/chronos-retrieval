"""Recalculate the published PAER score summaries using only the Python standard library."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("Chronos-2", "PatchTST", "iTransformer", "RAFT")
SEEDS = (42, 123, 2026)
PAER, RAW, DIRECT = "PAER", "Raw Expert", "Revision without acceptance"
ARMS = (PAER, DIRECT, RAW)
METRICS = ("mse", "mae")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(rows, fields, expected):
    data = {}
    for row in rows:
        key = tuple(int(row[k]) if k == "seed" else row[k] for k in fields)
        if key in data:
            raise ValueError(f"Duplicate result {key}")
        data[key] = row
    if set(data) != set(expected):
        raise ValueError(f"Unexpected coverage for {fields}")
    return data


def equal(actual, stored, scale=0):
    if not math.isclose(actual, float(stored), rel_tol=1e-11,
                        abs_tol=max(1e-12, abs(scale) * 2e-14)):
        raise ValueError(f"Recomputed value {actual} differs from recorded value {stored}")


def summarize(values, stored, mean_column, sd_column):
    mean = statistics.fmean(values)
    equal(mean, stored[mean_column])
    equal(statistics.stdev(values), stored[sd_column], mean)
    return mean


def verify(root):
    manifest = json.loads((root / "sources.json").read_text())
    for name, entry in manifest["tables"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["public_sha256"]:
            raise ValueError(f"Table checksum mismatch: {name}")
        if len(read_csv(path)) != entry["records"]:
            raise ValueError(f"Row count mismatch: {name}")
    for name, expected in manifest["metadata"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Metadata checksum mismatch: {name}")

    metadata = {x["dataset"]: x for x in json.loads((root / "datasets.json").read_text())}
    tasks = list(metadata)
    keys = ("dataset", "arm", "model")
    main = indexed(read_csv(root / "main/per_seed.csv"), (*keys, "seed"),
                   itertools.product(tasks, ARMS, MODELS, SEEDS))
    summaries = indexed(read_csv(root / "main/summary.csv"), keys,
                        itertools.product(tasks, ARMS, MODELS))
    means = {}
    counts = {}

    def contrast(name, metric, new, reference):
        ri = 100 * (reference-new) / reference
        counts.setdefault(f"{name} / {metric.upper()}", []).append(ri)
        return ri

    for group, row in summaries.items():
        ds, arm, model = group
        for metric in METRICS:
            for component in ("base", "memory", "fused"):
                values = [float(main[(*group, seed)][f"{component}_test_{metric}"]) for seed in SEEDS]
                key = f"{component}_{metric}"
                means[group, key] = summarize(values, row, key+"_mean", key+"_sd")
            ri = contrast(arm+" vs Base", metric, means[group, "fused_"+metric], means[group, "base_"+metric])
            equal(ri, row[f"{metric}_RI_vs_base_pct"])

    pairs = ((PAER, DIRECT), (PAER, RAW), (DIRECT, RAW))
    compare = indexed(read_csv(root / "main/comparisons.csv"), ("dataset","model","arm","reference"),
                      ((d,m,a,r) for d,m in itertools.product(tasks,MODELS) for a,r in pairs))
    for (ds, model, arm, ref), row in compare.items():
        for metric in METRICS:
            new = means[(ds,arm,model), "fused_"+metric]
            prior = means[(ds,ref,model), "fused_"+metric]
            equal(new,row[metric+"_arm"])
            equal(prior,row[metric+"_reference"])
            equal(new-prior,row[metric+"_difference"],prior)
            equal(contrast(arm+" vs "+ref, metric, new, prior),row[metric+"_RI_pct"])

    memory = indexed(read_csv(root/"main/memory_per_seed.csv"), ("dataset","arm","seed"),
                     itertools.product(tasks,ARMS,SEEDS))
    for (ds,arm,seed), row in memory.items():
        for model,metric in itertools.product(MODELS,METRICS):
            equal(float(row["test_aligned_"+metric]),main[ds,arm,model,seed]["memory_test_"+metric])
    for ds in tasks:
        task=metadata[ds]
        intervals=read_csv(root/"main/block_intervals"/f"{ds}_block_bootstrap.csv")
        horizon,context=task["horizon"],task["context"]
        lengths=tuple(min(length,task["window_counts"]["test"])
                      for length in (horizon,context+horizon,2*(context+horizon)))
        covered=set()
        for row in intervals:
            arm,ref,model,metric=(row[k] for k in ("arm","reference","model","metric"))
            length=int(row["block_length"])
            key=(arm,ref,model,metric,length)
            if row["dataset"]!=ds or key in covered or int(row["replicates"])!=2000:
                raise ValueError(f"Unexpected bootstrap record {ds}, {key}")
            covered.add(key)
            if row["primary_length"]!=str(length==context+horizon):
                raise ValueError(f"Bootstrap primary length mismatch {ds}")
            component="memory_" if model=="memory" else "fused_"
            base=MODELS[0] if model=="memory" else model
            first=means[(ds,arm,base),component+metric]
            second=means[(ds,ref,base),component+metric]
            equal(first-second,row["mean_delta"],max(first,second)*4)
            if not (math.isfinite(float(row["ci_low"])) and math.isfinite(float(row["ci_high"]))
                    and float(row["ci_low"])<=float(row["ci_high"])):
                raise ValueError(f"Invalid time-block interval {key}")
        expected={(a,r,m,k,b) for a,r in pairs for m,k,b in itertools.product((*MODELS,"memory"),METRICS,lengths)}
        if covered!=expected:
            raise ValueError(f"Incomplete bootstrap coverage for {ds}")

    controls = read_csv(root / "controls/per_seed.csv")
    control_tasks = ("etth1","etth2","exchange_rate","traffic","weather")
    control_arms = ("Numeric retrieval","Retrieval without experts",RAW,"Screening only",
                    "Revision only",DIRECT,"Unconditional experts","Shuffled expert text")
    controls = indexed(controls, (*keys,"seed"),itertools.product(control_tasks,control_arms,MODELS,SEEDS))
    for group,row in controls.items():
        if group[1] in (RAW,DIRECT):
            for field in ("lambda_value","selected_val_mse","base_test_mse","fused_test_mse",
                          "base_test_mae","fused_test_mae"):
                equal(float(row[field]), main[group][field])
    for ds,ref,model,metric in itertools.product(control_tasks,
        ("Retrieval without experts","Screening only"), MODELS, METRICS):
        prior=statistics.fmean(float(controls[(ds,ref,model,s)]["fused_test_"+metric]) for s in SEEDS)
        contrast("PAER vs "+ref+" (five tasks)",metric,means[(ds,PAER,model),"fused_"+metric],prior)

    external = root / "weather_2021_01"
    weather_arms = ("Base","Retrieval without experts", RAW, PAER)
    wseed = indexed(read_csv(external/"per_seed.csv"),("model","arm","seed"),
                    itertools.product(MODELS,weather_arms,SEEDS))
    wsumm = indexed(read_csv(external/"summary.csv"),("model","arm"),
                    itertools.product(MODELS,weather_arms))
    weather_means = {}
    for (model,arm),row in wsumm.items():
        for metric in ("mse","mae","filled_mse","filled_mae"):
            values = [float(wseed[model,arm,s][metric]) for s in SEEDS]
            weather_means[model,arm,metric] = summarize(values,row,metric,metric+"_sd")
        for seed in SEEDS:
            record=wseed[model,arm,seed]
            if record["dataset"] != "weather" or int(record["windows"]) != 3857 or int(record["observed_cells"]) != 370272:
                raise ValueError("Expected the registered January 2021 Weather evaluation")
            weight=0 if arm=="Base" else (controls if arm=="Retrieval without experts" else main)["weather",arm,model,seed]["lambda_value"]
            equal(float(weight),record["lambda"])
    wc = indexed(read_csv(external/"comparisons.csv"),("model","reference","metric"),
                 itertools.product(MODELS,weather_arms[:-1],METRICS))
    for (model,ref,metric),row in wc.items():
        value=weather_means[model,PAER,metric]
        prior=weather_means[model,ref,metric]
        equal(value-prior,row["paer_minus_reference"],prior)
        equal(contrast("January PAER vs "+ref,metric,value,prior),row["ri_pct"])

    gates = read_csv(root / "main/revision_acceptance.csv")
    accepted = 0
    for row in gates:
        old,new=float(row["old_mean"]),float(row["new_mean"])
        equal(new-old,row["delta"],old)
        if (new<old) != (row["new_lt_old"]=="True"):
            raise ValueError(f"Acceptance decision mismatch for {row['dataset']} {row['expert_id']}")
        accepted+=int(new<old)
    if len(gates)!=70 or accepted!=16:
        raise ValueError("Unexpected proposal counts")
    reports={}
    for group,ratios in counts.items():
        direction=Counter("improved" if x>1e-10 else "worsened" if x< -1e-10 else "tied" for x in ratios)
        reports[group]={"improved":direction["improved"], "tied":direction["tied"],
                       "worsened":direction["worsened"],"mean_RI_percent":statistics.fmean(ratios)}
    return {"main_seed_rows":len(main),"component_control_rows":len(controls),
            "external_seed_rows":len(wseed),"revisions_accepted":accepted,"proposals":len(gates),
            "comparisons":reports}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json",action="store_true",help="Print all comparisons as JSON.")
    args=parser.parse_args()
    report=verify(ROOT/"evidence")
    if args.json:
        print(json.dumps(report,indent=2))
    else:
        print("Verified 252 main, 480 component-control and 48 external per-seed rows.")
        print("Accepted 16 of 70 revision proposals.")
        print("Comparison                                          improved  tied  worsened   mean RI")
        for name,row in report["comparisons"].items():
            print(f"{name:52s}{row['improved']:>6}{row['tied']:>6}{row['worsened']:>10}{row['mean_RI_percent']:>9.3f}%")
