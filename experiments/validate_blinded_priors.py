"""Validate the two blinded prior sets against the existing TRAIN/VAL V2.1 pipeline.

Loads the priors already saved by run_blinded_prompt_control.py (no LLM
regeneration), maps the anonymized variable names back to the real ETT names,
and runs each prior through ``leave_one_in_executable`` independently.

No prior combination, no TEST access. Every forecasting/retrieval setting is the
pipeline default (alpha grid, lambda=0.1, chunk=256, context=512, pred=96).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.executable import ExecutablePrior, leave_one_in_executable

# anonymized placeholder -> real ETT variable (order irrelevant: priors use the
# full load set together, and sum/ewma/concatenate are permutation-invariant).
BLIND_TO_REAL = {
    "Y": "OT",
    "X1": "HUFL", "X2": "HULL", "X3": "MUFL",
    "X4": "MULL", "X5": "LUFL", "X6": "LULL",
}


def load_blind_priors(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for d in data["parsed_priors"]:
        inputs = [BLIND_TO_REAL[v] for v in d["input_variables"]]
        out.append(ExecutablePrior(
            name=d["name"],
            source=d["source"],
            input_variables=inputs,
            transform=d["transform"],
            temporal_aggregation=d["temporal_aggregation"],
            similarity=d["similarity"],
            expected_mechanism=d.get("expected_future_relationship")
                              or d.get("domain_fact", ""),
            power=None,
            confidence=float(d.get("confidence", 0.0)),
        ))
    return out


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def print_table(title, rows):
    print("\n" + title)
    print("-" * 118)
    print(f"{'Prior':<42}{'struct':>9}{'Spearman':>10}{'best a':>7}"
          f"{'memMSE':>11}{'fusMSE':>11}{'d%':>8}")
    print("-" * 118)
    for r in rows:
        name = r["name"] if r["feature"] is not None else f"{r['name']} [rejected: {r['reason']}]"
        print(f"{name[:41]:<42}"
              f"{fmt(r['structural'], 3):>9}"
              f"{fmt(r['spearman'], 3):>10}"
              f"{fmt(r['best_alpha'], 2):>7}"
              f"{fmt(r['val_mem_mse']):>11}"
              f"{fmt(r['val_fus_mse']):>11}"
              f"{fmt(r['improvement'], 2):>8}")
    print("-" * 118)


def main():
    pa = load_blind_priors("results/blinding/blind_metadata.json")
    pb = load_blind_priors("results/blinding/blind_expert.json")

    print(f"ETTh1 TRAIN/VAL V2.1 leave-one-in  "
          f"A={len(pa)} priors (blind_metadata)  B={len(pb)} priors (blind_expert)")

    rows_a = leave_one_in_executable(pa, dataset="ETTh1")
    rows_b = leave_one_in_executable(pb, dataset="ETTh1")

    # baselines are identical across both runs (same settings / data)
    base = rows_a[0]
    print(f"\nBaselines:  shape-only fused MSE = {base['shape_fus_mse']:.4f}   "
          f"Chronos-only val MSE = {base['chronos_val_mse']:.4f}   "
          f"shape-only mem MSE = {base['shape_mem_mse']:.4f}")

    print_table("BLIND METADATA (A) -- per-prior", rows_a)
    print_table("BLIND + EXPERT (B) -- per-prior", rows_b)

    # best prior per condition = lowest VAL fused MSE (== highest improvement)
    def best(rows):
        exec_ = [r for r in rows if r["feature"] is not None]
        return min(exec_, key=lambda r: r["val_fus_mse"]) if exec_ else None

    ba, bb = best(rows_a), best(rows_b)

    print("\nBEST PRIOR PER CONDITION (side-by-side)")
    print("-" * 118)
    fields = [
        ("name", "name", lambda r: r["name"]),
        ("structural", "structural", lambda r: fmt(r["structural"], 3)),
        ("Spearman (train)", "spearman", lambda r: fmt(r["spearman"], 3)),
        ("best VAL alpha", "best_alpha", lambda r: fmt(r["best_alpha"], 2)),
        ("VAL memory MSE", "val_mem_mse", lambda r: fmt(r["val_mem_mse"])),
        ("VAL fused MSE", "val_fus_mse", lambda r: fmt(r["val_fus_mse"])),
        ("improvement vs shape (%)", "improvement", lambda r: fmt(r["improvement"], 2)),
    ]
    print(f"{'':<28}{'Blind Metadata':<30}{'Blind + Expert':<30}")
    print("-" * 118)
    for label, _k, get in fields:
        print(f"{label:<28}{get(ba) if ba else '-':<30}{get(bb) if bb else '-':<30}")
    print("-" * 118)


if __name__ == "__main__":
    main()
