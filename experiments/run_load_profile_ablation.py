"""Matched 2x2 ablation around the best expert prior (squared load profile).

The best expert prior from the V2.1 leave-one-in table was
``squared_load_profile`` (square / window / correlation over the 6 load series).
This ablation varies ONLY two axes while holding everything else fixed:

  * transform : raw load  vs  squared load   (square of the summed load)
  * aggregation: level (mean scalar)  vs  profile (full window)

Everything else is identical to the V2.1 pipeline: the same 6 load variables,
the same per-point transform + per-window slicing, the same similarity step
(correlation -> (r+1)/2 for profiles, standardized Gaussian kernel for scalars),
the same Top-K / mean-anchor transfer / softmax aggregation, fusion lambda=0.1,
and the same VAL alpha grid. TRAIN/VAL only; TEST is never accessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.executable import ExecutablePrior, leave_one_in_executable

LOAD_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def _cell(name, transform, agg, sim, mechanism):
    return ExecutablePrior(
        name=name,
        source="expert_llm",
        input_variables=LOAD_VARS,
        transform=transform,
        temporal_aggregation=agg,
        similarity=sim,
        expected_mechanism=mechanism,
        confidence=1.0,
    )


def main():
    dataset = "ETTh1"

    # (row, col, prior)
    cells = [
        ("raw", "level", _cell("raw load level", "raw", "mean", "abs_diff",
                               "raw total load level")),
        ("raw", "profile", _cell("raw load profile", "raw", "window", "correlation",
                                 "raw total load temporal profile")),
        ("squared", "level", _cell("squared load level", "square", "mean", "abs_diff",
                                   "squared total load level (copper-loss proxy)")),
        ("squared", "profile", _cell("squared load profile", "square", "window", "correlation",
                                     "squared total load temporal profile")),
    ]

    priors = [p for _, _, p in cells]
    rows = leave_one_in_executable(priors, dataset=dataset)
    result = {p.name: r for p, r in zip(priors, rows)}
    name_of = {(row, col): p.name for row, col, p in cells}

    shape_mem = rows[0]["shape_mem_mse"]
    shape_fus = rows[0]["shape_fus_mse"]
    chronos = rows[0]["chronos_val_mse"]

    def cell(row, col):
        r = result[name_of[(row, col)]]
        return (
            f"a*={r['best_alpha']:.2f} mem={r['val_mem_mse']:.3f} "
            f"fus={r['val_fus_mse']:.3f} imp={r['improvement']:+.2f}%"
        )

    W = 43
    c1 = "level (mean, abs_diff)"
    c2 = "profile (window, correlation)"
    print("=" * (13 + 3 + (W + 3) * 2))
    print("MATCHED 2x2 ABLATION -- best expert prior family (squared load profile)")
    print(f"dataset={dataset}  lambda=0.1  alpha_grid=[0,.1,.25,.5,.75,1.0]")
    print("variables = HUFL+HULL+MUFL+MULL+LUFL+LULL")
    print("=" * (13 + 3 + (W + 3) * 2))
    print(f"{'':<13} | {c1:<{W}} | {c2:<{W}}")
    print("-" * (13 + 3 + (W + 3) * 2))
    print(f"{'raw load':<13} | {cell('raw', 'level'):<{W}} | {cell('raw', 'profile'):<{W}}")
    print(f"{'squared load':<13} | {cell('squared', 'level'):<{W}} | {cell('squared', 'profile'):<{W}}")
    print("-" * (13 + 3 + (W + 3) * 2))
    print(f"  shape-only (a=0) VAL memory MSE = {shape_mem:.4f}   fusion MSE = {shape_fus:.4f}")
    print(f"  Chronos-only VAL MSE = {chronos:.4f}")
    print("  imp% = (shape_fus - prior_fus)/shape_fus*100 (positive = better than shape-only).")


if __name__ == "__main__":
    main()
