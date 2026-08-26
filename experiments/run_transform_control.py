"""TRAIN/VAL transformation control (NOT a power search).

Compares 4 predefined transforms of the summed 6 load series, all under
window profile + correlation similarity with the identical Top-K / mean-anchor
transfer / softmax aggregation / fusion lambda=0.1 / VAL alpha grid.

Transform family: f_p(x) = sign(x) * |x|^p  (sign-preserving where needed)

  p=1.0   raw load
  p=1.5   signed power
  p=2.0   squared load
  p=3.0   signed power

Only these four p values are run. TEST is never accessed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.executable import ExecutablePrior, leave_one_in_executable

LOAD_VARS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def _cell(label, transform, power):
    return ExecutablePrior(
        name=label,
        source="expert_llm",
        input_variables=LOAD_VARS,
        transform=transform,
        temporal_aggregation="window",
        similarity="correlation",
        power=power,
        expected_mechanism=f"load transform f(x)=sign(x)|x|^{power}",
        confidence=1.0,
    )


def main():
    dataset = "ETTh1"
    cells = [
        _cell("raw load (p=1)", "raw", None),
        _cell("signed power p=1.5", "signed_power", 1.5),
        _cell("squared load (p=2)", "square", None),
        _cell("signed power p=3", "signed_power", 3.0),
    ]

    rows = leave_one_in_executable(cells, dataset=dataset)

    shape_mem = rows[0]["shape_mem_mse"]
    shape_fus = rows[0]["shape_fus_mse"]
    chronos = rows[0]["chronos_val_mse"]

    print("=" * 84)
    print("TRANSFORMATION CONTROL -- sign(x)*|x|^p  (window profile, correlation)")
    print(f"dataset={dataset}  lambda=0.1  alpha_grid=[0,.1,.25,.5,.75,1.0]")
    print("variables = HUFL+HULL+MUFL+MULL+LUFL+LULL")
    print("=" * 84)
    print(f"{'transform':<20}{'a*':>6}{'VAL mem MSE':>13}{'VAL fus MSE':>13}{'imp vs shape':>14}")
    print("-" * 84)
    for p, r in zip(cells, rows):
        if r["best_alpha"] is None:
            print(f"{p.name:<20}{'-':>6}{'-':>13}{'-':>13}{'-':>14}  (rejected: {r['reason']})")
            continue
        print(f"{p.name:<20}{r['best_alpha']:>6.2f}{r['val_mem_mse']:>13.4f}"
              f"{r['val_fus_mse']:>13.4f}{r['improvement']:>+13.2f}%")
    print("-" * 84)
    print(f"  shape-only (a=0) VAL memory MSE = {shape_mem:.4f}   fusion MSE = {shape_fus:.4f}")
    print(f"  Chronos-only VAL MSE = {chronos:.4f}")
    print("  imp% = (shape_fus - prior_fus)/shape_fus*100 (positive = better than shape-only).")


if __name__ == "__main__":
    main()
