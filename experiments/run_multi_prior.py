"""Multi-Prior Retrieval (TRAIN/VAL only).

Combine validated executable priors into one retrieval score and compare five
combination strategies (A shape-only, B best single, C equal weight,
D LLM-confidence weight, E validation-score weight). No TEST access, no
adaptive lambda/gate, no new priors or transforms, Chronos / Top-K / transfer /
softmax / lambda=0.1 unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.executable import multi_prior_combination
from priors.llm_prior import generate_priors
from priors.prompts import PROMPT_VERSION_V21

METHOD_LABELS = {
    "A_shape_only": "A: shape-only",
    "B_best_single": "B: best single (squared_load_profile)",
    "C_equal_weight": "C: equal weight",
    "D_confidence_weight": "D: LLM-confidence weight",
    "E_validation_weight": "E: validation-score (Spearman) weight",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_priors(dataset):
    metadata = load_json(f"configs/metadata/{dataset}.json")
    expert_cfg = load_json(f"configs/expert_hints/{dataset}.json")
    mechanism = expert_cfg.get("expert_hint", "") or ""
    meta = generate_priors(
        metadata, expert_hint=None, prompt_type="metadata_only_v21",
        prompt_version=PROMPT_VERSION_V21,
    )["priors"]
    expert = generate_priors(
        metadata, expert_hint=mechanism, prompt_type="expert_mechanism_v21",
        prompt_version=PROMPT_VERSION_V21,
    )["priors"]
    return meta + expert  # metadata first -> dedup keeps metadata representative


def main():
    dataset = "ETTh1"
    priors = load_priors(dataset)
    out = multi_prior_combination(priors, dataset=dataset)

    print("=" * 100)
    print("MULTI-PRIOR RETRIEVAL -- TRAIN/VAL  (lambda=0.1, alpha_grid=[0,.1,.25,.5,.75,1.0])")
    print(f"dataset={dataset}  validation gate: structural>=0.05 AND Spearman>0  "
          f"(dedup by feature key)")
    print("=" * 100)

    # ---- accepted candidates ----
    print("\nACCEPTED PRIORS (used in C/D/E):")
    print(f"{'name':<34}{'source':<13}{'feature':<40}{'struct':>7}{'spear':>8}{'conf':>6}")
    for c in out["candidates"]:
        print(f"{c['name']:<34}{c['source']:<13}{c['key']:<40}"
              f"{c['structural']:>7.3f}{c['spearman']:>+8.3f}{c['confidence']:>6.2f}")

    # ---- rejected ----
    if out["rejected"]:
        print("\nREJECTED (excluded from combination):")
        for r in out["rejected"]:
            print(f"  - {r['name']}: {r['reason']}")

    # ---- strategy comparison table ----
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    hdr = (f"{'method':<40}{'n':>3}{'a*':>6}{'VAL mem MSE':>13}{'VAL fus MSE':>13}"
           f"{'vs shape':>10}{'vs best':>9}")
    print(hdr)
    print("-" * 100)
    for r in out["strategies"]:
        n = len(r["included"])
        label = METHOD_LABELS.get(r["method"], r["method"])
        print(f"{label:<40}{n:>3}{r['best_alpha']:>6.2f}"
              f"{r['val_mem_mse']:>13.4f}{r['val_fus_mse']:>13.4f}"
              f"{r['imp_shape']:>+9.2f}%{r['imp_best']:>+8.2f}%")
    print("-" * 100)
    print(f"  shape-only VAL memory MSE = {out['shape_mem_mse']:.4f}   "
          f"fusion MSE = {out['shape_fus_mse']:.4f}")
    print(f"  Chronos-only VAL MSE = {out['chronos_mse']:.4f}")
    print("  vs shape = (shape_fus - fus)/shape_fus*100   "
          "vs best = (best_single_fus - fus)/best_single_fus*100")

    # ---- weight detail ----
    print("\nPRIOR WEIGHTS (sum = 1.0):")
    for r in out["strategies"]:
        if not r["included"]:
            print(f"  {METHOD_LABELS[r['method']]}: (none)")
            continue
        wstr = ", ".join(f"{nm}={w:.4f}" for nm, w in zip(r["included"], r["weights"]))
        print(f"  {METHOD_LABELS[r['method']]}:")
        print(f"    {wstr}")


if __name__ == "__main__":
    main()
