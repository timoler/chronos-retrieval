"""Expert / LLM prior ablation -- generation (STEP 4), validation (STEP 6),
and leave-one-in (STEP 6.5).

Modes (run from the project root):

  * default:         generate and print the three LLM prior sets.
  * --validate:      TRAIN-ONLY data-driven validation table (STEP 6).
  * --leave-one-in:  TRAIN/VAL leave-one-in alpha sweep per prior (STEP 6.5).

The leave-one-in experiment keeps memory bank / Top-K / transfer / aggregation
fixed, sweeps alpha on VAL only, and never touches test data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from priors.llm_prior import generate_priors
from priors.prompts import (
    GENERIC_EXPERT_HINT,
    PROMPT_VERSION,
    PROMPT_VERSION_V2,
    PROMPT_VERSION_V21,
)
from priors.validator import leave_one_in, operationalize, validate_priors
from priors.executable import AR_OT_CONTROL, compile_prior, leave_one_in_executable


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def print_prompt(prompt):
    print("----- EXACT PROMPT (system) -----")
    print(prompt["system"])
    print("----- EXACT PROMPT (user) -----")
    print(prompt["user"])


def print_priors(result):
    print(f"model         : {result['model']}")
    print(f"prompt_version: {result['prompt_version']}")
    print(f"prompt_type   : {result['prompt_type']}")
    print(f"from_cache    : {result['from_cache']}")
    print(f"cache_path    : {result['cache_path']}")
    print(f"n_priors      : {len(result['priors'])}")
    print()
    print("----- PARSED PRIORS (structured) -----")
    for i, p in enumerate(result["priors"], 1):
        print(f"[{i}] " + json.dumps(p.to_dict(), ensure_ascii=False, indent=2))


def print_validation_table(condition_label, results, model, prompt_version):
    print("\n" + "=" * 100)
    print(f"VALIDATION TABLE -- {condition_label}   (model={model}, prompt_version={prompt_version})")
    print("=" * 100)
    hdr = (
        f"{'Prior':<34} {'Type':<13} {'LLMconf':>7} {'Struct':>7} "
        f"{'PredRel':>7} {'Spearman':>9} {'Status':>24}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        struct = f"{r['structural_score']:.3f}" if r["structural_score"] is not None else "  -"
        pred = f"{r['predictive_relevance_score']:.3f}" if r["predictive_relevance_score"] is not None else "  -"
        sp = f"{r['spearman']:+.3f}" if r["spearman"] is not None else "  -"
        print(
            f"{r['name']:<34} {r['type']:<13} {r['llm_confidence']:>7.2f} {struct:>7} "
            f"{pred:>7} {sp:>9} {r['status']:>24}"
        )
    print("-" * len(hdr))
    print("detail (feature / reason / top-K future MSE):")
    for r in results:
        feat = r["feature"] or "-"
        ptk = f"{r['prior_topk_mse']:.4f}" if r["prior_topk_mse"] is not None else "   -"
        stk = f"{r['shape_topk_mse']:.4f}" if r["shape_topk_mse"] is not None else "   -"
        print(f"  - {r['name']}  [feature={feat}]  prior_topk_MSE={ptk}  shape_topk_MSE={stk}  -> {r['reason']}")


def _generate_all(datasets, model):
    """Generate (from cache) the three prior sets and return per-condition priors."""
    out = {}
    for ds in [d.strip() for d in datasets.split(",")]:
        metadata = load_json(f"configs/metadata/{ds}.json")
        expert_cfg = load_json(f"configs/expert_hints/{ds}.json")
        specific_hint = expert_cfg.get("expert_hint", "") or ""

        conditions = [
            ("metadata_only", None, "Metadata-only LLM"),
            ("generic_expert", GENERIC_EXPERT_HINT, "Generic Expert-guided LLM"),
            ("specific_expert", specific_hint, "Specific Expert-guided LLM"),
        ]
        out[ds] = []
        for prompt_type, hint, label in conditions:
            result = generate_priors(
                metadata,
                expert_hint=hint,
                model=model,
                prompt_type=prompt_type,
                prompt_version=PROMPT_VERSION,
            )
            out[ds].append((label, result))
    return out


def dedupe_priors(generated):
    """Collapse the three conditions' priors to one representative per feature."""
    seen = {}
    for ds, conds in generated.items():
        for _label, result in conds:
            for p in result["priors"]:
                f = operationalize(p)
                key = f if f is not None else ("not_op", p.type)
                if key not in seen:
                    seen[key] = p
    return list(seen.values())


def print_leave_one_in_table(priors, val_rows, li_rows, model, prompt_version):
    print("\n" + "=" * 110)
    print(
        f"LEAVE-ONE-IN (TRAIN/VAL, alpha sweep, lambda fixed) -- model={model}, prompt_version={prompt_version}"
    )
    print("=" * 110)
    hdr = (
        f"{'Prior':<32} {'Feat':<12} {'Type':<13} {'conf':>4} {'Struct':>7} {'Spear':>8} "
        f"{'a*':>5} {'VALmemMSE':>10} {'VALfusMSE':>10} {'vsShape%':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p, vr, lr in zip(priors, val_rows, li_rows):
        if lr["best_alpha"] is None:
            print(f"{p.name:<32} {'-':<12} {p.type:<13} {p.confidence:>4.2f} {'  -':>7} "
                  f"{'  -':>8} {'  -':>5} {'      -':>10} {'      -':>10} {'    -':>9}  (not_operationalizable)")
            continue
        struct = f"{vr['structural_score']:.3f}"
        sp = f"{vr['spearman']:+.3f}"
        mem = f"{lr['val_mem_mse']:.4f}"
        fus = f"{lr['val_fus_mse']:.4f}"
        imp = f"{lr['improvement']:+.2f}"
        print(f"{p.name:<32} {lr['feature']:<12} {p.type:<13} {p.confidence:>4.2f} {struct:>7} "
              f"{sp:>8} {lr['best_alpha']:>5.2f} {mem:>10} {fus:>10} {imp:>9}%")
    print("-" * len(hdr))
    # baselines (shared across rows)
    base = next((lr for lr in li_rows if lr["best_alpha"] is not None), None)
    if base is not None:
        print(f"  shape-only (a=0) VAL memory MSE = {base['shape_mem_mse']:.4f}  "
              f"VAL fusion MSE = {base['shape_fus_mse']:.4f}")
        print(f"  Chronos-only VAL MSE = {base['chronos_val_mse']:.4f}")
        print("  NOTE: lambda fixed at 0.1; alpha selected on VAL fusion MSE. vsShape% = "
              "(shape_fus - prior_fus) / shape_fus * 100 (positive = better).")


def _generate_v2(datasets, model):
    """Generate the two Prompt V2 conditions (metadata-only, expert-mechanism)."""
    out = {}
    for ds in [d.strip() for d in datasets.split(",")]:
        metadata = load_json(f"configs/metadata/{ds}.json")
        expert_cfg = load_json(f"configs/expert_hints/{ds}.json")
        mechanism = expert_cfg.get("expert_hint", "") or ""
        conditions = [
            ("metadata_only_v2", None, "Metadata-only V2"),
            ("expert_mechanism_v2", mechanism, "Expert-mechanism + LLM V2"),
        ]
        out[ds] = []
        for prompt_type, hint, label in conditions:
            result = generate_priors(
                metadata,
                expert_hint=hint,
                model=model,
                prompt_type=prompt_type,
                prompt_version=PROMPT_VERSION_V2,
            )
            out[ds].append((label, result))
    return out


def print_v2_results(meta_result, expert_result):
    print("\n" + "=" * 120)
    print(
        f"PROMPT V2 -- model={meta_result['model']}  prompt_version={meta_result['prompt_version']}"
    )
    print("=" * 120)

    print("\n----- EXACT PROMPT (system, shared) -----")
    print(meta_result["prompt"]["system"])
    print("\n----- EXACT PROMPT (user) -- Metadata-only V2 -----")
    print(meta_result["prompt"]["user"])
    print("\n----- EXACT PROMPT (user) -- Expert-mechanism V2 -----")
    print(expert_result["prompt"]["user"])

    mp = meta_result["priors"]
    ep = expert_result["priors"]
    n = max(len(mp), len(ep))

    print("\n----- SIDE-BY-SIDE (name | latent state | observable proxy) -----")
    for i in range(n):
        m = mp[i] if i < len(mp) else None
        e = ep[i] if i < len(ep) else None
        mline = (
            f"{m.name} ({m.confidence:.2f}) | {m.latent_state} | {m.observable_proxy}"
            if m
            else "(none)"
        )
        eline = (
            f"{e.name} ({e.confidence:.2f}) | {e.latent_state} | {e.observable_proxy}"
            if e
            else "(none)"
        )
        print(f"  [{i+1:>2}] META: {mline}")
        print(f"       EXP : {eline}")

    print("\n----- FULL DETAIL: Metadata-only V2 -----")
    for p in mp:
        print(json.dumps(p.to_dict(), ensure_ascii=False, indent=2))
    print("\n----- FULL DETAIL: Expert-mechanism V2 -----")
    for p in ep:
        print(json.dumps(p.to_dict(), ensure_ascii=False, indent=2))


def _generate_v21(datasets, model):
    out = {}
    for ds in [d.strip() for d in datasets.split(",")]:
        metadata = load_json(f"configs/metadata/{ds}.json")
        expert_cfg = load_json(f"configs/expert_hints/{ds}.json")
        mechanism = expert_cfg.get("expert_hint", "") or ""
        conditions = [
            ("metadata_only_v21", None, "Metadata-only V2.1"),
            ("expert_mechanism_v21", mechanism, "Expert-mechanism V2.1"),
        ]
        out[ds] = []
        for prompt_type, hint, label in conditions:
            result = generate_priors(
                metadata,
                expert_hint=hint,
                model=model,
                prompt_type=prompt_type,
                prompt_version=PROMPT_VERSION_V21,
            )
            out[ds].append((label, result))
    return out


def _feat_label(p):
    if p is None:
        return "-"
    vars_s = (
        "+".join(p.input_variables)
        if len(p.input_variables) <= 3
        else f"{len(p.input_variables)}vars"
    )
    return f"{p.transform}.{p.temporal_aggregation}.{p.similarity}({vars_s})"


def print_v21_table(priors, rows, model, prompt_version):
    print("\n" + "=" * 118)
    print(f"V2.1 EXECUTABLE PRIORS -- TRAIN/VAL -- model={model}  prompt_version={prompt_version}")
    print("=" * 118)
    hdr = (
        f"{'Prior':<30} {'Source':<12} {'Feature':<34} {'Struct':>6} {'Spear':>7} "
        f"{'a*':>5} {'VALmem':>9} {'VALfus':>9} {'vsShape%':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for p, r in zip(priors, rows):
        if r["best_alpha"] is None:
            print(f"{p.name:<30} {p.source:<12} {_feat_label(p):<34} {'  -':>6} {'  -':>7} "
                  f"{'  -':>5} {'      -':>9} {'      -':>9} {'    -':>8}  (rejected: {r['reason']})")
            continue
        print(f"{p.name:<30} {p.source:<12} {_feat_label(p):<34} {r['structural']:>6.3f} "
              f"{r['spearman']:>+7.3f} {r['best_alpha']:>5.2f} {r['val_mem_mse']:>9.4f} "
              f"{r['val_fus_mse']:>9.4f} {r['improvement']:>+8.2f}")
    print("-" * len(hdr))
    base = next((r for r in rows if r["best_alpha"] is not None), None)
    if base is not None:
        print(f"  shape-only (a=0) VAL memory MSE = {base['shape_mem_mse']:.4f}  "
              f"fusion MSE = {base['shape_fus_mse']:.4f}")
        print(f"  Chronos-only VAL MSE = {base['chronos_val_mse']:.4f}")
        print("  NOTE: lambda fixed at 0.1; alpha selected on VAL fusion MSE; "
              "vsShape% = (shape_fus - prior_fus)/shape_fus*100 (positive = better).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ETTh1", help="comma-separated dataset names")
    ap.add_argument("--model", default=None, help="override LLM model (default: env ANTHROPIC_MODEL)")
    ap.add_argument("--validate", action="store_true", help="run TRAIN-ONLY prior validation (STEP 6)")
    ap.add_argument("--leave-one-in", action="store_true", help="run TRAIN/VAL leave-one-in alpha sweep (STEP 6.5)")
    ap.add_argument("--prompt-v2", action="store_true", help="generate + print the two Prompt V2 prior sets")
    ap.add_argument("--v21", action="store_true", help="compile V2.1 executable priors and run TRAIN/VAL evaluation")
    args = ap.parse_args()

    if args.prompt_v2:
        v2 = _generate_v2(args.datasets, args.model)
        for ds, conds in v2.items():
            print_v2_results(conds[0][1], conds[1][1])
        return

    if args.v21:
        v21 = _generate_v21(args.datasets, args.model)
        for ds, conds in v21.items():
            priors = conds[0][1]["priors"] + conds[1][1]["priors"] + [AR_OT_CONTROL]
            rows = leave_one_in_executable(priors, dataset=ds)
            print_v21_table(priors, rows, conds[0][1]["model"], conds[0][1]["prompt_version"])
        return

    generated = _generate_all(args.datasets, args.model)

    if args.leave_one_in:
        for ds in generated:
            priors = dedupe_priors({ds: generated[ds]})
            val_rows = validate_priors(priors, dataset=ds)
            li_rows = leave_one_in(priors, dataset=ds)
            model = next(iter(generated[ds]))[1]["model"]
            prompt_version = next(iter(generated[ds]))[1]["prompt_version"]
            print_leave_one_in_table(priors, val_rows, li_rows, model, prompt_version)
        return

    for ds, conds in generated.items():
        for label, result in conds:
            if args.validate:
                rows = validate_priors(result["priors"], dataset=ds)
                print_validation_table(f"{ds} / {label}", rows, result["model"], result["prompt_version"])
            else:
                source = "expert+llm" if "expert" in result["prompt_type"] else "llm"
                print("\n" + "=" * 78)
                print(f"DATASET={ds}  CONDITION={label}  prompt_type={result['prompt_type']}  source={source}")
                print("=" * 78)
                print_prompt(result["prompt"])
                print()
                print_priors(result)
                print()
                print("----- RAW RESPONSE -----")
                print(result["raw_response"])
                print()


if __name__ == "__main__":
    main()
