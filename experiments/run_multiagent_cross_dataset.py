"""Frozen Multi-Agent pipeline on ETTh2 / ETTm1 / ETTm2 (cross-dataset test).

Reuses the EXACT ETTh1 protocol from run_multiagent_control.py -- identical
anonymized metadata, expert mechanism, Proposal -> Critic -> Compiler workflow,
executable whitelist, and prompts. The protocol is invoked per dataset; its LLM
stages are dataset-agnostic (identical anonymized metadata), so they cache-hit
and yield the same priors, and only the TRAIN/VAL validation differs.

No TEST, no Chronos execution (cached val forecasts only), no prior combination,
no adaptive fusion. Every prompt/output/decision/compiled prior is saved.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_multiagent_control as ma
from priors.executable import leave_one_in_executable

DATASETS = ["ETTh2", "ETTm1", "ETTm2"]


def run_frozen_protocol(model):
    """Run the frozen proposal -> critique -> compile protocol (cache-hit)."""
    prop_sys, prop_user = ma.build_proposal_prompt()
    prop_raw, _, prop_cached = ma._call_llm(prop_sys, prop_user, "proposal", model)
    hypotheses = ma._list_from(prop_raw, "hypotheses")

    crit_sys, crit_user = ma.build_critique_prompt(hypotheses)
    crit_raw, _, crit_cached = ma._call_llm(crit_sys, crit_user, "critique", model)
    critiques = ma._list_from(crit_raw, "critiques")

    verdict_by_name = {str(c.get("name", "")).strip().lower(): c for c in critiques}
    accepted = []
    for h in hypotheses:
        crit = verdict_by_name.get(str(h.get("name", "")).strip().lower())
        if crit is None or str(crit.get("verdict", "accept")).strip().lower() == "accept":
            hc = dict(h)
            hc["_critique"] = crit if crit else {"verdict": "accept (no critique)"}
            accepted.append(hc)

    comp_sys, comp_user = ma.build_compile_prompt(accepted)
    comp_raw, _, comp_cached = ma._call_llm(comp_sys, comp_user, "compile", model)
    compiled = ma._list_from(comp_raw, "priors")

    exec_priors, unsupported = [], []
    for d in compiled:
        supported = str(d.get("supported", "true")).strip().lower() in ("true", "1", "yes")
        e = ma.compiled_to_exec(d) if supported else None
        if supported and e is not None:
            exec_priors.append(e)
        else:
            unsupported.append({
                "name": d.get("name"),
                "reason": d.get("unsupported_reason", ""),
                "unmappable": supported and e is None,
            })

    return {
        "proposal": {"system": prop_sys, "user": prop_user, "raw_response": prop_raw,
                     "parsed": hypotheses},
        "critique": {"system": crit_sys, "user": crit_user, "raw_response": crit_raw,
                     "parsed": critiques, "accepted": accepted},
        "compile": {"system": comp_sys, "user": comp_user, "raw_response": comp_raw,
                    "parsed": compiled},
        "exec_priors": exec_priors,
        "unsupported": unsupported,
        "n_proposed": len(hypotheses),
    }


def _spec_str(p):
    v = p.input_variables
    if set(v) == {"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"}:
        inp = "6-load"
    elif v == ["OT"]:
        inp = "OT"
    else:
        inp = "+".join(v)
    return f"{p.transform}/{p.temporal_aggregation}/{p.similarity} on {inp}"


def summarize_run(ds, proto, rows):
    base = rows[0]
    exec_pairs = [(r, p) for r, p in zip(rows, proto["exec_priors"])
                  if r["feature"] is not None]
    n_exec = len(exec_pairs)
    best_r, best_p = min(exec_pairs, key=lambda t: t[0]["val_fus_mse"])
    return {
        "dataset": ds,
        "chronos_val_mse": base["chronos_val_mse"],
        "shape_fus_mse": base["shape_fus_mse"],
        "n_proposed": proto["n_proposed"],
        "n_exec": n_exec,
        "best_name": best_r["name"],
        "best_spec": _spec_str(best_p),
        "best_alpha": best_r["best_alpha"],
        "best_mem_mse": best_r["val_mem_mse"],
        "best_fus_mse": best_r["val_fus_mse"],
        "best_improvement": best_r["improvement"],
    }


def main():
    model = ma._default_model()
    ts = datetime.now().isoformat(timespec="seconds")

    reports, all_rows, protocol_saved = [], {}, None
    for ds in DATASETS:
        proto = run_frozen_protocol(model)  # frozen protocol invoked per dataset (cache-hit)
        if protocol_saved is None:
            protocol_saved = proto  # identical across datasets; save once
        rows = leave_one_in_executable(proto["exec_priors"], dataset=ds)
        rep = summarize_run(ds, proto, rows)
        reports.append(rep)
        all_rows[ds] = rows

        # ---- per-dataset report block ----
        print("\n" + "=" * 72)
        print(f"DATASET: {ds}")
        print("=" * 72)
        print(f"  Chronos VAL MSE:         {rep['chronos_val_mse']:.4f}")
        print(f"  Shape-only fusion MSE:   {rep['shape_fus_mse']:.4f}")
        print(f"  # proposed priors:       {rep['n_proposed']}")
        print(f"  # accepted executable:   {rep['n_exec']}")
        print(f"  Best prior:              {rep['best_name']}")
        print(f"  Executable specification: {rep['best_spec']}")
        print(f"  Best alpha:              {rep['best_alpha']}")
        print(f"  VAL memory MSE:          {rep['best_mem_mse']:.4f}")
        print(f"  VAL fused MSE:           {rep['best_fus_mse']:.4f}")
        print(f"  Improvement vs shape:    {rep['best_improvement']:.2f}%")

    # ---- summary table ----
    print("\n" + "=" * 100)
    print("SUMMARY  (frozen Multi-Agent protocol; identical settings, lambda=0.1)")
    print("=" * 100)
    print(f"{'Dataset':<10}{'Shape':>10}{'Best Agent Prior':<32}{'Fusion MSE':>12}{'Improvement':>13}")
    print("-" * 100)
    for r in reports:
        print(f"{r['dataset']:<10}{r['shape_fus_mse']:>10.4f}{r['best_name']:<32}"
              f"{r['best_fus_mse']:>12.4f}{r['best_improvement']:>12.2f}%")
    print("-" * 100)

    # ---- save ----
    ma._save("results/multiagent_cross_dataset/protocol.json", {
        "note": "frozen protocol (identical prompts/outputs across datasets)",
        "model": model, "timestamp": ts, "datasets": DATASETS,
        "proposal": protocol_saved["proposal"],
        "critique": protocol_saved["critique"],
        "compile": protocol_saved["compile"],
        "exec_priors": [p.to_dict() for p in protocol_saved["exec_priors"]],
        "unsupported": protocol_saved["unsupported"],
    })
    ma._save("results/multiagent_cross_dataset/validation.json", {
        "model": model, "timestamp": ts,
        "reports": reports,
        "rows": {ds: all_rows[ds] for ds in DATASETS},
    })
    print("\nSaved to results/multiagent_cross_dataset/ (protocol.json, validation.json)")


if __name__ == "__main__":
    main()
