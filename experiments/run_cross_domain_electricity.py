"""Cross-domain validation, step 1: electricity load (non-ETT).

Runs the FROZEN anonymized Multi-Agent pipeline (Proposal -> Critic -> Compiler)
on a new electricity-load domain, reusing the exact prompt builders / whitelist /
LLM cache from the ETT protocol. Only the anonymized metadata + expert mechanism
facts change.
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
import run_multiagent_cross_dataset as mcd

# ---- anonymized electricity metadata ----
ELECTRICITY_METADATA = {
    "target_variable": "Y",
    "target_description": "total electricity demand (aggregate load of all consumers)",
    "observed_variables": ["X1", "X2", "X3", "X4", "X5", "X6"],
    "observed_description": "electricity load of six individual consumers",
    "sampling_frequency": "15 minutes",
}

# ---- domain expert mechanism facts (mechanisms, NOT retrieval features) ----
ELECTRICITY_EXPERT_MECHANISM = (
    "electricity demand has a strong recurring temporal usage structure\n"
    "recent demand tends to persist into the near future because consumption patterns change gradually\n"
    "working and non-working periods correspond to systematically different demand regimes\n"
    "aggregate demand is the sum of many individual consumers whose loads rise and fall together\n"
    "aggregate demand changes smoothly because it averages over many independent consumers"
)


def main():
    # monkey-patch the frozen pipeline's anonymized inputs (same code path)
    ma.BLIND_METADATA = ELECTRICITY_METADATA
    ma.BLIND_EXPERT_MECHANISM = ELECTRICITY_EXPERT_MECHANISM

    model = ma._default_model()
    proto = mcd.run_frozen_protocol(model)  # proposal -> critique -> compile (cache-aware)

    print("=" * 90)
    print("ELECTRICITY Multi-Agent pipeline (frozen, anonymized)")
    print("=" * 90)
    print(f"model={model}")
    print(f"# proposed hypotheses: {proto['n_proposed']}")
    print(f"# accepted (by critic): {len(proto['critique']['accepted'])}")
    print(f"# compiled executable:  {len(proto['exec_priors'])}")
    print(f"# unsupported:          {len(proto['unsupported'])}")

    print("\nCOMPILED EXECUTABLE PRIORS:")
    for p in proto["exec_priors"]:
        print(f"  - {p.name}: {p.transform}/{p.temporal_aggregation}/{p.similarity} "
              f"on {p.input_variables}")

    print("\nUNSUPPORTED / REJECTED:")
    for u in proto["unsupported"]:
        print(f"  - {u.get('name')}: {u.get('reason')}")

    # save
    Path("results/cross_domain").mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": "electricity",
        "model": model,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metadata": ELECTRICITY_METADATA,
        "expert_mechanism": ELECTRICITY_EXPERT_MECHANISM,
        "n_proposed": proto["n_proposed"],
        "n_accepted": len(proto["critique"]["accepted"]),
        "proposal": proto["proposal"],
        "critique": proto["critique"],
        "compile": proto["compile"],
        "exec_priors": [p.to_dict() for p in proto["exec_priors"]],
        "unsupported": proto["unsupported"],
    }
    Path("results/cross_domain/electricity_pipeline.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSaved results/cross_domain/electricity_pipeline.json")


if __name__ == "__main__":
    main()
