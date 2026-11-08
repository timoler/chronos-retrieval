"""Prepare PAER prompts and validate saved replies. Each invocation is offline."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.data import compile_pool, knowledge_folds
from paer.features import FEATURE_REGISTRY
from paer.refine import decision


EVIDENCE = (
    "Only these TRAIN expanding-fold error differences and activation coverage are evidence. "
    "The inner temporal holdout, VAL and TEST are unavailable. Do not invent regime or causal explanations. "
    "A leave-one-out retraining reward is conditional predictive utility, not factual truth or measured isolated effect. "
    "The verdict is fixed by code. Do not override it. State uncertainty when evidence does not identify a modification. "
)
EXPERT_KEYS = set(("id mechanism scope activation_condition failure_condition observables retrieval_action "
                   "expected_effect reason condition_spec retrieval_spec claim observable_features replaced "
                   "compiled revision_round lineage_id").split())
CRITIC_KEYS = ("expert_id", "verdict", "diagnosis", "revision_instruction")
SCHEMA = """OUTPUT SCHEMA (strict JSON):
Return {"experts": [ ... ]}. Each entry has exactly these keys:
  - "id": "E1", "E2", ...
  - "mechanism": the physical/domain mechanism (1 sentence)
  - "scope": where it applies (e.g. global / summer / high volatility / workday)
  - "activation_condition": the precise "when [observable state]" condition
  - "failure_condition": when it is expected to FAIL or should NOT apply
  - "observables": list of feature names from AVAILABLE OBSERVABLES (only current/history info)
  - "retrieval_action": how it concretely changes historical retrieval
  - "expected_effect": why it should make the retrieved future better match
  - "reason": mechanism rationale (1-2 sentences)
  - "condition_spec": {"logic": "AND"|"OR", "conditions": [{"feature": "...", "qualifier": "..."}]}
  - "retrieval_spec": {"ops": [{"type": ..., "feature": ...}]}  (OPTIONAL structured form of retrieval_action)

retrieval_spec.ops use ONLY these types:
  - {"type":"prefer_similar","feature":"recent_level|recent_volatility|recent_slope_magnitude|recent_diff_magnitude|recent_autocorr"}
  - {"type":"prefer_same","feature":"hour_of_day|working_day|seasonal_level"}
  - {"type":"restrict_same","feature":"hour_of_day|working_day|seasonal_level"}
  - {"type":"cross","feature":"working_day|seasonal_level"}
  - {"type":"lag","direction":"short"}

For condition_spec.qualifier: continuous features use "high" or "low"; calendar features use the category name (e.g. "summer", "night", "workday").
Do NOT invent features outside the catalog. Do NOT use future/target values as observables. Output only the JSON object."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, obj):
    content = (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing artifact differs: {path.name}. Use a new output directory.")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def summary(rewards):
    numbers = np.asarray(rewards, dtype=np.float64)
    if numbers.shape != (4,) or not np.isfinite(numbers).all():
        raise ValueError("Expected four finite TRAIN fold rewards")
    return dict(fold_rewards=numbers.tolist(), mean_pool_reward=float(numbers.mean()),
                median_pool_reward=float(np.median(numbers)), positive_fold_ratio=float((numbers>0).mean()),
                worst_pool_reward=float(numbers.min()))


def from_fits(record, folds):
    full, minus = record["full"], record["masked"]
    if len(full)!=4 or len(minus)!=4:
        raise ValueError("Expected four pairs of training fold fits")
    eid=record["expert"]["id"]
    for i,(a,b) in enumerate(zip(full,minus)):
        if (a["fold"]!=i or b["fold"]!=i or (a["cutoff"],a["end"])!=folds[i]
                or (b["cutoff"],b["end"])!=folds[i]):
            raise ValueError("Expected a matched sequential TRAIN fold partition")
        if i and a["cutoff"]!=full[i-1]["end"]:
            raise ValueError("Unexpected gap or overlap in the TRAIN fold query intervals")
        for key in ("raw_mse","aligned_mse"):
            if any(not math.isfinite(x[key]) or x[key]<0 for x in (a,b)):
                raise ValueError("Invalid fold MSE")
    cov=float(np.mean([x["coverage"][eid] for x in full]))
    if not math.isclose(cov,record["coverage"],rel_tol=1e-12,abs_tol=1e-14):
        raise ValueError("Reported coverage differs from fold records")
    return tuple(summary([(b[k]-a[k])/b[k] if b[k]>0 else 0.0 for a,b in zip(full,minus)])
                 for k in ("raw_mse","aligned_mse"))


def checked_summary(value):
    expected=summary(value["fold_rewards"])
    for k in expected.keys()-{"fold_rewards"}:
        if not math.isclose(value[k],expected[k],rel_tol=1e-12,abs_tol=1e-14):
            raise ValueError(f"Inconsistent contribution summary: {k}")
    return {k:value[k] for k in expected}


def evidence_records(source, dataset, number):
    protocol=read(ROOT/"inputs"/dataset/"protocol.json")
    folds=knowledge_folds(protocol["split"]["train"][1],protocol["context"],protocol["horizon"])
    if isinstance(source, dict):
        if (source.get("dataset")!=dataset or source.get("stages")!=[f"fold{i}" for i in range(4)]
                or source.get("seeds")!=[42] or source.get("used_for_revision") is not True
                or not source.get("label","").endswith(f"_round{number-1}")):
            raise ValueError("Expected the active TRAIN evaluation for the preceding round")
        records=source["records"]
    else:
        records=source
        if not isinstance(records,list) or any("full" not in r or "masked" not in r for r in records):
            raise ValueError("Portable evaluation records must include their TRAIN fold fits")
    if not isinstance(records,list) or len(records)>8 or number not in (1,2):
        raise ValueError("Expected up to eight active experts and revision round 1 or 2")
    all_ids=[item["expert"]["id"] for item in records]
    if len(set(all_ids))!=len(all_ids):
        raise ValueError("Duplicate expert ID in evaluation")
    result=[]
    for item in records:
        expert={k:v for k,v in item["expert"].items() if k in EXPERT_KEYS}
        compile_pool([expert],calendar=dataset!="exchange_rate")
        if "full" in item:
            raw, aligned=from_fits(item,folds)
            if raw["fold_rewards"]!=item["fold_rewards"]:
                raise ValueError("Fold rewards differ from the paired fits")
        else:
            raw,aligned=(checked_summary(item[k]) for k in ("pool_stats","aligned_stats"))
        c=item["coverage"]
        verdict,reward=decision(raw["fold_rewards"],c)
        if (verdict,reward)!=(item["verdict"],item["reward"]) or item.get("pool_size",len(records))!=len(records):
            raise ValueError("Verdict, reward or active pool size differs from the screening record")
        if verdict!="KEEP":
            result.append(dict(expert=expert,pool_stats=raw,aligned_stats=aligned,
                               coverage=c,reward=reward,verdict=verdict,pool_size=len(records)))
    return result


def catalog():
    lines=["AVAILABLE OBSERVABLES (choose 'observables' and 'condition_spec.feature' ONLY from this list):"]
    for name,meta in FEATURE_REGISTRY.items():
        description=("calendar (categories: "+", ".join(sorted(meta["categories"]))+")"
                     if meta["kind"]=="calendar" else "continuous (qualifier: high or low)")
        lines.append(f"  - {name}: {description}")
    return "\n".join(lines)


def request(system, user):
    return {"model":"deepseek-v4-pro","system":system,"messages":[{"role":"user","content":user}],"max_tokens":20000}


def critic_request(records):
    fields=[{k:r[k] for k in ("expert","pool_stats","coverage","verdict","reward","pool_size")} for r in records]
    return request("You are the PAER critic. "+EVIDENCE+"Return JSON only.",
                   json.dumps(fields,ensure_ascii=False)+'\nReturn {"critiques":[{"expert_id":"E1","verdict":"REVISE",'
                   '"diagnosis":"...","revision_instruction":"..."}]}; one entry for each supplied expert.')


def entries(path, key, id_field, expected):
    obj=read(path)
    if isinstance(obj,dict) and "text" in obj and key not in obj:
        text=re.sub(r"^```(?:json)?\s*|\s*```$","",obj["text"].strip())
        try:
            obj=json.loads(text)
        except json.JSONDecodeError:
            obj=json.loads(text[text.index("{"):text.rindex("}")+1])
    rows=obj.get(key) if isinstance(obj,dict) else None
    if not isinstance(rows,list) or not all(isinstance(row,dict) for row in rows):
        raise ValueError(f"Expected JSON object with array {key}")
    ids=[row.get(id_field) for row in rows]
    if len(ids)!=len(expected) or set(ids)!=expected:
        raise ValueError(f"Expected one {key} entry for each of {sorted(expected)}")
    return rows


def expert_request(records, critiques, dataset):
    by_id={row["expert_id"]:row for row in critiques}
    feedback=[]
    for row in records:
        critic=by_id[row["expert"]["id"]]
        if critic.get("verdict")!=row["verdict"] or any(not isinstance(critic.get(k),str)
            or not critic[k].strip() for k in ("diagnosis","revision_instruction")):
            raise ValueError("Critic reply must preserve the program verdict and include diagnosis and revision_instruction")
        feedback.append({**row,"critic":{k:critic[k] for k in CRITIC_KEYS}})
    restriction=("Calendar features are prohibited; dataset dates are ordinal placeholders. " if dataset=="exchange_rate" else "")
    if dataset in ("traffic","solar"):
        restriction+="Calendar is reconstructed; do not assert authenticated day-of-week or seasonal evidence. "
    system=("You revise conditional retrieval knowledge using PAER predictive feedback. "+EVIDENCE+
            "Preserve each id. Revise non-KEEP entries. If changing mechanism family, set replaced=true. "
            "Conditions must be executable using current context or known calendar only. "+restriction+
            "Continuous qualifiers must be high or low. Calendar qualifiers must occur in the catalog. "
            "Only flat AND/OR conditions; no negation, arbitrary numerical thresholds, nested conditions, unknown features. Return strict JSON.")
    return request(system,catalog()+"\n"+SCHEMA+"\nFEEDBACK:\n"+json.dumps(feedback,ensure_ascii=False)+
                   '\nReturn {"experts":[...]} with the same ids, a nonempty mechanism, condition_spec, observables, retrieval_action and replaced boolean.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,choices=("etth1","etth2","exchange_rate","traffic","weather","solar","electricity"))
    parser.add_argument("--round",type=int,choices=(1,2),required=True)
    parser.add_argument("--evaluation",required=True,type=Path)
    parser.add_argument("--critic-reply",type=Path)
    parser.add_argument("--expert-reply",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.expert_reply and not args.critic_reply:
        parser.error("--expert-reply requires --critic-reply")
    records=evidence_records(read(args.evaluation),args.dataset,args.round)
    args.out.mkdir(parents=True,exist_ok=True)
    ids={row["expert"]["id"] for row in records}
    save(args.out/"context.json",{"dataset":args.dataset,"revision_round":args.round,"pending_ids":sorted(ids),
         "training_evidence_sha256":hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False).encode()).hexdigest()})
    if not ids:
        save(args.out/"proposals.json",{str(args.round):[]})
        return
    save(args.out/"critic_request.json",critic_request(records))
    if args.critic_reply:
        critiques=entries(args.critic_reply,"critiques","expert_id",ids)
        save(args.out/"expert_request.json",expert_request(records,critiques,args.dataset))
    if args.expert_reply:
        proposed=entries(args.expert_reply,"experts","id",ids)
        for e in proposed:
            if type(e.get("replaced")) is not bool:
                raise ValueError(f"Expected boolean replaced flag for {e['id']}")
            e["lineage_id"]=e["id"]
            e["revision_round"]=args.round
        compile_pool(proposed,calendar=args.dataset!="exchange_rate")
        save(args.out/"proposals.json",{str(args.round):proposed})


if __name__=="__main__":
    main()
