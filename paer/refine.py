"""Four-fold contribution screening, KEEP archival and paired revision acceptance."""
from copy import deepcopy
import json

import numpy as np

from .data import compile_pool, knowledge_folds, slot
from .gate import decide, merge_context, replace_one
from .retrieval import Training, fit, mse, predict


def decision(rewards, coverage):
    values = np.asarray(rewards, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or not 0 <= coverage <= 1:
        raise ValueError("Expected four finite per-fold rewards and coverage in [0, 1]")
    positive = (values > 0).mean()
    if coverage < 0.01:
        return "UNSUPPORTED", 0
    if positive >= 0.75 and np.median(values) > 0:
        return "KEEP", 2
    if positive >= 0.5 and values.mean() > 0:
        return "KEEP", 1
    if positive <= 0.25 and np.median(values) < 0:
        return "REJECT", -2
    return "REVISE", -1


class Evaluator:
    def __init__(self, series, vectors, device="cpu", training=Training()):
        self.series = series
        self.vectors = vectors
        self.device = device
        self.training = training
        self.folds = knowledge_folds(series.train_end, series.context, series.horizon)
        self.cache = {}

    def score(self, pool, drop=None):
        # All objects in this cache share one series, feature protocol, seed and budget.
        key = json.dumps([pool, drop], sort_keys=True, ensure_ascii=False)
        if key not in self.cache:
            rows = []
            for fold, (cutoff, end) in enumerate(self.folds):
                stage = self.series.stage(cutoff, end)
                model = fit(stage, pool, self.vectors, 42, self.device, self.training, drop)
                pred = predict(model, stage, pool, drop)
                rows.append(dict(fold=fold, cutoff=cutoff, end=end,
                                 raw_mse=mse(stage["Y"]["query"], pred["raw"]),
                                 aligned_mse=mse(stage["Y"]["query"], pred["aligned"]),
                                 coverage=pred["coverage"]))
            self.cache[key] = rows
        return self.cache[key]

    def evaluate(self, pool):
        full = self.score(pool)
        records = []
        for expert in pool:
            eid = expert["id"]
            minus = self.score(pool, drop=eid)
            rewards = [(m["raw_mse"] - f["raw_mse"]) / m["raw_mse"] if m["raw_mse"] > 0 else 0.0
                       for f, m in zip(full, minus)]
            coverage = float(np.mean([row["coverage"][eid] for row in full]))
            verdict, reward = decision(rewards, coverage)
            records.append(dict(expert=deepcopy(expert), coverage=coverage, verdict=verdict, reward=reward,
                                fold_rewards=rewards, full=full, masked=minus))
        return records


def refine(initial, proposal_rounds, evaluator, calendar=True):
    """Run the paired protocol with recorded, externally supplied revision proposals.

    proposal_rounds is a mapping from one-based round numbers to a list of expert
    objects. Each executed round must cover exactly the pending IDs. A caller may
    also supply a function (round_number, records) -> proposals.
    """
    active = compile_pool(initial, calendar)
    archive = {}
    rounds = []
    for number in range(3):
        records = evaluator.evaluate(active)
        run = dict(round=number, evaluation=records, gate=[])
        rounds.append(run)
        for record in records:
            if record["verdict"] == "KEEP":
                expert = record["expert"]
                archive[expert["id"]] = expert
        pending = {r["expert"]["id"] for r in records if r["verdict"] != "KEEP"}
        if number == 2 or not pending:
            break
        if callable(proposal_rounds):
            proposed = proposal_rounds(number + 1, deepcopy(records))
        else:
            proposed = proposal_rounds.get(str(number + 1))
        if proposed is None:
            raise ValueError(f"Missing proposals for round {number + 1}; pending IDs: {sorted(pending)}")
        replacements = compile_pool(proposed, calendar)
        if {e["id"] for e in replacements} != pending:
            raise ValueError(f"Round {number + 1} proposals must contain pending IDs {sorted(pending)}")
        context = merge_context(archive, active)
        old_losses = [r["raw_mse"] for r in evaluator.score(context)]
        for candidate in sorted(replacements, key=slot):
            updated = replace_one(context, candidate)
            new_losses = [r["raw_mse"] for r in evaluator.score(updated)]
            gate = decide(old_losses, new_losses)
            run["gate"].append(dict(expert_id=candidate["id"], old_fold_mse=old_losses,
                                    new_fold_mse=new_losses, **gate))
            if gate["accepted"]:
                context, old_losses = updated, new_losses
        active = [e for e in context if e["id"] in pending]
    return sorted(archive.values(), key=slot), rounds
