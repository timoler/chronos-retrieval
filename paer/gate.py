"""Pure acceptance rules for paired expert revisions.

The comparison context contains archived KEEP entries and the current active
entries. Replacing an entry is different from masking it. These helpers do not
train models, select final pool membership, or read validation/test outcomes.
The caller must supply the same ordered TRAIN folds for both loss vectors.
"""

from collections.abc import Mapping
from copy import deepcopy
import math


_IDS = tuple(f"E{i}" for i in range(1, 9))


def _expert_id(expert):
    if not isinstance(expert, Mapping):
        raise ValueError("An expert must be a mapping with an id in E1..E8")
    expert_id = expert.get("id")
    if expert_id not in _IDS:
        raise ValueError("Expert id must be one of E1..E8")
    return expert_id


def _unique_current(experts):
    """Validate an active/context list without silently dropping duplicates."""
    result = {}
    for expert in experts:
        expert_id = _expert_id(expert)
        if expert_id in result:
            raise ValueError(f"Duplicate active expert id: {expert_id}")
        result[expert_id] = expert
    return result


def merge_context(archive, current):
    """Return independent expert copies, ordered E1..E8; current wins.

    ``archive`` can be a list of expert objects or an id-to-expert mapping.
    Repeated ids in an archive list resolve to the most recent list entry.
    Duplicate active ids are an error. Archived entries whose ids also appear
    in ``current`` do not create a second slot.
    """
    merged = {}
    if isinstance(archive, Mapping):
        for key, expert in archive.items():
            expert_id = _expert_id(expert)
            if key != expert_id:
                raise ValueError("Archive key must match its expert id")
            merged[expert_id] = expert
    else:
        for expert in archive:
            merged[_expert_id(expert)] = expert
    merged.update(_unique_current(current))
    return [deepcopy(merged[expert_id]) for expert_id in _IDS if expert_id in merged]


def replace_one(context, proposal):
    """Copy the context and replace exactly one existing expert id.

    The original context and proposal are never modified. No entry is added,
    deleted, or masked; final membership is a separate contribution decision.
    """
    entries = _unique_current(context)
    expert_id = _expert_id(proposal)
    if expert_id not in entries:
        raise ValueError(f"Cannot replace absent expert: {expert_id}")
    entries[expert_id] = proposal
    return [deepcopy(entries[item]) for item in _IDS if item in entries]


def _losses(values, name):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an ordered loss sequence")
    try:
        items = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an ordered loss sequence") from error
    result = []
    for item in items:
        if isinstance(item, (str, bytes, bool)):
            raise ValueError(f"{name} must contain numeric MSE values")
        try:
            value = float(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must contain numeric MSE values") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must contain finite nonnegative MSE values")
        result.append(value)
    return result


def decide(old_losses, new_losses, expected_count=4):
    """Accept only a strict decrease in the arithmetic mean paired-fold MSE.

    Exact ties retain the incumbent. Fold losses are equally weighted; neither
    positive-fold counts nor deployment scores choose an accepted revision.
    ``accepted`` and ``new_lt_old`` deliberately report the same Boolean.
    """
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    old = _losses(old_losses, "old_losses")
    new = _losses(new_losses, "new_losses")
    if len(old) != len(new) or len(old) != expected_count:
        raise ValueError(f"Both loss vectors must have exactly {expected_count} paired folds")
    # Scale before summing to avoid overflow for finite, very large MSE values.
    old_mean = math.fsum(value / expected_count for value in old)
    new_mean = math.fsum(value / expected_count for value in new)
    accepted = new_mean < old_mean
    return {
        "old_mean": old_mean,
        "new_mean": new_mean,
        "delta": new_mean - old_mean,
        "new_lt_old": accepted,
        "accepted": accepted,
        "per_fold_delta": [after - before for before, after in zip(old, new)],
        "criterion": "arithmetic_mean_new_mse < arithmetic_mean_old_mse; exact ties rejected",
    }
