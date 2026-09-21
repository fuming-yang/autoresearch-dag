"""Credit-guided parent sampler for the global experiment DAG."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

QUALITY_SCALE = 0.01
LAMBDA_CREDIT = 0.5
GAMMA_EXPLORE = 0.5
TEMPERATURE = 1.0
MERGE_PROBABILITY = 0.5
MAX_MERGE_PAIR_ATTEMPTS = 3

def _loss(node):
    result = node.get("result")
    if isinstance(result, dict):
        result = result.get("val_bpb")
    try:
        value = float(result)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None

def credit_config(state):
    """Return configured credit parameters and counters, with safe defaults."""
    stored = state.get("credit_guidance", {})
    return {
        "quality_scale": float(stored.get("quality_scale", QUALITY_SCALE)),
        "lambda_credit": float(stored.get("lambda_credit", LAMBDA_CREDIT)),
        "gamma_explore": float(stored.get("gamma_explore", GAMMA_EXPLORE)),
        "temperature": float(stored.get("temperature", TEMPERATURE)),
        "total_parent_selections": int(stored.get("total_parent_selections", 0)),
        "parent_uses": dict(stored.get("parent_uses", {})),
    }

def compute_credits(state):
    """Compute Q, E, H, and C for every finished node with a finite val_bpb."""
    config = credit_config(state)
    scale = config["quality_scale"]
    temperature = config["temperature"]
    if scale <= 0 or temperature <= 0:
        raise ValueError("quality_scale and temperature must be positive")
    eligible = [node for node in state.get("nodes", [])
                if node.get("status") == "finished" and _loss(node) is not None]
    if not eligible:
        raise ValueError("the DAG has no finished valid node to expand")
    best_loss = min(_loss(node) for node in eligible)
    valid_children = [node for node in eligible if node.get("mode") != "baseline"]
    total = config["total_parent_selections"]
    records = []
    for node in eligible:
        node_id, loss = node["id"], _loss(node)
        improvements = [(loss - _loss(child)) / scale for child in valid_children
                        if node_id in child.get("parents", [])]
        quality = (best_loss - loss) / scale
        offspring = sum(improvements) / len(improvements) if improvements else 0.0
        uses = int(config["parent_uses"].get(node_id, 0))
        exploration = math.sqrt(math.log(total + 1) / (uses + 1))
        credit = quality + config["lambda_credit"] * offspring + config["gamma_explore"] * exploration
        records.append({"id": node_id, "val_bpb": loss, "Q": quality, "E": offspring,
                        "H": exploration, "credit": credit, "n_v_before": uses,
                        "current_T": total})
    max_credit = max(record["credit"] for record in records)
    weights = [math.exp((record["credit"] - max_credit) / temperature) for record in records]
    normalizer = sum(weights)
    for record, weight in zip(records, weights):
        record["selection_probability"] = weight / normalizer
    return records

def _weighted_choice(items, weights, rng):
    draw, cumulative = rng.random() * sum(weights), 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if draw < cumulative:
            return item
    return items[-1]

def _selection_snapshot(record, probability=None):
    snapshot = dict(record)
    if probability is not None:
        snapshot["selection_probability"] = probability
    return snapshot

def _pair_options(state, credits):
    """Return frozen ordered-pair options using DAG-A's merge weights."""
    by_id = {record["id"]: record for record in credits}
    config = credit_config(state)
    attempted = {tuple(sorted(node.get("parents", []))) for node in state.get("nodes", [])
                 if node.get("mode") == "merge"}
    attempted.update(tuple(sorted(pair)) for pair in state.get("rejected_merges", []))
    untried_pairs = [pair for pair in itertools.combinations(by_id, 2)
                     if tuple(sorted(pair)) not in attempted]
    options = []
    for left, right in untried_pairs:
        for first, second in ((left, right), (right, left)):
            first_p = by_id[first]["selection_probability"]
            remaining = [record for record in credits if record["id"] != first]
            remaining_max = max(record["credit"] for record in remaining)
            remaining_weights = [math.exp((record["credit"] - remaining_max) /
                                          config["temperature"])
                                 for record in remaining]
            remaining_total = sum(remaining_weights)
            second_p = next(weight / remaining_total for record, weight
                            in zip(remaining, remaining_weights) if record["id"] == second)
            options.append({
                "parents": [first, second],
                "pair_key": tuple(sorted((first, second))),
                "weight": first_p * second_p,
                "first_probability": first_p,
                "second_probability": second_p,
            })
    return options


def _expand_choice(credits, rng, mode_draw, reason):
    probabilities = [record["selection_probability"] for record in credits]
    parent = _weighted_choice(credits, probabilities, rng)
    return {"mode": "expand", "parents": [parent["id"]], "score": mode_draw,
            "reason": reason, "parent_selection": {"operation": "expand",
                "current_T": parent["current_T"], "selected": [_selection_snapshot(parent)]}}


def plan_operation(state, seed=None, max_merge_pair_attempts=MAX_MERGE_PAIR_ATTEMPTS):
    """Freeze one operation draw and its credit-guided parent choices."""
    if max_merge_pair_attempts < 1:
        raise ValueError("max_merge_pair_attempts must be positive")
    rng = random.Random(seed)
    credits = compute_credits(state)
    by_id = {record["id"]: record for record in credits}
    mode_draw = rng.random()
    operation_draw = ("merge" if mode_draw >= 1.0 - MERGE_PROBABILITY else "expand")
    plan = {
        "operation_draw": operation_draw,
        "operation_draw_value": mode_draw,
        "max_merge_pair_attempts": max_merge_pair_attempts,
        "current_T": credits[0]["current_T"],
        "credit_snapshot": credits,
        "merge_choices": [],
        "fallback_expand_choice": None,
    }
    if operation_draw == "expand":
        plan["expand_choice"] = _expand_choice(
            credits, rng, mode_draw, "credit-softmax expand selection"
        )
        return plan

    # Produce a weighted sample without replacement over exact unordered pairs.
    # Credits and raw pair weights stay frozen for the entire Merge intent.
    remaining = _pair_options(state, credits)
    for _ in range(max_merge_pair_attempts):
        if not remaining:
            break
        selected = _weighted_choice(remaining, [item["weight"] for item in remaining], rng)
        pair_key = selected["pair_key"]
        pool_total = sum(item["weight"] for item in remaining)
        first, second = selected["parents"]
        plan["merge_choices"].append({
            "mode": "merge",
            "parents": [first, second],
            "score": mode_draw,
            "pair_id": "+".join(pair_key),
            "pair_probability": selected["weight"] / pool_total,
            "reason": "credit-softmax merge sampled without replacement from remaining untried pairs",
            "parent_selection": {"operation": "merge", "current_T": by_id[first]["current_T"],
                "selected": [
                    _selection_snapshot(by_id[first], selected["first_probability"]),
                    _selection_snapshot(by_id[second], selected["second_probability"]),
                ]},
        })
        remaining = [item for item in remaining if item["pair_key"] != pair_key]

    plan["fallback_expand_choice"] = _expand_choice(
        credits, rng, mode_draw,
        "merge pair attempts exhausted; fell back to credit-softmax expand selection",
    )
    return plan


def choose(state, seed=None):
    """Backward-compatible first choice for one DAG-B operation."""
    plan = plan_operation(state, seed=seed)
    if plan["operation_draw"] == "expand":
        return plan["expand_choice"]
    if plan["merge_choices"]:
        return plan["merge_choices"][0]
    return plan["fallback_expand_choice"]

def record_parent_selection(state, choice):
    """Persist one use for each parent after the proposal is actually reserved."""
    config = state.setdefault("credit_guidance", {})
    config.setdefault("quality_scale", QUALITY_SCALE)
    config.setdefault("lambda_credit", LAMBDA_CREDIT)
    config.setdefault("gamma_explore", GAMMA_EXPLORE)
    config.setdefault("temperature", TEMPERATURE)
    uses = config.setdefault("parent_uses", {})
    config.setdefault("total_parent_selections", 0)
    for parent_id in choice["parents"]:
        uses[parent_id] = int(uses.get(parent_id, 0)) + 1
        config["total_parent_selections"] += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(choose(json.loads(args.dag.read_text()), args.seed), indent=2))

if __name__ == "__main__":
    main()
