"""Uniform random sampler for the global experiment DAG."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path


def _loss(node):
    result = node.get("result")
    if isinstance(result, dict):
        result = result.get("val_bpb")
    return float(result) if result is not None else None


def choose(state, seed=None):
    """Randomly choose expand/merge, then uniformly sample its parent(s)."""
    rng = random.Random(seed)
    nodes = state.get("nodes", [])
    finished = [
        node
        for node in nodes
        if node.get("status") == "finished" and _loss(node) is not None
    ]
    if not finished:
        raise ValueError("the DAG has no finished node to expand")

    attempted = {
        tuple(sorted(node.get("parents", [])))
        for node in nodes
        if node.get("mode") == "merge"
    }
    attempted.update(
        tuple(sorted(pair)) for pair in state.get("rejected_merges", [])
    )
    merge_candidates = [
        pair
        for pair in itertools.combinations(finished, 2)
        if tuple(sorted((pair[0]["id"], pair[1]["id"]))) not in attempted
    ]

    mode_draw = rng.random()
    if mode_draw >= 0.5 and merge_candidates:
        left, right = rng.choice(merge_candidates)
        return {
            "mode": "merge",
            "parents": [left["id"], right["id"]],
            "score": mode_draw,
            "reason": "randomly selected merge and an untried parent pair",
        }

    parent = rng.choice(finished)
    reason = (
        "randomly selected expand and a finished parent"
        if mode_draw < 0.5
        else "randomly selected merge, but no untried pair was available; fell back to expand"
    )
    return {
        "mode": "expand",
        "parents": [parent["id"]],
        "score": mode_draw,
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(choose(json.loads(args.dag.read_text()), args.seed), indent=2))


if __name__ == "__main__":
    main()
