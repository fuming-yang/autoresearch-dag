"""Uniform random sampler for the random research tree."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _loss(node):
    result = node.get("result")
    if isinstance(result, dict):
        result = result.get("val_bpb")
    return float(result) if result is not None else None


def choose(state, seed=None):
    """Choose exactly one eligible finished parent uniformly, without pruning."""
    rng = random.Random(seed)
    finished = [
        node for node in state.get("nodes", [])
        if node.get("status") == "finished" and _loss(node) is not None
    ]
    if not finished:
        raise ValueError("the tree has no finished node to expand")
    parent = rng.choice(finished)
    return {
        "mode": "expand",
        "parents": [parent["id"]],
        "score": None,
        "reason": "uniform random selection from all eligible finished nodes",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dag", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(choose(json.loads(args.dag.read_text()), args.seed), indent=2))


if __name__ == "__main__":
    main()
