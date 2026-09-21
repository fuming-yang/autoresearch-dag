# Uniform Random DAG Search

## Why was this experiment run?

To extend branching search with two-parent semantic recombination and examine whether Merge could combine distinct historical candidates.

## What changed from the previous stage?

Compared with the Random Tree, the sampler could request either a one-parent Expand or a two-parent Merge. The final valid set contains 15 of each.

## Experimental setup

The completed run contains 30 valid candidates, plus the 1.885642 baseline. Two failed nodes are retained in `dag.json` but excluded from the valid count. Candidates ran under the time-based 300-second protocol.

## Main outputs

`results/dag.json` preserves all parent relationships and rejected merges; `results/candidates.tsv` gives baseline- and parent-relative outcomes. The unified topology figures distinguish operations and local success.

## Key observations

The best candidate, `exp_000001`, reached 1.879146. Three candidates beat the recorded baseline and three of 30 operations beat their parent or better parent. The run demonstrates executable multi-parent topology, not a general advantage over Tree or Linear search.

## Limitations

This is a single trajectory with two failed nodes and 20 recorded rejected Merge pairs. Its 30-candidate completion supersedes the earlier 29-of-40 checkpoint.

## Which files should the reader look at?

Start with `figures/performance_trajectory_common_scale.png`, `figures/search_topology.png`, `figures/search_topology_success.png`, `figures/parent_relative_delta.png`, `results/dag.json`, and `code/orchestrator.py`.
