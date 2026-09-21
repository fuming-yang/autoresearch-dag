# Uniform Random Tree Search

## Why was this experiment run?

To test branching search in which any finished historical candidate remains eligible for future work, without introducing recombination.

## What changed from the previous stage?

The linear chain was replaced by a tree. Parent selection was uniform over eligible finished nodes, every candidate had exactly one parent, and there was no pruning, credit weighting, or Merge.

## Experimental setup

One serial worker produced 30 valid Expand candidates from a recorded baseline of 1.885642 under a nominal 300-second training budget. The stored DAG has 31 nodes including the baseline and no failed candidates.

## Main outputs

`results/dag.json` records topology and candidate metadata; `results/candidates.tsv` gives parent-relative metrics. The representative topology and trajectory figures are in `figures/`.

## Key observations

The best candidate was `exp_000003` at 2.002502; none beat the recorded baseline. Nine of 30 candidates improved on their own parent, showing local recovery inside branches even though the run did not recover baseline quality.

## Limitations

The baseline result was inherited from the preceding setup rather than freshly re-measured for the reset tree root. This is one trajectory, and its poor absolute results should not be read as a general verdict on tree search.

## Which files should the reader look at?

Read `figures/performance_trajectory_common_scale.png`, `figures/search_topology.png`, `figures/search_topology_success.png`, `figures/parent_relative_delta.png`, `results/dag.json`, and `code/sample.py`.
