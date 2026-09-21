# Merge-Retry Credit-Guided DAG

## Why was this experiment run?

To address the Credit-Guided Parent DAG's Merge starvation without changing its node-level credit formula.

## What changed from the previous stage?

A Merge intent could retry up to three distinct incompatible pairs. If the retry pool was exhausted, the operation fell back to a credit-guided Expand.

## Experimental setup

Thirty valid candidates completed: 18 Expands and 12 Merges. Nineteen Merge intents produced 24 semantic rejects, 12 trained Merges, six fallback Expands, and one quota-stopped intent. The recorded baseline was 1.885642.

## Main outputs

`results/dag.json`, `results/candidates.tsv`, `results/summary.json`, and `results/parent_relative_summary.json` contain the compact evidence.

## Key observations

Retry/fallback increased finished Merge conversion from the preceding run's 4/36 to 12/19 intents. That structural change did not establish better optimization: the best candidate was 1.890996, none beat baseline, two of 18 Expands beat their parent, and one of 12 Merges beat its better parent.

## Limitations

The preceding and current experiments are separate adaptive trajectories, not paired replicates. A quota circuit breaker stopped one intent without producing a candidate.

## Which files should the reader look at?

Read `figures/performance_trajectory_common_scale.png`, both unified topology figures, `figures/parent_relative_delta.png`, and the JSON summaries in `results/`.
