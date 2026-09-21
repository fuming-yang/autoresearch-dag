# Outcome-Aware Proposal-Memory DAG

## Why was this experiment run?

To investigate proposal collapse after Merge retry improved execution but continued to explore a narrow set of mechanisms.

## What changed from the previous stage?

The experiment retained credit-guided selection and Merge retry/fallback, while adding online proposal memory, explicit diversity guidance, structured mechanism metadata, a novelty/preparation guard, and proposal retries. The memory exposed per-family attempt counts and parent-relative success/failure counts to proposal generation, making it outcome-aware.

## Experimental setup

The run contains 30 valid candidates—20 Expands and ten Merges—plus one failed preparation node. The recorded baseline was 1.885642. The state records operations and proposal history in addition to topology.

## Main outputs

`results/dag.json`, `results/candidates.tsv`, and `results/summary.json` record candidates and proposal metadata. Diversity and lineage tables/figures are retained in `figures/`; `report.md` gives the source interpretation.

## Key observations

The best candidate was `exp_000027` at 1.855929. Ten of 30 operations improved on their parent or better parent (six Expands, four Merges), and the analysis identified 18 mechanism families. These are descriptive results from one trajectory; they do not prove that proposal memory caused the differences from DAG-B.

## Limitations

This run cannot isolate which bundled proposal-generation change mattered. Cross-run comparisons are not paired randomized evidence, and baseline provenance complicates comparisons with DAG-D.

## Which files should the reader look at?

Read `figures/performance_trajectory_common_scale.png`, both unified topology figures, `figures/parent_relative_delta.png`, `figures/mechanism_family_distribution.png`, `report.md`, and `code/proposal_memory.py`.
