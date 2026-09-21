# Outcome-Blind Proposal-Memory DAG

## Why was this experiment run?

To test proposal generation without explicit historical success/failure labels while retaining structural exploration memory and outcome-conditioned parent selection.

## What changed from the previous stage?

The proposal memory removed per-family `beat-parent` and failure counts, and the Merge prompt stopped describing the checkout as the better-result parent. Credit-guided parent selection still used measured outcomes; Merge logic, novelty controls, mechanism attempts, and structural history remained. “Outcome-blind” is therefore more accurate than “history-blind” or “memoryless.”

## Experimental setup

Thirty valid candidates completed: 20 Expands and ten Merges. One interrupted preparation node (`exp_000018`) is retained as failed and excluded from the valid count. The recorded baseline was 1.906318; final parent-selection count `T` was 40.

## Main outputs

`results/dag.json`, `results/candidates.tsv`, and `results/summary.json` preserve topology, metrics, and proposal-history evidence. Unified left-to-right topology and parent-relative views are retained.

## Key observations

The best candidate was `exp_000029` at 1.869937. Nine operations improved locally: eight Expands and one Merge. The analysis identified 21 mechanism families. Nineteen of the 28 candidates below the recorded baseline inherited that status without improving locally, emphasizing the difference between lineage inheritance and new local gain.

## Limitations

The own-baseline win rate is not clean causal evidence: later calibration showed that the nominal wall-clock budget produced 20–23 optimizer steps. This single trajectory cannot establish the effect of hiding outcome labels.

## Which files should the reader look at?

Read `figures/performance_trajectory_common_scale.png`, the unified left-to-right topology figures, `figures/parent_relative_delta.png`, the supplementary `figures/outcome_blind_discovery_vs_inheritance.png` and `figures/outcome_blind_mechanism_family_distribution.png`, `results/summary.json`, and `code/proposal_memory.py`.
