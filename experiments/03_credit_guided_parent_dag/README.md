# Credit-Guided Parent DAG

## Why was this experiment run?

To ask whether parent opportunities could be allocated using candidate quality, historical child improvement, and an exploration bonus instead of uniform sampling.

## What changed from the previous stage?

Parent selection became stochastic credit-softmax sampling. For each eligible finished node, credit combined current quality, mean direct-child improvement, and an exploration bonus. The 50/50 Expand/Merge intent, no-pruning policy, and one-pair-per-Merge-intent rejection behavior remained unchanged.

## Experimental setup

The run completed 30 valid candidates: 26 Expands and four Merges. Two failed nodes remain in the state. The recorded baseline was 1.885642; the best candidate was `exp_000004` at 1.895905.

## Main outputs

`results/dag.json`, `results/candidates.tsv`, and `results/merge_attempts.tsv` preserve the compact evidence. `report.md` contains the source audit and interpretation.

## Key observations

No candidate beat the baseline. Three operations improved locally. Thirty-one of 36 Merge attempts were semantically rejected, and only four reached training; all four finished Merges failed to beat their better parent. Credit allocation operated, but the run exposed a mismatch between node quality and pair compatibility, alongside severe proposal concentration around `MATRIX_LR`.

## Limitations

The run does not provide a controlled causal comparison with the earlier random DAG, and the small number of executed Merges limits inference.

## Which files should the reader look at?

Read `figures/performance_trajectory_common_scale.png`, both unified topology figures, `figures/parent_relative_delta.png`, `report.md`, and `results/merge_attempts.tsv`.
