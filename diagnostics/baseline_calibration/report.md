# Baseline Variance Calibration

## Purpose

This small calibration measures repeated-run variation of the exact baseline used by the DAG-C/D experiments. It is separate from both search experiments: no DAG search, candidate, Merge, or proposal agent was run. Four repetitions were executed serially on one RTX 5070 Laptop GPU with seed 42 and a 300-second training-time budget.

## Source and controls

The calibration copies `train.py` and `prepare.py` from the archived DAG-D snapshot. Their SHA-256 hashes are `8b3ed658...5945cfd` and `c5aa2c6b...1738851`; both are byte-identical to the DAG-C and DAG-D baseline commits. Every run used the same shared dataset/tokenizer, model (8 layers, width 512, 50.3M parameters), sequence length 2048, vocabulary 8192, device batch 16, gradient accumulation 16, total batch 524,288 tokens, optimizer/schedule, BF16 evaluation, and `AUTORESEARCH_SEED=42`.

The command was `AUTORESEARCH_SEED=42 CUDA_VISIBLE_DEVICES=0 <D-venv-python> -u train.py`. Complete stdout/stderr is retained in `logs/`.

## Results

| Run | val_bpb | Steps | Exact tokens | Training seconds | Total seconds |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.906045 | 20 | 10,485,760 | 310.2 | 825.5 |
| 2 | 1.902874 | 21 | 11,010,048 | 327.9 | 836.7 |
| 3 | 1.905279 | 20 | 10,485,760 | 304.7 | 816.3 |
| 4 | 1.888005 | 23 | 12,058,624 | 315.0 | 748.6 |

Summary: mean **1.900551**, median **1.904077**, sample standard deviation **0.008472**, minimum **1.888005**, maximum **1.906045**, range **0.018040**.

## Historical baselines

- Historical C: 1.885642, 23 steps. It is 0.002363 below the calibration minimum and 0.002363 better than the calibration 23-step run.
- Historical D: 1.906318, 20 steps. It is only 0.000273 above the calibration maximum and 0.000273 worse than calibration run 1.
- Historical C-D gap: 0.020676. This is 1.146 times the four-run observed range, but is clearly on the same scale once the observed 20-versus-23-step regimes are considered.

The two 20-step repetitions differ by only 0.000766. The 23-step repetition improves over their mean by about 0.017657. The 21-step result lies between the 20- and 23-step outcomes. This ordered relationship is strong evidence that variable step/token exposure, rather than ordinary same-step numerical noise, dominates baseline variation here.

## Interpretation

1. Baseline variation across the complete wall-clock protocol is large: observed range 0.018040 and sample SD 0.008472. Within the same 20-step regime it is much smaller (range 0.000766).
2. The historical 0.020676 gap is slightly larger than the four-run range, but it is consistent in scale with switching from a 20-step to 23-step regime. Historical D is essentially reproduced by the 20-step runs; historical C is closely approached by the 23-step run.
3. The 300-second wall-clock budget demonstrably produces different step counts: `[20, 21, 20, 23]` under identical code, seed, and nominal hardware.
4. Step/token exposure is sufficient to explain most of the C/D baseline difference. The time-dependent LR/weight-decay schedules also occur at different optimizer steps, amplifying the exposure difference.
5. Four runs already reveal the diagnostic mechanism. A fifth or sixth run would refine variance estimates but is not necessary to establish that throughput-induced step-count variation is material. Stop here unless a publication requires a confidence interval or a redesigned fixed-step calibration.

## Consequence for the ablation

The historical own-baseline rates should remain descriptive and should not be compared as if the thresholds were exchangeable. Absolute val_bpb also reflects different realized compute exposure. Parent-relative, Merge-relative, proposal-diversity, and topology analyses remain the more useful evidence, with ordinary run-noise caveats for small deltas.
