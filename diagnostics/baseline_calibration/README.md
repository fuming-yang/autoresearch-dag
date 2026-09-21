# Baseline calibration

## Why was this diagnostic run?

An exploratory comparison between the outcome-aware and outcome-blind proposal-memory runs revealed inconsistent baseline provenance. This calibration tested whether an identical nominal 300-second wall-clock protocol delivers stable optimization exposure.

## Method

The byte-identical baseline `train.py` and `prepare.py` were run four times serially on one RTX 5070 Laptop GPU, with seed 42 and the same data, model, optimizer, schedule, and evaluation settings. No DAG search or proposal agent was involved.

## Results

The runs completed 20, 21, 20, and 23 optimizer steps, exposing 10,485,760 to 12,058,624 tokens. `val_bpb` ranged from 1.888005 to 1.906045 (range 0.018040). The two 20-step runs differed by only 0.000766; the 23-step run was substantially lower.

## Interpretation and limitation

Step/token exposure variation is material under this wall-clock protocol. Historical own-baseline rates for the two proposal-memory runs should remain descriptive, and their direct headline comparison is not treated as causal evidence. Four runs identify the issue but are not a full variance study.

## Files

- `results/runs.tsv` and `results/summary.json`: compact measurements and configuration.
- `code/train.py` and `code/prepare.py`: calibrated source.
The original full logs were reviewed in the source archive but intentionally omitted from this public-facing curation.
