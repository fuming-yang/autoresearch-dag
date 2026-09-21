# Linear Search

## Why was this experiment run?

To establish the behavior of the original sequential keep/discard AutoResearch loop on the RTX 5070-compatible setup.

## What changed from the previous stage?

This is the first formal stage. Each new proposal modified the current retained program; improvements were kept and regressions discarded. There was no branching, revisiting, or Merge.

## Experimental setup

The archive contains one baseline row followed by 40 completed iterations. The metric is validation bits per byte (`val_bpb`, lower is better), with five-minute training runs. The trajectory moved from a baseline of 1.904472 to a best recorded value of 1.320959.

The original linear run contained 40 iterations. The first 30 are used in the standardized view for consistency with later 30-candidate experiments. The standardized TSV contains the baseline reference plus iterations 1–30; the complete 40-iteration TSV remains available.

## Main outputs

- `results/full_40_iterations.tsv`: complete baseline plus 40 iterations.
- `results/standardized_first_30.tsv`: baseline plus the first 30 iterations.
- `figures/linear_40_trajectory.png`: full trajectory showing rapid early gains and a later plateau.
- `figures/performance_trajectory_common_scale.png`: standardized view, **30 shown / 40 total**.
- `figures/performance_trajectory_local_scale.png`: the same first 30 iterations on a local y-axis.

## Key observations

Large early improvements came from batch-size and depth changes. Later iterations mostly tuned architecture and optimization near a much flatter frontier; improvements continued, but at far smaller scale.

## Limitations

This is one adaptive trajectory. Later proposals depend on earlier keep/discard decisions, and its wall-clock protocol is not directly exchangeable with every later run.

## Which files should the reader look at?

Start with `figures/performance_trajectory_common_scale.png`, then compare the two TSVs and the full-run `figures/linear_40_trajectory.png`. `code/train.py`, `code/prepare.py`, and `code/program.md` preserve the executed experiment implementation and instructions.
