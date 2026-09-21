# Curation audit

Source audited: `~/projects/4-papers` at branch `autoresearch-time-40`, commit `562f5af`, including all local branches, checkpoints, untracked `autoresearch-*` directories, nested run histories, results, analysis, reports, figures, and source Git history available on 2026-09-22.

## Run inventory and disposition

| Evidence-backed run | State | Disposition | Reason |
|---|---:|---|---|
| Linear Search | 40/40 iterations | INCLUDE | Complete sequential baseline and plateau evidence |
| Original Random DAG target-40 checkpoint | 29 valid candidates | EXCLUDE | Interrupted/quota-affected checkpoint; superseded by completed DAG-30 |
| Uniform Random DAG Search | 30/30 valid candidates | INCLUDE | Complete branching and two-parent Merge experiment |
| Uniform Random Tree Search | 30/30 valid candidates | INCLUDE | Complete branching/no-Merge experiment |
| Credit-Guided Parent DAG | 30/30 valid candidates | INCLUDE | Parent-credit and Merge-compatibility evidence |
| Merge-Retry Credit-Guided DAG | 30/30 valid candidates | INCLUDE | Merge retry/fallback mechanism evidence |
| Outcome-Aware Proposal-Memory DAG | 30/30 valid candidates | INCLUDE | Proposal-diversity and outcome-labelled history evidence |
| Outcome-Blind Proposal-Memory DAG | 30/30 valid candidates | INCLUDE | Outcome-label-blind proposal-memory ablation within the DAG research sequence |
| Baseline calibration | 4/4 repetitions | DIAGNOSTIC | Demonstrates optimization-step/token-exposure variation |

No second, separately auditable partial **linear** trajectory was found. The `autoresearch-master`, `autoresearch-master-5070`, and `autoresearch-time-40-template` directories are templates/code states without a separate linear result table or candidate trajectory. The clearly interrupted run in the source evidence is the original Random DAG target-40 run at 29 valid candidates; it is not relabelled as Linear.

## Other excluded material

- Pre-run Tree reset/audit state with zero launched candidates.
- Failed candidate reservations retained inside completed DAG metadata but never counted as valid experiments.
- Debug-only runs, quota stops, backups, copied snapshots, stale worktrees, caches, virtual environments, and agent transcripts.
- The confounded `dag_c_vs_d` headline comparison and its figures.
- Obsolete or duplicate figure versions and full raw logs.

`EXCLUDE` means “not copied into this repository”; nothing was deleted from or changed in the source archive.

## Linear standardization

`experiments/00_linear_search/results/full_40_iterations.tsv` retains the baseline and all 40 completed iterations. `standardized_first_30.tsv` retains the same baseline reference and the first 30 experimental iterations (rows 1–30 after the baseline). No values were recomputed or altered.

## Public-safety curation

The copied DAG state retains candidate IDs, statuses, operations, parents, hypotheses, metrics, commits, and analysis metadata. Machine-specific absolute source paths were replaced only in the curated copies with `<source-archive>/`; source data is unchanged. Raw logs, credentials, environments, weights, and nested Git data were not copied.
