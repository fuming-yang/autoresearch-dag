# Figure topology audit

Generated from the curated `results/dag.json` files. Both topology figures for each experiment use the same audited node and edge sets. Parent-relative success affects styling only; it never changes topology.

| Experiment | Plotted nodes | Valid candidates | Expand edges | Merge edges | Failed nodes | Total edges | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Uniform Random Tree Search | 31 | 30 | 30 | 0 | 0 | 30 | PASS |
| Uniform Random DAG Search | 33 | 30 | 15 | 30 | 2 | 48 | PASS |
| Credit-Guided Parent DAG | 33 | 30 | 26 | 8 | 2 | 37 | PASS |
| Merge-Retry Credit-Guided DAG | 31 | 30 | 18 | 24 | 0 | 42 | PASS |
| Outcome-Aware Proposal-Memory DAG | 32 | 30 | 20 | 20 | 1 | 41 | PASS |
| Outcome-Blind Proposal-Memory DAG | 32 | 30 | 20 | 20 | 1 | 42 | PASS |

## Checks applied

### Uniform Random Tree Search

- PASS — node set exact
- PASS — edge set exact
- PASS — all 30 valid nodes
- PASS — no invented edge
- PASS — all source edges plotted

### Uniform Random DAG Search

- PASS — node set exact
- PASS — edge set exact
- PASS — all 30 valid nodes
- PASS — no invented edge
- PASS — all source edges plotted

### Credit-Guided Parent DAG

- PASS — node set exact
- PASS — edge set exact
- PASS — all 30 valid nodes
- PASS — no invented edge
- PASS — all source edges plotted

### Merge-Retry Credit-Guided DAG

- PASS — node set exact
- PASS — edge set exact
- PASS — all 30 valid nodes
- PASS — no invented edge
- PASS — all source edges plotted

### Outcome-Aware Proposal-Memory DAG

- PASS — node set exact
- PASS — edge set exact
- PASS — all 30 valid nodes
- PASS — no invented edge
- PASS — all source edges plotted

### Outcome-Blind Proposal-Memory DAG

- PASS — node set exact
- PASS — edge set exact
- PASS — all 30 valid nodes
- PASS — no invented edge
- PASS — all source edges plotted

## Counting convention

`Expand edges` and `Merge edges` count incoming edges of the 30 valid candidates only. Failed/interrupted nodes are plotted in gray and their source-recorded edges are included in `Total edges`. Root/baseline is included in plotted-node counts but not in valid-candidate counts.

Success colors use only local comparisons: Expand child versus its parent; Merge child versus the lower-BPB parent. Lower BPB is better.
