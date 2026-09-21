# Exploring DAG Search for Automated Research

*An exploratory study of branching, recombination, credit-guided allocation, and proposal memory in automated program-improvement search.*

## Overview

Linear automated program improvement can make rapid early progress, but a single surviving edit trajectory may later enter a flatter search phase. This project asks: **what changes when alternative historical candidates are retained, revisited, and recombined instead of following only one linear trajectory?**

The study progresses from Linear Search to Tree and directed acyclic graph (DAG) search, then examines multi-parent Merge, credit-guided parent allocation, Merge retry/fallback, proposal memory, and an ablation of explicit outcome information in proposal generation. It is a mechanism study—not a claim that DAG search universally outperforms linear search.

![Linear Search performance trajectory](experiments/00_linear_search/figures/performance_trajectory_common_scale.png)

*The original Linear Search contained 40 iterations; the standardized visualization shows the first 30. It found strong early improvements followed by a later, flatter search phase.*

## Research progression

| Stage | Search structure | Main change | Question |
|---|---|---|---|
| **Linear Search** | Single adaptive chain | Keep or discard edits along one trajectory | What does sequential automated improvement find, and where does it flatten? |
| **Uniform Random Tree Search** | Single-parent tree | Revisit any eligible historical candidate uniformly | Is branching alone useful? |
| **Uniform Random DAG Search** | Multi-parent DAG | Add 50/50 Expand/Merge intent and two-parent semantic recombination | What changes when separate branches can be recombined? |
| **Credit-Guided Parent DAG** | Credit-selected DAG | Allocate parent opportunities using quality, direct-child improvement, and exploration | Which historical candidates deserve more search budget? |
| **Merge-Retry Credit-Guided DAG** | Credit DAG with fallback | Retry up to three distinct Merge pairs after semantic rejection, then fall back to Expand | Can pair resampling reduce Merge starvation? |
| **Outcome-Aware Proposal-Memory DAG** | Credit DAG with proposal memory | Add mechanism history, outcome summaries, diversity guidance, and proposal retries | Can proposal-level memory address repeated mechanisms? |
| **Outcome-Blind Proposal-Memory DAG** | Proposal-memory diagnostic | Remove explicit outcome labels from proposal context while retaining exploration memory | What remains when proposals cannot see labelled historical outcomes? |

## From Tree to DAG

![Uniform Random Tree Search topology](experiments/01_uniform_random_tree/figures/search_topology_success.png)

![Uniform Random DAG Search topology](experiments/02_uniform_random_dag/figures/search_topology_success.png)

In these audited left-to-right graphs, arrows mean **parent → child**. The root is neutral, green marks a local parent-relative improvement, red marks no local improvement or regression, and gray marks a failed or interrupted candidate. Green never means merely “below this run's baseline.” Tree candidates have one incoming parent; a DAG Merge may have two.

## Search mechanisms

- **Branching.** Finished historical candidates remain eligible for later expansion, preserving alternative edit lineages instead of collapsing search into one chain.
- **Recombination.** DAG search can ask a proposal agent to synthesize compatible modifications from two parent lineages. A Merge counts as a local success only when its child beats the better (lower-BPB) parent.
- **Credit-guided allocation.** Parent probability reflects current candidate quality, mean improvement among direct children, and an exploration bonus. A node can therefore receive credit for producing useful descendants rather than only for being the current best node.

The Merge-Retry variant did **not** introduce a new credit algorithm. It froze the credit distribution for one Merge intent, tried up to three distinct pairs after semantic rejection, and fell back to a credit-guided Expand if the retry budget or pair pool was exhausted.

## Proposal memory and outcome information

Outcome-Aware Proposal-Memory DAG exposed mechanism-family history, attempt counts, effective transitions, parent-relative outcomes, failures, novelty information, and retry context to the proposal agent.

Outcome-Blind Proposal-Memory DAG retained mechanism history, novelty and retry information, structural context, and credit-guided parent selection, but removed explicit `beat-parent` and failure counts from proposal context. **Outcome-Blind is not history-blind or memoryless:** the search allocator still used measured results to calculate parent credit.

![Outcome-Aware Proposal-Memory DAG topology](experiments/05_outcome_aware_proposal_memory_dag/figures/search_topology_success.png)

![Outcome-Blind Proposal-Memory DAG topology](experiments/06_outcome_blind_proposal_memory_dag/figures/search_topology_success.png)

## Key observations

1. **Linear Search made substantial early progress and later flattened.** The plateau motivated preserving alternatives; it does not make the linear run a failed baseline.
2. **Branching alone was insufficient in the observed Tree run.** Nine of 30 candidates improved over their own parent, while none exceeded the recorded root baseline. This is a result from one trajectory, not a universal judgment on tree search.
3. **Merge exposed separate feasibility and usefulness problems.** Pairs could be rejected as semantically redundant or incompatible; accepted Merges often failed to improve over their better parent.
4. **Credit changed where search effort was spent.** It could reward productive ancestors, but allocation alone did not resolve proposal collapse or pair compatibility.
5. **Evaluation exposure mattered.** Variable optimizer-step and token exposure under a wall-clock budget motivated greater emphasis on within-run, parent-relative evidence.

## Evaluation caveat

Four byte-identical baseline calibration runs under the nominal 300-second protocol completed **20, 21, 20, and 23 optimizer steps**; their `val_bpb` range was **0.018040**. A fixed wall-clock budget was therefore not a fixed optimization budget. The project does not rank the seven experiments by best BPB or treat cross-run own-baseline win rates as strict causal evidence; it emphasizes parent-relative improvement, topology, lineage, and proposal behavior.

## Repository structure

```text
autoresearch-dag/
├── experiments/
│   ├── 00_linear_search/
│   ├── 01_uniform_random_tree/
│   ├── 02_uniform_random_dag/
│   ├── 03_credit_guided_parent_dag/
│   ├── 04_merge_retry_credit_dag/
│   ├── 05_outcome_aware_proposal_memory_dag/
│   └── 06_outcome_blind_proposal_memory_dag/
├── diagnostics/baseline_calibration/
├── figures/selected/
├── docs/
├── report/
│   ├── research_report.md
│   └── claim_audit.md
└── scripts/
```

Each experiment folder contains its own executed code snapshot, compact results, selected figures, and explanatory README or report where available. Repeated code is intentional: experiment-level snapshots preserve implementation provenance rather than retroactively replacing historical implementations with one unified version.

## Limitations

- Mostly single-seed experiments with 30-candidate search budgets
- A single-GPU experimental setting
- Adaptive trajectories that visit different candidate states
- Variable optimization exposure under wall-clock stopping
- Proposal-agent stochasticity and limited statistical power
- Merge compatibility and synthesis limitations

See the full report for detailed methodology, quantitative results, negative findings, and limitations.

## Read more

- [Full Technical Report](report/research_report.md)
- [Claim Audit](report/claim_audit.md)
- [Experiment Naming and Methodology](docs/naming.md)
- [Topology Figure Audit](docs/figure_audit.md)
- [Baseline Calibration](diagnostics/baseline_calibration/README.md)

## Status

An exploratory independent research project. The repository contains completed experiment archives and a first technical report; it does not claim statistical significance, state-of-the-art performance, or universal superiority of DAG search.
