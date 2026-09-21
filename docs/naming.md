# Final experiment naming

| Legacy name | Final display name | Final directory name | Main change |
|---|---|---|---|
| Linear AutoResearch | Linear Search | `00_linear_search` | Sequential keep/discard search from the current retained state |
| Random Tree-30 | Uniform Random Tree Search | `01_uniform_random_tree` | Uniformly select one eligible historical parent; Expand only, no pruning |
| Random DAG-30 | Uniform Random DAG Search | `02_uniform_random_dag` | 50/50 Expand/Merge intent with uniformly sampled parents or untried pairs |
| Credit-Guided DAG-A | Credit-Guided Parent DAG | `03_credit_guided_parent_dag` | Replace uniform parent selection with credit-softmax allocation using quality, mean child improvement, and exploration |
| Credit-Guided DAG-B | Merge-Retry Credit-Guided DAG | `04_merge_retry_credit_dag` | Retain credit selection and 50/50 intent; retry up to three Merge pairs from a frozen distribution, then fallback Expand |
| Proposal-Memory DAG-C | Outcome-Aware Proposal-Memory DAG | `05_outcome_aware_proposal_memory_dag` | Add mechanism/proposal memory with attempt and parent-relative outcome counts, diversity guidance, novelty/preparation checks, and proposal retries |
| History-Blind DAG-D | Outcome-Blind Proposal-Memory DAG | `06_outcome_blind_proposal_memory_dag` | Remove explicit outcome counts and “better-result parent” wording from proposal context while retaining mechanism history and outcome-based parent credit |

## Why “outcome-aware” and “outcome-blind”?

The controlled prompt-level difference is narrower than general history access. Both experiments preserve proposal/mechanism history, novelty classification, prior transitions, retry context, and credit-guided parent selection. The outcome-aware version exposes per-family `beat-parent` and failure counts. The outcome-blind version exposes attempt counts without those outcome labels and describes the Merge checkout as the selected base parent rather than the better-result parent. It is therefore not history-blind or memoryless.

## Credit and Merge mechanics

The Credit-Guided Parent DAG scores every eligible finished finite-result node with:

`credit = quality + 0.5 × mean direct-child improvement + 0.5 × exploration bonus`

with `quality_scale = 0.01`, softmax temperature 1.0, and no hard pruning or inactivity removal. The operation intent remains 50% Expand / 50% Merge. A Merge pair is sampled by sequential credit-softmax selection without replacement, conditioned on being untried; one semantic rejection ends that intent.

The Merge-Retry Credit-Guided DAG leaves that credit formula, eligibility set, no-pruning policy, and 50/50 intent unchanged. For a Merge intent it freezes the eligible nodes and credit distribution, samples up to three distinct pairs without replacement, and falls back to a credit-guided Expand if the pair pool or retry budget is exhausted.
