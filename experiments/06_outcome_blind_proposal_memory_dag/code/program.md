# autoresearch: Credit-Guided DAG

This repository runs a controlled local single-GPU experiment on one NVIDIA RTX 5070 Laptop GPU. The objective is the lowest validation bits per byte (`val_bpb`). The orchestrator must run with `workers = 1`.

## Fixed experiment protocol

- Every run uses `AUTORESEARCH_SEED=42`.
- Every run uses `TRAIN_TIME_BUDGET = 300.0` seconds.
- Termination is based on accumulated synchronized training time. Each measured optimizer step is bracketed by CUDA synchronization, and post-warmup durations accumulate into `total_training_time`.
- Training schedule progress is exactly `progress = min(total_training_time / TRAIN_TIME_BUDGET, 1.0)`.
- The fixed evaluation, result parser, validator, and controlled-invariant checks remain authoritative. The RTX 5070 memory-efficient SDPA backend, query chunking, and other compatibility modifications must remain unchanged.
- This protocol does not use a fixed optimizer-step termination rule; optimizer-step count is not an experiment stopping condition.

The research agent may edit only `train.py` and may make one coherent experimental change. Normal research variables include learning rate, weight decay, warmup and warmdown, optimizer choices, model architecture, attention, and RoPE. It must not modify the controlled seed, time budget, time/progress rules, evaluation, validator, controlled invariants, RTX 5070 compatibility code, or orchestrator experiment rules.

Run locally with `uv run orchestrator.py --workers 1 --max-experiments N`. `--watchdog-seconds` is only an optional abnormal-hang safeguard and is not the training budget.

## DAG operations

This is a DAG, not a tree. `expand` asks for one logical change from one finished parent. `merge` supplies two finished parents, their hypotheses, and direct Git diff. The agent decides whether their ideas are distinct and compatible and, if so, semantically synthesizes one coherent `train.py` from the selected base parent. It does not invoke `git merge` or search for a common ancestor. Redundant or conflicting proposals are rejected. A Merge operation may try at most three distinct credit-guided pairs from one frozen selection snapshot; after three semantic rejects, or earlier pair-pool exhaustion, it falls back to the ordinary credit-guided Expand selection from that same snapshot.

Git and the research DAG have the same topology: baseline has no parent, expand has one parent, and merge has two parents. All finished valid nodes with finite `val_bpb` remain eligible; no node is hard-pruned.

## Credit-guided parent allocation

For eligible node `v`, let `L_v` be its `val_bpb`, and `L_best` the minimum eligible loss. With `quality_scale = 0.01`:

`Q(v) = (L_best - L_v) / 0.01`

For every finished valid child `c` that names `v` as any parent, `improvement(v,c) = (L_v - L_c) / 0.01`. `E(v)` is the mean of these improvements, or zero when there are none. A merge child contributes independently relative to each parent. Let `n_v` be the number of times `v` has actually been selected as a parent, and `T` the total parent selections:

`H(v) = sqrt(log(T + 1) / (n_v + 1))`

Using `lambda_credit = 0.5` and `gamma_explore = 0.5`:

`C(v) = Q(v) + 0.5 * E(v) + 0.5 * H(v)`

Parent probabilities use a stable softmax with `temperature = 1.0`:

`P(v) = exp((C(v) - max_u C(u)) / temperature) / sum_u exp((C(u) - max_u C(u)) / temperature)`

Expand samples one parent from this distribution. Merge retains the 50/50 operation probability and untried-pair rule, and samples distinct pairs using the same sequential credit-softmax distribution as before. Semantic rejects remove only the exact pair and do not update `T` or `n_v`; an accepted executable Merge stops resampling immediately. If all available attempts fail, fallback Expand uses the frozen original node distribution. Each actual executable parent use is counted (`T += 1`, `n_v += 1`); a merge adds two uses. Operation metadata records the original draw, pair attempts, rejects, fallback, final operation, actual parents, and counter snapshots.

## Candidate agent contract

The inner candidate-generation command explicitly uses `codex exec --model gpt-5.6-luna`; it must never depend on a default or fallback model. The agent writes an untracked `decision.json` with `status`, `hypothesis`, and (for accepted DAG-C proposals) a `proposal` object containing `family`, `mechanism`, `effective_transition`, `summary`, and `mechanistic_justification`. It does not commit and does not run the full training experiment. The orchestrator alone creates commits, launches training, parses results, updates the DAG, and releases the GPU. Failed runs remain non-eligible. DAG-C's proposal memory contains only history created during the current DAG-C run.
