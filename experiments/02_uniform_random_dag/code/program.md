# autoresearch

This repository runs an asynchronous swarm of agents that improve `train.py`.
The objective is the lowest validation bits per byte (`val_bpb`).

## Fixed experiment protocol

- Every experiment runs exactly `TRAIN_STEPS` optimizer steps as defined in `prepare.py`.
- `prepare.py`, evaluation, tokenizer, data, and step count are fixed.
- An agent may edit only `train.py` and may make one coherent experimental change.
- Runtime and VRAM are reported, but elapsed time never determines when training stops.
- A run is valid only when `num_steps == target_steps`.

Run a standalone Modal experiment with
`uv run modal run --env main modal_app.py::train`.

## Parallel execution

First commit the repository baseline, prepare the data, and verify one standalone run.
Deploy the Modal App, then launch the desired number of concurrent workers:

```bash
uv run modal deploy modal_app.py --env main
uv run orchestrator.py --workers 4 --modal-env main --max-experiments 20
```

Use `--max-experiments N` for a bounded run. `--watchdog-seconds` is an optional
abnormal-hang safeguard; it is disabled by default and is not an experiment budget.

The orchestrator maintains one global DAG and schedules a new experiment whenever
any GPU becomes idle. There is no batch barrier between GPUs.

## DAG operations

`expand` asks an agent to make one logical change from one finished parent.

`merge` gives an agent two finished commits, both hypotheses, both results, and their
direct `git diff`. The agent decides whether the ideas are distinct and compatible.
It does not run `git merge` and does not find a common ancestor. If compatible, it
synthesizes a single coherent `train.py` starting from the better-result parent. If
redundant or conflicting, it rejects the pair.

Git and the research DAG have the same topology:

- baseline: parentless Git commit;
- expand: one-parent Git commit;
- merge: two-parent Git commit created from the synthesized tree.

`dag.json` stores information Git does not: status (`running`, `finished`, `failed`),
parents, descendant count, result for finished nodes, best direct-child result, GPU,
timestamps, and hypothesis. Only the orchestrator writes this file.

## Sampling

`sample.py` draws a uniform random number in `[0, 1)`: values below `0.5` select
`expand`, while values at or above `0.5` select `merge`. It then uniformly samples
a finished parent or an untried pair of finished parents. If no merge pair is
available, it falls back to `expand`.

## Agent contract

The orchestrator checks out an isolated worktree and asks the agent to edit
`train.py`. The agent writes an untracked `decision.json`:

```json
{"status": "accept", "hypothesis": "short semantic description of the change"}
```

For a redundant or conflicting merge, use `status: "reject"`. The orchestrator alone
creates commits, launches training, parses results, updates the DAG, and releases GPUs.
Failed runs stay in the DAG without a result and are not eligible as future parents.
