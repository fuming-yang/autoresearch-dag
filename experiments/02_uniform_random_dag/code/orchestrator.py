"""Asynchronous multi-GPU autoresearch orchestrator.

Git commits carry the experiment DAG topology. JSON stores live status and metrics.
Expand commits have one parent; semantic merge commits have two explicit parents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sample import choose


def now():
    return datetime.now(timezone.utc).isoformat()


class ControlledInvariantError(RuntimeError):
    pass


def git(repo, *args, input_text=None, check=True):
    result = subprocess.run(
        ["git", *args], cwd=repo, input=input_text, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result


async def stop_process(process):
    if process.returncode is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        process.kill()
    await process.wait()


class Dag:
    def __init__(self, path):
        self.path = path
        self.data = {"version": 1, "nodes": [], "rejected_merges": []}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @property
    def nodes(self):
        return self.data["nodes"]

    def get(self, node_id):
        return next(node for node in self.nodes if node["id"] == node_id)

    def next_id(self):
        used = [int(node["id"].split("_")[-1]) for node in self.nodes]
        return f"exp_{max(used, default=-1) + 1:06d}"

    def recompute(self):
        children = {node["id"]: set() for node in self.nodes}
        for node in self.nodes:
            for parent in node.get("parents", []):
                children.setdefault(parent, set()).add(node["id"])

        by_id = {node["id"]: node for node in self.nodes}
        for node in self.nodes:
            seen, stack = set(), list(children.get(node["id"], ()))
            while stack:
                child = stack.pop()
                if child in seen:
                    continue
                seen.add(child)
                stack.extend(children.get(child, ()))
            node["descendants"] = len(seen)
            direct_losses = []
            for child_id in children.get(node["id"], ()):
                child = by_id[child_id]
                if child.get("status") == "finished":
                    direct_losses.append(child["result"]["val_bpb"])
            node["best_child_result"] = min(direct_losses) if direct_losses else None

    def validate(self):
        seen = set()
        for node in self.nodes:
            node_id = node.get("id")
            if not node_id or node_id in seen:
                raise ValueError(f"duplicate or missing DAG node id: {node_id}")
            parents = node.get("parents", [])
            expected = {"baseline": 0, "expand": 1, "merge": 2}.get(node.get("mode"))
            if expected is None or len(parents) != expected or any(parent not in seen for parent in parents):
                raise ValueError(f"invalid parents or mode for {node_id}")
            if node.get("status") == "finished":
                if not isinstance(node.get("result"), dict):
                    raise ValueError(f"finished node {node_id} has no result")
            elif "result" in node:
                raise ValueError(f"non-finished node {node_id} must not store a result")
            seen.add(node_id)

    def save(self):
        self.validate()
        self.recompute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


class Orchestrator:
    def __init__(self, args):
        self.args = args
        project_dir = Path(__file__).parent.resolve()
        top_level = git(project_dir, "rev-parse", "--show-toplevel").stdout.strip()
        self.repo = Path(top_level).resolve()
        self.project_relative = project_dir.relative_to(self.repo)
        self.dag = Dag(args.state.resolve())
        self.worktrees = self.repo.parent / f".{self.repo.name}-autoresearch-worktrees"
        self.results = project_dir / "results"
        self.worktrees.mkdir(exist_ok=True)
        self.results.mkdir(exist_ok=True)

    def project_in_worktree(self, worktree):
        return worktree / self.project_relative

    def reserve(self, choice, worker_slot):
        node_id = self.dag.next_id()
        parents = [self.dag.get(parent_id) for parent_id in choice["parents"]]
        base = min(parents, key=lambda item: item["result"]["val_bpb"])
        tree = git(self.repo, "rev-parse", f"{base['commit']}^{{tree}}").stdout.strip()
        commit_args = ["commit-tree", tree]
        for parent in parents:
            commit_args.extend(["-p", parent["commit"]])
        placeholder = git(
            self.repo, *commit_args, input_text=f"{node_id}: running {choice['mode']} experiment\n"
        ).stdout.strip()
        node = {
            "id": node_id,
            "status": "running",
            "mode": choice["mode"],
            "parents": choice["parents"],
            "descendants": 0,
            "best_child_result": None,
            "commit": placeholder,
            "hypothesis": "pending agent decision",
            "worker_slot": worker_slot,
            "created_at": now(),
            "started_at": now(),
        }
        self.dag.nodes.append(node)
        git(self.repo, "update-ref", f"refs/autoresearch/nodes/{node['id']}", placeholder)
        self.dag.save()
        return node

    def add_worktree(self, path, commit):
        git(self.repo, "worktree", "add", "--detach", str(path), commit)

    def remove_worktree(self, path):
        if path.exists():
            git(self.repo, "worktree", "remove", "--force", str(path), check=False)

    def agent_args(self, worktree, prompt):
        if self.args.agent_command:
            parts = shlex.split(self.args.agent_command, posix=os.name != "nt")
            return [part.format(worktree=str(worktree), prompt=prompt) for part in parts]
        return [
            "codex", "exec", "--ephemeral", "--sandbox", "workspace-write",
            "--ask-for-approval", "never", "-C", str(worktree), prompt,
        ]

    async def run_agent(self, worktree, prompt, log_path):
        with log_path.open("wb") as output:
            process = await asyncio.create_subprocess_exec(
                *self.agent_args(worktree, prompt), stdout=output,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                return await process.wait()
            except asyncio.CancelledError:
                await stop_process(process)
                raise

    def prompt(self, node):
        decision = (
            "Write decision.json containing exactly a JSON object with keys status and hypothesis. "
            "Do not commit and do not run the full training experiment. Only train.py may be edited."
        )
        if node["mode"] == "expand":
            parent = self.dag.get(node["parents"][0])
            return (
                "Design one coherent autoresearch experiment starting from the checked-out parent. "
                "Read program.md and train.py, make one logical change intended to lower val_bpb. "
                f"The parent hypothesis was: {parent.get('hypothesis') or 'baseline'}. {decision} "
                "Use status=accept."
            )
        left, right = (self.dag.get(parent_id) for parent_id in node["parents"])
        return (
            "Consider a semantic merge of two completed experiments. Do not use git merge and do not "
            "search for a common ancestor. Inspect both versions with git show and compare them with "
            f"git diff {left['commit']} {right['commit']} -- train.py. The hypotheses are:\n"
            f"A ({left['commit']}): {left.get('hypothesis')}\n"
            f"B ({right['commit']}): {right.get('hypothesis')}\n"
            "The worktree starts from the better-result parent. Decide whether the ideas are distinct, "
            "compatible, and worth combining. If they are, synthesize a coherent train.py and use "
            "status=accept. If they are redundant or semantically conflicting, leave train.py alone and "
            f"use status=reject. {decision}"
        )

    def read_decision(self, worktree):
        path = worktree / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        if decision.get("status") not in {"accept", "reject"}:
            raise ValueError("decision status must be accept or reject")
        if not isinstance(decision.get("hypothesis"), str) or not decision["hypothesis"].strip():
            raise ValueError("decision hypothesis must be a non-empty string")
        return decision

    def create_commit(self, worktree, node, base_commit, hypothesis):
        # Rebuild the index from the chosen base so an agent can never accidentally
        # include decision.json or unrelated files in the experiment tree.
        git(worktree, "read-tree", base_commit)
        git(worktree, "add", "--", "train.py")
        unchanged = git(worktree, "diff", "--cached", "--quiet", base_commit, "--", "train.py", check=False)
        if unchanged.returncode == 0:
            raise ValueError("agent accepted the experiment without changing train.py")
        tree = git(worktree, "write-tree").stdout.strip()
        args = ["commit-tree", tree]
        parent_commits = [self.dag.get(parent)["commit"] for parent in node["parents"]]
        for parent_commit in parent_commits:
            args.extend(["-p", parent_commit])
        commit = git(
            worktree, *args, input_text=f"{node['id']} {node['mode']}: {hypothesis}\n"
        ).stdout.strip()
        git(self.repo, "update-ref", f"refs/autoresearch/nodes/{node['id']}", commit)
        actual = git(self.repo, "show", "-s", "--format=%P", commit).stdout.split()
        if actual != parent_commits:
            raise RuntimeError("created commit parents do not match the experiment DAG")
        return commit

    def validate_controlled_invariants(self, candidate_source, baseline_source):
        required = {
            "fixed seed source": r'^seed = int\(os\.environ\.get\("AUTORESEARCH_SEED", "42"\)\)$',
            "fixed torch seed": r'^torch\.manual_seed\(seed\)$',
            "fixed CUDA seed": r'^torch\.cuda\.manual_seed\(seed\)$',
            "300-second budget": r'^TRAIN_TIME_BUDGET = 300\.0(?:\s+#.*)?$',
            "time-based termination": r'^while total_training_time < TRAIN_TIME_BUDGET:$',
            "time-based progress": r'^    progress = min\(total_training_time / TRAIN_TIME_BUDGET, 1\.0\)$',
            "synchronized step timing": (
                r'torch\.cuda\.synchronize\(\).*?t0 = time\.time\(\).*?'
                r'torch\.cuda\.synchronize\(\).*?t1 = time\.time\(\).*?dt = t1 - t0'
            ),
            "accumulated step timing": r'if step > 10:\s+total_training_time \+= dt',
            "RTX 5070 SDPA backend": r'with sdpa_kernel\(SDPBackend\.EFFICIENT_ATTENTION\):',
            "RTX 5070 query chunking": r'^SDPA_QUERY_CHUNK_SIZE = 128(?:\s+#.*)?$',
            "evaluation": r'val_bpb = evaluate_bpb\(model, tokenizer, DEVICE_BATCH_SIZE\)',
        }
        missing = [
            name for name, pattern in required.items()
            if re.search(pattern, candidate_source, re.MULTILINE | re.DOTALL) is None
        ]
        if "while step < TRAIN_STEPS:" in candidate_source:
            missing.append("fixed-step termination is forbidden")
        if missing:
            raise ControlledInvariantError(
                "candidate violates controlled invariants: " + ", ".join(missing)
            )

        for name, pattern in required.items():
            if re.search(pattern, baseline_source, re.MULTILINE | re.DOTALL) is None:
                raise RuntimeError(f"baseline is missing controlled invariant: {name}")

    async def train(self, node, project_worktree, worker_slot):
        log_path = self.results / f"{node['id']}.log"
        seed = 42
        env = os.environ.copy()
        env["AUTORESEARCH_SEED"] = str(seed)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "train.py",
            cwd=project_worktree,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            if self.args.watchdog_seconds:
                output, _ = await asyncio.wait_for(
                    process.communicate(), self.args.watchdog_seconds
                )
            else:
                output, _ = await process.communicate()
        except asyncio.TimeoutError as exc:
            await stop_process(process)
            raise RuntimeError("Local experiment exceeded the safety watchdog") from exc

        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, [sys.executable, "train.py"])
        text = output.decode()
        log_path.write_text(str(text), encoding="utf-8")
        response = {}
        integer_fields = {"num_steps", "target_steps"}
        for name in (
            "val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
            "num_steps", "target_steps",
        ):
            match = re.search(rf"^{name}:\s+([0-9.]+)$", text, re.MULTILINE)
            if match:
                value = float(match.group(1))
                response[name] = int(value) if name in integer_fields else value
        fields = {
            name: response[name]
            for name in (
                "val_bpb", "training_seconds", "total_seconds", "peak_vram_mb",
                "num_steps", "target_steps",
            )
            if name in response
        }
        required = {"val_bpb", "peak_vram_mb", "num_steps", "target_steps"}
        if not required <= fields.keys() or int(fields["num_steps"]) != int(fields["target_steps"]):
            raise RuntimeError("training summary is missing or did not complete the fixed step count")
        fields["num_steps"] = int(fields["num_steps"])
        fields["target_steps"] = int(fields["target_steps"])
        return fields, log_path

    async def run_baseline(self, worker_slot):
        source = git(self.repo, "rev-parse", "HEAD").stdout.strip()
        tree = git(self.repo, "rev-parse", f"{source}^{{tree}}").stdout.strip()
        # Start a self-contained experiment graph whose Git root has no parent.
        node_id = self.dag.next_id()
        commit = git(
            self.repo, "commit-tree", tree, input_text=f"{node_id}: autoresearch baseline\n"
        ).stdout.strip()
        node = {
            "id": node_id, "status": "running", "mode": "baseline",
            "parents": [], "descendants": 0, "best_child_result": None,
            "commit": commit, "hypothesis": "baseline", "worker_slot": worker_slot,
            "created_at": now(), "started_at": now(),
        }
        self.dag.nodes.append(node)
        git(self.repo, "update-ref", f"refs/autoresearch/nodes/{node['id']}", commit)
        self.dag.save()
        worktree = self.worktrees / node["id"]
        try:
            self.add_worktree(worktree, commit)
            project_worktree = self.project_in_worktree(worktree)
            result, log_path = await self.train(node, project_worktree, worker_slot)
            node.update(status="finished", result=result, finished_at=now(), log=str(log_path))
        except Exception as exc:
            node.update(status="failed", error=str(exc), finished_at=now())
            raise
        finally:
            self.remove_worktree(worktree)
            self.dag.save()

    async def run_experiment(self, node, worker_slot):
        worktree = self.worktrees / node["id"]
        agent_log = self.results / f"{node['id']}.agent.log"
        try:
            parents = [self.dag.get(parent) for parent in node["parents"]]
            base = min(parents, key=lambda item: item["result"]["val_bpb"])
            self.add_worktree(worktree, base["commit"])
            project_worktree = self.project_in_worktree(worktree)
            if await self.run_agent(project_worktree, self.prompt(node), agent_log):
                raise RuntimeError(f"research agent failed; see {agent_log}")
            decision = self.read_decision(project_worktree)
            if decision["status"] == "reject":
                if node["mode"] != "merge":
                    raise RuntimeError("an expand experiment cannot be rejected")
                self.dag.data["rejected_merges"].append(sorted(node["parents"]))
                self.dag.nodes.remove(node)
                git(self.repo, "update-ref", "-d", f"refs/autoresearch/nodes/{node['id']}")
                self.dag.save()
                return
            candidate_source = (project_worktree / "train.py").read_text(encoding="utf-8")
            baseline = next(node for node in self.dag.nodes if node["mode"] == "baseline")
            baseline_source = git(
                self.repo, "show", f"{baseline['commit']}:train.py"
            ).stdout
            self.validate_controlled_invariants(candidate_source, baseline_source)
            node["hypothesis"] = decision["hypothesis"].strip()
            node["commit"] = self.create_commit(
                project_worktree, node, base["commit"], node["hypothesis"]
            )
            node["training_started_at"] = now()
            self.dag.save()
            result, log_path = await self.train(node, project_worktree, worker_slot)
            node.update(status="finished", result=result, finished_at=now(), log=str(log_path))
        except ControlledInvariantError:
            self.dag.nodes.remove(node)
            git(self.repo, "update-ref", "-d", f"refs/autoresearch/nodes/{node['id']}")
            return False
        except Exception as exc:
            if node.get("hypothesis") == "pending agent decision":
                node["hypothesis"] = "experiment preparation failed"
            node.update(status="failed", error=str(exc), finished_at=now())
        finally:
            self.remove_worktree(worktree)
            self.dag.save()

    async def run(self):
        if git(self.repo, "rev-parse", "--is-inside-work-tree", check=False).returncode:
            raise RuntimeError(f"{self.repo} is not a Git repository")
        if git(self.repo, "diff", "--quiet", check=False).returncode or git(
            self.repo, "diff", "--cached", "--quiet", check=False
        ).returncode:
            raise RuntimeError("commit tracked changes before starting the orchestrator")

        # Drop legacy/incomplete reservations that predate placeholder commits.
        reservations = [node for node in self.dag.nodes if node.get("commit") is None]
        for node in reservations:
            self.dag.nodes.remove(node)
        stale = [node for node in self.dag.nodes if node.get("status") == "running"]
        for node in stale:
            node.update(status="failed", error="orchestrator restarted while node was running", finished_at=now())
        if stale or reservations:
            self.dag.save()

        by_id = {node["id"]: node for node in self.dag.nodes}
        for node in self.dag.nodes:
            expected = [by_id[parent]["commit"] for parent in node["parents"]]
            actual = git(self.repo, "show", "-s", "--format=%P", node["commit"]).stdout.split()
            if actual != expected:
                raise RuntimeError(f"Git parents disagree with dag.json for {node['id']}")

        launched = 0
        if not self.dag.nodes:
            await self.run_baseline(self.args.worker_slots[0])
            launched += 1
        if not any(node.get("status") == "finished" for node in self.dag.nodes):
            raise RuntimeError("no successful baseline or finished node exists")

        active = {}
        while active or not self.args.max_experiments or launched < self.args.max_experiments:
            free = [slot for slot in self.args.worker_slots if slot not in active.values()]
            while free and (not self.args.max_experiments or launched < self.args.max_experiments):
                worker_slot = free.pop(0)
                choice = choose(self.dag.data)
                node = self.reserve(choice, worker_slot)
                task = asyncio.create_task(self.run_experiment(node, worker_slot))
                active[task] = worker_slot
                launched += 1
            if not active:
                break
            done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                active.pop(task)
                if await task is False:
                    launched -= 1


def parse_args():
    parser = argparse.ArgumentParser(description="Run asynchronous autoresearch on Modal GPUs")
    parser.add_argument("--workers", type=int, required=True, help="maximum concurrent Modal jobs")
    parser.add_argument("--modal-app", default="autoresearch", help="deployed Modal App name")
    parser.add_argument("--modal-env", default="main", help="Modal Environment name")
    parser.add_argument("--state", type=Path, default=Path(__file__).with_name("dag.json"))
    parser.add_argument("--max-experiments", type=int, default=0, help="0 means run until interrupted")
    parser.add_argument("--watchdog-seconds", type=float, default=0, help="safety timeout only; 0 disables it")
    parser.add_argument(
        "--agent-command",
        help="optional command template containing {worktree} and {prompt}; defaults to codex exec",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    args.worker_slots = list(range(args.workers))
    if not args.state.is_absolute():
        args.state = Path(__file__).parent / args.state
    return args


if __name__ == "__main__":
    try:
        asyncio.run(Orchestrator(parse_args()).run())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
