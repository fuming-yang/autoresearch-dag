"""Run autoresearch data preparation and training on Modal.

Usage:
    uv run modal setup
    uv run modal run --env main modal_app.py::prepare --num-shards 1
    uv run modal run --env main modal_app.py::train
    uv run modal run --env main modal_app.py::parallel
    uv run modal deploy modal_app.py --env main

The dataset and tokenizer are persisted in the ``autoresearch-cache`` Volume.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal


APP_NAME = "autoresearch"
LOCAL_PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = "/workspace/autoresearch"
CACHE_DIR = "/root/.cache/autoresearch"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("autoresearch-cache", create_if_missing=True)

# Dependencies are cached in the Image independently of the source code. The
# orchestrator sends each experiment's train.py as a Function argument.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_sync(str(LOCAL_PROJECT_DIR), extra_options="--no-dev")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    .add_local_dir(
        LOCAL_PROJECT_DIR,
        PROJECT_DIR,
        ignore=[".git", ".venv", "__pycache__", "*.pyc"],
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={CACHE_DIR: cache_volume},
)
def prepare(num_shards: int = 8, download_workers: int = 8) -> None:
    """Download data shards and build the tokenizer in the persistent Volume."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")

    subprocess.run(
        [
            "python",
            "prepare.py",
            "--num-shards",
            str(num_shards),
            "--download-workers",
            str(download_workers),
        ],
        cwd=PROJECT_DIR,
        check=True,
    )
    cache_volume.commit()


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65536,
    timeout=24 * 60 * 60,
    max_containers=16,
    volumes={CACHE_DIR: cache_volume},
)
def train(
    run_id: str | int = 0,
    seed: int = 42,
    train_source: str | None = None,
) -> dict[str, int | float | str]:
    """Run one independent experiment in its own single-H100 container."""
    env = os.environ.copy()
    env["AUTORESEARCH_SEED"] = str(seed)
    print(f"Starting parallel run {run_id} with seed {seed}")

    with tempfile.TemporaryDirectory(prefix="autoresearch-") as temporary_dir:
        run_dir = Path(temporary_dir) / "project"
        shutil.copytree(PROJECT_DIR, run_dir)
        if train_source is not None:
            (run_dir / "train.py").write_text(train_source, encoding="utf-8")

        process = subprocess.Popen(
            ["python", "train.py"],
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output = []
        for line in process.stdout:
            output.append(line)
            print(f"[run {run_id}] {line}", end="", flush=True)

        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, ["python", "train.py"])

    text = "".join(output)
    metrics: dict[str, int | float | str] = {
        "run_id": run_id,
        "seed": seed,
        "log": text,
    }
    integer_fields = {"num_steps", "target_steps"}
    for name in (
        "val_bpb",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "num_steps",
        "target_steps",
    ):
        match = re.search(rf"^{name}:\s+([0-9.]+)$", text, re.MULTILINE)
        if match:
            value = float(match.group(1))
            metrics[name] = int(value) if name in integer_fields else value
    return metrics


@app.local_entrypoint()
def parallel(count: int = 2, base_seed: int = 42) -> None:
    """Fan out independent tests across `count` single-GPU Modal containers."""
    if not 1 <= count <= 16:
        raise ValueError("count must be between 1 and 16")

    inputs = [(run_id, base_seed + run_id) for run_id in range(count)]
    print(f"Launching {count} parallel H100 experiments...")
    results = list(train.starmap(inputs, order_outputs=True))
    print("Parallel experiments completed:")
    for result in results:
        print(result)
