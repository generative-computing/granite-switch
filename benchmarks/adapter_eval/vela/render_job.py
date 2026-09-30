#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the values file for one adapter-benchmark job on Vela.

The output is a values file for the ``mlbatch/pytorchjob-generator`` Helm
chart, written as JSON (which Helm reads as YAML). ``submit.sh`` calls this;
see ``docs/ADAPTER_BENCHMARK.md``.

The harness is shipped inside the job itself, as a base64 tarball of
``benchmarks/`` in an environment variable. The pod unpacks it and runs
``pod_entry.sh``. So the harness is exactly the local working tree, and the
benchmarked commit's own copy (if any) is never used.

The reference job also ships the SR model code the same way: only its model
package, taken with ``git archive`` from a local shadow-residual checkout at
a pinned commit (``SR_REPO``, ``SR_REF``). It is never added to this repo;
the results record only its commit sha.

Cluster names, storage paths and secret names come from the environment,
which ``submit.sh`` loads from the gitignored ``local/local.env``. Standard
library only.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXCLUDE_DIRS = {"local", ".rendered", "__pycache__"}
# The SR model package inside a shadow-residual checkout.
SR_CODE_DIR = "src/shadow_residual/shadow_residual"
# One environment string must stay below Linux's 128 KiB MAX_ARG_STRLEN.
MAX_PAYLOAD = 120_000

RESOURCES = {
    # mode: (gpus, cpus, memory)
    # discover and stage use no GPU, but the storage mount needs ~250 GB of
    # free local disk for its cache, which only the GPU nodes have.
    "discover": (1, 4, "16Gi"),
    "stage": (1, 4, "16Gi"),
    "bench": (1, 16, "128Gi"),
    # reference.py spreads its cells over every GPU of the pod.
    "reference": (4, 32, "256Gi"),
    "script": (1, 16, "128Gi"),
}


def fail(msg: str) -> None:
    print(f"render_job: {msg}", file=sys.stderr)
    sys.exit(1)


def need(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        fail(f"{name} is not set (see vela/local.env.example)")
    return value


def harness_payload(extra: dict[str, Path]) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted((REPO / "benchmarks").rglob("*")):
            rel = path.relative_to(REPO)
            if path.is_file() and not EXCLUDE_DIRS & set(rel.parts):
                tar.add(path, arcname=str(rel))
        for name, path in extra.items():
            tar.add(path, arcname=f"extra/{name}")
    payload = base64.b64encode(buf.getvalue()).decode()
    if len(payload) > MAX_PAYLOAD:
        fail(f"harness payload is {len(payload)} bytes, limit {MAX_PAYLOAD}")
    return payload


def sr_payload() -> tuple[str, str]:
    """The SR model code at ``SR_REF``, as a base64 tarball, and its full sha."""
    repo, ref = need("SR_REPO"), need("SR_REF")

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", repo, *args], check=True, capture_output=True
        ).stdout

    try:
        sha = git("rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()
        tgz = git("archive", "--format=tar.gz", sha, SR_CODE_DIR)
    except subprocess.CalledProcessError as e:
        fail(f"SR code at {ref}: {e.stderr.decode().strip()}")
    payload = base64.b64encode(tgz).decode()
    if len(payload) > MAX_PAYLOAD:
        fail(f"SR code payload is {len(payload)} bytes, limit {MAX_PAYLOAD}")
    return payload, sha


def harness_version() -> tuple[str, bool]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    return git("rev-parse", "HEAD"), bool(
        git("status", "--porcelain", "--", "benchmarks")
    )


def env_var(name: str, value) -> dict:
    return {"name": name, "value": str(value)}


def build(args) -> dict:
    mode = args.mode
    gpus, cpus, memory = RESOURCES[mode]
    mount = need("PVC_MOUNT")
    sha, dirty = harness_version()

    extra: dict[str, Path] = {}
    if mode == "stage":
        extra["selection.json"] = Path(need("SELECTION_FILE"))
        if os.environ.get("JUDGE_PROMPT_FILE"):
            extra["judge_prompt.txt"] = Path(os.environ["JUDGE_PROMPT_FILE"])
    if mode == "script":
        extra["script.py"] = Path(args.script)
    for path in extra.values():
        if not path.is_file():
            fail(f"{path} does not exist")

    # The run timestamp goes first, as in every Vela job of this team.
    env = [
        env_var("RUN_TS", args.run_ts),
        env_var("PYTHONUNBUFFERED", "1"),
        env_var("ADAPTER_BENCH_MODE", mode),
        env_var("ADAPTER_BENCH_HARNESS_SHA", sha),
        env_var("ADAPTER_BENCH_HARNESS_DIRTY", int(dirty)),
        env_var("XDG_CACHE_HOME", "/workspace/.cache"),
        env_var("UV_CACHE_DIR", "/workspace/.cache/uv"),
        env_var("UV_HTTP_TIMEOUT", "600"),
    ]
    optional = {
        "HF_HOME": "HF_HOME",
        "BENCH_ROOT": "ADAPTER_BENCH_ROOT",
        "WORK_ROOT": "ADAPTER_BENCH_WORK_ROOT",
        "BASE_MODEL_PATH": "ADAPTER_BENCH_BASE_MODEL",
        "ADAPTER_SOURCE_ROOTS": "ADAPTER_BENCH_ADAPTER_SOURCES",
        "EVAL_SOURCE_ROOTS": "ADAPTER_BENCH_EVAL_SOURCES",
        "UV_SYNC_ARGS": "ADAPTER_BENCH_UV_SYNC_ARGS",
        "JUDGE_URL": "ADAPTER_BENCH_JUDGE_URL",
        "JUDGE_MODEL": "ADAPTER_BENCH_JUDGE_MODEL",
    }
    for local_name, pod_name in optional.items():
        if os.environ.get(local_name):
            env.append(env_var(pod_name, os.environ[local_name]))
    # submit.sh has already resolved the model's own settings (BENCH_ROOT, ...).
    if args.model:
        env.append(env_var("ADAPTER_BENCH_MODEL", args.model))
    if os.environ.get("JUDGE_SECRET_NAME"):
        env.append(
            {
                "name": "RITS_API_KEY",
                "secret": {
                    "name": os.environ["JUDGE_SECRET_NAME"],
                    "key": need("JUDGE_SECRET_KEY"),
                },
            }
        )
    if mode in ("bench", "reference", "script"):
        need("BENCH_ROOT")
        need("WORK_ROOT")
        if mode != "reference":
            env.append(env_var("ADAPTER_BENCH_COMMIT", args.sha))
        if args.limit:
            env.append(env_var("ADAPTER_BENCH_LIMIT", args.limit))
        if args.only:
            env.append(env_var("ADAPTER_BENCH_ONLY", args.only))
        if args.extra_args:
            env.append(env_var("ADAPTER_BENCH_EXTRA_ARGS", args.extra_args))
    if mode == "reference":
        payload, sr_sha = sr_payload()
        env.append(env_var("ADAPTER_BENCH_SR_REF", sr_sha))
        env.append(env_var("ADAPTER_BENCH_SR_TGZ", payload))
    if mode == "stage":
        need("BENCH_ROOT")
        if args.replace:
            env.append(env_var("ADAPTER_BENCH_STAGE_REPLACE", "1"))
    env.append(env_var("ADAPTER_BENCH_HARNESS_TGZ", harness_payload(extra)))

    return {
        "namespace": need("NAMESPACE"),
        "jobName": args.job_name,
        "queueName": os.environ.get("QUEUE_NAME", "default-queue"),
        "priority": os.environ.get("PRIORITY", "default-priority"),
        "numPods": 1,
        "numCpusPerPod": cpus,
        "numGpusPerPod": gpus,
        "totalMemoryPerPod": memory,
        "containerImage": need("CONTAINER_IMAGE"),
        "imagePullSecrets": [{"name": need("IMAGE_PULL_SECRET")}],
        # Never retry, and keep a failed pod a day for its logs.
        "retryLimit": 0,
        "deletionOnFailureGracePeriodDuration": "24h",
        "failureGracePeriodDuration": "24h",
        "environmentVariables": env,
        "volumes": [
            {"name": "bench-storage", "claimName": need("PVC_NAME"), "mountPath": mount}
        ],
        "setupCommands": [
            "mkdir -p /workspace/harness && cd /workspace/harness"
            ' && printf %s "$ADAPTER_BENCH_HARNESS_TGZ" | base64 -d | tar xzf -'
            " && bash benchmarks/adapter_eval/vela/pod_entry.sh"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=sorted(RESOURCES), required=True)
    p.add_argument("--job-name", required=True)
    p.add_argument("--run-ts", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sha", help="bench: full commit sha")
    p.add_argument("--model", help="adapters.yaml model id (default: the first)")
    p.add_argument("--limit", type=int)
    p.add_argument("--only")
    p.add_argument(
        "--extra-args", help="bench, reference, script: extra flags for the program"
    )
    p.add_argument("--script", help="script: local Python file to run in the pod")
    p.add_argument("--replace", action="store_true", help="stage: overwrite cells")
    args = p.parse_args(argv)
    if args.mode in ("bench", "script") and not args.sha:
        fail(f"--sha is required for {args.mode}")
    if args.mode == "script" and not args.script:
        fail("--script is required for script")
    values = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(values, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
