"""Helpers for launching and monitoring local vLLM servers in tutorials."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence

import requests

DEFAULT_MAX_MODEL_LEN = 32768  # 32k, fits comfortably on an A100 (40/80 GiB).


def launch_vllm(
    model: str,
    port: int,
    log_file: str,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 1,
    enforce_eager: bool = True,
    extra_args: Sequence[str] = (),
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
) -> subprocess.Popen:
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
        *(["--enforce-eager"] if enforce_eager else []),
        *extra_args,
    ]

    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
    print(f"Launching {model} on :{port} (pid {proc.pid}, log -> {log_file})")
    return proc


# vLLM log keywords in order of progression (last match wins = most advanced stage)
_VLLM_STAGES = [
    ("Starting to load model", "Downloading / loading model"),
    ("Loading safetensors", "Loading model weights into GPU"),
    ("GPU KV cache size", "Allocating KV cache"),
    ("Capturing CUDA graphs", "Warming up — capturing CUDA graphs"),
    ("Application startup complete", "Starting API server"),
]


def _current_stage(log_file: str) -> str:
    """Return the most advanced stage seen so far in the vLLM log."""
    try:
        with open(log_file) as f:
            content = f.read()
        stage = "Starting up"
        for keyword, label in _VLLM_STAGES:
            if keyword in content:
                stage = label
        return stage
    except Exception:
        return "Starting up"


_HEARTBEAT_INTERVAL = 30  # seconds between heartbeat prints when stage hasn't changed


def wait_for_server(port: int, timeout: int = 600, log_file: str | None = None) -> bool:
    """Poll /v1/models until vLLM is ready, showing log-based stage progress."""
    t0 = time.time()
    print("Waiting for vLLM server — this may take a few minutes...")
    last_stage = ""
    last_print_time = -_HEARTBEAT_INTERVAL  # force print on first iteration
    while time.time() - t0 < timeout:
        try:
            if (
                requests.get(
                    f"http://localhost:{port}/v1/models", timeout=2
                ).status_code
                == 200
            ):
                print(f"  Server ready on :{port} in {int(time.time() - t0)}s")
                return True
        except Exception:
            pass

        elapsed = int(time.time() - t0)
        stage = _current_stage(log_file) if log_file else "waiting"
        stage_changed = stage != last_stage
        heartbeat_due = (elapsed - last_print_time) >= _HEARTBEAT_INTERVAL

        if stage_changed or heartbeat_due:
            print(f"  [{elapsed:3d}s] {stage}...")
            last_stage = stage
            last_print_time = elapsed

        time.sleep(5)

    print(f"\n  timed out after {timeout}s - check the log file")
    return False


def tail_log(log_file: str, n: int = 20) -> None:
    r = subprocess.run(["tail", f"-{n}", log_file], capture_output=True, text=True)
    print(r.stdout)


def kill_stale_vllm_processes(wait_seconds: int = 5) -> None:
    """Terminate stale vLLM processes that can hold GPU memory after a notebook restart."""
    r = subprocess.run(
        ["pgrep", "-f", "vllm.entrypoints"], capture_output=True, text=True
    )
    pids = [p for p in r.stdout.strip().split("\n") if p]
    if pids:
        print(f"Killing stale vLLM processes: {pids}")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(wait_seconds)
    else:
        print("No stale vLLM processes found.")


def print_gpu_state() -> None:
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    print("GPU state:", r.stdout.strip())
