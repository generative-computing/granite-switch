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
    extra_args: Sequence[str] | None = None,
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
    ]
    if extra_args:
        cmd += extra_args

    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
    print(f"Launching {model} on :{port} (pid {proc.pid}, log -> {log_file})")
    return proc


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Poll /health until vLLM is ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"http://localhost:{port}/health", timeout=2).status_code == 200:
                print(f"\n  Server ready on :{port} in {int(time.time() - t0)}s")
                return True
        except Exception:
            pass

        elapsed = int(time.time() - t0)
        print(f"  waiting for :{port} ... {elapsed}s", end="\r")
        time.sleep(5)

    print(f"\n  timed out after {timeout}s - check the log file")
    return False


def tail_log(log_file: str, n: int = 20) -> None:
    r = subprocess.run(["tail", f"-{n}", log_file], capture_output=True, text=True)
    print(r.stdout)


def kill_stale_vllm_processes(wait_seconds: int = 5) -> None:
    """Terminate stale vLLM processes that can hold GPU memory after a notebook restart."""
    r = subprocess.run(["pgrep", "-f", "vllm.entrypoints"], capture_output=True, text=True)
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
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print("GPU state:", r.stdout.strip())
