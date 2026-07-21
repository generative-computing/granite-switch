# SPDX-License-Identifier: Apache-2.0
"""Granite Switch vLLM pipeline-parallel generation test.

This is a regression test for the 41-vs-40 layer-count mismatch under PP>1:
the saved config includes one SingleSwitch cache slot plus 40 decoder layers,
while the decoder ModuleList contains only the 40 real decoder layers.

The test intentionally asserts that PP=2 generation succeeds. On the current
buggy implementation it is expected to fail on a 2-GPU machine; after fixing
the vLLM layer-count reporting or PP structure, it should pass.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
_WORKER = Path(__file__).parent / "_pp_generation_worker.py"
_REPO_ROOT = _WORKER.parents[2]
_TIMEOUT = 600
_LOG_TAIL = 30000


def _visible_cuda_device_count():
    """Count visible NVIDIA GPUs without importing torch or initializing CUDA."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible = visible.strip()
        if not visible or visible == "-1":
            return 0
        return len([dev for dev in visible.split(",") if dev.strip()])

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0

    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


pytestmark = [
    pytest.mark.vllm,
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed"),
    pytest.mark.skipif(
        _visible_cuda_device_count() < 2,
        reason="requires at least 2 visible CUDA GPUs",
    ),
]


def test_single_switch_generation_with_pipeline_parallel_size_2(tmp_path):
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) if not pythonpath else f"{_REPO_ROOT}{os.pathsep}{pythonpath}"
    )
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        str(_WORKER),
        "--tmpdir",
        str(tmp_path),
    ]
    stdout_log = tmp_path / "pp_generation_stdout.log"
    stderr_log = tmp_path / "pp_generation_stderr.log"

    try:
        with stdout_log.open("w", encoding="utf-8") as stdout:
            with stderr_log.open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    cmd,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=_TIMEOUT,
                    env=env,
                )
    except subprocess.TimeoutExpired as exc:
        stdout = stdout_log.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_log.read_text(encoding="utf-8", errors="replace")
        if stdout:
            print(stdout[-_LOG_TAIL:])
        if stderr:
            print("STDERR:", stderr[-_LOG_TAIL:])
        raise AssertionError(
            "Granite Switch PP=2 generation timed out.\n"
            f"Full stdout: {stdout_log}\n"
            f"Full stderr: {stderr_log}\n"
            f"STDOUT (last {_LOG_TAIL} chars):\n{stdout[-_LOG_TAIL:]}\n"
            f"STDERR (last {_LOG_TAIL} chars):\n{stderr[-_LOG_TAIL:]}"
        ) from exc

    stdout = stdout_log.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_log.read_text(encoding="utf-8", errors="replace")

    if stdout:
        print(stdout[-_LOG_TAIL:])
    if stderr:
        print("STDERR:", stderr[-_LOG_TAIL:])

    assert result.returncode == 0, (
        "Granite Switch PP=2 generation failed.\n"
        f"Full stdout: {stdout_log}\n"
        f"Full stderr: {stderr_log}\n"
        f"STDOUT (last {_LOG_TAIL} chars):\n{stdout[-_LOG_TAIL:]}\n"
        f"STDERR (last {_LOG_TAIL} chars):\n{stderr[-_LOG_TAIL:]}"
    )
