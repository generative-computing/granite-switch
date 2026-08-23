# SPDX-License-Identifier: Apache-2.0
"""PP=2 vs PP=1 generation equivalence for the Granite Switch SWITCH kernel.

The SWITCH kernel relies on a recompute-per-rank contract under pipeline
parallelism: the first PP rank runs the switch and ships token-leading
``adapter_indices`` across the stage boundary, and every later rank *recomputes*
its per-module LoRA kernel metadata (bitmasks + remapped indices) locally from
that tensor (see granite_switch.vllm.granite_switch_model.forward and
_finalize_fused_lora). The existing PP test only asserts liveness (PP=2 emits N
tokens) — a metadata-recompute bug would still emit N (wrong) tokens.

This test asserts *correctness*: the same pre-composed checkpoint, with an
adapter active, must generate identical greedy tokens under PP=2 and PP=1.
Identical tokens ⇒ the per-rank recompute matches what a single rank computes.

Requires ≥2 GPUs (skipped otherwise) and downloads a real model
(requires_model). Worker: _pp_equivalence_worker.py (run / compare phases).
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
_WORKER = Path(__file__).parent / "_pp_equivalence_worker.py"
_REPO_ROOT = _WORKER.parents[2]

MODEL_ID = "ibm-granite/granite-switch-4.1-3b-preview"
_TIMEOUT = 1800  # download + 2× (vLLM load + generate)


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
    pytest.mark.requires_model,
    pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed"),
    pytest.mark.skipif(
        _visible_cuda_device_count() < 2,
        reason="requires at least 2 visible CUDA GPUs",
    ),
]


def _subprocess_env():
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) if not pythonpath else f"{_REPO_ROOT}{os.pathsep}{pythonpath}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_step(step_name, *cmd_args, timeout):
    """Run one worker step as a fresh subprocess so CUDA is released between steps."""
    cmd = [sys.executable, str(_WORKER), *cmd_args]
    print(
        f"\n{'=' * 60}\n  Step: {step_name}\n  Command: {' '.join(map(str, cmd))}\n{'=' * 60}"
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_subprocess_env(),
    )

    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    assert result.returncode == 0, (
        f"PP equivalence step '{step_name}' failed (exit code {result.returncode}).\n"
        f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n"
        f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}"
    )


def test_pp2_matches_pp1_token_equivalence(tmp_path):
    """Same checkpoint + active adapter: PP=2 greedy tokens must equal PP=1."""
    work_dir = str(tmp_path)

    # 1. PP=1 generation (GPU)
    _run_step(
        "run pp1",
        "run",
        "--model",
        MODEL_ID,
        "--work-dir",
        work_dir,
        "--pp-size",
        "1",
        "--tag",
        "pp1",
        timeout=_TIMEOUT,
    )

    # 2. PP=2 generation (GPU) — fresh process so the PP=1 engine is gone
    _run_step(
        "run pp2",
        "run",
        "--model",
        MODEL_ID,
        "--work-dir",
        work_dir,
        "--pp-size",
        "2",
        "--tag",
        "pp2",
        timeout=_TIMEOUT,
    )

    # 3. Token-for-token comparison (CPU)
    _run_step(
        "compare",
        "compare",
        "--work-dir",
        work_dir,
        timeout=60,
    )
