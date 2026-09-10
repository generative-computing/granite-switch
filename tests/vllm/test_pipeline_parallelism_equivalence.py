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

Requires ≥2 GPUs (skipped otherwise) and composes a real MultiSwitch checkpoint
(requires_model, E2E-gated). Worker: _pp_equivalence_worker.py (run / compare).
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

# The published previews are legacy SingleSwitch checkpoints and no longer load;
# compose a real MultiSwitch checkpoint on demand (warm-reused under
# GRANITE_SWITCH_E2E_DIR), the same one test_multi_switch_mixed_tech builds. PP
# only needs a loadable checkpoint with an active adapter — any MultiSwitch does.
BASE_MODEL = "ibm-granite/granite-4.1-3b"
ADAPTER_REPOS = [
    "ibm-granite/granitelib-rag-r1.0",
    "ibm-granite/granitelib-guardian-r1.0",
]
_E2E_ROOT = Path(os.environ.get("GRANITE_SWITCH_E2E_DIR", "/tmp/granite_switch_e2e"))
_TIMEOUT = 1800  # compose (warm-reused) + 2× (vLLM load + generate)


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
    pytest.mark.skipif(
        os.environ.get("GRANITE_SWITCH_E2E_MODELS") != "1",
        reason="composes a real ~3B checkpoint; set GRANITE_SWITCH_E2E_MODELS=1",
    ),
]


@pytest.fixture(scope="module")
def model_path():
    """Compose (or warm-reuse) a mixed MultiSwitch checkpoint; return its dir."""
    out_dir = _E2E_ROOT / "multi-mixed"
    if (out_dir / "config.json").exists():
        print(f"warm-reuse {out_dir}", file=sys.stderr)
        return str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        BASE_MODEL,
        *[arg for r in ADAPTER_REPOS for arg in ("--adapters", r)],
        "--output",
        str(out_dir),
    ]
    print("composing (mixed):", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        pytest.fail(
            f"mixed compose failed (exit {r.returncode})\n"
            f"--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}",
            pytrace=False,
        )
    return str(out_dir)


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


def test_pp2_matches_pp1_token_equivalence(tmp_path, model_path):
    """Same checkpoint + active adapter: PP=2 greedy tokens must equal PP=1."""
    work_dir = str(tmp_path)

    # 1. PP=1 generation (GPU)
    _run_step(
        "run pp1",
        "run",
        "--model",
        model_path,
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
        model_path,
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
