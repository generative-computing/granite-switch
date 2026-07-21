# SPDX-License-Identifier: Apache-2.0
"""Verify skinned GraniteSwitch models produce bit-exact logits vs the originals.

Each model runs in its own subprocess so CUDA context is fully torn down between
tests (avoids OOM and stale-state issues that plague in-process model reloading).

Two worker scripts:
  - _skinning_equivalence_worker.py     — HF backend (CPU, float32)
  - _skinning_equivalence_worker_vllm.py — vLLM backend (GPU, native dtype)

The vLLM worker uses separate subprocesses for each phase (build, run original,
run skin, compare) so only one vLLM model is ever on GPU at a time.  This
prevents OOM on 7B+ models where GPU memory isn't fully reclaimed within a
single process.

vLLM tests come in two tiers:
  - Fast (requires_model only): single 64-token request per model.
  - Thorough (slow + requires_model): 8 requests, lengths 3–256 tokens.

Markers: slow, requires_model (consistent with test_build_e2e.py).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HF_WORKER = Path(__file__).parent / "_skinning_equivalence_worker.py"
VLLM_WORKER = Path(__file__).parent / "_skinning_equivalence_worker_vllm.py"
HF_TIMEOUT = 1800  # 30 min per model (download + 2× load + forward)
VLLM_TIMEOUT = 1800  # 30 min per model (download + build skin + 2× vLLM load)

# Granite models for skinning equivalence tests.
ALL_MODELS = [
    "ibm-granite/granite-4.0-micro",  # Granite 4.x Dense (small, fast)
]

# Models tested via vLLM (requires GPU).
VLLM_MODELS = [
    "ibm-granite/granite-4.0-micro",  # Granite 4.x Dense (small, fast)
]


def _short_name(model: str) -> str:
    """Extract short display name for pytest parametrize IDs."""
    return model.rsplit("/", 1)[-1]


def _run_worker(worker_script, model_name, timeout, extra_args=()):
    """Run a worker subprocess and assert success."""
    result = subprocess.run(
        [sys.executable, str(worker_script), *extra_args, model_name],
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    # Always print worker output for visibility
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    assert result.returncode == 0, (
        f"Skinning equivalence failed for {model_name} "
        f"(exit code {result.returncode}).\n"
        f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n"
        f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}"
    )


def _run_step(step_name, *cmd_args, timeout):
    """Run a single vLLM worker step as a subprocess and assert success.

    Each step is a fresh process so CUDA memory is fully released between steps.
    """
    cmd = [sys.executable, str(VLLM_WORKER), *cmd_args]
    print(f"\n{'=' * 60}")
    print(f"  Step: {step_name}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    assert result.returncode == 0, (
        f"vLLM worker step '{step_name}' failed "
        f"(exit code {result.returncode}).\n"
        f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n"
        f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}"
    )


def _run_vllm_test(model_name, timeout, fast=False):
    """Run the 4-step vLLM skinning equivalence test.

    Each step runs in a separate subprocess so only one vLLM model is ever
    on GPU at a time:
      1. build  — Build skin + save inputs (CPU only)
      2. run    — Original model logprobs (GPU)
      3. run    — Skinned model logprobs (GPU)
      4. compare — Check bit-exact match (CPU)
    """
    with tempfile.TemporaryDirectory(prefix="skinning_equiv_vllm_") as work_dir:
        skin_dir = os.path.join(work_dir, "skin")
        inputs_json = os.path.join(work_dir, "inputs.json")
        ref_json = os.path.join(work_dir, "ref.json")
        sw_json = os.path.join(work_dir, "sw.json")
        fast_flag = ["--fast"] if fast else []

        # 1. Build skin (CPU, no GPU needed)
        _run_step(
            "build skin",
            "build",
            "--model",
            model_name,
            "--work-dir",
            work_dir,
            *fast_flag,
            timeout=timeout,
        )

        # 2. Run original model in vLLM (GPU)
        _run_step(
            "run original",
            "run",
            "--model",
            model_name,
            "--inputs",
            inputs_json,
            "--output",
            ref_json,
            timeout=timeout,
        )

        # 3. Run skinned model in vLLM (GPU)
        _run_step(
            "run skin",
            "run",
            "--model",
            skin_dir,
            "--inputs",
            inputs_json,
            "--output",
            sw_json,
            timeout=timeout,
        )

        # 4. Compare logprobs (CPU)
        _run_step(
            "compare",
            "compare",
            "--ref",
            ref_json,
            "--switch",
            sw_json,
            "--label",
            model_name,
            timeout=60,
        )


# ── HF tests (slow) ──────────────────────────────────────────────
#
# SKIPPED: The GraniteSwitch HF backend uses fused QKV/gate-up projections
# (symmetric with vLLM) rather than the separate projections used by upstream
# HuggingFace GraniteMoeHybrid.  Fused projections change the floating-point
# reduction order, so bit-exact equivalence with the upstream HF model is not
# achievable.  The vLLM skinning tests below are the authoritative equivalence
# check — both sides use the same fused-projection architecture.


@pytest.mark.skip(
    reason="HF backend uses fused projections (not bit-exact with upstream HF)"
)
@pytest.mark.slow
@pytest.mark.requires_model
@pytest.mark.parametrize("model_name", ALL_MODELS, ids=_short_name)
def test_skinning_equivalence_hf(model_name):
    """Skin *model_name* to GraniteSwitch and assert bit-exact logits via HF."""
    _run_worker(HF_WORKER, model_name, HF_TIMEOUT)


# ── vLLM tests: fast (single request) ────────────────────────────


@pytest.mark.requires_model
@pytest.mark.parametrize("model_name", VLLM_MODELS, ids=_short_name)
def test_skinning_equivalence_vllm(model_name):
    """Skin *model_name* and assert bit-exact logprobs via vLLM.

    Fast: single 64-token request per model.
    Each phase runs in its own subprocess for clean CUDA teardown.
    """
    _run_vllm_test(model_name, VLLM_TIMEOUT, fast=True)


# ── vLLM tests: thorough (8 requests, varying lengths) ───────────


@pytest.mark.slow
@pytest.mark.requires_model
@pytest.mark.parametrize("model_name", VLLM_MODELS, ids=_short_name)
def test_skinning_equivalence_vllm_thorough(model_name):
    """Skin *model_name* and assert bit-exact logprobs via vLLM.

    Thorough: 8 individual requests of varying lengths (3–256 tokens).
    Each phase runs in its own subprocess for clean CUDA teardown.
    """
    _run_vllm_test(model_name, VLLM_TIMEOUT, fast=False)
