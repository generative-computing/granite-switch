# SPDX-License-Identifier: Apache-2.0
"""Verify greedy generation equivalence: upstream model vs zero-adapter switch model.

Tests that autoregressive generation produces identical token sequences when a
GraniteSwitch model has a single built-in adapter with zero LoRA weights.

No control tokens appear in the prompt, so:
- Switch layer → adapter_indices=0 everywhere, no token rewrite
- LoRA delta = 0 → decoder layers produce identical output

Each model runs in its own set of subprocesses so CUDA context is fully torn
down between tests (prevents OOM and stale-state issues).

Worker: _generation_equivalence_worker.py (build / run / compare phases).

Markers: requires_model (consistent with test_skinning_equivalence.py).
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


WORKER = Path(__file__).parent / "_generation_equivalence_worker.py"
TIMEOUT = 1200  # 20 min per model (download + build + 2× vLLM load + generate)

MODELS = [
    "ibm-granite/granite-4.0-micro",        # Granite 4.x Dense (small, fast)
]


def _short_name(model: str) -> str:
    """Extract short display name for pytest parametrize IDs."""
    return model.rsplit("/", 1)[-1]


def _run_step(step_name, *cmd_args, timeout):
    """Run a single worker step as a subprocess and assert success.

    Each step is a fresh process so CUDA memory is fully released between steps.
    """
    cmd = [sys.executable, str(WORKER), *cmd_args]
    print(f"\n{'='*60}")
    print(f"  Step: {step_name}")
    print(f"  Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")

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
        f"Generation equivalence step '{step_name}' failed "
        f"(exit code {result.returncode}).\n"
        f"STDOUT (last 2000 chars):\n{result.stdout[-2000:]}\n"
        f"STDERR (last 1000 chars):\n{result.stderr[-1000:]}"
    )


def _run_generation_test(model_name, timeout):
    """Run the 4-step generation equivalence test.

    Each step runs in a separate subprocess so only one vLLM model is ever
    on GPU at a time:
      1. build   — Build switch model + save inputs (CPU only)
      2. run ref — Upstream model generation (GPU)
      3. run sw  — Switch model generation (GPU)
      4. compare — Check token-for-token match (CPU)
    """
    with tempfile.TemporaryDirectory(prefix="gen_equiv_") as work_dir:
        switch_dir = os.path.join(work_dir, "switch")

        # 1. Build switch model (CPU, no GPU needed)
        _run_step(
            "build switch",
            "build", "--model", model_name,
            "--work-dir", work_dir,
            timeout=timeout,
        )

        # 2. Run upstream model in vLLM (GPU)
        _run_step(
            "run upstream",
            "run", "--model", model_name,
            "--work-dir", work_dir,
            "--tag", "ref",
            timeout=timeout,
        )

        # 3. Run switch model in vLLM (GPU)
        _run_step(
            "run switch",
            "run", "--model", switch_dir,
            "--work-dir", work_dir,
            "--tag", "switch",
            timeout=timeout,
        )

        # 4. Compare token sequences (CPU)
        _run_step(
            "compare",
            "compare",
            "--work-dir", work_dir,
            "--label", model_name,
            timeout=60,
        )


@pytest.mark.requires_model
@pytest.mark.parametrize("model_name", MODELS, ids=_short_name)
def test_generation_equivalence(model_name):
    """Generate with upstream vs zero-adapter switch and assert token-exact match."""
    _run_generation_test(model_name, TIMEOUT)
