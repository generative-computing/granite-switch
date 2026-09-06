# SPDX-License-Identifier: Apache-2.0
"""Quantization tests for GraniteSwitch vLLM backend.
Subprocess wrapper — runs _quantization_tests.py in a subprocess.

All GPU work happens in the subprocess so the parent pytest process
never creates a CUDA context (required for Exclusive_Process GPU mode).

Tests BitsAndBytes INT4 and FP8 quantization:
1. Base model weights are actually quantized
2. LoRA/aLoRA weights remain in full precision (bfloat16)
3. Adapters still activate under quantization
4. LoRA dimensions are correct (not corrupted by packed weight shapes)

Each quantization method runs in a single subprocess so the module-scoped
fixture (model load) is shared across all tests for that method.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None
_CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(
        not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
        reason="requires CUDA GPU and vLLM installed",
    ),
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
]

_INNER = Path(__file__).parent / "_quantization_tests.py"
_TIMEOUT = 600  # 10 min — model download + load + inference


def _run_inner(pattern):
    """Run inner tests matching pattern in a subprocess."""
    cmd = [sys.executable, "-m", "pytest", str(_INNER),
           "-v", "-s", "--tb=short", "-k", pattern]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"Inner tests failed (exit {result.returncode})"


# ---------------------------------------------------------------------------
# BitsAndBytes INT4 (NF4)
# All INT4 tests run in a single subprocess (one model load).
# ---------------------------------------------------------------------------

class TestBnBInt4:
    """BnB INT4: quantization, LoRA precision, adapter activation, dimensions, memory."""

    def test_suite(self):
        _run_inner("BnBInt4")


# ---------------------------------------------------------------------------
# FP8 (vLLM native)
# All FP8 tests run in a single subprocess (one model load).
# ---------------------------------------------------------------------------

class TestFP8:
    """FP8: quantization, LoRA precision, adapter activation."""

    def test_suite(self):
        _run_inner("FP8")
