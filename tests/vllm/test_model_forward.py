# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward pass tests for GraniteSwitchForCausalLM (vLLM backend).
Subprocess wrapper — runs _model_forward_tests.py in a subprocess.

All GPU work happens in the subprocess so the parent pytest process
never creates a CUDA context (required for Exclusive_Process GPU mode).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = [
    pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed (GPU checked by inner tests)"),
    pytest.mark.vllm,
    pytest.mark.gpu,
]

_INNER = Path(__file__).parent / "_model_forward_tests.py"
_TIMEOUT = 600


def _run_inner_class(class_name):
    cmd = [sys.executable, "-m", "pytest", str(_INNER),
           "-v", "-s", "--tb=short", "-k", class_name]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"Inner tests failed (exit {result.returncode})"


class TestModelInstantiation:
    def test_suite(self):
        _run_inner_class("TestModelInstantiation")


class TestForwardOutputShape:
    def test_suite(self):
        _run_inner_class("TestForwardOutputShape")


class TestAdapterIndicesWiring:
    def test_suite(self):
        _run_inner_class("TestAdapterIndicesWiring")


class TestKVVisibility:
    def test_suite(self):
        _run_inner_class("TestKVVisibility")
