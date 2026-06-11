# SPDX-License-Identifier: Apache-2.0
"""TP-aware LoRA tests (subprocess wrapper).

Runs the actual tests from _tp_lora_tests.py in a subprocess so the parent
pytest process never creates a CUDA context.
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

_INNER = Path(__file__).parent / "_tp_lora_tests.py"
_TIMEOUT = 300


def _run_inner_class(class_name):
    cmd = [sys.executable, "-m", "pytest", str(_INNER),
           "-v", "-s", "--tb=short", "-k", class_name]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"Inner tests failed (exit {result.returncode})"


class TestTP1IsNoop:
    def test_suite(self):
        _run_inner_class("TestTP1IsNoop")


class TestColumnParallelSlicing:
    def test_suite(self):
        _run_inner_class("TestColumnParallelSlicing")


class TestRowParallelSlicing:
    def test_suite(self):
        _run_inner_class("TestRowParallelSlicing")


class TestPackedSlicing:
    def test_suite(self):
        _run_inner_class("TestPackedSlicing")


class TestWeightLoaderAttached:
    def test_suite(self):
        _run_inner_class("TestWeightLoaderAttached")


class TestWeightLoaderSlices:
    def test_suite(self):
        _run_inner_class("TestWeightLoaderSlices")
