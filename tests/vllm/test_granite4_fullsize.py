# SPDX-License-Identifier: Apache-2.0
"""Full-size Granite 4 family equivalence tests with random weights (vLLM).

Split into:
- TestGranite4FullSizeWeightTransfer: HF-level CPU-only weight transfer (in-process)
- TestGranite4FullSizeEquivalence: GPU tests in subprocess via _granite4_fullsize_tests.py

MoE models excluded — expert weights exceed GPU memory.
"""

import gc
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridForCausalLM,
)

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM

from tests.shared.granite4_equivalence import (
    transfer_weights_strict,
    GRANITE4_FULLSIZE,
)

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

_MODEL_NAMES = sorted(GRANITE4_FULLSIZE.keys())


# ── Weight transfer tests (HF-level, no vLLM) ────────────────────


@pytest.mark.slow
class TestGranite4FullSizeWeightTransfer:
    """HF-level weight transfer at full model dimensions.

    CPU-only — no vLLM or CUDA required.
    """

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_weight_transfer(self, model_name):
        cfg = GRANITE4_FULLSIZE[model_name]

        torch.manual_seed(0)
        upstream = GraniteMoeHybridForCausalLM(
            GraniteMoeHybridConfig(**cfg)
        ).eval()
        upstream_sd = upstream.state_dict()
        del upstream
        gc.collect()

        switch = GraniteSwitchForCausalLM(
            GraniteSwitchConfig(**cfg, num_adapters=0)
        ).eval()

        transfer_weights_strict(upstream_sd, switch.state_dict())

        del switch, upstream_sd
        gc.collect()


# ── GPU test class wrappers (subprocess) ──────────────────────────

_INNER = Path(__file__).parent / "_granite4_fullsize_tests.py"
_TIMEOUT = 1200


def _run_inner_class(class_name):
    cmd = [sys.executable, "-m", "pytest", str(_INNER),
           "-v", "-s", "--tb=short", "-k", class_name]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"Inner tests failed (exit {result.returncode})"


@pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed")
@pytest.mark.vllm
@pytest.mark.gpu
@pytest.mark.slow
class TestGranite4FullSizeEquivalence:
    def test_suite(self):
        _run_inner_class("TestGranite4FullSizeEquivalence")
