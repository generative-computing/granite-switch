# SPDX-License-Identifier: Apache-2.0
"""Verify vLLM GraniteSwitch matches upstream for every Granite 4 variant.

Miniaturized configs with real scaling multipliers (from GRANITE4_MINI).
All 9 Granite 4 family models are tested: dense, hybrid, and hybrid+MoE.

Split into:
- TestGranite4FamilyWeightTransfer: HF-level CPU-only weight transfer (in-process)
- TestZeroAdapterWeightTransfer: CPU-only weight transfer with adapters (in-process)
- GPU test classes: run in subprocess via _granite4_mini_tests.py

vLLM counterpart to tests/hf/test_granite4_mini.py.
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
    augment_cfg_with_adapters,
    transfer_weights,
    transfer_weights_strict,
    GRANITE4_MINI,
)

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

_MODEL_NAMES = sorted(GRANITE4_MINI.keys())


# ── Weight transfer tests (HF-level, no vLLM) ────────────────────


class TestGranite4FamilyWeightTransfer:
    """HF-level weight transfer: all switch params populated from upstream.

    CPU-only — no vLLM or CUDA required.
    """

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_weight_transfer(self, model_name):
        cfg = GRANITE4_MINI[model_name]

        torch.manual_seed(0)
        upstream = GraniteMoeHybridForCausalLM(
            GraniteMoeHybridConfig(**cfg)
        ).eval()

        switch = GraniteSwitchForCausalLM(
            GraniteSwitchConfig(**cfg, num_adapters=0)
        ).eval()

        transfer_weights_strict(upstream.state_dict(), switch.state_dict())

        del upstream, switch
        gc.collect()


class TestZeroAdapterWeightTransfer:
    """HF-level weight transfer with adapter infrastructure.

    CPU-only — no vLLM or CUDA required.
    """

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_weight_transfer(self, model_name):
        cfg = GRANITE4_MINI[model_name]

        torch.manual_seed(0)
        upstream = GraniteMoeHybridForCausalLM(
            GraniteMoeHybridConfig(**cfg)
        ).eval()

        switch_cfg_dict = augment_cfg_with_adapters(cfg)
        switch = GraniteSwitchForCausalLM(
            GraniteSwitchConfig(**switch_cfg_dict)
        ).eval()

        unloaded = transfer_weights(upstream.state_dict(), switch.state_dict())

        for name in unloaded:
            assert any(k in name for k in (
                "lora_A", "lora_B", "switch", "adapter_token_ids",
                "control_to_substitute_lut",
            )), f"Unexpected unloaded parameter: {name}"

        assert len(unloaded) > 0, "Expected LoRA/switch params to be unloaded"

        del upstream, switch
        gc.collect()


# ── GPU test class wrappers (subprocess) ──────────────────────────

_INNER = Path(__file__).parent / "_granite4_mini_tests.py"
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
class TestGranite4FamilyEquivalence:
    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_suite(self, model_name):
        _run_inner_class(f"TestGranite4FamilyEquivalence and {model_name}")


@pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed")
class TestZeroAdapterNoHiding:
    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_suite(self, model_name):
        _run_inner_class(f"TestZeroAdapterNoHiding and {model_name}")


@pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed")
class TestZeroAdapterEquivalence:
    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_suite(self, model_name):
        _run_inner_class(f"TestZeroAdapterEquivalence and {model_name}")
