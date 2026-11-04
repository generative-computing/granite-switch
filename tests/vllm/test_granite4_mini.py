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
    GRANITE4_MINI,
    augment_cfg_with_adapters,
    transfer_weights,
    transfer_weights_strict,
)

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

_MODEL_NAMES = sorted(GRANITE4_MINI.keys())

# The vLLM equivalence classes cost minutes per variant, so PR runs keep one
# representative and the release tier runs the rest. The CPU weight-transfer
# classes stay on every variant: they take seconds.
_PR_MODEL = "4.0-micro"
_GPU_MODEL_PARAMS = [
    pytest.param(name, marks=() if name == _PR_MODEL else pytest.mark.release)
    for name in _MODEL_NAMES
]


# ── Weight transfer tests (HF-level, no vLLM) ────────────────────


class TestGranite4FamilyWeightTransfer:
    """HF-level weight transfer: all switch params populated from upstream.

    CPU-only — no vLLM or CUDA required.
    """

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_weight_transfer(self, model_name):
        cfg = GRANITE4_MINI[model_name]

        torch.manual_seed(0)
        upstream = GraniteMoeHybridForCausalLM(GraniteMoeHybridConfig(**cfg)).eval()

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
        upstream = GraniteMoeHybridForCausalLM(GraniteMoeHybridConfig(**cfg)).eval()

        switch_cfg_dict = augment_cfg_with_adapters(cfg)
        switch = GraniteSwitchForCausalLM(GraniteSwitchConfig(**switch_cfg_dict)).eval()

        unloaded = transfer_weights(upstream.state_dict(), switch.state_dict())

        for name in unloaded:
            assert any(
                k in name
                for k in (
                    "lora_A",
                    "lora_B",
                    "switch",
                    "adapter_token_ids",
                    "control_to_substitute_lut",
                )
            ), f"Unexpected unloaded parameter: {name}"

        assert len(unloaded) > 0, "Expected LoRA/switch params to be unloaded"

        del upstream, switch
        gc.collect()


# ── GPU tests (subprocess) ────────────────────────────────────────

_INNER = Path(__file__).parent / "_granite4_mini_tests.py"
# One subprocess now runs all three classes for a variant (three engine boots in
# all), where each class used to get its own subprocess and its own two engines.
_TIMEOUT = 1800


def _run_inner(selector):
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(_INNER),
        "-v",
        "-s",
        "--tb=short",
        "-k",
        selector,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT)
    if result.stdout:
        print(result.stdout[-6000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    assert result.returncode == 0, f"Inner tests failed (exit {result.returncode})"


@pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed")
class TestGranite4VllmEquivalence:
    """All vLLM equivalence classes for one variant, in ONE pytest process.

    TestGranite4FamilyEquivalence, TestZeroAdapterNoHiding and
    TestZeroAdapterEquivalence in _granite4_mini_tests.py share a module fixture
    that boots each of the variant's three models once. Running them in a single
    process is what lets them share it: per-class processes would rebuild it,
    and the boots, three times.
    """

    @pytest.mark.parametrize("model_name", _GPU_MODEL_PARAMS)
    def test_suite(self, model_name):
        _run_inner(model_name)
