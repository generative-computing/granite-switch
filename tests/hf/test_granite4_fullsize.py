# SPDX-License-Identifier: Apache-2.0
"""Full-size Granite 4 family equivalence tests with random weights.

Instantiates models at their REAL HuggingFace dimensions with random weights.
This catches bugs tied to specific dimension values (e.g., head_dim=64 vs 32,
or interaction with real scaling multipliers at real hidden sizes).

Memory management: models are created and destroyed sequentially (upstream first,
then switch) to minimize peak memory.

MoE models excluded — expert weights exceed CPU memory. See
tests/shared/granite4_equivalence.py for the full config reference.
"""

import gc

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
    assert_close,
    transfer_weights_strict,
    get_tolerances,
    GRANITE4_FULLSIZE,
)


def _run_equivalence(cfg_dict, *, seq_len=8):
    """Run full equivalence test with sequential model creation.

    Returns (upstream_logits, switch_logits).
    """
    torch.manual_seed(42)
    input_ids = torch.randint(0, cfg_dict["vocab_size"], (1, seq_len))

    # Phase 1: upstream model
    torch.manual_seed(0)
    upstream = GraniteMoeHybridForCausalLM(
        GraniteMoeHybridConfig(**cfg_dict)
    ).eval()

    with torch.no_grad():
        upstream_logits = upstream(input_ids=input_ids, use_cache=False).logits.clone()

    upstream_sd = upstream.state_dict()
    del upstream
    gc.collect()

    # Phase 2: switch model
    switch = GraniteSwitchForCausalLM(
        GraniteSwitchConfig(**cfg_dict, num_adapters=0)
    ).eval()

    transfer_weights_strict(upstream_sd, switch.state_dict())
    del upstream_sd
    gc.collect()

    with torch.no_grad():
        switch_logits = switch(input_ids=input_ids, use_cache=False).logits.clone()

    del switch
    gc.collect()

    return upstream_logits, switch_logits


pytestmark = pytest.mark.slow

_MODEL_NAMES = sorted(GRANITE4_FULLSIZE.keys())


class TestGranite4FullSize:
    """Full-size equivalence: GraniteSwitch sans LoRA == upstream GraniteMoeHybrid."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_weight_transfer(self, model_name):
        """All switch parameters can be populated from upstream weights."""
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

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_logits_match(self, model_name):
        """Logits match on short sequence at full model dimensions."""
        cfg = GRANITE4_FULLSIZE[model_name]
        layer_types = cfg.get("layer_types", [])

        upstream_logits, switch_logits = _run_equivalence(cfg, seq_len=8)

        tol = get_tolerances(layer_types)
        if tol is None:
            torch.testing.assert_close(
                switch_logits, upstream_logits,
                atol=0.0, rtol=0.0,
                msg=f"{model_name}: mamba-only logits should be bit-exact",
            )
        else:
            assert_close(
                switch_logits, upstream_logits,
                atol=tol[0], rtol=tol[1],
                msg=f"{model_name}: full-size logits diverge",
            )
