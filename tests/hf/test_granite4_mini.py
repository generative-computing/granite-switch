# SPDX-License-Identifier: Apache-2.0
"""Verify GraniteSwitch matches upstream for every Granite 4 family variant.

Tests:
- TestGranite4FamilyEquivalence: Sans-LoRA (num_adapters=0) matches upstream.
- TestZeroAdapterEquivalence: Full adapter infrastructure with active switching
  but zero LoRA weights — exercises switch, LoRA wrappers, and adapter routing
  while still matching upstream output.

Miniaturized configs with real scaling multipliers. See
tests/shared/granite4_equivalence.py for the full config registry and
Granite 4 family reference table.
"""

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
    assert_close,
    augment_cfg_with_adapters,
    get_tolerances,
    get_visible_mask,
    make_active_adapter_input,
    transfer_weights,
    transfer_weights_strict,
    zero_lora_weights,
)


def _make_pair(cfg_dict):
    """Create upstream + switch model pair with transferred weights."""
    torch.manual_seed(0)
    upstream = GraniteMoeHybridForCausalLM(GraniteMoeHybridConfig(**cfg_dict)).eval()
    switch = GraniteSwitchForCausalLM(
        GraniteSwitchConfig(**cfg_dict, num_adapters=0)
    ).eval()
    transfer_weights_strict(upstream.state_dict(), switch.state_dict())
    return upstream, switch


_MODEL_NAMES = sorted(GRANITE4_MINI.keys())


@pytest.fixture(params=_MODEL_NAMES)
def model_pair(request):
    """Upstream + switch model pair for each Granite 4 family variant."""
    return request.param, *_make_pair(GRANITE4_MINI[request.param])


class TestGranite4FamilyEquivalence:
    """Verify GraniteSwitch sans LoRA matches upstream for every Granite 4 variant."""

    def test_weight_transfer(self, model_pair):
        """All switch parameters populated from upstream weights."""
        name, upstream, switch = model_pair
        assert upstream is not None

    def test_logits_short(self, model_pair):
        """Short sequence logits match."""
        name, upstream, switch = model_pair
        layer_types = GRANITE4_MINI[name].get("layer_types", [])

        torch.manual_seed(42)
        input_ids = torch.randint(0, 256, (1, 16))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        tol = get_tolerances(layer_types, long_sequence=False)
        if tol is None:
            torch.testing.assert_close(
                switch_out.logits,
                upstream_out.logits,
                atol=0.0,
                rtol=0.0,
                msg=f"{name}: mamba-only logits should be bit-exact",
            )
        else:
            assert_close(
                switch_out.logits,
                upstream_out.logits,
                atol=tol[0],
                rtol=tol[1],
                msg=f"{name}: short sequence logits diverge",
            )

    def test_logits_long(self, model_pair):
        """Longer sequence — fused QKV error compounds through mamba."""
        name, upstream, switch = model_pair
        layer_types = GRANITE4_MINI[name].get("layer_types", [])

        torch.manual_seed(123)
        input_ids = torch.randint(0, 256, (1, 64))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        tol = get_tolerances(layer_types, long_sequence=True)
        if tol is None:
            torch.testing.assert_close(
                switch_out.logits,
                upstream_out.logits,
                atol=0.0,
                rtol=0.0,
                msg=f"{name}: mamba-only logits should be bit-exact",
            )
        else:
            assert_close(
                switch_out.logits,
                upstream_out.logits,
                atol=tol[0],
                rtol=tol[1],
                msg=f"{name}: long sequence logits diverge",
            )

    def test_logits_batch(self, model_pair):
        """Batched input."""
        name, upstream, switch = model_pair
        layer_types = GRANITE4_MINI[name].get("layer_types", [])

        torch.manual_seed(7)
        input_ids = torch.randint(0, 256, (3, 16))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        tol = get_tolerances(layer_types, long_sequence=False)
        if tol is None:
            torch.testing.assert_close(
                switch_out.logits,
                upstream_out.logits,
                atol=0.0,
                rtol=0.0,
                msg=f"{name}: mamba-only batched logits should be bit-exact",
            )
        else:
            assert_close(
                switch_out.logits,
                upstream_out.logits,
                atol=tol[0],
                rtol=tol[1],
                msg=f"{name}: batched logits diverge",
            )


# ── Zero-adapter infrastructure tests ──────────────────────────
#
# These tests exercise adapter infrastructure with zero LoRA weights.
# Vanilla tokens serve as control tokens so the switch computes non-zero
# adapter_indices and the LoRA forward path runs (with zero weights → zero
# delta). The upstream model sees the same tokens as plain text.


def _make_zero_adapter_pair(cfg_dict):
    """Create upstream + zero-adapter switch model pair.

    The switch model has full adapter infrastructure (LoRA wrappers, switch layer)
    with zero LoRA weights. Vanilla tokens are used as control tokens so
    the switch actively computes adapter_indices during forward.
    """
    torch.manual_seed(0)
    upstream = GraniteMoeHybridForCausalLM(GraniteMoeHybridConfig(**cfg_dict)).eval()

    switch_cfg_dict = augment_cfg_with_adapters(cfg_dict)
    switch = GraniteSwitchForCausalLM(GraniteSwitchConfig(**switch_cfg_dict)).eval()

    # Transfer base weights (non-strict: LoRA/switch params left unloaded)
    unloaded = transfer_weights(upstream.state_dict(), switch.state_dict())

    # Verify unloaded params are only LoRA and switch related
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

    # Zero all LoRA weights defensively
    zero_lora_weights(switch)

    return upstream, switch


class TestZeroAdapterNoHiding:
    """Zero LoRA weights, adapter infrastructure active.

    No control tokens in input -- adapter_indices=0 everywhere.
    No hiding triggered (no adapter tokens in input).

    SingleSwitch: bit-exact (no counting head, no position perturbation).
    """

    @pytest.fixture(params=_MODEL_NAMES)
    def model_pair(self, request):
        model_name = request.param
        upstream, switch = _make_zero_adapter_pair(GRANITE4_MINI[model_name])
        return model_name, upstream, switch

    def test_no_control_tokens(self, model_pair):
        """No control tokens -- adapter_indices=0."""
        name, upstream, switch = model_pair

        input_ids = torch.randint(0, 100, (1, 16))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        # SingleSwitch is bit-exact (no counting head, no position perturbation)
        torch.testing.assert_close(
            switch_out.logits,
            upstream_out.logits,
            atol=0.0,
            rtol=0.0,
            msg=f"{name}: should be bit-exact with no control tokens",
        )


class TestZeroAdapterEquivalence:
    """GraniteSwitch with active switching but zero LoRA weights must match upstream.

    Uses vanilla tokens as adapter control tokens -- the switch actually
    computes non-zero adapter_indices and the LoRA forward path runs.
    With zero LoRA weights the delta is zero, so output matches upstream
    at visible (non-hidden) positions.

    All control tokens are KV-hidden (K=finfo.min masking on control dims).
    Hidden positions are intentionally different from upstream because their
    KV contribution is zeroed out by the hiding mechanism. Error at visible
    positions comes from removing the hidden token's attention contribution
    and expanded tensor FP rounding.
    """

    @pytest.fixture(params=_MODEL_NAMES)
    def model_pair(self, request):
        model_name = request.param
        upstream, switch = _make_zero_adapter_pair(GRANITE4_MINI[model_name])
        return model_name, upstream, switch

    def test_logits_short(self, model_pair):
        """Short sequence with active adapter switching."""
        name, upstream, switch = model_pair
        layer_types = GRANITE4_MINI[name].get("layer_types", [])

        input_ids = make_active_adapter_input(1, 16, seed=42)

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        # Mask out kv_hidden positions (intentionally different due to hiding)
        visible = get_visible_mask(input_ids)
        tol = get_tolerances(layer_types, long_sequence=False, has_kv_hidden=True)
        if tol is None:
            torch.testing.assert_close(
                switch_out.logits[visible],
                upstream_out.logits[visible],
                atol=0.0,
                rtol=0.0,
                msg=f"{name}: mamba-only logits should be bit-exact",
            )
        else:
            assert_close(
                switch_out.logits[visible],
                upstream_out.logits[visible],
                atol=tol[0],
                rtol=tol[1],
                msg=f"{name}: short sequence logits diverge (zero-adapter)",
            )

    def test_logits_long(self, model_pair):
        """Longer sequence with active adapter switching."""
        name, upstream, switch = model_pair
        layer_types = GRANITE4_MINI[name].get("layer_types", [])

        input_ids = make_active_adapter_input(1, 64, seed=123)

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        # Mask out kv_hidden positions (intentionally different due to hiding)
        visible = get_visible_mask(input_ids)
        tol = get_tolerances(layer_types, long_sequence=True, has_kv_hidden=True)
        if tol is None:
            torch.testing.assert_close(
                switch_out.logits[visible],
                upstream_out.logits[visible],
                atol=0.0,
                rtol=0.0,
                msg=f"{name}: mamba-only logits should be bit-exact",
            )
        else:
            assert_close(
                switch_out.logits[visible],
                upstream_out.logits[visible],
                atol=tol[0],
                rtol=tol[1],
                msg=f"{name}: long sequence logits diverge (zero-adapter)",
            )

    def test_logits_batch(self, model_pair):
        """Batched input with active adapter switching."""
        name, upstream, switch = model_pair
        layer_types = GRANITE4_MINI[name].get("layer_types", [])

        input_ids = make_active_adapter_input(3, 16, seed=7)

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        # Mask out kv_hidden positions (intentionally different due to hiding)
        visible = get_visible_mask(input_ids)
        tol = get_tolerances(layer_types, long_sequence=False, has_kv_hidden=True)
        if tol is None:
            torch.testing.assert_close(
                switch_out.logits[visible],
                upstream_out.logits[visible],
                atol=0.0,
                rtol=0.0,
                msg=f"{name}: mamba-only batched logits should be bit-exact",
            )
        else:
            assert_close(
                switch_out.logits[visible],
                upstream_out.logits[visible],
                atol=tol[0],
                rtol=tol[1],
                msg=f"{name}: batched logits diverge (zero-adapter)",
            )
