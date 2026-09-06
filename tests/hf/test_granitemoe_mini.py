# SPDX-License-Identifier: Apache-2.0
"""Verify GraniteSwitch matches upstream ``granitemoe``.

``granitemoe`` is a *pure sparse* MoE: every layer has an expert bank and there
is no dense ``shared_mlp``, which upstream encodes as
``shared_intermediate_size == 0``.  The switch model must therefore drop the
shared-MLP module entirely — not merely size it to zero — while the frozen
expert bank transfers through untouched.

Tests:
- TestGraniteMoeFamilyEquivalence: sans-LoRA (num_adapters=0) matches upstream.
- TestGraniteMoeNoSharedMLP: the shared MLP and its parameters are truly absent.
- TestZeroAdapterNoHiding / TestZeroAdapterEquivalence: full adapter
  infrastructure with active switching but zero LoRA weights still matches.

Miniaturized configs with real scaling multipliers. See
tests/shared/granite4_equivalence.py for the config registry.
"""

import pytest
import torch
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeForCausalLM

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM
from tests.shared.granite4_equivalence import (
    GRANITEMOE_MINI,
    assert_close,
    augment_cfg_with_adapters,
    get_tolerances,
    get_visible_mask,
    make_active_adapter_input,
    transfer_weights,
    transfer_weights_strict,
    zero_lora_weights,
)

_MODEL_NAMES = sorted(GRANITEMOE_MINI.keys())


def _make_pair(cfg_dict):
    """Create upstream + switch model pair with transferred weights.

    ``cfg_dict`` carries a few keys ``GraniteMoeConfig`` does not define
    (``shared_intermediate_size``, ``layer_types``, ...).  They are inert there:
    ``PretrainedConfig`` keeps unknown kwargs as attributes and
    ``GraniteMoeModel`` does not consult them.
    """
    torch.manual_seed(0)
    upstream = GraniteMoeForCausalLM(GraniteMoeConfig(**cfg_dict)).eval()
    switch = GraniteSwitchForCausalLM(
        GraniteSwitchConfig(**cfg_dict, num_adapters=0)
    ).eval()
    transfer_weights_strict(upstream.state_dict(), switch.state_dict())
    return upstream, switch


@pytest.fixture(params=_MODEL_NAMES)
def model_pair(request):
    """Upstream + switch model pair for each pure-sparse MoE variant."""
    return request.param, *_make_pair(GRANITEMOE_MINI[request.param])


class TestGraniteMoeFamilyEquivalence:
    """GraniteSwitch sans LoRA is bit-exact against upstream granitemoe.

    Bit-exact and not merely close: fused QKV is a concatenation of the same
    three matmuls, and the expert bank is called by the same upstream module.
    Any drift here is a real weight-mapping or forward-path bug, so the
    tolerance is deliberately zero.
    """

    def test_logits_short(self, model_pair):
        name, upstream, switch = model_pair

        torch.manual_seed(42)
        input_ids = torch.randint(0, 256, (1, 16))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        torch.testing.assert_close(
            switch_out.logits,
            upstream_out.logits,
            atol=0.0,
            rtol=0.0,
            msg=f"{name}: short sequence logits should be bit-exact",
        )

    def test_logits_long(self, model_pair):
        name, upstream, switch = model_pair

        torch.manual_seed(123)
        input_ids = torch.randint(0, 256, (1, 64))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        torch.testing.assert_close(
            switch_out.logits,
            upstream_out.logits,
            atol=0.0,
            rtol=0.0,
            msg=f"{name}: long sequence logits should be bit-exact",
        )

    def test_logits_batch(self, model_pair):
        name, upstream, switch = model_pair

        torch.manual_seed(7)
        input_ids = torch.randint(0, 256, (3, 16))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        torch.testing.assert_close(
            switch_out.logits,
            upstream_out.logits,
            atol=0.0,
            rtol=0.0,
            msg=f"{name}: batched logits should be bit-exact",
        )


class TestGraniteMoeNoSharedMLP:
    """The dense shared MLP must be absent, and the expert bank untouched."""

    def test_config_keeps_zero_shared_intermediate(self, model_pair):
        """0 must survive construction.

        ``GraniteSwitchConfig`` defaults ``shared_intermediate_size`` from
        ``intermediate_size`` when it ``is None``.  A falsy test there would
        silently resurrect a phantom shared MLP on every granitemoe base.
        """
        _name, _upstream, switch = model_pair
        assert switch.config.shared_intermediate_size == 0

    def test_layers_have_experts_and_no_shared_mlp(self, model_pair):
        _name, _upstream, switch = model_pair

        for layer in switch.model.layers:
            assert layer.has_experts
            assert not layer.has_shared_mlp
            assert layer.shared_mlp is None
            assert hasattr(layer, "block_sparse_moe")

    def test_no_shared_mlp_or_zero_width_params(self, model_pair):
        """A gated-off shared MLP must leave no ``[0, H]`` tensors behind."""
        _name, _upstream, switch = model_pair

        for name, param in switch.named_parameters():
            assert "shared_mlp" not in name, f"unexpected shared_mlp param {name}"
            assert 0 not in param.shape, (
                f"zero-width param {name}: {tuple(param.shape)}"
            )

    def test_expert_weights_match_upstream(self, model_pair):
        """Expert tensors are named identically, so they transfer by identity."""
        _name, upstream, switch = model_pair
        upstream_sd = upstream.state_dict()

        for i, layer in enumerate(switch.model.layers):
            src = f"model.layers.{i}.block_sparse_moe"
            for attr, key in (
                (layer.block_sparse_moe.input_linear.weight, "input_linear.weight"),
                (layer.block_sparse_moe.output_linear.weight, "output_linear.weight"),
                (layer.block_sparse_moe.router.layer.weight, "router.layer.weight"),
            ):
                torch.testing.assert_close(attr, upstream_sd[f"{src}.{key}"])


# ── Zero-adapter infrastructure tests ──────────────────────────
#
# These exercise the adapter infrastructure with zero LoRA weights. Vanilla
# tokens serve as control tokens so the switch computes non-zero
# adapter_indices and the LoRA forward path runs (with zero weights -> zero
# delta). The upstream model sees the same tokens as plain text.


def _make_zero_adapter_pair(cfg_dict):
    """Create upstream + zero-adapter switch model pair."""
    torch.manual_seed(0)
    upstream = GraniteMoeForCausalLM(GraniteMoeConfig(**cfg_dict)).eval()

    switch_cfg_dict = augment_cfg_with_adapters(cfg_dict)
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

    zero_lora_weights(switch)

    return upstream, switch


class TestZeroAdapterNoHiding:
    """Zero LoRA weights, adapter infrastructure active, no control tokens.

    adapter_indices=0 everywhere and no hiding is triggered, so SingleSwitch is
    bit-exact.
    """

    @pytest.fixture(params=_MODEL_NAMES)
    def model_pair(self, request):
        model_name = request.param
        upstream, switch = _make_zero_adapter_pair(GRANITEMOE_MINI[model_name])
        return model_name, upstream, switch

    def test_no_control_tokens(self, model_pair):
        name, upstream, switch = model_pair

        input_ids = torch.randint(0, 100, (1, 16))

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        torch.testing.assert_close(
            switch_out.logits,
            upstream_out.logits,
            atol=0.0,
            rtol=0.0,
            msg=f"{name}: should be bit-exact with no control tokens",
        )


class TestZeroAdapterEquivalence:
    """Active switching with zero LoRA weights must still match upstream.

    Control positions hold the token-exchange substitute embedding in the switch
    model versus the original control id upstream, so they are excluded via
    get_visible_mask(); visible positions still pick up that delta through
    attention, which is what the tolerance covers.
    """

    @pytest.fixture(params=_MODEL_NAMES)
    def model_pair(self, request):
        model_name = request.param
        upstream, switch = _make_zero_adapter_pair(GRANITEMOE_MINI[model_name])
        return model_name, upstream, switch

    def test_logits_short(self, model_pair):
        name, upstream, switch = model_pair
        layer_types = GRANITEMOE_MINI[name]["layer_types"]

        input_ids = make_active_adapter_input(1, 16, seed=42)

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        visible = get_visible_mask(input_ids)
        atol, rtol = get_tolerances(layer_types, has_kv_hidden=True)
        assert_close(
            switch_out.logits[visible],
            upstream_out.logits[visible],
            atol=atol,
            rtol=rtol,
            msg=f"{name}: short sequence logits diverge (zero-adapter)",
        )

    def test_logits_long(self, model_pair):
        name, upstream, switch = model_pair
        layer_types = GRANITEMOE_MINI[name]["layer_types"]

        input_ids = make_active_adapter_input(1, 64, seed=123)

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        visible = get_visible_mask(input_ids)
        atol, rtol = get_tolerances(layer_types, long_sequence=True, has_kv_hidden=True)
        assert_close(
            switch_out.logits[visible],
            upstream_out.logits[visible],
            atol=atol,
            rtol=rtol,
            msg=f"{name}: long sequence logits diverge (zero-adapter)",
        )

    def test_logits_batch(self, model_pair):
        name, upstream, switch = model_pair
        layer_types = GRANITEMOE_MINI[name]["layer_types"]

        input_ids = make_active_adapter_input(3, 16, seed=7)

        with torch.no_grad():
            upstream_out = upstream(input_ids=input_ids, use_cache=False)
            switch_out = switch(input_ids=input_ids, use_cache=False)

        visible = get_visible_mask(input_ids)
        atol, rtol = get_tolerances(layer_types, has_kv_hidden=True)
        assert_close(
            switch_out.logits[visible],
            upstream_out.logits[visible],
            atol=atol,
            rtol=rtol,
            msg=f"{name}: batched logits diverge (zero-adapter)",
        )
