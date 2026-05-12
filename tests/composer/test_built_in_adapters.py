# SPDX-License-Identifier: Apache-2.0
"""Tests for built-in adapter support (Mode A native / Mode B mixed).

Tests config-level behavior:
- Mode A: control_dims=0, no hiding
- Mode B: mixed built-in + external → control_dims>0, third_party = external only
- SSM rejection for mixed mode
- Model construction with control_dims=0
"""

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mode_a_config():
    """Mode A (native): built-in adapters only, control_dims=0, token-exchange."""
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 1 switch + 2 decoder
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        # Built-in adapters: substitute = BOS (arbitrary id 1 for tests).
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["router", "planner"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=16,
        control_dims=0,
        # No hiding groups
        hiding_groups=None,
        hiding_policy=None,
        adapter_third_party=None,
    )


@pytest.fixture
def mode_b_config():
    """Mode B (mixed): 1 external + 1 built-in adapter, control_dims=8."""
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 1 switch + 2 decoder
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_names=["external_rag", "router"],
        hiding_groups={"all_controls": ["external_rag", "router"]},
        hiding_policy={
            "base": ["all_controls"],
            "external_rag": ["all_controls"],
            "router": ["all_controls"],
        },
        adapter_third_party=["external_rag"],  # Only external is third-party
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=16,
        control_dims=8,
    )


# ── Mode A Config Tests ──────────────────────────────────────────────


class TestModeAConfig:
    """Config-level checks for Mode A (native, control_dims=0)."""

    def test_control_dims_zero_allowed(self, mode_a_config):
        """control_dims=0 should be accepted by config validation."""
        assert mode_a_config.control_dims == 0

    def test_no_hiding_groups(self, mode_a_config):
        """Mode A has no hiding groups."""
        assert mode_a_config.num_hiding_groups == 0
        assert mode_a_config.hiding_group_names == []
        assert mode_a_config.get_hiding_group_token_ids() == {}

    def test_no_third_party(self, mode_a_config):
        """Mode A has no third-party adapters."""
        assert mode_a_config.adapter_third_party is None
        mask = mode_a_config.get_third_party_adapter_mask()
        assert all(v is False for v in mask)

    def test_adapters_present(self, mode_a_config):
        """Mode A still has adapters with LoRA."""
        assert mode_a_config.num_adapters == 2
        assert mode_a_config.adapter_ranks == [4, 4]


# ── Mode A Model Tests ───────────────────────────────────────────────


class TestModeAModel:
    """Model construction and forward pass with control_dims=0."""

    def test_model_creates_successfully(self, mode_a_config):
        """GraniteSwitchForCausalLM should construct with control_dims=0."""
        model = GraniteSwitchForCausalLM(mode_a_config)
        assert model is not None
        assert model.config.control_dims == 0

    def test_attention_no_expansion(self, mode_a_config):
        """Decoder attention layers should NOT expand control dims."""
        model = GraniteSwitchForCausalLM(mode_a_config)
        for layer in model.model.layers:
            attn = layer.self_attn
            assert not attn.expand_control_dims, (
                "expand_control_dims should be False when control_dims=0"
            )
            assert attn.expanded_head_dim == attn.head_dim, (
                "expanded_head_dim should equal head_dim when control_dims=0"
            )

    def test_forward_pass(self, mode_a_config):
        """Forward pass should work with control_dims=0."""
        model = GraniteSwitchForCausalLM(mode_a_config).eval()
        model.model.adapter_token_ids.data = torch.tensor(
            mode_a_config.adapter_token_ids, dtype=torch.long
        )

        input_ids = torch.tensor([[10, 250, 20, 30, 40]])
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 5, mode_a_config.vocab_size)

    def test_no_hiding_buffers(self, mode_a_config):
        """Model should have no hiding-related buffers when control_dims=0."""
        model = GraniteSwitchForCausalLM(mode_a_config)
        assert model.model.token_to_group_mask is None
        assert model.model.adapter_hiding_matrix is None

    def test_lora_shapes_correct(self, mode_a_config):
        """LoRA weight shapes should reflect num_adapters."""
        model = GraniteSwitchForCausalLM(mode_a_config)
        layer = model.model.layers[0]  # First decoder layer
        attn = layer.self_attn
        # QKV has LoRA with 2 adapters, rank 4
        if hasattr(attn.qkv_proj, "lora_A_slices"):
            for lora_a in attn.qkv_proj.lora_A_slices:
                assert lora_a.shape[0] == 2, "num_adapters should be 2"
                assert lora_a.shape[2] == 4, "max_lora_rank should be 4"

    def test_adapter_routing_works(self, mode_a_config):
        """Adapter routing should still work with control_dims=0."""
        model = GraniteSwitchForCausalLM(mode_a_config).eval()
        model.model.adapter_token_ids.data = torch.tensor(
            mode_a_config.adapter_token_ids, dtype=torch.long
        )

        # Set non-zero lora_B to make adapter effect visible
        with torch.no_grad():
            for layer in model.model.layers:
                if hasattr(layer.self_attn.o_proj, "lora_B"):
                    layer.self_attn.o_proj.lora_B.data = (
                        torch.randn_like(layer.self_attn.o_proj.lora_B) * 0.1
                    )

        # All base tokens
        base_ids = torch.tensor([[10, 20, 30, 40, 50]])
        # With adapter control token
        adapter_ids = torch.tensor([[250, 20, 30, 40, 50]])

        with torch.no_grad():
            out_base = model(input_ids=base_ids)
            out_adapter = model(input_ids=adapter_ids)

        # Logits should differ when adapter is active
        # (tokens after control token see different LoRA)
        diff = (out_base.logits[0, -1] - out_adapter.logits[0, -1]).abs().max()
        assert diff > 1e-6, "Adapter should produce different logits than base"

    def test_control_token_logits_finite(self, mode_a_config):
        """Control token logits should be finite."""
        model = GraniteSwitchForCausalLM(mode_a_config).eval()
        model.model.adapter_token_ids.data = torch.tensor(
            mode_a_config.adapter_token_ids, dtype=torch.long
        )

        input_ids = torch.tensor([[250, 20, 30]])
        with torch.no_grad():
            output = model(input_ids=input_ids)

        control_token_logits = output.logits[:, :, mode_a_config.adapter_token_ids]
        assert torch.isfinite(control_token_logits).all(), (
            "Control token logits should be finite"
        )


# ── Mode B Config Tests ──────────────────────────────────────────────


class TestModeBConfig:
    """Config-level checks for Mode B (mixed, control_dims>0)."""

    def test_control_dims_positive(self, mode_b_config):
        """Mode B should have control_dims > 0."""
        assert mode_b_config.control_dims == 8

    def test_only_external_is_third_party(self, mode_b_config):
        """Only external adapter should be third-party."""
        assert mode_b_config.adapter_third_party == ["external_rag"]
        mask = mode_b_config.get_third_party_adapter_mask()
        # [base=False, external_rag=True, router=False]
        assert mask == [False, True, False]

    def test_hiding_groups_present(self, mode_b_config):
        """Mode B should have hiding groups."""
        assert mode_b_config.num_hiding_groups == 1

    def test_third_party_mask(self, mode_b_config):
        """Third-party mask marks only external adapter."""
        mask = mode_b_config.get_third_party_adapter_mask()
        assert mask == [False, True, False]


# ── Negative Tests ────────────────────────────────────────────────────


class TestNegative:
    """Validation errors that should be raised."""

    def test_control_dims_negative_rejected(self):
        """control_dims < 0 should still be rejected."""
        with pytest.raises(ValueError, match="control_dims must be >= 0"):
            GraniteSwitchConfig(
                vocab_size=256,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                num_adapters=0,
                control_dims=-1,
            )

    def test_hiding_groups_require_control_dims(self):
        """Hiding groups require control_dims >= num_hiding_groups.

        A build with 1 hiding group and control_dims=1 works; with 2 groups and
        control_dims=1 it must fail. Substitute ids are supplied only to get
        past the newer "no-hiding-and-no-exchange" validator; the assertion is
        specifically about the hiding-vs-control_dims arithmetic.
        """
        with pytest.raises(ValueError, match="control_dims.*must be >= number of hiding groups"):
            GraniteSwitchConfig(
                vocab_size=300,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=4,
                num_adapters=2,
                adapter_token_ids=[250, 251],
                adapter_substitute_token_ids=[1, 2],
                adapter_names=["a", "b"],
                hiding_groups={"g1": ["a"], "g2": ["b"]},  # 2 groups > control_dims=1
                max_lora_rank=4,
                adapter_ranks=[4, 4],
                control_dims=1,
            )
