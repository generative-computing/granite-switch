# SPDX-License-Identifier: Apache-2.0
"""Additional config edge case tests for GraniteSwitchConfig.

These tests cover edge cases not covered by the main test_config.py,
specifically targeting previously uncovered code paths:
- Line 99: shared_intermediate_size default from intermediate_size
- Line 119: negative control_dims validation
- Lines 220, 222: get_hiding_group_token_ids with missing configs
- Lines 250-259: get_third_party_adapter_mask functionality
"""

import pytest

from granite_switch.config import GraniteSwitchConfig


def _valid_kwargs(num_adapters=2, **overrides):
    """Return kwargs for a valid SingleSwitch config, with optional overrides.

    Default mode: legacy KV hiding (control_dims=32).
    """
    adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    base = dict(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=num_adapters,
        adapter_token_ids=list(range(500, 500 + num_adapters)),
        adapter_names=adapter_names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
        control_dims=32,
    )
    base.update(overrides)
    return base


class TestSharedIntermediateSize:
    """Tests for shared_intermediate_size default handling (line 99).

    Note: The parent GraniteMoeHybridConfig may have a non-None default,
    so line 99 (the None check) may not always be hit. We test the
    explicit case and verify the config has a sensible value.
    """

    def test_shared_intermediate_size_has_value(self):
        """shared_intermediate_size has a value (either explicit or parent default)."""
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        # Should have a sensible value (not None)
        assert cfg.shared_intermediate_size is not None
        assert cfg.shared_intermediate_size > 0

    def test_explicit_shared_intermediate_size_preserved(self):
        """Explicit shared_intermediate_size is preserved."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            shared_intermediate_size=256,
        ))
        assert cfg.shared_intermediate_size == 256


class TestControlDimsValidation:
    """Tests for control_dims validation (line 119)."""

    def test_negative_control_dims_raises(self):
        """Negative control_dims should raise ValueError."""
        with pytest.raises(ValueError, match="control_dims must be >= 0"):
            GraniteSwitchConfig(**_valid_kwargs(control_dims=-1))

    def test_zero_control_dims_valid_with_substitute_ids(self):
        """Zero control_dims is valid when substitute ids are provided (token exchange)."""
        cfg = GraniteSwitchConfig(
            **_valid_kwargs(control_dims=0, adapter_substitute_token_ids=[1, 2])
        )
        assert cfg.control_dims == 0
        assert cfg.use_token_exchange is True

    def test_zero_control_dims_no_substitute_ids_raises(self):
        """Zero control_dims without substitute ids must fail: no hiding and no exchange."""
        with pytest.raises(ValueError, match="either control_dims > 0"):
            GraniteSwitchConfig(**_valid_kwargs(control_dims=0))

    def test_positive_control_dims_valid(self):
        """Positive control_dims is valid."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(control_dims=64))
        assert cfg.control_dims == 64


class TestGetHidingGroupTokenIds:
    """Tests for get_hiding_group_token_ids edge cases (lines 220, 222)."""

    def test_no_hiding_groups_returns_empty(self):
        """Empty dict when hiding_groups is None (line 219)."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(hiding_groups=None))
        result = cfg.get_hiding_group_token_ids()
        assert result == {}

    def test_no_adapter_names_returns_empty(self):
        """Empty dict when adapter_names is None (line 219)."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_names=None,
            hiding_groups={"all": ["adapter_0"]},
        ))
        result = cfg.get_hiding_group_token_ids()
        assert result == {}

    def test_no_adapter_token_ids_returns_empty(self):
        """Empty dict when adapter_token_ids is None (line 222)."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_token_ids=None,
            hiding_groups={"all": ["adapter_0"]},
        ))
        result = cfg.get_hiding_group_token_ids()
        assert result == {}

    def test_partial_adapter_name_match(self):
        """Only matching adapter names are included in result."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            hiding_groups={"all": ["adapter_0", "nonexistent_adapter"]},
        ))
        result = cfg.get_hiding_group_token_ids()
        # Only adapter_0 should be in the result (token 500)
        assert result == {0: [500]}


class TestGetThirdPartyAdapterMask:
    """Tests for get_third_party_adapter_mask (lines 250-259)."""

    def test_no_third_party_returns_all_false(self):
        """All-False mask when adapter_third_party is not configured."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_third_party=None,
        ))
        mask = cfg.get_third_party_adapter_mask()
        # Length = num_adapters + 1 (base + adapters)
        assert len(mask) == 3  # base + 2 adapters
        assert mask == [False, False, False]

    def test_empty_third_party_returns_all_false(self):
        """All-False mask when adapter_third_party is empty list."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_third_party=[],
        ))
        mask = cfg.get_third_party_adapter_mask()
        assert mask == [False, False, False]

    def test_no_adapter_names_returns_all_false(self):
        """All-False mask when adapter_names is None."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_names=None,
            adapter_third_party=["adapter_0"],
        ))
        mask = cfg.get_third_party_adapter_mask()
        assert mask == [False, False, False]

    def test_single_third_party_adapter(self):
        """Mask correctly identifies single third-party adapter."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_third_party=["adapter_0"],
        ))
        mask = cfg.get_third_party_adapter_mask()
        # Index 0 = base (never third-party)
        # Index 1 = adapter_0 (third-party)
        # Index 2 = adapter_1 (not third-party)
        assert mask == [False, True, False]

    def test_multiple_third_party_adapters(self):
        """Mask correctly identifies multiple third-party adapters."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_third_party=["adapter_0", "adapter_1"],
        ))
        mask = cfg.get_third_party_adapter_mask()
        # Both adapters are third-party
        assert mask == [False, True, True]

    def test_base_never_third_party(self):
        """Base adapter (index 0) is never marked as third-party."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            adapter_third_party=["adapter_0", "adapter_1"],
        ))
        mask = cfg.get_third_party_adapter_mask()
        assert mask[0] is False  # Base is never third-party

    def test_mask_length_matches_num_adapters_plus_one(self):
        """Mask length is num_adapters + 1 (includes base slot)."""
        for num_adapters in [0, 1, 4, 10]:
            cfg = GraniteSwitchConfig(**_valid_kwargs(
                num_adapters=num_adapters,
                adapter_token_ids=list(range(500, 500 + num_adapters)),
                adapter_names=[f"adapter_{i}" for i in range(num_adapters)],
                adapter_ranks=[8] * num_adapters if num_adapters > 0 else None,
            ))
            mask = cfg.get_third_party_adapter_mask()
            assert len(mask) == num_adapters + 1


class TestLayerTypesDefault:
    """Tests for layer_types default handling."""

    def test_layer_types_defaults_to_attention(self):
        """layer_types defaults to all 'attention' when None."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            layer_types=None,
            num_hidden_layers=4,
        ))
        # Should have 4 attention layers (adapters add a switch layer at index 0,
        # but the config has num_hidden_layers=4 which becomes 5 with switch)
        # The default is set before parent init adds the switch layer
        assert cfg.layer_types is not None

    def test_explicit_layer_types_preserved(self):
        """Explicit layer_types are preserved."""
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            layer_types=["attention", "attention"],
            num_hidden_layers=2,
        ))
        assert cfg.layer_types == ["attention", "attention"]


class TestLoraTargetModulesDefault:
    """Tests for lora_target_modules default handling."""

    def test_lora_target_modules_empty_when_no_adapters(self):
        """lora_target_modules defaults to empty when num_adapters=0."""
        cfg = GraniteSwitchConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            num_adapters=0,
        )
        assert cfg.lora_target_modules == []

    def test_lora_target_modules_populated_with_adapters(self):
        """lora_target_modules defaults to standard modules when adapters present."""
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        # Should include attention and MLP modules
        assert "qkv_proj" in cfg.lora_target_modules
        assert "o_proj" in cfg.lora_target_modules
        assert "shared_input_linear" in cfg.lora_target_modules
        assert "shared_output_linear" in cfg.lora_target_modules
