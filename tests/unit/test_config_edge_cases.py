# SPDX-License-Identifier: Apache-2.0
"""Additional config edge case tests for GraniteSwitchConfig."""

import pytest

from granite_switch.config import GraniteSwitchConfig

pytestmark = pytest.mark.local_fast


def _valid_kwargs(num_adapters=2, **overrides):
    """Return kwargs for a valid token-exchange config."""
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
        adapter_substitute_token_ids=[1] * num_adapters,
        adapter_names=adapter_names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
    )
    base.update(overrides)
    return base


class TestSharedIntermediateSize:
    """The parent GraniteMoeHybridConfig may have a non-None default for
    shared_intermediate_size. Verify our config has a sensible value."""

    def test_shared_intermediate_size_has_value(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert cfg.shared_intermediate_size is not None
        assert cfg.shared_intermediate_size > 0

    def test_explicit_shared_intermediate_size_preserved(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs(
            shared_intermediate_size=256,
        ))
        assert cfg.shared_intermediate_size == 256


class TestLayerTypesDefault:
    """layer_types defaults to all-attention with length == num_hidden_layers."""

    def test_default_layer_types_when_omitted(self):
        cfg = GraniteSwitchConfig(num_adapters=0, num_hidden_layers=4)
        assert cfg.layer_types == ["attention"] * 4

    def test_explicit_layer_types_preserved(self):
        cfg = GraniteSwitchConfig(
            num_adapters=0,
            num_hidden_layers=3,
            layer_types=["attention", "attention", "attention"],
        )
        assert cfg.layer_types == ["attention", "attention", "attention"]


class TestLoraTargetModulesDefault:
    """lora_target_modules defaults to qkv_proj/o_proj + shared_mlp pair
    when num_adapters > 0; empty when num_adapters == 0."""

    def test_no_adapters_empty_target_modules(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.lora_target_modules == []

    def test_adapters_populate_target_modules(self):
        cfg = GraniteSwitchConfig(**_valid_kwargs())
        assert "qkv_proj" in cfg.lora_target_modules
        assert "o_proj" in cfg.lora_target_modules
        assert "shared_input_linear" in cfg.lora_target_modules
        assert "shared_output_linear" in cfg.lora_target_modules

    def test_explicit_target_modules_preserved(self):
        cfg = GraniteSwitchConfig(
            **_valid_kwargs(lora_target_modules=["qkv_proj"])
        )
        assert cfg.lora_target_modules == ["qkv_proj"]
