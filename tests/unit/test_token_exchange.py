# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the token-exchange config path.

Covers the new fields and validators on GraniteSwitchConfig:
- adapter_substitute_token_ids length check
- use_token_exchange property
- rejection of num_adapters>0 with no hiding and no substitute ids
- rejection of duplicate adapter_token_ids (LUT would collide)
- default control_dims flipped to 0
"""

import pytest

from granite_switch.config import GraniteSwitchConfig


def _base(num_adapters=2, **overrides):
    names = [f"a{i}" for i in range(num_adapters)]
    base = dict(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=num_adapters,
        adapter_token_ids=list(range(500, 500 + num_adapters)),
        adapter_names=names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
    )
    base.update(overrides)
    return base


class TestDefaults:
    def test_control_dims_default_is_zero(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.control_dims == 0

    def test_no_adapters_no_validation(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.use_token_exchange is False


class TestUseTokenExchangeProperty:
    def test_true_when_substitute_and_zero_dims(self):
        cfg = GraniteSwitchConfig(
            **_base(
                control_dims=0,
                adapter_substitute_token_ids=[1, 2],
            )
        )
        assert cfg.use_token_exchange is True

    def test_false_when_legacy_hiding(self):
        cfg = GraniteSwitchConfig(**_base(control_dims=32))
        assert cfg.use_token_exchange is False

    def test_false_when_no_substitute_ids_even_with_zero_dims_requires_validator(self):
        # This combo is invalid — validator rejects it, so the property
        # cannot be observed in a built config. Covered in TestValidation.
        pass


class TestValidation:
    def test_zero_dims_plus_missing_substitute_ids_raises(self):
        with pytest.raises(ValueError, match="either control_dims > 0"):
            GraniteSwitchConfig(**_base(control_dims=0))

    def test_substitute_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_substitute_token_ids length"):
            GraniteSwitchConfig(
                **_base(control_dims=0, adapter_substitute_token_ids=[1])
            )

    def test_duplicate_adapter_token_ids_raises(self):
        with pytest.raises(ValueError, match="adapter_token_ids must be unique"):
            GraniteSwitchConfig(
                **_base(
                    adapter_token_ids=[100, 100],
                    adapter_substitute_token_ids=[1, 2],
                    control_dims=0,
                )
            )


class TestLegacyPathStillWorks:
    def test_control_dims_positive_without_substitute_ids(self):
        cfg = GraniteSwitchConfig(**_base(control_dims=32))
        assert cfg.control_dims == 32
        assert cfg.use_token_exchange is False
        # Expanded head_dim reflects the legacy path.
        assert cfg.expanded_head_dim == cfg.projection_head_dim + 32


class TestExpandedHeadDim:
    def test_token_exchange_gives_native_head_dim(self):
        cfg = GraniteSwitchConfig(
            **_base(control_dims=0, adapter_substitute_token_ids=[1, 2])
        )
        assert cfg.expanded_head_dim == cfg.projection_head_dim
