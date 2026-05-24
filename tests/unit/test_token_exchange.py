# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the token-exchange config path.

Verifies the validators and required-field semantics on
GraniteSwitchConfig, now that token-exchange is the only mode.
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
        adapter_substitute_token_ids=[1] * num_adapters,
        adapter_names=names,
        max_lora_rank=8,
        adapter_ranks=[8] * num_adapters,
    )
    base.update(overrides)
    return base


class TestDefaults:
    def test_no_adapters_no_validation(self):
        cfg = GraniteSwitchConfig(num_adapters=0)
        assert cfg.adapter_substitute_token_ids is None


class TestValidation:
    def test_substitute_ids_required_when_adapters_present(self):
        with pytest.raises(ValueError, match="adapter_substitute_token_ids is required"):
            GraniteSwitchConfig(**_base(adapter_substitute_token_ids=None))

    def test_substitute_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapter_substitute_token_ids length"):
            GraniteSwitchConfig(**_base(adapter_substitute_token_ids=[1]))

    def test_duplicate_adapter_token_ids_raises(self):
        with pytest.raises(ValueError, match="adapter_token_ids must be unique"):
            GraniteSwitchConfig(**_base(adapter_token_ids=[100, 100]))

    def test_negative_substitute_id_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            GraniteSwitchConfig(**_base(adapter_substitute_token_ids=[-1, 1]))


class TestProjectionHeadDim:
    def test_inferred_from_hidden_size(self):
        cfg = GraniteSwitchConfig(**_base())
        assert cfg.projection_head_dim == cfg.hidden_size // cfg.num_attention_heads
