# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the token-exchange config path.

Verifies the validators and required-field semantics on
GraniteSwitchConfig, now that token-exchange is the only mode.
"""

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.token_exchange import (
    build_adapter_kind_lut,
    build_classifier_control_luts,
    split_adapter_indices,
)


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
        with pytest.raises(
            ValueError, match="adapter_substitute_token_ids is required"
        ):
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


class TestClassifierLoRACoactivation:
    def _config(self):
        return GraniteSwitchConfig(
            **_base(
                adapter_token_ids=[499, 500, 501],  # leading base-reset token
                adapter_substitute_token_ids=[1, 1, 1],
                adapter_kinds=["lora", "classifier"],
                classifier_label_token_ids=[None, [100, 101]],
            )
        )

    def test_classifier_probe_lut_skips_leading_base_reset_token(self):
        cfg = self._config()
        marker_lut, slot_lut = build_classifier_control_luts(cfg)

        assert not marker_lut[499]
        assert not marker_lut[500]
        assert marker_lut[501]
        assert slot_lut[499] == slot_lut[500] == 0
        assert slot_lut[501] == 2
        assert cfg.classifier_control_token_ids == [501]

    def test_classifier_probe_preserves_lora_indices_and_separates_probe(self):
        cfg = self._config()
        adapter_indices = torch.tensor([0, 1, 1, 1])
        probe_indices = torch.tensor([0, 0, 0, 2])

        lora_indices, classifier_indices = split_adapter_indices(
            build_adapter_kind_lut(cfg),
            adapter_indices,
            probe_indices,
        )

        assert lora_indices.tolist() == [0, 1, 1, 1]
        assert classifier_indices.tolist() == [0, 0, 0, 2]
