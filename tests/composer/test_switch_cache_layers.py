# SPDX-License-Identifier: Apache-2.0
"""Composer layer-inflation guard for the switch-cache-slot count.

The composer inflates ``num_hidden_layers`` by ``_switch_cache_layers(switch_type)``
so that, after ``modeling_granite_switch.py`` subtracts ``switch.num_cache_layers``,
all base decoder layers are retained. This must stay in sync with each switch
class's ``num_cache_layers`` property:

  - "single" -> SingleSwitch.num_cache_layers     == 1
  - "multi"  -> MultiSwitch.num_cache_layers == 2

The regression risk is subtle: the coded engine's move to 2 slots changed the
inflation from a hardcoded ``+1`` to a slot-count lookup. These tests pin the
mapping AND assert it matches the switch classes' actual ``num_cache_layers`` so
the single-switch path (``+1``, unchanged behavior) cannot silently drift.

CPU-only: reads a pure mapping function and constructs the switches over a tiny
mock config; no model download.
"""

import pytest
import torch

from granite_switch.composer.compose_utils import _switch_cache_layers
from granite_switch.hf.switch import create_switch


class _MockSwitchConfig:
    """Minimal GraniteSwitchConfig-shaped object for create_switch dispatch."""

    def __init__(self, switch_type):
        self.switch_type = switch_type
        self.num_adapters = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.projection_head_dim = 64
        self.attention_multiplier = 0.125
        self.vocab_size = 2000
        self.hidden_size = 256
        self.adapter_token_ids = [101, 102]
        self.adapter_substitute_token_ids = [1, 2]
        self.switch_head_dim = 32
        self.control_token_gain = 15.0
        # coded-engine params (ignored by the single switch)
        self.ms_code_m = 6
        self.ms_code_type = "kerdock"
        self.ms_memory_gain = 28.0
        self.ms_counting_head_dim = 32
        self._pre_quantization_dtype = torch.float32


class TestSwitchCacheLayers:
    """The inflation lookup and its agreement with the switch classes."""

    @pytest.mark.parametrize(
        "switch_type,expected",
        [("single", 1), ("multi", 2)],
    )
    def test_mapping(self, switch_type, expected):
        assert _switch_cache_layers(switch_type) == expected

    @pytest.mark.parametrize("switch_type", ["single", "multi"])
    def test_lookup_matches_switch_class(self, switch_type):
        """The inflation count MUST equal the built switch's num_cache_layers —
        otherwise the composed model keeps the wrong number of decoder layers."""
        switch = create_switch(_MockSwitchConfig(switch_type), layer_idx=0)
        assert _switch_cache_layers(switch_type) == switch.num_cache_layers

    def test_unknown_switch_type_raises(self):
        with pytest.raises(KeyError):
            _switch_cache_layers("bogus")
