# SPDX-License-Identifier: Apache-2.0
"""Composer layer-inflation guard for the switch-cache-slot count.

The composer inflates ``num_hidden_layers`` by ``SWITCH_CACHE_LAYERS`` so that,
after ``modeling_granite_switch.py`` subtracts ``switch.num_cache_layers``, all
base decoder layers are retained. MultiSwitch is the only engine, so this is a
fixed constant now (the coded engine owns 2 slots: counting + memory). This test
pins that the composer's constant still equals the switch's actual
``num_cache_layers`` — if the two drift, the composed model keeps the wrong
number of decoder layers.

CPU-only: constructs the switch over a tiny mock config; no model download.
"""

import torch

from granite_switch.composer.compose_utils import SWITCH_CACHE_LAYERS
from granite_switch.hf.switch import create_switch


class _MockSwitchConfig:
    """Minimal GraniteSwitchConfig-shaped object for create_switch."""

    def __init__(self):
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
        # coded-engine params
        self.ms_code_m = 6
        self.ms_code_type = "kerdock"
        self.ms_memory_gain = 28.0
        self.ms_counting_head_dim = 32
        self._pre_quantization_dtype = torch.float32


class TestSwitchCacheLayers:
    """The inflation constant and its agreement with the switch class."""

    def test_constant_is_two(self):
        assert SWITCH_CACHE_LAYERS == 2

    def test_constant_matches_switch_class(self):
        """The inflation count MUST equal the built switch's num_cache_layers —
        otherwise the composed model keeps the wrong number of decoder layers."""
        switch = create_switch(_MockSwitchConfig(), layer_idx=0)
        assert SWITCH_CACHE_LAYERS == switch.num_cache_layers
