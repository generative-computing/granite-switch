# SPDX-License-Identifier: Apache-2.0
"""Adapter switching implementation for Granite Switch (HuggingFace).

The switch engine is the Kerdock/DG coded-memory ``MultiSwitch`` -- the only
engine. ``create_switch`` builds it.
"""

from .multi import MultiSwitch

__all__ = [
    "MultiSwitch",
    "create_switch",
]


def create_switch(config, layer_idx=0):
    """Build the switch (the coded-memory ``MultiSwitch``).

    Args:
        config: GraniteSwitchConfig
        layer_idx: Layer index for cache management (default: 0)

    Returns:
        A ``MultiSwitch`` module.
    """
    return MultiSwitch(
        num_adapters=config.num_adapters,
        config=config,
        control_token_gain=getattr(config, "control_token_gain", 15.0),
        switch_head_dim=config.switch_head_dim,
        layer_idx=layer_idx,
    )
