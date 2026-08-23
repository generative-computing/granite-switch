# SPDX-License-Identifier: Apache-2.0
"""Adapter switching implementations for Granite Switch (HuggingFace).

Provides the adapter-selection engines, dispatched by ``config.switch_type``:

- ``"single"`` -> ``SingleSwitch``  (attention ±gain cumsum; default)
- ``"multi"``  -> ``MultiSwitch``   (Kerdock/DG coded-memory routing)
"""

from .multi import MultiSwitch
from .single import SingleSwitch

__all__ = [
    "MultiSwitch",
    "SingleSwitch",
    "create_switch",
]


def create_switch(config, layer_idx=0):
    """Factory function to create the switch selected by ``config.switch_type``.

    Args:
        config: GraniteSwitchConfig
        layer_idx: Layer index for cache management (default: 0)

    Returns:
        A switch module (SingleSwitch or MultiSwitch).
    """
    switch_type = getattr(config, "switch_type", "single")
    common = dict(
        num_adapters=config.num_adapters,
        config=config,
        control_token_gain=getattr(config, "control_token_gain", 15.0),
        switch_head_dim=config.switch_head_dim,
        layer_idx=layer_idx,
    )
    # "multi" selects the coded-memory engine.
    if switch_type == "multi":
        return MultiSwitch(**common)
    return SingleSwitch(**common)
