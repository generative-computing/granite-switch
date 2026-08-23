# SPDX-License-Identifier: Apache-2.0
"""Adapter switching implementations for Granite Switch (vLLM).

Provides the adapter-selection engines, dispatched by ``config.switch_type``:

- ``"single"`` -> ``SingleSwitch``  (replicated one-hot attention; default)
- ``"multi"``  -> ``MultiSwitch``   (Kerdock/DG coded-memory routing)
"""

from .multi import MultiSwitch
from .single import SingleSwitch

__all__ = [
    "MultiSwitch",
    "SingleSwitch",
    "create_switch",
]


def create_switch(config, vllm_config=None):
    """Factory function to create the switch selected by ``config.switch_type``.

    Args:
        config: GraniteSwitchConfig
        vllm_config: vLLM configuration (for vLLM implementation)

    Returns:
        A switch module (SingleSwitch or MultiSwitch).
    """
    switch_type = getattr(config, "switch_type", "single")
    common = dict(
        num_adapters=config.num_adapters,
        vllm_config=vllm_config,
        control_token_gain=getattr(config, "control_token_gain", 15.0),
        switch_head_dim=config.switch_head_dim,
        config=config,
    )
    # "multi" selects the coded-memory engine.
    if switch_type == "multi":
        return MultiSwitch(**common)
    return SingleSwitch(**common)
