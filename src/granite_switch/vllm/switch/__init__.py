# SPDX-License-Identifier: Apache-2.0
"""Adapter switching implementation for Granite Switch (vLLM).

The switch engine is the Kerdock/DG coded-memory ``MultiSwitch`` -- the only
engine. ``create_switch`` builds it.
"""

from .multi import MultiSwitch

__all__ = [
    "MultiSwitch",
    "create_switch",
]


def create_switch(config, vllm_config=None):
    """Build the switch (the coded-memory ``MultiSwitch``).

    Args:
        config: GraniteSwitchConfig
        vllm_config: vLLM configuration (for vLLM implementation)

    Returns:
        A ``MultiSwitch`` module.
    """
    return MultiSwitch(
        num_adapters=config.num_adapters,
        vllm_config=vllm_config,
        control_token_gain=getattr(config, "control_token_gain", 15.0),
        switch_head_dim=config.switch_head_dim,
        config=config,
    )
