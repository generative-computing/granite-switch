# SPDX-License-Identifier: Apache-2.0
"""Granite Switch: HuggingFace backend for adapter switching."""

from granite_switch.config import GraniteSwitchConfig

from .modeling_granite_switch import GraniteSwitchForCausalLM
from .switch.single import SingleSwitch

__all__ = [
    "GraniteSwitchConfig",
    "GraniteSwitchForCausalLM",
    "SingleSwitch",
]

# Register with transformers AutoConfig and AutoModel
try:
    from transformers import AutoConfig, AutoModelForCausalLM

    AutoConfig.register("granite_switch", GraniteSwitchConfig)
    AutoModelForCausalLM.register(GraniteSwitchConfig, GraniteSwitchForCausalLM)
except Exception:
    # Registration may fail if already registered or transformers not available
    pass
