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


def load_model(path: str):
    """Load a GraniteSwitch model from a saved checkpoint.

    Args:
        path: Path to saved model directory (with config.json).

    Returns:
        GraniteSwitchForCausalLM instance.
    """
    from pathlib import Path

    config_file = Path(path) / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No config.json found at {path}")

    config = GraniteSwitchConfig.from_pretrained(path)
    return GraniteSwitchForCausalLM.from_pretrained(path, config=config)
