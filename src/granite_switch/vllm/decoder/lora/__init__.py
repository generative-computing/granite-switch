# SPDX-License-Identifier: Apache-2.0
"""LoRA/aLoRA decoder tier for the shared Granite Switch vLLM model.

Single-stream ``[M, H]`` decoder: attention + MLP with conditional
:class:`~granite_switch.vllm.core.lora.SwitchedLoRALinear` on the projections,
selected per-token by the switch. This is the host default adaptation.
"""

from .decoder import (
    GraniteLoRAEmbeddedAttention,
    GraniteSwitchDecoderLayer,
    rms_norm_select,
)

__all__ = [
    "GraniteLoRAEmbeddedAttention",
    "GraniteSwitchDecoderLayer",
    "rms_norm_select",
]
