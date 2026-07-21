# SPDX-License-Identifier: Apache-2.0
"""Core LoRA and decoder layer implementations for Granite Switch.

This package provides the foundational building blocks:
- lora: Low-level LoRA primitives using vLLM's Punica kernels
- lora_kernel_meta: Torch.compile-friendly LoRA kernel metadata
- decoder: High-level decoder layers (attention, MLP, full decoder layer)
"""

from .decoder import (
    GraniteLoRAEmbeddedAttention,
    GraniteSwitchDecoderLayer,
)
from .lora import SwitchedLoRALinear
from .lora_kernel_meta import CompileFriendlyLoRAKernelMeta, LoRAContext

__all__ = [
    "CompileFriendlyLoRAKernelMeta",
    "GraniteLoRAEmbeddedAttention",
    "GraniteSwitchDecoderLayer",
    "LoRAContext",
    "SwitchedLoRALinear",
]
