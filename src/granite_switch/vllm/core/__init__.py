# SPDX-License-Identifier: Apache-2.0
"""Shared switch-kernel substrate for Granite Switch (both adaptations).

Foundational building blocks used by BOTH the LoRA and Shadow-Residual decoder
tiers — not LoRA-specific despite the historical ``lora`` naming:
- lora: Fused switch-LoRA linear layer (SwitchedLoRALinear)
- lora_ops: torch.compile-opaque custom ops wrapping the Triton kernel
- lora_kernel_meta: Bitmask + remap metadata for the fused kernel
"""

from .lora import SwitchedLoRALinear
from .lora_kernel_meta import FusedLoRAKernelMeta, LoRAContext

__all__ = [
    "FusedLoRAKernelMeta",
    "LoRAContext",
    "SwitchedLoRALinear",
]
