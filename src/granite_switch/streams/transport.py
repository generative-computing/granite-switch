# SPDX-License-Identifier: Apache-2.0
"""Exact slot IDs carried alongside vLLM hidden states, including in bf16."""

import torch


def pack_slots(slots: torch.Tensor, columns: int, dtype: torch.dtype) -> torch.Tensor:
    """Encode nonnegative slot IDs as bytes, each exactly representable in bf16."""
    return torch.stack([(slots >> (8 * i)) & 255 for i in range(columns)], -1).to(dtype)


def unpack_slots(packed: torch.Tensor) -> torch.Tensor:
    """Decode bytes without host reads; invalid profiling rows become slot zero."""
    parts = packed.round().long()
    valid = ((parts >= 0) & (parts <= 255)).all(dim=-1)
    slots = torch.zeros_like(parts[..., 0])
    for i in range(parts.shape[-1]):
        slots = slots | (parts[..., i] << (8 * i))
    return torch.where(valid, slots, torch.zeros_like(slots))
