# SPDX-License-Identifier: Apache-2.0
"""The split of the switch's index stream into adapter and head streams.

A slot is either an adapter stream or a head stream (any type other than
``lora``), selected by the same control token and adapter index. These helpers,
shared by both backends' switches, split the switch's single per-token index
stream so that head positions leave the adapters' path (unless a head names an
adapter, which the models route separately) and feed the head streams.

Both switch engines use the shared stream routing table.
"""

import torch

from .routing import head_slot_mask
from .spec import stream_specs


def build_head_slot_lut(config) -> torch.Tensor | None:
    """Bool LUT over adapter indices marking which slots are head streams.

    ``[num_adapters + 1]``, indexed by ADAPTER INDEX: entry ``i`` is True when
    adapter index ``i`` is a head slot -- any stream type other than ``lora``
    (a probe, or a registered head type; see :mod:`granite_switch.streams`).
    Index 0, base, is always False. ``None`` when no slot is a head, which lets
    the switch skip the split entirely.
    """
    if config is None or getattr(config, "adapter_kinds", None) is None:
        return None
    if not any(s.is_head for s in stream_specs(config)):
        return None
    return head_slot_mask(config)


def split_head_indices(
    kind_lut: torch.Tensor | None, adapter_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-token index stream into LoRA and head streams.

    Shapes are preserved (``[batch, seq_len]`` on HF, ``[total_tokens]`` on
    vLLM), so this serves both backends:

      - ``lora_indices``: head positions zeroed, so the LoRA path no-ops
        there (a head that names an adapter is routed back by the models).
      - ``head_indices``: the adapter index at head positions, else 0.

    With ``kind_lut`` None (no head slots) the LoRA stream passes through
    and the head stream is all zeros.
    """
    if kind_lut is None:
        return adapter_indices, torch.zeros_like(adapter_indices)
    is_head = kind_lut[adapter_indices]
    zeros = torch.zeros_like(adapter_indices)
    return (
        torch.where(is_head, zeros, adapter_indices),
        torch.where(is_head, adapter_indices, zeros),
    )
