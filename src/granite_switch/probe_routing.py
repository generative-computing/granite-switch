# SPDX-License-Identifier: Apache-2.0
"""Activation-probe slot routing helpers, shared by both backends' switches.

An adapter slot is either a generation adapter or an activation probe head,
selected by the same control token and adapter index. These helpers split the
switch's single per-token index stream so that probe positions no-op the LoRA
path and feed the probe head instead.

Backend-neutral on purpose: the table is a pure function of
``config.adapter_kinds``, so both switch engines (HF and vLLM MultiSwitch)
build the identical table.
"""

import torch


def build_adapter_kind_lut(config) -> torch.Tensor | None:
    """Bool LUT over adapter indices marking which slots are probes.

    ``[num_adapters + 1]``, indexed by ADAPTER INDEX: entry ``i`` is True when
    adapter index ``i`` is a probe slot (index 0, base, is always False).
    ``None`` when no slot is a probe, which lets the switch skip the split
    entirely. Built from ``config.adapter_kinds``, so both backends' switches
    agree on the table by construction.
    """
    kinds = getattr(config, "adapter_kinds", None) if config is not None else None
    if kinds is None or not any(k == "probe" for k in kinds):
        return None
    lut = torch.zeros(len(kinds) + 1, dtype=torch.bool)
    for slot, kind in enumerate(kinds):  # slot is the 0-based adapter position
        if kind == "probe":
            lut[slot + 1] = True  # adapter index is slot + 1
    return lut


def split_adapter_indices(
    kind_lut: torch.Tensor | None, adapter_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-token index stream into LoRA and probe streams.

    Shapes are preserved (``[batch, seq_len]`` on HF, ``[total_tokens]`` on
    vLLM), so this serves both backends:

      - ``lora_indices``: probe positions zeroed, so the LoRA path no-ops
        there.
      - ``probe_indices``: the adapter index at probe positions, else 0.

    With ``kind_lut`` None (no probe slots) the LoRA stream passes through
    and the probe stream is all zeros.
    """
    if kind_lut is None:
        return adapter_indices, torch.zeros_like(adapter_indices)
    is_probe = kind_lut[adapter_indices]
    zeros = torch.zeros_like(adapter_indices)
    return (
        torch.where(is_probe, zeros, adapter_indices),
        torch.where(is_probe, adapter_indices, zeros),
    )
