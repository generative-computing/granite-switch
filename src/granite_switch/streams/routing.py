# SPDX-License-Identifier: Apache-2.0
"""Routing tables a model builds from its streams, shared by both backends.

From ``config.streams`` (via :func:`~.spec.stream_specs`) each backend builds,
once at load time:

* the head slots' control-token ids (to find marker positions);
* ``row_lut`` -- per slot index, ``1 + head_row`` in that slot's type bank, or 0
  (a LoRA slot, or base); the head modules take these "row + 1" indices, with
  0 meaning "not this type's marker";
* ``outputs`` -- per slot, its output mode, token ids and merge weight, as
  tables a row gathers from;
* the read groups -- per read layer, the head types that read it, and which
  slots of each.

Nothing here depends on a particular head type.
"""

import torch

from .outputs import output_tables
from .registry import get_head_type
from .spec import stream_specs


def head_specs(config):
    """``{type name: [StreamSpec, ...]}`` for the config's head slots."""
    by_type: dict[str, list] = {}
    for s in stream_specs(config):
        if s.is_head:
            by_type.setdefault(s.type, []).append(s)
    return by_type


def build_head_modules(config, backend: str) -> dict:
    """``{type name: nn.Module}`` -- one module per head type, for ``backend``."""
    modules = {}
    for name, specs in head_specs(config).items():
        head_type = get_head_type(name)
        for s in specs:
            head_type.check(s, config)
        modules[name] = (
            head_type.vllm_module(specs, config)
            if backend == "vllm"
            else head_type.hf_module(specs, config)
        )
    return modules


def module_attr(name: str) -> str:
    """The model attribute holding head type ``name``'s module.

    A type may pin its checkpoint names with ``module_name`` (the probe keeps
    ``probe_head``); others live at ``stream_<name>``.
    """
    return getattr(get_head_type(name), "module_name", None) or f"stream_{name}"


def head_control_token_ids(config) -> list[int]:
    """Control-token ids of the head slots (a base-reset token is not one)."""
    offset = len(config.adapter_token_ids or []) - config.num_adapters
    return [
        config.adapter_token_ids[s.slot - 1 + offset]
        for s in stream_specs(config)
        if s.is_head
    ]


def row_lut(config, type_name: str) -> torch.Tensor:
    """``[num_adapters + 1]``: slot -> 1 + head_row for ``type_name``, else 0."""
    lut = torch.zeros(config.num_adapters + 1, dtype=torch.long)
    for s in stream_specs(config):
        if s.type == type_name:
            lut[s.slot] = s.head_row + 1
    return lut


def adapter_lut(config) -> torch.Tensor:
    """``[num_adapters + 1]``: slot -> the adapter slot its LoRA layers run.

    The identity for adapter slots and base (0); 0 for a head slot on the base
    model; the named adapter's slot for a head slot with an ``adapter``. The
    model routes LoRA through this table, so a head's marker token runs its
    adapter, and the head then reads that adapter's state.
    """
    lut = torch.zeros(config.num_adapters + 1, dtype=torch.long)
    names = config.adapter_names or []
    for s in stream_specs(config):
        if not s.is_head:
            lut[s.slot] = s.slot
        elif s.adapter is not None:
            # The marker runs the adapter. On Shadow Residual that alone gives
            # the adapter's state, because the adapter writes no K/V. A
            # single-stream adapter writes the K/V of every token it runs on,
            # so its own control token must also sit in the prompt, where that
            # adapter goes (the chat template places it), for the head to read
            # the state the adapter computes.
            target = names.index(s.adapter) + 1 if s.adapter in names else 0
            if target == 0 or stream_specs(config)[target - 1].is_head:
                raise ValueError(
                    f"stream slot {s.slot} names adapter {s.adapter!r}, which is not "
                    f"an adapter slot of this checkpoint ({names})."
                )
            lut[s.slot] = target
    return lut


def lora_rows(config) -> list[int]:
    """The adapter slots that own a LoRA row, in row order (1-based slots).

    A pre-streams checkpoint gave every slot a row in every LoRA tensor (head
    slots' rows stay zero), so its rows are all slots. A streams checkpoint
    keeps rows for adapter slots only.
    """
    specs = stream_specs(config)
    if config.streams is None:
        return [s.slot for s in specs]
    return [s.slot for s in specs if not s.is_head]


def lora_row_lut(config) -> torch.Tensor:
    """``[num_adapters + 1]``: slot -> 1 + its LoRA row (0 = no LoRA).

    Composed with :func:`adapter_lut`, a token's LoRA row is
    ``lora_row_lut[adapter_lut[slot]]``: a head on an adapter reads that
    adapter's row.
    """
    lut = torch.zeros(config.num_adapters + 1, dtype=torch.long)
    for row, slot in enumerate(lora_rows(config)):
        lut[slot] = row + 1
    return lut


def head_slot_mask(config) -> torch.Tensor:
    """``[num_adapters + 1]`` bool: which slot indices are head slots."""
    mask = torch.zeros(config.num_adapters + 1, dtype=torch.bool)
    for s in stream_specs(config):
        if s.is_head:
            mask[s.slot] = True
    return mask


def head_outputs(config) -> dict:
    """The head slots' output tables (:func:`~.outputs.output_tables`)."""
    return output_tables(stream_specs(config), config.num_adapters)


def max_head_outputs(config) -> int:
    """Width of a head score row: the widest head's output count (0 if none)."""
    return max((s.num_outputs for s in stream_specs(config) if s.is_head), default=0)


def read_groups(config) -> dict:
    """``{read_layer: {type name: [slot, ...]}}`` (``None`` = after the final norm)."""
    groups: dict = {}
    for s in stream_specs(config):
        if s.is_head:
            groups.setdefault(s.read_layer, {}).setdefault(s.type, []).append(s.slot)
    return groups
