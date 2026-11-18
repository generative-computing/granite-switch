# SPDX-License-Identifier: Apache-2.0
"""Streams: pluggable per-slot heads for Granite Switch.

A *stream* is what a control token turns on. Every adapter slot is a stream,
and a stream type decides what happens at that slot's tokens:

* ``lora`` -- a LoRA / aLoRA / Shadow Residual adapter. Built into the decoder
  layers; its weights are the stacked LoRA banks. The LM head makes the logits.
* a *head type* (``probe``, or any registered type) -- a small module that reads
  the hidden state at one layer, at the slot's marker token, and returns scores
  that the model writes into that row's logits (see :mod:`.outputs`).

A head type plugs in with three things and no change to the model files:

1. its config entry -- ``config.streams[slot]`` (see :mod:`.spec`);
2. its forward code -- a :class:`~.registry.HeadType` with an HF and a vLLM
   module (see :mod:`.registry`);
3. its weights -- a safetensors file in its folder, loaded by compose.

This is part 1 of the stream API: single-token streams, whose output is one
logits row at the marker. Block streams (several tokens at once) come later.
"""

from .outputs import OUTPUT_MODES
from .registry import HeadType, get_head_type, head_type_names, register_head_type
from .spec import StreamSpec, stream_specs

__all__ = [
    "OUTPUT_MODES",
    "HeadType",
    "StreamSpec",
    "get_head_type",
    "head_type_names",
    "register_head_type",
    "stream_specs",
]
