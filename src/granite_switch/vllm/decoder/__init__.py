# SPDX-License-Identifier: Apache-2.0
"""Decoder tier for the shared Granite Switch vLLM model.

Granite Switch hosts two *adaptations* — LoRA and Shadow-Residual (SR) — that
diverge only at the decoder. This package holds that divergence:

* :mod:`.interface` — the :class:`DecoderInterface` ABC + the LoRA/SR
  implementations + :func:`select_decoder_interface` (keyed on
  ``config.cross_stream_rank``). The shared model calls its hooks and never
  branches on which adaptation is in use.
* :mod:`.lora` — the single-stream LoRA decoder layer.
* :mod:`.shadow_residual` — the dual-stream SR decoder layer + its kernel metadata,
  cross-stream shunt, and head ops.

Everything above the decoder (vocab sizing, PP handoff, ``compute_logits``,
``sample``, the four host interfaces) is shared, adaptation-agnostic host code.
"""

from .interface import (
    DecoderInterface,
    LoRADecoderInterface,
    SRDecoderInterface,
    is_shadow_residual,
    select_decoder_interface,
)
from .lora import (
    GraniteLoRAEmbeddedAttention,
    GraniteSwitchDecoderLayer,
    rms_norm_select,
)
from .shadow_residual.decoder import ShadowResidualDecoderLayer

__all__ = [
    "DecoderInterface",
    "GraniteLoRAEmbeddedAttention",
    "GraniteSwitchDecoderLayer",
    "LoRADecoderInterface",
    "SRDecoderInterface",
    "ShadowResidualDecoderLayer",
    "is_shadow_residual",
    "rms_norm_select",
    "select_decoder_interface",
]
