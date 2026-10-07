# SPDX-License-Identifier: Apache-2.0
"""Size FlashAttention 3's ahead-of-time schedule for the layers it serves.

vLLM builds one FlashAttention metadata builder per KV-cache group and sizes
FA3's ahead-of-time (AOT) scheduler metadata from the *model config*: its query
heads, KV heads and head size. A Granite Switch checkpoint has attention layers
whose shape differs from the model config:

- the Shadow-Residual decoder doubles its query heads
  (``Attention(2 * num_heads, ...)``) against base-only K/V, but the model
  config reports the un-doubled ``num_attention_heads``; and
- the switch's counting and memory heads use their own ``head_size``.

For such a group FA3's schedule is computed for the wrong shape. Eager steps
raise ``scheduler_metadata must have shape (metadata_size)``; CUDA-graph steps
compute wrong attention silently. It shows only in small, prefix-cached steps
(one conversation or one game served alone), so parity checks on batches of
fresh prompts pass while the served model misreads its context. See
``docs/FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md`` / issue #139.

This patch wraps ``FlashAttentionMetadataBuilder.__init__`` and, for a
``granite_switch`` model, resizes the schedule from the group's *own* layers:

- Single-shape group: overwrite ``num_heads_q``/``num_heads_kv``/``headdim`` with
  the group's real shape. AOT stays on (the FULL-cudagraph buffer is sized from
  batch, not shape, so it remains valid). This is the correctness fix for the SR
  decoder group and keeps the FULL latency win.
- Genuinely mixed-shape group (rare): disable AOT *consistently* —
  ``aot_schedule = False`` **and** tear down the FULL commitment
  (``scheduler_metadata = None``, ``max_num_splits = 0``) so the builder state is
  self-consistent whatever path ``build()`` takes, regardless of vLLM's internal
  ordering.

Other models are untouched. The patch targets vLLM 0.26-0.30 internals.

NOTE on reproducing the bug: it only manifests under FlashAttention **3**, which
requires a Hopper GPU (compute capability 9.x). On older GPUs vLLM uses FA2,
whose cudagraph support forces ``cudagraph_mode=FULL`` down to
``FULL_AND_PIECEWISE`` and the bug cannot appear. This patch is still correct to
install everywhere; it is simply inert where FA3 is not used.
"""

from __future__ import annotations

_PATCHED = False

# Builder instance attributes the patch reads/writes; asserted present so a
# future vLLM refactor fails loudly instead of silently installing a no-op.
_REQUIRED_BUILDER_ATTRS = ("num_heads_q", "num_heads_kv", "headdim", "aot_schedule")


def patch_flash_attn_schedule() -> None:
    """Install the FA3-schedule patch (idempotent).

    Soft-returns only when vLLM is genuinely absent (CPU-only / import error).
    When vLLM is present but its internals have moved, it logs loudly and does
    **not** mark the patch installed — a silent no-op here would reintroduce
    wrong tokens under FA3 + FULL cudagraphs.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.logger import init_logger
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.attention.backends import flash_attn as fa
    except ImportError:
        # vLLM not installed (e.g. CPU-only or a non-vLLM backend). Not an error;
        # do NOT mark _PATCHED so a later call in a vLLM-capable process retries.
        return

    logger = init_logger(__name__)

    builder = getattr(fa, "FlashAttentionMetadataBuilder", None)
    if builder is None:
        logger.warning(
            "granite_switch: could not find FlashAttentionMetadataBuilder in "
            "vllm.v1.attention.backends.flash_attn; FA3 AOT-schedule patch NOT "
            "installed. Under FlashAttention 3 + cudagraph_mode=FULL this will "
            "produce wrong tokens (see docs/FA3_SCHEDULE_FULL_CUDAGRAPH_BUG.md). "
            "vLLM internals may have changed; the patch needs updating."
        )
        # Not marked installed: a vLLM version bump that restores the symbol
        # (or a fixed patch) should get another chance.
        return

    if getattr(builder, "_granite_switch_schedule", False):
        _PATCHED = True
        return

    original = builder.__init__

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        original(self, kv_cache_spec, layer_names, vllm_config, device)

        hf = getattr(vllm_config.model_config, "hf_config", None)
        if getattr(hf, "model_type", None) != "granite_switch":
            return

        # The attributes we are about to touch must exist; if vLLM renamed them
        # the correction would silently miss, so fail loudly instead.
        missing = [a for a in _REQUIRED_BUILDER_ATTRS if not hasattr(self, a)]
        if missing:
            raise AttributeError(
                "granite_switch: FlashAttentionMetadataBuilder is missing "
                f"attribute(s) {missing}; the FA3 AOT-schedule patch cannot size "
                "the schedule for the switch's own layers. vLLM internals have "
                "changed — update fa3_schedule.py."
            )

        if not self.aot_schedule:
            # No AOT schedule to size (e.g. FA2 fallback, or already disabled).
            return

        layers = get_layers_from_vllm_config(vllm_config, Attention, layer_names)
        if not layers:
            # This builder serves a non-attention group; nothing to resize.
            return

        shapes = {
            (a.num_heads, a.num_kv_heads, a.head_size) for a in layers.values()
        }
        if len(shapes) == 1:
            # One real shape for the whole group — size the schedule to it. This
            # corrects the SR doubled-Q decoder group (and any switch group whose
            # shape disagrees with the model config) while keeping AOT on.
            self.num_heads_q, self.num_heads_kv, self.headdim = shapes.pop()
        else:
            # Genuinely mixed shapes in one group (rare). No single AOT schedule
            # is correct, so disable AOT and tear down the FULL-cudagraph
            # commitment consistently, leaving the builder self-consistent for
            # whatever path build() takes.
            self.aot_schedule = False
            if hasattr(self, "scheduler_metadata"):
                self.scheduler_metadata = None
            if hasattr(self, "max_num_splits"):
                self.max_num_splits = 0

    builder.__init__ = __init__
    builder._granite_switch_schedule = True
    _PATCHED = True
