# SPDX-License-Identifier: Apache-2.0
"""Keep FlashAttention 3 attention out of the FULL CUDA graph for Granite Switch.

On vLLM 0.26 under FlashAttention **3**, serving a Granite Switch checkpoint with
``cudagraph_mode=FULL`` produces attention that diverges — grossly, from the very
first prefill position — from the eager / ``FULL_AND_PIECEWISE`` reference. The
divergence is in the attention op *as captured inside the full CUDA graph*:
turning off FULL (eager), or dropping attention out of the graph
(``FULL_AND_PIECEWISE`` / ``PIECEWISE``), matches the reference exactly. See
issue #139.

Why FA3 + FULL specifically
---------------------------
``FlashAttentionMetadataBuilder.get_cudagraph_support`` reports
``AttentionCGSupport.ALWAYS`` when FA3 is active and ``UNIFORM_BATCH`` otherwise
(``vllm/v1/attention/backends/flash_attn.py``). ``ALWAYS`` is what lets
``cudagraph_mode=FULL`` capture attention for mixed prefill-decode batches; with
``UNIFORM_BATCH`` vLLM downgrades ``FULL`` to ``FULL_AND_PIECEWISE`` and runs
attention eagerly. On an FA2 (non-Hopper) box the backend already reports
``UNIFORM_BATCH``, FULL silently downgrades, and the bug never appears — which is
why it had never been seen until an FA3-capable (Hopper, compute 9.x) run.

The break is **not** Granite-Switch-specific in mechanism: a plain Granite base
served through this same vLLM 0.26 + FA3 build diverges under FULL in exactly the
same way. It is an upstream FA3 full-cudagraph correctness problem on this vLLM
line. But Granite Switch is where it bites in production (FULL was chosen for the
latency of tiny prefix-cached decode steps), so the backend's broad FA3-ALWAYS
promise is wrong *for this model* and we narrow it here.

The fix
-------
Wrap ``get_cudagraph_support`` so that, **for a granite_switch model only**, the
FlashAttention builder reports ``UNIFORM_BATCH`` instead of ``ALWAYS``. vLLM then
downgrades ``cudagraph_mode=FULL`` to ``FULL_AND_PIECEWISE`` on its own
(logging ``CUDAGraphMode.FULL is not supported ... setting
cudagraph_mode=FULL_AND_PIECEWISE``): the compute-heavy MLP/projection regions
stay in the graph, attention runs eagerly, and output matches the eager
reference token-for-token. Every other model keeps FA3's native ``ALWAYS``
support untouched.

An earlier revision of this module tried to *resize* FA3's ahead-of-time (AOT)
schedule — overwriting ``num_heads_q`` for the SR decoder's doubled query heads —
on the theory that a mis-sized schedule was the cause. A GPU reproduction
disproved it: the AOT head count was already being resized correctly, yet FULL
still diverged, and disabling AOT entirely did not help either (attention stayed
captured). The schedule shape was a red herring; the capture of attention itself
is the fault. See the doc for the full trace.

This costs the FULL-vs-FULL_AND_PIECEWISE latency delta on attention launches,
which is the price of correctness until the upstream FA3 full-cudagraph path is
fixed on this vLLM line. ``PIECEWISE``/``FULL_AND_PIECEWISE`` remain fully
captured everywhere except the attention op.

The patch targets vLLM 0.26-0.30 internals and is inert on any build without an
FA3 ``ALWAYS`` promise (e.g. FA2), where it would be a no-op anyway.
"""

from __future__ import annotations

_PATCHED = False


def patch_flash_attn_schedule() -> None:
    """Install the FA3 full-cudagraph downgrade for granite_switch (idempotent).

    Soft-returns only when vLLM is genuinely absent (CPU-only / import error).
    When vLLM is present but its internals have moved, it logs loudly and does
    **not** mark the patch installed — a silent no-op here would reintroduce
    wrong tokens under FlashAttention 3 + cudagraph_mode=FULL.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from vllm.logger import init_logger
        from vllm.v1.attention.backends import flash_attn as fa
    except ImportError:
        # vLLM not installed (e.g. CPU-only or a non-vLLM backend). Not an error;
        # do NOT mark _PATCHED so a later call in a vLLM-capable process retries.
        return

    logger = init_logger(__name__)

    builder = getattr(fa, "FlashAttentionMetadataBuilder", None)
    get_support = getattr(builder, "get_cudagraph_support", None)
    if builder is None or get_support is None:
        logger.warning(
            "granite_switch: could not find "
            "FlashAttentionMetadataBuilder.get_cudagraph_support in "
            "vllm.v1.attention.backends.flash_attn; the FA3 full-cudagraph "
            "downgrade is NOT installed. Under FlashAttention 3 + "
            "cudagraph_mode=FULL this will produce wrong tokens (see "
            "issue #139). vLLM internals may have "
            "changed; the patch needs updating."
        )
        # Not marked installed: a vLLM version bump that restores the symbol
        # (or a fixed patch) should get another chance.
        return

    if getattr(builder, "_granite_switch_cg_support", False):
        _PATCHED = True
        return

    # UNIFORM_BATCH is the FA2 support level: it tells vLLM the backend cannot
    # take attention into a FULL graph for mixed prefill-decode, so FULL is
    # downgraded to FULL_AND_PIECEWISE. Resolve the enum defensively so a rename
    # fails loudly rather than silently installing a no-op.
    cg_support_enum = getattr(fa, "AttentionCGSupport", None)
    uniform_batch = getattr(cg_support_enum, "UNIFORM_BATCH", None)
    if uniform_batch is None:
        logger.warning(
            "granite_switch: could not resolve "
            "AttentionCGSupport.UNIFORM_BATCH; the FA3 full-cudagraph downgrade "
            "is NOT installed and FlashAttention 3 + cudagraph_mode=FULL will "
            "produce wrong tokens (see issue #139). "
            "vLLM internals may have changed; the patch needs updating."
        )
        return

    # get_cudagraph_support is a classmethod; unwrap to the plain function so the
    # wrapper can re-bind it as a classmethod cleanly (and call through for every
    # non-granite_switch model).
    original = get_support.__func__ if hasattr(get_support, "__func__") else get_support

    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        hf = getattr(vllm_config.model_config, "hf_config", None)
        if getattr(hf, "model_type", None) == "granite_switch":
            # Force FULL -> FULL_AND_PIECEWISE: attention runs eagerly (correct)
            # while the rest of the model stays in the CUDA graph.
            return uniform_batch
        return original(cls, vllm_config, kv_cache_spec)

    builder.get_cudagraph_support = classmethod(get_cudagraph_support)
    builder._granite_switch_cg_support = True
    _PATCHED = True
