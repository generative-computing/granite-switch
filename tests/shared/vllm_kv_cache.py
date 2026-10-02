# SPDX-License-Identifier: Apache-2.0
"""Allocate a real KV cache for an :class:`Attention` layer, on any vLLM 0.26+.

These harnesses hand the tensor to a real FlashAttention kernel (they build real
``FlashAttentionMetadata``: slot_mapping, block_table, query_start_loc, seq_lens),
so the shape has to be genuinely right, not merely plausible — a wrong one reads
back garbage instead of raising.

Two APIs, because vLLM 0.29 deleted the first:

* 0.26 - 0.28: ``attn_backend.get_kv_cache_shape(num_blocks, block_size,
  num_kv_heads, head_size)``, a per-backend staticmethod. 29 definitions at 0.28,
  ZERO at 0.29 (0.30 keeps one, on flashinfer_mla_sparse_sm90).
* 0.29+: ``compute_layer_kv_cache_shape_bytes(spec, num_blocks)``, a module
  function returning ``(B, H, N, C)`` with C in BYTES rather than elements.

The two are provably equal for dense bf16 attention, which is why this is a port
and not a guess. FlashAttention returns
``(num_blocks, num_kv_heads, block_size, 2 * head_size)``; for a
``FullAttentionSpec`` with no packing overrides, ``num_heads == num_kv_heads``,
``get_num_kernel_states(bs) == bs`` (``tokens_per_state == 1``), and
``state_content_size_bytes == (head_size + head_size_v) * itemsize``, so dividing
the last dim by the itemsize reproduces the old tuple exactly.

No shipped version offers both APIs for FlashAttention — 0.26-0.28 have only the
backend staticmethod, 0.29+ only the spec function — so that equality is an
argument from the source above, not something a run can confirm. The cross-check
below therefore never fires today; it is there so that a future version which
reintroduces the staticmethod tells us immediately if the two ever disagree.
"""

import torch


def kv_cache_shape(attn, vllm_config, num_blocks: int, block_size: int):
    """The KV-cache shape, in ELEMENTS, for one ``Attention`` layer."""
    from vllm.utils.torch_utils import get_dtype_size

    try:
        from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes
    except ImportError:  # 0.26 - 0.28
        compute_layer_kv_cache_shape_bytes = None

    legacy = None
    if hasattr(attn.attn_backend, "get_kv_cache_shape"):  # 0.26 - 0.28
        legacy = tuple(
            attn.attn_backend.get_kv_cache_shape(
                num_blocks,
                block_size,
                attn.num_kv_heads,
                attn.head_size,
            )
        )
        if compute_layer_kv_cache_shape_bytes is None:
            return legacy
    assert compute_layer_kv_cache_shape_bytes is not None, (
        "this vLLM has neither attn_backend.get_kv_cache_shape nor "
        "compute_layer_kv_cache_shape_bytes"
    )

    spec = attn.get_kv_cache_spec(vllm_config)
    assert spec is not None, (
        f"{attn.layer_name} contributes no KV cache; it cannot be set up here"
    )
    # compute_layer_kv_cache_shape_bytes derives the leading dim from
    # spec.block_size (which tracks vllm_config.cache_config.block_size), so a
    # harness block_size that disagrees would silently shift every block index.
    assert spec.block_size == block_size, (
        f"harness block_size={block_size} but the spec says {spec.block_size}; "
        "set cache_config.block_size to match"
    )
    *lead, content_bytes = compute_layer_kv_cache_shape_bytes(spec, num_blocks)
    itemsize = get_dtype_size(spec.dtype)
    assert content_bytes % itemsize == 0, (
        f"{content_bytes} bytes per cell is not a whole number of "
        f"{spec.dtype} elements; this layer's cache is packed, not dense"
    )
    shape = (*lead, content_bytes // itemsize)

    if legacy is not None:
        assert shape == legacy, (
            f"the 0.29+ spec path gives {shape} but the backend's own "
            f"get_kv_cache_shape gives {legacy}"
        )
    return shape


def setup_kv_cache(attn, vllm_config, num_blocks: int, block_size: int, device):
    """Attach a zeroed bf16 KV cache to ``attn`` and return it."""
    attn.kv_cache_torch_dtype = torch.bfloat16
    shape = kv_cache_shape(attn, vllm_config, num_blocks, block_size)
    kv_cache = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    attn.kv_cache = kv_cache
    return kv_cache
