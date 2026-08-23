# SPDX-License-Identifier: Apache-2.0
"""Long-lived subprocess worker for vLLM MultiSwitch tests (coded engine).

Generalizes ``_single_switch_worker.py`` to the Kerdock/DG coded-memory
MultiSwitch engine:

- ``multi`` — 2 Attention layers (counting slot + memory slot). Each needs its
  own KV-cache tensor and its own entry in the ForwardContext's
  ``attn_metadata`` / ``slot_mapping`` maps, keyed by the layer's ``layer_name``
  (the ``prefix`` passed at construction: ``switch.multi.0`` / ``switch.multi.1``).
  This worker auto-discovers every ``Attention`` submodule of the switch and
  wires up all of them, so it is agnostic to how many there are.

Protocol (JSON-line over stdin/stdout), started with argv[1] = switch_type:
  Startup: {"ready": true, "backend_name": "...", "switch_type": "...",
            "num_attn_layers": N}
       OR  {"fatal": "...", "hint": "...", "backend_name": "..."} then exit
           (e.g. FA kernel-image mismatch on the auto-selected backend).
  Request: {"seq": [...], "adapter_token_ids": [...]}
  Response: {"result": [...]}
  Error: {"error": "..."}
  Shutdown: EOF on stdin

All diagnostic output goes to stderr; only JSON on stdout.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tests.shared.vllm_distributed import ensure_distributed

BLOCK_SIZE = 16
MAX_TOKENS = 8192
NUM_ADAPTERS = 2
# Token conventions mirror tests/shared/multi_switch_cases.py, but the worker
# receives the concrete adapter_token_ids in each request (layout may vary).
VOCAB_SIZE = 2000
# Deterministic substitute mapping large enough for either layout.
ADAPTER_SUBSTITUTE_TOKEN_IDS = [1, 2, 3]


def _mock_config(switch_type, adapter_token_ids, adapter_substitute_token_ids):
    """GraniteSwitchConfig-shaped mock with realistic backbone geometry.

    The coded engine fixes its expert-id offset and token-exchange LUT at
    __init__ from ``adapter_token_ids`` length (no-base-slot vs base-reset),
    so the worker rebuilds the switch when the requested layout changes.
    """
    return SimpleNamespace(
        switch_type=switch_type,
        num_adapters=NUM_ADAPTERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        projection_head_dim=64,
        attention_multiplier=0.125,
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        adapter_token_ids=list(adapter_token_ids),
        adapter_substitute_token_ids=list(adapter_substitute_token_ids),
        switch_head_dim=32,
        control_token_gain=15.0,
        ms_code_m=6,
        ms_code_type="kerdock",
        ms_memory_gain=28.0,
        ms_counting_head_dim=32,
    )


def _substitutes_for(adapter_token_ids):
    """base-reset layout (len == num_adapters+1) maps slot0->0; else i+1."""
    if len(adapter_token_ids) == NUM_ADAPTERS + 1:
        return list(range(len(adapter_token_ids)))  # [0, 1, 2]
    return [i + 1 for i in range(len(adapter_token_ids))]  # [1, 2]


def _discover_attention_layers(switch):
    """Return [(layer_name, attn_module)] for every vLLM Attention submodule."""
    from vllm.model_executor.layers.attention.attention import Attention

    layers = []
    for module in switch.modules():
        if isinstance(module, Attention):
            layers.append((module.layer_name, module))
    return layers


def _build_switch(harness, adapter_token_ids):
    """(Re)build the switch for a given adapter_token_ids layout.

    Pops any prior Attention layers from the vLLM static forward context first
    (their ``prefix`` names would otherwise collide on reconstruction), then
    builds a fresh switch, discovers its Attention layers, and allocates one KV
    cache per layer. Updates ``harness`` in place and returns it.
    """
    from vllm.config import set_current_vllm_config

    from granite_switch.vllm.switch import create_switch

    device = harness["device"]
    vllm_config = harness["vllm_config"]
    switch_type = harness["switch_type"]

    # Drop previous layers' names so create_switch can re-register its prefixes.
    sfc = vllm_config.compilation_config.static_forward_context
    for layer_name, _attn in harness.get("attn_layers", []):
        sfc.pop(layer_name, None)

    subs = _substitutes_for(adapter_token_ids)
    config = _mock_config(switch_type, adapter_token_ids, subs)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with set_current_vllm_config(vllm_config):
            switch = create_switch(config, vllm_config=vllm_config)
        switch = switch.to(device)
    finally:
        torch.set_default_dtype(old_dtype)

    attn_layers = _discover_attention_layers(switch)

    kv_caches = {}
    backend_name = None
    num_blocks = (MAX_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    for layer_name, attn in attn_layers:
        attn.kv_cache_torch_dtype = torch.bfloat16
        if backend_name is None:
            backend_name = attn.attn_backend.get_name()
        cache_shape = attn.attn_backend.get_kv_cache_shape(
            num_blocks,
            BLOCK_SIZE,
            attn.num_kv_heads,
            attn.head_size,
        )
        kv_cache = torch.zeros(cache_shape, device=device, dtype=torch.bfloat16)
        attn.kv_cache = kv_cache
        kv_caches[layer_name] = kv_cache

    harness["switch"] = switch
    harness["attn_layers"] = attn_layers
    harness["kv_caches"] = kv_caches
    harness["config"] = config
    harness["layout_len"] = len(adapter_token_ids)
    if backend_name is not None:
        harness["backend_name"] = backend_name
    return harness


def _setup(switch_type):
    """Create VllmConfig + the switch (no-base-slot layout) + KV caches."""
    # Redirect fd 1 -> fd 2 during setup so native CUDA/FA init doesn't
    # contaminate the JSON-line protocol on stdout (same as the single worker).
    _saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    _saved_fd1 = os.dup(1)
    os.dup2(2, 1)

    from vllm.config import VllmConfig

    device = torch.device("cuda")
    vllm_config = VllmConfig()
    ensure_distributed(vllm_config)

    harness = {
        "switch_type": switch_type,
        "vllm_config": vllm_config,
        "device": device,
        "attn_layers": [],
        "block_size": BLOCK_SIZE,
        "backend_name": "NONE",
    }
    # Seed with the no-base-slot layout (2 control tokens).
    _build_switch(harness, [101, 102])

    # Restore stdout for the JSON protocol.
    os.dup2(_saved_fd1, 1)
    os.close(_saved_fd1)
    sys.stdout = _saved_stdout

    return harness


def _build_metadata(harness, seq_len):
    """Build FlashAttention metadata for a single-sequence prefill.

    Returns (metadata, slot_mapping). The metadata is shared across all the
    switch's Attention layers (they have the same seq geometry); each layer gets
    its own ForwardContext entry keyed by layer_name.
    """
    device = harness["device"]
    block_size = harness["block_size"]
    backend_name = harness["backend_name"]

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(
        num_blocks_needed,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    query_start_loc = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    if backend_name == "FLASH_ATTN":
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        # scheduler_metadata is FA3-only (Hopper SM90+); passing it on FA2
        # forces FA3 dispatch and crashes. Only compute it when FA version == 3.
        scheduler_metadata = None
        try:
            from vllm.v1.attention.backends.fa_utils import (
                get_flash_attn_version,
                get_scheduler_metadata,
            )

            if get_flash_attn_version() == 3:
                # Use the first attention layer's head geometry (counting head).
                _name, attn = harness["attn_layers"][0]
                scheduler_metadata = get_scheduler_metadata(
                    batch_size=1,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                    num_heads_q=attn.num_heads,
                    num_heads_kv=attn.num_kv_heads,
                    headdim=attn.head_size,
                    cache_seqlens=seq_lens,
                    qkv_dtype=torch.bfloat16,
                    cu_seqlens_q=query_start_loc,
                    page_size=block_size,
                    causal=True,
                    num_splits=0,
                )
        except ImportError:
            pass

        metadata = FlashAttentionMetadata(
            num_actual_tokens=seq_len,
            max_query_len=seq_len,
            query_start_loc=query_start_loc,
            max_seq_len=seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            causal=True,
            scheduler_metadata=scheduler_metadata,
        )
    else:
        raise RuntimeError(f"Backend {backend_name}: not supported by worker")

    return metadata, slot_mapping


def _run(harness, seq, adapter_token_ids):
    """Execute switch.forward and return adapter_indices as a list."""
    device = harness["device"]

    # The coded engine fixes its expert-id offset + LUT at __init__ from the
    # adapter_token_ids layout, so rebuild when the layout length changes.
    if len(adapter_token_ids) != harness.get("layout_len"):
        _build_switch(harness, adapter_token_ids)

    switch = harness["switch"]
    input_ids = torch.tensor(seq, dtype=torch.long, device=device)
    atok = torch.tensor(adapter_token_ids, dtype=torch.long, device=device)

    # coded: set up ForwardContext across all attention layers.
    from vllm.forward_context import ForwardContext, override_forward_context

    vllm_config = harness["vllm_config"]
    seq_len = len(seq)
    for kv in harness["kv_caches"].values():
        kv.zero_()

    metadata, slot_mapping = _build_metadata(harness, seq_len)

    attn_metadata = {}
    slot_mapping_map = {}
    for layer_name, _attn in harness["attn_layers"]:
        attn_metadata[layer_name] = metadata
        slot_mapping_map[layer_name] = slot_mapping

    forward_ctx = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping_map,
    )

    saved_direct = []
    for _name, attn in harness["attn_layers"]:
        saved_direct.append((attn, attn.use_direct_call))
        attn.use_direct_call = True

    try:
        with override_forward_context(forward_ctx):
            # Single request => positions are arange(seq_len): exactly one anchor,
            # at index 0. Passed explicitly because the switch no longer
            # synthesizes a fallback -- a synthesized arange is correct ONLY for
            # the single-request case and silently mis-routes real batches.
            adapter_indices, _modified = switch.forward(
                input_ids=input_ids,
                adapter_token_ids=atok,
                positions=torch.arange(seq_len, device=device),
            )
    finally:
        for attn, old in saved_direct:
            attn.use_direct_call = old

    return adapter_indices.cpu().tolist()


def _build_batch_metadata(harness, seq_lens):
    """Build FlashAttention metadata for a MULTI-request prefill batch.

    This is the batching test's core: it emulates vLLM continuous batching by
    packing several independent sequences into one flat token stream, with
    per-request ``query_start_loc`` / ``seq_lens`` / block tables so the FA
    kernel applies per-request causal masking (no cross-request attention).

    Returns (metadata, slot_mapping, positions) where ``positions`` RESETS to 0
    at each request boundary — exactly what vLLM feeds the model, and what the
    coded switch's ``positions == 0`` anchor relies on to place one counting
    anchor per request.
    """
    device = harness["device"]
    block_size = harness["block_size"]
    backend_name = harness["backend_name"]
    total = int(sum(seq_lens))

    # Per-request positions reset to 0 (vLLM's real layout for a fresh batch).
    positions = torch.cat(
        [torch.arange(n, dtype=torch.long, device=device) for n in seq_lens]
    )

    # query_start_loc: cumulative offsets [0, l0, l0+l1, ...].
    starts = [0]
    for n in seq_lens:
        starts.append(starts[-1] + int(n))
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(list(seq_lens), dtype=torch.int32, device=device)

    # Per-request block tables (each request gets its own contiguous blocks so
    # KV of different requests never overlaps in the paged cache).
    slot_parts, block_rows, next_block = [], [], 0
    max_blocks = max((n + block_size - 1) // block_size for n in seq_lens)
    for n in seq_lens:
        nb = (n + block_size - 1) // block_size
        base = next_block * block_size
        slot_parts.append(
            torch.arange(base, base + n, dtype=torch.int64, device=device)
        )
        row = torch.arange(
            next_block, next_block + nb, dtype=torch.int32, device=device
        )
        if nb < max_blocks:  # pad ragged rows
            row = torch.cat([row, row.new_zeros(max_blocks - nb)])
        block_rows.append(row.unsqueeze(0))
        next_block += nb
    slot_mapping = torch.cat(slot_parts)
    block_table = torch.cat(block_rows, dim=0)

    if backend_name != "FLASH_ATTN":
        raise RuntimeError(f"Backend {backend_name}: not supported by worker")

    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    scheduler_metadata = None
    try:
        from vllm.v1.attention.backends.fa_utils import (
            get_flash_attn_version,
            get_scheduler_metadata,
        )

        if get_flash_attn_version() == 3:
            _name, attn = harness["attn_layers"][0]
            scheduler_metadata = get_scheduler_metadata(
                batch_size=len(seq_lens),
                max_seqlen_q=int(max(seq_lens)),
                max_seqlen_k=int(max(seq_lens)),
                num_heads_q=attn.num_heads,
                num_heads_kv=attn.num_kv_heads,
                headdim=attn.head_size,
                cache_seqlens=seq_lens_t,
                qkv_dtype=torch.bfloat16,
                cu_seqlens_q=query_start_loc,
                page_size=block_size,
                causal=True,
                num_splits=0,
            )
    except ImportError:
        pass

    metadata = FlashAttentionMetadata(
        num_actual_tokens=total,
        max_query_len=int(max(seq_lens)),
        query_start_loc=query_start_loc,
        max_seq_len=int(max(seq_lens)),
        seq_lens=seq_lens_t,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
        scheduler_metadata=scheduler_metadata,
    )
    return metadata, slot_mapping, positions


def _build_step_metadata(harness, query_lens, seq_lens, block_base):
    """Metadata for ONE forward where query_len may be < seq_len per request.

    Generalizes ``_build_batch_metadata`` to the cases real vLLM serving produces
    but the batch builder cannot express, because it assumes ``query_len ==
    seq_len`` (a whole-request prefill):

      * CHUNKED PREFILL — a request's prompt is split across forwards, so a later
        chunk has ``query_len < seq_len`` and, critically, contains NO position-0
        anchor (it lives in the first chunk, now only in the KV cache).
      * DECODE — ``query_len == 1`` with ``seq_len`` the full history.
      * MIXED — some requests prefilling (or chunk-prefilling) while others decode
        in the SAME forward.

    ``positions`` are the real absolute per-request positions of the queried
    tokens: for a request with ``cached = seq_len - query_len`` already in the
    cache, its query rows carry positions ``cached .. seq_len-1``. So position 0
    appears ONLY in a request's first chunk — which is exactly the property the
    coded engine's ``1/(1+n)`` anchor depends on, and exactly what a fabricated
    ``arange`` would destroy.

    ``block_base`` keeps each request's paged blocks stable ACROSS steps so the KV
    written by an earlier chunk is still addressable by a later one.
    """
    device = harness["device"]
    block_size = harness["block_size"]
    backend_name = harness["backend_name"]
    total = int(sum(query_lens))

    # Absolute positions of the QUERIED tokens only.
    positions = torch.cat(
        [
            torch.arange(s - q, s, dtype=torch.long, device=device)
            for q, s in zip(query_lens, seq_lens)
        ]
    )

    starts = [0]
    for n in query_lens:
        starts.append(starts[-1] + int(n))
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(list(seq_lens), dtype=torch.int32, device=device)

    # Slots for the queried tokens land at their ABSOLUTE offset in the request's
    # own block range, so a later chunk appends after the earlier one.
    max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
    slot_parts, block_rows = [], []
    for q, s, base_blk in zip(query_lens, seq_lens, block_base):
        cached = s - q
        base = base_blk * block_size
        slot_parts.append(
            torch.arange(base + cached, base + s, dtype=torch.int64, device=device)
        )
        nb = (s + block_size - 1) // block_size
        row = torch.arange(base_blk, base_blk + nb, dtype=torch.int32, device=device)
        if nb < max_blocks:
            row = torch.cat([row, row.new_zeros(max_blocks - nb)])
        block_rows.append(row.unsqueeze(0))
    slot_mapping = torch.cat(slot_parts)
    block_table = torch.cat(block_rows, dim=0)

    if backend_name != "FLASH_ATTN":
        raise RuntimeError(f"Backend {backend_name}: not supported by worker")

    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    scheduler_metadata = None
    try:
        from vllm.v1.attention.backends.fa_utils import (
            get_flash_attn_version,
            get_scheduler_metadata,
        )

        if get_flash_attn_version() == 3:
            _name, attn = harness["attn_layers"][0]
            scheduler_metadata = get_scheduler_metadata(
                batch_size=len(seq_lens),
                max_seqlen_q=int(max(query_lens)),
                max_seqlen_k=int(max(seq_lens)),
                num_heads_q=attn.num_heads,
                num_heads_kv=attn.num_kv_heads,
                headdim=attn.head_size,
                cache_seqlens=seq_lens_t,
                qkv_dtype=torch.bfloat16,
                cu_seqlens_q=query_start_loc,
                page_size=block_size,
                causal=True,
                num_splits=0,
            )
    except ImportError:
        pass

    metadata = FlashAttentionMetadata(
        num_actual_tokens=total,
        max_query_len=int(max(query_lens)),
        query_start_loc=query_start_loc,
        max_seq_len=int(max(seq_lens)),
        seq_lens=seq_lens_t,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
        scheduler_metadata=scheduler_metadata,
    )
    return metadata, slot_mapping, positions


def _run_steps(harness, steps, adapter_token_ids, num_requests, seq_capacity):
    """Run a SEQUENCE of forwards against one persistent KV cache.

    ``steps`` is a list of forwards; each is a list of per-request entries
    ``{"req": i, "tokens": [...], "cached": k}`` meaning: request ``i``
    contributes ``tokens`` as its query rows, with ``k`` tokens already in the
    cache (so ``seq_len = k + len(tokens)`` and ``positions`` run ``k..seq_len-1``).

    The cache is zeroed ONCE before the first step, never between steps — that
    persistence is the whole point: it is what lets a later chunk (or a decode
    token) attend back to a control token that is no longer in ``input_ids``.

    Returns a list per step of ``{req_index: [adapter indices for its query rows]}``.
    """
    device = harness["device"]
    if len(adapter_token_ids) != harness.get("layout_len"):
        _build_switch(harness, adapter_token_ids)
    switch = harness["switch"]

    from vllm.forward_context import ForwardContext, override_forward_context

    vllm_config = harness["vllm_config"]
    atok = torch.tensor(adapter_token_ids, dtype=torch.long, device=device)
    block_size = harness["block_size"]

    # Stable per-request block ranges for the whole run, sized for the largest
    # sequence each request will reach, so blocks never collide across steps.
    blocks_per_req = (seq_capacity + block_size - 1) // block_size
    block_base = [i * blocks_per_req for i in range(num_requests)]

    for kv in harness["kv_caches"].values():
        kv.zero_()

    saved = []
    for _name, attn in harness["attn_layers"]:
        saved.append((attn, attn.use_direct_call))
        attn.use_direct_call = True

    results = []
    try:
        for step in steps:
            req_ids = [e["req"] for e in step]
            query_lens = [len(e["tokens"]) for e in step]
            seq_lens = [e["cached"] + len(e["tokens"]) for e in step]
            flat = [t for e in step for t in e["tokens"]]
            input_ids = torch.tensor(flat, dtype=torch.long, device=device)

            metadata, slot_mapping, positions = _build_step_metadata(
                harness, query_lens, seq_lens, [block_base[i] for i in req_ids]
            )
            attn_metadata, slot_mapping_map = {}, {}
            for layer_name, _attn in harness["attn_layers"]:
                attn_metadata[layer_name] = metadata
                slot_mapping_map[layer_name] = slot_mapping

            forward_ctx = ForwardContext(
                no_compile_layers=vllm_config.compilation_config.static_forward_context,
                attn_metadata=attn_metadata,
                slot_mapping=slot_mapping_map,
            )
            with override_forward_context(forward_ctx):
                adapter_indices, _modified = switch.forward(
                    input_ids=input_ids,
                    adapter_token_ids=atok,
                    positions=positions,
                )
            flat_out = adapter_indices.cpu().tolist()
            per_req, off = {}, 0
            for rid, q in zip(req_ids, query_lens):
                per_req[str(rid)] = flat_out[off : off + q]
                off += q
            results.append(per_req)
    finally:
        for attn, old in saved:
            attn.use_direct_call = old

    return results


def _run_batch(harness, seqs, adapter_token_ids):
    """Run several sequences as ONE batched flat forward; return per-seq indices.

    Proves vLLM continuous batching: the flat stream is seqs concatenated, with
    per-request positions/metadata. Asserts nothing here — returns each request's
    adapter_indices slice so the test can compare against the same sequences run
    individually (which must match exactly: no cross-request contamination).
    """
    device = harness["device"]
    if len(adapter_token_ids) != harness.get("layout_len"):
        _build_switch(harness, adapter_token_ids)
    switch = harness["switch"]

    from vllm.forward_context import ForwardContext, override_forward_context

    vllm_config = harness["vllm_config"]
    seq_lens = [len(s) for s in seqs]
    flat = [t for s in seqs for t in s]
    input_ids = torch.tensor(flat, dtype=torch.long, device=device)
    atok = torch.tensor(adapter_token_ids, dtype=torch.long, device=device)

    for kv in harness["kv_caches"].values():
        kv.zero_()

    metadata, slot_mapping, positions = _build_batch_metadata(harness, seq_lens)
    attn_metadata, slot_mapping_map = {}, {}
    for layer_name, _attn in harness["attn_layers"]:
        attn_metadata[layer_name] = metadata
        slot_mapping_map[layer_name] = slot_mapping

    forward_ctx = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping_map,
    )

    saved = []
    for _name, attn in harness["attn_layers"]:
        saved.append((attn, attn.use_direct_call))
        attn.use_direct_call = True
    try:
        with override_forward_context(forward_ctx):
            adapter_indices, _modified = switch.forward(
                input_ids=input_ids,
                adapter_token_ids=atok,
                positions=positions,
            )
    finally:
        for attn, old in saved:
            attn.use_direct_call = old

    flat_out = adapter_indices.cpu().tolist()
    # Slice back into per-request lists.
    out, off = [], 0
    for n in seq_lens:
        out.append(flat_out[off : off + n])
        off += n
    return out


def _query_geometry(harness):
    switch = harness["switch"]
    info = {
        "switch_type": harness["switch_type"],
        "num_cache_layers": int(switch.num_cache_layers),
        "num_attn_layers": len(harness["attn_layers"]),
        "attn_layer_names": [name for name, _ in harness["attn_layers"]],
        "backend_name": harness["backend_name"],
    }
    return info


def _probe_attention(harness):
    """Smoke-test the auto-selected attention kernel on a tiny input.

    Surfaces FA kernel-image mismatches on the coded engine's two Attention
    heads before signaling ready.
    """
    try:
        _run(harness, seq=[0, 101, 0], adapter_token_ids=[101, 102])
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        hint = (
            "Auto-selected attention backend "
            f"{harness['backend_name']!r} crashed during the coded-engine "
            "startup smoke test. 'no kernel image is available' means the FA "
            "kernels were compiled for a different SM than this GPU."
        )
        return {"fatal": msg, "hint": hint, "backend_name": harness["backend_name"]}
    return None


def main():
    switch_type = sys.argv[1] if len(sys.argv) > 1 else "multi"
    try:
        harness = _setup(switch_type)
    except Exception as exc:
        msg = {
            "fatal": f"{type(exc).__name__}: {exc}",
            "hint": "Worker setup failed before attention probe.",
            "backend_name": "unknown",
        }
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()
        traceback.print_exc(file=sys.stderr)
        return

    probe_failure = _probe_attention(harness)
    if probe_failure is not None:
        sys.stdout.write(json.dumps(probe_failure) + "\n")
        sys.stdout.flush()
        return

    ready_msg = {
        "ready": True,
        "backend_name": harness["backend_name"],
        "switch_type": switch_type,
        "num_attn_layers": len(harness["attn_layers"]),
    }
    sys.stdout.write(json.dumps(ready_msg) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            command = req.get("command", "forward")
            if command == "query_geometry":
                resp = {"result": _query_geometry(harness)}
            elif command == "forward":
                result = _run(
                    harness,
                    seq=req["seq"],
                    adapter_token_ids=req["adapter_token_ids"],
                )
                resp = {"result": result}
            elif command == "forward_batch":
                result = _run_batch(
                    harness,
                    seqs=req["seqs"],
                    adapter_token_ids=req["adapter_token_ids"],
                )
                resp = {"result": result}
            elif command == "forward_steps":
                result = _run_steps(
                    harness,
                    steps=req["steps"],
                    adapter_token_ids=req["adapter_token_ids"],
                    num_requests=req["num_requests"],
                    seq_capacity=req["seq_capacity"],
                )
                resp = {"result": result}
            else:
                resp = {"error": f"Unknown command: {command}"}
        except Exception:
            resp = {"error": traceback.format_exc()}
            print(traceback.format_exc(), file=sys.stderr)

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    # Clean up static forward context entries so a later worker can reuse names.
    sfc = harness["vllm_config"].compilation_config.static_forward_context
    for layer_name, _attn in harness["attn_layers"]:
        sfc.pop(layer_name, None)


if __name__ == "__main__":
    main()
