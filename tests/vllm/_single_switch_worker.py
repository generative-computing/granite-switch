# SPDX-License-Identifier: Apache-2.0
"""Long-lived subprocess worker for vLLM SingleSwitch tests.

Protocol (JSON-line over stdin/stdout):
  Startup: prints {"ready": true, "backend_name": "..."} to stdout
       OR  prints {"fatal": "...", "hint": "...", "backend_name": "..."} and exits
           when the auto-selected attention backend's kernel image is incompatible
           with this GPU (e.g. FA3 Hopper-only kernels on a non-Hopper card). The
           parent reads this and converts it into one clear pytest failure instead
           of letting hundreds of subsequent tests cascade into BrokenPipeErrors.
  Request: {"seq": [...], "num_adapters": N, "control_token_gain": G}
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


def _setup():
    """Create VllmConfig, SingleSwitch, KV cache. Return harness dict."""
    # Redirect fd 1 → fd 2 at the OS level during setup so that C-level
    # output (CUDA, FA3 kernel init) doesn't contaminate the JSON-line
    # protocol on stdout.  Python-level sys.stdout redirect alone is not
    # enough — native code writes directly to fd 1.
    _saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    _saved_fd1 = os.dup(1)
    os.dup2(2, 1)

    from vllm.config import VllmConfig, set_current_vllm_config
    from granite_switch.vllm.switch.single import SingleSwitch

    BLOCK_SIZE = 16
    MAX_TOKENS = 131_072
    NUM_ADAPTERS = 32
    ADAPTER_TOKEN_IDS_LIST = list(range(1000, 1000 + NUM_ADAPTERS))

    # Mock config with realistic backbone geometry (GQA: 4Q/2KV, head_dim=64)
    # so unit tests exercise the multi-head path, not the fallback.
    mock_config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=2,
        expanded_head_dim=64,
        attention_multiplier=0.125,
    )

    device = torch.device("cuda")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    try:
        vllm_config = VllmConfig()
        ensure_distributed(vllm_config)
        with set_current_vllm_config(vllm_config):
            switch = SingleSwitch(
                num_adapters=NUM_ADAPTERS,
                vllm_config=vllm_config,
                control_token_gain=15.0,
                config=mock_config,
            )
    finally:
        torch.set_default_dtype(old_dtype)

    attn = switch.attn
    attn.kv_cache_torch_dtype = torch.bfloat16
    layer_name = "switch.layers.0"
    backend_name = attn.attn_backend.get_name()

    num_blocks = (MAX_TOKENS + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    cache_shape = attn.attn_backend.get_kv_cache_shape(
        num_blocks, BLOCK_SIZE, switch.num_kv_heads, switch.head_dim,
    )
    kv_cache = torch.zeros(cache_shape, device=device, dtype=torch.bfloat16)
    attn.kv_cache = kv_cache

    # Restore real stdout (both Python-level and OS fd 1) for JSON protocol
    os.dup2(_saved_fd1, 1)
    os.close(_saved_fd1)
    sys.stdout = _saved_stdout

    return {
        "switch": switch,
        "vllm_config": vllm_config,
        "kv_cache": kv_cache,
        "device": device,
        "layer_name": layer_name,
        "backend_name": backend_name,
        "block_size": BLOCK_SIZE,
        "adapter_token_ids_list": ADAPTER_TOKEN_IDS_LIST,
    }


def _build_metadata(harness, seq_len):
    """Build FlashAttention metadata for a single-sequence prefill."""
    device = harness["device"]
    block_size = harness["block_size"]
    backend_name = harness["backend_name"]

    slot_mapping = torch.arange(seq_len, dtype=torch.int64, device=device)
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(
        num_blocks_needed, dtype=torch.int32, device=device,
    ).unsqueeze(0)
    query_start_loc = torch.tensor(
        [0, seq_len], dtype=torch.int32, device=device,
    )
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    if backend_name == "FLASH_ATTN":
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        # scheduler_metadata is FA3-only — passing it on FA2 (Ampere/A100)
        # forces FA3 kernel dispatch and crashes with "no kernel image
        # is available". Only compute it when get_flash_attn_version() == 3
        # (Hopper SM90+).
        scheduler_metadata = None
        try:
            from vllm.v1.attention.backends.fa_utils import (
                get_flash_attn_version,
                get_scheduler_metadata,
            )
            if get_flash_attn_version() == 3:
                switch = harness["switch"]
                scheduler_metadata = get_scheduler_metadata(
                    batch_size=1,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                    num_heads_q=switch.num_heads,
                    num_heads_kv=switch.num_kv_heads,
                    headdim=switch.head_dim,
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


def _run(harness, seq, num_adapters, control_token_gain):
    """Execute SingleSwitch.forward and return result as list."""
    from vllm.forward_context import ForwardContext, override_forward_context

    switch = harness["switch"]
    vllm_config = harness["vllm_config"]
    kv_cache = harness["kv_cache"]
    device = harness["device"]
    layer_name = harness["layer_name"]
    adapter_token_ids_list = harness["adapter_token_ids_list"]

    seq_len = len(seq)
    kv_cache.zero_()

    orig_gain = switch.control_token_gain
    orig_effective_gain = switch.effective_gain
    orig_num_adapters = switch.num_adapters
    switch.control_token_gain = control_token_gain
    switch.effective_gain = control_token_gain / switch.scaling
    switch.num_adapters = num_adapters

    input_ids = torch.tensor(seq, dtype=torch.long, device=device)
    adapter_token_ids = torch.tensor(
        adapter_token_ids_list[:num_adapters], dtype=torch.long, device=device,
    )

    metadata, slot_mapping = _build_metadata(harness, seq_len)

    forward_ctx = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        attn_metadata={layer_name: metadata},
        slot_mapping={layer_name: slot_mapping},
    )

    old_direct = switch.attn.use_direct_call
    switch.attn.use_direct_call = True

    try:
        with override_forward_context(forward_ctx):
            result = switch.forward(
                input_ids=input_ids,
                adapter_token_ids=adapter_token_ids,
            )
    finally:
        switch.attn.use_direct_call = old_direct
        switch.control_token_gain = orig_gain
        switch.effective_gain = orig_effective_gain
        switch.num_adapters = orig_num_adapters

    return result.cpu().tolist()


def _query_geometry(harness):
    """Return switch geometry and cache info for infrastructure tests."""
    switch = harness["switch"]
    kv_cache = harness["kv_cache"]
    return {
        "num_heads": int(switch.num_heads),
        "num_kv_heads": int(switch.num_kv_heads),
        "head_dim": int(switch.head_dim),
        "scaling": float(switch.scaling),
        "effective_gain": float(switch.effective_gain),
        "control_token_gain": float(switch.control_token_gain),
        "num_adapters": int(switch.num_adapters),
        "kv_cache_shape": list(kv_cache.shape),
    }


def _probe_attention(harness):
    """Run a 1-token forward to verify the auto-selected attention kernel is usable.

    vLLM picks FLASH_ATTN by default. If the installed vllm-flash-attn was built
    only for Hopper (SM 9.0) but this GPU is a different architecture, the very
    first kernel launch raises "no kernel image is available for execution on
    the device" — and in some builds this kills the worker process at the C
    layer before any Python handler runs. Catching it here, on a tiny synthetic
    input, lets us emit a structured fatal message before signaling ready.

    Returns None on success, or a dict {"fatal": ..., "hint": ...} on failure.
    """
    try:
        _run(harness, seq=[0], num_adapters=1, control_token_gain=15.0)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        hint = (
            "Auto-selected attention backend "
            f"{harness['backend_name']!r} crashed during the startup smoke "
            "test. If 'no kernel image is available' appears above, the FA "
            "kernels in this venv were compiled for a different SM than the "
            "runtime GPU. Common cause: a stale standalone 'vllm-flash-attn' "
            "PyPI package shadowing vLLM's bundled FA kernels — try "
            "`pip uninstall vllm-flash-attn` and re-run."
        )
        return {"fatal": msg, "hint": hint, "backend_name": harness["backend_name"]}
    return None


def main():
    try:
        harness = _setup()
    except Exception as exc:
        # Setup itself blew up — surface it on stdout so the parent can show a
        # clean failure instead of "Worker failed to start: <empty>".
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

    # Signal ready
    ready_msg = {"ready": True, "backend_name": harness["backend_name"]}
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
                    num_adapters=req.get("num_adapters", 32),
                    control_token_gain=req.get("control_token_gain", 15.0),
                )
                resp = {"result": result}
            else:
                resp = {"error": f"Unknown command: {command}"}
        except Exception:
            resp = {"error": traceback.format_exc()}
            print(traceback.format_exc(), file=sys.stderr)

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    # Clean up
    sfc = harness["vllm_config"].compilation_config.static_forward_context
    sfc.pop(harness["layer_name"], None)


if __name__ == "__main__":
    main()
