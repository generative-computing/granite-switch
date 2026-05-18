# SPDX-License-Identifier: Apache-2.0
"""Force a compatible attention backend in vLLM tests on non-Hopper GPUs.

The CI/test environment has a `vllm-flash-attn` build that ships only Hopper
(SM 9.0) kernels. On Ampere/Ada GPUs vLLM still auto-selects ``FLASH_ATTN``
because validation passes (the build is present and importable), but the very
first kernel launch crashes with::

    CUDA error (.../hopper/flash_api.cpp:697):
        no kernel image is available for execution on the device

That C-level abort kills the worker process before any Python handler runs,
which is what produces the cascade of ``BrokenPipeError``s in the long-lived
SingleSwitch worker and the opaque "Worker died unexpectedly" message in the
short-lived per-class subprocess tests.

Until the environment is fixed (rebuild ``vllm-flash-attn`` with
``TORCH_CUDA_ARCH_LIST`` covering the runtime GPU), force a backend whose
kernels are universally compiled. ``FLASHINFER`` validates cleanly for
head_dim=64 / bf16 on SM 8.0 and is the default fallback. ``TRITON_ATTN`` and
``FLEX_ATTENTION`` are also valid choices and can be selected via env.

Activation rules
----------------
- Only fires when ``torch.cuda.get_device_capability() < (9, 0)``. On Hopper+,
  the FA3 build matches the GPU and we leave selection alone so the FA3 path
  remains under test.
- Override the gate (e.g. on a CI runner with a known-broken FA build on
  Hopper) by setting ``GS_TEST_FORCE_ATTN_BACKEND=1``.
- Pick a different backend with ``GS_TEST_ATTN_BACKEND=TRITON_ATTN`` (or
  ``FLEX_ATTENTION``). Default is ``FLASHINFER``.

The helper is a no-op outside vLLM/CUDA so importing it from CPU-only test
contexts doesn't break collection.
"""

from __future__ import annotations

import os
import sys
from typing import Callable


def _should_force(force_flag: str | None, capability: tuple[int, int] | None) -> bool:
    if force_flag and force_flag.lower() not in ("0", "false", ""):
        return True
    if capability is None:
        return False
    return capability < (9, 0)


def force_compatible_attn_backend() -> Callable[[], None] | None:
    """Monkey-patch vLLM's attention selector to a fixed, validated backend.

    Returns a restorer callable that undoes the patch, or ``None`` if no patch
    was applied (Hopper+, or vLLM not importable, or no CUDA).
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    capability = torch.cuda.get_device_capability()
    force_flag = os.environ.get("GS_TEST_FORCE_ATTN_BACKEND")
    if not _should_force(force_flag, capability):
        return None

    backend_name = os.environ.get("GS_TEST_ATTN_BACKEND", "FLASHINFER")

    try:
        from vllm.v1.attention import selector as _sel
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return None

    try:
        backend_enum = AttentionBackendEnum[backend_name]
    except KeyError as exc:
        valid = ", ".join(b.name for b in AttentionBackendEnum)
        raise ValueError(
            f"GS_TEST_ATTN_BACKEND={backend_name!r} is not a known vLLM "
            f"attention backend. Valid: {valid}"
        ) from exc

    forced_cls = backend_enum.get_class()

    def _patched_get_attn_backend(*args, **kwargs):
        return forced_cls

    # vLLM's Attention layer and friends do
    #   from vllm.v1.attention.selector import get_attn_backend
    # at module import time, so patching only `selector.get_attn_backend`
    # leaves their already-bound local references untouched. Patch each
    # known consumer's module namespace too. Missing modules (older/newer
    # vLLM) are ignored — at least one of these will be the live one.
    consumer_modules = [
        "vllm.v1.attention.selector",
        "vllm.model_executor.layers.attention.attention",
        "vllm.model_executor.layers.attention.chunked_local_attention",
        "vllm.model_executor.layers.attention.cross_attention",
        "vllm.model_executor.layers.attention.encoder_only_attention",
        "vllm.model_executor.layers.attention.mla_attention",
        "vllm.model_executor.layers.attention.static_sink_attention",
    ]

    import importlib

    originals: list[tuple[object, object]] = []
    for mod_name in consumer_modules:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(mod, "get_attn_backend"):
            originals.append((mod, mod.get_attn_backend))
            mod.get_attn_backend = _patched_get_attn_backend

    if not originals:
        # Nothing to patch — vLLM internals must have moved.
        sys.stderr.write(
            "[vllm_attn_backend] WARNING: no get_attn_backend symbols found "
            "in known vLLM modules; backend override is a no-op. The test "
            "will likely still hit the FA3 kernel-image crash.\n"
        )
        sys.stderr.flush()
        return None

    sys.stderr.write(
        f"[vllm_attn_backend] GPU compute capability {capability} < (9, 0): "
        f"forcing {backend_name} attention backend in "
        f"{len(originals)} vLLM module(s) (set GS_TEST_FORCE_ATTN_BACKEND=0 "
        f"to disable, GS_TEST_ATTN_BACKEND=<NAME> to pick a different one).\n"
    )
    sys.stderr.flush()

    def _restore() -> None:
        for mod, original in originals:
            mod.get_attn_backend = original

    return _restore
