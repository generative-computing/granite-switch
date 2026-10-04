# SPDX-License-Identifier: Apache-2.0
"""Pick a ``gpu_memory_utilization`` that fits the memory actually free right now.

vLLM measures the fraction against TOTAL device memory and refuses to start when
that exceeds what is free::

    Free memory on device cuda:0 (4.6/79.25 GiB) ... is less than desired GPU
    memory utilization (0.92, 72.91 GiB)   -- vllm/v1/worker/utils.py

So a fixed value — including vLLM's own default — only works on an idle card, and
these suites do not have one: they start several engines in a row and the previous
engine's subprocess is often still releasing, which is enough to trip the guard
in-pod with the second GPU completely idle. The guard is byte-identical back to
0.26, so this is not a version problem.
"""

import gc
import os
import time

#: Ceiling, i.e. what to ask for on an empty card.
GPU_MEM_UTIL = float(os.environ.get("GS_GPU_MEM_UTIL", "0.85"))
#: Below this the card is too full to be worth starting on -- a 3B model plus its
#: KV cache needs roughly a tenth of an 80 GiB card, so this is generous.
GPU_MEM_UTIL_FLOOR = float(os.environ.get("GS_GPU_MEM_UTIL_FLOOR", "0.15"))
#: Free fraction that counts as "the previous engine is gone". Not 1.0: the parent
#: holds a small CUDA context of its own once it has called ``mem_get_info``.
GPU_MEM_RECLAIM_TARGET = float(os.environ.get("GS_GPU_MEM_RECLAIM_TARGET", "0.80"))
#: How long to wait for that. Generous, because the alternative to waiting is a
#: spurious red, and a shut-down engine normally releases within a few seconds.
GPU_MEM_RECLAIM_TIMEOUT = float(os.environ.get("GS_GPU_MEM_RECLAIM_TIMEOUT", "120"))


def gpu_mem_util(ceiling: float = GPU_MEM_UTIL, floor: float = GPU_MEM_UTIL_FLOOR):
    """90% of the free fraction, capped at ``ceiling``.

    Returns ``None`` without CUDA, so a caller can leave the kwarg unset and let
    vLLM pick (there is no device to measure, and no engine to start either).

    Raises ``SystemExit`` rather than returning a doomed value when the result is
    below ``floor``: at that point something else is holding the card and a test
    failure would be the wrong diagnosis.
    """
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    util = min(ceiling, 0.9 * free / total)
    print(
        f"  GPU memory: {free / 2**30:.1f}/{total / 2**30:.1f} GiB free -> "
        f"gpu_memory_utilization={util:.3f} (ceiling {ceiling:g})"
    )
    if util < floor:
        raise SystemExit(
            f"FATAL: only {free / 2**30:.1f} GiB free on this device, which leaves "
            f"gpu_memory_utilization={util:.3f} below the {floor:g} floor. Another "
            "process is holding the card; this is not a test failure."
        )
    return util


def shutdown_llm(
    llm,
    *,
    target: float = GPU_MEM_RECLAIM_TARGET,
    timeout: float = GPU_MEM_RECLAIM_TIMEOUT,
) -> None:
    """Stop an ``LLM``'s EngineCore and wait for the device memory to come back.

    ``del llm; gc.collect(); torch.cuda.empty_cache()`` does not do this, which is
    why the suites that start engines back to back were tripping ``gpu_mem_util``'s
    floor on a card with nothing else running:

    * The weights and KV cache belong to the **EngineCore child process** (v1 forks
      one, and these runners set ``VLLM_WORKER_MULTIPROC_METHOD=spawn``).
      ``empty_cache()`` drains the *parent's* allocator, which never held them.
    * Teardown is registered as a ``weakref.finalize``, so ``del`` schedules it at
      collection time rather than performing it, and nothing waits for the child.

    ``engine_core.shutdown()`` is the synchronous path: it chains to
    ``CoreEngineProcManager.shutdown``, which terminates *and joins* the children.
    Its ``shutdown(timeout=None)`` signature is byte-identical on 0.26 - 0.30, and
    it is reached defensively here so a future rename degrades to today's
    behaviour rather than an ``AttributeError``.

    The caller's ``llm`` reference does not have to be dropped before calling:
    once the child has exited the driver owns the reclaim, so refcounts in this
    process no longer gate it.

    A timeout prints and returns instead of raising. ``gpu_mem_util`` is the one
    place that decides a card is too full to use, and two components disagreeing
    about that is worse than waiting a bounded time and letting it rule.
    """
    engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if shutdown is not None:
        shutdown()

    gc.collect()

    import torch

    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()

    deadline = time.monotonic() + timeout
    while True:
        free, total = torch.cuda.mem_get_info()
        frac = free / total
        if frac >= target:
            print(
                f"  GPU memory reclaimed: {free / 2**30:.1f}/{total / 2**30:.1f} GiB "
                f"free ({frac:.2f} >= {target:g})"
            )
            return
        if time.monotonic() >= deadline:
            print(
                f"  WARNING: {free / 2**30:.1f}/{total / 2**30:.1f} GiB free "
                f"({frac:.2f}) still below {target:g} after {timeout:g}s; "
                "continuing and letting gpu_mem_util rule on it"
            )
            return
        time.sleep(1.0)
