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
import subprocess
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
#: How long to wait for the DEVICE itself (not its memory) to become acquirable by
#: a new CUDA context after an engine exits. See device_is_free / wait_device_free.
GPU_DEVICE_READY_TIMEOUT = float(os.environ.get("GS_GPU_DEVICE_READY_TIMEOUT", "60"))


def _nvsmi_free_total_bytes():
    """(free, total) bytes for the worker's visible GPU, via ``nvidia-smi``.

    CUDA-FREE on purpose -- NOT ``torch.cuda.mem_get_info``. ``gpu_mem_util`` runs
    in the worker *before* ``LLM()``, and ``LLM()`` spawns the ``EngineCore`` child
    that creates the real CUDA context. Under the host's NVIDIA
    ``Exclusive_Process`` compute mode only ONE process may hold a device's primary
    context at a time, so if this (the parent worker) process created a context
    first -- which ``torch.cuda.mem_get_info`` does: it flips
    ``torch.cuda.is_initialized()`` to True -- the ``EngineCore`` child's first CUDA
    op dies with ``cudaErrorDevicesUnavailable`` at ``init_device`` ->
    ``MemorySnapshot`` -> ``get_memory_info``. That is the real cluster-2 Cause B
    (the doc's "self-sustaining non-eager window" was a mis-diagnosis; the actual
    precondition is deterministic -- parent holds a context before the child spawns;
    minimal repro in git history / scratch). ``nvidia-smi`` reads NVML without a
    context, so the parent stays CUDA-free.

    Honours ``CUDA_VISIBLE_DEVICES`` by taking the FIRST visible index (the worker
    pins to one device before this runs). Returns ``None`` on any failure so the
    caller falls back to letting vLLM size itself.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    index = "0"
    if visible is not None:
        entries = [v.strip() for v in visible.split(",") if v.strip() != ""]
        if not entries:
            return None  # device explicitly hidden -- no GPU to measure
        index = entries[0]
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        free_mib, total_mib = (int(x.strip()) for x in out.splitlines()[0].split(","))
    except Exception:
        return None
    return free_mib * 2**20, total_mib * 2**20


def gpu_mem_util(ceiling: float = GPU_MEM_UTIL, floor: float = GPU_MEM_UTIL_FLOOR):
    """90% of the free fraction, capped at ``ceiling``.

    Reads free/total memory via ``nvidia-smi`` (:func:`_nvsmi_free_total_bytes`),
    NOT ``torch.cuda`` -- this runs in the worker before ``LLM()`` spawns the
    ``EngineCore`` child, and creating a CUDA context here would block that child
    under ``Exclusive_Process`` (see :func:`_nvsmi_free_total_bytes`).

    Returns ``None`` when there is no visible GPU to measure (or the query fails),
    so a caller can leave the kwarg unset and let vLLM pick.

    Raises ``SystemExit`` rather than returning a doomed value when the result is
    below ``floor``: at that point something else is holding the card and a test
    failure would be the wrong diagnosis.
    """
    ft = _nvsmi_free_total_bytes()
    if ft is None:
        return None
    free, total = ft
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


def device_is_free() -> bool:
    """True if no compute process is holding a visible GPU, per ``nvidia-smi``.

    Queried through ``nvidia-smi`` on purpose, NOT a ``torch.cuda`` probe: this
    runs in the **parent** pytest process, and under the host's
    ``Exclusive_Process`` compute mode (see ``docs/FIXING_VLLM.md`` cluster 2) only
    one process may hold a device's primary context at a time. A
    ``torch.zeros(1, device=...)`` here would create exactly the parent context
    that then blocks the next ``EngineCore`` child -- the very failure we are
    guarding against. So we observe the device from the outside instead.

    Honours ``CUDA_VISIBLE_DEVICES``. Conservative: any error (no ``nvidia-smi``,
    parse failure) returns ``True`` so a probe failure never wedges a run -- the
    engine start itself remains the real gate.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return True

    # Map visible indices -> UUIDs so CUDA_VISIBLE_DEVICES is respected. If that
    # query fails, fall back to "any running compute app means busy".
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed_uuids = None
    if visible is not None:
        allowed = {v.strip() for v in visible.split(",") if v.strip() != ""}
        try:
            idx_out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            allowed_uuids = set()
            for line in idx_out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] in allowed:
                    allowed_uuids.add(parts[1])
        except Exception:
            allowed_uuids = None

    for line in out.splitlines():
        uuid = line.strip()
        if not uuid:
            continue
        if allowed_uuids is not None and uuid not in allowed_uuids:
            continue
        return False
    return True


def wait_device_free(*, timeout: float = GPU_DEVICE_READY_TIMEOUT) -> None:
    """Block until no compute process holds a visible GPU, or ``timeout`` elapses.

    Complements the memory-reclaim wait in :func:`shutdown_llm`. The cluster-2
    evidence (``docs/FIXING_VLLM.md``) is a device that reports its memory fully
    free (~78.7/79.2 GiB) yet still fails a *new* engine's first host->device copy
    with ``cudaErrorDevicesUnavailable``: the exiting ``EngineCore`` child has
    released memory but not yet relinquished the device's primary context. Memory
    free != device acquirable. We wait for the child to actually drop off
    ``nvidia-smi``'s compute-app list before letting the next engine start.

    A timeout prints and returns rather than raising: like the memory wait, a
    spurious red is worse than a bounded wait, and the engine start is still the
    authoritative gate.
    """
    deadline = time.monotonic() + timeout
    while True:
        if device_is_free():
            return
        if time.monotonic() >= deadline:
            print(
                f"  WARNING: a compute process still holds the device after "
                f"{timeout:g}s; continuing and letting the engine start rule on it"
            )
            return
        time.sleep(1.0)


def shutdown_llm(
    llm,
    *,
    target: float = GPU_MEM_RECLAIM_TARGET,
    timeout: float = GPU_MEM_RECLAIM_TIMEOUT,
    device_ready_timeout: float = GPU_DEVICE_READY_TIMEOUT,
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

    After the memory wait, this also waits for the **device** itself to become
    acquirable again (:func:`wait_device_free`): under ``Exclusive_Process`` the
    exiting ``EngineCore`` child can return its memory before it relinquishes the
    device's primary context, so the next engine's first CUDA op fails with
    ``cudaErrorDevicesUnavailable`` even on a card that reports fully free. See
    ``docs/FIXING_VLLM.md`` cluster 2.
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
            break
        if time.monotonic() >= deadline:
            print(
                f"  WARNING: {free / 2**30:.1f}/{total / 2**30:.1f} GiB free "
                f"({frac:.2f}) still below {target:g} after {timeout:g}s; "
                "continuing and letting gpu_mem_util rule on it"
            )
            break
        time.sleep(1.0)

    # Memory can be back while the device is momentarily unacquirable by a NEW
    # CUDA context -- the device-availability race in cluster 2. Wait that out too.
    wait_device_free(timeout=device_ready_timeout)
