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

import os

#: Ceiling, i.e. what to ask for on an empty card.
GPU_MEM_UTIL = float(os.environ.get("GS_GPU_MEM_UTIL", "0.85"))
#: Below this the card is too full to be worth starting on -- a 3B model plus its
#: KV cache needs roughly a tenth of an 80 GiB card, so this is generous.
GPU_MEM_UTIL_FLOOR = float(os.environ.get("GS_GPU_MEM_UTIL_FLOOR", "0.15"))


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
