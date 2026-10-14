# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: run ONE vLLM engine and emit teacher-forced prompt logprobs.

Invoked as a fresh process per engine so the **pytest process never creates a
CUDA context**::

    python _vllm_logprobs_worker.py run \
        --model <dir> --inputs <inputs.json> --out <logprobs.pt>

This is load-bearing under the GPU host's NVIDIA **Exclusive_Process** compute
mode: only one OS process may hold a device's primary CUDA context at a time, so
if the pytest process touched ``torch.cuda`` (even a ``mem_get_info`` probe) it
would own the device and every ``EngineCore`` child it spawned afterward would die
with ``cudaErrorDevicesUnavailable``. See ``docs/FIXING_VLLM.md`` cluster 2.

The engine lives entirely inside this short-lived process. Because the process
exits after a single engine there is no back-to-back-engine handoff within it, so
the ``gpu_mem_util`` / ``shutdown_llm`` helpers (which DO touch ``torch.cuda``) are
called HERE -- in the process that legitimately holds the context -- rather than in
the pytest process.

Contract:
- ``inputs.json``: ``{"input_ids": [int, ...], "vocab_size": int,
  "llm_kwargs": {...}}``. ``llm_kwargs`` is forwarded verbatim to ``LLM(...)`` and
  carries exactly what the caller passed (e.g. ``max_model_len``, ``enforce_eager``,
  ``enable_prefix_caching``, ``gpu_memory_utilization``).
- ``--out``: a ``torch.save`` of the dense ``[seq_len-1, vocab_size]`` CPU logprob
  tensor, loadable back in the pytest process without CUDA.
"""

import argparse
import json
import os
import sys

# Make tests.shared importable when run as a bare subprocess (cwd-independent),
# mirroring tests/vllm/_generation_equivalence_worker.py.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _pin_single_visible_device():
    """Narrow ``CUDA_VISIBLE_DEVICES`` to a single device for this worker.

    Each worker runs exactly one ``tensor_parallel_size=1`` engine, so it never
    needs more than one GPU. Leaving two devices visible makes vLLM's distributed
    init (``init_worker_distributed_environment`` -> NCCL) set up across both even
    at world_size=1, which widens the device-acquisition surface -- and under the
    host's NVIDIA Exclusive_Process compute mode that is exactly where engine init
    intermittently dies with ``cudaErrorDevicesUnavailable`` (see
    ``docs/FIXING_VLLM.md`` cluster 2). Pinning to one device removes the
    cross-device NCCL step and halves the surface.

    Honours an existing allowlist: picks the FIRST entry of ``CUDA_VISIBLE_DEVICES``
    (so a harness that already restricts/orders devices is respected), or ``"0"``
    if unset. Must run BEFORE any ``torch``/``vllm`` import so no context has formed
    yet. ``GS_LOGPROBS_NO_PIN=1`` opts out (e.g. to A/B this behaviour).
    """
    if os.environ.get("GS_LOGPROBS_NO_PIN"):
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        chosen = "0"
    else:
        entries = [v.strip() for v in visible.split(",") if v.strip() != ""]
        if len(entries) <= 1:
            return  # already a single device (or none) -- nothing to narrow
        chosen = entries[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    print(f"  pinned CUDA_VISIBLE_DEVICES={chosen} (was {visible!r})")


def cmd_run(args):
    """Build one vLLM engine, capture dense prompt logprobs, save to --out."""
    # Pin to one GPU BEFORE importing torch/vllm (see _pin_single_visible_device).
    _pin_single_visible_device()
    # vLLM v1 forks a subprocess for EngineCore; spawn avoids inheriting any CUDA
    # state. (This process has no CUDA context yet -- the engine child creates it.)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch
    from tests.shared.vllm_equivalence import extract_logprobs_tensor
    from tests.shared.vllm_gpu_mem import gpu_mem_util, shutdown_llm

    register_granite_switch()

    with open(args.inputs) as f:
        spec = json.load(f)
    input_ids_list = spec["input_ids"]
    vocab_size = spec["vocab_size"]
    llm_kwargs = dict(spec.get("llm_kwargs") or {})

    # Allow requesting logprobs for all vocab tokens.
    llm_kwargs.setdefault("max_logprobs", vocab_size)

    # gpu_memory_utilization: vLLM's default is a fraction of TOTAL memory, which
    # assumes an idle card. These suites start engines back to back (one worker
    # process each) and the previous one may still be releasing, so size against
    # what is actually free. setdefault, so an explicit caller value still wins.
    # See tests/shared/vllm_gpu_mem.py. Safe to probe torch.cuda here: this is the
    # process that will hold the engine's context anyway.
    mem_util = gpu_mem_util()
    if mem_util is not None:
        llm_kwargs.setdefault("gpu_memory_utilization", mem_util)

    print(f"Creating vLLM LLM for {args.model} with kwargs {sorted(llm_kwargs)}...")
    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        **llm_kwargs,
    )

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=vocab_size,
    )
    prompt = TokensPrompt(prompt_token_ids=input_ids_list)
    outputs = llm.generate(prompt, sampling_params=sampling_params)

    logprobs_tensor = extract_logprobs_tensor(outputs, vocab_size)
    torch.save(logprobs_tensor, args.out)
    print(f"  saved logprobs {tuple(logprobs_tensor.shape)} to {args.out}")

    # Synchronous shutdown + device/memory reclaim, in the process that owns the
    # context. The process then exits, which under Exclusive_Process is what
    # actually releases the device for the next worker.
    shutdown_llm(llm)
    del llm
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run", help="Run one engine, emit dense prompt logprobs")
    p_run.add_argument("--model", required=True, help="Model dir or HF name to load")
    p_run.add_argument(
        "--inputs", required=True, help="JSON: {input_ids, vocab_size, llm_kwargs}"
    )
    p_run.add_argument("--out", required=True, help="Path to torch.save the tensor to")

    args = parser.parse_args()
    if args.mode == "run":
        return cmd_run(args)
    raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
