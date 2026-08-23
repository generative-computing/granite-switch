# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: PP=2 vs PP=1 generation equivalence for the SWITCH kernel.

The Granite Switch SWITCH kernel relies on a recompute-per-rank contract under
pipeline parallelism: only the first PP rank runs the switch and ships the
token-leading ``adapter_indices`` across the stage boundary; every later rank
*recomputes* its per-module LoRA kernel metadata (bitmasks + remapped indices)
locally from that tensor. This worker verifies that contract end-to-end —
greedy generation under PP=2 must match PP=1 token-for-token, with an adapter
active so the kernel's delta path runs on every rank.

Both arms load the SAME pre-composed checkpoint; pipeline parallelism is a
runtime ``LLM(pipeline_parallel_size=N)`` argument, not a checkpoint property.
The embedded adapters load as plain weights — no vLLM LoRA flags, no LoRARequest.

Two modes, each invoked as a separate subprocess so only one vLLM engine is ever
resident on GPU at a time::

    python worker.py run     --model <id> --work-dir <dir> --pp-size <N> --tag <tag>
    python worker.py compare --work-dir <dir>

**run**: builds a deterministic token prompt with one control token injected
early (fires a nonzero adapter index), loads the model in vLLM at the requested
pipeline_parallel_size, runs greedy generation, and saves the generated token
IDs to ``<work-dir>/<tag>.json``.

**compare**: loads ``pp1.json`` / ``pp2.json`` and checks token-for-token match.
"""

import argparse
import faulthandler
import gc
import json
import os
import sys
import traceback

# Number of prompt tokens and the position the control token is injected at.
PROMPT_LEN = 16
CONTROL_POS = 2
MAX_NEW_TOKENS = 32
GENERATE_TIMEOUT_SECONDS = 300


def _log(message: str) -> None:
    print(f"PP_EQUIVALENCE_PHASE {message}", flush=True)


def _build_prompt_ids(adapter_token_ids, vocab_size):
    """Deterministic prompt of benign low IDs with one control token early.

    The control token at CONTROL_POS makes the switch compute a nonzero adapter
    index, so the SWITCH kernel's delta path runs on every PP rank (the whole
    point of the test). Fill tokens stay well below any control-token range.
    """
    import torch

    torch.manual_seed(42)
    max_fill = min(vocab_size, 1000)
    prompt_ids = torch.randint(1, max_fill, (PROMPT_LEN,)).tolist()
    # Fire the first adapter.
    prompt_ids[CONTROL_POS] = int(adapter_token_ids[0])
    return prompt_ids


# ── run mode ──────────────────────────────────────────────────────


def cmd_run(args):
    """Load the model in vLLM at a given PP size, greedily generate, save tokens."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # vllm22 + CUDA12/13 FlashInfer sampler mismatch: force the native sampler so
    # greedy decoding is deterministic and identical across PP sizes.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    model_path = args.model
    work_dir = args.work_dir
    pp_size = args.pp_size
    tag = args.tag

    _log(f"config_load_start model={model_path}")
    config = AutoConfig.from_pretrained(model_path)
    adapter_token_ids = config.adapter_token_ids
    vocab_size = config.vocab_size
    assert adapter_token_ids, (
        "model config has no adapter_token_ids; cannot activate an adapter"
    )
    _log(f"config_load_done vocab={vocab_size} adapters={len(adapter_token_ids)}")

    prompt_ids = _build_prompt_ids(adapter_token_ids, vocab_size)
    _log(f"prompt_ready len={len(prompt_ids)} ctrl_pos={CONTROL_POS}")

    _log(f"llm_init_start pp_size={pp_size}")
    llm = LLM(
        model=model_path,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        distributed_executor_backend="mp",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.4,
    )
    _log("llm_init_done")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        ignore_eos=True,
    )

    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    _log("generate_start")
    faulthandler.dump_traceback_later(
        GENERATE_TIMEOUT_SECONDS,
        file=sys.stderr,
        exit=True,
    )
    try:
        outputs = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
    finally:
        faulthandler.cancel_dump_traceback_later()
    _log("generate_done")

    generated_ids = list(outputs[0].outputs[0].token_ids)
    _log(f"generated {len(generated_ids)} tokens: {generated_ids[:10]}...")

    output_path = os.path.join(work_dir, f"{tag}.json")
    with open(output_path, "w") as f:
        json.dump({"token_ids": generated_ids, "pp_size": pp_size}, f)
    _log(f"saved {output_path}")

    del llm
    gc.collect()
    return 0


# ── compare mode ──────────────────────────────────────────────────


def cmd_compare(args):
    """Load pp1.json / pp2.json and assert token-for-token match."""
    work_dir = args.work_dir
    pp1_path = os.path.join(work_dir, "pp1.json")
    pp2_path = os.path.join(work_dir, "pp2.json")

    with open(pp1_path) as f:
        pp1_ids = json.load(f)["token_ids"]
    with open(pp2_path) as f:
        pp2_ids = json.load(f)["token_ids"]

    print(f"  pp1 tokens ({len(pp1_ids)}): {pp1_ids}")
    print(f"  pp2 tokens ({len(pp2_ids)}): {pp2_ids}")

    if len(pp1_ids) != len(pp2_ids):
        print(f"\nFAIL: length mismatch: pp1={len(pp1_ids)}, pp2={len(pp2_ids)}")
        return 1

    for i, (a, b) in enumerate(zip(pp1_ids, pp2_ids)):
        if a != b:
            print(
                f"\nFAIL: first divergence at position {i}: pp1={a}, pp2={b}\n"
                "  PP=2 must recompute per-rank LoRA kernel metadata identically "
                "to PP=1; a mismatch means the recompute-per-rank contract is broken."
            )
            return 1

    print(
        f"\nPASS: PP=2 matches PP=1 token-for-token [{len(pp1_ids)} tokens] — "
        "per-rank metadata recompute is correct."
    )
    return 0


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run", help="Load model at a PP size, generate tokens")
    p_run.add_argument("--model", required=True, help="Model name or path to load")
    p_run.add_argument(
        "--work-dir", required=True, help="Working directory for outputs"
    )
    p_run.add_argument(
        "--pp-size", type=int, required=True, help="pipeline_parallel_size"
    )
    p_run.add_argument("--tag", required=True, help="Output tag (pp1 or pp2)")

    p_compare = sub.add_parser("compare", help="Compare pp1.json vs pp2.json")
    p_compare.add_argument(
        "--work-dir", required=True, help="Working directory with pp1/pp2 json"
    )

    args = parser.parse_args()

    if args.mode == "run":
        return cmd_run(args)
    elif args.mode == "compare":
        return cmd_compare(args)
    return 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

    # Isolated from pytest to contain vLLM multiprocessing state. Avoid hanging in
    # interpreter teardown if a vLLM helper process/thread survives the assertion.
    os._exit(exit_code)
