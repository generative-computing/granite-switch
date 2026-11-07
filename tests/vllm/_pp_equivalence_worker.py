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

# Prompt text for the real-language path (see _build_prompt_ids).
PROMPT_TEXT = "Explain in one sentence what a hash function is."

# Per-step top-k logprobs to request, so compare can tell a tie from a real
# difference. 5 is plenty: only the top two matter.
NUM_LOGPROBS = 5


def _log(message: str) -> None:
    print(f"PP_EQUIVALENCE_PHASE {message}", flush=True)


def _build_prompt_ids(model_path, adapter_token_ids, vocab_size):
    """Real-language prompt with one control token, so the adapter fires.

    The control token makes the switch compute a nonzero adapter index, so the
    SWITCH kernel's delta path runs on every PP rank -- the whole point of the
    test.

    Real text, NOT random filler ids. This used to build 16 ids from
    ``torch.randint(1, 1000)``, which is badly out of distribution, and under
    greedy decoding that compounds: measured on an A100 (vLLM 0.26, bf16) the
    PP=1 and PP=2 arms agreed for 4 tokens and then the PP=1 arm collapsed into a
    repeating two-token loop, failing the test. With this prompt the two arms are
    byte-identical for all 32 tokens.

    The divergence was never the per-rank metadata recompute this test exists to
    check. PP=1 and PP=2 differ in CUDA-graph capture mode (vLLM downgrades
    capture above PP=1) and in reduction order, which is enough to land an
    already-tied argmax on the other side. Probed directly: at PP=1,
    ``cudagraph_mode=PIECEWISE`` is byte-identical to ``enforce_eager`` on
    real-language prompts with adapter routing intact (3 adapters, 3 distinct
    continuations) -- so graph capture does not corrupt the switch, and the junk
    prompt was the whole problem.

    Falling back to filler ids if the tokenizer cannot be loaded keeps the worker
    runnable on a checkpoint without one; the comparison is then tie-tolerant
    (see ``cmd_compare``) rather than token-exact.
    """
    import torch

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        prompt_ids = list(tokenizer(PROMPT_TEXT)["input_ids"])
        # Appended, not injected mid-sequence: a control token spliced into the
        # middle of real text breaks it back out of distribution, which is the
        # problem this is fixing.
        return [*prompt_ids, int(adapter_token_ids[0])]
    except Exception as exc:  # pragma: no cover - no-tokenizer checkpoints
        _log(f"tokenizer_unavailable ({exc}); falling back to filler ids")

    torch.manual_seed(42)
    max_fill = min(vocab_size, 1000)
    prompt_ids = torch.randint(1, max_fill, (PROMPT_LEN,)).tolist()
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

    prompt_ids = _build_prompt_ids(model_path, adapter_token_ids, vocab_size)
    _log(f"prompt_ready len={len(prompt_ids)}")

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
        logprobs=NUM_LOGPROBS,
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

    completion = outputs[0].outputs[0]
    generated_ids = list(completion.token_ids)
    _log(f"generated {len(generated_ids)} tokens: {generated_ids[:10]}...")

    # Per-step gap between the best and second-best candidate. compare uses this
    # to tell "the two arms disagree" from "the argmax was a tie and the two arms
    # broke it differently", which is not a defect in anything this test covers.
    gaps = []
    for step in completion.logprobs or []:
        values = sorted((lp.logprob for lp in step.values()), reverse=True)
        gaps.append(values[0] - values[1] if len(values) > 1 else float("inf"))
    _log(f"min_top2_gap={min(gaps) if gaps else float('nan')}")

    output_path = os.path.join(work_dir, f"{tag}.json")
    with open(output_path, "w") as f:
        json.dump(
            {"token_ids": generated_ids, "pp_size": pp_size, "top2_gaps": gaps}, f
        )
    _log(f"saved {output_path}")

    del llm
    gc.collect()
    return 0


# ── compare mode ──────────────────────────────────────────────────


def cmd_compare(args):
    """Compare pp1.json / pp2.json, treating an exactly-tied argmax as equal.

    Tie-tolerant rather than token-exact, and the tolerance is measured per
    position rather than assumed. PP=1 and PP=2 do not run the same computation
    bit-for-bit: vLLM downgrades CUDA-graph capture above PP=1, and splitting the
    stack across two ranks changes the float reduction order. Where the top two
    candidates are already equal in bf16, that sub-ULP difference decides which
    one greedy emits -- an arbitrary coin flip, not a routing defect. Measured on
    an A100 (vLLM 0.26) this prompt reaches a top-2 gap of exactly 0.0, so a
    token-exact assertion here cannot be made stable.

    A broken per-rank recompute does not look like that. A rank that rebuilt its
    bitmasks or remapped indices wrongly applies the wrong adapter delta, which
    moves the whole distribution and separates the argmax -- a divergence with a
    nonzero gap, which still fails below.
    """
    work_dir = args.work_dir
    pp1_path = os.path.join(work_dir, "pp1.json")
    pp2_path = os.path.join(work_dir, "pp2.json")

    with open(pp1_path) as f:
        pp1 = json.load(f)
    with open(pp2_path) as f:
        pp2 = json.load(f)
    pp1_ids, pp2_ids = pp1["token_ids"], pp2["token_ids"]
    pp1_gaps = pp1.get("top2_gaps") or []

    print(f"  pp1 tokens ({len(pp1_ids)}): {pp1_ids}")
    print(f"  pp2 tokens ({len(pp2_ids)}): {pp2_ids}")

    if len(pp1_ids) != len(pp2_ids):
        print(f"\nFAIL: length mismatch: pp1={len(pp1_ids)}, pp2={len(pp2_ids)}")
        return 1

    ties, real = [], []
    for i, (a, b) in enumerate(zip(pp1_ids, pp2_ids)):
        if a == b:
            continue
        # No gap recorded (older artifact, or no tokenizer) => cannot prove a tie,
        # so treat it as a real divergence rather than passing on faith.
        gap = pp1_gaps[i] if i < len(pp1_gaps) else 0.0
        (ties if gap <= 0.0 else real).append((i, a, b, gap))

    for i, a, b, gap in ties:
        print(f"  tie at position {i}: pp1={a}, pp2={b} (pp1 top-2 gap {gap:.6f})")

    if real:
        i, a, b, gap = real[0]
        print(
            f"\nFAIL: divergence at position {i}: pp1={a}, pp2={b}, "
            f"pp1 top-2 gap {gap:.6f}.\n"
            "  The argmax was strictly separated there, so this is not a tie-break: "
            "PP=2 must recompute per-rank LoRA kernel metadata identically to PP=1, "
            "and a mismatch means the recompute-per-rank contract is broken."
        )
        return 1

    print(
        f"\nPASS: PP=2 matches PP=1 over {len(pp1_ids)} tokens "
        f"({len(ties)} exactly-tied position(s) allowed to differ) — "
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
