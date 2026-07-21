# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: verify a skinned GraniteSwitch model is logit-exact vs the original via vLLM.

Three modes, each invoked as a separate subprocess so only one vLLM model is
ever resident on GPU at a time::

    python worker.py build   --model <name> --work-dir <dir> [--fast]
    python worker.py run     --model <name-or-path> --output <json> [--fast]
    python worker.py compare --ref <json> --switch <json> --label <model_name>

**build**: Loads config for dtype/vocab, generates deterministic inputs
(``torch.manual_seed(42)``), saves them to ``<work-dir>/inputs.json``.  Builds
skin via ``GraniteSwitchComposer.from_base_and_adapters()``, saves to
``<work-dir>/skin/``.

**run**: Loads inputs from ``--inputs`` JSON, loads model in vLLM, extracts
top-K logprobs per position, saves to ``--output`` as JSON.

**compare**: Loads two JSON logprob files, checks bit-exact match, exits 0 on
match, 1 on diff.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig

FAST_LENGTHS = [64]  # Single medium-length request for quick regression checks.
FULL_LENGTHS = [3, 7, 16, 32, 64, 128, 192, 256]  # Thorough: short to long.
TOP_K = 100  # Compare top-100 logprobs per position (sufficient to detect divergence).


def _native_dtype(config):
    """Determine the model's native dtype from its HuggingFace config."""
    dt = getattr(config, "torch_dtype", None)
    if dt is None:
        return torch.float32
    if isinstance(dt, torch.dtype):
        return dt
    if isinstance(dt, str):
        return getattr(torch, dt, torch.float32)
    return torch.float32


def _dtype_str(dtype):
    """Convert torch.dtype to vLLM dtype string."""
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }.get(dtype, "auto")


def _generate_inputs(vocab_size, request_lengths):
    """Generate deterministic test inputs from model config."""
    torch.manual_seed(42)
    max_tok = min(vocab_size, 1000)
    return [torch.randint(1, max_tok, (l,)).tolist() for l in request_lengths]


# ── build mode ────────────────────────────────────────────────────


def cmd_build(args):
    """Build a GraniteSwitch skin and save inputs + skin to work-dir."""
    from granite_switch.composer import GraniteSwitchComposer
    from granite_switch.composer.arch import load_base_config

    model_name = args.model
    work_dir = args.work_dir
    request_lengths = FAST_LENGTHS if args.fast else FULL_LENGTHS

    print(f"Loading config for {model_name}...")
    base_config = load_base_config(model_name)
    dtype = _native_dtype(base_config)
    print(f"  model_type={base_config.model_type}  native_dtype={dtype}")

    # Generate and save deterministic inputs
    all_ids = _generate_inputs(base_config.vocab_size, request_lengths)
    inputs_path = os.path.join(work_dir, "inputs.json")
    with open(inputs_path, "w") as f:
        json.dump({"request_lengths": request_lengths, "inputs": all_ids}, f)
    print(f"  saved {len(all_ids)} input sequences to {inputs_path}")

    # Build skin
    print("\nBuilding GraniteSwitch skin (num_adapters=0)...")
    skin_dir = os.path.join(work_dir, "skin")
    model = GraniteSwitchComposer.from_base_and_adapters(
        model_name,
        torch_dtype=dtype,
    )
    print(f"  saving skinned model to {skin_dir}...")
    model.save_pretrained(skin_dir)
    del model
    print("  build complete")
    return 0


# ── run mode ──────────────────────────────────────────────────────


def cmd_run(args):
    """Load a model in vLLM, extract logprobs, save to JSON."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    model_path = args.model
    output_path = args.output
    inputs_path = args.inputs

    # Load inputs
    with open(inputs_path) as f:
        data = json.load(f)
    all_ids = data["inputs"]

    # Resolve dtype from config.
    print(f"Loading config for {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    dtype = _native_dtype(config)
    dtype_s = _dtype_str(dtype)
    print(f"  native_dtype={dtype}")

    # Create vLLM instance
    print(f"Creating vLLM LLM for {model_path}...")
    llm_kwargs = dict(
        model=model_path,
        dtype=dtype_s,
        max_logprobs=TOP_K,
        enforce_eager=True,
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
    )

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=TOP_K,
    )

    # Extract logprobs one request at a time
    all_logprobs = []
    for i, ids in enumerate(all_ids):
        print(f"  request {i + 1}/{len(all_ids)} (len={len(ids)})...")
        prompt = TokensPrompt(prompt_token_ids=ids)
        outputs = llm.generate(prompt, sampling_params=sampling_params)
        # Extract top-K prompt logprobs (skip position 0)
        req_logprobs = []
        for pos_logprobs in outputs[0].prompt_logprobs[1:]:
            if pos_logprobs is not None:
                req_logprobs.append(
                    {str(tid): lp.logprob for tid, lp in pos_logprobs.items()}
                )
            else:
                req_logprobs.append({})
        all_logprobs.append(req_logprobs)
        print(f"    positions: {len(req_logprobs)}, top-k: {TOP_K}")

    # Save logprobs
    with open(output_path, "w") as f:
        json.dump({"logprobs": all_logprobs}, f)
    print(f"  saved logprobs to {output_path}")

    del llm
    return 0


# ── compare mode ──────────────────────────────────────────────────


def cmd_compare(args):
    """Load two logprob JSONs and check bit-exact match."""
    with open(args.ref) as f:
        ref_data = json.load(f)
    with open(args.switch) as f:
        sw_data = json.load(f)

    ref_all = ref_data["logprobs"]
    sw_all = sw_data["logprobs"]
    label = args.label

    assert len(ref_all) == len(sw_all), (
        f"{label}: request count mismatch {len(ref_all)} vs {len(sw_all)}"
    )

    rc = 0
    for i, (ref_req, sw_req) in enumerate(zip(ref_all, sw_all)):
        rc_i = _compare_logprobs(ref_req, sw_req, f"req[{i}]")
        rc = max(rc, rc_i)

    if rc == 0:
        print(
            f"\nPASS: {label} — bit-exact equivalence via vLLM "
            f"[{len(ref_all)} individual requests]"
        )
    else:
        print(f"\nFAIL: {label} — logprobs differ via vLLM")
    return rc


def _compare_logprobs(reference, switch, label):
    """Compare two top-K logprob lists. Returns 0 on exact match, 1 on diff.

    Each input is a list of dicts (one per position), mapping str(token_id) → logprob.
    """
    assert len(reference) == len(switch), (
        f"{label}: length mismatch {len(reference)} vs {len(switch)}"
    )
    total_entries = 0
    mismatched_keys = 0
    mismatched_values = 0
    max_diff = 0.0

    for pos, (ref_d, sw_d) in enumerate(zip(reference, switch)):
        total_entries += len(ref_d)
        if set(ref_d.keys()) != set(sw_d.keys()):
            mismatched_keys += 1
            continue
        for tid in ref_d:
            d = abs(ref_d[tid] - sw_d[tid])
            if d > 0:
                mismatched_values += 1
                max_diff = max(max_diff, d)

    positions = len(reference)
    print(f"  [{label}] positions: {positions}, entries: {total_entries}")
    if mismatched_keys > 0:
        print(
            f"  [{label}] FAIL: {mismatched_keys} positions have different top-K token sets"
        )
        return 1
    if mismatched_values > 0:
        print(
            f"  [{label}] FAIL: {mismatched_values} logprob values differ, max |diff| = {max_diff:.6e}"
        )
        return 1
    print(f"  [{label}] OK: bit-exact")
    return 0


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    # build
    p_build = sub.add_parser("build", help="Build skin and save inputs")
    p_build.add_argument(
        "--model", required=True, help="HuggingFace model name or path"
    )
    p_build.add_argument(
        "--work-dir", required=True, help="Working directory for outputs"
    )
    p_build.add_argument(
        "--fast",
        action="store_true",
        help="Single medium-length request (quick regression check)",
    )

    # run
    p_run = sub.add_parser("run", help="Load model in vLLM, extract logprobs")
    p_run.add_argument("--model", required=True, help="Model name or path to load")
    p_run.add_argument("--inputs", required=True, help="Path to inputs.json")
    p_run.add_argument("--output", required=True, help="Path to save logprobs JSON")

    # compare
    p_compare = sub.add_parser("compare", help="Compare two logprob JSONs")
    p_compare.add_argument("--ref", required=True, help="Reference logprobs JSON")
    p_compare.add_argument("--switch", required=True, help="Switch logprobs JSON")
    p_compare.add_argument("--label", required=True, help="Model label for output")

    args = parser.parse_args()

    if args.mode == "build":
        return cmd_build(args)
    elif args.mode == "run":
        return cmd_run(args)
    elif args.mode == "compare":
        return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
