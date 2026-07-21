# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: verify generation equivalence between upstream and zero-adapter switch models.

Three modes, each invoked as a separate subprocess so only one vLLM model is
ever resident on GPU at a time::

    python worker.py build   --model <name> --work-dir <dir>
    python worker.py run     --model <name-or-path> --work-dir <dir> --tag <tag>
    python worker.py compare --work-dir <dir> --label <model_name>

**build**: Loads config for dtype/vocab, generates a deterministic 64-token prompt,
builds a GraniteSwitch model with 1 built-in adapter (zero LoRA weights).
Saves the switch model and inputs to ``<work-dir>/``.

**run**: Loads inputs from ``<work-dir>/inputs.json``, loads model in vLLM, runs
greedy autoregressive generation (temperature=0, max_tokens=32), saves generated
token IDs to ``<work-dir>/<tag>.json``.

**compare**: Loads two token-ID JSONs and checks token-for-token match.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig


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


# ── build mode ────────────────────────────────────────────────────


def cmd_build(args):
    """Build a GraniteSwitch model with 1 zero-weight built-in adapter."""
    from granite_switch.composer import GraniteSwitchComposer

    model_name = args.model
    work_dir = args.work_dir

    print(f"Loading config for {model_name}...")
    base_config = AutoConfig.from_pretrained(model_name)
    dtype = _native_dtype(base_config)
    vocab_size = base_config.vocab_size
    print(
        f"  model_type={base_config.model_type}  native_dtype={dtype}  vocab_size={vocab_size}"
    )

    # Deterministic prompt (no control tokens — all IDs in [1, 1000))
    torch.manual_seed(42)
    max_tok = min(vocab_size, 1000)
    prompt_ids = torch.randint(1, max_tok, (64,)).tolist()

    # Adapter token IDs placed far from the prompt range
    adapter_token_id = vocab_size - 100

    # Save inputs
    inputs_path = os.path.join(work_dir, "inputs.json")
    with open(inputs_path, "w") as f:
        json.dump(
            {
                "prompt_ids": prompt_ids,
                "adapter_token_id": adapter_token_id,
                "vocab_size": vocab_size,
            },
            f,
        )
    print(f"  saved inputs to {inputs_path}")

    # Build switch model with 1 built-in adapter
    print("\nBuilding GraniteSwitch (1 built-in adapter)...")
    skin_dir = os.path.join(work_dir, "switch")
    model = GraniteSwitchComposer.from_base_and_adapters(
        model_name,
        built_in_adapter_names=["test"],
        adapter_names=["test"],
        adapter_token_ids=[adapter_token_id],
        adapter_substitute_token_ids=[1],
        torch_dtype=dtype,
    )

    # Zero all LoRA weights
    print("  zeroing all LoRA weights...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.zero_()
                print(f"    zeroed {name} {tuple(param.shape)}")

    print(f"  saving switch model to {skin_dir}...")
    model.save_pretrained(skin_dir)
    del model
    print("  build complete")
    return 0


# ── run mode ──────────────────────────────────────────────────────


def cmd_run(args):
    """Load a model in vLLM, run greedy generation, save token IDs."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register as register_granite_switch

    register_granite_switch()

    model_path = args.model
    work_dir = args.work_dir
    tag = args.tag

    # Load inputs
    inputs_path = os.path.join(work_dir, "inputs.json")
    with open(inputs_path) as f:
        data = json.load(f)
    prompt_ids = data["prompt_ids"]

    # Resolve dtype
    print(f"Loading config for {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    dtype = _native_dtype(config)
    dtype_s = _dtype_str(dtype)
    print(f"  native_dtype={dtype}")

    # Create vLLM instance
    print(f"Creating vLLM LLM for {model_path}...")
    llm = LLM(
        model=model_path,
        skip_tokenizer_init=True,
        dtype=dtype_s,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=32,
        ignore_eos=True,
    )

    # Generate
    print(f"  generating (prompt_len={len(prompt_ids)}, max_tokens=32)...")
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    outputs = llm.generate(prompt, sampling_params=sampling_params)
    generated_ids = list(outputs[0].outputs[0].token_ids)
    print(f"  generated {len(generated_ids)} tokens: {generated_ids[:10]}...")

    # Save
    output_path = os.path.join(work_dir, f"{tag}.json")
    with open(output_path, "w") as f:
        json.dump({"token_ids": generated_ids}, f)
    print(f"  saved to {output_path}")

    del llm
    return 0


# ── compare mode ──────────────────────────────────────────────────


def cmd_compare(args):
    """Load two token-ID JSONs and check token-for-token match."""
    work_dir = args.work_dir
    label = args.label

    ref_path = os.path.join(work_dir, "ref.json")
    sw_path = os.path.join(work_dir, "switch.json")
    inputs_path = os.path.join(work_dir, "inputs.json")

    with open(ref_path) as f:
        ref_ids = json.load(f)["token_ids"]
    with open(sw_path) as f:
        sw_ids = json.load(f)["token_ids"]
    with open(inputs_path) as f:
        inputs = json.load(f)

    adapter_token_id = inputs["adapter_token_id"]

    print(f"  ref tokens ({len(ref_ids)}): {ref_ids}")
    print(f"  switch tokens ({len(sw_ids)}): {sw_ids}")

    if len(ref_ids) != len(sw_ids):
        print(
            f"\nFAIL: {label} — length mismatch: ref={len(ref_ids)}, switch={len(sw_ids)}"
        )
        return 1

    for i, (r, s) in enumerate(zip(ref_ids, sw_ids)):
        if r != s:
            msg = f"\nFAIL: {label} — first divergence at position {i}: ref={r}, switch={s}"
            if r == adapter_token_id or s == adapter_token_id:
                msg += (
                    f"\n  NOTE: adapter_token_id={adapter_token_id} was generated. "
                    f"This is expected to cause divergence due to KV hiding."
                )
            print(msg)
            return 1

    print(
        f"\nPASS: {label} — token-for-token generation equivalence "
        f"[{len(ref_ids)} tokens]"
    )
    return 0


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    # build
    p_build = sub.add_parser("build", help="Build switch model and save inputs")
    p_build.add_argument(
        "--model", required=True, help="HuggingFace model name or path"
    )
    p_build.add_argument(
        "--work-dir", required=True, help="Working directory for outputs"
    )

    # run
    p_run = sub.add_parser("run", help="Load model in vLLM, generate tokens")
    p_run.add_argument("--model", required=True, help="Model name or path to load")
    p_run.add_argument(
        "--work-dir", required=True, help="Working directory with inputs.json"
    )
    p_run.add_argument("--tag", required=True, help="Output tag (ref or switch)")

    # compare
    p_compare = sub.add_parser("compare", help="Compare two token-ID JSONs")
    p_compare.add_argument(
        "--work-dir",
        required=True,
        help="Working directory with ref.json and switch.json",
    )
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
