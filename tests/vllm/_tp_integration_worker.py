# SPDX-License-Identifier: Apache-2.0
"""TP integration test worker — runs in subprocess to avoid CUDA fork issues.

Commands:
  build            — Compose a zero-adapter switch model and save to disk (CPU)
  build-compose    — Compose via the CLI compose script with adapter repos (CPU)
  build-granitemoe — Compose a tiny synthetic pure-sparse MoE checkpoint (CPU)
  run              — Load model in vLLM with given TP size, generate, save output (GPU)
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLAIN_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    # >512 tokens to exercise FA3 non-CUDA-graph path under TP (#104)
    ("Summarize the following document in detail. " * 100).strip(),
]

CHAT_MESSAGES = [
    {"role": "user", "content": "Is this document relevant to the query?"},
]

# Used with ``--token-prompts``: a synthetic checkpoint ships no tokenizer, so
# the string prompts above are unusable.  The two rows differ only at position 2
# -- control token vs. an ordinary id -- so the switch's own attention layer runs
# under sharding on one of them and not the other.
_CONTROL_TOKEN_ID = 250
TOKEN_PROMPTS = [
    [10, 11, _CONTROL_TOKEN_ID, 12, 13, 14, 15, 16],
    [10, 11, 30, 12, 13, 14, 15, 16],
]


def cmd_build(args):
    """Build a zero-adapter GraniteSwitch model."""
    from granite_switch.composer import GraniteSwitchComposer

    base_model = args.base_model
    output_dir = args.output_dir

    base_config = AutoConfig.from_pretrained(base_model)
    vocab_size = base_config.vocab_size

    adapter_token_id = vocab_size - 100
    muted_token_id = vocab_size - 101

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model,
        built_in_adapter_names=["test"],
        adapter_names=["test"],
        adapter_token_ids=[adapter_token_id],
        adapter_substitute_token_ids=[1],
        muted_adapter_token_ids=[muted_token_id],
        switch_type="single",
    )

    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(output_dir)
    del model

    print("BUILD_OK")
    return 0


def cmd_build_compose(args):
    """Build a GraniteSwitch model using the CLI compose script."""
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        args.base_model,
        "--output",
        args.output_dir,
    ]
    for repo in args.adapter_repos:
        cmd.extend(["--adapters", repo])
    # MultiSwitch owns 2 cache layers vs SingleSwitch's 1; the composer inflates
    # num_hidden_layers from switch_type itself (compose_utils._switch_cache_layers),
    # so passing the flag is all that is needed -- no geometry changes here.
    if getattr(args, "switch_type", None):
        cmd.extend(["--switch-type", args.switch_type])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    if result.returncode != 0:
        print(f"Compose failed (exit {result.returncode})")
        return result.returncode

    print("BUILD_COMPOSE_OK")
    return 0


def cmd_build_granitemoe(args):
    """Compose a tiny synthetic pure-sparse MoE checkpoint (no download).

    Deliberately not built with ``save_switch_model``: ``SwitchedLoRALinear``
    zero-initializes ``lora_B``, so a synthetic switch model has an identically
    zero adapter delta and a TP comparison over it degenerates into comparing two
    base-only runs.  Composing a real PEFT adapter is what makes the adapter live,
    and the guards below fail the build rather than let that pass silently.
    """
    from tests.shared.granitemoe_compose import (
        ADAPTER_BUILDERS,
        GPU_GEOMETRY,
        compose_granitemoe,
    )

    output_dir = Path(args.output_dir)
    work = output_dir.parent

    build = compose_granitemoe(
        name=args.variant,
        base_path=work / f"{args.variant}_moe_base",
        adapter_path=work / f"{args.variant}_moe_adapter",
        output_dir=output_dir,
        make_adapter=ADAPTER_BUILDERS[args.variant],
        geometry=GPU_GEOMETRY,
    )

    live = [
        name
        for name, param in build.model.named_parameters()
        if "lora_B" in name and param.abs().sum() > 0
    ]
    if not live:
        print("BUILD_FAIL: every lora_B is zero; the adapter is dead", file=sys.stderr)
        return 1
    if args.variant == "sr" and not any("cross_stream" in n for n in live):
        print(
            "BUILD_FAIL: cross_stream.lora_B is zero; the SR cross-stream "
            "injection is dead and only the q/o deltas would be compared",
            file=sys.stderr,
        )
        return 1

    config = json.loads((output_dir / "config.json").read_text())
    if config.get("shared_intermediate_size") != 0:
        print(
            f"BUILD_FAIL: shared_intermediate_size is "
            f"{config.get('shared_intermediate_size')!r}, not 0 — this is not the "
            f"pure-sparse path under test",
            file=sys.stderr,
        )
        return 1
    if not config.get("num_local_experts"):
        print("BUILD_FAIL: no expert bank in the composed config", file=sys.stderr)
        return 1

    del build
    print(f"BUILD_GRANITEMOE_OK live_lora_B={len(live)}")
    return 0


def cmd_run(args):
    """Load model in vLLM and generate."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    model_path = args.model_path
    tp_size = args.tp_size
    output_path = args.output_path

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
    )
    if args.token_prompts:
        # A synthetic checkpoint has no tokenizer, so vLLM must not try to load
        # one.  bf16 keeps the numerics on the same footing as the real-checkpoint
        # arms above, which is what the tolerances downstream were measured for.
        llm_kwargs.update(
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=64,
            gpu_memory_utilization=0.3,
        )

    llm = LLM(**llm_kwargs)

    # Request top-K logprobs at the first generated token of each prompt.
    # We compare distributions across TP sizes (within a tolerance) instead
    # of asserting byte-equality of greedy text — the latter is not a
    # well-defined invariant in bf16 across different all-reduce orderings.
    # See docs/TENSOR_PARALLEL_FIX.md for the analysis.
    sampling = SamplingParams(temperature=0.0, max_tokens=20, logprobs=20)

    def _top_logprobs(first_step_logprobs):
        """Convert the first-step Logprob dict to a JSON-serialisable list of
        (token_id, logprob) sorted descending by logprob."""
        if first_step_logprobs is None:
            return None
        items = [
            (int(tid), float(lp.logprob)) for tid, lp in first_step_logprobs.items()
        ]
        items.sort(key=lambda x: -x[1])
        return items

    records = []

    if args.token_prompts:
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in TOKEN_PROMPTS]
    else:
        prompts = PLAIN_PROMPTS

    outputs = llm.generate(prompts, sampling)
    for o in outputs:
        completion = o.outputs[0]
        first_step = completion.logprobs[0] if completion.logprobs else None
        records.append(
            {
                "text": completion.text,
                "first_token_topk": _top_logprobs(first_step),
            }
        )

    if args.intrinsic_name:
        # The composed chat template gates control-token injection on
        # ``adapter_name`` (composer/tokenizer_setup.py::configure_chat_template
        # emits "{%- if adapter_name is defined and adapter_name in adapter_map %}").
        # This call previously passed ``intrinsic_name``; Jinja silently treats an
        # unknown kwarg as not-defined, so no control token was ever injected and
        # the chat arm of the TP comparison agreed vacuously -- two servers both
        # running plain base. Guard it rather than trust the key.
        tok = AutoTokenizer.from_pretrained(model_path)
        rendered = tok.apply_chat_template(
            CHAT_MESSAGES,
            tokenize=False,
            add_generation_prompt=True,
            adapter_name=args.intrinsic_name,
        )
        expected_token = f"<|{args.intrinsic_name}|>"
        if expected_token not in rendered:
            print(
                f"RUN_FAIL: chat template did not inject {expected_token} for "
                f"adapter_name={args.intrinsic_name!r}; the chat arm would compare "
                f"two base-only runs and agree vacuously. Rendered: {rendered[:300]!r}",
                file=sys.stderr,
            )
            return 1
        chat_outputs = llm.chat(
            CHAT_MESSAGES,
            sampling_params=sampling,
            chat_template_kwargs={"adapter_name": args.intrinsic_name},
        )
        for o in chat_outputs:
            completion = o.outputs[0]
            first_step = completion.logprobs[0] if completion.logprobs else None
            records.append(
                {
                    "text": completion.text,
                    "first_token_topk": _top_logprobs(first_step),
                }
            )

    with open(output_path, "w") as f:
        json.dump(records, f)

    del llm
    print("RUN_OK")
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build")
    p_build.add_argument("--base-model", required=True)
    p_build.add_argument("--output-dir", required=True)

    p_compose = sub.add_parser("build-compose")
    p_compose.add_argument("--base-model", required=True)
    p_compose.add_argument("--output-dir", required=True)
    p_compose.add_argument("--adapter-repos", nargs="+", required=True)
    p_compose.add_argument(
        "--switch-type",
        default=None,
        help="'single' (default composer behaviour) or 'multi'",
    )

    p_moe = sub.add_parser("build-granitemoe")
    p_moe.add_argument("--output-dir", required=True)
    p_moe.add_argument("--variant", choices=("lora", "sr"), required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--model-path", required=True)
    p_run.add_argument("--tp-size", type=int, required=True)
    p_run.add_argument("--output-path", required=True)
    p_run.add_argument(
        "--intrinsic-name",
        default=None,
        help="If set, adds a chat-template prompt activating this adapter",
    )
    p_run.add_argument(
        "--token-prompts",
        action="store_true",
        help="Feed raw token ids instead of strings (checkpoint has no tokenizer)",
    )

    args = parser.parse_args()
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "build-compose":
        return cmd_build_compose(args)
    elif args.command == "build-granitemoe":
        return cmd_build_granitemoe(args)
    elif args.command == "run":
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
