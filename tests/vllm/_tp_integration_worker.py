# SPDX-License-Identifier: Apache-2.0
"""TP integration test worker — runs in subprocess to avoid CUDA fork issues.

Commands:
  build         — Compose a zero-adapter switch model and save to disk (CPU)
  build-compose — Compose via the CLI compose script with adapter repos (CPU)
  run           — Load model in vLLM with given TP size, generate, save output (GPU)
"""

import argparse
import json
import os
import subprocess
import sys

from transformers import AutoConfig, AutoTokenizer

PLAIN_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    # >512 tokens to exercise FA3 non-CUDA-graph path under TP (#104)
    ("Summarize the following document in detail. " * 100).strip(),
]

CHAT_MESSAGES = [
    {"role": "user", "content": "Is this document relevant to the query?"},
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


def cmd_run(args):
    """Load model in vLLM and generate."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    model_path = args.model_path
    tp_size = args.tp_size
    output_path = args.output_path

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
    )

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

    outputs = llm.generate(PLAIN_PROMPTS, sampling)
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
        chat_outputs = llm.chat(
            CHAT_MESSAGES,
            sampling_params=sampling,
            chat_template_kwargs={"intrinsic_name": args.intrinsic_name},
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

    p_run = sub.add_parser("run")
    p_run.add_argument("--model-path", required=True)
    p_run.add_argument("--tp-size", type=int, required=True)
    p_run.add_argument("--output-path", required=True)
    p_run.add_argument(
        "--intrinsic-name",
        default=None,
        help="If set, adds a chat-template prompt activating this adapter",
    )

    args = parser.parse_args()
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "build-compose":
        return cmd_build_compose(args)
    elif args.command == "run":
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
