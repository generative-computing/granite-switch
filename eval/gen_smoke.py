# SPDX-License-Identifier: Apache-2.0
"""Adapter-generation smoke test for a composed Granite Switch checkpoint.

Loads a composed switch model in vLLM and generates the same question three
ways — with no control token (base) and with each adapter's control token
appended — then prints the decoded text side by side. The point is to eyeball
that (a) the composed switch model loads and generates at all under the
de-hybridized (granitemoeshared) backend, and (b) different adapter control
tokens actually change the output rather than all collapsing to base.

Exit code: 0 if generation succeeded for base + every adapter, 1 otherwise.
Mirrors the routing/generation approach in
tests/integration/test_multi_switch_vllm_generate.py.

Usage:
    python eval/gen_smoke.py --model /path/to/composed-switch-model
"""

import argparse
import os
import sys


QUESTIONS = [
    "Explain in one sentence what a hash function is.",
    "Briefly: what is entropy in thermodynamics?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="composed switch checkpoint dir")
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.config import GraniteSwitchConfig

    config = GraniteSwitchConfig.from_pretrained(args.model)
    adapter_tokens = list(config.adapter_token_ids or [])
    adapter_names = list(config.adapter_names or [])
    print(f"[gen_smoke] num_adapters={config.num_adapters}")
    print(f"[gen_smoke] adapter_names={adapter_names}")
    print(f"[gen_smoke] adapter_token_ids={adapter_tokens}")
    print(f"[gen_smoke] shared_intermediate_size={config.shared_intermediate_size}")
    print(f"[gen_smoke] num_local_experts={getattr(config, 'num_local_experts', 0)}")
    if not adapter_tokens:
        print("[gen_smoke] FAIL: composed model has no adapter control tokens")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.7,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Probe the first two adapters (or fewer if the model has fewer).
    probe = adapter_tokens[: min(2, len(adapter_tokens))]
    ok = True
    for question in QUESTIONS:
        base_ids = tokenizer(question)["input_ids"]
        variants = [("base", base_ids)]
        for i, ctrl in enumerate(probe):
            name = adapter_names[i] if i < len(adapter_names) else f"adapter_{i}"
            variants.append((name, base_ids + [ctrl]))

        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for _, ids in variants]
        outs = llm.generate(prompts, sampling_params=sp)
        print(f"\n=== Q: {question}")
        texts = []
        for (label, _), out in zip(variants, outs):
            toks = [int(t) for t in out.outputs[0].token_ids]
            text = tokenizer.decode(toks, skip_special_tokens=True).strip()
            texts.append(text)
            print(f"  [{label:16s}] {text[:160]!r}")
            if not toks:
                print(f"  [gen_smoke] WARN: {label} produced no tokens")
                ok = False
        # Soft signal: at least one adapter should differ from base.
        if len(texts) > 1 and all(t == texts[0] for t in texts[1:]):
            print("  [gen_smoke] NOTE: all adapters matched base text for this Q "
                  "(may indicate routing collapsed to base)")

    print("\n[gen_smoke] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
