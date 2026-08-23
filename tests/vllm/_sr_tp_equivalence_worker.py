# SPDX-License-Identifier: Apache-2.0
"""Worker for test_sr_tp_equivalence — runs in its own subprocess (one per TP size)
because vLLM does not free GPU memory on ``del``.

Loads a composed Shadow-Residual checkpoint under vLLM at ``tensor_parallel_size=<tp>``
and dumps the decision-position top-k next-token logprobs (one dict per prompt).

  python _sr_tp_equivalence_worker.py <tp_size> <out.json>

Env: SR_COMPOSED_DIR (composed SR checkpoint), SR_TP_TOPK (default 200). The prompt builder
auto-detects a granite-4.2 chat template (no ``documents`` handling → inline the docs into the
user message in the training format + enable_thinking=False); granite-4.1 keeps the inline-doc
form. The adapter control token (config ``adapter_token_ids[0]``) is appended so the SR adapter
fires; the same prompt is used for every TP leg, so the equivalence check is format-independent.
"""

import json
import os
import sys

COMPOSED = os.environ["SR_COMPOSED_DIR"]
TOPK = int(os.environ.get("SR_TP_TOPK", "200"))

PROMPTS = [
    (
        "What is the capital of France?",
        "France is a country in Western Europe; its capital and largest city is Paris.",
    ),
    (
        "In what year did the company report its first profit?",
        "The firm was founded in 1998 and expanded rapidly across three continents.",
    ),
    (
        "How many moons does Mars have?",
        "Mars is the fourth planet from the Sun; it has two moons, Phobos and Deimos.",
    ),
    (
        "What treatment did the study recommend?",
        "The passage describes the geology of the Grand Canyon and the Colorado River.",
    ),
]


def _build_decision_ids(tok, cfg):
    ctrl = cfg["adapter_token_ids"][0]
    tmpl = getattr(tok, "chat_template", None) or ""
    is_g42 = "document" not in tmpl.lower()
    print(
        "SR_TP_PROMPT: "
        + (
            "granite-4.2 (inline docs + enable_thinking=False)"
            if is_g42
            else "granite-4.1 (inline doc)"
        ),
        flush=True,
    )
    ids_list = []
    for q, doc in PROMPTS:
        if is_g42:
            content = f"Documents:\n\nDocument 0\n{doc}\n\n{q}"
            kw = {"enable_thinking": False}
        else:
            content = f"{q}\n\nDocument: {doc}"
            kw = {}
        msg = [{"role": "user", "content": content}]
        try:
            text = tok.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False, **kw
            )
        except Exception:
            text = content + "\n"
        base_ids = list(tok(text, add_special_tokens=False).input_ids)
        ids_list.append([*base_ids, ctrl])
    return ids_list


def main(tp, out_path):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoTokenizer

    from granite_switch.vllm import register

    register()
    from vllm import LLM, SamplingParams

    cfg = json.load(open(os.path.join(COMPOSED, "config.json")))
    tok = AutoTokenizer.from_pretrained(COMPOSED)
    ids_list = _build_decision_ids(tok, cfg)

    kw = dict(
        model=COMPOSED,
        tensor_parallel_size=tp,
        enforce_eager=True,
        dtype="bfloat16",
        max_logprobs=TOPK,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        skip_tokenizer_init=True,
    )
    if tp > 1:
        kw["distributed_executor_backend"] = "mp"
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=TOPK)
    outs = llm.generate([{"prompt_token_ids": ids} for ids in ids_list], sp)

    decisions = [
        {str(tid): float(lp.logprob) for tid, lp in o.outputs[0].logprobs[0].items()}
        for o in outs
    ]
    json.dump({"tp": tp, "decisions": decisions}, open(out_path, "w"))
    print(f"SR_TP_DUMP: tp={tp} wrote {len(decisions)} decisions", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: _sr_tp_equivalence_worker.py <tp_size> <out.json>", flush=True)
        sys.exit(2)
    sys.exit(main(int(sys.argv[1]), sys.argv[2]))
