# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker for the vLLM classifier chunked-prefill test.

Each sub-command is a separate process invocation so only one vLLM engine is ever
resident (same discipline as ``_generation_equivalence_worker``)::

    python worker.py compose-multilabel           # CPU only; prints {"result": meta}
    python worker.py run <budget|none> <out.json> # one engine, one budget
"""

import json
import os
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# Single-token label candidates, in priority order (only single-token ones are kept).
_LABEL_CANDIDATES = [
    "safe",
    "unsafe",
    "yes",
    "no",
    "true",
    "false",
    "good",
    "bad",
    "high",
    "low",
    "hot",
    "cold",
]


def compose(base, outdir, control_offset=5, num_labels=6, scale=0.5):
    """Compose a single-classifier-slot checkpoint.

    ``num_labels`` single-token labels are taken from ``_LABEL_CANDIDATES`` and
    ``scale`` is the std of the random head weight.
    """
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoTokenizer

    from granite_switch.composer.compose_utils import GraniteSwitchComposer
    from granite_switch.composer.weight_transfer import CLASSIFIER_HEAD_FILE

    tok = AutoTokenizer.from_pretrained(base)
    hidden = AutoConfig.from_pretrained(base).hidden_size

    labels, ids = [], []
    for c in _LABEL_CANDIDATES:
        e = tok.encode(c, add_special_tokens=False)
        if len(e) == 1 and e[0] not in ids:
            labels.append(c)
            ids.append(e[0])
        if len(labels) == num_labels:
            break
    if len(labels) < num_labels:
        raise RuntimeError(
            f"only {len(labels)} single-token labels available from "
            f"{_LABEL_CANDIDATES}, need {num_labels}"
        )
    nl = len(labels)

    g = torch.Generator().manual_seed(0)
    # Weight-based (not bias-only) head so the verdict depends on the hidden state.
    weight = torch.randn((nl, hidden), generator=g, dtype=torch.float32) * scale
    bias = torch.zeros((nl,), dtype=torch.float32)

    slot = os.path.join(outdir, "detect")
    os.makedirs(slot, exist_ok=True)
    save_file(
        {"weight": weight.contiguous(), "bias": bias.contiguous()},
        os.path.join(slot, CLASSIFIER_HEAD_FILE),
    )
    json.dump(
        {"kind": "classifier", "labels": labels},
        open(os.path.join(slot, "adapter_config.json"), "w"),
    )
    open(os.path.join(slot, "io.yaml"), "w").write("name: detect\nmodel: ~\n")

    control_id = tok.vocab_size - control_offset
    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base,
        adapter_paths=[slot],
        adapter_token_ids=[control_id],
        adapter_substitute_token_ids=[control_id],
        adapter_names=["detect"],
        adapter_kinds=["classifier"],
        classifier_label_token_ids=[ids],
    )
    ckpt = os.path.join(outdir, "composed")
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    words = (
        "machine translation quietly reshaped how distant villages argued "
        "about rainfall while engineers debated whether copper cables or "
        "glass fibre carried gossip faster than a startled horse could run "
        "downhill past the old mill where children once traded marbles for "
        "stories about comets volcanoes submarines and the strange arithmetic "
        "of tides that neither king nor merchant ever fully trusted yet "
        "everyone quoted at dinner as though the ocean kept a ledger of debts "
        "owed to the moon each evening without fail or apparent complaint "
        "although sailors insisted otherwise over cheap wine and louder songs"
    )
    # The chat template's classifier layout: the user turn, its close, then the
    # marker as the very last token, which is itself the read point.
    body = tok.encode(
        f"<|start_of_role|>user<|end_of_role|>Is this text safe? {words}"
        "<|end_of_text|>\n",
        add_special_tokens=False,
    )
    prompt_ids = [*body, control_id]
    return {
        "ckpt": ckpt,
        "labels": labels,
        "control_id": control_id,
        "label_token_ids": ids,
        "prompt_ids": prompt_ids,
    }


def run_budget(ckpt, prompt_ids, budget, out_path):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=ckpt,
        enforce_eager=True,
        gpu_memory_utilization=0.55,
        max_model_len=4096,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    if budget is None:
        kw["enable_chunked_prefill"] = False
    else:
        kw["max_num_batched_tokens"] = budget
        kw["enable_chunked_prefill"] = True
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    o = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0].outputs[0]
    out = {"budget": budget, "token_id": int(o.token_ids[0]), "text": o.text}
    json.dump(out, open(out_path, "w"))


def main():
    mode = sys.argv[1]
    if mode == "compose-multilabel":
        try:
            meta = compose(os.environ["CLS_BASE"], os.environ["CLS_OUT"])
            print(json.dumps({"result": meta}))
        except Exception as e:
            print(json.dumps({"error": f"{e}\n{traceback.format_exc()}"}))
        return
    if mode == "run":
        budget = None if sys.argv[2] == "none" else int(sys.argv[2])
        out_path = sys.argv[3]
        prompt_ids = json.load(open(os.environ["CLS_PIDS"]))
        try:
            run_budget(os.environ["CLS_CKPT"], prompt_ids, budget, out_path)
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
