# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker for the vLLM probe chunked-prefill test.

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
    """Compose a single-probe-slot checkpoint.

    ``num_labels`` single-token labels are taken from ``_LABEL_CANDIDATES`` and
    ``scale`` is the std of the random head weight.
    """
    from transformers import AutoConfig, AutoTokenizer

    from granite_switch.composer.compose_utils import GraniteSwitchComposer

    tok = AutoTokenizer.from_pretrained(base)
    labels, ids = _pick_labels(tok, num_labels)
    # Weight-based (not bias-only) head so the verdict depends on the hidden state.
    weight, bias = _fixed_head(
        0, len(ids), AutoConfig.from_pretrained(base).hidden_size, scale
    )
    slot = os.path.join(outdir, "detect")
    _write_head(slot, weight, bias, labels)

    control_id = tok.vocab_size - control_offset
    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base,
        adapter_paths=[slot],
        adapter_token_ids=[control_id],
        adapter_substitute_token_ids=[control_id],
        adapter_names=["detect"],
        adapter_kinds=["probe"],
        probe_label_token_ids=[ids],
    )
    ckpt = os.path.join(outdir, "composed")
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    del model

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
    # The chat template's probe layout: the user turn, its close, then the
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


def _reference_hidden(model, prompt_ids, read_layer=None):
    """Read the marker's raw decoder output, or final post-norm output.

    HF replaces hidden_states[-1] with the normalized final output, so a
    decoder hook is necessary for the last raw layer. Save only the marker
    row and always remove the hook, including when forward raises.
    """
    import torch

    captured = []
    handle = None
    if read_layer is not None:

        def capture(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden[0, -1].detach().clone())

        handle = model.model.layers[read_layer].register_forward_hook(capture)
    try:
        with torch.no_grad():
            output = model.model(input_ids=torch.tensor([prompt_ids]), use_cache=False)
        return (
            captured[0] if read_layer is not None else output.last_hidden_state[0, -1]
        )
    finally:
        if handle is not None:
            handle.remove()


def _pick_labels(tok, num_labels):
    """The first ``num_labels`` single-token words from ``_LABEL_CANDIDATES``."""
    labels, ids = [], []
    for c in _LABEL_CANDIDATES:
        e = tok.encode(c, add_special_tokens=False)
        if len(e) == 1 and e[0] not in ids:
            labels.append(c)
            ids.append(e[0])
        if len(labels) == num_labels:
            return labels, ids
    raise RuntimeError(f"only {len(labels)} single-token labels, need {num_labels}")


def _fixed_head(seed, num_labels, hidden_size, scale=0.5):
    """Fixed seed-``seed`` weight (std ``scale``) and zero bias. Do not search seeds."""
    import torch

    g = torch.Generator().manual_seed(seed)
    weight = torch.randn((num_labels, hidden_size), generator=g, dtype=torch.float32)
    return weight * scale, torch.zeros(num_labels, dtype=torch.float32)


def _write_head(slot_dir, weight, bias, labels):
    from safetensors.torch import save_file

    from granite_switch.composer.weight_transfer import PROBE_HEAD_FILE

    os.makedirs(slot_dir, exist_ok=True)
    save_file(
        {"weight": weight.contiguous(), "bias": bias.contiguous()},
        os.path.join(slot_dir, PROBE_HEAD_FILE),
    )
    json.dump(
        {"kind": "probe", "labels": labels},
        open(os.path.join(slot_dir, "adapter_config.json"), "w"),
    )
    open(os.path.join(slot_dir, "io.yaml"), "w").write(
        f"name: {os.path.basename(slot_dir)}\nmodel: ~\n"
    )


def _prompt(tok, control_id):
    return [*tok.encode("Is this text safe?", add_special_tokens=False), control_id]


def compose_at_layer(base, outdir, read_layer, control_offset=5, num_labels=2):
    """Compose one probe slot reading ``read_layer``, plus its fp32 HF reference."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from granite_switch.composer.compose_utils import GraniteSwitchComposer

    tok = AutoTokenizer.from_pretrained(base)
    labels, ids = _pick_labels(tok, num_labels)
    control_id = tok.vocab_size - control_offset
    prompt_ids = _prompt(tok, control_id)

    plain = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    reference_hidden = _reference_hidden(plain, prompt_ids, read_layer)
    del plain

    weight, bias = _fixed_head(
        0, len(ids), AutoConfig.from_pretrained(base).hidden_size
    )
    slot = os.path.join(outdir, "detect")
    _write_head(slot, weight, bias, labels)

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base,
        adapter_paths=[slot],
        adapter_token_ids=[control_id],
        adapter_substitute_token_ids=[control_id],
        adapter_names=["detect"],
        adapter_kinds=["probe"],
        probe_label_token_ids=[ids],
        probe_read_layers=[read_layer],
    )
    ckpt = os.path.join(outdir, "composed")
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    del model

    scores = weight.double() @ reference_hidden.double() + bias.double()
    return {
        "ckpt": ckpt,
        "control_id": control_id,
        "label_token_ids": ids,
        "prompt_ids": prompt_ids,
        "read_layer": read_layer,
        "ground_truth_label_id": ids[int(scores.argmax())],
        "ground_truth_scores": scores.tolist(),
    }


def compose_two_slot(base, outdir, read_layer, control_offset=5, num_labels=2):
    """Compose a final-layer and a ``read_layer`` slot, plus fp32 HF references."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from granite_switch.composer.compose_utils import GraniteSwitchComposer

    tok = AutoTokenizer.from_pretrained(base)
    labels, ids = _pick_labels(tok, num_labels)
    hidden_size = AutoConfig.from_pretrained(base).hidden_size
    control_ids = [tok.vocab_size - control_offset, tok.vocab_size - control_offset - 1]
    heads = [_fixed_head(seed, len(ids), hidden_size) for seed in (0, 1)]
    slot_dirs = [
        os.path.join(outdir, "detect_final"),
        os.path.join(outdir, "detect_mid"),
    ]
    for d, (w, b) in zip(slot_dirs, heads):
        _write_head(d, w, b, labels)

    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=base,
        adapter_paths=slot_dirs,
        adapter_token_ids=control_ids,
        adapter_substitute_token_ids=control_ids,
        adapter_names=["detect_final", "detect_mid"],
        adapter_kinds=["probe", "probe"],
        probe_label_token_ids=[ids, ids],
        probe_read_layers=[None, read_layer],
    )
    ckpt = os.path.join(outdir, "composed_two_slot")
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    del model

    plain = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    result = {"ckpt": ckpt, "label_token_ids": ids, "read_layer": read_layer}
    for name, control_id, (w, b), layer in zip(
        ("slot1", "slot2"), control_ids, heads, (None, read_layer)
    ):
        prompt_ids = _prompt(tok, control_id)
        hidden = _reference_hidden(plain, prompt_ids, layer)
        scores = w.double() @ hidden.double() + b.double()
        result[f"{name}_prompt_ids"] = prompt_ids
        result[f"{name}_reference_hidden"] = hidden.tolist()
        result[f"{name}_ground_truth_label_id"] = ids[int(scores.argmax())]
        result[f"{name}_ground_truth_scores"] = scores.tolist()
    return result


def _record_probe_rows(llm):
    """Install after startup; PP callbacks must be importable for standard pickle."""
    from tests.shared.probe_recorder import install_probe_recorder

    llm.apply_model(install_probe_recorder)


def _read_probe_rows(llm):
    from tests.shared.probe_recorder import read_probe_records

    return [row for rows in llm.apply_model(read_probe_records) for row in rows]


def run_two_slot(ckpt, prompt_ids_1, prompt_ids_2, label_token_ids, out_path, *, eager):
    """Serve both probe slots' prompts, one request each, eager or FULL.

    Under FULL both requests share one captured batch shape, exercising
    replay with two different slots/read layers. Reports per-label logprobs
    and the recorded probe row (packed scores, rewritten label logits,
    and for the final-layer slot its actual activation and loaded head).
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=ckpt,
        enforce_eager=eager,
        gpu_memory_utilization=0.55,
        max_model_len=4096,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    if not eager:
        kw["compilation_config"] = {"cudagraph_mode": "FULL"}
    llm = LLM(**kw)
    _record_probe_rows(llm)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(label_token_ids))

    results = []
    for expected_slot, prompt_ids in ((1, prompt_ids_1), (2, prompt_ids_2)):
        out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0].outputs[
            0
        ]
        records = _read_probe_rows(llm)
        recorded_slots = [r["slot"] for r in records]
        if recorded_slots != [expected_slot]:
            raise RuntimeError(
                f"expected one recorded slot-{expected_slot} row, got {recorded_slots}"
            )
        lp_dict = out.logprobs[0]
        label_logprobs = [
            lp_dict[tid].logprob if tid in lp_dict else None for tid in label_token_ids
        ]
        results.append(
            {
                "token_id": int(out.token_ids[0]),
                "text": out.text,
                "label_logprobs": label_logprobs,
                **{k: v for k, v in records[0].items() if k != "slot"},
            }
        )

    json.dump({"slot1": results[0], "slot2": results[1]}, open(out_path, "w"))


def run_at_layer(ckpt, prompt_ids, label_token_ids, out_path, full_cudagraph=False):
    """Serve one prompt; report the emitted token and every label's logprob."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(
        model=ckpt,
        gpu_memory_utilization=0.55,
        max_model_len=4096,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    if full_cudagraph:
        kw["enforce_eager"] = False
        kw["compilation_config"] = {"cudagraph_mode": "FULL"}
    else:
        kw["enforce_eager"] = True
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(label_token_ids))
    out = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sp)[0].outputs[0]
    lp_dict = out.logprobs[0]
    label_logprobs = [
        lp_dict[tid].logprob if tid in lp_dict else None for tid in label_token_ids
    ]
    json.dump(
        {
            "token_id": int(out.token_ids[0]),
            "text": out.text,
            "label_logprobs": label_logprobs,
        },
        open(out_path, "w"),
    )


def run_sr_probe(ckpt, out_path, *, eager, pp_size=1):
    """Serve each SR-checkpoint probe and an SR request, sequentially and mixed.

    Each slot is selected on its own: a probe request carries only its marker,
    and the SR request carries only the SR control token.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if pp_size > 1:
        # Only trusted, importable test callbacks cross the local RPC boundary.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from tests.composer.test_sr_compose_e2e import CONTROL_TOKEN_ID, HIDDEN

    labels = list(range(100, 100 + HIDDEN))
    config = AutoConfig.from_pretrained(ckpt)
    markers = config.probe_control_token_ids
    kw = dict(
        model=ckpt,
        skip_tokenizer_init=True,
        enforce_eager=eager,
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        max_logprobs=HIDDEN + 1,
        enable_prefix_caching=False,
        pipeline_parallel_size=pp_size,
    )
    if pp_size > 1:
        kw["distributed_executor_backend"] = "mp"
    if not eager:
        kw["compilation_config"] = {
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 16],
        }
    llm = LLM(**kw)
    _record_probe_rows(llm)
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=HIDDEN + 1)
    sr_request = [10, CONTROL_TOKEN_ID, 20, 30]

    def serve(prompts):
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], sampling
        )
        return {
            "records": _read_probe_rows(llm),
            "outputs": [
                {
                    "token_id": int(o.outputs[0].token_ids[0]),
                    "label_logprobs": [
                        o.outputs[0].logprobs[0][tid].logprob
                        if tid in o.outputs[0].logprobs[0]
                        else None
                        for tid in labels
                    ],
                    "has_non_label_support": any(
                        tid not in labels and lp.logprob > float("-inf")
                        for tid, lp in o.outputs[0].logprobs[0].items()
                    ),
                }
                for o in outputs
            ],
        }

    # Each probe twice: the second run is the replay of the first's shape.
    result = {
        "sequential": [
            serve([[10, 21, 20, marker]]) for marker in markers for _ in range(2)
        ]
    }
    mixed = [[10, 21, 20, markers[0]], sr_request, [11, 21, 20, markers[-1]]]
    if len(markers) > 1:
        mixed.append([12, 21, 20, markers[1]])
    result["mixed"] = serve(mixed)
    with open(out_path, "w") as f:
        json.dump(result, f)


def run_full_cudagraph_replay(
    ckpt, control_id, label_token_ids, out_path, *, eager=False
):
    """Run identical fixed-length prompts in eager or FULL mode.

    Includes six changed inputs, a plain request, and a mixed batch. Markers
    remain last. Equal prompt lengths permit graph reuse; this test does not
    instrument the dispatcher to count actual graph replays.
    """
    import math
    import random

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    label_ids = set(label_token_ids)
    kw = dict(
        model=ckpt,
        enforce_eager=eager,
        gpu_memory_utilization=0.55,
        max_model_len=4096,
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    if not eager:
        kw["compilation_config"] = {"cudagraph_mode": "FULL"}
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(label_ids) + 1)

    def summarize(output):
        return {
            "token_id": int(output.token_ids[0]),
            "text": output.text,
            "is_label": int(output.token_ids[0]) in label_ids,
            "has_non_label_support": any(
                tid not in label_ids and math.isfinite(lp.logprob)
                for tid, lp in output.logprobs[0].items()
            ),
        }

    rng = random.Random(0)
    base_len = 40
    results = []
    for _ in range(6):
        body = [rng.randint(1000, 2000) for _ in range(base_len)]
        prompt = TokensPrompt(prompt_token_ids=[*body, control_id])
        o = llm.generate([prompt], sp)[0].outputs[0]
        results.append(summarize(o))

    # Same shape (base_len + 1 tokens), no probe marker at all -- the
    # verdict from the PRECEDING call must not leak into this one. This is a
    # SEQUENTIAL check (separate generate() calls), not a same-batch mix.
    plain_body = [rng.randint(1000, 2000) for _ in range(base_len + 1)]
    plain_prompt = TokensPrompt(prompt_token_ids=plain_body)
    o = llm.generate([plain_prompt], sp)[0].outputs[0]
    plain_result = summarize(o)

    # A GENUINE mixed batch: 3 requests in ONE generate() call -- probe,
    # plain, probe -- so vLLM schedules and samples all three rows in the
    # same forward pass. Each must get its own correct result: the plain
    # row's logits must not pick up either probe row's verdict, and the
    # two probe rows (same slot here, since this checkpoint has one)
    # must each still fire independently.
    mix_cls_body_1 = [rng.randint(1000, 2000) for _ in range(base_len)]
    mix_plain_body = [rng.randint(1000, 2000) for _ in range(base_len + 1)]
    mix_cls_body_2 = [rng.randint(1000, 2000) for _ in range(base_len)]
    mix_prompts = [
        TokensPrompt(prompt_token_ids=[*mix_cls_body_1, control_id]),
        TokensPrompt(prompt_token_ids=mix_plain_body),
        TokensPrompt(prompt_token_ids=[*mix_cls_body_2, control_id]),
    ]
    mix_outputs = llm.generate(mix_prompts, sp)
    mixed_results = [summarize(o.outputs[0]) for o in mix_outputs]

    json.dump(
        {
            "results": results,
            "plain_result": plain_result,
            "mixed_results": mixed_results,
        },
        open(out_path, "w"),
    )


def run_eager_reference(ckpt, control_id, out_path):
    # Reuse the exact prompt sequence, including the plain and mixed requests.
    run_full_cudagraph_replay(
        ckpt, control_id, json.loads(os.environ["CLS_LABEL_IDS"]), out_path, eager=True
    )


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
    if mode in ("compose-sr-probe", "compose-sr-probes"):
        from pathlib import Path

        from tests.composer.test_sr_compose_e2e import _compose_sr_probe

        read_layers = [0, 1, None] if mode == "compose-sr-probes" else [None]
        model = _compose_sr_probe(Path(os.environ["CLS_OUT"]), read_layers=read_layers)
        ckpt = os.path.join(os.environ["CLS_OUT"], "composed")
        model.save_pretrained(ckpt)
        print(json.dumps({"result": {"ckpt": ckpt}}))
        return
    if mode in ("run-sr-eager", "run-sr-full", "run-sr-pp-eager", "run-sr-pp-full"):
        run_sr_probe(
            os.environ["CLS_CKPT"],
            sys.argv[3] if mode.startswith("run-sr-pp-") else sys.argv[2],
            eager=mode.endswith("eager"),
            pp_size=int(sys.argv[2]) if mode.startswith("run-sr-pp-") else 1,
        )
        return
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
    if mode == "run-full-cudagraph":
        out_path = sys.argv[2]
        control_id = int(os.environ["CLS_CONTROL_ID"])
        label_token_ids = json.loads(os.environ["CLS_LABEL_IDS"])
        try:
            run_full_cudagraph_replay(
                os.environ["CLS_CKPT"], control_id, label_token_ids, out_path
            )
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    if mode == "run-eager-reference":
        out_path = sys.argv[2]
        control_id = int(os.environ["CLS_CONTROL_ID"])
        try:
            run_eager_reference(os.environ["CLS_CKPT"], control_id, out_path)
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    if mode == "compose-at-layer":
        read_layer = int(sys.argv[2])
        try:
            meta = compose_at_layer(
                os.environ["CLS_BASE"], os.environ["CLS_OUT"], read_layer
            )
            print(json.dumps({"result": meta}))
        except Exception as e:
            print(json.dumps({"error": f"{e}\n{traceback.format_exc()}"}))
        return
    if mode == "run-at-layer":
        out_path = sys.argv[2]
        prompt_ids = json.load(open(os.environ["CLS_PIDS"]))
        label_token_ids = json.loads(os.environ["CLS_LABEL_IDS"])
        try:
            run_at_layer(os.environ["CLS_CKPT"], prompt_ids, label_token_ids, out_path)
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    if mode == "run-at-layer-full-cudagraph":
        out_path = sys.argv[2]
        prompt_ids = json.load(open(os.environ["CLS_PIDS"]))
        label_token_ids = json.loads(os.environ["CLS_LABEL_IDS"])
        try:
            run_at_layer(
                os.environ["CLS_CKPT"],
                prompt_ids,
                label_token_ids,
                out_path,
                full_cudagraph=True,
            )
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    if mode == "compose-two-slot":
        read_layer = int(sys.argv[2])
        try:
            meta = compose_two_slot(
                os.environ["CLS_BASE"], os.environ["CLS_OUT"], read_layer
            )
            print(json.dumps({"result": meta}))
        except Exception as e:
            print(json.dumps({"error": f"{e}\n{traceback.format_exc()}"}))
        return
    if mode in ("run-two-slot-eager", "run-two-slot-full-cudagraph"):
        out_path = sys.argv[2]
        prompt_ids_1 = json.loads(os.environ["CLS_PIDS_1"])
        prompt_ids_2 = json.loads(os.environ["CLS_PIDS_2"])
        label_token_ids = json.loads(os.environ["CLS_LABEL_IDS"])
        try:
            run_two_slot(
                os.environ["CLS_CKPT"],
                prompt_ids_1,
                prompt_ids_2,
                label_token_ids,
                out_path,
                eager=mode == "run-two-slot-eager",
            )
        except Exception as e:
            json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
            sys.exit(1)
        return
    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    if sys.argv[1].startswith("run-sr-pp-"):
        # Like _pp_generation_worker, contain distributed-worker teardown in
        # this subprocess so a completed test cannot hang in Python finalizers.
        exit_code = 1
        try:
            main()
            exit_code = 0
        except BaseException:
            traceback.print_exc()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(exit_code)
    else:
        main()
