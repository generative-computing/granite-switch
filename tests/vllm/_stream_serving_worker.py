# SPDX-License-Identifier: Apache-2.0
"""Worker for ``test_stream_serving.py``: a streams checkpoint through vLLM.

Each mode runs in its own subprocess, so only one engine is resident::

    compose <out_dir> <out.json>
        CPU, before any CUDA init: compose the tiny checkpoint, save it, and
        write the HF fp32 reference rows.
    serve <ckpt> <eager|full> <tp> <pp> <out.json>
        GPU: serve every head request and an SR request, one at a time (twice,
        the second a replay of the first's shape) and in one mixed batch.

The checkpoint is composed with ``streams=`` (compact LoRA rows), from a tiny
base and one Shadow Residual adapter, with three heads:

* ``tox``    -- an identity probe on the base model, read after the final norm;
* ``tox_sr`` -- the same identity probe, on the SR adapter (option B);
* ``nudge``  -- the example ``mlp_head`` plugin, in merge mode.

The plugin type is visible only through the ``granite_switch.streams`` entry
point: the test puts a dist-info on PYTHONPATH, and nothing here imports the
plugin, so compose, HF and every vLLM process must discover it.
"""

import json
import os
import sys
import traceback
from pathlib import Path

HEADS = ("tox", "tox_sr", "nudge")
PROMPT = [10, 21, 20]
NUDGE_LABELS = [100, 101, 102, 103]
MERGE_WEIGHT = 0.5
WIDTH = 8


def _tokens():
    from tests.composer.test_sr_compose_e2e import CONTROL_TOKEN_ID

    return {name: CONTROL_TOKEN_ID + i for i, name in enumerate(("sr", *HEADS))}


def _labels():
    from tests.composer.test_sr_compose_e2e import HIDDEN

    return list(range(100, 100 + HIDDEN))


def _requests():
    tokens = _tokens()
    requests = {name: [*PROMPT, tokens[name]] for name in HEADS}
    requests["sr"] = [10, tokens["sr"], 20, 30]
    return requests


def compose(out_dir, out_path):
    import torch
    from safetensors.torch import save_file

    from granite_switch.composer import GraniteSwitchComposer
    from granite_switch.hf import GraniteSwitchForCausalLM
    from granite_switch.streams.types.probe import PROBE_HEAD_FILE
    from tests.composer.test_sr_compose_e2e import (
        HIDDEN,
        _create_base_model,
        _create_sr_adapter,
    )

    out = Path(out_dir)
    base, sr, probe, nudge = (out / n for n in ("base", "sr", "probe", "nudge"))
    for d in (base, probe, nudge):
        d.mkdir(parents=True)
    _create_base_model(base)
    _create_sr_adapter(sr)
    save_file(
        {"weight": torch.eye(HIDDEN), "bias": torch.zeros(HIDDEN)},
        str(probe / PROBE_HEAD_FILE),
    )
    g = torch.Generator().manual_seed(3)
    save_file(
        {
            "w1": torch.randn(WIDTH, HIDDEN, generator=g) * 0.3,
            "w2": torch.randn(len(NUDGE_LABELS), WIDTH, generator=g),
        },
        str(nudge / "mlp_head.safetensors"),
    )
    labels = _labels()
    tokens = _tokens()
    streams = [
        {"type": "lora"},
        {
            "type": "probe",
            "head_row": 0,
            "output": {"mode": "select", "token_ids": labels},
        },
        {
            "type": "probe",
            "head_row": 1,
            "adapter": "sr",
            "output": {"mode": "select", "token_ids": labels},
        },
        {
            "type": "mlp_head",
            "head_row": 0,
            "output": {
                "mode": "merge",
                "token_ids": NUDGE_LABELS,
                "weight": MERGE_WEIGHT,
            },
            "settings": {"width": WIDTH},
        },
    ]
    model = GraniteSwitchComposer.from_base_and_adapters(
        base_model_name_or_path=str(base),
        adapter_paths=[str(sr), str(probe), str(probe), str(nudge)],
        adapter_token_ids=[tokens[n] for n in ("sr", *HEADS)],
        adapter_substitute_token_ids=[1, 2, 2, 2],
        adapter_names=["sr", *HEADS],
        streams=streams,
        switch_head_dim=16,
    ).eval()
    ckpt = out / "composed"
    model.save_pretrained(ckpt)

    # The HF fp32 reference, from the saved checkpoint as vLLM will load it.
    hf = GraniteSwitchForCausalLM.from_pretrained(ckpt, dtype=torch.float32).eval()

    def last_row(ids):
        with torch.no_grad():
            return hf(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]

    requests = _requests()
    reference = {}
    for name in ("tox", "tox_sr"):
        reference[name] = last_row(requests[name])[labels].tolist()
    merged = last_row(requests["nudge"])
    with torch.no_grad():
        hf.model.stream_mlp_head.w2.zero_()
    lm = last_row(requests["nudge"])
    reference["nudge"] = merged[NUDGE_LABELS].tolist()
    reference["nudge_lm"] = lm[NUDGE_LABELS].tolist()
    json.dump(
        {"ckpt": str(ckpt), "reference": reference, "labels": labels},
        open(out_path, "w"),
    )


def _install(llm):
    from tests.shared.probe_recorder import install_probe_recorder

    llm.apply_model(install_probe_recorder)


def _records(llm):
    """This step's head rows, from one worker.

    With tensor parallelism every rank computes the logits and records the
    same rows; with pipeline parallelism only the last stage does.
    """
    from tests.shared.probe_recorder import read_probe_records

    return next((rows for rows in llm.apply_model(read_probe_records) if rows), [])


def serve(ckpt, mode, tp, pp, out_path):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if tp * pp > 1:
        # Only trusted, importable test callbacks cross the RPC boundary.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    labels = _labels()
    kw = dict(
        model=ckpt,
        skip_tokenizer_init=True,
        enforce_eager=mode == "eager",
        dtype="bfloat16",
        gpu_memory_utilization=0.3,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        max_logprobs=len(labels) + 1,
        enable_prefix_caching=False,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )
    if tp * pp > 1:
        kw["distributed_executor_backend"] = "mp"
    if mode == "full":
        kw["compilation_config"] = {
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4, 16],
        }
    llm = LLM(**kw)
    _install(llm)
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(labels) + 1)
    requests = _requests()

    def run(names):
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=requests[n]) for n in names], sampling
        )
        return {
            "records": _records(llm),
            "outputs": [
                {
                    "token_id": int(o.outputs[0].token_ids[0]),
                    "has_non_label_support": any(
                        tid not in labels and lp.logprob > float("-inf")
                        for tid, lp in o.outputs[0].logprobs[0].items()
                    ),
                }
                for o in outputs
            ],
        }

    order = [*HEADS, "sr"]
    result = {
        "sequential": {name: [run([name]), run([name])] for name in order},
        "mixed": run(order),
    }
    json.dump(result, open(out_path, "w"))


def main():
    mode = sys.argv[1]
    out_path = sys.argv[-1]
    try:
        if mode == "compose":
            compose(sys.argv[2], out_path)
        elif mode == "serve":
            serve(
                sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), out_path
            )
        else:
            raise SystemExit(f"unknown mode {mode!r}")
    except Exception as e:
        json.dump({"fatal": f"{e}\n{traceback.format_exc()}"}, open(out_path, "w"))
        raise


if __name__ == "__main__":
    distributed = len(sys.argv) > 5 and int(sys.argv[4]) * int(sys.argv[5]) > 1
    if distributed:
        # Contain distributed-worker teardown, as the PP probe worker does, so
        # a completed run cannot hang in Python finalizers.
        code = 1
        try:
            main()
            code = 0
        except BaseException:
            traceback.print_exc()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
        os._exit(code)
    else:
        main()
