# SPDX-License-Identifier: Apache-2.0
"""Serve a composed Sudoku loop checkpoint with vLLM and score its answers.

    python serve_vllm.py <composed ckpt> <data dir> [--first 2000] [--count 20000]
        [--json out.json] [--reference N] [--eager] [--mix]

Each test puzzle is sent as the loop slot's prompt (the same layout as
training: grid, the template's turn close, then the slot's marker). The
server emits the whole answer, 81 digits then EOS, in the request's prefill
step; this script checks that and scores the answers like eval_head.py.

``--reference N`` re-solves the first N puzzles with the HF backend from the
same fixed start the server uses (``LoopHead.fixed_start``) and reports how
many answers agree exactly. ``--mix`` interleaves ordinary chat prompts with
the loop prompts, to check that the other requests still decode normally.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sudoku_data import check


def prompts_for(tok, slot_name, end_offset, boards):
    from train_sudoku import Prompt

    pr = Prompt(tok, slot_name, end_offset)
    cells = [pr.cell_ids[0].item()] + [pr.cell_ids[d].item() for d in range(1, 10)]
    return pr, [pr.head + [cells[c] for c in b] + pr.tail for b in boards]


def hf_reference(ckpt, prompts, pr, spec, device):
    """The HF backend's answers to ``prompts`` from the server's fixed start."""
    import torch

    from granite_switch.hf import GraniteSwitchForCausalLM

    model = (
        GraniteSwitchForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    slot = model.config.adapter_kinds.index("loop") + 1
    head = model.model.loop_heads.head(slot)
    z0 = head.fixed_start()
    layer = spec["read_layer"]
    answers = []
    for ids in prompts:
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(input_ids=x, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        # hidden_states: [embeddings (scaled), layer 0 out, ..., final norm].
        read = hs[0] if layer == -1 else hs[-1] if layer is None else hs[layer + 1]
        grid = read[:, pr.grid_start : pr.grid_start + spec["grid_size"]]
        amp = spec.get("solve_dtype") == "bfloat16"
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, _ = head.solve_each(
                grid,
                z0,
                max_iter=spec["max_iter"],
                fp_thresh=spec["fp_thresh"],
                decay=spec["stepsize_decay"],
                patience=spec["decay_patience"],
                min_iter=spec.get("min_iter", 0),
            )
        answers.append((logits[0].float().argmax(-1) + 1).tolist())
    del model
    torch.cuda.empty_cache()
    return answers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("data")
    p.add_argument("--first", type=int, default=2000)
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--json", default=None)
    p.add_argument("--reference", type=int, default=0)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--mix", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=64)
    a = p.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.vllm import register

    register()
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = GraniteSwitchConfig.from_pretrained(a.model)
    slot = cfg.adapter_kinds.index("loop")
    spec = cfg.loop_specs[slot]
    digit_ids = spec["output_token_ids"]
    test = np.load(Path(a.data) / "test.npz")
    lo, hi = a.first, min(len(test["x"]), a.first + a.count)
    boards, sols = test["x"][lo:hi], test["y"][lo:hi]
    pr, loop_prompts = prompts_for(
        tok, cfg.adapter_names[slot], spec.get("end_offset", 0), boards
    )

    reference = None
    if a.reference:
        reference = hf_reference(
            a.model, loop_prompts[: a.reference], pr, spec, torch.device("cuda")
        )

    chats = []
    if a.mix:
        for q in ("What is the capital of France?", "Name three prime numbers."):
            text = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            chats.append(tok.encode(text, add_special_tokens=False))

    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=a.max_num_seqs,
        gpu_memory_utilization=0.85,
        enforce_eager=a.eager,
    )
    requests = [TokensPrompt(prompt_token_ids=ids) for ids in loop_prompts]
    mixed = []
    for i, ids in enumerate(chats):
        mixed.append(len(requests) // (len(chats) + 1) * (i + 1) + i)
        requests.insert(mixed[-1], TokensPrompt(prompt_token_ids=ids))
    params = SamplingParams(temperature=0.0, max_tokens=120)
    t0 = time.time()
    outs = llm.generate(requests, params)
    elapsed = time.time() - t0
    chat_out = [outs[i].outputs[0] for i in mixed]
    outs = [o for i, o in enumerate(outs) if i not in set(mixed)]

    digit_of = {t: d + 1 for d, t in enumerate(digit_ids)}
    eos = tok.eos_token_id
    cells = exact = valid = blank_right = shape_ok = agree = 0
    blanks = int((boards == 0).sum())
    sample = []
    for k, (o, b, s) in enumerate(zip(outs, boards, sols)):
        ids = list(o.outputs[0].token_ids)
        # The burst: exactly 81 digit tokens, then EOS, finished by stop.
        ok = (
            len(ids) == spec["grid_size"] + 1
            and ids[-1] == eos
            and all(t in digit_of for t in ids[:-1])
            and o.outputs[0].finish_reason == "stop"
        )
        shape_ok += ok
        pred = np.array([digit_of.get(t, 0) for t in ids[: spec["grid_size"]]])
        if pred.size < spec["grid_size"]:
            pred = np.pad(pred, (0, spec["grid_size"] - pred.size))
        cells += int((pred == s).sum())
        blank_right += int(((pred == s) & (b == 0)).sum())
        exact += int((pred == s).all())
        valid += int(check(b, pred))
        if reference is not None and k < len(reference):
            agree += int(pred.tolist() == reference[k])
        if k < 3:
            sample.append("".join(map(str, pred.tolist())))
    n = len(boards)
    res = {
        "n": n,
        "test_slice": [lo, hi],
        "answer_as_one_burst": shape_ok / n,
        "exact_acc": exact / n,
        "cell_acc": cells / (81 * n),
        "blank_cell_acc": blank_right / blanks,
        "valid_acc": valid / n,
        "seconds": round(elapsed, 1),
        "requests_per_s": round(n / elapsed, 2),
        "eager": a.eager,
        "sample_answers": sample,
    }
    if reference is not None:
        res["hf_reference_agree"] = agree / len(reference)
        res["hf_reference_n"] = len(reference)
    if chats:
        res["mixed_chat_answers"] = [c.text[:80] for c in chat_out]
        res["mixed_chat_ok"] = all(len(c.token_ids) > 1 for c in chat_out)
    print("LOOP_VLLM " + json.dumps(res), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
