# SPDX-License-Identifier: Apache-2.0
"""Serve the composed loop-slot checkpoint with vLLM and check its one-step answers.

Each loop request's answer (81 digit tokens and end of text) comes out of the
step that completes its prompt (see ``granite_switch.vllm.loop``). This script
measures that on Sudoku test puzzles, next to ordinary requests in the same
engine:

* **answers**: each loop request's completion is parsed back into 81 digits and
  scored like the HF evaluation (``eval_head.py``): exact, cells, blank cells;
* **one step**: driving the engine step by step, a loop request gets its 82
  tokens in a single step while an ordinary request next to it gets one per
  step;
* **HF agreement** (``--hf N``): for the first N puzzles, the composed
  checkpoint's HF forward with the same seeded start (``solve_rows``) must give
  the same digits;
* **mixed batch**: ordinary prompts (no loop marker) decode normally in the same
  ``generate`` call, and match a run of those prompts alone;
* **chunked prefill** (``--chunk N``): the same puzzles served with a small
  token budget, so grids span prefill chunks, give the same answers.

    python serve_vllm.py <composed ckpt> <data dir> --first 2000 --count 2000 --json out.json
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
from train_sudoku import Prompt

ORDINARY = [
    "Write one sentence about the sea.",
    "List three prime numbers.",
    "What is the capital of France?",
    "Say hello in Spanish.",
]


def loop_prompts(tok, slot, end_offset, boards):
    import torch

    pr = Prompt(tok, slot, end_offset)
    ids = pr.batch(torch.as_tensor(boards)).tolist()
    return pr, ids


def ordinary_prompts(tok):
    out = []
    for text in ORDINARY:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append(tok.encode(rendered, add_special_tokens=False))
    return out


def digits_of(token_ids, digit_ids):
    lut = {t: i + 1 for i, t in enumerate(digit_ids)}
    return [lut.get(t, 0) for t in token_ids[:81]]


def score(answers, boards, sols):
    exact = cells = blank_right = valid = 0
    blanks = int((boards == 0).sum())
    for a, b, s in zip(answers, boards, sols):
        pred = np.array(a)
        cells += int((pred == s).sum())
        blank_right += int(((pred == s) & (b == 0)).sum())
        exact += int((pred == s).all())
        valid += int(check(b, pred))
    n = len(answers)
    return {
        "exact_acc": exact / n,
        "cell_acc": cells / (81 * n),
        "blank_cell_acc": blank_right / blanks,
        "valid_acc": valid / n,
        "n": n,
    }


def install_solve_recorder(model):
    """Record each solve the engine runs: its seed, read grid, iterations, answer.

    Wraps the loop head's ``solve_rows`` inside the worker (``apply_model``),
    so the chunked-prefill check can tell a grid read that differs from one
    that is merely rounded differently.
    """
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM

    target = next(m for m in model.modules() if isinstance(m, GraniteSwitchForCausalLM))
    records = []
    target._loop_records = records
    for head in target.model.loop_heads.heads.values():
        original = head.solve_rows

        def recording(read, seeds, *args, _original=original, **kwargs):
            scores, state = _original(read, seeds, *args, **kwargs)
            for i, seed in enumerate(seeds):
                records.append(
                    {
                        "seed": int(seed),
                        "read": read[i].float().cpu(),
                        "iters": int(state["iters"][i]),
                        "residual": float(state["residual"][i]),
                        "answer": (scores[i].argmax(-1) + 1).tolist(),
                    }
                )
            return scores, state

        head.solve_rows = recording
    return True


def read_solve_records(model):
    from granite_switch.vllm.granite_switch_model import GraniteSwitchForCausalLM

    target = next(m for m in model.modules() if isinstance(m, GraniteSwitchForCausalLM))
    out = list(target._loop_records)
    target._loop_records.clear()
    return out


def run_engine(
    model,
    prompts,
    params,
    max_num_batched_tokens=None,
    logprobs=None,
    max_num_seqs=None,
    record=False,
):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kwargs = dict(
        model=model,
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs
    llm = LLM(**kwargs)
    if record:
        llm.apply_model(install_solve_recorder)
    sp = [
        SamplingParams(
            temperature=0.0,
            max_tokens=params[i][0],
            seed=params[i][1],
            logprobs=logprobs,
        )
        for i in range(len(prompts))
    ]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    elapsed = time.time() - t0
    records = llm.apply_model(read_solve_records)[0] if record else None
    del llm
    if record:
        return outs, elapsed, records
    return outs, elapsed


def step_trace(model, loop_prompt, plain_prompt, seed):
    """Drive the engine one step at a time: tokens each request gets per step.

    One loop request and one ordinary request run together. The loop request
    must get all its tokens in a single step; the ordinary one gets one per
    step, as usual.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.85
    )
    engine = llm.llm_engine
    engine.add_request(
        "loop",
        TokensPrompt(prompt_token_ids=loop_prompt),
        SamplingParams(temperature=0.0, max_tokens=200, seed=seed),
    )
    engine.add_request(
        "plain",
        TokensPrompt(prompt_token_ids=plain_prompt),
        SamplingParams(temperature=0.0, max_tokens=12),
    )
    seen = {"loop": 0, "plain": 0}
    trace = []
    while engine.has_unfinished_requests():
        step = {}
        for out in engine.step():
            n = len(out.outputs[0].token_ids)
            step[out.request_id] = n - seen[out.request_id]
            seen[out.request_id] = n
        trace.append(step)
    del llm
    return trace


def hf_answers(model_path, prompt_ids, seeds, spec):
    """The HF backend's answers, one request at a time, from the same seeded start.

    The composed checkpoint's ordinary HF forward, with ``loop_seeds`` set so
    it solves as vLLM serves (``solve_rows``); each grid row's answer is the
    argmax of its rewritten logits.
    """
    import torch

    from granite_switch.hf import GraniteSwitchForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GraniteSwitchForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    m = m.to(device).eval()
    digit_ids = torch.tensor(spec["output_token_ids"], device=device)
    answers = []
    for ids, seed in zip(prompt_ids, seeds):
        end = len(ids) - 1 - spec.get("end_offset", 0)
        start = end - spec["grid_size"]
        m.model.loop_seeds = [seed]
        with torch.no_grad():
            logits = m(
                input_ids=torch.tensor([ids], device=device), use_cache=False
            ).logits
        rows = logits[0, start:end][:, digit_ids]
        answers.append((rows.argmax(-1) + 1).tolist())
    del m
    return answers


def chunked_record(a, loop_ids, seeds, budget):
    """One engine (in this process): each solve's seed, read grid and answer.

    ``budget`` 0 serves the prompts whole, else with that prefill token budget.
    Both use the same small batch (max_num_seqs 8), so the budget is the only
    change. Saved with torch so the compare step can read both runs.
    """
    import torch

    n = min(a.chunk_count, len(loop_ids))
    outs, _, records = run_engine(
        a.model,
        loop_ids[:n],
        [(200, s) for s in seeds[:n]],
        max_num_batched_tokens=budget or None,
        max_num_seqs=8,
        record=True,
    )
    tokens = [list(o.outputs[0].token_ids) for o in outs]
    torch.save({"records": records, "tokens": tokens}, a.record_out)
    print(f"recorded {len(records)} solves (budget {budget}) to {a.record_out}")


def chunked_compare(whole_path, chunked_path, sols, lo, spec, budget):
    """Solve by solve: a read grid that differs by more than bf16 rounding would
    be a capture bug; a matching grid that solves to another answer is the
    solver amplifying rounding."""
    import torch

    whole = torch.load(whole_path, weights_only=False)
    chunked = torch.load(chunked_path, weights_only=False)
    by_seed = {r["seed"]: r for r in whole["records"]}
    rows = []
    for r in chunked["records"]:
        w = by_seed.get(r["seed"])
        if w is None:
            continue
        diff = (r["read"] - w["read"]).abs().max().item()
        scale = w["read"].abs().max().item()
        truth = sols[r["seed"] - lo].tolist()
        rows.append(
            {
                "seed": r["seed"],
                "read_rel_diff": diff / max(scale, 1e-6),
                "same": r["answer"] == w["answer"],
                "converged": r["iters"] < spec["max_iter"]
                and w["iters"] < spec["max_iter"],
                "whole_solved": w["answer"] == truth,
                "chunked_solved": r["answer"] == truth,
                "iters": [w["iters"], r["iters"]],
            }
        )
    conv = [x for x in rows if x["converged"]]
    k = max(len(rows), 1)
    pairs = list(zip(whole["tokens"], chunked["tokens"]))
    return {
        "budget": budget,
        "n": len(rows),
        "same_answers": sum(x["same"] for x in rows) / k,
        "same_when_both_converged": sum(x["same"] for x in conv) / max(len(conv), 1),
        "num_both_converged": len(conv),
        "max_read_rel_diff": max((x["read_rel_diff"] for x in rows), default=None),
        "median_read_rel_diff": float(np.median([x["read_rel_diff"] for x in rows]))
        if rows
        else None,
        "exact_whole": sum(x["whole_solved"] for x in rows) / k,
        "exact_chunked": sum(x["chunked_solved"] for x in rows) / k,
        "tokens_match_whole_run": sum(x == y for x, y in pairs) / max(len(pairs), 1),
        "solved_whole_not_chunked": sum(
            x["whole_solved"] and not x["chunked_solved"] for x in rows
        ),
        "solved_chunked_not_whole": sum(
            x["chunked_solved"] and not x["whole_solved"] for x in rows
        ),
        "differing": [x for x in rows if not x["same"]],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("data")
    p.add_argument("--first", type=int, default=2000)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--hf", type=int, default=0, help="puzzles to check against HF")
    p.add_argument(
        "--chunk", type=int, default=0, help="token budget for a chunked run"
    )
    p.add_argument("--chunk-count", type=int, default=64)
    p.add_argument("--json", default=None)
    p.add_argument("--tag", default="LOOP_VLLM")
    p.add_argument(
        "--record",
        type=int,
        default=None,
        metavar="BUDGET",
        help="only record one engine's solves (0: whole prompts), to --record-out",
    )
    p.add_argument("--record-out", default=None)
    p.add_argument(
        "--compare",
        nargs=2,
        default=None,
        metavar=("WHOLE", "CHUNKED"),
        help="only compare two --record files",
    )
    a = p.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if a.record is not None:
        # Each solve is read back through apply_model, as the probe tests do:
        # the engine runs in this process, so no function is sent to a worker.
        # One engine per process (run_serve_eval.sh), since an in-process
        # engine keeps its GPU memory until the process exits.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.vllm import register

    register()
    tok = AutoTokenizer.from_pretrained(a.model)
    config = GraniteSwitchConfig.from_pretrained(a.model)
    slot = config.adapter_kinds.index("loop")
    spec = config.loop_specs[slot]
    digit_ids = spec["output_token_ids"]
    eos = tok.eos_token_id

    test = np.load(Path(a.data) / "test.npz")
    lo, hi = a.first, min(len(test["x"]), a.first + a.count)
    boards, sols = test["x"][lo:hi], test["y"][lo:hi]
    _, loop_ids = loop_prompts(
        tok, config.adapter_names[slot], spec["end_offset"], boards
    )
    seeds = list(range(lo, hi))  # a fixed start per puzzle
    plain = ordinary_prompts(tok)

    if a.record is not None:
        chunked_record(a, loop_ids, seeds, a.record)
        return
    if a.compare:
        res = chunked_compare(*a.compare, sols, lo, spec, a.chunk)
        summary = {k: v for k, v in res.items() if k != "differing"}
        print(f"{a.tag}_CHUNKED " + json.dumps(summary), flush=True)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=2))
        return

    # 1. The main run: every puzzle plus the ordinary prompts, one engine.
    prompts = loop_ids + plain
    params = [(200, s) for s in seeds] + [(40, None)] * len(plain)
    outs, elapsed = run_engine(a.model, prompts, params, logprobs=1)
    loop_outs, plain_outs = outs[: len(loop_ids)], outs[len(loop_ids) :]

    answers, lengths, ends = [], [], 0
    for o in loop_outs:
        toks = list(o.outputs[0].token_ids)
        lengths.append(len(toks))
        ends += int(bool(toks) and toks[-1] == eos)
        answers.append(digits_of(toks, digit_ids))
    res = score(answers, boards, sols)
    res.update(
        {
            "test_slice": [lo, hi],
            "elapsed_s": round(elapsed, 1),
            "answer_lengths": sorted(set(lengths)),
            "answers_ending_in_eos": ends / len(loop_outs),
            "ordinary_outputs": [o.outputs[0].text for o in plain_outs],
            "logprob_example": [
                {str(k): round(v.logprob, 4) for k, v in lp.items()}
                for lp in (loop_outs[0].outputs[0].logprobs or [])[:3]
            ],
            "example": {
                "board": "".join(map(str, boards[0].tolist())),
                "answer": "".join(map(str, answers[0])),
                "solution": "".join(map(str, sols[0].tolist())),
                "text": loop_outs[0].outputs[0].text,
            },
        }
    )
    print(
        f"{a.tag}_MAIN "
        + json.dumps({k: v for k, v in res.items() if k != "ordinary_outputs"}),
        flush=True,
    )

    # 2. Step by step: a loop request and an ordinary one in the same engine.
    trace = step_trace(a.model, loop_ids[0], plain[0], seeds[0])
    loop_steps = [s["loop"] for s in trace if s.get("loop")]
    res["step_trace"] = {
        "steps": len(trace),
        "loop_tokens_per_step": loop_steps,
        "plain_tokens_per_step": [s["plain"] for s in trace if s.get("plain")],
    }
    print(f"{a.tag}_STEPS " + json.dumps(res["step_trace"]), flush=True)

    # 3. Ordinary prompts alone: the loop requests must not change them.
    alone, _ = run_engine(a.model, plain, [(40, None)] * len(plain))
    res["ordinary_match_alone"] = [
        x.outputs[0].token_ids == y.outputs[0].token_ids
        for x, y in zip(plain_outs, alone)
    ]
    print(f"{a.tag}_ORDINARY " + json.dumps(res["ordinary_match_alone"]), flush=True)

    # 4. Chunked prefill: run_serve_eval.sh runs it after this process, one
    #    engine per process (--record), then --compare.

    # 5. HF agreement on the first puzzles, same seeded start.
    if a.hf:
        n = min(a.hf, len(loop_ids))
        ref = hf_answers(a.model, loop_ids[:n], seeds[:n], spec)
        agree = [r == x for r, x in zip(ref, answers[:n])]
        cells = [
            sum(int(u == v) for u, v in zip(r, x)) for r, x in zip(ref, answers[:n])
        ]
        res["hf"] = {
            "n": n,
            "same_answer": sum(agree) / n,
            "mean_cells_same": float(np.mean(cells)),
        }
        print(f"{a.tag}_HF " + json.dumps(res["hf"]), flush=True)

    print(f"{a.tag} " + json.dumps(res), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
