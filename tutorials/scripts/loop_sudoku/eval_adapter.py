# SPDX-License-Identifier: Apache-2.0
"""Score a generation adapter (LoRA, aLoRA or SR) on Sudoku through Granite Switch.

The adapter is composed into a Granite Switch checkpoint like any other slot;
each test puzzle is rendered with the checkpoint's chat template and the
adapter's name (``adapter_name=``), and vLLM decodes the answer greedily. The
answer is parsed back into 81 digits (``adapter_data.parse_grid``) and scored
like the loop slot: cells right, puzzles solved exactly, and valid grids.

    python eval_adapter.py <composed ckpt> <data dir> --adapter sudoku [--first S] [--count N] [--json out.json]

``--adapter none`` scores the base model with no adapter (the same prompt).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from adapter_data import INSTRUCTION, format_grid, parse_grid
from sudoku_data import check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("data", help="folder with test.npz (sudoku_data.py)")
    p.add_argument(
        "--adapter", required=True, help="the slot's name, or 'none' for the base model"
    )
    p.add_argument(
        "--first",
        type=int,
        default=2000,
        help="first test puzzle (after the selection slice)",
    )
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    p.add_argument("--json", default=None)
    p.add_argument("--tag", default="ADAPTER_EVAL")
    a = p.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Greedy decoding needs no sampler kernel; FlashInfer's is compiled at start
    # and fails under an older system CUDA (the adapter benchmark turns it off too).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from granite_switch.vllm import register

    register()
    tok = AutoTokenizer.from_pretrained(a.model)
    test = np.load(Path(a.data) / "test.npz")
    lo, hi = a.first, min(len(test["x"]), a.first + a.count)
    boards, sols = test["x"][lo:hi], test["y"][lo:hi]
    kwargs = {} if a.adapter == "none" else {"adapter_name": a.adapter}
    prompts = []
    for b in boards:
        msgs = [{"role": "user", "content": f"{INSTRUCTION}\n\n{format_grid(b)}"}]
        # Render to text, then tokenize: the control token is a special token
        # in the rendered text, so it encodes to its own id.
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs
        )
        ids = tok.encode(text, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    llm = LLM(
        model=a.model,
        dtype="bfloat16",
        tensor_parallel_size=a.tp,
        max_model_len=1024,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
    )
    outs = llm.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    )

    cells = exact = valid = parsed = blank_right = 0
    blanks = int((boards == 0).sum())
    sample = []
    for k, (o, b, s) in enumerate(zip(outs, boards, sols)):
        text = o.outputs[0].text
        grid = parse_grid(text)
        if k < 3:
            sample.append(text)
        if grid is None:
            continue
        parsed += 1
        pred = np.array(grid)
        cells += int((pred == s).sum())
        blank_right += int(((pred == s) & (b == 0)).sum())
        exact += int((pred == s).all())
        valid += int(check(b, pred))
    n = len(boards)
    res = {
        "adapter": a.adapter,
        "cell_acc": cells / (81 * n),
        # Only the cells the puzzle leaves blank: the givens are copied.
        "blank_cell_acc": blank_right / blanks,
        "exact_acc": exact / n,
        "valid_acc": valid / n,
        "parsed": parsed / n,
        "n": n,
        "test_slice": [lo, hi],
        "sample_answers": sample,
    }
    print(f"{a.tag} " + json.dumps(res), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
