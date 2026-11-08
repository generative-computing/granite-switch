# SPDX-License-Identifier: Apache-2.0
"""Record what one served loop-slot request returns, for one test puzzle.

    python dump_output.py <composed ckpt> <data dir> --index 2000 --out output.json

Runs the composed checkpoint's ordinary HF forward pass once on the puzzle's
prompt and keeps what comes back: the 81 grid rows of the logits (each -inf
except at the nine digit tokens), as an 81 x 9 score matrix with the token ids
of its columns, the answer read off by argmax, the solver's iteration count,
and one full logits row, summarized (which ids are finite).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_sudoku import Prompt
from transformers import AutoTokenizer

from granite_switch.hf import GraniteSwitchForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("data")
    p.add_argument("--index", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="repeat the request with this many random solver starts (FPRM draws one per solve)",
    )
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    m = (
        GraniteSwitchForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    spec = m.config.loop_specs[0]
    pr = Prompt(tok, m.config.adapter_names[0], spec["end_offset"])
    test = np.load(Path(a.data) / "test.npz")
    board, solution = test["x"][a.index], test["y"][a.index]
    ids = pr.batch(torch.as_tensor(board[None], device=device))

    # The solver starts from a random state (FPRM's), so one request can land on
    # a different fixed point than another: record how often each seed solves it.
    seeds = []
    for seed in range(1, a.seeds):
        torch.manual_seed(seed)
        with torch.no_grad():
            lg = m(input_ids=ids, use_cache=False).logits[0].float()
        ans = (
            lg[pr.grid_start : pr.grid_start + 81][:, spec["output_token_ids"]].argmax(
                -1
            )
            + 1
        )
        seeds.append(
            {
                "seed": seed,
                "cells_right": int((ans.cpu().numpy() == solution).sum()),
                "iters": int(m.model._last_loop_iters[0]),
            }
        )
    torch.manual_seed(0)
    with torch.no_grad():
        logits = m(input_ids=ids, use_cache=False).logits[0].float()
    rows = logits[pr.grid_start : pr.grid_start + 81]  # [81, vocab]
    digit_ids = spec["output_token_ids"]
    scores = rows[:, digit_ids]  # [81, 9]
    answer = scores.argmax(-1) + 1
    finite = torch.isfinite(rows[0]).nonzero().flatten().tolist()
    out = {
        "index": a.index,
        "board": "".join(map(str, board.tolist())),
        "solution": "".join(map(str, solution.tolist())),
        "answer": "".join(map(str, answer.tolist())),
        "cells_right": int((answer.cpu().numpy() == solution).sum()),
        "solved": bool((answer.cpu().numpy() == solution).all()),
        "solver_iters": int(m.model._last_loop_iters[0]),
        "digit_token_ids": digit_ids,
        "digit_tokens": tok.convert_ids_to_tokens(digit_ids),
        "scores": [[round(v, 3) for v in r] for r in scores.tolist()],
        "probs": [[round(v, 4) for v in r] for r in torch.softmax(scores, -1).tolist()],
        "prompt_tokens": ids.shape[1],
        "logits_shape": list(logits.shape),
        "grid_rows": [pr.grid_start, pr.grid_start + 81],
        "row0_finite_ids": finite,
        "row0_vocab": rows.shape[1],
        "other_seeds": seeds,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(
        "OUTPUT_DUMP "
        + json.dumps(
            {
                k: out[k]
                for k in (
                    "index",
                    "cells_right",
                    "solved",
                    "solver_iters",
                    "logits_shape",
                    "row0_finite_ids",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
