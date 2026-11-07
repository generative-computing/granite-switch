# SPDX-License-Identifier: Apache-2.0
"""Score loop heads on the test split through the served HF path (GPU).

    python tutorials/scripts/loop_sudoku/eval_head.py <composed ckpt> <data dir> <head>... [--count N] [--first S]
        [--max-iter K] [--min-iter M] [--json out.json]

A head is a loop_head.safetensors, loaded into the slot's loop head, or the word
``composed``: the checkpoint as composed, with nothing loaded. Each is scored on
test puzzles [S, S + N) (the split is shuffled once, so a slice is a uniform
sample). Under torchrun every GPU scores its own shard of the slice. (The
options avoid the prefixes of torchrun's own, such as --start-method and
--nnodes, which torchrun would reject as ambiguous.)
"""

import argparse
import copy
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
from train_sudoku import Prompt, evaluate
from transformers import AutoTokenizer

from granite_switch.hf import GraniteSwitchForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("data")
    p.add_argument("heads", nargs="+")
    p.add_argument("--count", type=int, default=10000, help="test puzzles to score")
    p.add_argument(
        "--first",
        type=int,
        default=0,
        help="first test puzzle (skip the selection slice)",
    )
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--min-iter", type=int, default=None)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--json", default=None, help="write the last head's result here")
    p.add_argument("--tag", default="HEAD_EVAL", help="prefix of each result line")
    p.add_argument(
        "--solve-dtype",
        choices=["spec", "float32", "bfloat16"],
        default="spec",
        help="run the head's solve in this precision (spec: as the checkpoint's spec says)",
    )
    p.add_argument(
        "--outer-autocast",
        action="store_true",
        help="also wrap the whole forward in bf16 autocast (the trainer's eval did)",
    )
    a = p.parse_args()
    cuda = torch.cuda.is_available()
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            "nccl" if cuda else "gloo", timeout=datetime.timedelta(hours=3)
        )
        if cuda:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if cuda
        else torch.device("cpu")
    )
    tok = AutoTokenizer.from_pretrained(a.model)
    m = (
        GraniteSwitchForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    spec = m.config.loop_specs[0]
    if a.max_iter is not None:
        spec["max_iter"] = a.max_iter
    if a.min_iter is not None:
        spec["min_iter"] = a.min_iter
    if a.solve_dtype == "bfloat16":
        spec["solve_dtype"] = "bfloat16"
    elif a.solve_dtype == "float32":
        spec.pop("solve_dtype", None)
    pr = Prompt(tok, m.config.adapter_names[0], spec["end_offset"])
    test = np.load(Path(a.data) / "test.npz")
    ids = torch.tensor(spec["output_token_ids"], device=device)
    lo = a.first
    hi = min(len(test["x"]), lo + a.count)
    head = m.model.loop_heads.head(1)
    composed = copy.deepcopy(head.state_dict())
    for h in a.heads:
        head.load_state_dict(composed if h == "composed" else load_file(h))
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=a.outer_autocast
        ):
            res = evaluate(
                m, pr, test["x"][lo:hi], test["y"][lo:hi], a.batch, ids, device
            )
        res.update(
            head=h,
            test_slice=[lo, hi],
            max_iter=spec["max_iter"],
            min_iter=spec.get("min_iter", 0),
            solve_dtype=spec.get("solve_dtype", "float32"),
            outer_autocast=a.outer_autocast,
        )
        if is_main:
            print(f"{a.tag} " + json.dumps(res), flush=True)
            if a.json:
                Path(a.json).write_text(json.dumps(res, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
