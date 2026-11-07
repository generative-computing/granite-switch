# SPDX-License-Identifier: Apache-2.0
"""Sudoku-Extreme as chat rows, for training generation adapters (LoRA, aLoRA, SR).

The loop slot reads the grid and writes one digit per cell in a single forward
pass. A generation adapter must instead answer in text, so each puzzle becomes
a two-turn conversation: the grid in the user turn, the solved grid in the
assistant turn. Both spell the grid as nine rows of nine digits separated by
spaces, ``0`` for a blank, so every cell is one token and every puzzle has the
same length (``format_grid``). The answer is parsed back with ``parse_grid``.

    python adapter_data.py --data <dir with train.npz/test.npz> --out <dir> [--train-rows N] [--val-rows N]

Train rows come from the same augmented train split the loop head trains on,
one random augmentation of a base puzzle per row, as the loop trainer draws
them; validation rows come from the test split's selection slice [0, val);
the final test slice [2000, 22000) is never written.
"""

import argparse
import json
from pathlib import Path

import numpy as np

INSTRUCTION = (
    "Solve this sudoku. The grid is given row by row, one row per line, "
    "cells separated by spaces, 0 for a blank cell. Answer with the solved "
    "grid in the same format."
)


def format_grid(cells) -> str:
    """81 digits (0 = blank) -> nine lines of nine space-separated digits."""
    cells = [int(c) for c in cells]
    return "\n".join(
        " ".join(str(c) for c in cells[r * 9 : (r + 1) * 9]) for r in range(9)
    )


def parse_grid(text: str):
    """The first 81 digits of an answer, or ``None`` when it has fewer."""
    digits = [int(ch) for ch in text if ch.isdigit()]
    return digits[:81] if len(digits) >= 81 else None


def row(board, solution) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"{INSTRUCTION}\n\n{format_grid(board)}"},
            {"role": "assistant", "content": format_grid(solution)},
        ]
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        required=True,
        help="folder with train.npz and test.npz (sudoku_data.py)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--train-rows", type=int, default=100_000)
    p.add_argument("--val-rows", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    # Read each array once: indexing an NpzFile re-reads (and decompresses) it.
    train = dict(np.load(Path(a.data) / "train.npz"))
    test = dict(np.load(Path(a.data) / "test.npz"))
    group = int(train["group_size"]) if "group_size" in train else 1
    groups = len(train["x"]) // group
    rng = np.random.default_rng(a.seed)
    # One pass over the base puzzles per epoch, a random augmentation of each,
    # as the loop trainer samples them.
    order = np.concatenate(
        [rng.permutation(groups) for _ in range(-(-a.train_rows // groups))]
    )
    picks = order[: a.train_rows] * group + rng.integers(0, group, a.train_rows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as f:
        for i in picks:
            f.write(json.dumps(row(train["x"][i], train["y"][i])) + "\n")
    with open(out / "val.jsonl", "w") as f:
        for i in range(a.val_rows):
            f.write(json.dumps(row(test["x"][i], test["y"][i])) + "\n")
    print(f"wrote {a.train_rows} train and {a.val_rows} val rows to {out}")


if __name__ == "__main__":
    main()
