# SPDX-License-Identifier: Apache-2.0
"""Sudoku-Extreme data for the loop head, built as in HRM / FPRM.

Train: 1,000 puzzles sampled from the train split, each kept once plus
``num_aug`` random symmetry transforms (digit relabeling, transpose, band/row
and stack/column shuffles). Test: the full test split, untransformed.

A puzzle is 81 cells in row-major order: 0 = blank, 1..9 = digit. The prompt
spells it as single tokens (the tokenizer merges digit runs, so ids are built
directly, never by encoding the string).
"""

import csv
import os

import numpy as np


def _read(path, limit=None):
    qs, ans = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for _source, q, a, _rating in r:
            assert len(q) == 81 and len(a) == 81
            qs.append(np.frombuffer(q.replace(".", "0").encode(), np.uint8) - ord("0"))
            ans.append(np.frombuffer(a.encode(), np.uint8) - ord("0"))
            if limit and len(qs) >= limit:
                break
    return np.stack(qs).astype(np.uint8), np.stack(ans).astype(np.uint8)


def shuffle_sudoku(board, solution, rng):
    """One random symmetry of a puzzle and its solution (HRM's augmentation)."""
    digit_map = np.pad(rng.permutation(np.arange(1, 10)), (1, 0))
    transpose = rng.random() < 0.5
    bands = rng.permutation(3)
    row_perm = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    stacks = rng.permutation(3)
    col_perm = np.concatenate([s * 3 + rng.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply(x):
        x = x.reshape(9, 9)
        if transpose:
            x = x.T
        return digit_map[x.flatten()[mapping]]

    return apply(board).astype(np.uint8), apply(solution).astype(np.uint8)


def build_test(raw_dir, out_dir):
    """Write only test.npz (the same shuffled order build() writes)."""
    os.makedirs(out_dir, exist_ok=True)
    tq, ta = _read(os.path.join(raw_dir, "test.csv"))
    perm = np.random.default_rng(1234).permutation(len(tq))
    np.savez_compressed(os.path.join(out_dir, "test.npz"), x=tq[perm], y=ta[perm])
    return len(tq)


def build(raw_dir, out_dir, subsample=1000, num_aug=1000, seed=42):
    """Write train.npz (augmented) and test.npz from the HF csv files."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    q, a = _read(os.path.join(raw_dir, "train.csv"))
    pick = rng.choice(len(q), size=subsample, replace=False)
    xs, ys = [], []
    for i in pick:
        xs.append(q[i])
        ys.append(a[i])
        for _ in range(num_aug):
            bq, ba = shuffle_sudoku(q[i], a[i], rng)
            xs.append(bq)
            ys.append(ba)
    np.savez_compressed(
        os.path.join(out_dir, "train.npz"),
        x=np.stack(xs),
        y=np.stack(ys),
        group_size=1 + num_aug,
    )
    tq, ta = _read(os.path.join(raw_dir, "test.csv"))
    # Shuffled with a fixed seed, so any prefix is a uniform sample of the split.
    perm = np.random.default_rng(1234).permutation(len(tq))
    np.savez_compressed(os.path.join(out_dir, "test.npz"), x=tq[perm], y=ta[perm])
    return len(xs), len(tq)


def check(board, solution):
    """True when ``solution`` (81 digits) is a valid grid that keeps the givens."""
    s = solution.reshape(9, 9)
    full = set(range(1, 10))
    for i in range(9):
        if set(s[i]) != full or set(s[:, i]) != full:
            return False
        r, c = 3 * (i // 3), 3 * (i % 3)
        if set(s[r : r + 3, c : c + 3].flatten()) != full:
            return False
    given = board > 0
    return bool((solution[given] == board[given]).all())
