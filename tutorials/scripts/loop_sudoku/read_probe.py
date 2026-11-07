# SPDX-License-Identifier: Apache-2.0
"""Which base layer carries each cell's given digit best? (Linear probe, GPU.)

Reads the hidden state at the 81 grid cells from several decoder layers of the
composed checkpoint, then fits a linear map per layer (ridge, closed form) from
a cell's state to that cell's GIVEN value (blank or 1..9) and to its SOLUTION
digit. Prints held-out accuracy per layer.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_sudoku import Prompt
from transformers import AutoTokenizer

from granite_switch.hf import GraniteSwitchForCausalLM


def ridge_acc(xtr, ytr, xte, yte, classes, lam=1e-2):
    xtr = torch.cat([xtr, torch.ones(len(xtr), 1, device=xtr.device)], 1)
    xte = torch.cat([xte, torch.ones(len(xte), 1, device=xte.device)], 1)
    y = torch.nn.functional.one_hot(ytr, classes).float()
    a = xtr.T @ xtr + lam * len(xtr) * torch.eye(xtr.shape[1], device=xtr.device)
    w = torch.linalg.solve(a, xtr.T @ y)
    return float(((xte @ w).argmax(-1) == yte).float().mean())


@torch.no_grad()
def main():
    ck, data, out = sys.argv[1], sys.argv[2], sys.argv[3]
    tok = AutoTokenizer.from_pretrained(ck)
    m = GraniteSwitchForCausalLM.from_pretrained(ck, dtype=torch.bfloat16).cuda().eval()
    spec = m.config.loop_specs[0]
    pr = Prompt(tok, "sudoku", spec["end_offset"])
    d = np.load(Path(data) / "train.npz")
    x, y = d["x"][::997][:1200], d["y"][::997][:1200]  # distinct base puzzles
    n_layers = len(m.model.layers)
    layers = sorted({0, 4, 8, 12, 16, 20, 24, 28, 32, n_layers - 1})
    feats = {L: [] for L in [*layers, "final"]}
    for i in range(0, len(x), 64):
        ids = pr.batch(torch.as_tensor(x[i : i + 64], device="cuda"))
        o = m.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        g = slice(pr.grid_start, pr.grid_start + 81)
        for L in layers:
            feats[L].append(o.hidden_states[L + 1][:, g].float())
        feats["final"].append(o.last_hidden_state[:, g].float())
    given = torch.as_tensor(x.astype(np.int64), device="cuda").reshape(-1)
    sol = torch.as_tensor(y.astype(np.int64) - 1, device="cuda").reshape(-1)
    n_tr = int(0.8 * len(given))
    res = {}
    for L, f in feats.items():
        f = torch.cat(f).reshape(-1, f[0].shape[-1])
        f = (f - f[:n_tr].mean(0)) / (f[:n_tr].std(0) + 1e-5)
        res[str(L)] = {
            "given_acc": ridge_acc(f[:n_tr], given[:n_tr], f[n_tr:], given[n_tr:], 10),
            "solution_acc": ridge_acc(f[:n_tr], sol[:n_tr], f[n_tr:], sol[n_tr:], 9),
        }
        print("LAYER", L, json.dumps(res[str(L)]), flush=True)
    Path(out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
