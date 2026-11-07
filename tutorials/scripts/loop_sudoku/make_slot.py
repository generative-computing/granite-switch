# SPDX-License-Identifier: Apache-2.0
"""Write a Sudoku loop-slot folder and its compose manifest.

The folder holds loop_spec.json (the head's shape and solver settings) and an
io.yaml; after training it also holds loop_head.safetensors. The spec names its
output tokens ('1'..'9'), and compose checks them against the tokenizer.
"""

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer

# FPRM's Sudoku settings (config/cfg_pretrain_sudoku.yaml of the paper's code).
SPEC = {
    "width": 512,
    "num_layers": 2,
    "num_heads": 8,
    "expansion": 4,
    "grid_size": 81,
    "prefix_len": 16,
    "conv_kernel_size": 3,
    "alpha_1_init": 0.75,
    "alpha_2_init": 0.25,
    "rope_theta": 10000.0,
    "init_std": 6.0,  # cfg_pretrain_sudoku.yaml overrides FPRM's default 1.0
    "read_layer": None,
    # The served solve uses FPRM's evaluation settings (paper, Table 4): stop at
    # residual < 0.1, step-size decay 0.997 after 10 non-improving iterations,
    # and a budget of up to 35,000 iterations.
    "max_iter": 35000,
    "fp_thresh": 0.1,
    "stepsize_decay": 0.997,
    "decay_patience": 10,
    # The template closes the user turn before the marker, so the grid ends
    # this many tokens before it (<|end_of_text|> and a newline on 4.1).
    "end_offset": 2,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="ibm-granite/granite-4.1-3b")
    p.add_argument("--out", required=True, help="folder for the slot and the manifest")
    p.add_argument("--name", default="sudoku")
    p.add_argument("--read-layer", type=int, default=None)
    p.add_argument(
        "--head", default=None, help="trained loop_head.safetensors to include"
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    tokens = [str(d) for d in range(1, 10)]
    ids = [tok.convert_tokens_to_ids(t) for t in tokens]
    assert all(len(tok.encode(t, add_special_tokens=False)) == 1 for t in tokens)

    slot = Path(args.out) / args.name
    slot.mkdir(parents=True, exist_ok=True)
    spec = dict(
        SPEC, read_layer=args.read_layer, output_tokens=tokens, output_token_ids=ids
    )
    (slot / "loop_spec.json").write_text(json.dumps(spec, indent=2))
    (slot / "io.yaml").write_text(f"name: {args.name}\nmodel: ~\n")
    if args.head:
        shutil.copy(args.head, slot / "loop_head.safetensors")
    entry = f"{args.name}:\n  path: {slot}\n  type: lora\n  kind: loop\n"
    if args.read_layer is not None:
        entry += f"  layer: {args.read_layer}\n"
    manifest = Path(args.out) / "manifest.yaml"
    manifest.write_text(entry)
    print(f"wrote {slot} and {manifest}")


if __name__ == "__main__":
    main()
