#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copied from the switch benchmark (granite-switch-staging, branch
# feature/switch-benchmark, commit 0359602) and kept unchanged, so this benchmark
# measures switching exactly as it does. Update it only by copying it again.
"""Reject a composed Granite Switch checkpoint whose adapters did not land.

The composer skips any adapter tensor it cannot remap onto the switch model with
a bare ``continue``, and ``validate_cross_stream_population`` only warns. The
resulting checkpoint loads, serves, and is *faster* than it should be, because
the missing modules simply run as base. That is the worst possible failure for a
throughput benchmark, so it gets a hard gate rather than a log line.

This is not hypothetical. The pre-existing checkpoint
``agentic-4adapter-toolobject-alora-qkvo-mlp-r32-4.1-3b`` has entirely zero
``shared_mlp.input_linear.lora_B`` and ``shared_mlp.output_linear.lora_B``
despite the ``-mlp-`` in its name: its source adapters stored MLP LoRA without
the ``mlp.`` parent segment, so every MLP tensor was dropped at compose time.

Checks, all fatal:

* every expected LoRA group is present in the weight index, and its ``lora_A``
  and ``lora_B`` are non-zero **for each adapter slot** at sampled layers;
* ``num_adapters`` / ``adapter_ranks`` / ``max_lora_rank`` / ``vocab_size`` match
  what was asked for;
* Shadow-Residual checkpoints carry ``dual_stream`` and ``cross_stream_rank``,
  with non-zero ``cross_stream`` weights;
* the Granite scaling constants survived composition — silently losing
  ``logits_scaling`` would change outputs without changing speed.

Reads tensors by byte offset straight out of the safetensors shards, so it needs
neither torch nor a GPU and costs a few hundred KB of I/O per check.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# LoRA groups a composed dense-Granite switch checkpoint must contain. The MLP
# entries are the ones the parentless-key bug silently empties.
EXPECTED_GROUPS = (
    "self_attn.qkv_proj.lora_A_slices.0",
    "self_attn.qkv_proj.lora_B_slices.0",
    "self_attn.o_proj.lora_A",
    "self_attn.o_proj.lora_B",
    "shared_mlp.input_linear.lora_A_slices.0",
    "shared_mlp.input_linear.lora_B_slices.0",
    "shared_mlp.input_linear.lora_A_slices.1",
    "shared_mlp.input_linear.lora_B_slices.1",
    "shared_mlp.output_linear.lora_A",
    "shared_mlp.output_linear.lora_B",
)
SR_GROUPS = ("cross_stream.lora_A", "cross_stream.lora_B")
# Present only when the adapters target k/v — Shadow Residual never does.
KV_GROUPS = (
    "self_attn.qkv_proj.lora_A_slices.1",
    "self_attn.qkv_proj.lora_B_slices.1",
    "self_attn.qkv_proj.lora_A_slices.2",
    "self_attn.qkv_proj.lora_B_slices.2",
)
GRANITE_SCALARS = (
    "logits_scaling",
    "attention_multiplier",
    "residual_multiplier",
    "embedding_multiplier",
)

_BF16 = "BF16"


def read_tensor(ckpt: Path, weight_map, name):
    """Read one tensor by byte offset from its shard. Returns a float32 ndarray."""
    shard = ckpt / weight_map[name]
    with shard.open("rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        entry = header[name]
        start, end = entry["data_offsets"]
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    dtype = entry["dtype"]
    if dtype == _BF16:
        arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(
            np.float32
        )
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        raise SystemExit(f"FATAL: unhandled dtype {dtype} for {name}")
    return arr.reshape(entry["shape"]), entry["shape"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument("--expect-adapters", type=int, required=True)
    ap.add_argument("--expect-rank", type=int, default=32)
    ap.add_argument(
        "--expect-alpha",
        type=float,
        default=64.0,
        help="Adapter lora_alpha, used to check the cross-stream scale.",
    )
    ap.add_argument("--base-vocab", type=int, default=100352)
    ap.add_argument(
        "--expect-switch-type",
        default=None,
        choices=["single", "multi"],
        help="Assert the composed switch engine. Unset skips the check -- but a "
        "checkpoint composed for a transition benchmark that silently landed on "
        "'single' would route by averaging and publish plausible numbers for the "
        "wrong spans, so pass it whenever you care.",
    )
    ap.add_argument(
        "--base-reset",
        action="store_true",
        help="The checkpoint was composed with --base-reset-token, so "
        "adapter_token_ids carries a leading base slot: the list and the "
        "substitute list are num_adapters + 1 long and vocab_size is "
        "base + num_adapters + 1. Without this flag those three checks are "
        "off by one and the compose looks broken when it is correct.",
    )
    ap.add_argument(
        "--layers",
        default="1,5,20,39",
        help="Layer indices to sample. Layer 0 is the prepended switch layer "
        "and carries no adapter LoRA.",
    )
    ap.add_argument(
        "--expect-kv",
        action="store_true",
        help="Adapters target k_proj/v_proj (standard aLoRA, not SR).",
    )
    ap.add_argument(
        "--expect-names",
        default=None,
        help="Comma-separated adapter names in canonical task order; the first "
        "--expect-adapters of them must be exactly the set embedded here. Checked as a "
        "set, not a sequence: the composer allocates control tokens in discovery order, "
        "which need not match the CLI order, and since every adapter here has identical "
        "rank and module coverage the slot ordering cannot affect throughput. The wrong "
        "SUBSET would silently change what the N axis means, so that is fatal.",
    )
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / "config.json").read_text())
    weight_map = json.loads((ckpt / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    fails, notes = [], []

    def fail(msg):
        fails.append(msg)
        print(f"  FAIL  {msg}")

    def ok(msg):
        print(f"  ok    {msg}")

    print(f"verifying {ckpt}")

    # ---- config -----------------------------------------------------------
    n = args.expect_adapters
    is_sr = bool(cfg.get("dual_stream"))
    # --base-reset-token prepends a base slot to adapter_token_ids, so the control
    # lists and the vocabulary each grow by one beyond num_adapters. num_adapters
    # itself does NOT change: the base slot is not an adapter.
    n_ctl = n + 1 if args.base_reset else n
    checks = {
        "num_adapters": (cfg.get("num_adapters"), n),
        "max_lora_rank": (cfg.get("max_lora_rank"), args.expect_rank),
        "adapter_ranks": (cfg.get("adapter_ranks"), [args.expect_rank] * n),
        "vocab_size": (cfg.get("vocab_size"), args.base_vocab + n_ctl),
        "len(adapter_token_ids)": (len(cfg.get("adapter_token_ids") or []), n_ctl),
        "len(adapter_substitute_token_ids)": (
            len(cfg.get("adapter_substitute_token_ids") or []),
            n_ctl,
        ),
        "len(adapter_names)": (len(cfg.get("adapter_names") or []), n),
    }
    for key, (got, want) in checks.items():
        (ok if got == want else fail)(f"{key} = {got!r} (expected {want!r})")

    ids = cfg.get("adapter_token_ids") or []
    (ok if len(set(ids)) == len(ids) else fail)(f"adapter_token_ids unique: {ids}")

    if args.expect_switch_type is not None:
        got_st = cfg.get("switch_type", "single")
        (ok if got_st == args.expect_switch_type else fail)(
            f"switch_type = {got_st!r} (expected {args.expect_switch_type!r})"
        )
    if args.base_reset:
        # MultiSwitch keys the base slot off the list being num_adapters + 1 long
        # (validator.validate_base_reset_switch_type). If the flag was dropped the
        # list is n long, every span still routes to some adapter, and a
        # return-to-base benchmark measures nothing while looking fine.
        st = cfg.get("switch_type", "single")
        (ok if st in ("multi",) else fail)(
            f"base-reset requires a multi engine; switch_type = {st!r}"
        )

    got_names = list(cfg.get("adapter_names") or [])
    if args.expect_names:
        want = [x.strip() for x in args.expect_names.split(",") if x.strip()][:n]
        if set(got_names) == set(want):
            ok(f"adapter_names subset correct: {got_names}")
            if got_names != want:
                notes.append(
                    f"adapter slot order is {got_names}, not the CLI order {want}; "
                    f"harmless for throughput but the report should quote the "
                    f"composed order"
                )
        else:
            fail(f"adapter_names {got_names} != expected subset {want}")

    for key in GRANITE_SCALARS:
        val = cfg.get(key)
        (ok if val is not None else fail)(f"{key} = {val}")

    if cfg.get("unfused_qkv"):
        fail(
            "unfused_qkv is set — this is a pre-fusion checkpoint and must be re-composed"
        )

    if is_sr:
        csr = cfg.get("cross_stream_rank")
        (ok if csr == args.expect_rank else fail)(
            f"cross_stream_rank = {csr!r} (expected {args.expect_rank})"
        )
        notes.append(
            f"cross-stream scale is alpha/rank; source alpha={args.expect_alpha} "
            f"rank={args.expect_rank} -> expected {args.expect_alpha / args.expect_rank:g}"
        )
    else:
        (ok if cfg.get("cross_stream_rank") is None else fail)(
            "cross_stream_rank is None for a non-SR checkpoint"
        )

    # ---- weights ----------------------------------------------------------
    groups = list(EXPECTED_GROUPS)
    if is_sr:
        groups += list(SR_GROUPS)
    if args.expect_kv:
        groups += list(KV_GROUPS)
    if is_sr and args.expect_kv:
        fail(
            "--expect-kv with a Shadow-Residual checkpoint: SR takes K/V from the base "
            "stream and must never carry k_proj/v_proj LoRA"
        )

    layers = [int(x) for x in args.layers.split(",")]
    print(f"  sampling layers {layers}; {len(groups)} LoRA groups; SR={is_sr}")
    for layer in layers:
        for group in groups:
            name = f"model.layers.{layer}.{group}"
            if name not in weight_map:
                fail(f"missing weight {name}")
                continue
            arr, shape = read_tensor(ckpt, weight_map, name)
            # Adapter slot is the leading axis; check every slot independently so a
            # single dropped adapter cannot hide behind its populated neighbours.
            flat = (
                arr.reshape(shape[0], -1)
                if shape and shape[0] == n
                else arr.reshape(1, -1)
            )
            dead = [i for i in range(flat.shape[0]) if not np.any(flat[i])]
            if dead:
                fail(
                    f"{name} is ALL ZERO for adapter slot(s) {dead} "
                    f"(shape {shape}) — the composer dropped these weights"
                )
            else:
                ok(
                    f"{name} nonzero in all {flat.shape[0]} slot(s), "
                    f"absmax={float(np.abs(arr).max()):.4g}"
                )

    print()
    for note in notes:
        print(f"  note: {note}")
    if fails:
        print(f"\nVERIFY_FAIL {ckpt}: {len(fails)} problem(s)")
        for f_ in fails:
            print(f"  - {f_}")
        return 1
    print(f"VERIFY_OK {ckpt} N={n} SR={is_sr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
