# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for KV hiding gap equivalence tests.

Used by both HF (tests/hf/) and vLLM (tests/vllm/) gap equivalence tests.
Backend-agnostic input generation and visible-position extraction.
"""

import torch

from tests.shared.granite4_equivalence import _ADAPTER_TOKEN_BASE

# ── Constants ─────────────────────────────────────────────────────

# Switch types available for testing (MVP: SingleSwitch only)
SWITCH_TYPES = ["single"]

# Attention-only configs (RoPE, no Mamba) — KV hiding only works with attention.
ATTN_ONLY_NAMES = ["4.0-350m", "4.0-1b", "4.0-micro"]


# ── Helpers ───────────────────────────────────────────────────────


def ctrl_token():
    """Return the control token ID for SingleSwitch.

    SingleSwitch (num_adapters=1): adapter_token_ids=[101] → 101 is adapter_0.
    """
    return _ADAPTER_TOKEN_BASE  # 101


def make_gapped_inputs(seq_len, ctrl_pos, seed=42):
    """Create upstream and switch input_ids with a hidden control token gap.

    Upstream: [1, seq_len] random tokens from [0, 100).
    Switch:   same tokens with the adapter_0 control token inserted at ctrl_pos.
              Shape: [1, seq_len + 1].

    Returns:
        (upstream_ids, switch_ids) — both [1, seq_len(+1)] LongTensors.
    """
    torch.manual_seed(seed)
    upstream_ids = torch.randint(0, 100, (1, seq_len))

    ctrl = ctrl_token()
    switch_ids = torch.cat(
        [
            upstream_ids[:, :ctrl_pos],
            torch.tensor([[ctrl]]),
            upstream_ids[:, ctrl_pos:],
        ],
        dim=1,
    )

    return upstream_ids, switch_ids


def extract_visible_batched(tensor, ctrl_pos):
    """Remove the ctrl_pos index on dim=1 — for HF [batch, seq, ...] tensors."""
    return torch.cat(
        [
            tensor[:, :ctrl_pos],
            tensor[:, ctrl_pos + 1 :],
        ],
        dim=1,
    )


def extract_visible_flat(tensor, ctrl_pos):
    """Remove the ctrl_pos index on dim=0 — for vLLM [seq, ...] tensors."""
    return torch.cat(
        [
            tensor[:ctrl_pos],
            tensor[ctrl_pos + 1 :],
        ],
        dim=0,
    )
