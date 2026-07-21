# SPDX-License-Identifier: Apache-2.0
"""Shared configs and helpers for Granite 4 upstream equivalence tests.

Used by both HF (tests/hf/) and vLLM (tests/vllm/) equivalence tests.

All miniaturized configs use head_dim=32, which is compatible with both
HF (any head_dim) and vLLM FlashAttention (requires 32/64/96/128/256).

Real Granite 4 dense models (as of 2025-02):

  Model         Params  Arch    GQA   PosEmb  Layers
  ─────────────────────────────────────────────────────
  4.0-350m      350M    dense   4:1   rope    28 attn
  4.0-1b        1B      dense   4:1   rope    40 attn
  4.0-micro     3B      dense   5:1   rope    40 attn

Dense models use exclusively attention layers.
"""

import torch

# ── Assertion helper ──────────────────────────────────────────────


def assert_close(actual, expected, *, atol, rtol, msg=""):
    """Assert tensors are close using independent absolute and relative checks.

    Each element passes if it satisfies EITHER criterion:
        |a - b| <= atol       (absolute: good for values near zero)
        |a - b| <= rtol * |b| (relative: good for large values)

    This avoids the joint formula (atol + rtol * |b|) where the two
    tolerances inflate each other's budget.
    """
    diff = (actual - expected).abs()
    abs_ok = diff <= atol
    rel_ok = diff <= rtol * expected.abs()
    ok = abs_ok | rel_ok

    if not ok.all():
        num_bad = (~ok).sum().item()
        worst_idx = diff[~ok].argmax()
        worst_diff = diff[~ok][worst_idx].item()
        worst_expected = expected[~ok][worst_idx].abs().item()
        raise AssertionError(
            f"{msg}: {num_bad} elements exceed tolerance "
            f"(worst: diff={worst_diff:.4e}, |expected|={worst_expected:.4e}, "
            f"atol={atol:.1e}, rtol={rtol:.1e})"
        )


# ── Weight transfer ──────────────────────────────────────────────


def _resolve_base_layer(name):
    """Strip '.base_layer.' from LoRA-wrapped parameter names.

    When num_adapters > 0, LoRA wrappers add a '.base_layer.' indirection:
      switch: model.layers.0.self_attn.o_proj.base_layer.weight
      upstream: model.layers.0.self_attn.o_proj.weight

    Returns the resolved name (or the original if no '.base_layer.' found).
    """
    return name.replace(".base_layer.", ".")


def transfer_weights(upstream_sd, switch_sd):
    """Transfer weights from upstream state dict to switch state dict.

    Both arguments are {name: tensor} dicts (e.g. from model.state_dict()).
    Modifies switch_sd tensors in-place.

    Handles:
    1. Fused QKV: upstream q_proj/k_proj/v_proj → switch qkv_proj
       (including LoRA-wrapped: qkv_proj.base_layer.weight)
    2. LoRA base_layer indirection: .base_layer. stripped for upstream lookup

    Returns list of switch parameter names that could NOT be loaded.
    """
    loaded = set()
    for name, param in switch_sd.items():
        # Resolve .base_layer. for LoRA-wrapped params
        resolved = _resolve_base_layer(name)

        if ".qkv_proj." in name:
            suffix = resolved.split(".qkv_proj.")[-1]
            prefix = resolved.split(".qkv_proj.")[0]
            q_name = f"{prefix}.q_proj.{suffix}"
            k_name = f"{prefix}.k_proj.{suffix}"
            v_name = f"{prefix}.v_proj.{suffix}"
            if q_name in upstream_sd:
                fused = torch.cat(
                    [upstream_sd[q_name], upstream_sd[k_name], upstream_sd[v_name]],
                    dim=0,
                )
                param.data.copy_(fused)
                loaded.add(name)
        elif resolved in upstream_sd:
            param.data.copy_(upstream_sd[resolved])
            loaded.add(name)

    unloaded = [n for n in switch_sd if n not in loaded]
    return unloaded


def transfer_weights_strict(upstream_sd, switch_sd):
    """Like transfer_weights but raises on any unloaded parameters."""
    unloaded = transfer_weights(upstream_sd, switch_sd)
    if unloaded:
        raise RuntimeError(
            f"{len(unloaded)} switch parameters not loaded from upstream: "
            f"{unloaded[:10]}"
        )


# ── Zero-adapter test helpers ─────────────────────────────────────
#
# Strategy: use regular ("vanilla") tokens as adapter control tokens.
# The upstream model processes these tokens normally. The switch model
# recognizes them as control tokens and activates adapters, but with
# zero LoRA weights the delta is zero for the LoRA path.
#
# All control tokens (adapter_token_ids) are KV-hidden by default.
# Test sequences use adapter tokens; the control position is excluded
# from comparison via get_visible_mask().
#
# Token-exchange: at the switch, each control token's id is rewritten to
# its configured substitute id before the decoder embeds. Upstream sees
# the original control id; switch sees the substitute. The two embeddings
# differ at the control position only.
#
# Error sources:
#
# 1. Embedding divergence at control positions (primary):
#    Upstream embeds the original control id; switch embeds the substitute.
#    The control position itself is excluded from comparison. Visible
#    positions attend to the control position too — the attention
#    contribution from the (substitute vs original) embedding propagates.
#    Error scales with control_tokens / seq_len.
#
# 2. Mamba conv1d effects (hybrid only):
#    Conv1d's sliding window over substituted vs original tokens at the
#    control position perturbs K-1 subsequent real tokens.
#
# Token allocation (within vocab_size=256):
#   101+ = adapter_token_ids (rewritten to substitute by switch)
#   Random fill: [0, 100) -- guaranteed no collisions with control tokens

# Control token IDs -- low-vocab, valid embeddings in the base model
_ADAPTER_TOKEN_BASE = 101  # 101, 102, ... for adapter tokens


def augment_cfg_with_adapters(cfg_dict, num_adapters=2, rank=8):
    """Augment a base GRANITE4_MINI config dict with adapter infrastructure.

    Uses vanilla (low-vocab) tokens as adapter control tokens so the switch
    actually computes non-zero adapter_indices when these tokens appear in
    the input. With zero LoRA weights, the adapter delta is zero and the
    output matches upstream.

    Returns a new dict suitable for GraniteSwitchConfig(**result) that has:
    - num_hidden_layers += 1 (1 cache slot for SingleSwitch)
    - layer_types prepended with "attention" (switch layer type)
    - LoRA adapter config fields
    - adapter_token_ids (rewritten to substitute ids by the switch)
    - adapter_substitute_token_ids (token-exchange substitutes)
    - adapter_names for name-to-index mapping
    """
    cfg = dict(cfg_dict)

    # Prepend placeholder entry for switch cache slot (SingleSwitch: 1 slot)
    cfg["num_hidden_layers"] = cfg["num_hidden_layers"] + 1
    cfg["layer_types"] = ["attention", *list(cfg["layer_types"])]

    # Adapter configuration
    cfg["num_adapters"] = num_adapters
    cfg["adapter_ranks"] = [rank] * num_adapters
    cfg["max_lora_rank"] = rank
    adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    cfg["adapter_names"] = adapter_names

    # SingleSwitch: num_adapters entries
    cfg["adapter_token_ids"] = [_ADAPTER_TOKEN_BASE + i for i in range(num_adapters)]
    # Token-exchange substitute ids — use a benign shared id (the BOS-or-
    # equivalent doesn't matter for these synthetic equivalence tests since
    # all LoRA weights are zero, so the embedding is what feeds the decoder).
    cfg["adapter_substitute_token_ids"] = [1] * num_adapters

    return cfg


def make_active_adapter_input(batch_size, seq_len, seed=42):
    """Generate input_ids that trigger adapter switching.

    Places control tokens at specific positions so the switch computes
    non-zero adapter_indices. Fill tokens are drawn from [0, 100) to
    avoid collisions with control tokens (101+).

    SingleSwitch: single adapter activation (one-shot, no mid-sequence
    transitions). Place one control token early in the sequence.

    Returns:
        input_ids: [batch_size, seq_len] LongTensor
    """
    torch.manual_seed(seed)
    # Fill with tokens from [0, 100) — no control token collisions
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # SingleSwitch: single adapter activation (one-shot, no mid-sequence
    # transitions). Place one control token early in the sequence.
    input_ids[:, 2] = _ADAPTER_TOKEN_BASE

    return input_ids


def zero_lora_weights(model):
    """Zero all LoRA A and B parameters in-place.

    Defensive — makes the zero-adapter test independent of early-exit behavior.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.zero_()


def get_visible_mask(input_ids):
    """Return boolean mask of non-control (visible) positions.

    Control positions hold the substitute embedding in the switch model
    versus the original control-token embedding upstream — their logits
    are intentionally different. This mask identifies positions that
    should be compared in equivalence tests.

    All adapter tokens (>= _ADAPTER_TOKEN_BASE) are control positions.
    Fill tokens from [0, 100) are visible.
    """
    is_adapter = input_ids >= _ADAPTER_TOKEN_BASE
    return ~is_adapter


# ── Tolerance selection ──────────────────────────────────────────


def get_tolerances(layer_types, long_sequence=False, has_kv_hidden=False):
    """Return (atol, rtol) for a given architecture.

    Error sources:

    1. **No adapters**: GraniteSwitch with num_adapters=0 is bit-exact vs
       upstream Granite. Fused QKV matmul is bit-identical to separate
       Q/K/V matmuls in float32.

    2. **Token-exchange embedding divergence**: With adapters and a control
       token in the input, the switch embeds the substitute id at that
       position while upstream embeds the original control id. Visible
       positions attending to the control position pick up that delta.

    Args:
        layer_types: list of "attention" strings
        long_sequence: unused (kept for API compatibility)
        has_kv_hidden: True when adapters are active and control tokens
            are present (kept name for API compatibility — the parameter
            now means "control tokens get substituted").

    Returns:
        (atol, rtol) tuple, or None if bit-exact match expected.
    """
    if not has_kv_hidden:
        # Pure base-model path: bit-exact (fused QKV numerically identical).
        return None
    else:
        # Substitute-embedding propagates through attention to visible
        # positions. Worst observed: ~5.0e-2 (multi 1b, seed-dependent).
        return (6e-2, 6e-2)


# ── Miniaturized Granite 4 configs ───────────────────────────────
#
# Each mini config matches the REAL model's head_dim (64 or 128) and GQA
# ratio.  hidden_size = num_attention_heads * real_head_dim, with minimal
# head counts.  This ensures page-size relationships between attention
# and switch layers faithfully reproduce deployment behavior.

_FIXED = dict(
    vocab_size=256,
    hidden_act="silu",
    max_position_embeddings=512,
    attention_bias=False,
    mlp_bias=False,
)

GRANITE4_MINI = {
    # ── Dense (all attention, rope) ──────────────────────────────
    "4.0-350m": {
        # Real: hidden=1024, heads=16, kv=4, GQA 4:1, head_dim=64, rope
        **_FIXED,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,  # GQA 4:1, head_dim=64
        "intermediate_size": 512,
        "shared_intermediate_size": 512,
        "num_local_experts": 0,
        "num_experts_per_tok": 0,
        "layer_types": ["attention"] * 4,
        "position_embedding_type": "rope",
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.263,
        "attention_multiplier": 0.015625,
        "logits_scaling": 4.0,
    },
    "4.0-1b": {
        # Real: hidden=2048, heads=16, kv=4, GQA 4:1, head_dim=128, rope
        **_FIXED,
        "hidden_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,  # GQA 4:1, head_dim=128
        "intermediate_size": 1024,
        "shared_intermediate_size": 1024,
        "num_local_experts": 0,
        "num_experts_per_tok": 0,
        "layer_types": ["attention"] * 4,
        "position_embedding_type": "rope",
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.0078125,
        "logits_scaling": 8.0,
    },
    "4.0-micro": {
        # Real: hidden=2560, heads=40, kv=8, GQA 5:1, head_dim=64, rope
        **_FIXED,
        "hidden_size": 320,
        "num_hidden_layers": 4,
        "num_attention_heads": 5,
        "num_key_value_heads": 1,  # GQA 5:1, head_dim=64
        "intermediate_size": 640,
        "shared_intermediate_size": 640,
        "num_local_experts": 0,
        "num_experts_per_tok": 0,
        "layer_types": ["attention"] * 4,
        "position_embedding_type": "rope",
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.015625,
        "logits_scaling": 10.0,
    },
}


# ── Full-size configs (real HF dimensions) ───────────────────────

_FULLSIZE_FIXED = dict(
    vocab_size=100352,
    hidden_act="silu",
    max_position_embeddings=131072,
    attention_bias=False,
    mlp_bias=False,
)

GRANITE4_FULLSIZE = {
    "4.0-350m": {
        **_FULLSIZE_FIXED,
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 2048,
        "shared_intermediate_size": 2048,
        "num_local_experts": 0,
        "num_experts_per_tok": 0,
        "layer_types": ["attention"] * 28,
        "position_embedding_type": "rope",
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.263,
        "attention_multiplier": 0.015625,
        "logits_scaling": 4.0,
    },
    "4.0-1b": {
        **_FULLSIZE_FIXED,
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "intermediate_size": 4096,
        "shared_intermediate_size": 4096,
        "num_local_experts": 0,
        "num_experts_per_tok": 0,
        "layer_types": ["attention"] * 40,
        "position_embedding_type": "rope",
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.0078125,
        "logits_scaling": 8.0,
    },
    "4.0-micro": {
        **_FULLSIZE_FIXED,
        "hidden_size": 2560,
        "num_hidden_layers": 40,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "shared_intermediate_size": 8192,
        "num_local_experts": 0,
        "num_experts_per_tok": 0,
        "layer_types": ["attention"] * 40,
        "position_embedding_type": "rope",
        "embedding_multiplier": 12.0,
        "residual_multiplier": 0.22,
        "attention_multiplier": 0.015625,
        "logits_scaling": 10.0,
    },
}
