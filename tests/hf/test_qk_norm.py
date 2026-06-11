# SPDX-License-Identifier: Apache-2.0
"""QK-norm unit tests for GraniteLoRAEmbeddedAttention (HF backend).

Verifies that optional QK-norm (Qwen3-style per-head RMSNorm on Q/K before
RoPE) is correctly wired: parameters exist when enabled, forward pass differs
from non-normed, and outputs are correct shape.

All tests run on CPU with random weights (no pretrained checkpoint needed).
"""

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf.core.lora import GraniteLoRAEmbeddedAttention

pytestmark = pytest.mark.local_fast


# ── Helpers ────────────────────────────────────────────────────────

def _make_config(qk_norm: bool, num_adapters: int = 0) -> GraniteSwitchConfig:
    """Minimal config for attention layer tests."""
    config = GraniteSwitchConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_multiplier=(64 // 4) ** -0.5,
        num_adapters=num_adapters,
        adapter_token_ids=[],
        adapter_names=[],
        adapter_ranks=[],
        qk_norm=qk_norm,
    )
    config._attn_implementation = "sdpa"
    return config


def _make_position_embeddings(seq_len: int, head_dim: int):
    """Create dummy (cos, sin) position embeddings.

    Shape [batch=1, seq_len, head_dim] — broadcastable to [batch, heads, seq, dim]
    after transpose in the attention forward pass.
    """
    cos = torch.ones(1, seq_len, head_dim)
    sin = torch.zeros(1, seq_len, head_dim)
    return cos, sin


# ── Tests ─────────────────────────────────────────────────────────

class TestQKNormParameters:
    """Verify QK-norm layers exist (or not) based on config."""

    def test_qk_norm_true_creates_norm_layers(self):
        config = _make_config(qk_norm=True)
        attn = GraniteLoRAEmbeddedAttention(config, layer_idx=0)
        assert hasattr(attn, "q_norm")
        assert hasattr(attn, "k_norm")

    def test_qk_norm_false_no_norm_layers(self):
        config = _make_config(qk_norm=False)
        attn = GraniteLoRAEmbeddedAttention(config, layer_idx=0)
        assert not hasattr(attn, "q_norm")
        assert not hasattr(attn, "k_norm")

    def test_qk_norm_default_false(self):
        """Config without qk_norm field should default to no norm."""
        config = _make_config(qk_norm=False)
        # Remove the attribute to simulate old configs
        del config.qk_norm
        attn = GraniteLoRAEmbeddedAttention(config, layer_idx=0)
        assert attn.qk_norm is False

    def test_q_norm_weight_shape(self):
        config = _make_config(qk_norm=True)
        attn = GraniteLoRAEmbeddedAttention(config, layer_idx=0)
        head_dim = config.hidden_size // config.num_attention_heads
        assert attn.q_norm.weight.shape == (head_dim,)
        assert attn.k_norm.weight.shape == (head_dim,)


class TestQKNormForward:
    """Verify QK-norm affects the forward pass output."""

    def test_output_differs_with_qk_norm(self):
        """Same weights, same input: qk_norm=True should produce different output."""
        torch.manual_seed(42)
        config_off = _make_config(qk_norm=False)
        attn_off = GraniteLoRAEmbeddedAttention(config_off, layer_idx=0)

        torch.manual_seed(42)
        config_on = _make_config(qk_norm=True)
        attn_on = GraniteLoRAEmbeddedAttention(config_on, layer_idx=0)

        # Copy base weights from off → on so only qk_norm differs
        attn_on.qkv_proj.load_state_dict(attn_off.qkv_proj.state_dict())
        attn_on.o_proj.load_state_dict(attn_off.o_proj.state_dict())

        bsz, seq_len = 1, 8
        head_dim = config_off.hidden_size // config_off.num_attention_heads
        hidden = torch.randn(bsz, seq_len, config_off.hidden_size)
        adapter_indices = torch.zeros(bsz, seq_len, dtype=torch.long)
        pos_emb = _make_position_embeddings(seq_len, head_dim)

        with torch.no_grad():
            out_off, _, _ = attn_off(
                hidden, adapter_indices,
                position_embeddings=pos_emb,
            )
            out_on, _, _ = attn_on(
                hidden, adapter_indices,
                position_embeddings=pos_emb,
            )

        assert not torch.allclose(out_off, out_on, atol=1e-6), (
            "QK-norm should change output, but outputs are identical"
        )

    def test_output_shape_preserved(self):
        """QK-norm should not change output shape."""
        config = _make_config(qk_norm=True)
        attn = GraniteLoRAEmbeddedAttention(config, layer_idx=0)

        bsz, seq_len = 2, 16
        head_dim = config.hidden_size // config.num_attention_heads
        hidden = torch.randn(bsz, seq_len, config.hidden_size)
        adapter_indices = torch.zeros(bsz, seq_len, dtype=torch.long)
        pos_emb = _make_position_embeddings(seq_len, head_dim)

        with torch.no_grad():
            out, _, _ = attn(
                hidden, adapter_indices,
                position_embeddings=pos_emb,
            )

        assert out.shape == (bsz, seq_len, config.hidden_size)
