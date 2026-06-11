# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward pass tests for GraniteSwitchForCausalLM (HF backend).

Tests the full model wiring: switch → adapter_indices → decoder layers.
All tests run on CPU with random weights (no pretrained checkpoint needed).
"""

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM
from granite_switch.hf.switch.single import SingleSwitch

pytestmark = pytest.mark.local_fast


# ── Helpers ────────────────────────────────────────────────────────

def _set_adapter_token_ids(model, token_ids):
    """Populate model.model.adapter_token_ids from a list of ints."""
    model.model.adapter_token_ids.data = torch.tensor(token_ids, dtype=torch.long)


def _set_nonzero_lora_B(model, scale=0.1):
    """Set non-zero lora_B on every LoRA layer so adapters produce visible deltas."""
    with torch.no_grad():
        for layer in model.model.layers:
            attn = layer.self_attn
            mlp = layer.shared_mlp
            # QKV (MergedSwitchedLoRALinear)
            if hasattr(attn.qkv_proj, "lora_B_slices"):
                for b in attn.qkv_proj.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            # O proj (SwitchedLoRALinear)
            if hasattr(attn.o_proj, "lora_B"):
                attn.o_proj.lora_B.data = torch.randn_like(attn.o_proj.lora_B) * scale
            # input_linear (MergedSwitchedLoRALinear)
            if hasattr(mlp.input_linear, "lora_B_slices"):
                for b in mlp.input_linear.lora_B_slices:
                    b.data = torch.randn_like(b) * scale
            # output_linear (SwitchedLoRALinear)
            if hasattr(mlp.output_linear, "lora_B"):
                mlp.output_linear.lora_B.data = torch.randn_like(mlp.output_linear.lora_B) * scale


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tiny_single_config():
    """Minimal SingleSwitch config for CPU tests (token exchange)."""
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 1 switch + 2 decoder
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["adapter_1", "adapter_2"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=16,
    )


# ════════════════════════════════════════════════════════════════════
# 1. Model instantiation
# ════════════════════════════════════════════════════════════════════

class TestModelInstantiation:

    def test_single_switch_model_creates(self, tiny_config):
        model = GraniteSwitchForCausalLM(tiny_config)
        assert isinstance(model.model.switch, SingleSwitch)
        assert len(model.model.layers) == 2  # num_hidden_layers - 1 switch

    def test_no_adapter_model_creates(self, tiny_config_no_adapters):
        model = GraniteSwitchForCausalLM(tiny_config_no_adapters)
        assert model.model.switch is None
        assert len(model.model.layers) == 2


# ════════════════════════════════════════════════════════════════════
# 2. Forward output shape
# ════════════════════════════════════════════════════════════════════

class TestForwardOutputShape:

    def test_basic_output_shape(self, tiny_config):
        model = GraniteSwitchForCausalLM(tiny_config).eval()
        _set_adapter_token_ids(model, tiny_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 8, tiny_config.vocab_size)

    def test_batch_output_shape(self, tiny_config):
        model = GraniteSwitchForCausalLM(tiny_config).eval()
        _set_adapter_token_ids(model, tiny_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (2, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (2, 8, tiny_config.vocab_size)

    def test_no_adapter_output_shape(self, tiny_config_no_adapters):
        model = GraniteSwitchForCausalLM(tiny_config_no_adapters).eval()
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 8, 256)


# ════════════════════════════════════════════════════════════════════
# 4. CausalLM output fields
# ════════════════════════════════════════════════════════════════════

class TestCausalLMOutputFields:

    def test_returns_causal_lm_output(self, tiny_config):
        model = GraniteSwitchForCausalLM(tiny_config).eval()
        _set_adapter_token_ids(model, tiny_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert isinstance(output, CausalLMOutputWithPast)
        assert output.logits is not None

    def test_use_cache_returns_past_key_values(self, tiny_config):
        model = GraniteSwitchForCausalLM(tiny_config).eval()
        _set_adapter_token_ids(model, tiny_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
        assert output.past_key_values is not None

    def test_labels_produce_loss(self, tiny_config):
        model = GraniteSwitchForCausalLM(tiny_config).eval()
        _set_adapter_token_ids(model, tiny_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=input_ids)
        assert output.loss is not None
        assert output.loss.dim() == 0  # scalar

    def test_output_hidden_states(self, tiny_config):
        model = GraniteSwitchForCausalLM(tiny_config).eval()
        _set_adapter_token_ids(model, tiny_config.adapter_token_ids)
        input_ids = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            output = model(input_ids=input_ids, output_hidden_states=True)
        assert output.hidden_states is not None
        # num_decoder_layers + 1 (input to first layer + output of each layer after norm)
        num_decoder_layers = tiny_config.num_hidden_layers - 1  # minus switch
        assert len(output.hidden_states) == num_decoder_layers + 1


# ════════════════════════════════════════════════════════════════════
# 5. Adapter indices wiring
# ════════════════════════════════════════════════════════════════════

class TestAdapterIndicesWiring:
    """End-to-end: switch → adapter_indices → LoRA modifies logits."""

    def _make_model(self, config):
        torch.manual_seed(42)
        model = GraniteSwitchForCausalLM(config).eval()
        _set_adapter_token_ids(model, config.adapter_token_ids)
        _set_nonzero_lora_B(model)
        return model

    def test_control_token_activates_adapter(self, tiny_config):
        """Logits diverge after a control token activates an adapter."""
        model = self._make_model(tiny_config)

        # Sequence with control token at position 2
        with_ctrl = torch.tensor([[10, 20, 250, 30, 40, 50, 60, 70]])
        # All-text sequence (same tokens, no control)
        no_ctrl = torch.tensor([[10, 20, 100, 30, 40, 50, 60, 70]])

        with torch.no_grad():
            logits_ctrl = model(input_ids=with_ctrl).logits
            logits_text = model(input_ids=no_ctrl).logits

        # Pre-control positions (0, 1): identical (causal, can't see position 2)
        torch.testing.assert_close(logits_ctrl[0, :2], logits_text[0, :2])

        # Post-control positions (3+): must differ (adapter active via LoRA)
        post_ctrl = logits_ctrl[0, 3:]
        post_text = logits_text[0, 3:]
        assert not torch.allclose(post_ctrl, post_text), \
            "Post-control logits should differ when adapter is active"

    def test_different_adapters_produce_different_post_control_logits(self, tiny_config):
        """Different control tokens → different adapters → different logits."""
        model = self._make_model(tiny_config)

        seq_a1 = torch.tensor([[10, 20, 250, 30, 40, 50, 60, 70]])  # adapter 1
        seq_a2 = torch.tensor([[10, 20, 251, 30, 40, 50, 60, 70]])  # adapter 2

        with torch.no_grad():
            logits_a1 = model(input_ids=seq_a1).logits
            logits_a2 = model(input_ids=seq_a2).logits

        # Pre-control positions: identical
        torch.testing.assert_close(logits_a1[0, :2], logits_a2[0, :2])

        # Post-control positions: differ (different LoRA weights)
        assert not torch.allclose(logits_a1[0, 3:], logits_a2[0, 3:]), \
            "Different adapters should produce different post-control logits"


# ════════════════════════════════════════════════════════════════════
# 6. Activating tokens: switch behavior (explicit adapter_indices)
# ════════════════════════════════════════════════════════════════════

class TestActivatingTokenSwitch:
    """Test that activating tokens properly trigger adapter switching."""

    def test_activating_adapter_indices_nonzero(self, tiny_single_config):
        """SingleSwitch: activating token produces adapter_indices > 0 at and after."""
        config = tiny_single_config
        model = GraniteSwitchForCausalLM(config).eval()
        _set_adapter_token_ids(model, config.adapter_token_ids)

        input_ids = torch.tensor([[10, 20, 250, 30, 40, 50, 60, 70]])
        with torch.no_grad():
            model(input_ids=input_ids)

        ai = model.model._last_adapter_indices
        assert (ai[:, :2] == 0).all(), "Pre-control positions should be base"
        assert (ai[:, 2:] > 0).all(), \
            f"Activating token should set adapter_indices > 0 at pos 2+, got {ai}"


# ════════════════════════════════════════════════════════════════════
# 7. Token-exchange forward pass tests
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def tiny_native_config():
    """Minimal config for token-exchange mode."""
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,  # 1 switch + 2 decoder
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
        adapter_names=["router", "planner"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=16,
    )


class TestNativeModeForward:
    """Forward pass tests with token-exchange enabled."""

    def test_forward_produces_logits(self, tiny_native_config):
        """Basic forward pass succeeds and produces correct-shaped logits."""
        config = tiny_native_config
        model = GraniteSwitchForCausalLM(config).eval()
        _set_adapter_token_ids(model, config.adapter_token_ids)

        input_ids = torch.tensor([[10, 250, 20, 30, 40]])
        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output.logits.shape == (1, 5, config.vocab_size)
        assert torch.isfinite(output.logits).all()

    def test_control_token_logits_finite(self, tiny_native_config):
        """Control token logits should be finite."""
        config = tiny_native_config
        model = GraniteSwitchForCausalLM(config).eval()
        _set_adapter_token_ids(model, config.adapter_token_ids)

        input_ids = torch.tensor([[250, 20, 30]])
        with torch.no_grad():
            output = model(input_ids=input_ids)

        # All control token logits should be finite
        for tid in config.adapter_token_ids:
            assert torch.isfinite(output.logits[:, :, tid]).all(), (
                f"Token {tid} logits should be finite"
            )

    def test_adapter_effect_visible(self, tiny_native_config):
        """Adapter activation should change logits."""
        config = tiny_native_config
        model = GraniteSwitchForCausalLM(config).eval()
        _set_adapter_token_ids(model, config.adapter_token_ids)
        _set_nonzero_lora_B(model, scale=0.1)

        base_ids = torch.tensor([[10, 20, 30, 40, 50]])
        adapter_ids = torch.tensor([[250, 20, 30, 40, 50]])

        with torch.no_grad():
            out_base = model(input_ids=base_ids)
            out_adapter = model(input_ids=adapter_ids)

        diff = (out_base.logits[0, -1] - out_adapter.logits[0, -1]).abs().max()
        assert diff > 1e-6, "Adapter should produce different logits"

    def test_batch_forward(self, tiny_native_config):
        """Batched forward pass works."""
        config = tiny_native_config
        model = GraniteSwitchForCausalLM(config).eval()
        _set_adapter_token_ids(model, config.adapter_token_ids)

        input_ids = torch.tensor([
            [10, 250, 20, 30, 40],
            [10, 251, 20, 30, 40],
        ])
        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output.logits.shape == (2, 5, config.vocab_size)
        assert torch.isfinite(output.logits).all()
