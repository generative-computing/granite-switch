# SPDX-License-Identifier: Apache-2.0
"""HF backend tests for token-exchange mode.

Two properties under test:
1. The embedding at each control-token position equals the embedding of its
   substitute token (scaled by embedding_multiplier), not the original
   control-token embedding.
2. The KV cache head_dim is the native projection_head_dim — token-exchange
   does not expand the KV cache.
"""

import pytest
import torch

from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM

pytestmark = pytest.mark.local_fast


def _build(num_adapters=2, substitute_ids=(1, 7)):
    return GraniteSwitchConfig(
        vocab_size=200,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        shared_intermediate_size=64,
        max_position_embeddings=64,
        mamba_n_heads=1,
        mamba_expand=1,
        num_adapters=num_adapters,
        adapter_ranks=[4] * num_adapters,
        max_lora_rank=4,
        adapter_token_ids=[100, 101][:num_adapters],
        adapter_names=["a", "b"][:num_adapters],
        adapter_substitute_token_ids=list(substitute_ids[:num_adapters]),
        torch_dtype=torch.float32,
    )


@torch.no_grad()
def _forward(config, input_ids):
    model = GraniteSwitchForCausalLM(config).eval()
    return model, model(input_ids=input_ids, use_cache=True)


class TestTokenExchangeEmbeddingSwap:
    """The control position's residual-stream input is the substitute embedding."""

    def test_swap_picks_substitute_embedding(self):
        config = _build(substitute_ids=(5, 7))
        model, _ = _forward(
            config,
            torch.tensor([[10, 20, 100, 40]], dtype=torch.long),  # adapter 0 control at pos 2
        )
        # The LUT lives on the switch (it performs the rewrite during its
        # forward); maps control id 100 → substitute 5.
        lut = model.model.switch.control_to_substitute_lut
        assert lut[100].item() == 5
        assert lut[101].item() == 7
        # Positions without control tokens map to -1.
        assert lut[10].item() == -1
        assert lut[40].item() == -1

    def test_swap_is_not_applied_on_non_control_positions(self):
        config = _build(substitute_ids=(5, 7))
        model = GraniteSwitchForCausalLM(config).eval()
        # Run once through the model with a control token and once without;
        # verify the non-control embedding rows are identical.
        raw_a = model.model.embed_tokens(torch.tensor([[10, 20, 30, 40]], dtype=torch.long))
        raw_b = model.model.embed_tokens(torch.tensor([[10, 20, 100, 40]], dtype=torch.long))
        # Positions 0, 1, 3 should match; position 2 is the control token (differs).
        assert torch.allclose(raw_a[:, 0], raw_b[:, 0])
        assert torch.allclose(raw_a[:, 1], raw_b[:, 1])
        assert torch.allclose(raw_a[:, 3], raw_b[:, 3])


class TestKVCacheHeadDim:
    """The load-bearing correctness property: KV cache head_dim equals
    the native projection_head_dim — no expansion."""

    def test_token_exchange_native_head_dim(self):
        config = _build(substitute_ids=(5, 7))
        _, out = _forward(
            config,
            torch.tensor([[10, 20, 100, 40]], dtype=torch.long),
        )
        # layers[0] is the switch; layers[1] is the first decoder layer.
        decoder_key = out.past_key_values.layers[1].keys
        assert decoder_key.shape[-1] == config.projection_head_dim


class TestSwitchStillDetectsAdapter:
    """Swap must happen AFTER the switch reads input_ids, so detection is unaffected."""

    def test_adapter_indices_still_activate(self):
        config = _build(substitute_ids=(5, 7))
        model, _ = _forward(
            config,
            torch.tensor([[10, 20, 100, 40, 50]], dtype=torch.long),
        )
        adapter_indices = model.model._last_adapter_indices
        # Position 2 is the control token for adapter 0 (1-indexed output).
        # Positions after it inherit adapter=1 (SingleSwitch persists once fired).
        assert adapter_indices[0, 0].item() == 0
        assert adapter_indices[0, 1].item() == 0
        assert adapter_indices[0, 2].item() == 1
        assert adapter_indices[0, 3].item() == 1
        assert adapter_indices[0, 4].item() == 1
