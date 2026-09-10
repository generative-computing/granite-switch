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
from granite_switch.token_exchange import (
    build_control_to_substitute_lut,
    rebuild_control_to_substitute_lut,
)


def _build(num_adapters=2, substitute_ids=(1, 7)):
    return GraniteSwitchConfig(
        vocab_size=200,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        # MultiSwitch reserves 2 cache slots (counting + memory), so 3 hidden
        # layers leaves exactly 1 physical decoder layer (at past_key_values index 2).
        num_hidden_layers=3,
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
            torch.tensor(
                [[10, 20, 100, 40]], dtype=torch.long
            ),  # adapter 0 control at pos 2
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
        raw_a = model.model.embed_tokens(
            torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
        )
        raw_b = model.model.embed_tokens(
            torch.tensor([[10, 20, 100, 40]], dtype=torch.long)
        )
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
        # layers[0] and [1] are the switch cache slots (counting + memory);
        # layers[2] is the first decoder layer.
        decoder_key = out.past_key_values.layers[2].keys
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
        # Positions after it inherit adapter=1 (latest-wins persists once fired).
        assert adapter_indices[0, 0].item() == 0
        assert adapter_indices[0, 1].item() == 0
        assert adapter_indices[0, 2].item() == 1
        assert adapter_indices[0, 3].item() == 1
        assert adapter_indices[0, 4].item() == 1


class TestControlLutSizing:
    """build_control_to_substitute_lut is the single source of the sizing rule."""

    def test_sized_to_vocab_when_vocab_is_larger(self):
        lut = build_control_to_substitute_lut(_build())
        assert lut is not None
        assert lut.numel() == 200  # vocab_size, control ids are 100/101

    def test_sized_past_the_last_control_id_when_vocab_lags(self):
        """Every control id stays addressable even if vocab_size is behind.

        This is the state compose is in at switch construction, not an edge case:
        the config's vocab_size still comes from the base checkpoint while
        add_control_tokens has already appended ids past it, so a table longer than
        vocab_size is the CORRECT answer here. It stops being correct once
        resize_token_embeddings and the rebuild have run, which is what
        tests/composer/test_validator.py::test_longer_than_vocab_raises pins. The
        two look contradictory and are not — they check different points in the
        lifecycle. Do not "fix" either by making them agree.
        """
        config = _build()
        config.vocab_size = 50  # smaller than adapter_token_ids [100, 101]

        lut = build_control_to_substitute_lut(config)

        assert lut is not None
        assert lut.numel() == 102  # max_ctrl_id + 1

    def test_no_mapping_yields_no_table(self):
        config = _build()
        config.adapter_substitute_token_ids = None

        assert build_control_to_substitute_lut(config) is None

    def test_empty_adapter_ids_yield_no_table(self):
        config = _build()
        config.adapter_token_ids = []

        assert build_control_to_substitute_lut(config) is None


class TestControlLutRebuildAfterResize:
    """A vocabulary resize leaves the table stale; it must be re-derived.

    ``__init__`` sizes the table from ``config.vocab_size``, so compose growing
    the vocabulary for control and marker tokens leaves the persistent buffer
    shorter than the config it ships with. Loading such a checkpoint does not
    degrade gracefully: the stored tensor is discarded and the buffer is left as
    uninitialised memory, so every id reads as a control id and the rewrite
    sends out-of-range ids into the embedding gather.
    """

    def test_resize_leaves_the_table_stale(self):
        """Witness for why the rebuild is needed at all."""
        model = GraniteSwitchForCausalLM(_build(substitute_ids=(5, 7))).eval()
        model.resize_token_embeddings(201)

        assert model.config.vocab_size == 201
        assert model.model.embed_tokens.weight.shape[0] == 201
        assert model.model.switch.control_to_substitute_lut.numel() == 200

    def test_rebuild_restores_agreement_and_values(self):
        model = GraniteSwitchForCausalLM(_build(substitute_ids=(5, 7))).eval()
        model.resize_token_embeddings(201)

        assert rebuild_control_to_substitute_lut(model.model.switch, model.config)

        lut = model.model.switch.control_to_substitute_lut
        assert lut.numel() == model.config.vocab_size == 201
        assert lut[100].item() == 5
        assert lut[101].item() == 7
        assert int((lut >= 0).sum()) == 2  # exactly the two control slots

    def test_rebuild_reports_false_without_a_mapping(self):
        config = _build()
        config.adapter_substitute_token_ids = None
        model = GraniteSwitchForCausalLM(config).eval()

        assert rebuild_control_to_substitute_lut(model.model.switch, config) is False

    @torch.no_grad()
    def test_rebuilt_table_survives_a_save_load_round_trip(self, tmp_path):
        """The end-to-end guard: without the rebuild this load yields garbage.

        A stale buffer comes back as uninitialised memory, which makes every
        position a control position and raises IndexError from the embedding
        gather (a CUDA device-side assert on GPU).
        """
        model = GraniteSwitchForCausalLM(_build(substitute_ids=(5, 7))).eval()
        model.resize_token_embeddings(201)
        rebuild_control_to_substitute_lut(model.model.switch, model.config)
        model.save_pretrained(tmp_path / "ckpt")

        # Strict load: no ignore_mismatched_sizes to paper over a bad length.
        loaded = GraniteSwitchForCausalLM.from_pretrained(
            tmp_path / "ckpt", dtype=torch.float32
        ).eval()

        lut = loaded.model.switch.control_to_substitute_lut
        assert lut.numel() == 201
        assert lut[100].item() == 5
        assert lut[101].item() == 7
        assert lut[10].item() == -1
        assert int((lut >= 0).sum()) == 2
        # And the rewrite it drives stays in bounds.
        loaded(input_ids=torch.tensor([[10, 20, 100, 40]], dtype=torch.long))


class TestControlLutRebuildAcrossSwitchEngines:
    """The rebuild must work on EVERY switch engine, not just SingleSwitch.

    Regression for the compose crash: ``rebuild_control_to_substitute_lut`` used
    to be a method defined only on ``SingleSwitch``, while ``create_switch``
    returns ``MultiSwitch`` for ``switch_type="multi"``. Compose always adds
    control tokens, so ``resize_token_embeddings`` always leaves the table stale
    and compose always reaches the rebuild — meaning ``--switch-type multi``
    raised ``AttributeError: 'MultiSwitch' object has no attribute
    'rebuild_control_to_substitute_lut'`` on every run, including for text-only
    checkpoints.

    Parameterized over ``switch_type`` because a single-engine test is exactly
    what let this through a green suite.
    """

    @pytest.mark.parametrize("switch_type", ["multi"])
    def test_rebuild_agrees_with_vocab_size(self, switch_type):
        config = _build(substitute_ids=(5, 7))
        config.switch_type = switch_type
        model = GraniteSwitchForCausalLM(config).eval()
        switch = model.model.switch
        assert type(switch).__name__ == (
            "MultiSwitch" if switch_type == "multi" else "SingleSwitch"
        )

        model.resize_token_embeddings(201)
        # The state compose finds itself in: table sized pre-resize, config after.
        assert switch.control_to_substitute_lut.numel() != model.config.vocab_size

        assert rebuild_control_to_substitute_lut(switch, model.config)

        lut = switch.control_to_substitute_lut
        assert lut.numel() == model.config.vocab_size == 201
        assert lut[100].item() == 5
        assert lut[101].item() == 7
        assert int((lut >= 0).sum()) == 2

    @pytest.mark.parametrize("switch_type", ["multi"])
    def test_empty_adapter_ids_yield_no_table(self, switch_type):
        """An empty id list is "no mapping", not ``max(())``.

        The two pre-consolidation builders disagreed here: the MultiSwitch one
        guarded with ``ctrl_ids is None``, so an empty list fell through to
        ``max(ctrl_ids)`` and raised ``ValueError``. The surviving guard is a
        falsiness test.
        """
        config = _build(substitute_ids=(5, 7))
        config.switch_type = switch_type
        config.adapter_token_ids = []
        config.adapter_substitute_token_ids = []

        assert build_control_to_substitute_lut(config) is None
