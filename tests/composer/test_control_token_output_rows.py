# SPDX-License-Identifier: Apache-2.0
"""Tests for the control tokens' output (LM head) rows.

``resize_token_embeddings`` appends a row per control token, sampled from the
distribution of the *trained* rows (``mean_resizing=True`` since transformers
4.46), so without a fixup each control token carries an arbitrary,
compose-run-dependent output logit despite never having been trained. Nothing
suppresses control tokens at generation time, so the model can emit one.

The fixup copies a reserved ``<|unused_N|>`` row — which the base model was
trained not to emit — into every control token's row. Covered here:

* the reserved-token lookup (found, absent, highest-id-wins)
* the row copy on the untied path (Granite 4.2, distinct ``lm_head``)
* the row copy on the tied path (Granite 4.0/4.1, shared matrix)
* control rows are *not* copied from their token-exchange substitutes, which was
  the previous policy (it made a control token's logit equal its substitute's)
* neighbouring rows are left alone
* the tied path's shared-matrix write is inert: a control token's *input* row is
  never read, because the switch rewrites the id before the embedding lookup
"""

import pytest
import torch

from granite_switch.composer.compose_granite_switch import (
    initialize_control_token_output_rows,
)
from granite_switch.composer.tokenizer_setup import (
    find_reserved_never_emitted_token_id,
)
from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM

# Ids inside _tiny_config's 300-token vocabulary.
_CONTROL_IDS = [250, 251]
_SUBSTITUTE_IDS = [1, 7]
_RESERVED_ID = 253
_BYSTANDER_ID = 254
# Stands in for the <|base_reset|> slot MultiSwitch prepends at index 0.
_BASE_RESET_ID = 249


def _tiny_config(tie: bool) -> GraniteSwitchConfig:
    return GraniteSwitchConfig(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=list(_CONTROL_IDS),
        adapter_substitute_token_ids=list(_SUBSTITUTE_IDS),
        adapter_names=["adapter_a", "adapter_b"],
        max_lora_rank=4,
        adapter_ranks=[4, 4],
        switch_head_dim=16,
        tie_word_embeddings=tie,
    )


class _FakeTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


class TestFindReservedNeverEmittedTokenId:
    def test_returns_highest_reserved_id(self):
        tok = _FakeTokenizer(
            {"hello": 1, "<|unused_1|>": 90, "<|unused_7|>": 96, "<|unused_3|>": 92}
        )
        # Highest *id*, not highest N — the ids are what index the weight matrix.
        assert find_reserved_never_emitted_token_id(tok) == 96

    def test_returns_none_when_vocab_has_no_reserved_tokens(self):
        assert find_reserved_never_emitted_token_id(_FakeTokenizer({"a": 0})) is None

    def test_does_not_match_lookalike_tokens(self):
        tok = _FakeTokenizer(
            {"<|unused|>": 5, "<|unused_x|>": 6, "unused_1": 7, "<|unused_1|>x": 8}
        )
        assert find_reserved_never_emitted_token_id(tok) is None


class TestInitializeControlTokenOutputRows:
    """Every control token's output row becomes an exact copy of the reserved row."""

    @pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
    def test_rows_copied_from_reserved_row(self, tie):
        model = GraniteSwitchForCausalLM(_tiny_config(tie=tie))
        head = model.get_output_embeddings().weight

        reserved_before = head[_RESERVED_ID].clone()
        # Guard the test itself: the rows must differ beforehand, otherwise the
        # assertions below would pass without the copy doing anything.
        for control_id in _CONTROL_IDS:
            assert not torch.equal(head[control_id], reserved_before)

        initialize_control_token_output_rows(model, _CONTROL_IDS, _RESERVED_ID)

        head = model.get_output_embeddings().weight
        for control_id in _CONTROL_IDS:
            assert torch.equal(head[control_id], reserved_before)
        # The source row is copied from, not moved.
        assert torch.equal(head[_RESERVED_ID], reserved_before)

    @pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
    def test_rows_not_copied_from_substitutes(self, tie):
        """Regression guard against the previous substitute-row policy.

        Copying the substitute's row gave the control token a logit identical to
        the substitute's, so the two split the probability mass wherever the
        substitute was the natural next token.
        """
        model = GraniteSwitchForCausalLM(_tiny_config(tie=tie))
        head = model.get_output_embeddings().weight
        with torch.no_grad():
            head[_SUBSTITUTE_IDS[0]] = 3.0
            head[_SUBSTITUTE_IDS[1]] = -5.0

        initialize_control_token_output_rows(model, _CONTROL_IDS, _RESERVED_ID)

        head = model.get_output_embeddings().weight
        for control_id, substitute_id in zip(_CONTROL_IDS, _SUBSTITUTE_IDS):
            assert not torch.equal(head[control_id], head[substitute_id])

    @pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
    def test_other_rows_untouched(self, tie):
        model = GraniteSwitchForCausalLM(_tiny_config(tie=tie))
        head = model.get_output_embeddings().weight
        bystander = head[_BYSTANDER_ID].clone()
        substitutes = head[_SUBSTITUTE_IDS].clone()

        initialize_control_token_output_rows(model, _CONTROL_IDS, _RESERVED_ID)

        head = model.get_output_embeddings().weight
        assert torch.equal(head[_BYSTANDER_ID], bystander)
        assert torch.equal(head[_SUBSTITUTE_IDS], substitutes)

    @pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
    def test_covers_the_base_reset_slot(self, tie):
        """MultiSwitch's ``<|base_reset|>`` rides in the same list, at index 0.

        Composed with ``--base-reset-token``, ``adapter_token_ids`` is one longer
        than ``num_adapters``. The base-reset token is added and never trained
        like every other control token, so it must get the reserved row too —
        which it does by being in the list the loop walks.
        """
        model = GraniteSwitchForCausalLM(_tiny_config(tie=tie))
        head = model.get_output_embeddings().weight
        reserved_before = head[_RESERVED_ID].clone()
        control_ids = [_BASE_RESET_ID, *_CONTROL_IDS]

        initialize_control_token_output_rows(model, control_ids, _RESERVED_ID)

        head = model.get_output_embeddings().weight
        for control_id in control_ids:
            assert torch.equal(head[control_id], reserved_before)

    @pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
    def test_empty_control_list_is_a_noop(self, tie):
        """Zero-adapter skinning has no control tokens to fix up."""
        model = GraniteSwitchForCausalLM(_tiny_config(tie=tie))
        before = model.get_output_embeddings().weight.clone()

        initialize_control_token_output_rows(model, [], _RESERVED_ID)

        assert torch.equal(model.get_output_embeddings().weight, before)

    def test_tied_path_also_rewrites_the_shared_input_rows(self):
        """On the tied path the write lands in the shared matrix.

        That is intentional and inert — see
        ``TestControlTokenInputRowIsNeverRead`` below, which pins the property
        this relies on.
        """
        model = GraniteSwitchForCausalLM(_tiny_config(tie=True))
        initialize_control_token_output_rows(model, _CONTROL_IDS, _RESERVED_ID)
        embed = model.get_input_embeddings().weight
        head = model.get_output_embeddings().weight
        for control_id in _CONTROL_IDS:
            assert torch.equal(embed[control_id], head[_RESERVED_ID])

    def test_untied_path_leaves_the_input_embeddings_alone(self):
        """On the untied path only the head is touched; the input rows are separate."""
        model = GraniteSwitchForCausalLM(_tiny_config(tie=False))
        embed_before = model.get_input_embeddings().weight[_CONTROL_IDS].clone()

        initialize_control_token_output_rows(model, _CONTROL_IDS, _RESERVED_ID)

        assert torch.equal(
            model.get_input_embeddings().weight[_CONTROL_IDS], embed_before
        )


class TestControlTokenInputRowIsNeverRead:
    """The property that makes the tied-path (Granite 4.1) fixup safe.

    Overwriting a control token's row in a *tied* checkpoint also overwrites its
    input embedding. That is only acceptable because the row is never read: the
    switch rewrites each control-token id to its substitute id before the
    embedding lookup, so the decoder embeds the substitute instead.

    ``tests/vllm/_model_forward_tests.py::TestKVVisibility`` covers the vLLM side
    but needs a GPU; this is the CPU-runnable HF equivalent, and it is what
    justifies dropping the ``tie_word_embeddings`` gate in the composer.
    """

    def test_perturbing_a_control_row_does_not_change_the_forward(self):
        torch.manual_seed(0)
        model = GraniteSwitchForCausalLM(_tiny_config(tie=False)).eval()
        # A control token mid-sequence, so there are positions both before and
        # after it whose hidden states could pick up a leak.
        input_ids = torch.tensor([[10, 20, _CONTROL_IDS[0], 30, 40, 50]])

        with torch.no_grad():
            before = model(input_ids=input_ids).logits

        with torch.no_grad():
            model.get_input_embeddings().weight[_CONTROL_IDS[0]] += (
                torch.randn(model.config.hidden_size) * 10.0
            )
            after = model(input_ids=input_ids).logits

        torch.testing.assert_close(before, after)
