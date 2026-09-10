# SPDX-License-Identifier: Apache-2.0
"""Tests for the audio marker's output (LM head) row.

``resize_token_embeddings`` appends a row for ``<|audio|>`` sampled from the
distribution of the *trained* rows (``mean_resizing=True``), so without a fixup
the marker carries an arbitrary, compose-run-dependent output logit even though
it was never trained. Nothing suppresses control tokens or the marker at
generation time, so the model can emit ``<|audio|>`` — and a marker in a reply
that is fed back on a later turn makes the processor's marker/audio-item counts
disagree and the request is rejected.

The fixup copies a reserved ``<|unused_N|>`` row, whose logit the base model was
trained to keep low, into the marker's row. Covered here:

* the reserved-token lookup (found, absent, highest-id-wins)
* the row copy on the untied path (distinct ``lm_head``)
* the row copy on the tied path (shared matrix)
* neighbouring rows are left alone

The row copy is also parametrized over the MLP topology. The fixup only ever
touches embedding rows, so a dense base and a pure sparse MoE base
(``shared_intermediate_size == 0``, no ``shared_mlp`` module at all) must behave
identically -- and if a shared-MLP-shaped assumption ever creeps into the model
construction the fixup runs against, the sparse arm is what notices.
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
_AUDIO_ID = 252
_RESERVED_ID = 253
_BYSTANDER_ID = 254


# (tie, sparse_moe) ids for the row-copy cases.
_TOPOLOGIES = [
    pytest.param(False, False, id="untied-dense"),
    pytest.param(True, False, id="tied-dense"),
    pytest.param(False, True, id="untied-sparse_moe"),
    pytest.param(True, True, id="tied-sparse_moe"),
]


def _tiny_config(tie: bool, sparse_moe: bool = False) -> GraniteSwitchConfig:
    # shared_intermediate_size == 0 is upstream's encoding for "no shared MLP",
    # so it is a meaningful value, never a falsy one. A layer needs at least one
    # MLP path, hence the expert bank.
    moe_fields = (
        {
            "shared_intermediate_size": 0,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
        }
        if sparse_moe
        else {}
    )
    return GraniteSwitchConfig(
        **moe_fields,
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_adapters=2,
        adapter_token_ids=[250, 251],
        adapter_substitute_token_ids=[1, 1],
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


@pytest.mark.audio
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

    def test_finds_reserved_token_on_real_granite_tokenizer(self):
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-3b")
        except Exception as e:
            pytest.skip(f"could not fetch Granite tokenizer: {e}")
        token_id = find_reserved_never_emitted_token_id(tok)
        assert token_id is not None
        assert tok.convert_ids_to_tokens(token_id).startswith("<|unused_")


@pytest.mark.audio
class TestInitializeAudioMarkerOutputRow:
    """The marker's output row becomes an exact copy of the reserved row.

    The marker goes through ``initialize_control_token_output_rows`` like every
    other never-trained id (it used to have its own near-identical function).
    These cases stay marker-specific on purpose: the marker is the one id whose
    emission breaks a *later* turn, via the processor's marker/audio-item count,
    so it is worth pinning separately from the control tokens.
    """

    @pytest.mark.parametrize(("tie", "sparse_moe"), _TOPOLOGIES)
    def test_marker_row_matches_reserved_row(self, tie, sparse_moe):
        model = GraniteSwitchForCausalLM(_tiny_config(tie=tie, sparse_moe=sparse_moe))
        head = model.get_output_embeddings().weight

        reserved_before = head[_RESERVED_ID].clone()
        # Guard the test itself: the rows must differ beforehand, otherwise the
        # assertion below would pass without the copy doing anything.
        assert not torch.equal(head[_AUDIO_ID], reserved_before)

        initialize_control_token_output_rows(model, [_AUDIO_ID], _RESERVED_ID)

        head = model.get_output_embeddings().weight
        assert torch.equal(head[_AUDIO_ID], reserved_before)
        # The source row is copied from, not moved.
        assert torch.equal(head[_RESERVED_ID], reserved_before)

    @pytest.mark.parametrize(("tie", "sparse_moe"), _TOPOLOGIES)
    def test_other_rows_untouched(self, tie, sparse_moe):
        model = GraniteSwitchForCausalLM(_tiny_config(tie=tie, sparse_moe=sparse_moe))
        head = model.get_output_embeddings().weight
        bystander = head[_BYSTANDER_ID].clone()
        control_rows = head[[250, 251]].clone()

        initialize_control_token_output_rows(model, [_AUDIO_ID], _RESERVED_ID)

        head = model.get_output_embeddings().weight
        assert torch.equal(head[_BYSTANDER_ID], bystander)
        assert torch.equal(head[[250, 251]], control_rows)

    def test_tied_path_also_rewrites_the_shared_input_row(self):
        """On the tied path the write lands in the shared matrix.

        That is intentional and inert: the marker is replaced by the transcript's
        token ids before the decoder runs, and a marker with no matching audio
        item is rejected up-front, so the marker's input row is never read. This
        test documents the aliasing rather than guarding against it.
        """
        model = GraniteSwitchForCausalLM(_tiny_config(tie=True))
        initialize_control_token_output_rows(model, [_AUDIO_ID], _RESERVED_ID)
        embed = model.get_input_embeddings().weight
        head = model.get_output_embeddings().weight
        assert torch.equal(embed[_AUDIO_ID], head[_RESERVED_ID])

    def test_untied_path_leaves_the_input_embedding_alone(self):
        """On the untied path only the head is touched; the input row is separate."""
        model = GraniteSwitchForCausalLM(_tiny_config(tie=False))
        embed_before = model.get_input_embeddings().weight[_AUDIO_ID].clone()
        initialize_control_token_output_rows(model, [_AUDIO_ID], _RESERVED_ID)
        assert torch.equal(model.get_input_embeddings().weight[_AUDIO_ID], embed_before)
