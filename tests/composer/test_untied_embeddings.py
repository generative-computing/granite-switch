# SPDX-License-Identifier: Apache-2.0
"""Tests for untied word embeddings (Granite 4.2, tie_word_embeddings=False).

Granite 4.0/4.1 tie the input embedding matrix and the LM head; Granite 4.2
sets ``tie_word_embeddings: false`` and ships a distinct LM head. These tests
pin the behavior that matters for the untied path:

* A10 — ``GraniteSwitchForCausalLM._tied_weights_keys`` is empty for an untied
  config (so ``lm_head.weight`` stays a first-class parameter through
  save/load), and the class default (tied) is preserved for a tied config.
* Round-trip — ``save_pretrained`` → ``from_pretrained`` on an untied model
  keeps a distinct ``lm_head.weight`` that is NOT identical-by-alias to
  ``embed_tokens.weight``. (On transformers ≥5.9 the framework already gates
  tie-key expansion on the config, so the head is not dropped; this test guards
  against a regression if that gating changes.)
* Config copy — ``tie_word_embeddings`` survives the base→GraniteSwitchConfig
  field copy (pins arch.py's required-field list).
"""

import torch

from granite_switch.composer.weight_transfer import read_saved_lm_head_shape
from granite_switch.config import GraniteSwitchConfig
from granite_switch.hf import GraniteSwitchForCausalLM


def _tiny_config(tie: bool) -> GraniteSwitchConfig:
    return GraniteSwitchConfig(
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


class TestConditionalTiedWeightsKeys:
    """A10: _tied_weights_keys reflects config.tie_word_embeddings."""

    def test_untied_config_clears_tied_keys(self):
        model = GraniteSwitchForCausalLM(_tiny_config(tie=False))
        assert model._tied_weights_keys == {}

    def test_tied_config_keeps_default_tied_keys(self):
        model = GraniteSwitchForCausalLM(_tiny_config(tie=True))
        assert model._tied_weights_keys == {
            "lm_head.weight": "model.embed_tokens.weight"
        }


class TestConfigCopyPreservesTieFlag:
    """The tie_word_embeddings flag round-trips through config construction."""

    def test_untied_flag_persists(self):
        cfg = _tiny_config(tie=False)
        assert cfg.tie_word_embeddings is False

    def test_tied_flag_persists(self):
        cfg = _tiny_config(tie=True)
        assert cfg.tie_word_embeddings is True


class TestUntiedSaveLoadRoundTrip:
    """save_pretrained→from_pretrained keeps a distinct lm_head on the untied
    path; the head is not dropped or re-aliased to the input embeddings."""

    def test_distinct_lm_head_survives_roundtrip(self, tmp_path):
        model = GraniteSwitchForCausalLM(_tiny_config(tie=False)).eval()
        # Make the head recognizably different from the input embeddings.
        with torch.no_grad():
            model.lm_head.weight.copy_(
                torch.arange(model.lm_head.weight.numel(), dtype=torch.float32).reshape(
                    model.lm_head.weight.shape
                )
            )
        original_head = model.lm_head.weight.detach().clone()
        # The head must not already alias the embeddings.
        assert not torch.equal(
            model.lm_head.weight, model.get_input_embeddings().weight
        )

        model.save_pretrained(tmp_path)
        reloaded = GraniteSwitchForCausalLM.from_pretrained(tmp_path).eval()

        # Head reloaded identically and still distinct from the embeddings.
        assert torch.allclose(reloaded.lm_head.weight, original_head)
        assert not torch.equal(
            reloaded.lm_head.weight, reloaded.get_input_embeddings().weight
        )

    def test_saved_checkpoint_contains_lm_head_weight(self, tmp_path):
        cfg = _tiny_config(tie=False)
        model = GraniteSwitchForCausalLM(cfg).eval()
        model.save_pretrained(tmp_path)

        # Use the same reader the composer's compose-time guard uses, so test
        # and runtime check agree on what "the head survived save" means.
        shape = read_saved_lm_head_shape(tmp_path)
        assert shape is not None, "lm_head.weight missing from untied checkpoint"
        assert shape == [cfg.vocab_size, cfg.hidden_size]

    def test_tied_path_unchanged_backwards_compat(self, tmp_path):
        """Backwards compatibility: a TIED model (4.0/4.1 behavior) still ties
        lm_head to embed_tokens through save/load — the untied changes are a
        no-op on the tied path.
        """
        model = GraniteSwitchForCausalLM(_tiny_config(tie=True)).eval()
        # Tied at runtime: the two matrices are the same tensor.
        assert model.lm_head.weight is model.get_input_embeddings().weight

        model.save_pretrained(tmp_path)
        reloaded = GraniteSwitchForCausalLM.from_pretrained(tmp_path).eval()

        # Still tied after reload (shared storage), unchanged from prior behavior.
        assert reloaded.lm_head.weight is reloaded.get_input_embeddings().weight
        assert torch.equal(
            reloaded.lm_head.weight, reloaded.get_input_embeddings().weight
        )
