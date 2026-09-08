# SPDX-License-Identifier: Apache-2.0
"""The ``--base-reset-token`` compose option: CLI wiring and compose layout.

The base-reset slot reads ``adapter_token_ids[0]`` as expert id 0 when the
control-token list is ``num_adapters + 1`` long, so one request can return to
base mid-stream. MultiSwitch is the only engine, so the option is always valid;
these tests pin the CLI flag wiring and that compose emits the layout the engine
reads as base-reset.
"""

from unittest.mock import patch

from granite_switch.composer.compose_granite_switch import (
    _compose_argparser,
    build_control_token_lists,
)
from tests.hf.test_multi_switch import _MockSwitchConfig
from tests.shared.multi_switch_cases import TEXT_TOKEN

from .test_tokenizer_setup import MockTokenizer

_PROBE = (
    "granite_switch.composer.compose_granite_switch._probe_lora_substitute_token_id"
)
_ALORA = "granite_switch.composer.tokenizer_setup.get_alora_first_invocation_token_id"


class TestBaseResetCliFlag:
    def test_flag_defaults_to_off(self):
        """Existing behaviour must be unchanged unless asked for."""
        args = _compose_argparser().parse_args(["--adapters", "org/lib"])
        assert args.base_reset_token is False

    def test_flag_sets_true(self):
        args = _compose_argparser().parse_args(
            ["--adapters", "org/lib", "--base-reset-token"]
        )
        assert args.base_reset_token is True


class TestBuildControlTokenLists:
    """The compose seam: both lists must come out aligned, base slot first.

    ``GraniteSwitchConfig`` requires equal lengths and the token-exchange LUT
    zips them, so if only one list grew, every adapter's substitute would shift
    by one and each control token would swap to the wrong embedding.
    """

    _ADAPTERS = [("/a", "rag", "alora", None), ("/b", "code", "lora", None)]

    def _run(self, base_reset):
        tokenizer = MockTokenizer(initial_vocab_size=500)
        with patch(_PROBE, return_value=42), patch(_ALORA, return_value=77):
            return build_control_token_lists(tokenizer, self._ADAPTERS, base_reset)

    def test_base_reset_grows_both_lists_and_leads_with_base(self):
        token_ids, special_tokens, substitute_ids = self._run(base_reset=True)

        assert special_tokens[0] == "<|base_reset|>"
        assert len(token_ids) == len(self._ADAPTERS) + 1
        assert len(substitute_ids) == len(token_ids)
        assert substitute_ids[0] == 42  # base-reset slot takes the probed id
        assert substitute_ids[1:] == [77, 42]  # rag (alora), code (lora)

    def test_default_leaves_both_lists_at_num_adapters(self):
        token_ids, special_tokens, substitute_ids = self._run(base_reset=False)

        assert special_tokens == ["<|rag|>", "<|code|>"]
        assert len(token_ids) == len(self._ADAPTERS)
        assert substitute_ids == [77, 42]

    def test_compose_layout_is_the_one_the_engine_reads_as_base_reset(self):
        """The point of the whole option: compose output must route back to base.

        This is the link that was missing. The engine has always supported the
        base-reset layout and ``tests/hf/test_multi_switch.py`` proves it against
        a hand-written fixture, but the composer never produced that layout, so
        nothing connected the two. Feed the engine exactly what compose emits.
        """
        import torch

        from granite_switch.hf.switch import create_switch

        from .test_tokenizer_setup import MockTokenizer as _Tok

        tokenizer = _Tok(initial_vocab_size=500)
        with patch(_PROBE, return_value=42), patch(_ALORA, return_value=77):
            token_ids, _special, _subs = build_control_token_lists(
                tokenizer, self._ADAPTERS, base_reset=True
            )

        cfg = _MockSwitchConfig(
            "multi", token_ids, _subs, backend="sdpa", num_adapters=2
        )
        cfg.vocab_size = 2000
        switch = create_switch(cfg, layer_idx=0)
        assert switch._expert_id_offset == 0, (
            "compose emitted num_adapters+1 control tokens but the engine did not "
            "switch to the base-reset layout, so slot 0 would fire adapter 1"
        )

        # Look the ids up BY NAME, never by position: deriving them from the same
        # list the engine indexes would only prove self-consistency, and would pass
        # even if the base-reset token had been emitted in the wrong slot.
        base_tok = tokenizer.convert_tokens_to_ids("<|base_reset|>")
        rag_tok = tokenizer.convert_tokens_to_ids("<|rag|>")
        seq = torch.tensor([[rag_tok, TEXT_TOKEN, base_tok, TEXT_TOKEN]])
        indices, _ = switch.forward(
            input_ids=seq,
            adapter_token_ids=torch.tensor(token_ids),
            cache_position=torch.arange(seq.shape[1]),
        )
        assert [int(x) for x in indices[0]] == [1, 1, 0, 0], (
            "rag should hold until the base-reset token, then routing returns to base"
        )

    def test_default_layout_keeps_the_adapter_offset(self):
        """Without the option the engine must stay on the offset-1 layout.

        Guards against inverting the wiring: if offset 0 were selected for a
        plain compose, every adapter would be off by one.
        """
        from granite_switch.hf.switch import create_switch

        from .test_tokenizer_setup import MockTokenizer as _Tok

        tokenizer = _Tok(initial_vocab_size=500)
        with patch(_PROBE, return_value=42), patch(_ALORA, return_value=77):
            token_ids, _special, _subs = build_control_token_lists(
                tokenizer, self._ADAPTERS, base_reset=False
            )

        cfg = _MockSwitchConfig(
            "multi", token_ids, _subs, backend="sdpa", num_adapters=2
        )
        cfg.vocab_size = 2000
        assert create_switch(cfg, layer_idx=0)._expert_id_offset == 1
