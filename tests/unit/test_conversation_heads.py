# SPDX-License-Identifier: Apache-2.0
"""Head turns: a head asks about the conversation; it is not a turn in it.

A head stream reads the hidden state at its marker, the prompt's last token,
and emits one token. Two things go wrong if ``Conversation`` treats it as an
ordinary turn:

* ``PRESERVE_MIXED_HISTORY`` keeps the ids sent so far. A control token kept
  there runs its adapter over the history, and a LoRA or aLoRA writes the K/V of
  those tokens, so a head "on base" would read a state that adapter changed.
* Recording a head turn would keep its marker inside every later prompt, where
  it routes the tokens after it through the head's slot (HF raises on such a
  prompt; vLLM does not check).

So a head turn is rendered from text whenever the kept ids hold a control
token, and its verdict joins the transcript as text, never as kept ids. These
run on CPU against the real chat template (``tests/shared/conversation_stubs``).
"""

import pytest

from granite_switch import Conversation, KVHistoryPolicy
from granite_switch.composer.tokenizer_setup import ANCHOR_MODE_HEAD
from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

AL, TOX, PAIR = "uncertainty", "tox", "tox_on_uncertainty"
# An aLoRA invoked inside the user message, a head on the base model, and the
# same kind of head on the aLoRA (option B).
ADAPTERS = [
    (AL, "alora", "<certainty>"),
    (TOX, ANCHOR_MODE_HEAD, None),
    (PAIR, ANCHOR_MODE_HEAD, None),
]
PRESERVE = KVHistoryPolicy.PRESERVE_MIXED_HISTORY
RE_PREFILL = KVHistoryPolicy.RE_PREFILL


@pytest.fixture
def tok():
    return make_stub_tokenizer(ADAPTERS, head_adapters={PAIR: AL})


@pytest.fixture
def config(tok):
    names = [name for name, _tech, _inv in ADAPTERS]
    return StubConfig(
        [tok.token_id(f"<|{name}|>") for name in names],
        adapter_names=names,
        adapter_kinds=["lora", "probe", "probe"],
    )


def _token(tok, name):
    return tok.token_id(f"<|{name}|>")


def _controls(ids, config):
    return [t for t in ids if t in set(config.adapter_token_ids)]


def _scratch(conv, adapter):
    """The transcript rendered from text, with only this turn's control token."""
    return list(conv._encode(conv._render(conv._messages, gen=True, adapter=adapter)))


def _after_an_alora_turn(tok, config, policy):
    conv = Conversation(tok, policy=policy, config=config)
    conv.user("Is this answerable from the context? <certainty>")
    conv.build_prompt(adapter=AL)
    conv.record_answer("Yes, with high confidence.", adapter=AL)
    return conv


class TestHeadReadsOnlyThisTurn:
    def test_kept_control_tokens_are_not_in_the_head_prompt(self, tok, config):
        conv = _after_an_alora_turn(tok, config, PRESERVE)
        kept = conv.sent_token_ids
        assert _token(tok, AL) in kept  # the history holds the aLoRA's token

        conv.user("Is the answer rude?")
        ids = list(conv.build_prompt(adapter=TOX))

        assert ids[-1] == _token(tok, TOX)
        assert _controls(ids, config) == [_token(tok, TOX)]
        assert ids == _scratch(conv, TOX)
        # Not a re-prefill: the kept history is not touched.
        assert conv.reprefills == 0
        assert conv.sent_token_ids == kept

    def test_an_adapter_turn_still_appends_to_the_kept_ids(self, tok, config):
        """The twin: only a head turn leaves the kept ids aside."""
        conv = _after_an_alora_turn(tok, config, PRESERVE)
        kept = conv.sent_token_ids
        conv.user("And now? <certainty>")
        ids = list(conv.build_prompt(adapter=AL))
        assert ids[: len(kept)] == kept

    def test_clean_history_is_reused(self, tok, config):
        """Kept ids with no control token change nothing the head reads."""
        conv = Conversation(tok, policy=PRESERVE, config=config)
        conv.user("Hello.")
        conv.build_prompt()
        conv.record_answer("Hi, how can I help?")
        kept = conv.sent_token_ids
        assert not _controls(kept, config)

        conv.user("Is that reply rude?")
        ids = list(conv.build_prompt(adapter=TOX))
        assert ids[: len(kept)] == kept
        assert ids[-1] == _token(tok, TOX)

    def test_re_prefill_head_turn_has_only_its_marker(self, tok, config):
        conv = _after_an_alora_turn(tok, config, RE_PREFILL)
        conv.user("Is the answer rude?")
        ids = list(conv.build_prompt(adapter=TOX))
        assert _controls(ids, config) == [_token(tok, TOX)]
        assert ids[-1] == _token(tok, TOX)


class TestRecordingAHeadTurn:
    @pytest.mark.parametrize("policy", [PRESERVE, RE_PREFILL])
    def test_the_marker_never_reaches_a_later_prompt(self, tok, config, policy):
        conv = _after_an_alora_turn(tok, config, policy)
        kept = conv.sent_token_ids
        conv.user("Is the answer rude?")
        conv.build_prompt(adapter=TOX)
        assert conv.record_answer("no", adapter=TOX) == "no"

        # The verdict is in the transcript, as text.
        assert conv.messages[-2:] == [
            {"role": "user", "content": "Is the answer rude?"},
            {"role": "assistant", "content": "no"},
        ]
        # The kept ids are where they were: the next turn's delta carries the
        # head turn's text.
        assert conv.sent_token_ids == kept

        conv.user("Summarize it. <certainty>")
        ids = list(conv.build_prompt(adapter=AL))
        assert _token(tok, TOX) not in ids
        assert "Is the answer rude?" in tok.decode(ids)
        if policy is PRESERVE:
            assert ids[: len(kept)] == kept
        else:
            assert ids == _scratch(conv, AL)

    def test_a_head_turn_can_be_the_first_turn(self, tok, config):
        conv = Conversation(tok, policy=PRESERVE, config=config)
        conv.user("Is this rude?")
        ids = list(conv.build_prompt(adapter=TOX))
        assert ids[-1] == _token(tok, TOX)
        conv.record_answer("yes", adapter=TOX)
        assert conv.sent_token_ids == []

        conv.user("Why? <certainty>")
        ids = list(conv.build_prompt(adapter=AL))
        assert _token(tok, TOX) not in ids
        assert ids == _scratch(conv, AL)


class TestHeadOnAnAdapter:
    def test_pair_on_an_in_message_alora_appends_with_both_tokens(self, tok, config):
        """Option B: the aLoRA's token inside the message, the pair's marker last."""
        conv = Conversation(tok, policy=PRESERVE, config=config)
        conv.user("Hello.")
        conv.build_prompt()
        conv.record_answer("Hi.")
        kept = conv.sent_token_ids

        conv.user("Is it answerable? <certainty>")
        ids = list(conv.build_prompt(adapter=PAIR))
        assert ids[: len(kept)] == kept
        assert _controls(ids[len(kept) :], config) == [
            _token(tok, AL),
            _token(tok, PAIR),
        ]
        assert ids[-1] == _token(tok, PAIR)

    def test_pair_after_kept_control_tokens_renders_from_text(self, tok, config):
        conv = _after_an_alora_turn(tok, config, PRESERVE)
        conv.user("Is it answerable? <certainty>")
        ids = list(conv.build_prompt(adapter=PAIR))
        assert ids == _scratch(conv, PAIR)
        assert _controls(ids, config) == [_token(tok, AL), _token(tok, PAIR)]


class TestCompletionPayload:
    def test_a_head_turn_asks_for_one_token(self, tok, config):
        conv = Conversation(tok, policy=PRESERVE, config=config)
        conv.user("Is this rude?")
        assert conv.completion_payload(adapter=TOX)["max_tokens"] == 1
        assert conv.completion_payload(adapter=TOX, max_tokens=4)["max_tokens"] == 4

    def test_other_turns_are_unchanged(self, tok, config):
        conv = Conversation(tok, policy=PRESERVE, config=config)
        conv.user("Is it answerable? <certainty>")
        assert "max_tokens" not in conv.completion_payload(adapter=AL)
        assert "max_tokens" not in conv.completion_payload()
