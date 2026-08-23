# SPDX-License-Identifier: Apache-2.0
"""A model can name an adapter in its own answer; that must be visible.

Control tokens are freely generatable -- there is no runtime suppression -- so a
model may emit one mid-answer and re-route the rest of its own generation. Under
PRESERVE the token then persists into every later turn and keeps re-routing that
region of history.

Deliberately not prevented. But it must not be invisible, and by default it is:
detokenization strips special tokens, so the returned text shows nothing and the
only other evidence is a routing trace. ``generated_control_tokens`` reports it.

The three-state result is the point. ``[]`` is "checked, none"; ``[id]`` is
"emitted these"; ``None`` is "not checkable, the answer arrived as text" -- by
which point the token is already gone and no re-encoding can recover it.
"""

import pytest

from granite_switch import Conversation, KVHistoryPolicy
from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

A_NAME, B_NAME = "unc", "req"
ADAPTERS = [
    (A_NAME, "alora", "<certainty>"),
    (B_NAME, "alora", "<|start_of_role|>assistant<|end_of_role|>"),
]
Q1 = "Is this answerable? <certainty>"

BOTH = [KVHistoryPolicy.RE_PREFILL, KVHistoryPolicy.PRESERVE_MIXED_HISTORY]


@pytest.fixture
def tok():
    return make_stub_tokenizer(ADAPTERS)


@pytest.fixture
def config(tok):
    return StubConfig([tok.token_id(f"<|{A_NAME}|>"), tok.token_id(f"<|{B_NAME}|>")])


def _turn(conv, adapter, answer):
    conv.user(Q1)
    conv.build_prompt(adapter=adapter)
    return conv.record_answer(answer, adapter=adapter)


class TestReporting:
    @pytest.mark.parametrize("policy", BOTH)
    def test_clean_answer_reports_an_empty_list(self, tok, config, policy):
        """[] is a positive statement: checked, and the model emitted none."""
        conv = Conversation(tok, policy=policy, config=config)
        clean = tok("Yes, with high confidence.")["input_ids"]
        _turn(conv, A_NAME, clean)

        assert conv.generated_control_tokens == [[]]

    @pytest.mark.parametrize("policy", BOTH)
    def test_self_named_adapter_is_reported(self, tok, config, policy):
        conv = Conversation(tok, policy=policy, config=config)
        stray = tok.token_id(f"<|{B_NAME}|>")
        answer = tok("Yes")["input_ids"] + [stray] + tok(" indeed.")["input_ids"]
        _turn(conv, A_NAME, answer)

        assert conv.generated_control_tokens == [[stray]]

    @pytest.mark.parametrize("policy", BOTH)
    def test_text_answer_reports_not_checkable(self, tok, config, policy):
        """None, not [] -- the difference between "none" and "cannot tell".

        Reporting [] here would be a false negative: by the time an answer is
        text, detokenization has already removed any control token, so a
        re-encoding cannot find one and claiming none were present is a guess.
        """
        conv = Conversation(tok, policy=policy, config=config)
        _turn(conv, A_NAME, "Yes, with high confidence.")

        assert conv.generated_control_tokens == [None]

    def test_one_entry_per_answer_in_order(self, tok, config):
        """Entries stay aligned with the turns, so a report can name the turn."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        stray = tok.token_id(f"<|{B_NAME}|>")
        _turn(conv, A_NAME, tok("clean")["input_ids"])
        _turn(conv, A_NAME, tok("dirty")["input_ids"] + [stray])
        _turn(conv, A_NAME, "text answer")

        assert conv.generated_control_tokens == [[], [stray], None]

    def test_a_discarded_prompt_adds_no_entry(self, tok, config):
        """Only recorded answers count, matching how the transcript advances."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        conv.build_prompt(adapter=A_NAME)  # a judge call, answer discarded

        assert conv.generated_control_tokens == []


class TestConsequenceItReports:
    """The report matters because the two policies then diverge."""

    def test_preserve_keeps_the_self_named_token_and_re_prefill_drops_it(
        self, tok, config
    ):
        """Same answer, two policies, two different histories from here on.

        This is what a non-empty report is warning about: the object's two records
        no longer describe the same token stream, so the policies are no longer
        interchangeable for this conversation.
        """
        stray = tok.token_id(f"<|{B_NAME}|>")
        answer = tok("Yes")["input_ids"] + [stray]

        preserve = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        _turn(preserve, A_NAME, answer)
        reprefill = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
        _turn(reprefill, A_NAME, answer)

        # Both saw it...
        assert preserve.generated_control_tokens == [[stray]]
        assert reprefill.generated_control_tokens == [[stray]]
        # ...but only one carries it forward.
        assert stray in preserve.sent_token_ids
        assert reprefill.sent_token_ids == []
        # And neither keeps it in the text record, which is why the report exists.
        assert not any(f"<|{B_NAME}|>" in m["content"] for m in preserve.messages)
