# SPDX-License-Identifier: Apache-2.0
"""LoRA adapters always use RE_PREFILL; PRESERVE_MIXED_HISTORY is aLoRA-only.

Why the rule exists. The template emits a LoRA adapter's control token at
sequence position 0 (its LoRA prefix insertion, which also suppresses the role
marker that would follow). Position 0 is inside the already-sent prefix on every
turn after the first, so the delta can never be derived and the policy cannot
hold.

Why the refusal is up front rather than at the point of failure. Turn 1 takes the
full-render path, where there is no prefix to preserve, so a LoRA adapter WOULD
work there -- and then fail on turn 2, and on every turn after it. It is refused
anyway, because a conversation does not change adapter technology mid-dialogue: a
LoRA turn 1 is a LoRA turn 2, so "it works on turn 1" is not a case anyone can
actually use. Succeeding once and failing forever is also worse than refusing
immediately, because by turn 2 a caller has a transcript that cannot continue
under the policy it chose.

Position 0 is a reliable signature: an aLoRA adapter activates either inside a
user message (Pass 2) or at the assistant boundary (the fallback), and both sit
after the opening role marker.

CPU-only; stub tokenizer over the real Granite fixture template.
"""

import pytest

from granite_switch import Conversation, KVHistoryPolicy
from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

LORA, ALORA = "ctx", "unc"
ADAPTERS = [(LORA, "lora", None), (ALORA, "alora", "<certainty>")]

Q1 = "Map this report to MITRE techniques."
Q2 = "Is that answerable? <certainty>"


@pytest.fixture
def tok():
    return make_stub_tokenizer(ADAPTERS)


@pytest.fixture
def config(tok):
    return StubConfig([tok.token_id(f"<|{LORA}|>"), tok.token_id(f"<|{ALORA}|>")])


class TestLoraIsRePrefillOnly:
    def test_refused_on_the_very_first_turn(self, tok, config):
        """Refuse immediately, not after a turn has already been committed."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)

        with pytest.raises(RuntimeError) as exc:
            conv.build_prompt(adapter=LORA)

        message = str(exc.value)
        assert "LoRA" in message
        assert "position 0" in message
        assert "RE_PREFILL" in message, (
            "the error must name the policy to use instead; the caller's next "
            "action is to switch policy, so the message should say so"
        )
        assert LORA in message, "name the adapter, so the caller knows which one"

    def test_transcript_is_untouched_by_the_refusal(self, tok, config):
        """A refused build_prompt must not leave half a turn behind.

        Otherwise a caller that catches the error and switches policy would carry
        a transcript that no longer matches anything the model was sent.
        """
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        with pytest.raises(RuntimeError):
            conv.build_prompt(adapter=LORA)

        assert conv.sent_token_ids == []
        with pytest.raises(RuntimeError):
            conv.record_answer("T1566.", adapter=LORA)

    def test_same_adapter_is_fine_under_re_prefill(self, tok, config):
        """The rule is about the policy, not about the adapter being unusable."""
        conv = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
        conv.user(Q1)
        ids = conv.build_prompt(adapter=LORA)

        assert ids, "a LoRA adapter under RE_PREFILL is an ordinary request"
        assert tok.token_id(f"<|{LORA}|>") in ids
        assert ids.requires_token_ids is False

    def test_alora_still_works_under_preserve(self, tok, config):
        """Non-vacuity: the guard must not reject the case the policy is for."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q2)
        p1 = conv.build_prompt(adapter=ALORA)
        conv.record_answer("Yes.", adapter=ALORA)
        conv.user(Q2)
        p2 = conv.build_prompt(adapter=ALORA)

        assert p2[: len(p1)] == list(p1), "PRESERVE must still preserve for aLoRA"
        assert tok.token_id(f"<|{ALORA}|>") in p1

    def test_base_turns_are_not_mistaken_for_lora(self, tok, config):
        """adapter=None emits no control token, so it cannot be at position 0."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        ids = conv.build_prompt(adapter=None)

        assert ids
        assert not any(t in ids for t in config.adapter_token_ids)
