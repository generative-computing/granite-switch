# SPDX-License-Identifier: Apache-2.0
"""PRESERVE_MIXED_HISTORY is only correct if the prompt is sent as token ids.

That requirement lives outside ``Conversation``: the class builds ids and never
learns how they travel. Posting ``messages`` to ``/v1/chat/completions`` instead
makes the server re-render and re-tokenize, which drops earlier turns' control
tokens -- reverting to RE_PREFILL with no exception, no changed return value, and
no symptom a caller can see except a prefix-cache hit rate they were not
measuring.

So the class cannot make the mistake impossible; it can only make the obvious
route correct and the wrong route loud. These tests pin both halves:

  * ``build_prompt`` returns ``PromptTokenIds``, a ``list`` subclass carrying
    ``requires_token_ids`` so a transport layer can check rather than assume;
  * ``completion_payload`` is the ready-to-post body for either policy;
  * ``chat_payload`` REFUSES under PRESERVE rather than returning a body that
    would silently behave as the other policy.

CPU-only; stub tokenizer over the real Granite fixture template.
"""

import pytest

from granite_switch import Conversation, KVHistoryPolicy
from granite_switch.conversation import PromptTokenIds
from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

A_NAME, B_NAME = "unc", "req"
ADAPTERS = [
    (A_NAME, "alora", "<certainty>"),
    (B_NAME, "alora", "<|start_of_role|>assistant<|end_of_role|>"),
]
Q1 = "Is this answerable from the context? <certainty>"
ANSWER_1 = "Yes, with high confidence."
Q2 = "Now summarize it."

BOTH_POLICIES = [KVHistoryPolicy.RE_PREFILL, KVHistoryPolicy.PRESERVE_MIXED_HISTORY]


@pytest.fixture
def tok():
    return make_stub_tokenizer(ADAPTERS)


@pytest.fixture
def config(tok):
    return StubConfig([tok.token_id(f"<|{A_NAME}|>"), tok.token_id(f"<|{B_NAME}|>")])


def _conv(tok, config, policy, turns=1):
    """A conversation with ``turns`` completed turns, ready for the next one."""
    conv = Conversation(tok, policy=policy, config=config)
    conv.user(Q1)
    if turns >= 1:
        conv.build_prompt(adapter=A_NAME)
        conv.record_answer(ANSWER_1, adapter=A_NAME)
        conv.user(Q2)
    return conv


class TestPromptTokenIds:
    """The return type must stay a list, and say whether ids are mandatory."""

    @pytest.mark.parametrize("policy", BOTH_POLICIES)
    def test_behaves_as_a_plain_list(self, tok, config, policy):
        """Existing callers index, slice, compare and concatenate the result.

        A wrapper that broke any of those would be a silent break for every
        caller, which is worse than the problem it addresses.
        """
        conv = Conversation(tok, policy=policy, config=config)
        conv.user(Q1)
        ids = conv.build_prompt(adapter=A_NAME)

        assert isinstance(ids, PromptTokenIds)
        assert isinstance(ids, list)
        assert ids == list(ids)
        assert ids[:3] == list(ids)[:3]
        # `+` (not [*ids, 0]) on purpose: this asserts list.__add__ survives
        # subclassing. Unpacking would exercise iteration instead.
        assert len(ids + [0]) == len(ids) + 1  # noqa: RUF005
        assert all(isinstance(t, int) for t in ids)

    def test_requires_token_ids_only_under_preserve(self, tok, config):
        """The flag is the whole point: it distinguishes must-send-ids from may."""
        preserve = _conv(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY, turns=0)
        reprefill = _conv(tok, config, KVHistoryPolicy.RE_PREFILL, turns=0)

        assert preserve.build_prompt(adapter=A_NAME).requires_token_ids is True
        assert reprefill.build_prompt(adapter=A_NAME).requires_token_ids is False


class TestCompletionPayload:
    """The correct route, for both policies."""

    @pytest.mark.parametrize("policy", BOTH_POLICIES)
    def test_carries_ids_and_merges_extras(self, tok, config, policy):
        conv = _conv(tok, config, policy)
        body = conv.completion_payload(adapter=B_NAME, max_tokens=24, temperature=0.0)

        assert body["max_tokens"] == 24 and body["temperature"] == 0.0
        assert isinstance(body["prompt"], list)
        assert all(isinstance(t, int) for t in body["prompt"])
        assert "messages" not in body, (
            "a completions body must not carry messages; the server would have to "
            "choose one, and which it picks is not something this class controls"
        )

    def test_preserve_payload_extends_what_was_already_sent(self, tok, config):
        """The prompt must begin with the ids the model has already seen.

        This is the property the whole policy rests on: if the body did not start
        with the previous prompt, the cached blocks could not match it and PRESERVE
        would cost more than RE_PREFILL while claiming to cost less.
        """
        conv = _conv(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        sent = conv.sent_token_ids
        body = conv.completion_payload(adapter=B_NAME)

        assert sent, "turn 1 should have left a transcript"
        assert body["prompt"][: len(sent)] == sent


class TestChatPayloadRefusesUnderPreserve:
    """The wrong route must fail loudly instead of degrading quietly."""

    def test_raises_and_says_what_to_do(self, tok, config):
        conv = _conv(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)

        with pytest.raises(RuntimeError) as exc:
            conv.chat_payload(adapter=B_NAME)

        message = str(exc.value)
        assert "chat/completions" in message
        assert "completion_payload" in message, (
            "the error must name the call to use instead; an error that only says "
            "no leaves the caller to guess, and the guess is usually the bug"
        )
        assert "RE_PREFILL" in message

    def test_re_prefill_is_allowed_and_carries_the_adapter(self, tok, config):
        """RE_PREFILL is exactly what the chat endpoint does, so permit it."""
        conv = _conv(tok, config, KVHistoryPolicy.RE_PREFILL)
        body = conv.chat_payload(adapter=B_NAME, max_tokens=24)

        assert body["chat_template_kwargs"]["adapter_name"] == B_NAME
        assert body["max_tokens"] == 24
        assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]

    def test_messages_view_is_why_chat_payload_refuses(self, tok, config):
        """Concretely: the text view has lost turn 1's control token.

        Sending it would be sending a different conversation, not a slower render
        of the same one -- which is what makes the refusal a correctness matter
        rather than a performance one.
        """
        conv = _conv(tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY)
        control = f"<|{A_NAME}|>"
        control_id = tok.token_id(control)

        assert not any(control in m["content"] for m in conv.messages)
        assert control_id in conv.sent_token_ids, (
            "the ids kept it while the text view dropped it; that divergence is "
            "the reason the two transports are not interchangeable"
        )
