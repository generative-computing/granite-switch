# SPDX-License-Identifier: Apache-2.0
"""A LoRA turn falls back to a full text render; it is never refused.

Why the fallback exists. The template emits a LoRA adapter's control token at
sequence position 0 (its LoRA prefix insertion, which also suppresses the role
marker that would follow). Position 0 is inside the already-sent prefix on every
turn after the first, so the delta can never carry it. Rather than refuse,
``build_prompt`` falls back to a full text render for that turn -- exactly what
RE_PREFILL does anyway. Nothing detects "LoRA"; the append test
(:meth:`Conversation._appendable`) simply fails because the control token is not
in the delta region, and a full render is the natural result.

Position 0 is a reliable signature only in that it is never >= len(prev): an
aLoRA adapter activates inside a user message or at the assistant boundary, both
after the opening marker, so it lands in the delta and appends.

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


class TestLoraFallsBackToFullRender:
    @pytest.mark.parametrize(
        "policy",
        [KVHistoryPolicy.RE_PREFILL, KVHistoryPolicy.PRESERVE_MIXED_HISTORY],
    )
    def test_turn_one_lora_does_not_raise(self, tok, config, policy):
        """Turn 1 is a full render under both policies, so a LoRA adapter works."""
        conv = Conversation(tok, policy=policy, config=config)
        conv.user(Q1)
        ids = conv.build_prompt(adapter=LORA)  # no raise

        assert ids
        assert ids[0] == tok.token_id(f"<|{LORA}|>"), "LoRA control token at position 0"
        assert ids.requires_token_ids is True

    def test_re_prefill_lora_matches_a_from_scratch_render(self, tok, config):
        """RE_PREFILL + LoRA sends exactly what rendering the transcript would."""
        conv = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
        conv.user(Q1)
        conv.build_prompt(adapter=LORA)
        conv.record_answer("T1566.", adapter=LORA)
        conv.user("And the tactic?")
        got = list(conv.build_prompt(adapter=LORA))

        want = list(conv._encode(conv._render(conv._messages, gen=True, adapter=LORA)))
        assert got == want

    def test_alora_still_works_under_preserve(self, tok, config):
        """Non-vacuity: aLoRA still appends and preserves under PRESERVE."""
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

    def test_base_turn_is_appendable(self, tok, config):
        """adapter=None emits no control token, so the append test passes it."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        ids = conv.build_prompt(adapter=None)

        assert ids
        assert not any(t in ids for t in config.adapter_token_ids)


class TestAppendableHelper:
    """`_appendable` decides append vs full render, from control-token position."""

    def test_control_token_in_delta_region_is_appendable(self, tok, config):
        conv = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
        prev = "<|start_of_role|>user<|end_of_role|>earlier"
        full = prev + "<|later|><|unc|>certainty"  # control string after len(prev)
        # Uses the real control texts of this checkpoint via _control_texts().
        assert conv._appendable(full, prev) is True

    def test_control_token_inside_prev_is_not_appendable(self, tok, config):
        conv = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
        control = f"<|{LORA}|>"
        full = control + "everything else"
        prev = control + "every"  # control token at index 0, inside prev
        assert conv._appendable(full, prev) is False

    def test_no_control_token_is_appendable_when_prefix_holds(self, tok, config):
        conv = Conversation(tok, policy=KVHistoryPolicy.RE_PREFILL, config=config)
        assert conv._appendable("abcdef", "abc") is True
        assert conv._appendable("xyz", "abc") is False  # not even a prefix
