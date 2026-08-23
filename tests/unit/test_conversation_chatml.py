# SPDX-License-Identifier: Apache-2.0
"""``Conversation`` against the ChatML (Granite 4.2) chat template.

Every other ``Conversation`` test uses the ``<|start_of_role|>`` role-marker
fixture, so this file is the only coverage of the 4.2 family. It exists because
the ChatML stub existed and nothing used it: ``chatml=True`` appeared nowhere in
``tests/``, and ``PRESERVE_MIXED_HISTORY`` was broken on 4.2 as a result.

Two template flags decide whether ``PRESERVE_MIXED_HISTORY`` is possible at all,
measured against the fixture with ``transformers``' own Jinja settings:

    | enable_thinking | truncate_history_thinking | terminator | append-only |
    |-----------------|---------------------------|------------|-------------|
    | True            | True                      | no         | no          |
    | True            | False                     | no         | yes         |
    | False           | True                      | yes        | no          |
    | False           | False                     | yes        | yes         |

Only the last row can preserve history today, so it is the configuration these
tests pin. The other three must be refused rather than mis-served; the errors are
the module's existing generic ones, and ``docs/SUPPORTED_MODELS.md`` carries the
remedy (both flags off).

The two "terminator: no" rows are limited by how ``_turn_end`` probes -- with
empty assistant content -- rather than by the template itself; see
``TestThinkingOnIsRefused``. The two "append-only: no" rows are the template
rewriting turns it already emitted, which no probe change can repair.
"""

import pytest

from granite_switch import Conversation, KVHistoryPolicy
from tests.shared.conversation_stubs import StubConfig, make_stub_tokenizer

NAME = "uncertainty"
# aLoRA with user-message invocation text: its control token lands in the NEWEST
# user message, which is the placement PRESERVE needs (a LoRA token goes to index
# 0, inside the already-sent region, and is rejected for that reason elsewhere).
ADAPTERS = [(NAME, "alora", "<certainty>")]

Q1 = "Is this answerable from the context? <certainty>"
ANSWER_1 = "Yes, with high confidence."
Q2 = "Now summarize it. <certainty>"

# The only configuration in which 4.2 can preserve history. Thinking off makes the
# assistant turn terminator derivable; truncation off keeps the render append-only.
NO_THINK = {"enable_thinking": False, "truncate_history_thinking": False}


@pytest.fixture
def tok():
    return make_stub_tokenizer(ADAPTERS, chatml=True)


@pytest.fixture
def config(tok):
    return StubConfig([tok.token_id(f"<|{NAME}|>")])


def _two_turns(tok, config, policy, **template_kwargs):
    conv = Conversation(tok, policy=policy, config=config)
    conv.user(Q1)
    p1 = conv.build_prompt(adapter=NAME, **template_kwargs)
    conv.record_answer(ANSWER_1, adapter=NAME)
    conv.user(Q2)
    p2 = conv.build_prompt(adapter=NAME, **template_kwargs)
    return conv, p1, p2


class TestStubFidelity:
    """The stub must render like ``transformers``, or these tests prove nothing.

    ``transformers`` builds its Jinja environment with ``trim_blocks=True,
    lstrip_blocks=True`` (``utils/chat_template_utils.py:489``). Without those the
    ChatML fixture leaks its own block indentation into the output -- 193 chars
    instead of 170 for a three-message render -- and the leaked whitespace lands
    between the history and the assistant marker, which is exactly where the
    terminator is derived and the delta is cut.
    """

    def test_chatml_render_has_no_leaked_block_indentation(self, tok):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            add_generation_prompt=False,
            tokenize=False,
        )
        indented = [ln for ln in rendered.split("\n") if ln.startswith(" ")]
        assert not indented, (
            "the stub is leaking Jinja block indentation that real transformers "
            f"trims, so every byte offset measured here is fiction: {indented!r}"
        )


class TestPreserveWithThinkingOff:
    """The supported configuration: both flags off."""

    def test_preserve_carries_the_earlier_control_token(self, tok, config):
        _conv, _p1, p2 = _two_turns(
            tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY, **NO_THINK
        )
        assert p2.count(tok.token_id(f"<|{NAME}|>")) == 2, (
            "both turns' control tokens must be present under PRESERVE; one means "
            "the earlier region routes to base and the policy is a no-op"
        )

    def test_re_prefill_keeps_only_the_current_one(self, tok, config):
        """The twin, so a bug making both policies identical cannot pass."""
        _conv, _p1, p2 = _two_turns(tok, config, KVHistoryPolicy.RE_PREFILL, **NO_THINK)
        assert p2.count(tok.token_id(f"<|{NAME}|>")) == 1

    def test_preserve_extends_the_previous_prompt(self, tok, config):
        conv, p1, p2 = _two_turns(
            tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY, **NO_THINK
        )
        assert p2[: len(p1)] == p1
        assert p2[: len(conv.sent_token_ids)] == conv.sent_token_ids
        assert len(p2) > len(conv.sent_token_ids), "the new turn added nothing"

    def test_delta_begins_on_a_special_token(self, tok, config):
        """ChatML opens a turn with ``<|im_start|>``, so the seam is merge-proof."""
        conv, _p1, p2 = _two_turns(
            tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY, **NO_THINK
        )
        assert tok.decode(p2[len(conv.sent_token_ids) :]).startswith("<|im_start|>")

    def test_join_does_not_shift_ids(self, tok, config):
        _conv, _p1, p2 = _two_turns(
            tok, config, KVHistoryPolicy.PRESERVE_MIXED_HISTORY, **NO_THINK
        )
        assert tok(tok.decode(p2))["input_ids"] == p2

    def test_turn_end_is_the_chatml_terminator(self, tok, config):
        """Derived, not hardcoded -- and it must be ChatML's, not 4.1's."""
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        conv.build_prompt(adapter=NAME, **NO_THINK)
        assert conv._turn_end() == "<|im_end|>\n"


class TestThinkingOnIsRefused:
    """Thinking on is refused today, at the terminator derivation.

    The refusal is a property of the probe, not of the template. ``_turn_end``
    renders an EMPTY assistant turn, and the template prepends ``<think></think>``
    to content with no think tags, so it never matches the generation prompt's open
    ``<think>``. Measured against this fixture, a real answer carrying its own
    opening ``<think>`` does extend the sent prompt, and the render is append-only
    with thinking on as long as ``truncate_history_thinking`` is off.

    So these tests pin CURRENT behaviour, not a permanent limit. Supporting
    thinking on means probing with a sentinel instead of empty content and having
    ``record_answer`` restore the opening ``<think>`` the model never emits. When
    that lands, these assertions change rather than disappear -- what must not
    change is that an unsupported configuration raises instead of mis-serving.
    """

    def test_thinking_on_is_refused_at_the_terminator(self, tok, config):
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        conv.build_prompt(adapter=NAME, enable_thinking=True)
        with pytest.raises(RuntimeError, match="cannot derive the assistant turn"):
            conv.record_answer(ANSWER_1, adapter=NAME)

    def test_re_prefill_is_unaffected_by_thinking(self, tok, config):
        """The twin: RE_PREFILL never derives a terminator, so it must still work."""
        _conv, _p1, p2 = _two_turns(
            tok, config, KVHistoryPolicy.RE_PREFILL, enable_thinking=True
        )
        assert tok.token_id(f"<|{NAME}|>") in p2


class TestHistoryTruncationIsRefused:
    """Truncation on rewrites earlier assistant turns, so the prefix breaks.

    The bytes of an already-sent turn change when a newer user turn arrives:
    ``<think>\\nreasoning\\n</think>\\na1`` becomes ``<think></think>\\na1``. The
    generic "not append-only" message would send the reader to the template; the
    cause is one flag, so the error has to name it.
    """

    def test_truncation_is_refused(self, tok, config):
        kwargs = {"enable_thinking": False, "truncate_history_thinking": True}
        conv = Conversation(
            tok, policy=KVHistoryPolicy.PRESERVE_MIXED_HISTORY, config=config
        )
        conv.user(Q1)
        conv.build_prompt(adapter=NAME, **kwargs)
        conv.record_answer("<think>\nreasoning\n</think>\nYes.", adapter=NAME)
        conv.user(Q2)
        with pytest.raises(RuntimeError, match="not append-only"):
            conv.build_prompt(adapter=NAME, **kwargs)
