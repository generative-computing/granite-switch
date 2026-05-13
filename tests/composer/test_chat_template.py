# SPDX-License-Identifier: Apache-2.0
"""Render-level tests for configure_chat_template() against the real Granite template.

Uses ``fixtures/granite_chat_template.jinja`` (copied from Granite 4.1, identical
to 4.0 at all injection anchor points) so that the injection code — regex search
for anchor patterns, ns namespace merge, Pass 1 / Pass 2 / fallback block
placement — is exercised against the real template rather than a hand-written
approximation.

``_decode_alora_invocation_text`` is patched in ``TestConfigureChatTemplate``;
those tests verify that the assembled template produces correct rendered output,
not adapter I/O.

``TestEndToEndAdapterConfigToRender`` exercises the full unpatched pipeline:
adapter_config.json → _decode_alora_invocation_text → configure_chat_template →
rendered output.  Uses minimal adapter fixtures in ``fixtures/``.

Code paths covered:
  - LoRA prefix path  (``ns.adapter_type == 'lora'``)
  - ALoRA Pass 1 + Pass 2  (invocation text found in last user message)
  - ALoRA fallback  (invocation text absent → ``ns.alora_target_idx == -1``)
  - No adapter  (``adapter_name`` undefined → no-op)
  - End-to-end: adapter_config.json → render (no patching)
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

from jinja2 import Environment

from granite_switch.composer.tokenizer_setup import configure_chat_template

_PATCH_TARGET = "granite_switch.composer.tokenizer_setup._decode_alora_invocation_text"

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
with open(os.path.join(_FIXTURES, "granite_chat_template.jinja")) as _f:
    _GRANITE_TEMPLATE = _f.read()


def _make_tokenizer():
    return SimpleNamespace(chat_template=_GRANITE_TEMPLATE)


def _render(tokenizer, **kwargs):
    return Environment().from_string(tokenizer.chat_template).render(**kwargs)


class TestConfigureChatTemplate:

    def test_lora_prefix_path(self):
        """LoRA: activation token emitted at the very start of the sequence.

        The skip-once flag set by lora_prefix_insertion suppresses the very
        next <|start_of_role|>, so the rendered output is
        '<|ctx_rel|>user<|end_of_role|>...', not
        '<|ctx_rel|><|start_of_role|>user<|end_of_role|>...'. This keeps the
        runtime embedding-swap from producing two identical consecutive
        embeddings (see tokenizer_setup.py lora_prefix_insertion comment).
        """
        tokenizer = _make_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "ctx_rel", "lora")])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
            adapter_name="ctx_rel",
        )
        assert result.startswith("<|ctx_rel|>user<|end_of_role|>"), (
            f"expected <|ctx_rel|> followed by 'user<|end_of_role|>' "
            f"(skip-once suppressed <|start_of_role|>), got {result[:80]!r}"
        )
        # Exactly one <|start_of_role|> should survive: the assistant turn.
        assert result.count("<|start_of_role|>") == 1

    def test_alora_pass1_pass2_path(self):
        """ALoRA Pass 1+2: token inserted in last user message, first char of
        invocation text dropped.

        Pass 1 finds the user message containing '<requirements>' and sets
        ns.alora_target_idx. Pass 2 splits content.val on '<requirements>'
        and rejoins with the control token followed by the invocation text
        MINUS its first character ('<' is dropped). The runtime swap
        replaces the control token's embedding with '<'s embedding, so the
        sequence tokenizes the same as '<requirements>' with no duplicate.
        The fallback block does NOT fire (alora_target_idx >= 0).
        """
        with patch(_PATCH_TARGET, return_value="<requirements>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer, [("/path/a", "req_check", "alora")]
            )

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "<requirements>req1\nreq2"}],
            add_generation_prompt=True,
            adapter_name="req_check",
        )
        # Token immediately precedes the invocation text (minus first char)
        # inside the user turn: "<|req_check|>requirements>" (no '<').
        user_turn_header = "<|start_of_role|>user<|end_of_role|>"
        assert user_turn_header + "<|req_check|>requirements>" in result
        # And the literal "<|req_check|><requirements>" should NOT appear —
        # the leading '<' must have been dropped.
        assert "<|req_check|><requirements>" not in result
        # Fallback did not fire: token is not immediately before generation prompt
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|req_check|>"):last_gen_pos] != "<|req_check|>"

    def test_alora_fallback_path(self):
        """ALoRA fallback: token emitted before generation prompt when invocation text is absent.

        Pass 1 scans all user messages and finds none containing the decoded invocation
        text (here the assistant role token sequence), so ns.alora_target_idx stays -1
        and the fallback block fires.
        """
        with patch(_PATCH_TARGET, return_value="<|start_of_role|>assistant<|end_of_role|>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer, [("/path/a", "answerability", "alora")]
            )

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
            adapter_name="answerability",
        )
        assert "<|answerability|>" in result
        # Token appears immediately before what would have been the generation
        # prompt's <|start_of_role|>. The skip-once flag set by alora_insertion
        # suppresses that <|start_of_role|>, so the rendered output has
        # "<|answerability|>assistant<|end_of_role|>" — no role marker between
        # the control token and the role name. Prevents a duplicate-embedding
        # OOD at position 1 after the runtime swap (see tokenizer_setup.py
        # alora_insertion comment).
        token = "<|answerability|>"
        token_pos = result.index(token)
        after = result[token_pos + len(token):]
        assert after.startswith("assistant<|end_of_role|>"), (
            f"expected 'assistant<|end_of_role|>' immediately after "
            f"{token!r}, got {after[:60]!r}"
        )
        # Only one <|start_of_role|> should survive: the one before the user turn.
        assert result.count("<|start_of_role|>") == 1

    def test_alora_pass1_pass2_iterable_content(self):
        """ALoRA Pass 1+2: token inserted correctly when message content is a list of parts.

        When content is iterable (multi-part), Pass 1 must record the *message* index
        (outer loop), not the entry index (inner loop).  A previous bug used the inner
        loop.index0, causing the wrong message to be targeted in Pass 2 and a
        subsequent crash on _parts[1] when rsplit found no separator.
        """
        with patch(_PATCH_TARGET, return_value="<requirements>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer, [("/path/a", "req_check", "alora")]
            )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this: <requirements>req1\nreq2"},
                ],
            },
        ]
        result = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="req_check",
        )
        # Token appears before the invocation text, and the invocation
        # text's first character ('<') has been dropped.
        assert "<|req_check|>requirements>" in result
        assert "<|req_check|><requirements>" not in result
        assert result.index("<|req_check|>") > result.index("<|start_of_role|>user<|end_of_role|>")
        # Fallback must NOT also fire
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|req_check|>"):last_gen_pos] != "<|req_check|>"

    def test_skip_once_is_single_shot(self):
        """Skip-once flag consumes itself: only the first <|start_of_role|>
        after a LoRA control token is suppressed; later role markers emit."""
        tokenizer = _make_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "my_lora", "lora")])

        # Two user turns so the template emits <|start_of_role|> three times:
        # once per user turn + once for the generation prompt. Only the very
        # first one should be suppressed.
        result = _render(
            tokenizer,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
            add_generation_prompt=True,
            adapter_name="my_lora",
        )
        assert result.startswith("<|my_lora|>user<|end_of_role|>"), (
            f"first <|start_of_role|> should be suppressed; got {result[:80]!r}"
        )
        # Four role markers would be emitted normally (first user, assistant,
        # second user, assistant-generation-prompt). Skip-once removes the
        # first → exactly three survive.
        assert result.count("<|start_of_role|>") == 3

    def test_no_adapter_no_tokens(self):
        """Without adapter_name the rendered output is identical to the original template."""
        messages = [{"role": "user", "content": "Hello"}]
        original = _render(_make_tokenizer(), messages=messages, add_generation_prompt=True)

        with patch(_PATCH_TARGET, return_value="<requirements>"):
            tokenizer = _make_tokenizer()
            configure_chat_template(
                tokenizer,
                [("/path/a", "ctx_rel", "lora"), ("/path/b", "req_check", "alora")],
            )
        modified = _render(tokenizer, messages=messages, add_generation_prompt=True)

        assert modified == original


class TestInvocationFirstCharDropProperty:
    """Standalone property test on a real Granite tokenizer: dropping the first
    character of an ALoRA invocation text yields the same tail-token sequence
    as tokenizing the full invocation text and dropping its first token. This
    is the BPE-level invariant the Pass-2 edit relies on — if a future
    tokenizer change breaks it, the template-level drop would silently corrupt
    the tail of the invocation.
    """

    _INVOCATIONS = [
        "<requirements>",
        "<certainty>",
        "<guardian>",
        "<context>",
    ]

    def _get_tokenizer(self):
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-3b")
        except Exception as e:
            import pytest
            pytest.skip(f"could not fetch Granite tokenizer: {e}")

    def test_first_char_drop_equals_first_token_drop(self):
        tok = self._get_tokenizer()
        for invocation in self._INVOCATIONS:
            full_ids = tok(invocation, add_special_tokens=False).input_ids
            dropped_ids = tok(invocation[1:], add_special_tokens=False).input_ids
            assert full_ids[1:] == dropped_ids, (
                f"invocation {invocation!r}: dropping first char of the "
                f"string produced tokens {dropped_ids} but the tail of the "
                f"full tokenization is {full_ids[1:]}"
            )

    def test_first_token_is_single_character(self):
        """Sanity: the first token of each invocation must be exactly one
        character (the leading '<'). Otherwise dropping invocation_text[1:]
        in Jinja would drop the wrong number of characters."""
        tok = self._get_tokenizer()
        for invocation in self._INVOCATIONS:
            first_id = tok(invocation, add_special_tokens=False).input_ids[0]
            first_str = tok.decode([first_id])
            assert first_str == invocation[0], (
                f"invocation {invocation!r}: first token decodes to "
                f"{first_str!r}, expected {invocation[0]!r}"
            )


class _FixtureTokenizer:
    """Tokenizer with a decode map for fixture adapter token IDs."""

    def __init__(self, chat_template, decode_map):
        self.chat_template = chat_template
        self._decode_map = decode_map

    def decode(self, token_ids, skip_special_tokens=False):
        return self._decode_map[tuple(token_ids)]


class TestEndToEndAdapterConfigToRender:
    """End-to-end: adapter_config.json → _decode_alora_invocation_text →
    configure_chat_template → rendered output.  No patching."""

    # Fixture adapter paths
    _ANSWERABILITY = os.path.join(_FIXTURES, "answerability_adapter")
    _CONTEXT_REL = os.path.join(_FIXTURES, "context_relevance_adapter")
    _SUMMARIZATION = os.path.join(_FIXTURES, "summarization_adapter")

    @staticmethod
    def _make_tokenizer(decode_map):
        return _FixtureTokenizer(_GRANITE_TEMPLATE, decode_map)

    def test_alora_fallback_from_adapter_config(self):
        """ALoRA adapter whose invocation tokens decode to the assistant role
        sequence → fallback path (token before generation prompt)."""
        tokenizer = self._make_tokenizer({
            # [100264, 78191, 100265] → assistant role sequence
            (100264, 78191, 100265): "<|start_of_role|>assistant<|end_of_role|>",
        })
        configure_chat_template(tokenizer, [
            (self._ANSWERABILITY, "answerability", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Is this answerable?"}],
            add_generation_prompt=True,
            adapter_name="answerability",
        )
        # Fallback: token immediately before generation prompt, with the
        # generation-prompt <|start_of_role|> suppressed by the skip-once flag
        # armed in alora_insertion. Output is "<|answerability|>assistant<|end_of_role|>".
        token = "<|answerability|>"
        assert token in result
        token_pos = result.index(token)
        after = result[token_pos + len(token):]
        assert after.startswith("assistant<|end_of_role|>"), (
            f"expected 'assistant<|end_of_role|>' immediately after "
            f"{token!r}, got {after[:60]!r}"
        )
        # Only the user-turn <|start_of_role|> should survive.
        assert result.count("<|start_of_role|>") == 1

    def test_alora_invocation_at_start_of_user_message(self):
        """ALoRA: invocation text is the first thing in the user message.

        Pass 2 drops the first character of the invocation text after
        inserting the control token, so "<context>" becomes
        "<|context_relevance|>context>" in the rendered output.
        """
        tokenizer = self._make_tokenizer({(27,): "<context>"})
        configure_chat_template(tokenizer, [
            (self._CONTEXT_REL, "context_relevance", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "<context>some documents</context>"}],
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        # Token injected right after the user role header; the '<' of
        # the invocation text is dropped.
        user_header = "<|start_of_role|>user<|end_of_role|>"
        assert user_header + "<|context_relevance|>context>" in result
        assert "<|context_relevance|><context>" not in result
        # Fallback must NOT fire
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|context_relevance|>"):last_gen_pos] != "<|context_relevance|>"

    def test_alora_invocation_mid_user_message(self):
        """ALoRA: invocation text appears in the middle of the user message.

        Same first-character drop as the start-of-message case.
        """
        tokenizer = self._make_tokenizer({(27,): "<context>"})
        configure_chat_template(tokenizer, [
            (self._CONTEXT_REL, "context_relevance", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Please review: <context>docs</context>"}],
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        # Token injected mid-message, invocation text's '<' dropped.
        assert "Please review: <|context_relevance|>context>" in result
        assert "<|context_relevance|><context>" not in result
        user_header = "<|start_of_role|>user<|end_of_role|>"
        assert result.index("<|context_relevance|>") > result.index(user_header)
        # Fallback must NOT fire
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert result[last_gen_pos - len("<|context_relevance|>"):last_gen_pos] != "<|context_relevance|>"

    def test_alora_multiple_occurrences_targets_last(self):
        """ALoRA: invocation text appears twice — token injected before the last occurrence.

        rsplit(..., 1) splits on the last occurrence, so the control token must
        land before the second <context>, not the first. First occurrence
        remains intact with its '<'; only the second has its '<' dropped.
        """
        tokenizer = self._make_tokenizer({(27,): "<context>"})
        configure_chat_template(tokenizer, [
            (self._CONTEXT_REL, "context_relevance", "alora"),
        ])

        result = _render(
            tokenizer,
            messages=[{
                "role": "user",
                "content": "<context>first batch</context> Also check <context>second batch</context>",
            }],
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        # First <context> untouched; second one has the control token
        # inserted with its '<' dropped.
        assert "<context>first batch</context> Also check <|context_relevance|>context>second batch" in result
        # Only one control token in the entire output
        assert result.count("<|context_relevance|>") == 1

    def test_lora_prefix_from_adapter_config(self):
        """LoRA adapter (no alora_invocation_tokens) → prefix path."""
        tokenizer = self._make_tokenizer({})  # no decode needed for LoRA
        configure_chat_template(tokenizer, [
            (self._SUMMARIZATION, "summarization", "lora"),
        ])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Summarize this."}],
            add_generation_prompt=True,
            adapter_name="summarization",
        )
        assert result.startswith("<|summarization|>")
        # Skip-once suppresses the user-turn <|start_of_role|>: output is
        # "<|summarization|>user<|end_of_role|>...", not
        # "<|summarization|><|start_of_role|>user...". Keeps the adapter
        # substitute token from duplicating at runtime.
        assert result.startswith("<|summarization|>user<|end_of_role|>")

    def test_mixed_adapters_from_adapter_config(self):
        """All three adapter types composed together, each activated independently."""
        tokenizer = self._make_tokenizer({
            (100264, 78191, 100265): "<|start_of_role|>assistant<|end_of_role|>",
            (27,): "<context>",
        })
        configure_chat_template(tokenizer, [
            (self._ANSWERABILITY, "answerability", "alora"),
            (self._CONTEXT_REL, "context_relevance", "alora"),
            (self._SUMMARIZATION, "summarization", "lora"),
        ])

        messages = [{"role": "user", "content": "<context>docs</context>"}]

        # Activate context_relevance → Pass 1+2 (drops first char of invocation).
        result = _render(
            tokenizer, messages=messages,
            add_generation_prompt=True, adapter_name="context_relevance",
        )
        assert "<|context_relevance|>context>" in result
        assert "<|context_relevance|><context>" not in result

        # Activate answerability → fallback (skip-once suppresses the
        # generation-prompt <|start_of_role|>).
        result = _render(
            tokenizer, messages=messages,
            add_generation_prompt=True, adapter_name="answerability",
        )
        token = "<|answerability|>"
        token_pos = result.index(token)
        after = result[token_pos + len(token):]
        assert after.startswith("assistant<|end_of_role|>")

        # Activate summarization → prefix
        result = _render(
            tokenizer, messages=messages,
            add_generation_prompt=True, adapter_name="summarization",
        )
        assert result.startswith("<|summarization|>")

        # No adapter → no tokens
        result_none = _render(
            tokenizer, messages=messages, add_generation_prompt=True,
        )
        assert "<|answerability|>" not in result_none
        assert "<|context_relevance|>" not in result_none
        assert "<|summarization|>" not in result_none
