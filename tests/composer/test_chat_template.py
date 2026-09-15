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

import contextlib
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from jinja2 import Environment

from granite_switch.composer.tokenizer_setup import (
    build_substitute_token_ids,
    configure_audio_chat_template,
    configure_chat_template,
    detect_template_format,
)

_PATCH_TARGET = "granite_switch.composer.tokenizer_setup._decode_alora_invocation_text"
_IDS_TARGET = "granite_switch.composer.tokenizer_setup._load_alora_invocation_token_ids"


def _fake_alora(invocation_text, tokenizer, trained_ids=None):
    """Patch both halves of an adapter's invocation metadata at once.

    ``configure_chat_template`` reads an aLoRA's invocation twice and for two
    different purposes: the decoded TEXT (what Pass 1 matches on) and the recorded
    TOKEN IDS (what the substitute and the Pass-2 tail come from). A test that
    patches only the first leaves the second reading a nonexistent
    adapter_config.json.

    ``trained_ids`` defaults to *tokenizer*'s own split of the text, which is the
    honest default: for these stubs the trainer's tokenizer and the local one are
    the same object. Pass it explicitly only to model the mismatch between them --
    see ``TestMergedFirstTokenRenderEndToEnd``.
    """
    if trained_ids is None:
        trained_ids = tokenizer(invocation_text)["input_ids"]
    return (
        patch(_PATCH_TARGET, return_value=invocation_text),
        patch(_IDS_TARGET, return_value=trained_ids),
    )


_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
with open(os.path.join(_FIXTURES, "granite_chat_template.jinja")) as _f:
    _GRANITE_TEMPLATE = _f.read()
with open(os.path.join(_FIXTURES, "granite_chatml_template.jinja")) as _f:
    _CHATML_TEMPLATE = _f.read()


class _StubTokenizer:
    """Tokenizer stub that can actually TOKENIZE, not just hold a template.

    ``configure_chat_template`` computes each aLoRA's Pass-2 tail with
    :func:`alora_invocation_tail`, which needs the first *token* of the
    invocation text. A stub exposing only ``chat_template`` (what these tests
    used before) cannot answer that, so every render-level test here silently
    fell back to the character rule and none of them covered the fix for
    generative-computing/granite-switch#108. There is no fallback any more, so
    the stub has to tokenize.

    ``merges`` names multi-character pieces that swallow their neighbours,
    longest-match-first. Empty (the default) reproduces the ByteLevel shape
    Granite 4.1 has under transformers 5.8.1 -- ``'<requirements>'`` ->
    ``['<', 'requirements', '>']`` -- so existing expectations are unchanged.
    Passing ``merges=('<context',)`` reproduces the 5.9.0 ``Split`` shape --
    ``'<context>'`` -> ``['<context', '>']`` -- which is the tokenization #108
    reports and the one the character rule corrupts.
    """

    _FIRST_ID = 1000

    def __init__(self, chat_template, decode_map=None, merges=()):
        self.chat_template = chat_template
        self._decode_map = decode_map or {}
        # Longest first, so '<context' wins over '<' at the same position.
        self._merges = sorted(merges, key=len, reverse=True)
        self._piece_to_id = {}
        self._id_to_piece = {}

    def _pieces(self, text):
        """Split *text* the way a BPE pre-tokenizer would, per ``merges``."""
        out, i = [], 0
        while i < len(text):
            for merge in self._merges:
                if text.startswith(merge, i):
                    out.append(merge)
                    i += len(merge)
                    break
            else:
                if text[i].isalnum():
                    j = i
                    while j < len(text) and text[j].isalnum():
                        j += 1
                    out.append(text[i:j])
                    i = j
                else:
                    out.append(text[i])
                    i += 1
        return out

    def _id_for(self, piece):
        if piece not in self._piece_to_id:
            token_id = self._FIRST_ID + len(self._piece_to_id)
            self._piece_to_id[piece] = token_id
            self._id_to_piece[token_id] = piece
        return self._piece_to_id[piece]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id_for(p) for p in self._pieces(text)]}

    def ids_for_pieces(self, pieces):
        """Ids for an explicit split, e.g. an adapter's recorded tokenization.

        Lets a test state 'the adapter was trained on [<, context, >]' even on a
        stub whose own ``merges`` would split that text differently -- which is
        the situation the fix exists for.
        """
        return [self._id_for(p) for p in pieces]

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode, preferring an exact whole-sequence mapping.

        Falls back to per-id mappings and then to pieces this instance minted, so
        a fixture can declare ids one at a time and still have the full sequence
        decode compositionally -- which ``alora_invocation_tail`` requires.
        """
        key = tuple(token_ids)
        if key in self._decode_map:
            return self._decode_map[key]
        out = []
        for i in token_ids:
            if (i,) in self._decode_map:
                out.append(self._decode_map[(i,)])
            else:
                out.append(self._id_to_piece[i])
        return "".join(out)


def _make_tokenizer(merges=()):
    return _StubTokenizer(_GRANITE_TEMPLATE, merges=merges)


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
        tokenizer = _make_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<requirements>", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "req_check", "alora")])

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
        assert (
            result[last_gen_pos - len("<|req_check|>") : last_gen_pos]
            != "<|req_check|>"
        )

    def test_alora_fallback_path(self):
        """ALoRA fallback: token emitted before generation prompt when invocation text is absent.

        Pass 1 scans all user messages and finds none containing the decoded invocation
        text (here the assistant role token sequence), so ns.alora_target_idx stays -1
        and the fallback block fires.
        """
        tokenizer = _make_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora(
                "<|start_of_role|>assistant<|end_of_role|>", tokenizer
            ):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "answerability", "alora")])

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
        after = result[token_pos + len(token) :]
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
        tokenizer = _make_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<requirements>", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "req_check", "alora")])

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
        assert result.index("<|req_check|>") > result.index(
            "<|start_of_role|>user<|end_of_role|>"
        )
        # Fallback must NOT also fire
        gen_prompt = "<|start_of_role|>assistant<|end_of_role|>"
        last_gen_pos = result.rindex(gen_prompt)
        assert (
            result[last_gen_pos - len("<|req_check|>") : last_gen_pos]
            != "<|req_check|>"
        )

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
        original = _render(
            _make_tokenizer(), messages=messages, add_generation_prompt=True
        )

        tokenizer = _make_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<requirements>", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(
                tokenizer,
                [("/path/a", "ctx_rel", "lora"), ("/path/b", "req_check", "alora")],
            )
        modified = _render(tokenizer, messages=messages, add_generation_prompt=True)

        assert modified == original


@pytest.mark.requires_model
class TestInvocationTailProperty:
    """The Pass-2 tail must reconstruct the sequence the ADAPTER was trained on.

    The invariant has two halves and they must read the same source. The swap side
    substitutes ``alora_invocation_tokens[0]``
    (``build_substitute_token_ids`` -> ``get_alora_first_invocation_token_id``); the
    emit side must therefore emit ``decode(alora_invocation_tokens[1:])``. Anything
    derived from re-tokenizing the decoded text can disagree with the substitute,
    because the installed tokenizer need not split the text the way the trainer did.

    Three rules are compared below on a real Granite tokenizer, because two of them
    look right and are not (generative-computing/granite-switch#108):

        rule                        '<context>' trained [27, 2196, 29]   '</documents>'
        character   text[1:]        'context>'   right (coincidence)     WRONG
        local token retokenize      '>'          WRONG on tf 5.9.0       right
        adapter     decode(ids[1:]) 'context>'   right                   right

    Only the third is right in both, and it is the only one that never
    re-tokenizes. ``#108``'s own fix sketch proposes the second.

    Pinned revision: #127 saw an unpinned fetch tokenize '<context>' as
    ['<context', '>'] on one CI runner and wrote it off as a partial download. That
    is the legitimate transformers >= 5.9.0 tokenization -- and the reason the rule
    under test must not depend on it at all.
    """

    _MODEL = "ibm-granite/granite-4.1-3b"
    _REVISION = "main"

    # A hypothetical adapter recording '<context>' as ['<', 'context', '>']. NOT a
    # published invocation: every aLoRA in granitelib-{rag,core,guardian}-r1.0
    # records the assistant role sequence, '<requirements>', '<certainty>' or
    # '<guardian>', and all four re-tokenize identically on 5.8.1 and 5.9.0.
    # '<context>' is the string #108 reports on, and it is the smallest case where
    # a recorded tokenization and a local one can diverge, which is the property
    # under test -- so it is a probe, not a claim about shipped adapters.
    _TRAINED_CONTEXT = [27, 2196, 29]

    def _get_tokenizer(self):
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(self._MODEL, revision=self._REVISION)
        except Exception as e:
            pytest.skip(f"could not fetch Granite tokenizer: {e}")

    @staticmethod
    def _trained_variants(tok):
        """Every training shape that must round-trip, as ``{label: ids}``."""
        return {
            "'<context>' recorded as ['<', 'context', '>']": [27, 2196, 29],
            "'<context>' as this tokenizer splits it": tok(
                "<context>", add_special_tokens=False
            ).input_ids,
            "'</documents>' (single vocab entry)": tok(
                "</documents>", add_special_tokens=False
            ).input_ids,
            "'<requirements>'": tok(
                "<requirements>", add_special_tokens=False
            ).input_ids,
        }

    def test_substitute_plus_tail_reconstructs_the_trained_sequence(self):
        """The invariant itself, over every training shape.

        ``[ids[0]] + tokenize(tail)`` must equal ``ids`` -- ids[0] because that is
        literally what the runtime swaps in, and ``tokenize`` because the tail
        reaches the model as text in the rendered prompt.
        """
        from granite_switch.composer.tokenizer_setup import alora_invocation_tail

        tok = self._get_tokenizer()
        for label, trained in self._trained_variants(tok).items():
            tail = alora_invocation_tail(trained, tok)
            on_wire = [trained[0], *tok(tail, add_special_tokens=False).input_ids]
            assert on_wire == trained, (
                f"{label}: control token (standing in for {trained[0]}) plus tail "
                f"{tail!r} reaches the model as {on_wire}, but the adapter was "
                f"trained on {trained}."
            )

    def test_retokenizing_rule_would_break_a_divergent_recording(self):
        """Anti-regression against #108's own suggested fix.

        Guards the specific way this went wrong once: taking the tail from the
        LOCAL tokenizer's first token silently drops 'context' when the installed
        transformers merges '<context'. No published adapter records '<context>',
        so this is the latent case rather than a live one -- but it is the case the
        rule has to be right about, because nothing keeps a recorded tokenization
        and a local one in step. Skips where the merge does not happen, so the test
        states its own precondition rather than assuming a version.
        """
        from granite_switch.composer.tokenizer_setup import alora_invocation_tail

        tok = self._get_tokenizer()
        trained = self._TRAINED_CONTEXT
        text = tok.decode(trained)
        local_first = tok.decode([tok(text, add_special_tokens=False).input_ids[0]])
        if len(local_first) == 1:
            pytest.skip(
                f"this transformers keeps {local_first!r} a lone token, so the "
                "retokenizing rule happens to agree here; nothing to discriminate"
            )

        local_rule_tail = text[len(local_first) :]
        local_on_wire = [
            trained[0],
            *tok(local_rule_tail, add_special_tokens=False).input_ids,
        ]
        assert local_on_wire != trained, (
            "the retokenizing rule no longer differs from the trained sequence "
            "here, so this guard is weaker than it looks"
        )
        assert alora_invocation_tail(trained, tok) != local_rule_tail, (
            "alora_invocation_tail must not agree with the retokenizing rule on "
            f"{text!r} under this tokenizer"
        )

    def test_character_rule_breaks_on_single_token_markers(self):
        """Anti-regression against main's rule.

        Granite's structural markers are single vocab entries, so the character
        rule emits a whole word after a whole-marker substitute. Version
        independent, unlike the '<context>' case above.
        """
        from granite_switch.composer.tokenizer_setup import alora_invocation_tail

        tok = self._get_tokenizer()
        for marker in ["</documents>", "</think>"]:
            trained = tok(marker, add_special_tokens=False).input_ids
            if len(trained) != 1:
                pytest.skip(f"{marker!r} is not a single token on this tokenizer")

            assert alora_invocation_tail(trained, tok) == "", (
                f"{marker!r} is one token, so the substitute carries all of it and "
                "the tail must be empty"
            )
            char_on_wire = [
                trained[0],
                *tok(marker[1:], add_special_tokens=False).input_ids,
            ]
            assert char_on_wire != trained, (
                f"{marker!r} no longer discriminates the character rule on this "
                "tokenizer -- find a marker that does, or drop this test"
            )
            assert len(char_on_wire) > len(trained), (
                f"{marker!r}: expected the character rule to ADD tokens, got "
                f"{char_on_wire} vs {trained}"
            )

    def test_single_token_invocation_yields_empty_tail(self):
        from granite_switch.composer.tokenizer_setup import alora_invocation_tail

        tok = self._get_tokenizer()
        assert alora_invocation_tail([27], tok) == ""

    def test_empty_invocation_tokens_raise(self):
        from granite_switch.composer.tokenizer_setup import alora_invocation_tail

        tok = self._get_tokenizer()
        with pytest.raises(ValueError, match="no first token"):
            alora_invocation_tail([], tok)


class TestMergedFirstTokenRenderEndToEnd:
    """End-to-end through ``configure_chat_template`` on the exact mismatch
    generative-computing/granite-switch#108 is about, offline and deterministically.

    The adapter records ``['<', 'context', '>']`` for ``'<context>'``. That is a
    hypothetical recording, not a published one (see
    ``TestInvocationTailProperty._TRAINED_CONTEXT``); it is the smallest shape on
    which a recorded and a local tokenization can diverge. The stub tokenizer merges
    ``'<context'``, so re-tokenizing the decoded text gives a different split --
    transformers 5.9.0 versus 5.8.1 on identical Granite 4.1 files. Here the merge
    comes from the stub, so no particular transformers version is required.

    Both halves of the invariant are checked together, which is what makes this an
    end-to-end test rather than another property test: the substitute comes from
    ``build_substitute_token_ids`` reading the adapter's tokens, and the tail comes
    from the rendered prompt.

    #108 asked for exactly this and named a test
    (``test_context_invocation_injection_not_corrupted_granite_4_1``) that was never
    committed; this is that test.
    """

    _MERGED = ("<context",)
    _TRAINED_PIECES = ["<", "context", ">"]
    _LOAD_IDS_TARGET = (
        "granite_switch.composer.tokenizer_setup._load_alora_invocation_token_ids"
    )

    def _configured(self, factory, merges):
        """Configure one aLoRA on '<context>' whose adapter recorded the ids above."""
        tokenizer = factory(merges=merges)
        trained = tokenizer.ids_for_pieces(self._TRAINED_PIECES)
        with (
            patch(_PATCH_TARGET, return_value="<context>"),
            patch(self._LOAD_IDS_TARGET, return_value=trained),
        ):
            configure_chat_template(tokenizer, [("/path/ctx", "ctx_rel", "alora")])
        return tokenizer, trained

    def test_stub_reproduces_the_issue_tokenization(self):
        """Guard the guard: without the merge there is nothing to catch."""
        plain = _StubTokenizer(_GRANITE_TEMPLATE)
        merged = _StubTokenizer(_GRANITE_TEMPLATE, merges=self._MERGED)

        assert plain._pieces("<context>") == ["<", "context", ">"], (
            "the default stub must reproduce the ByteLevel / transformers 5.8.1 "
            "shape, or the other render expectations here are being changed"
        )
        assert merged._pieces("<context>") == ["<context", ">"], (
            "the merged stub must reproduce the transformers 5.9.0 / Split shape "
            "from #108, or this class proves nothing"
        )

    @pytest.mark.parametrize("fmt", ["granite", "chatml"])
    def test_swapped_stream_reconstructs_the_trained_invocation(self, fmt):
        """The whole invariant, both halves, through a real render.

        Substitute from the adapter's tokens + tail from the rendered prompt must
        retokenize to what the adapter was trained on.
        """
        factory = _make_tokenizer if fmt == "granite" else _make_chatml_tokenizer
        tokenizer, trained = self._configured(factory, self._MERGED)

        rendered = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Judge <context> here"}],
            add_generation_prompt=True,
            adapter_name="ctx_rel",
        )
        assert "<|ctx_rel|>" in rendered, (
            f"control token missing from the {fmt} render: {rendered!r}"
        )

        with patch(self._LOAD_IDS_TARGET, return_value=trained):
            substitute = build_substitute_token_ids(
                [("/path/ctx", "ctx_rel", "alora", None)], lora_substitute_id=-1
            )
        assert substitute == [trained[0]], (
            f"the swap side must substitute the adapter's first trained token "
            f"{trained[0]}, got {substitute}"
        )

        tail_text = rendered.split("<|ctx_rel|>", 1)[1]
        on_wire = [
            substitute[0],
            *tokenizer(tail_text)["input_ids"][: len(trained) - 1],
        ]
        assert on_wire == trained, (
            f"{fmt}: control token + emitted tail reaches the model as {on_wire} "
            f"but the adapter was trained on {trained}. Rendered: {rendered!r}"
        )

    def test_render_emits_the_adapter_tail_not_the_retokenized_one(self):
        """The regression this class was rewritten for.

        Under the merge, the retokenizing rule emits '>' and drops 'context'
        entirely while the substitute is still '<'. The adapter rule emits
        'context>'. This pins which one lands in the prompt.
        """
        tokenizer, trained = self._configured(_make_tokenizer, self._MERGED)
        rendered = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Judge <context> here"}],
            add_generation_prompt=True,
            adapter_name="ctx_rel",
        )
        after = rendered.split("<|ctx_rel|>", 1)[1]

        assert after.startswith("context>"), (
            "the emitted tail must be the decoding of the adapter's tokens after "
            f"the first ('context>'), got {after[:20]!r}. Starting with '>' means "
            "the retokenizing rule is back and 'context' has been dropped."
        )
        assert not after.startswith("<context>"), (
            f"the control token is followed by the FULL invocation, so the swap "
            f"site carries it twice: {rendered!r}"
        )

    def test_character_rule_would_corrupt_this_render(self):
        """Anti-regression: main's rule must visibly fail this same render.

        Here the character rule agrees with the adapter rule (both give
        'context>'), so the discriminator has to be a single-token marker; this
        asserts that discrimination survives on the stub.
        """
        tokenizer = _make_tokenizer(merges=("</documents>",))
        trained = tokenizer.ids_for_pieces(["</documents>"])
        marker = "</documents>"

        char_on_wire = [trained[0], *tokenizer(marker[1:])["input_ids"]]
        assert char_on_wire != trained, (
            "'</documents>' no longer discriminates the character rule under this "
            "stub -- fix the merge or drop this test"
        )
        assert len(char_on_wire) > len(trained), (
            f"expected the character rule to ADD tokens, got {char_on_wire} vs "
            f"{trained}"
        )


#: Per-id decoding of the context-relevance fixture's alora_invocation_tokens
#: ([27, 2196, 29] -- '<context>' as Granite 4.1 splits it under ByteLevel). Given
#: per id, not as one whole-sequence entry, so decode() composes: the Pass-2 tail is
#: decode(ids[1:]) and must concatenate with decode(ids[:1]) to the full text.
_CONTEXT_DECODE_MAP = {(27,): "<", (2196,): "context", (29,): ">"}

#: Same, for the answerability fixture, whose invocation is the assistant role
#: sequence ([100264, 78191, 100265]). The whole-sequence entry stays: it is what
#: Pass 1 matches on, and it must equal the concatenation of the per-id ones.
_ANSWERABILITY_DECODE_MAP = {
    (100264, 78191, 100265): "<|start_of_role|>assistant<|end_of_role|>",
    (100264,): "<|start_of_role|>",
    (78191,): "assistant",
    (100265,): "<|end_of_role|>",
}


class _FixtureTokenizer(_StubTokenizer):
    """``_StubTokenizer`` with a decode map for fixture adapter token IDs.

    The decode map answers the adapter-config ids the fixtures declare; anything
    else falls through to the stub's own tokenize/decode, which is what lets the
    Pass-2 tail be computed for real here.
    """


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
        tokenizer = self._make_tokenizer(_ANSWERABILITY_DECODE_MAP)
        configure_chat_template(
            tokenizer,
            [
                (self._ANSWERABILITY, "answerability", "alora"),
            ],
        )

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
        after = result[token_pos + len(token) :]
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
        tokenizer = self._make_tokenizer(_CONTEXT_DECODE_MAP)
        configure_chat_template(
            tokenizer,
            [
                (self._CONTEXT_REL, "context_relevance", "alora"),
            ],
        )

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
        assert (
            result[last_gen_pos - len("<|context_relevance|>") : last_gen_pos]
            != "<|context_relevance|>"
        )

    def test_alora_invocation_mid_user_message(self):
        """ALoRA: invocation text appears in the middle of the user message.

        Same first-character drop as the start-of-message case.
        """
        tokenizer = self._make_tokenizer(_CONTEXT_DECODE_MAP)
        configure_chat_template(
            tokenizer,
            [
                (self._CONTEXT_REL, "context_relevance", "alora"),
            ],
        )

        result = _render(
            tokenizer,
            messages=[
                {"role": "user", "content": "Please review: <context>docs</context>"}
            ],
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
        assert (
            result[last_gen_pos - len("<|context_relevance|>") : last_gen_pos]
            != "<|context_relevance|>"
        )

    def test_alora_multiple_occurrences_targets_last(self):
        """ALoRA: invocation text appears twice — token injected before the last occurrence.

        rsplit(..., 1) splits on the last occurrence, so the control token must
        land before the second <context>, not the first. First occurrence
        remains intact with its '<'; only the second has its '<' dropped.
        """
        tokenizer = self._make_tokenizer(_CONTEXT_DECODE_MAP)
        configure_chat_template(
            tokenizer,
            [
                (self._CONTEXT_REL, "context_relevance", "alora"),
            ],
        )

        result = _render(
            tokenizer,
            messages=[
                {
                    "role": "user",
                    "content": "<context>first batch</context> Also check <context>second batch</context>",
                }
            ],
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        # First <context> untouched; second one has the control token
        # inserted with its '<' dropped.
        assert (
            "<context>first batch</context> Also check <|context_relevance|>context>second batch"
            in result
        )
        # Only one control token in the entire output
        assert result.count("<|context_relevance|>") == 1

    def test_lora_prefix_from_adapter_config(self):
        """LoRA adapter (no alora_invocation_tokens) → prefix path."""
        tokenizer = self._make_tokenizer({})  # no decode needed for LoRA
        configure_chat_template(
            tokenizer,
            [
                (self._SUMMARIZATION, "summarization", "lora"),
            ],
        )

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
        tokenizer = self._make_tokenizer(
            {**_ANSWERABILITY_DECODE_MAP, **_CONTEXT_DECODE_MAP}
        )
        configure_chat_template(
            tokenizer,
            [
                (self._ANSWERABILITY, "answerability", "alora"),
                (self._CONTEXT_REL, "context_relevance", "alora"),
                (self._SUMMARIZATION, "summarization", "lora"),
            ],
        )

        messages = [{"role": "user", "content": "<context>docs</context>"}]

        # Activate context_relevance → Pass 1+2 (drops first char of invocation).
        result = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="context_relevance",
        )
        assert "<|context_relevance|>context>" in result
        assert "<|context_relevance|><context>" not in result

        # Activate answerability → fallback (skip-once suppresses the
        # generation-prompt <|start_of_role|>).
        result = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="answerability",
        )
        token = "<|answerability|>"
        token_pos = result.index(token)
        after = result[token_pos + len(token) :]
        assert after.startswith("assistant<|end_of_role|>")

        # Activate summarization → prefix
        result = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="summarization",
        )
        assert result.startswith("<|summarization|>")

        # No adapter → no tokens
        result_none = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
        )
        assert "<|answerability|>" not in result_none
        assert "<|context_relevance|>" not in result_none
        assert "<|summarization|>" not in result_none


# ---------------------------------------------------------------------------
# ChatML (Granite 4.2) render-level tests
# ---------------------------------------------------------------------------


def _make_chatml_tokenizer(merges=()):
    return _StubTokenizer(_CHATML_TEMPLATE, merges=merges)


class TestDetectTemplateFormat:
    """detect_template_format() classifies the two Granite template families."""

    def test_detects_chatml(self):
        fmt = detect_template_format(_CHATML_TEMPLATE)
        assert fmt is not None
        assert fmt.name == "chatml"
        assert fmt.role_open_marker == "<|im_start|>"
        assert fmt.content_accessor == "content"
        assert fmt.loop_var_source == "loop_messages"
        assert fmt.ns_merge_last is True

    def test_detects_granite_format(self):
        fmt = detect_template_format(_GRANITE_TEMPLATE)
        assert fmt is not None
        assert fmt.name == "granite_format"
        assert fmt.role_open_marker == "<|start_of_role|>"
        assert fmt.content_accessor == "content.val"
        assert fmt.loop_var_source == "messages"
        assert fmt.ns_merge_last is False

    def test_unknown_returns_none(self):
        assert detect_template_format("no markers here") is None
        assert detect_template_format("") is None
        assert detect_template_format(None) is None

    def test_chatml_wins_when_both_markers_present(self):
        both = "<|start_of_role|> and <|im_start|>"
        assert detect_template_format(both).name == "chatml"


class TestConfigureChatTemplateChatML:
    """configure_chat_template() against the real Granite 4.2 ChatML template.

    The trained 4.2 aLoRA adapters use the assistant-boundary invocation
    (``<|im_start|>assistant\\n``) → ALoRA fallback path. A hypothetical
    user-message invocation (``<context>``) exercises Pass 1 / Pass 2.
    """

    def test_lora_prefix_path(self):
        """LoRA: control token at sequence start, first <|im_start|> suppressed.

        The rendered output opens with ``<|my_lora|>system\\n`` (skip-once
        consumed the leading ``<|im_start|>``), not
        ``<|my_lora|><|im_start|>system``.
        """
        tokenizer = _make_chatml_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "my_lora", "lora")])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
            adapter_name="my_lora",
        )
        # apply_chat_template strips leading whitespace at tokenize time; the
        # jinja renderer leaves a leading newline, so strip before asserting.
        assert result.lstrip("\n").startswith("<|my_lora|>system\n"), (
            f"expected '<|my_lora|>system' at start (skip-once suppressed "
            f"<|im_start|>), got {result[:80]!r}"
        )
        # The control token must be followed by the role name, not <|im_start|>.
        pos = result.index("<|my_lora|>")
        after = result[pos + len("<|my_lora|>") :]
        assert not after.startswith("<|im_start|>")

    def test_skip_once_is_single_shot(self):
        """Only the first <|im_start|> after a LoRA token is suppressed."""
        tokenizer = _make_chatml_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "my_lora", "lora")])

        no_adapter = _render(
            _make_chatml_tokenizer(),
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
            add_generation_prompt=True,
        )
        with_adapter = _render(
            tokenizer,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ],
            add_generation_prompt=True,
            adapter_name="my_lora",
        )
        # Exactly one <|im_start|> is suppressed relative to the no-adapter
        # render (and one control token is added).
        assert (
            with_adapter.count("<|im_start|>") == no_adapter.count("<|im_start|>") - 1
        )
        assert with_adapter.count("<|my_lora|>") == 1

    def test_alora_fallback_path(self):
        """ALoRA assistant-boundary: token before <|im_start|>assistant\\n<think>.

        The 4.2 adapters' invocation decodes to ``<|im_start|>assistant\\n``,
        which never appears in a user message, so Pass 1 leaves
        alora_target_idx == -1 and the fallback fires before the generation
        prompt. Skip-once suppresses the generation-prompt <|im_start|>, leaving
        ``<|gsm8k|>assistant\\n<think>`` — the runtime swap restores <|im_start|>.
        """
        tokenizer = _make_chatml_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<|im_start|>assistant\n", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "gsm8k", "alora")])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            add_generation_prompt=True,
            adapter_name="gsm8k",
        )
        assert "<|gsm8k|>" in result
        pos = result.rindex("<|gsm8k|>")
        after = result[pos + len("<|gsm8k|>") :]
        # <|im_start|> is suppressed; assistant\n<think> is preserved.
        assert after.startswith("assistant\n<think>"), (
            f"expected 'assistant\\n<think>' immediately after <|gsm8k|>, "
            f"got {after[:40]!r}"
        )
        # Control token appears exactly once.
        assert result.count("<|gsm8k|>") == 1

    def test_alora_fallback_thinking_off(self):
        """ALoRA fallback works with enable_thinking=False (<think></think>)."""
        tokenizer = _make_chatml_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<|im_start|>assistant\n", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "gsm8k", "alora")])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            enable_thinking=False,
            adapter_name="gsm8k",
        )
        pos = result.rindex("<|gsm8k|>")
        after = result[pos + len("<|gsm8k|>") :]
        assert after.startswith("assistant\n<think></think>"), f"got {after[:40]!r}"

    def test_alora_pass2_user_message(self):
        """ALoRA Pass 1+2 for a user-message invocation under ChatML.

        Uses a hypothetical ``<context>`` invocation to exercise the plain
        ``content`` string mutation (not ``content.val``). The control token is
        inserted before the invocation text with its first char dropped.
        """
        tokenizer = _make_chatml_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<context>", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "ctxrel", "alora")])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "<context>docs</context>"}],
            add_generation_prompt=True,
            adapter_name="ctxrel",
        )
        assert "<|ctxrel|>context>" in result
        assert "<|ctxrel|><context>" not in result
        # Fallback must NOT fire: no control token right before the gen prompt.
        gen = "<|im_start|>assistant\n"
        last = result.rindex(gen)
        assert result[last - len("<|ctxrel|>") : last] != "<|ctxrel|>"

    def test_alora_pass2_index_alignment_with_system(self):
        """Pass 1/Pass 2 index alignment when a system message is present.

        ChatML strips the system message into ``loop_messages``; Pass 1 must
        iterate the same list so its recorded index matches the main loop.
        A misaligned index would target the wrong message or crash.
        """
        tokenizer = _make_chatml_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<context>", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "ctxrel", "alora")])

        result = _render(
            tokenizer,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "<context>docs</context>"},
            ],
            add_generation_prompt=True,
            adapter_name="ctxrel",
        )
        assert "<|ctxrel|>context>" in result
        assert result.count("<|ctxrel|>") == 1

    def test_no_adapter_equals_original(self):
        """Without adapter_name the ChatML render is byte-identical to base."""
        messages = [{"role": "user", "content": "Hello"}]
        original = _render(
            _make_chatml_tokenizer(), messages=messages, add_generation_prompt=True
        )
        tokenizer = _make_chatml_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<|im_start|>assistant\n", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(
                tokenizer,
                [("/path/a", "gsm8k", "alora"), ("/path/b", "my_lora", "lora")],
            )
        modified = _render(tokenizer, messages=messages, add_generation_prompt=True)
        assert modified == original

    def test_multi_turn_tool_conversation_single_control_token(self):
        """Regression: a full multi-turn conversation (system + tools +
        assistant + tool responses) renders with the ALoRA control token in
        exactly one place under ChatML's more complex message handling."""
        tokenizer = _make_chatml_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<|im_start|>assistant\n", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "gsm8k", "alora")])

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
            },
            {"role": "tool", "content": "sunny, 72F"},
            {"role": "user", "content": "Thanks, and in Paris?"},
        ]
        result = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="gsm8k",
        )
        assert result.count("<|gsm8k|>") == 1
        # Assistant-boundary fallback: control token right before final gen prompt.
        pos = result.rindex("<|gsm8k|>")
        after = result[pos + len("<|gsm8k|>") :]
        assert after.startswith("assistant\n<think>")


# ════════════════════════════════════════════════════════════════════
# Audio: <|audio|> marker preservation, and coexistence with adapters
# ════════════════════════════════════════════════════════════════════


def _audio_messages(text="transcribe this", audio_type="audio"):
    """A user turn whose content is a parts list carrying an audio clip."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": audio_type, audio_type: "clip-placeholder"},
            ],
        }
    ]


@pytest.mark.audio
class TestAudioChatTemplate:
    """configure_audio_chat_template emits the <|audio|> marker for audio parts."""

    def test_audio_part_emits_marker(self):
        tokenizer = _make_tokenizer()
        configure_audio_chat_template(tokenizer)
        result = _render(
            tokenizer, messages=_audio_messages(), add_generation_prompt=True
        )
        assert "<|audio|>" in result
        assert "transcribe this" in result

    def test_marker_dropped_without_injection(self):
        # Regression canary: the un-injected base template drops the audio part.
        tokenizer = _make_tokenizer()
        result = _render(
            tokenizer, messages=_audio_messages(), add_generation_prompt=True
        )
        assert "<|audio|>" not in result
        assert "transcribe this" in result

    def test_input_audio_and_audio_url_types_emit_marker(self):
        for audio_type in ("input_audio", "audio_url"):
            tokenizer = _make_tokenizer()
            configure_audio_chat_template(tokenizer)
            result = _render(
                tokenizer,
                messages=_audio_messages(audio_type=audio_type),
                add_generation_prompt=True,
            )
            assert "<|audio|>" in result, f"marker missing for type={audio_type!r}"

    def test_custom_marker_string(self):
        tokenizer = _make_tokenizer()
        configure_audio_chat_template(tokenizer, marker="<|snd|>")
        result = _render(
            tokenizer, messages=_audio_messages(), add_generation_prompt=True
        )
        assert "<|snd|>" in result

    def test_missing_anchor_raises(self):
        # A granite_format template (has the role marker) whose content-part
        # loop is gone: the family is known but the anchor is not.
        tokenizer = SimpleNamespace(
            chat_template="{{- '<|start_of_role|>user<|end_of_role|>' }}{{ messages }}"
        )
        with pytest.raises(ValueError, match="content-part loop"):
            configure_audio_chat_template(tokenizer)

    def test_unrecognized_format_raises(self):
        # Neither role marker present, so the family cannot be detected. Audio
        # was explicitly requested by this point, so this is loud, not silent.
        tokenizer = SimpleNamespace(chat_template="{{ messages }}")
        with pytest.raises(ValueError, match="detect the chat-template family"):
            configure_audio_chat_template(tokenizer)

    def test_none_template_is_noop(self):
        tokenizer = SimpleNamespace(chat_template=None)
        configure_audio_chat_template(tokenizer)
        assert tokenizer.chat_template is None

    def test_string_content_render_is_byte_identical(self):
        """Injection must not perturb a plain string-content render.

        ``_probe_lora_substitute_token_id`` runs *after* the audio injection and
        reads token 0 of a rendered string-content chat, so any drift here would
        silently change the LoRA substitute token.
        """
        messages = [{"role": "user", "content": "plain text"}]
        before = _render(
            _make_tokenizer(), messages=messages, add_generation_prompt=True
        )
        tokenizer = _make_tokenizer()
        configure_audio_chat_template(tokenizer)
        after = _render(tokenizer, messages=messages, add_generation_prompt=True)
        assert before == after


@pytest.mark.audio
class TestAudioAndAdapterInjectionsCompose:
    """Adapter and audio injections applied in sequence leave both intact."""

    def test_lora_prefix_and_audio_marker_coexist(self):
        tokenizer = _make_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "ctx_rel", "lora")])
        configure_audio_chat_template(tokenizer)

        result = _render(
            tokenizer,
            messages=_audio_messages(),
            add_generation_prompt=True,
            adapter_name="ctx_rel",
        )
        assert result.startswith("<|ctx_rel|>"), result[:80]
        assert "<|audio|>" in result
        assert "transcribe this" in result

    def test_no_adapter_still_emits_audio_marker(self):
        tokenizer = _make_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "ctx_rel", "lora")])
        configure_audio_chat_template(tokenizer)

        result = _render(
            tokenizer, messages=_audio_messages(), add_generation_prompt=True
        )
        assert "<|audio|>" in result
        assert "<|ctx_rel|>" not in result


# ════════════════════════════════════════════════════════════════════
# Audio on the ChatML (Granite 4.2) template family
# ════════════════════════════════════════════════════════════════════
#
# ChatML has no content-part loop at all — its user/system branch does
# ``{%- set content = message.content | string %}`` — so a parts list would
# render as that list's Python repr rather than dropping the audio part the way
# granite_format does. The failure mode is therefore *louder* and worse: the
# base64 audio payload would land in the prompt. These tests pin both the fix
# and that specific regression.


def _make_chatml_audio_tokenizer(merges=()):
    return _StubTokenizer(_CHATML_TEMPLATE, merges=merges)


def _chatml_user_turn(rendered):
    """The text between the user role header and its closing marker."""
    return rendered.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]


@pytest.mark.audio
class TestAudioChatTemplateChatML:
    """configure_audio_chat_template wires audio into the ChatML family too."""

    def test_audio_part_emits_marker(self):
        tokenizer = _make_chatml_audio_tokenizer()
        configure_audio_chat_template(tokenizer)
        result = _render(
            tokenizer, messages=_audio_messages(), add_generation_prompt=True
        )
        assert "<|audio|>" in result
        assert "transcribe this" in result

    def test_parts_repr_does_not_leak_into_prompt(self):
        """The clip payload must never reach the prompt.

        Without the flattening block ChatML stringifies the parts list, so the
        rendered prompt contains ``[{'type': 'audio', ...}]`` — payload included.
        """
        tokenizer = _make_chatml_audio_tokenizer()
        configure_audio_chat_template(tokenizer)
        result = _render(
            tokenizer, messages=_audio_messages(), add_generation_prompt=True
        )
        assert "clip-placeholder" not in result
        assert "'type'" not in result
        assert _chatml_user_turn(result) == "transcribe this<|audio|>"

    def test_uninjected_chatml_leaks_the_repr(self):
        """Regression canary for the pre-fix behaviour."""
        result = _render(
            _make_chatml_audio_tokenizer(),
            messages=_audio_messages(),
            add_generation_prompt=True,
        )
        assert "<|audio|>" not in result
        assert "clip-placeholder" in result

    def test_input_audio_and_audio_url_types_emit_marker(self):
        for audio_type in ("input_audio", "audio_url"):
            tokenizer = _make_chatml_audio_tokenizer()
            configure_audio_chat_template(tokenizer)
            result = _render(
                tokenizer,
                messages=_audio_messages(audio_type=audio_type),
                add_generation_prompt=True,
            )
            assert "<|audio|>" in result, f"marker missing for type={audio_type!r}"

    def test_custom_marker_string(self):
        tokenizer = _make_chatml_audio_tokenizer()
        configure_audio_chat_template(tokenizer, marker="<|snd|>")
        result = _render(
            tokenizer, messages=_audio_messages(), add_generation_prompt=True
        )
        assert "<|snd|>" in result

    def test_string_content_render_is_byte_identical(self):
        messages = [{"role": "user", "content": "plain text"}]
        before = _render(
            _make_chatml_audio_tokenizer(),
            messages=messages,
            add_generation_prompt=True,
        )
        tokenizer = _make_chatml_audio_tokenizer()
        configure_audio_chat_template(tokenizer)
        after = _render(tokenizer, messages=messages, add_generation_prompt=True)
        assert before == after

    def test_part_order_and_multiple_clips_preserved(self):
        """Marker position tracks the clip's position among the parts.

        The ASR processor replaces each marker with that clip's transcript, so a
        reordered or collapsed marker sequence would splice transcripts into the
        wrong place.
        """
        tokenizer = _make_chatml_audio_tokenizer()
        configure_audio_chat_template(tokenizer)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "a"},
                    {"type": "text", "text": "and"},
                    {"type": "audio", "audio": "b"},
                ],
            }
        ]
        result = _render(tokenizer, messages=messages, add_generation_prompt=True)
        assert _chatml_user_turn(result) == "<|audio|>\nand<|audio|>"

    def test_pass2_anchor_survives_injection(self):
        """The ALoRA Pass 2 anchor must remain findable after audio injection.

        The flattening block is appended after the content assignment rather
        than replacing it precisely so configure_chat_template can still anchor
        on it, in either call order.
        """
        tokenizer = _make_chatml_audio_tokenizer()
        configure_audio_chat_template(tokenizer)
        assert (
            "{%- set content = message.content | string %}" in tokenizer.chat_template
        )

    def test_missing_chatml_anchor_raises(self):
        tokenizer = SimpleNamespace(
            chat_template="{{- '<|im_start|>user\\n' }}{{ messages }}"
        )
        with pytest.raises(ValueError, match="ChatML per-message content assignment"):
            configure_audio_chat_template(tokenizer)


@pytest.mark.audio
class TestAudioAndAdapterInjectionsComposeChatML:
    """Adapter + audio injection on ChatML, in the order compose applies them."""

    def test_lora_prefix_and_audio_marker_coexist(self):
        tokenizer = _make_chatml_audio_tokenizer()
        configure_chat_template(tokenizer, [("/path/a", "ctx_rel", "lora")])
        configure_audio_chat_template(tokenizer)

        result = _render(
            tokenizer,
            messages=_audio_messages(),
            add_generation_prompt=True,
            adapter_name="ctx_rel",
        )
        assert "<|ctx_rel|>" in result
        assert "<|audio|>" in result
        assert "clip-placeholder" not in result

    def test_alora_pass2_runs_on_flattened_content(self):
        """Pass 2 calls ``rsplit`` on ``content``, so it needs the flattened string.

        Compose applies configure_chat_template first, which anchors Pass 2 right
        after the content assignment; the audio block then lands between them.
        If that order inverted, Pass 2 would rsplit the list repr and the control
        token would never be placed.
        """
        tokenizer = _make_chatml_audio_tokenizer()
        with contextlib.ExitStack() as _stack:
            for _p in _fake_alora("<requirements>", tokenizer):
                _stack.enter_context(_p)
            configure_chat_template(tokenizer, [("/path/a", "req_check", "alora")])
            configure_audio_chat_template(tokenizer)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "clip"},
                    {"type": "text", "text": "<requirements>r1"},
                ],
            }
        ]
        result = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="req_check",
        )
        # Control token placed before the invocation text, first char dropped,
        # with the audio marker still ahead of it in part order.
        assert _chatml_user_turn(result) == "<|audio|>\n<|req_check|>requirements>r1"


# ---------------------------------------------------------------------------
# Shadow Residual anchor placement
# ---------------------------------------------------------------------------


class _SRTokenizer:
    """Tokenizer that can encode a Shadow Residual anchor to a single id.

    ``encode_map`` pins the ids the tests assert on; everything else falls
    through to a greedy special-token-then-character split, which is enough for
    the composer's only question — is this candidate substitution site exactly
    one token?  Real Granite tokenizers answer yes for ``<think>`` and no for
    ``\\n<think>``, and so does this.
    """

    _SPECIALS = (
        "<|start_of_role|>",
        "<|end_of_role|>",
        "<|end_of_text|>",
        "<|im_start|>",
        "<|im_end|>",
        "</think>",
        "<think>",
    )

    def __init__(self, chat_template, encode_map):
        self.chat_template = chat_template
        self._encode_map = encode_map

    def encode(self, text, add_special_tokens=False):
        if text in self._encode_map:
            return self._encode_map[text]
        ids, pos = [], 0
        while pos < len(text):
            for special in self._SPECIALS:
                if text.startswith(special, pos):
                    ids.append(hash(special) % 10000)
                    pos += len(special)
                    break
            else:
                ids.append(ord(text[pos]))
                pos += 1
        return ids


class TestShadowResidualAnchorPlacement:
    """SR: the control token *replaces* last_context_token at the end of the
    generation prompt, so the post-swap id sequence equals a no-adapter render.

    Unlike aLoRA there is no invocation-sequence search and no first-character
    drop: the anchor is a single token by construction, and the runtime
    embedding swap puts its embedding back at the control token's position.
    """

    _SR_GRANITE = os.path.join(_FIXTURES, "sr_answerability_adapter")
    _SR_CHATML = os.path.join(_FIXTURES, "sr_answerability_adapter_chatml")

    def test_granite_anchor_replaces_generation_prompt_tail(self):
        tokenizer = _SRTokenizer(_GRANITE_TEMPLATE, {"<|end_of_role|>": [49153]})
        configure_chat_template(tokenizer, [(self._SR_GRANITE, "answerability", "sr")])

        result = _render(
            tokenizer,
            messages=[{"role": "user", "content": "Is this answerable?"}],
            add_generation_prompt=True,
            adapter_name="answerability",
        )
        assert result.count("<|answerability|>") == 1
        # The generation prompt keeps its role marker and role name; only the
        # trailing <|end_of_role|> becomes the control token.
        assert result.endswith("<|start_of_role|>assistant<|answerability|>"), (
            f"expected the control token in place of the generation prompt's "
            f"<|end_of_role|>, got {result[-70:]!r}"
        )

    def test_granite_earlier_anchors_survive(self):
        """Only the generation prompt's anchor is replaced.

        <|end_of_role|> also closes every earlier role marker; replacing those
        would corrupt the history, so the transform is confined to the
        add_generation_prompt block.
        """
        tokenizer = _SRTokenizer(_GRANITE_TEMPLATE, {"<|end_of_role|>": [49153]})
        configure_chat_template(tokenizer, [(self._SR_GRANITE, "answerability", "sr")])

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        no_adapter = _render(
            _SRTokenizer(_GRANITE_TEMPLATE, {}),
            messages=messages,
            add_generation_prompt=True,
        )
        with_adapter = _render(
            tokenizer,
            messages=messages,
            add_generation_prompt=True,
            adapter_name="answerability",
        )
        # Exactly one <|end_of_role|> is consumed — the generation prompt's.
        assert with_adapter.count("<|end_of_role|>") == (
            no_adapter.count("<|end_of_role|>") - 1
        )
        # And substituting the anchor back reproduces the no-adapter render
        # verbatim: that is the invariant the embedding swap preserves.
        assert (
            with_adapter.replace("<|answerability|>", "<|end_of_role|>") == no_adapter
        )

    def test_granite_no_adapter_render_unchanged(self):
        """With no adapter_name the SR-patched template renders as the original."""
        tokenizer = _SRTokenizer(_GRANITE_TEMPLATE, {"<|end_of_role|>": [49153]})
        configure_chat_template(tokenizer, [(self._SR_GRANITE, "answerability", "sr")])

        messages = [{"role": "user", "content": "Hello"}]
        assert _render(tokenizer, messages=messages, add_generation_prompt=True) == (
            _render(
                _SRTokenizer(_GRANITE_TEMPLATE, {}),
                messages=messages,
                add_generation_prompt=True,
            )
        )

    def test_chatml_anchor_replaces_opening_think_in_both_branches(self):
        """4.2: the site moves back to <think>, which both render paths emit.

        The declared last_context_token </think> only exists in the
        enable_thinking=False prompt, so substituting it would leave the
        *default* render with no control token and ship an inert adapter.
        <think> is the last token of the two branches' common prefix, so one
        control token with one substitute embedding serves both.
        """
        tokenizer = _SRTokenizer(_CHATML_TEMPLATE, {"</think>": [49154]})
        configure_chat_template(tokenizer, [(self._SR_CHATML, "answerability", "sr")])

        messages = [{"role": "user", "content": "Is this answerable?"}]
        for enable_thinking, expected_tail in (
            (False, "<|im_start|>assistant\n<|answerability|></think>"),
            (True, "<|im_start|>assistant\n<|answerability|>\n"),
        ):
            result = _render(
                tokenizer,
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                adapter_name="answerability",
            )
            assert result.count("<|answerability|>") == 1, (
                f"enable_thinking={enable_thinking}: expected exactly one "
                f"control token, got {result[-70:]!r}"
            )
            assert result.endswith(expected_tail), (
                f"enable_thinking={enable_thinking}: expected the control token "
                f"in place of <think>, got {result[-70:]!r}"
            )

    def test_chatml_history_think_tags_survive(self):
        """<think> appears in history and truncation logic, not only in emissions.

        A blanket literal replacement would break `.split('</think>')` and the
        assistant turns' own tags; the transform must only touch the generation
        prompt.  Substituting the site back reproduces the no-adapter render
        verbatim, in *both* thinking modes — that is the invariant the runtime
        embedding swap preserves.
        """
        tokenizer = _SRTokenizer(_CHATML_TEMPLATE, {"</think>": [49154]})
        configure_chat_template(tokenizer, [(self._SR_CHATML, "answerability", "sr")])

        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "<think>reasoning</think>reply"},
            {"role": "user", "content": "second"},
        ]
        for enable_thinking in (False, True):
            no_adapter = _render(
                _SRTokenizer(_CHATML_TEMPLATE, {}),
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            with_adapter = _render(
                tokenizer,
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                adapter_name="answerability",
            )
            assert with_adapter.replace("<|answerability|>", "<think>") == no_adapter, (
                f"enable_thinking={enable_thinking}"
            )

    def test_chatml_substitute_id_is_the_resolved_site_not_the_anchor(self):
        """The returned substitute id restores <think>, not the declared </think>.

        The runtime swap keys on the control token's id, so there is exactly one
        substitute embedding per adapter; if it did not match the token actually
        displaced, the post-swap sequence would diverge from a no-adapter render.
        """
        tokenizer = _SRTokenizer(_CHATML_TEMPLATE, {"</think>": [49154]})
        _fmt, substitutes = configure_chat_template(
            tokenizer, [(self._SR_CHATML, "answerability", "sr")]
        )
        assert substitutes == {
            "answerability": tokenizer.encode("<think>", add_special_tokens=False)[0]
        }
        assert substitutes["answerability"] != 49154

    def test_every_generation_prompt_branch_must_be_patched(self):
        """A branch left without a control token is a compose-time error.

        This is the failure the old count>0 guard let through: on 4.2 only the
        enable_thinking=False branch emits </think>, so patching it satisfied
        "something was patched" while the *default* render shipped an adapter
        that never activates.
        """
        import pytest

        # Two branches whose only shared literal is the role marker, which the
        # anchor is not part of and which is not a candidate site here because
        # the branches diverge immediately after it.
        template = _CHATML_TEMPLATE.replace(
            "{{- '<|im_start|>assistant\\n<think>\\n' }}",
            "{{- '<|im_start|>assistant\\n' }}",
        )
        assert template != _CHATML_TEMPLATE
        tokenizer = _SRTokenizer(template, {"</think>": [49154]})
        with pytest.raises(ValueError, match="render paths that do not all emit"):
            configure_chat_template(
                tokenizer, [(self._SR_CHATML, "answerability", "sr")]
            )

    def test_sr_anchor_is_a_single_token(self):
        """A multi-token anchor is a compose-time error, not a silent miss.

        The runtime swap replaces exactly one embedding, so an anchor that does
        not encode to one id cannot be honored.
        """
        import pytest

        tokenizer = _SRTokenizer(_GRANITE_TEMPLATE, {"<|end_of_role|>": [1, 2, 3]})
        with pytest.raises(ValueError, match="encodes to 3 tokens"):
            configure_chat_template(
                tokenizer, [(self._SR_GRANITE, "answerability", "sr")]
            )

    def test_sr_anchor_id_must_match_the_tokenizer(self):
        """A recorded id that disagrees with this tokenizer is a hard error.

        It means the adapter was trained against a different tokenizer, which
        would place the activation somewhere else entirely.
        """
        import pytest

        tokenizer = _SRTokenizer(_GRANITE_TEMPLATE, {"<|end_of_role|>": [12345]})
        with pytest.raises(ValueError, match="last_context_token_id=49153"):
            configure_chat_template(
                tokenizer, [(self._SR_GRANITE, "answerability", "sr")]
            )
