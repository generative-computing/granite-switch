# SPDX-License-Identifier: Apache-2.0
"""Tests for _probe_lora_substitute_token_id.

The probe derives the LoRA substitute token from the tokenizer's chat
template rather than hardcoding a Granite-4.x-specific token string. These
tests verify:

1. On the Granite tokenizers, the probe returns the role marker the LoRA prefix
   insertion places immediately after the control token in the rendered prompt:
   <|start_of_role|> (id 100264) on 4.0/4.1, <|im_start|> (id 100256) on 4.2.
2. On a synthetic tokenizer with a different template, the probe returns
   whatever that template emits first for a user turn.
3. The probe raises a clear error when the template is missing, fails to
   render, or emits an unknown token.
"""

import os
from types import SimpleNamespace

import pytest

from granite_switch.composer.compose_granite_switch import (
    _probe_lora_substitute_token_id,
)
from tests.shared import model_sources

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestOnGraniteTokenizer:
    """The probe on each Granite base's tokenizer and chat template.

    ``fake`` (every PR run) is the captured tokenizer config and template on a tiny
    vocabulary, whose special tokens keep their real ids; ``real`` (release tier)
    is the published tokenizer. Same assertions on both.
    """

    @pytest.mark.parametrize("source", model_sources.sources())
    @pytest.mark.parametrize(
        "base_name, token, token_id",
        [
            ("granite-4.0-micro", "<|start_of_role|>", 100264),
            ("granite-4.1-3b", "<|start_of_role|>", 100264),
            # ChatML: the role marker is <|im_start|>.
            ("granite-4.2-3b", "<|im_start|>", 100256),
        ],
    )
    def test_probe_returns_the_role_marker(self, source, base_name, token, token_id):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_sources.base_model(source, base_name))
        sub_id = _probe_lora_substitute_token_id(tok)
        assert sub_id == token_id
        assert tok.convert_ids_to_tokens([sub_id])[0] == token


class TestChatMLMultiShapeConstancy:
    """A6: the probe's core assumption — a constant input_ids[0] across all
    render modes — must hold for the ChatML template, which branches more than
    the granite-format one (thinking on/off, system/no-system, tools/no-tools).

    Uses the ChatML fixture with a minimal fake tokenizer that records the
    first token of each rendered shape and asserts they all agree.
    TestOnGraniteTokenizer above pins the concrete id (<|im_start|>)."""

    def test_input_ids_0_constant_across_chatml_shapes(self):
        from jinja2 import Environment

        with open(os.path.join(_FIXTURES, "granite_chatml_template.jinja")) as f:
            template = f.read()
        env = Environment().from_string(template)

        shapes = [
            dict(
                messages=[{"role": "user", "content": "hi"}],
                add_generation_prompt=False,
            ),
            dict(
                messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True
            ),
            dict(
                messages=[{"role": "user", "content": "hi"}],
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            dict(
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"},
                ],
                add_generation_prompt=True,
            ),
        ]
        # The rendered text of every shape must start with the same role marker
        # (ignoring leading whitespace the tokenizer strips), so input_ids[0] is
        # constant regardless of which shape the probe happens to render.
        firsts = set()
        for shape in shapes:
            rendered = env.render(**shape).lstrip("\n")
            firsts.add(rendered[: len("<|im_start|>")])
        assert firsts == {"<|im_start|>"}, (
            f"ChatML template does not emit a constant first marker across "
            f"render shapes: {firsts}"
        )


class TestOnSyntheticTokenizer:
    """Verify the probe is generic — it returns whatever the template emits,
    not a Granite-specific hardcoded token."""

    def test_custom_template_gives_custom_token(self):
        """A template whose first emission is a different marker produces
        the id of that different marker."""

        class _FakeTokenizer:
            chat_template = "<dummy-jinja>"
            unk_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                assert tokenize is False
                assert add_generation_prompt is False
                return "<BOS>hello"

            def __call__(self, text, **kwargs):
                # Pretend <BOS> tokenizes as [42], "hello" as [7, 8, 9, 10, 11].
                assert kwargs.get("add_special_tokens") is False
                assert text == "<BOS>hello"
                return SimpleNamespace(input_ids=[42, 7, 8, 9, 10, 11])

        assert _probe_lora_substitute_token_id(_FakeTokenizer()) == 42


class TestErrorPaths:
    def _minimal_tokenizer_without_template(self):
        class _T:
            chat_template = None
            unk_token_id = 0

            def apply_chat_template(self, *a, **kw):
                raise AssertionError("should not be called")

            def __call__(self, text, **kw):
                raise AssertionError("should not be called")

        return _T()

    def _tokenizer_whose_template_fails(self):
        class _T:
            chat_template = "<jinja-source>"
            unk_token_id = 0

            def apply_chat_template(self, *a, **kw):
                raise RuntimeError("template exploded")

            def __call__(self, text, **kw):
                raise AssertionError("unreachable")

        return _T()

    def _tokenizer_emitting_unk(self):
        class _T:
            chat_template = "<jinja-source>"
            unk_token_id = 777

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "mystery"

            def __call__(self, text, **kw):
                return SimpleNamespace(input_ids=[777])

        return _T()

    def _tokenizer_emitting_empty(self):
        class _T:
            chat_template = "<jinja-source>"
            unk_token_id = 0

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return ""

            def __call__(self, text, **kw):
                return SimpleNamespace(input_ids=[])

        return _T()

    def test_missing_chat_template_raises(self):
        with pytest.raises(ValueError, match="no chat_template"):
            _probe_lora_substitute_token_id(self._minimal_tokenizer_without_template())

    def test_template_render_failure_raises(self):
        with pytest.raises(ValueError, match="Failed to render a probe chat"):
            _probe_lora_substitute_token_id(self._tokenizer_whose_template_fails())

    def test_unk_first_token_raises(self):
        with pytest.raises(ValueError, match="<unk>"):
            _probe_lora_substitute_token_id(self._tokenizer_emitting_unk())

    def test_empty_tokenization_raises(self):
        with pytest.raises(ValueError, match="empty id list"):
            _probe_lora_substitute_token_id(self._tokenizer_emitting_empty())
