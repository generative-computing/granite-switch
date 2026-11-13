# SPDX-License-Identifier: Apache-2.0
"""Chat-template content-shape tests using REAL Granite tokenizers.

The mock-tokenizer tests in ``test_chat_template.py`` hand-build the parts-list
shape and render with a bare jinja ``Environment``. These tests instead load the
actual published base tokenizers — ``ibm-granite/granite-4.1-3b`` (role-marker
family) and ``ibm-granite/granite-4.2-3b`` (ChatML family) — run the composer's
``configure_chat_template`` over their real templates, and render via the
tokenizer's own ``apply_chat_template``. That exercises the real Jinja
environment, the real role markers, and the real ``enable_thinking`` / document
branches.

The focus is the content SHAPE matrix that was previously uncovered:

* plain string content (baseline, must be unchanged),
* a single text parts-list ``[{"type":"text","text":...}]`` (the OpenAI shape
  vLLM feeds once it detects the ``openai`` content format),
* multiple text parts (join behaviour),
* parts-list across multiple turns and in a system message,
* parts-list while an adapter is active (the flatten must run before the ALoRA
  Pass 2 ``rsplit``),
* parts-list on every role vLLM wraps — assistant history, assistant-with-tool-
  calls, tool, and a role outside the known branches.

Break criterion: the rendered prompt must never contain the Python list-repr
marker ``[{'type'`` — that means a parts list leaked into the prompt instead of
being flattened.

**That criterion alone is not sufficient**, and relying on it is how the
assistant-with-tool-calls case slipped through a green 2886-test GPU run: the
ChatML assistant branch gates on ``content is string``, so a parts list there is
DROPPED rather than repr'd. An absence cannot be detected by looking for a
marker. Tests covering a role must therefore also assert the text is PRESENT.

A separate class pins the *trigger*: that the composed template resolves to the
vLLM content format the template can actually honour, using a local
reimplementation of vLLM's AST-based detector (and the real vLLM detector when
it is importable).

Two sources (tests/shared/model_sources.py): ``fake`` on every PR run, the
captured tokenizer config and chat template of each base on a tiny vocabulary,
whose special tokens keep their real ids and which renders exactly like the real
tokenizer; ``real``, the published tokenizer, in the release tier. These tests
render templates and compare strings, so the fake's byte-level vocabulary for
ordinary words does not affect them.
"""

from __future__ import annotations

import os

import pytest

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer

from granite_switch.composer.tokenizer_setup import (
    configure_audio_chat_template,
    configure_chat_template,
    detect_template_format,
)
from tests.shared import model_sources

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_REPR_MARKER = "[{'type'"

# (base model, family name, adapter fixture dir, adapter name, adapter type)
# A sequence-start LoRA is used for both families: content-flattening is
# independent of adapter type, and LoRA still injects the ALoRA Pass 1 content
# loop (the vLLM 'openai' detection trigger) plus a control token. SR/aLoRA
# fixtures can't be used here — their recorded anchor token id was built against
# a tiny training tokenizer and the composer (correctly) rejects it against the
# real base tokenizer's id.
_BASES = [
    ("granite-4.1-3b", "granite_format", "summarization_adapter", "summ", "lora"),
    ("granite-4.2-3b", "chatml", "summarization_adapter", "summ", "lora"),
]
# Each base once per source. ``model_id`` is a (source, base) pair, resolved to
# a path or Hub id only when a test loads it, so collection builds nothing; the
# real case is release-marked.
_MODELS = [
    pytest.param(
        (src.values[0], base),
        family,
        adapter_dir,
        name,
        atype,
        id=f"{base}-{src.id}",
        marks=src.marks,
    )
    for base, family, adapter_dir, name, atype in _BASES
    for src in model_sources.sources()
]


def _load_tokenizer(model_id):
    # No skip on a failed load: in the release tier a fetch failure should fail
    # the run, not hide as a skip, and the fake is always available locally.
    source, base = model_id
    return AutoTokenizer.from_pretrained(model_sources.base_model(source, base))


def _configure(model_id, adapter_dir: str, name: str, atype: str, *, audio=False):
    tok = _load_tokenizer(model_id)
    configure_chat_template(tok, [(os.path.join(_FIXTURES, adapter_dir), name, atype)])
    if audio:
        configure_audio_chat_template(tok)
    return tok


def _render(tok, messages, **kwargs):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        **kwargs,
    )


def _text_part(text):
    return [{"type": "text", "text": text}]


@pytest.mark.parametrize("model_id,family,adapter_dir,name,atype", _MODELS)
class TestContentShapeRealTokenizer:
    """Both families must flatten a parts list to text, for every message shape."""

    def test_family_detected(self, model_id, family, adapter_dir, name, atype):
        tok = _load_tokenizer(model_id)
        assert detect_template_format(tok.chat_template).name == family

    def test_plain_string_unchanged(self, model_id, family, adapter_dir, name, atype):
        """A plain-string render is a baseline: text present, no list-repr. The
        no-adapter render must also byte-match the base template's output so the
        injection is inert when no adapter_name is passed."""
        base = _load_tokenizer(model_id)
        msgs = [{"role": "user", "content": "Is it raining in Paris?"}]
        before = _render(base, msgs)

        tok = _configure(model_id, adapter_dir, name, atype)
        after = _render(tok, msgs)
        assert "Is it raining in Paris?" in after
        assert _REPR_MARKER not in after
        assert after == before, "no-adapter render drifted from the base template"

    def test_single_text_part_flattened(
        self, model_id, family, adapter_dir, name, atype
    ):
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok, [{"role": "user", "content": _text_part("Is it raining in Paris?")}]
        )
        assert "Is it raining in Paris?" in out
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"

    def test_multiple_text_parts_joined(
        self, model_id, family, adapter_dir, name, atype
    ):
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "[Document doc_1]\nThe Eiffel Tower is in Paris.",
                        },
                        {"type": "text", "text": "When was it completed?"},
                    ],
                }
            ],
        )
        assert "The Eiffel Tower is in Paris." in out
        assert "When was it completed?" in out
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"

    def test_parts_in_system_and_multi_turn(
        self, model_id, family, adapter_dir, name, atype
    ):
        """Realistic mixed multi-turn: user/system turns carry parts lists (the
        shape vLLM feeds under 'openai'), assistant history stays a string (model
        output always is). Every parts-list turn must flatten, not only the last.

        Assistant content is intentionally a string: vLLM only wraps incoming
        user/system/tool content into parts lists, and the ChatML assistant
        branch runs its own <think> handling that assumes a string. A parts-list
        assistant turn is not a shape any client or engine produces.
        """
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok,
            [
                {"role": "system", "content": _text_part("You are concise.")},
                {"role": "user", "content": _text_part("first question")},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": _text_part("second question")},
            ],
        )
        for needle in (
            "You are concise.",
            "first question",
            "first answer",
            "second question",
        ):
            assert needle in out, f"{needle!r} missing from {family} render: {out!r}"
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"

    def test_parts_with_active_adapter(
        self, model_id, family, adapter_dir, name, atype
    ):
        """With an adapter active, the control token is placed AND content is
        flattened — the flatten must run before the ALoRA Pass 2 rsplit."""
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok,
            [{"role": "user", "content": _text_part("Is this answerable?")}],
            adapter_name=name,
        )
        assert "Is this answerable?" in out
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"
        assert f"<|{name}|>" in out, f"control token not placed ({family}): {out!r}"

    def test_parts_with_audio_enabled_coexist(
        self, model_id, family, adapter_dir, name, atype
    ):
        """Text flattening and the audio marker coexist: a text+audio parts list
        keeps the text, emits <|audio|>, and never leaks a list-repr."""
        tok = _configure(model_id, adapter_dir, name, atype, audio=True)
        out = _render(
            tok,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "transcribe this"},
                        {"type": "audio", "audio": "clip-placeholder"},
                    ],
                }
            ],
        )
        assert "transcribe this" in out
        assert "<|audio|>" in out
        assert "clip-placeholder" not in out
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"

    def test_tool_role_parts_flattened(
        self, model_id, family, adapter_dir, name, atype
    ):
        """A tool-role message sent as a parts list must flatten too.

        Tool content is external-data-in (like user content), so a client may
        send it structured and vLLM wraps it under 'openai'. The ChatML tool
        branch emits message.content directly, so without a flatten it leaks the
        list-repr; the 4.1 template flattens all roles natively. Both families
        must render the tool text with no repr.
        """
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok,
            [
                {"role": "user", "content": "what is the weather?"},
                {"role": "assistant", "content": "let me check"},
                {"role": "tool", "content": _text_part("sunny, 72F")},
                {"role": "user", "content": "thanks"},
            ],
        )
        assert "sunny, 72F" in out, f"tool text missing ({family}): {out!r}"
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"

    def test_tool_role_string_unchanged(
        self, model_id, family, adapter_dir, name, atype
    ):
        """Regression: a plain-string tool response still renders normally (the
        flatten only triggers on a non-string iterable)."""
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok,
            [
                {"role": "user", "content": "what is the weather?"},
                {"role": "assistant", "content": "let me check"},
                {"role": "tool", "content": "sunny, 72F"},
                {"role": "user", "content": "thanks"},
            ],
        )
        assert "sunny, 72F" in out
        assert _REPR_MARKER not in out

    def test_assistant_history_parts_flattened(
        self, model_id, family, adapter_dir, name, atype
    ):
        """An assistant turn sent as a parts list must flatten.

        Under 'openai' vLLM wraps *every* message's content, assistant history
        included — it is vLLM that produces this shape, not the client. Patching
        only the roles thought of as "incoming" (user/system/tool) left each
        earlier assistant turn rendering as [{'type': 'text', 'text': 'a1'}].
        """
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok,
            [
                {"role": "user", "content": _text_part("q1")},
                {"role": "assistant", "content": _text_part("a1-assistant-reply")},
                {"role": "user", "content": _text_part("q2")},
            ],
        )
        assert "a1-assistant-reply" in out, (
            f"assistant text missing ({family}): {out!r}"
        )
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"

    def test_assistant_text_kept_with_tool_calls(
        self, model_id, family, adapter_dir, name, atype
    ):
        """A parts-list assistant turn carrying tool calls must keep its text.

        This asserts PRESENCE, not the absence of the list-repr, which is the
        whole point: the ChatML assistant branch gates on ``content is string``,
        so a parts list fails the check and the text is DROPPED rather than
        repr'd. A no-list-repr assertion reports this case as passing while the
        text silently vanishes from the history — so only a positive assertion
        catches it.
        """
        tok = _configure(model_id, adapter_dir, name, atype)
        out = _render(
            tok,
            [
                {"role": "user", "content": _text_part("what is the weather?")},
                {
                    "role": "assistant",
                    "content": _text_part("Let me check the weather"),
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": _text_part("sunny, 72F")},
                {"role": "user", "content": _text_part("thanks")},
            ],
        )
        assert "Let me check the weather" in out, (
            f"assistant text dropped next to tool_calls ({family}): {out!r}"
        )
        assert "get_weather" in out, f"tool call missing ({family}): {out!r}"
        assert _REPR_MARKER not in out, f"list repr leaked ({family}): {out!r}"

    def test_unknown_role_parts_renders_like_string(
        self, model_id, family, adapter_dir, name, atype
    ):
        """A role outside the known branches must render, not raise.

        ChatML's catch-all ``else`` concatenates message.content straight onto a
        string, so a parts list there raised TypeError before the normalize pass
        ran. Base 4.2 only escapes this because vLLM detects 'string' for it and
        flattens upstream — composing removes that protection, so this is a
        composed-only regression.

        Asserted as equality against the equivalent plain-string render rather
        than as text presence, because what a template DOES with an unknown role
        differs by family: 4.2 has a catch-all that emits it, 4.1 has none and
        drops it (true of a plain string there too, so not a bug). Equality is
        the family-agnostic claim and the stronger one — flattening must make a
        parts list behave exactly like the string it carries, whatever that is.
        """
        tok = _configure(model_id, adapter_dir, name, atype)
        as_parts = _render(
            tok,
            [
                {"role": "user", "content": _text_part("summarize")},
                {"role": "document", "content": _text_part("doc-body-text")},
            ],
        )
        as_string = _render(
            tok,
            [
                {"role": "user", "content": "summarize"},
                {"role": "document", "content": "doc-body-text"},
            ],
        )
        assert as_parts == as_string, (
            f"parts-list render diverged from the string render ({family}):\n"
            f"parts={as_parts!r}\nstring={as_string!r}"
        )
        assert _REPR_MARKER not in as_parts, (
            f"list repr leaked ({family}): {as_parts!r}"
        )


def test_chatml_injections_raise_on_missing_anchor():
    """Every ChatML injection must fail loudly when its anchor is absent.

    Two of the three originally returned the template unchanged, so a base-template
    revision that moved an anchor would compose GREEN and then render wrongly at
    serve time. A composer cannot no-op a correctness fix: the checkpoint is
    written and published before anyone renders a prompt.
    """
    from granite_switch.composer.tokenizer_setup import (
        _inject_chatml_content_flatten,
        _inject_chatml_system_message_flatten,
    )

    anchorless = "{{- '<|im_start|>user\\n' }}{{ messages }}"
    with pytest.raises(ValueError, match="ChatML message loop"):
        _inject_chatml_content_flatten(anchorless, marker=None)
    with pytest.raises(ValueError, match="leading-system-message"):
        _inject_chatml_system_message_flatten(anchorless)


# ────────────────────────────────────────────────────────────────────
# Detector-level: the composed template must resolve to a content format
# the template can actually honour.
# ────────────────────────────────────────────────────────────────────


def _vllm_style_detect(chat_template: str) -> str:
    """Reimplementation of vLLM's content-format auto-detector.

    vLLM (`vllm/renderers/hf.py:_detect_content_format`) parses the template to a
    Jinja AST and returns 'openai' iff some ``for`` loop iterates a message's
    ``.content`` (``for x in message.content`` / ``message['content']``), else
    'string'. We reproduce that rule here so the test does not require vLLM
    installed; a vLLM-backed variant below cross-checks it when importable.
    """
    import jinja2

    try:
        import transformers.utils.chat_template_utils as hf_cu

        ast = hf_cu._compile_jinja_template(chat_template).environment.parse(
            chat_template
        )
    except Exception:
        ast = jinja2.Environment().parse(chat_template)

    message_varnames = {"message", "m", "msg", "_msg"}

    def is_content_access(node) -> bool:
        # message.content
        if isinstance(node, jinja2.nodes.Getattr):
            return (
                node.attr == "content"
                and isinstance(node.node, jinja2.nodes.Name)
                and node.node.name in message_varnames
            )
        # message['content']
        if isinstance(node, jinja2.nodes.Getitem):
            key = node.arg
            return (
                isinstance(key, jinja2.nodes.Const)
                and key.value == "content"
                and isinstance(node.node, jinja2.nodes.Name)
                and node.node.name in message_varnames
            )
        return False

    for loop in ast.find_all(jinja2.nodes.For):
        if is_content_access(loop.iter):
            return "openai"
    return "string"


@pytest.mark.parametrize("model_id,family,adapter_dir,name,atype", _MODELS)
class TestContentFormatDetection:
    """Pin the actual trigger: what vLLM auto-detects for base vs composed."""

    def test_base_template_detection(self, model_id, family, adapter_dir, name, atype):
        """4.1 base already loops over content parts (-> openai); 4.2 base does
        not (-> string). This is the asymmetry that makes the bug 4.2-only."""
        tok = _load_tokenizer(model_id)
        detected = _vllm_style_detect(tok.chat_template)
        expected = "openai" if family == "granite_format" else "string"
        assert detected == expected, (
            f"{family} base detected {detected!r}, expected {expected!r}"
        )

    def test_composed_template_honours_detection(
        self, model_id, family, adapter_dir, name, atype
    ):
        """After composing, the template must be able to honour whatever vLLM
        detects: it either stays 'string', or if it flips to 'openai' it now
        carries a flatten block so a parts list renders cleanly. Either way, a
        parts-list render must not leak a list-repr."""
        tok = _configure(model_id, adapter_dir, name, atype)
        detected = _vllm_style_detect(tok.chat_template)
        out = _render(tok, [{"role": "user", "content": _text_part("hello")}])
        assert _REPR_MARKER not in out, (
            f"{family} composed template detected {detected!r} but a parts list "
            f"still rendered as a repr: {out!r}"
        )

    def test_matches_real_vllm_detector(
        self, model_id, family, adapter_dir, name, atype
    ):
        """Cross-check our local detector against the real vLLM one when vLLM is
        installed (skipped otherwise)."""
        hf_renderers = pytest.importorskip("vllm.renderers.hf")
        tok = _configure(model_id, adapter_dir, name, atype)
        mine = _vllm_style_detect(tok.chat_template)
        theirs = hf_renderers._detect_content_format(
            tok.chat_template, default="string"
        )
        assert mine == theirs
