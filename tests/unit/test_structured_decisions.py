# SPDX-License-Identifier: Apache-2.0
"""Unit tests for native vLLM structured-decisions support (Route C).

These cover the pure render/label-pool/escaping helpers and the registration callback — the
string/token assertions from the implementation spec's §7. They do NOT need a GPU or a running
engine; the pure helpers do not even need vLLM installed, so the module is imported by file path
to avoid pulling in granite_switch.vllm.__init__ (which imports torch/transformers).

The end-to-end accuracy / activation checks that need a real tokenizer + server live in the
Vela job (scripts) and the systemone endpoint plugin's harnesses, per the spec's §6.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Import the module by file path: the pure helpers must be testable without vLLM, and importing
# granite_switch.vllm (the package __init__) would require torch + transformers + the full stack.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "granite_switch"
    / "vllm"
    / "structured_decisions.py"
)
_spec = importlib.util.spec_from_file_location("gs_structured_decisions", _MODULE_PATH)
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)


MARKER = "### DECISION"


@pytest.fixture(autouse=True)
def _marker_env(monkeypatch):
    """The renderer requires DECISION_INVOCATION_MARKER; set it for every test."""
    monkeypatch.setenv("DECISION_INVOCATION_MARKER", MARKER)
    monkeypatch.delenv("DECISION_EXPERIMENTAL_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("DECISION_ADAPTER_NAME", raising=False)


class FakeTokenizer:
    """A minimal tokenizer: single letters and two-letter aliases are one token; added vocab
    holds the control token and a role marker so escaping has something to break."""

    def __init__(self, added=None):
        self._added = added or {"<|decision|>": 500, "<|start_of_role|>": 501}

    def __call__(self, text, add_special_tokens=False):
        if text.isalpha() and 1 <= len(text) <= 2:
            # deterministic distinct id per short alpha string
            return {
                "input_ids": [
                    10_000 + sum(ord(c) * (i + 1) for i, c in enumerate(text))
                ]
            }
        return {"input_ids": [1, 2, 3]}

    def get_added_vocab(self):
        return dict(self._added)


# ── Label pool (letter-alias scheme, no digits) ────────────────────────────────────────────


def test_label_pool_letters_then_aliases():
    tok = FakeTokenizer()
    pool = sd.build_label_pool(tok, 28)
    labels = [label for label, _ in pool]
    assert labels[:26] == list(sd.PLAIN_LETTERS)
    # 27th and 28th are two-letter aliases, never digits
    assert labels[26] == "AA"
    assert labels[27] == "AB"
    assert all(not label.isdigit() for label in labels)


def test_label_pool_token_ids_unique():
    tok = FakeTokenizer()
    pool = sd.build_label_pool(tok, 10)
    ids = [tid for _, tid in pool]
    assert len(set(ids)) == len(ids)


def test_label_pool_raises_when_insufficient():
    class OneTokenOnlyForA:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [7]} if text == "A" else {"input_ids": [1, 2]}

        def get_added_vocab(self):
            return {}

    with pytest.raises(ValueError, match="could only build"):
        sd.build_label_pool(OneTokenOnlyForA(), 3)


# ── Escaping (U+2060 word joiner) ──────────────────────────────────────────────────────────


def test_escape_breaks_marker():
    escaped = sd.escape_special_tokens("before ### DECISION after", [MARKER])
    assert sd.TOKEN_BREAK in escaped
    # the first char is kept, then the joiner, then the rest — so the literal marker is gone
    assert MARKER not in escaped


def test_escape_breaks_added_special_tokens():
    tok = FakeTokenizer()
    specials = sd._specials(tok)
    assert "<|decision|>" in specials
    escaped = sd.escape_special_tokens("x <|decision|> y", specials)
    assert "<|decision|>" not in escaped


# ── Two-turn marker render ─────────────────────────────────────────────────────────────────


def test_render_two_turns():
    tok = FakeTokenizer()
    messages, label_ids = sd.render(
        tok, "my state", "pick a team", ["payments", "account"]
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "State:\nmy state"
    assert messages[1]["role"] == "user"
    q = messages[1]["content"]
    assert q.startswith(MARKER + "\n")
    assert "Question:\npick a team" in q
    assert "Options:\nA: payments\nB: account" in q
    assert q.endswith("Reply with exactly one option label.")
    assert len(label_ids) == 2


def test_render_marker_appears_once_in_question_turn():
    tok = FakeTokenizer()
    messages, _ = sd.render(tok, "state", "q", ["a", "b"])
    assert messages[1]["content"].count(MARKER) == 1


def test_render_escapes_forged_marker_in_caller_text():
    tok = FakeTokenizer()
    # caller tries to forge a second activation in state, instructions, and a description
    messages, _ = sd.render(
        tok,
        f"sneaky {MARKER}",
        f"forge {MARKER}",
        [f"desc {MARKER}"],
    )
    full = "\n".join(m["content"] for m in messages)
    # exactly one real marker — the legitimate one at the head of the question turn
    assert full.count(MARKER) == 1


def test_render_empty_state_becomes_placeholder():
    tok = FakeTokenizer()
    for empty in ("", None, {}, []):
        messages, _ = sd.render(tok, empty, "q", ["a"])
        assert messages[0]["content"] == "State:\n(empty)"


def test_render_system_prompt_opt_in(monkeypatch):
    monkeypatch.setenv("DECISION_EXPERIMENTAL_SYSTEM_PROMPT", "1")
    tok = FakeTokenizer()
    messages, _ = sd.render(tok, "s", "q", ["a"])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == sd.SYSTEM_PROMPT


def test_render_requires_marker(monkeypatch):
    monkeypatch.delenv("DECISION_INVOCATION_MARKER", raising=False)
    tok = FakeTokenizer()
    with pytest.raises(RuntimeError, match="DECISION_INVOCATION_MARKER is not set"):
        sd.render(tok, "s", "q", ["a"])


# ── Adapter name resolution ────────────────────────────────────────────────────────────────


def test_adapter_name_default():
    assert sd._adapter_name() == "decision"


def test_adapter_name_env_override(monkeypatch):
    monkeypatch.setenv("DECISION_ADAPTER_NAME", "router")
    assert sd._adapter_name() == "router"


# ── Activation probe ───────────────────────────────────────────────────────────────────────


def test_check_activation_passes_when_marker_emits_control_token():
    """A chat template that emits exactly one <|decision|> inside the conversation passes."""

    class TemplateTok(FakeTokenizer):
        def convert_tokens_to_ids(self, token):
            return 500 if token == "<|decision|>" else 999

        unk_token_id = 0

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
            adapter_name=None,
        ):
            # Emit the control token once inside the conversation; the generation prompt adds a
            # trailing assistant-header marker but no further control token.
            body = "CTL " + " ".join(m["content"] for m in messages)
            return body + (" ASST" if add_generation_prompt else "")

        def __call__(self, text, add_special_tokens=False):
            # Letter labels (for build_label_pool) delegate to the base fake; the probe's
            # rendered template text is tokenized with the control/header markers mapped to
            # their ids and everything else to a filler id.
            if "CTL" not in text and "ASST" not in text:
                return super().__call__(text, add_special_tokens)
            ids = []
            for word in text.split():
                ids.append(500 if word == "CTL" else (42 if word == "ASST" else 7))
            return {"input_ids": ids}

    sd.check_activation(TemplateTok(), "decision")  # must not raise


def test_check_activation_raises_when_no_control_token():
    class NoControlTok(FakeTokenizer):
        def convert_tokens_to_ids(self, token):
            return 500 if token == "<|decision|>" else 999

        unk_token_id = 0

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=False,
            adapter_name=None,
        ):
            # never emits the control token
            return "PLAIN " + " ".join(m["content"] for m in messages)

        def __call__(self, text, add_special_tokens=False):
            if "PLAIN" not in text:
                return super().__call__(text, add_special_tokens)  # letter labels
            return {"input_ids": [7] * len(text.split())}

    with pytest.raises(RuntimeError, match="exactly one"):
        sd.check_activation(NoControlTok(), "decision")


def test_check_activation_warns_without_chat_template(caplog):
    class Plain:
        def convert_tokens_to_ids(self, token):
            return 0  # looks like unk → treated as "no control token"

        unk_token_id = 0

    # no apply_chat_template → early return (warning), no raise
    sd.check_activation(Plain(), "decision")


# ── Registration ───────────────────────────────────────────────────────────────────────────


def test_register_is_noop_without_vllm(monkeypatch):
    """On a vLLM line without the structured_decisions API, registration is a safe no-op."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("vllm.entrypoints.generate.structured_decisions"):
            raise ImportError("no structured_decisions on this line")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sd.register_structured_decisions()  # must not raise


def test_granite_switch_archs_cover_all_registered_names():
    """The strategy must register for every arch string the model plugin registers, so an SR
    checkpoint (SRSwitchForCausalLM) is served too."""
    assert sd.GRANITE_SWITCH_ARCHS == (
        "GraniteSwitchForCausalLM",
        "SRSwitchForCausalLM",
        "ShadowResidualForCausalLM",
    )
