# SPDX-License-Identifier: Apache-2.0
"""The vLLM ``/v1/systemone`` decision strategy, on CPU, against a minimal fake of
vLLM 0.32's structured-decisions modules (tests/shared/decision_fixtures.py).

A toy template stands in for the composed Granite one: it activates ``<|decision|>``
on the last user turn holding the marker, else just before the assistant header. The
strategy is the real class, so these tests pin what it adds over vLLM:

- the decision config comes from ``hf_config`` (no env vars, no constants);
- a declared adapter that does not activate stops the server (RuntimeError, which
  vLLM's ``except ValueError`` around strategy construction does not swallow);
- no declared adapter starts fine and refuses reads with a clear error;
- the rendered prompt follows the spec, all caller text escaped, and every real
  render is checked for exactly one control token;
- a reordered (noul) question maps label probabilities back to the right options;
- caller ``chat_template_kwargs`` are allowlisted; media is refused.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from granite_switch.config import VLLM_ARCHITECTURES
from tests.shared.decision_fixtures import (
    LABEL_READS,
    MARKER,
    STRATEGIES,
    FakeStructuredDecisionError,
    build_strategy,
    install_fake_vllm,
    load_strategy_module,
    question,
    reference_block,
    run_read,
)

sd = load_strategy_module()

CTL, CTL_ID, HEADER_ID, FILLER_ID = "<|decision|>", 500, 42, 7


class ToyTokenizer:
    """Words are tokens. Single letters get their own ids, ``<|decision|>`` is
    added vocab, ``ASSISTANT`` is the generation prompt's last added token."""

    unk_token_id = 0

    def __init__(self, invocation_text=MARKER, double_activation=False):
        self.invocation_text = invocation_text
        self.double_activation = double_activation
        self.rendered = []

    def get_added_vocab(self):
        return {CTL: CTL_ID, "ASSISTANT": HEADER_ID}

    def convert_tokens_to_ids(self, token):
        return self.get_added_vocab().get(token, self.unk_token_id)

    def _word_ids(self, word):
        if word in self.get_added_vocab():
            return [self.get_added_vocab()[word]]
        if len(word) == 1 and word.isalpha():
            return [1000 + ord(word)]
        return [FILLER_ID, FILLER_ID]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [i for w in text.split() for i in self._word_ids(w)]}

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        adapter_name=None,
        **kw,
    ):
        contents = [m["content"] for m in messages]
        fallback = False
        if adapter_name == "decision":
            hits = [
                i
                for i, m in enumerate(messages)
                if m["role"] == "user" and self.invocation_text in m["content"]
            ]
            if hits:
                i = hits[-1]
                contents[i] = contents[i].replace(
                    self.invocation_text, f"{CTL} {self.invocation_text}"
                )
            else:
                fallback = add_generation_prompt
        text = " ".join(f"ROLE {m['role']} {c}" for m, c in zip(messages, contents))
        if fallback:
            text += f" {CTL}"
        if add_generation_prompt:
            text += " ASSISTANT"
        if self.double_activation and len(self.rendered) > 1:
            text = f"{CTL} {text}"  # breaks only renders after the startup probe
        self.rendered.append(text)
        return self(text)["input_ids"] if tokenize else text


def _letter(c):
    return 1000 + ord(c)


def _hf_config(specs=None, default="decision"):
    return SimpleNamespace(
        adapter_names=["guardian", "decision"],
        adapter_kinds=["lora", "lora"],
        decision_adapter=default if specs else None,
        decision_specs=specs,
    )


def _strategy(monkeypatch, specs, winners=(), tok=None, default="decision"):
    tok = tok or ToyTokenizer()
    modules = install_fake_vllm(monkeypatch, tok, winners)
    return build_strategy(sd, _hf_config(specs, default)), modules, tok


def _specs(**overrides):
    return {"decision": reference_block(**overrides)}


def _seen(modules):
    return modules[LABEL_READS].seen


# ── Startup ──────────────────────────────────────────────────────────────────────────


def test_startup_reads_the_decision_config_and_probes_activation(monkeypatch):
    strategy, _, tok = _strategy(monkeypatch, _specs())
    assert strategy.default_adapter == "decision"
    assert strategy.specs["decision"]["invocation_marker"] == MARKER
    assert strategy.tails["decision"] == []  # nothing after the ASSISTANT header
    assert any(CTL in text for text in tok.rendered)


def test_startup_fails_when_the_marker_does_not_activate(monkeypatch):
    """RuntimeError, not ValueError: vLLM builds the strategy under
    ``except ValueError`` and would only disable the route at DEBUG level."""
    tok = ToyTokenizer(invocation_text="### SOMETHING-ELSE")
    with pytest.raises(RuntimeError, match="inside the question turn") as info:
        _strategy(monkeypatch, _specs(), tok=tok)
    assert not isinstance(info.value, ValueError)


def test_startup_without_a_decision_adapter_serves_a_clear_error(monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        strategy, _, _ = _strategy(monkeypatch, None)
    assert "declares no decision adapter" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert strategy.limits().max_options == 26
    with pytest.raises(FakeStructuredDecisionError, match="--decision-adapter"):
        run_read(strategy, [question("q", "choice", [("a", None)])])


def test_spec_for_an_undeclared_adapter_is_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="not an adapter of this checkpoint"):
        _strategy(monkeypatch, {"other": reference_block()}, default="other")


def test_limits_report_the_label_pool(monkeypatch):
    strategy, _, _ = _strategy(monkeypatch, _specs(labels=["Y", "N"]))
    assert strategy.limits().max_options == 2


# ── Reads ────────────────────────────────────────────────────────────────────────────


def test_choice_read_renders_the_spec_and_forces_the_adapter(monkeypatch):
    strategy, modules, _ = _strategy(monkeypatch, _specs(), winners=[_letter("B")])
    (read,) = run_read(
        strategy,
        [
            question(
                "queue", "choice", [("payments", "Payout failures"), ("account", None)]
            )
        ],
    )
    assert read.probs[1] > 0.9 and read.argmax_is_label
    assert read.label_mass == pytest.approx(0.95)
    (engine_input,) = _seen(modules)
    assert engine_input["request"]["chat_template_kwargs"] == {
        "adapter_name": "decision",
        "enable_thinking": False,
    }
    question_turn = engine_input["messages"][1]["content"]
    assert question_turn.startswith(MARKER + "\nQuestion:\npick one")
    assert "A: Payout failures\nB: account" in question_turn


def test_noul_read_maps_the_false_first_labels_back(monkeypatch):
    # Rendered false-first: A = "no", B = "yes". The engine favours A.
    strategy, modules, _ = _strategy(monkeypatch, _specs(), winners=[_letter("A")])
    (read,) = run_read(
        strategy, [question("urgent", "noul", [("yes", None), ("no", None)])]
    )
    # vLLM's answer is option order (yes, no): the mass must land on "no".
    assert read.probs[1] > 0.9
    (engine_input,) = _seen(modules)
    assert "A: No / false\nB: Yes / true" in engine_input["messages"][1]["content"]


def test_a_real_render_with_two_control_tokens_is_refused(monkeypatch):
    tok = ToyTokenizer(double_activation=True)
    strategy, _, _ = _strategy(monkeypatch, _specs(), tok=tok)
    with pytest.raises(FakeStructuredDecisionError, match="2 <\\|decision\\|> control"):
        run_read(strategy, [question("q", "choice", [("a", None), ("b", None)])])


def test_media_is_refused(monkeypatch):
    strategy, _, _ = _strategy(monkeypatch, _specs())
    with pytest.raises(FakeStructuredDecisionError, match="read text"):
        run_read(
            strategy,
            [question("q", "choice", [("a", None)])],
            media={"image": ["data:..."]},
        )


def test_state_renders_json_on_one_line_and_empty_as_the_placeholder(monkeypatch):
    strategy, _, _ = _strategy(monkeypatch, _specs())
    assert strategy.render_state({"a": 1}) == '{"a": 1}'
    assert strategy.render_state({}) == ""
    strategy, modules, _ = _strategy(monkeypatch, _specs(), winners=[_letter("A")])
    run_read(strategy, [question("q", "choice", [("a", None)])], state="")
    (engine_input,) = _seen(modules)
    assert engine_input["messages"][0]["content"] == "State:\n(empty)"


def test_caller_text_is_escaped(monkeypatch):
    strategy, modules, _ = _strategy(monkeypatch, _specs(), winners=[_letter("A")])
    run_read(
        strategy,
        [question("q", "choice", [("a", f"{CTL} {MARKER}")], instructions=MARKER)],
        state=f"{MARKER} {CTL}",
    )
    (engine_input,) = _seen(modules)
    full = "\n".join(m["content"] for m in engine_input["messages"])
    assert full.count(MARKER) == 1 and CTL not in full


def test_request_instructions_are_refused_by_the_reference_spec(monkeypatch):
    strategy, _, _ = _strategy(monkeypatch, _specs())
    with pytest.raises(FakeStructuredDecisionError, match="request-level"):
        run_read(
            strategy, [question("q", "choice", [("a", None)])], instructions="terse"
        )


def test_unlisted_question_type_is_a_request_error(monkeypatch):
    strategy, _, _ = _strategy(monkeypatch, _specs(question_types={"choice": {}}))
    with pytest.raises(FakeStructuredDecisionError, match="does not answer 'score'"):
        run_read(strategy, [question("q", "score", [("low", None), ("high", None)])])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"documents": []}, "not supported"),
        ({"enable_thinking": True}, "thinking must be off"),
        ({"adapter_name": "guardian"}, "not a decision adapter"),
    ],
)
def test_caller_chat_template_kwargs_are_allowlisted(monkeypatch, kwargs, match):
    strategy, _, _ = _strategy(monkeypatch, _specs())
    with pytest.raises(FakeStructuredDecisionError, match=match):
        run_read(strategy, [question("q", "choice", [("a", None)])], kwargs=kwargs)


def test_caller_may_name_the_declared_adapter(monkeypatch):
    strategy, _, _ = _strategy(monkeypatch, _specs(), winners=[_letter("A")])
    (read,) = run_read(
        strategy,
        [question("q", "choice", [("a", None), ("b", None)])],
        kwargs={"adapter_name": "decision", "enable_thinking": False},
    )
    assert read.probs[0] > 0.9


# ── Registration ─────────────────────────────────────────────────────────────────────


def test_register_inserts_the_strategy_for_every_architecture(monkeypatch):
    modules = install_fake_vllm(monkeypatch, ToyTokenizer())
    sd.register_structured_decisions()
    registered = modules[STRATEGIES].READ_STRATEGIES
    assert set(registered) == set(VLLM_ARCHITECTURES)
    assert registered["GraniteSwitchForCausalLM"].__name__ == (
        "GraniteSwitchDecisionStrategy"
    )


def _without_strategies(monkeypatch, version):
    import sys

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=version))
    monkeypatch.setitem(sys.modules, STRATEGIES, None)  # import raises


def test_register_is_a_quiet_noop_before_0_32(monkeypatch, caplog):
    _without_strategies(monkeypatch, "0.30.1")
    with caplog.at_level(logging.WARNING):
        sd.register_structured_decisions()
    assert "NOT registered" not in caplog.text


def test_register_warns_when_a_newer_vllm_moved_the_api(monkeypatch, caplog):
    _without_strategies(monkeypatch, "0.33.0")
    with caplog.at_level(logging.WARNING):
        sd.register_structured_decisions()
    assert "NOT registered" in caplog.text
