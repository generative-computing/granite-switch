# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch with aLoRA adapters, end to end on a real composed checkpoint.

The switch itself does not know about adapter technology -- it matches control
token ids and routes. The LoRA/aLoRA difference is entirely WHERE the chat template
puts the control token, and that changes the routing shape completely:

    LoRA:    <tok> user1 ... assistant1 ... user2 ... [gen]
    routing:   1     1   1       1      1     1   1    1      adapter throughout

    aLoRA:   user1 ... assistant1 ... user2 <tok> ... [gen]
    routing:   0    0      0      0     0     1   1    1      base, then adapter

So for aLoRA the "return to base" for earlier turns is achieved by PLACEMENT, not
by a base-reset token -- which is why ordinary aLoRA chat needs no such token.

Coverage gap this closes: every other real-checkpoint multiswitch test composes
with ``--technology-filter lora`` (tests/hf/test_multi_switch_e2e.py) or with no
filter at all. Nothing composed an aLoRA-only checkpoint and checked that an
aLoRA-placed control token yields base-before / adapter-after routing on a real
model. The template's placement logic is unit-tested in
tests/composer/test_chat_template.py with no model attached, so the two halves --
placement and routing -- were never joined. That is the same shape as all three
bugs fixed on this branch: a layer correct in isolation, never crossed end to end.
aLoRA is the technology whose entire semantics depend on that placement.

Heavy: composes a real granite-4.1-3b with --technology-filter alora. Gated on
GRANITE_SWITCH_E2E_MODELS=1 like the other real-checkpoint suites.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("GRANITE_SWITCH_E2E_MODELS") != "1",
        reason="composes a real ~3B alora checkpoint; set GRANITE_SWITCH_E2E_MODELS=1",
    ),
]

BASE_MODEL = "ibm-granite/granite-4.1-3b"
# granitelib-rag carries aLoRA intrinsics (answerability, citations, ...); the
# guardian/core libraries used by the LoRA e2e are filtered to lora there.
ADAPTER_REPOS = ["ibm-granite/granitelib-rag-r1.0"]

_E2E_ROOT = Path(os.environ.get("GRANITE_SWITCH_E2E_DIR", "/tmp/granite_switch_e2e"))


def _compose_alora():
    """Compose (or warm-reuse) an aLoRA-only multi checkpoint."""
    out_dir = _E2E_ROOT / "multi-alora"
    if (out_dir / "config.json").exists():
        print(f"warm-reuse {out_dir}", file=sys.stderr)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        BASE_MODEL,
        *[arg for r in ADAPTER_REPOS for arg in ("--adapters", r)],
        "--technology-filter",
        "alora",
        "--switch-type",
        "multi",
        "--output",
        str(out_dir),
    ]
    print("composing:", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        pytest.fail(
            f"alora compose failed (exit {r.returncode})\n"
            f"--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}",
            pytrace=False,
        )
    return out_dir


@pytest.fixture(scope="module")
def alora_model():
    """(model, config, tokenizer, adapter_names) for an aLoRA-only checkpoint."""
    from transformers import AutoTokenizer

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM
    from granite_switch.hf.switch import MultiSwitch

    out_dir = _compose_alora()
    config = GraniteSwitchConfig.from_pretrained(out_dir)
    assert config.switch_type == "multi", (
        f"composed switch_type={config.switch_type!r}, expected 'multi'"
    )
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = GraniteSwitchForCausalLM.from_pretrained(out_dir).eval()
    assert isinstance(model.model.switch, MultiSwitch)
    names = list(getattr(config, "adapter_names", []) or [])
    assert names, "composed config has no adapter_names"
    print(f"\n  alora checkpoint: {len(names)} adapters: {names}", file=sys.stderr)
    return model, config, tok, names


def _ctrl_positions(ids, atoks):
    ctrl = set(int(t) for t in atoks)
    return [i for i, t in enumerate(ids) if int(t) in ctrl]


def _render(tok, question, adapter_name):
    """Render with the adapter requested by name (the kwarg is adapter_name)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        adapter_name=adapter_name,
    )


class TestALoRAComposition:
    """The compose step must actually produce aLoRA adapters, not silently none."""

    def test_alora_adapters_present(self, alora_model):
        """--technology-filter alora yielded at least one adapter.

        Non-vacuity guard: if the filter matched nothing the compose would produce
        zero adapters and every routing assertion below would be meaningless.
        """
        _model, config, _tok, names = alora_model
        assert config.num_adapters >= 1, (
            f"alora compose produced num_adapters={config.num_adapters}; the "
            "technology filter matched nothing, so the routing tests would be vacuous"
        )
        assert len(config.adapter_token_ids) == config.num_adapters, (
            "expected one control token per adapter (no base-reset slot); got "
            f"{len(config.adapter_token_ids)} tokens for {config.num_adapters} adapters"
        )

    def test_template_carries_alora_invocation_text(self, alora_model):
        """The template must know each aLoRA adapter's invocation text.

        aLoRA placement works by locating the invocation sequence in the user
        message; without invocation_text in the adapter_map the template falls back
        to the generation-prompt path, which is a different routing shape.
        """
        _model, _config, tok, names = alora_model
        template = tok.chat_template or ""
        assert "invocation_text" in template, (
            "the composed chat template has no invocation_text entries, so no "
            "adapter was treated as aLoRA -- placement would use the LoRA/fallback path"
        )


class TestALoRARouting:
    """aLoRA routing shape: base before the control token, adapter at and after."""

    def test_base_before_adapter_after(self, alora_model):
        """The defining aLoRA property, on a real model.

        This is what no existing test covered: the template places the control
        token, and the switch must then route base for everything before it and the
        adapter from it onward.
        """
        import torch

        model, config, tok, names = alora_model
        rendered = _render(tok, "Is this answerable from the context?", names[0])
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        assert pos, (
            f"no control token in the rendered aLoRA prompt for {names[0]!r}; "
            f"rendered:\n{rendered[:400]!r}"
        )

        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
        route = model.model._last_adapter_indices[0].tolist()

        first = pos[0]
        assert all(v == 0 for v in route[:first]), (
            f"aLoRA must be BASE before the control token at {first}; "
            f"got {sorted(set(route[:first]))} over positions 0..{first - 1}"
        )
        assert route[first] != 0, (
            f"the control token's own position {first} routed to base"
        )
        assert all(v == route[first] for v in route[first:]), (
            f"the adapter must hold from {first} to the end; got "
            f"{sorted(set(route[first:]))}"
        )

    def test_control_token_is_not_at_sequence_start(self, alora_model):
        """aLoRA placement must differ from LoRA's sequence-start placement.

        If the token landed at index 0 the routing shape would be LoRA's (adapter
        throughout) and the test above would pass while proving nothing about aLoRA.
        """
        _model, config, tok, names = alora_model
        ids = tok(
            _render(tok, "Is this answerable?", names[0]), add_special_tokens=False
        )["input_ids"]
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        assert pos, "no control token rendered"
        assert pos[0] > 0, (
            "the aLoRA control token landed at index 0, which is the LoRA "
            "sequence-start placement -- the base-before-adapter assertion above "
            "would then be vacuous"
        )

    def test_each_adapter_routes_to_its_own_index(self, alora_model):
        """Different aLoRA adapters must select different expert ids."""
        import torch

        model, config, tok, names = alora_model
        seen = {}
        for name in names[:3]:
            ids = tok(
                _render(tok, "Is this answerable?", name), add_special_tokens=False
            )["input_ids"]
            pos = _ctrl_positions(ids, config.adapter_token_ids)
            if not pos:
                continue
            with torch.no_grad():
                model(input_ids=torch.tensor([ids]))
            seen[name] = model.model._last_adapter_indices[0].tolist()[pos[0]]
        assert len(seen) >= 2, f"needed >=2 renderable aLoRA adapters, got {seen}"
        assert len(set(seen.values())) == len(seen), (
            f"different aLoRA adapters routed to the SAME expert id: {seen}"
        )

    def test_alora_prompt_generates(self, alora_model):
        """An aLoRA prompt must generate, with the adapter still active at decode."""
        import torch

        model, config, tok, names = alora_model
        ids_list = tok(
            _render(tok, "Is this answerable from the context?", names[0]),
            add_special_tokens=False,
        )["input_ids"]
        ids = torch.tensor([ids_list])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        assert out.shape[1] > ids.shape[1], "aLoRA prompt generated nothing"
        route = model.model._last_adapter_indices[0].tolist()
        assert route[-1] != 0, (
            f"the adapter was lost during decode; routing tail {route[-6:]}"
        )
        print(
            f"\n  alora generated: "
            f"{tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()[:110]!r}",
            file=sys.stderr,
        )


class TestTwoALoRAAdaptersInOneSequence:
    """Two aLoRA adapters active in ONE sequence -- the case MultiSwitch exists for.

    The template renders one adapter per call, so nothing above puts two aLoRA
    control tokens in the same sequence. But that is precisely the shape MultiSwitch
    was built for and the shape SingleSwitch cannot do: SingleSwitch's +/-gain
    attention AVERAGES competing control tokens, so two adapters in one sequence
    mis-route. MultiSwitch takes the most recent exactly (latest-wins).

    Real flows that produce it: an agentic loop that judges the same context twice
    (answerability, then certainty), or a multi-turn conversation whose earlier
    turns are replayed verbatim with their control tokens still in the text.

    Built by concatenating two per-adapter renders, which is what such a flow
    yields -- not by hand-assembling ids, so the placement is still the template's.
    """

    def _two_turn_ids(self, tok, config, names):
        """Render turn 1 with names[0] and turn 2 with names[1]; concatenate."""
        r1 = _render(tok, "Is this answerable from the context?", names[0])
        r2 = _render(tok, "How certain are you about that?", names[1])
        ids1 = tok(r1, add_special_tokens=False)["input_ids"]
        ids2 = tok(r2, add_special_tokens=False)["input_ids"]
        joined = ids1 + ids2
        pos = _ctrl_positions(joined, config.adapter_token_ids)
        return joined, pos, len(ids1)

    def test_two_alora_tokens_present(self, alora_model):
        """Non-vacuity: the concatenation really carries TWO control tokens.

        If it carried one (or zero) the latest-wins assertion below would pass
        without ever exercising a transition.
        """
        _model, config, tok, names = alora_model
        if len(names) < 2:
            pytest.skip(f"needs >=2 alora adapters, got {names}")
        _joined, pos, _split = self._two_turn_ids(tok, config, names)
        assert len(pos) >= 2, (
            f"expected two aLoRA control tokens in the concatenated sequence, "
            f"found {len(pos)} at {pos} -- the latest-wins test would be vacuous"
        )

    def test_latest_wins_between_two_aloras(self, alora_model):
        """Adapter A holds until B's token, then B holds to the end.

        This is the property SingleSwitch gets wrong by averaging and MultiSwitch
        gets right, now on a real composed aLoRA checkpoint rather than hand-built
        token ids.
        """
        import torch

        model, config, tok, names = alora_model
        if len(names) < 2:
            pytest.skip(f"needs >=2 alora adapters, got {names}")
        joined, pos, _split = self._two_turn_ids(tok, config, names)
        if len(pos) < 2:
            pytest.skip(f"only {len(pos)} control token(s) rendered")

        with torch.no_grad():
            model(input_ids=torch.tensor([joined]))
        route = model.model._last_adapter_indices[0].tolist()

        i_a, i_b = pos[0], pos[1]
        exp_a, exp_b = route[i_a], route[i_b]

        assert all(v == 0 for v in route[:i_a]), (
            f"base expected before the first aLoRA token at {i_a}; "
            f"got {sorted(set(route[:i_a]))}"
        )
        assert exp_a != 0 and exp_b != 0, (
            f"a control token routed to base: pos{i_a}->{exp_a}, pos{i_b}->{exp_b}"
        )
        assert exp_a != exp_b, (
            f"both aLoRA tokens selected the SAME expert ({exp_a}); the second "
            "adapter did not take over, so latest-wins is not being exercised"
        )
        assert all(v == exp_a for v in route[i_a:i_b]), (
            f"adapter A ({exp_a}) must hold from {i_a} to {i_b}; "
            f"got {sorted(set(route[i_a:i_b]))}"
        )
        assert all(v == exp_b for v in route[i_b:]), (
            f"adapter B ({exp_b}) must hold from {i_b} to the end; "
            f"got {sorted(set(route[i_b:]))}"
        )

    def test_two_alora_sequence_generates(self, alora_model):
        """The two-adapter sequence must generate, with B still active at decode."""
        import torch

        model, config, tok, names = alora_model
        if len(names) < 2:
            pytest.skip(f"needs >=2 alora adapters, got {names}")
        joined, pos, _split = self._two_turn_ids(tok, config, names)
        if len(pos) < 2:
            pytest.skip(f"only {len(pos)} control token(s) rendered")
        ids = torch.tensor([joined])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        assert out.shape[1] > ids.shape[1], "two-adapter sequence generated nothing"
        route = model.model._last_adapter_indices[0].tolist()
        assert route[-1] != 0, (
            f"the second adapter was lost during decode; tail {route[-6:]}"
        )
