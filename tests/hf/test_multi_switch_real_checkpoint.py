# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch routing on REAL composed granite-4.1-3b checkpoints (HF backend).

Every other HF MultiSwitch test uses a mock config and hand-built token ids. This
module composes real checkpoints and drives the full
``GraniteSwitchForCausalLM`` forward / ``generate``, asserting
``model.model._last_adapter_indices`` against latest-wins routing:

  compose (-> config.json)
      -> from_pretrained -> create_switch -> MultiSwitch
      -> model.forward / generate -> _last_adapter_indices

Three checkpoints, each composed once under ``$GRANITE_SWITCH_E2E_DIR/<name>`` and
warm-reused when its ``config.json`` exists:

    multi        LoRA-only (guardian + core): raw multi-transition routing, buffer
                 integrity, generate, chat-template placement, real multi-turn,
                 long prompts
    multi-alora  aLoRA-only (rag): base-before / adapter-after placement, two
                 aLoRAs in one sequence
    multi-mixed  no technology filter (rag + guardian): per-technology placement
                 in one checkpoint, re-save stability

The LoRA/aLoRA difference is entirely WHERE the chat template puts the control
token, and that changes the routing shape:

    LoRA:    <tok> user1 ... assistant1 ... user2 ... [gen]
    routing:   1     1   1       1      1     1   1    1      adapter throughout

    aLoRA:   user1 ... assistant1 ... user2 <tok> ... [gen]
    routing:   0    0      0      0     0     1   1    1      base, then adapter

Heavy: marked slow / requires_model / gpu / release, and skipped unless
``GRANITE_SWITCH_E2E_MODELS=1`` is set (composing downloads the base model +
adapters).
"""

import gc
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
    # Real composed checkpoint; the release suite sets GRANITE_SWITCH_E2E_MODELS=1
    # so the gate below opens.
    pytest.mark.release,
    pytest.mark.skipif(
        os.environ.get("GRANITE_SWITCH_E2E_MODELS") != "1",
        reason="composes real ~3B checkpoints; set GRANITE_SWITCH_E2E_MODELS=1",
    ),
]

BASE_MODEL = "ibm-granite/granite-4.1-3b"

# Where composed checkpoints are cached. Override with GRANITE_SWITCH_E2E_DIR.
_E2E_ROOT = Path(os.environ.get("GRANITE_SWITCH_E2E_DIR", "/tmp/granite_switch_e2e"))

# checkpoint name -> (adapter repos, technology filter or None)
_CHECKPOINTS = {
    "multi": (
        ["ibm-granite/granitelib-guardian-r1.0", "ibm-granite/granitelib-core-r1.0"],
        "lora",
    ),
    "multi-alora": (["ibm-granite/granitelib-rag-r1.0"], "alora"),
    # rag carries aLoRA intrinsics; guardian carries LoRA ones. Deliberately no
    # filter: that is the composer's default.
    "multi-mixed": (
        ["ibm-granite/granitelib-rag-r1.0", "ibm-granite/granitelib-guardian-r1.0"],
        None,
    ),
}


def _compose(name):
    """Compose (or warm-reuse) checkpoint ``name``; return its directory.

    Also used by tests/integration/test_multi_switch_vllm_generate.py so both
    suites share one artifact.
    """
    repos, tech_filter = _CHECKPOINTS[name]
    out_dir = _E2E_ROOT / name
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
        # One flag, all repos: --adapters is nargs="*" with the default store
        # action, so a repeated flag keeps only the LAST list.
        "--adapters",
        *repos,
    ]
    if tech_filter is not None:
        cmd += ["--technology-filter", tech_filter]
    cmd += ["--output", str(out_dir)]
    print(f"composing ({name}):", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        pytest.fail(
            f"{name} compose failed (exit {r.returncode})\n"
            f"--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}",
            pytrace=False,
        )
    assert (out_dir / "config.json").exists(), "compose produced no config.json"
    return out_dir


# TEXT: any non-control id. 50 is far below the appended control ids.
TEXT = 50


# The one checkpoint currently loaded, {name: (model, config, tokenizer)}.
_loaded = {}


def _load(name):
    """(model, config, tokenizer) for a composed checkpoint, asserting MultiSwitch.

    Holds ONE ~3B model at a time: loading a different checkpoint frees the
    previous one. The classes below are grouped by checkpoint, so in file order
    each loads once.
    """
    if name in _loaded:
        return _loaded[name]
    _loaded.clear()
    gc.collect()

    from transformers import AutoTokenizer

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM
    from granite_switch.hf.switch import MultiSwitch

    out_dir = _compose(name)
    config = GraniteSwitchConfig.from_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = GraniteSwitchForCausalLM.from_pretrained(out_dir).eval()
    assert isinstance(model.model.switch, MultiSwitch), (
        f"from_pretrained built {type(model.model.switch).__name__}, "
        f"expected MultiSwitch"
    )
    print(
        f"\n  {name} checkpoint: {config.num_adapters} adapters: "
        f"{list(getattr(config, 'adapter_names', []) or [])}",
        file=sys.stderr,
    )
    _loaded[name] = (model, config, tok)
    return _loaded[name]


# Function-scoped on purpose: the model is cached in _load, and a module-scoped
# fixture would keep all three alive until the module ends.
@pytest.fixture
def lora_model():
    return _load("multi")


@pytest.fixture
def alora_model():
    return _load("multi-alora")


@pytest.fixture
def mixed_model():
    return _load("multi-mixed")


def _control_ids(config):
    """(A_TOK, B_TOK): control tokens for adapters 1 and 2.

    Real composed configs use the no-base-slot layout: adapter_token_ids[k]
    fires adapter k+1.
    """
    atoks = config.adapter_token_ids
    assert len(atoks) >= 2, f"need >= 2 adapters, got {len(atoks)}"
    return atoks[0], atoks[1]


def _route(model, ids):
    """Run one forward over ``ids``; return the per-position adapter indices."""
    import torch

    with torch.no_grad():
        model(input_ids=torch.tensor([ids]))
    return model.model._last_adapter_indices[0].tolist()


def _render(tok, question, adapter_name):
    """Render a chat prompt with the adapter requested by name.

    The kwarg is ``adapter_name``. Jinja treats an unknown kwarg as
    simply-not-defined, so a wrong name raises nothing and renders no control
    token -- which is why the tests below assert a control token was rendered.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        adapter_name=adapter_name,
    )


def _render_ids(tok, question, adapter_name):
    return tok(_render(tok, question, adapter_name), add_special_tokens=False)[
        "input_ids"
    ]


def _ctrl_positions(ids, atoks):
    """Indices in ``ids`` holding any adapter control token."""
    ctrl = set(int(t) for t in atoks)
    return [i for i, t in enumerate(ids) if int(t) in ctrl]


def _template_types(tok):
    """Parse {adapter_name: technology} out of the template's adapter_map."""
    tmpl = tok.chat_template or ""
    return dict(
        re.findall(r"'([^']+)':\s*\{'token':\s*'[^']*',\s*'type':\s*'([^']+)'", tmpl)
    )


def _assert_buffers_valid(switch):
    """Codebook is unit-norm and nonzero; substitute LUT is present and nonzero."""
    import torch

    cb = switch.codebook.float()
    assert not torch.all(cb == 0), (
        "checkpoint loaded an ALL-ZERO codebook -- the memory softmax will "
        "average adapters instead of selecting one"
    )
    norms = cb.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)
    lut = switch.control_to_substitute_lut
    assert lut is not None, "checkpoint has no substitute LUT"
    assert not torch.all(lut == 0), (
        "substitute LUT loaded ALL ZEROS -- token exchange would rewrite every "
        "token id to 0 (the 'not a control' sentinel is -1)"
    )
    return lut


# ── lora: raw multi-transition routing ────────────────────────────────────────


class TestMultiTransition:
    """Full-model routing on hand-built ids matches latest-wins."""

    def test_base_then_A_then_B(self, lora_model):
        """[T,A,T,B,T] -> [0,1,1,2,2]."""
        model, config, _ = lora_model
        a, b = _control_ids(config)
        assert _route(model, [TEXT, a, TEXT, b, TEXT]) == [0, 1, 1, 2, 2]

    def test_higher_then_lower(self, lora_model):
        """[B,T,A,T] -> [2,2,1,1]: the most recent token wins, not the larger."""
        model, config, _ = lora_model
        a, b = _control_ids(config)
        assert _route(model, [b, TEXT, a, TEXT]) == [2, 2, 1, 1]

    def test_sticky_single_transition(self, lora_model):
        """One control token persists through a long text tail."""
        model, config, _ = lora_model
        a, _b = _control_ids(config)
        result = _route(model, [TEXT, TEXT, a] + [TEXT] * 8)
        assert result[:2] == [0, 0]
        assert all(v == 1 for v in result[2:])

    def test_all_base(self, lora_model):
        """No control tokens -> every position is base (0)."""
        model, _config, _ = lora_model
        assert all(v == 0 for v in _route(model, [TEXT] * 12))

    def test_control_at_position_zero(self, lora_model):
        """Control token at pos 0 -> adapter active from the start."""
        model, config, _ = lora_model
        a, _b = _control_ids(config)
        assert all(v == 1 for v in _route(model, [a, TEXT, TEXT, TEXT]))

    def test_three_transitions(self, lora_model):
        """A -> B -> A latest-wins across three transitions."""
        model, config, _ = lora_model
        a, b = _control_ids(config)
        result = _route(model, [TEXT, a, TEXT, b, TEXT, a, TEXT])
        assert result == [0, 1, 1, 2, 2, 1, 1]


# ── lora: paths only a real checkpoint reaches ────────────────────────────────


class TestRealCheckpointGenerate:
    """Bugs that survived every mock-config test:

    1. ``codebook`` / ``control_to_substitute_lut`` registered
       ``persistent=False``, so ``from_pretrained`` returned them as ALL ZEROS.
       Only visible after a real save/load.
    2. The switch built its causal mask as ``q_len x q_len`` while its heads
       attend over the whole KV cache, so the first decode step raised. Only
       visible when actually generating.
    """

    def test_buffers_are_valid(self, lora_model):
        model, config, _ = lora_model
        lut = _assert_buffers_valid(model.model.switch)
        for ctrl, sub in zip(
            config.adapter_token_ids, config.adapter_substitute_token_ids
        ):
            assert int(lut[ctrl]) == int(sub), (
                f"LUT maps control {ctrl} -> {int(lut[ctrl])}, expected {sub}"
            )

    def test_generates(self, lora_model):
        import torch

        model, config, _ = lora_model
        a, _b = _control_ids(config)
        ids = torch.tensor([[TEXT, TEXT, a, TEXT]])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=5, do_sample=False)
        assert out.shape[1] == ids.shape[1] + 5, (
            f"generate produced {out.shape[1]} tokens, expected {ids.shape[1] + 5}"
        )

    def test_generate_carries_adapter(self, lora_model):
        """A control token in the prompt keeps its adapter active through decode."""
        import torch

        model, config, _ = lora_model
        _a, b = _control_ids(config)
        with torch.no_grad():
            model.generate(
                torch.tensor([[TEXT, b, TEXT]]), max_new_tokens=4, do_sample=False
            )
        # The final forward is a decode step; routing must still be adapter 2.
        route = model.model._last_adapter_indices[0].tolist()
        assert all(v == 2 for v in route), (
            f"decode routing {route} lost the adapter set by the control token"
        )


# ── lora: production-shaped inputs ────────────────────────────────────────────


class TestChatTemplatePlacement:
    """Wherever the template puts a control token, routing must be base before it
    and that adapter at and after it."""

    def test_template_emits_control_tokens(self, lora_model):
        """Non-vacuity: otherwise the tests below pass on an all-base sequence."""
        _model, config, tok = lora_model
        name = next(iter(config.adapter_names))
        rendered = _render(tok, "What is a hash function?", name)
        ids = tok(rendered, add_special_tokens=False)["input_ids"]
        map_keys = re.findall(r"'([^']+)':\s*\{'token'", tok.chat_template or "")
        assert _ctrl_positions(ids, config.adapter_token_ids), (
            "the composed chat template emitted NO control token.\n"
            f"  adapter_name passed : {name!r}\n"
            f"  template adapter_map keys: {map_keys}\n"
            f"  rendered: {rendered[:300]!r}"
        )

    def test_routing_follows_template_placement(self, lora_model):
        model, config, tok = lora_model
        name = next(iter(config.adapter_names))
        ids = _render_ids(tok, "Explain hashing in one sentence.", name)
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        if not pos:
            pytest.skip("template emitted no control token (see test above)")
        route = _route(model, ids)
        first = pos[0]
        assert all(v == 0 for v in route[:first]), (
            f"positions before the first control token ({first}) must be base, "
            f"got {route[:first]}"
        )
        assert route[first] != 0, (
            f"the control token's own position {first} routed to base"
        )
        assert route[-1] != 0, f"the generation position routed to base: {route[-8:]}"

    def test_template_prompt_generates(self, lora_model):
        import torch

        model, config, tok = lora_model
        name = next(iter(config.adapter_names))
        ids = torch.tensor([_render_ids(tok, "Name one use of a hash function.", name)])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        # > rather than == : generate() may stop early on EOS.
        assert out.shape[1] > ids.shape[1], "template prompt generated nothing"


class TestRealMultiTurn:
    """Generate, append the model's own reply, switch adapter, go again."""

    def test_two_turns_different_adapters(self, lora_model):
        """Turn 1's adapter holds through its turn; turn 2's wins at the tail."""
        import torch

        model, config, tok = lora_model
        a, b = _control_ids(config)

        turn1 = tok("Question: what is entropy?", add_special_tokens=False)[
            "input_ids"
        ] + [a]
        ids1 = torch.tensor([turn1])
        with torch.no_grad():
            out1 = model.generate(ids1, max_new_tokens=10, do_sample=False)
        reply1 = out1[0][ids1.shape[1] :].tolist()
        assert reply1, "turn 1 generated nothing"

        turn2 = (
            turn1
            + reply1
            + tok(" Question: and what is enthalpy?", add_special_tokens=False)[
                "input_ids"
            ]
            + [b]
        )
        route = _route(model, turn2)
        i_a, i_b = turn2.index(a), turn2.index(b)
        assert route[i_a] != 0 and route[i_b] != 0, (
            f"a control token routed to base: {i_a}->{route[i_a]}, {i_b}->{route[i_b]}"
        )
        assert route[i_b] != route[i_a], (
            f"turn 2's control token did not change the adapter ({route[i_a]})"
        )
        assert all(v == route[i_a] for v in route[i_a:i_b]), (
            f"turn 1's adapter did not hold through its own turn: {route[i_a:i_b]}"
        )
        assert all(v == route[i_b] for v in route[i_b:]), (
            f"turn 2's adapter did not hold to the end: {route[i_b:]}"
        )

    def test_second_turn_generates(self, lora_model):
        """The second turn generates too -- decode over a longer cache."""
        import torch

        model, config, tok = lora_model
        a, b = _control_ids(config)
        t1 = tok("Briefly: what is a graph?", add_special_tokens=False)["input_ids"] + [
            a
        ]
        with torch.no_grad():
            o1 = model.generate(torch.tensor([t1]), max_new_tokens=8, do_sample=False)
        t2 = (
            o1[0].tolist()
            + tok(" And a tree?", add_special_tokens=False)["input_ids"]
            + [b]
        )
        with torch.no_grad():
            o2 = model.generate(torch.tensor([t2]), max_new_tokens=8, do_sample=False)
        # generate() may stop early on EOS, so assert SOME new tokens, not exactly 8.
        assert o2.shape[1] > len(t2), "second turn generated nothing"


class TestLongPrompt:
    """The count is recovered from an anchor at position 0; check it still inverts
    exactly when the control tokens sit thousands of tokens behind it."""

    @pytest.mark.parametrize("length", [1024, 4096])
    def test_long_sequence_routing(self, lora_model, length):
        model, config, _ = lora_model
        a, b = _control_ids(config)
        i_a, i_b = length // 2, length - 20
        ids = [TEXT] * length
        ids[i_a] = a
        ids[i_b] = b
        route = _route(model, ids)
        assert all(v == 0 for v in route[:i_a]), (
            f"len={length}: base expected before {i_a}; saw {sorted(set(route[:i_a]))}"
        )
        assert all(v == route[i_a] for v in route[i_a:i_b]), (
            f"len={length}: A did not hold {i_a}..{i_b}; saw {sorted(set(route[i_a:i_b]))}"
        )
        assert all(v == route[i_b] for v in route[i_b:]), (
            f"len={length}: B did not hold from {i_b}; saw {sorted(set(route[i_b:]))}"
        )
        assert route[i_a] != route[i_b] and route[i_a] != 0, (
            f"len={length}: controls did not select distinct adapters "
            f"(A={route[i_a]}, B={route[i_b]})"
        )

    def test_many_transitions(self, lora_model):
        """20 transitions in one sequence -- stresses the counting head."""
        model, config, _ = lora_model
        a, b = _control_ids(config)
        seq, expected = [], []
        for k in range(20):
            ctrl, cur = (a, 1) if k % 2 == 0 else (b, 2)
            seq += [ctrl] + [TEXT] * 30
            expected += [cur] * 31
        route = _route(model, seq)
        wrong = [
            (i, route[i], expected[i])
            for i in range(len(seq))
            if route[i] != expected[i]
        ]
        assert not wrong, (
            f"{len(wrong)} of {len(seq)} positions mis-routed across 20 transitions; "
            f"first few (index, got, want): {wrong[:6]}"
        )


# ── alora: placement-driven routing ───────────────────────────────────────────


class TestALoRA:
    """For aLoRA the return to base for earlier turns is achieved by PLACEMENT."""

    def test_adapters_present(self, alora_model):
        """Non-vacuity: --technology-filter alora yielded adapters."""
        _model, config, tok = alora_model
        assert config.num_adapters >= 1, (
            f"alora compose produced num_adapters={config.num_adapters}"
        )
        assert len(config.adapter_token_ids) == config.num_adapters, (
            f"{len(config.adapter_token_ids)} control tokens for "
            f"{config.num_adapters} adapters"
        )
        # Without invocation_text the template falls back to the generation-prompt
        # path, which is a different routing shape.
        assert "invocation_text" in (tok.chat_template or ""), (
            "the composed chat template has no invocation_text entries"
        )

    def test_base_before_adapter_after(self, alora_model):
        model, config, tok = alora_model
        name = config.adapter_names[0]
        ids = _render_ids(tok, "Is this answerable from the context?", name)
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        assert pos, f"no control token in the rendered aLoRA prompt for {name!r}"
        first = pos[0]
        assert first > 0, (
            "the aLoRA control token landed at index 0 (LoRA's placement); the "
            "base-before assertion would be vacuous"
        )
        route = _route(model, ids)
        assert all(v == 0 for v in route[:first]), (
            f"aLoRA must be BASE before {first}; got {sorted(set(route[:first]))}"
        )
        assert route[first] != 0 and all(v == route[first] for v in route[first:]), (
            f"the adapter must hold from {first} to the end; "
            f"got {sorted(set(route[first:]))}"
        )

    def test_each_adapter_routes_to_its_own_index(self, alora_model):
        model, config, tok = alora_model
        seen = {}
        for name in config.adapter_names[:3]:
            ids = _render_ids(tok, "Is this answerable?", name)
            pos = _ctrl_positions(ids, config.adapter_token_ids)
            if pos:
                seen[name] = _route(model, ids)[pos[0]]
        assert len(seen) >= 2, f"needed >=2 renderable aLoRA adapters, got {seen}"
        assert len(set(seen.values())) == len(seen), (
            f"different aLoRA adapters routed to the SAME expert id: {seen}"
        )

    def test_generates_with_adapter_at_decode(self, alora_model):
        import torch

        model, config, tok = alora_model
        ids = torch.tensor(
            [
                _render_ids(
                    tok, "Is this answerable from the context?", config.adapter_names[0]
                )
            ]
        )
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        assert out.shape[1] > ids.shape[1], "aLoRA prompt generated nothing"
        route = model.model._last_adapter_indices[0].tolist()
        assert route[-1] != 0, f"the adapter was lost during decode; tail {route[-6:]}"


class TestTwoALoRAsInOneSequence:
    """Two aLoRA adapters in ONE sequence -- the shape MultiSwitch exists for.

    Built by concatenating two per-adapter renders (what an agentic loop or a
    replayed history yields), so placement is still the template's.
    """

    @staticmethod
    def _two_turn_ids(config, tok):
        names = config.adapter_names
        if len(names) < 2:
            pytest.skip(f"needs >=2 alora adapters, got {names}")
        joined = _render_ids(tok, "Is this answerable from the context?", names[0])
        joined += _render_ids(tok, "How certain are you about that?", names[1])
        return joined, _ctrl_positions(joined, config.adapter_token_ids)

    @pytest.fixture
    def two_turn(self, alora_model):
        _model, config, tok = alora_model
        joined, pos = self._two_turn_ids(config, tok)
        if len(pos) < 2:
            pytest.skip(f"only {len(pos)} control token(s) rendered")
        return joined, pos

    def test_two_alora_tokens_present(self, alora_model):
        """Non-vacuity: with fewer than two tokens no transition is exercised."""
        _model, config, tok = alora_model
        _joined, pos = self._two_turn_ids(config, tok)
        assert len(pos) >= 2, (
            f"expected two aLoRA control tokens in the concatenated sequence, "
            f"found {len(pos)} at {pos} -- the latest-wins test would be vacuous"
        )

    def test_latest_wins_between_two_aloras(self, alora_model, two_turn):
        model, _config, _tok = alora_model
        joined, pos = two_turn
        route = _route(model, joined)
        i_a, i_b = pos[0], pos[1]
        exp_a, exp_b = route[i_a], route[i_b]
        assert all(v == 0 for v in route[:i_a]), (
            f"base expected before {i_a}; got {sorted(set(route[:i_a]))}"
        )
        assert exp_a != 0 and exp_b != 0 and exp_a != exp_b, (
            f"expected two distinct adapters: pos{i_a}->{exp_a}, pos{i_b}->{exp_b}"
        )
        assert all(v == exp_a for v in route[i_a:i_b]), (
            f"A ({exp_a}) must hold {i_a}..{i_b}; got {sorted(set(route[i_a:i_b]))}"
        )
        assert all(v == exp_b for v in route[i_b:]), (
            f"B ({exp_b}) must hold from {i_b}; got {sorted(set(route[i_b:]))}"
        )

    def test_two_alora_sequence_generates(self, alora_model, two_turn):
        import torch

        model, _config, _tok = alora_model
        joined, _pos = two_turn
        ids = torch.tensor([joined])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        assert out.shape[1] > ids.shape[1], "two-adapter sequence generated nothing"
        route = model.model._last_adapter_indices[0].tolist()
        assert route[-1] != 0, (
            f"the second adapter was lost during decode; {route[-6:]}"
        )


# ── mixed: per-technology placement in one checkpoint ─────────────────────────


class TestMixed:
    """With no --technology-filter, each adapter must keep its own technology's
    placement and routing shape."""

    def test_unfiltered_compose_prefers_alora(self, mixed_model):
        """Discovery prefers aLoRA for dual-published adapters, by design.

        Pinned because a silent flip to LoRA preference would change the routing
        SHAPE of every unfiltered checkpoint without changing any adapter name.
        """
        _model, _config, tok = mixed_model
        kinds = set(_template_types(tok).values())
        assert "alora" in kinds, (
            f"an unfiltered compose should prefer aLoRA for dual-published "
            f"adapters; got {kinds}"
        )

    def test_one_control_token_per_adapter(self, mixed_model):
        _model, config, _tok = mixed_model
        atoks = config.adapter_token_ids
        assert len(atoks) == config.num_adapters, (
            f"{len(atoks)} control tokens for {config.num_adapters} adapters"
        )
        assert len(set(atoks)) == len(atoks), "duplicate control token ids"

    def test_lora_adapter_routes_from_sequence_start(self, mixed_model):
        model, config, tok = mixed_model
        lora = [n for n, t in _template_types(tok).items() if t == "lora"]
        if not lora:
            pytest.skip("no lora-typed adapter in this checkpoint")
        ids = _render_ids(tok, "Summarize the context.", lora[0])
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        assert pos and pos[0] == 0, (
            f"a LoRA control token must land at index 0; got {pos}"
        )
        route = _route(model, ids)
        assert route[0] != 0 and all(v == route[0] for v in route), (
            f"LoRA routing must be one adapter throughout; got {sorted(set(route))}"
        )

    def test_alora_adapter_routes_base_then_adapter(self, mixed_model):
        model, config, tok = mixed_model
        alora = [n for n, t in _template_types(tok).items() if t == "alora"]
        if not alora:
            pytest.skip("no alora-typed adapter in this checkpoint")
        ids = _render_ids(tok, "Is this answerable from the context?", alora[0])
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        assert pos and pos[0] > 0, (
            f"an aLoRA control token must not be at index 0; got {pos}"
        )
        route = _route(model, ids)
        first = pos[0]
        assert all(v == 0 for v in route[:first]), (
            f"aLoRA must be base before {first}; got {sorted(set(route[:first]))}"
        )
        assert route[first] != 0 and all(v == route[first] for v in route[first:]), (
            f"aLoRA adapter must hold from {first}; got {sorted(set(route[first:]))}"
        )

    def test_lora_and_alora_select_different_experts(self, mixed_model):
        model, config, tok = mixed_model
        types = _template_types(tok)
        lora = [n for n, t in types.items() if t == "lora"]
        alora = [n for n, t in types.items() if t == "alora"]
        if not lora or not alora:
            pytest.skip(f"need both technologies; got lora={lora} alora={alora}")
        got = {}
        for name in (lora[0], alora[0]):
            ids = _render_ids(tok, "Is this answerable?", name)
            pos = _ctrl_positions(ids, config.adapter_token_ids)
            if pos:
                got[name] = _route(model, ids)[pos[0]]
        if len(got) < 2:
            pytest.skip(f"could not render both technologies: {got}")
        assert len(set(got.values())) == 2, (
            f"a LoRA and an aLoRA adapter routed to the SAME expert id: {got}"
        )

    def test_generates_for_each_technology(self, mixed_model):
        import torch

        model, _config, tok = mixed_model
        for kind in ("lora", "alora"):
            names = [n for n, t in _template_types(tok).items() if t == kind]
            if not names:
                continue
            ids = torch.tensor([_render_ids(tok, "Is this answerable?", names[0])])
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=8, do_sample=False)
            assert out.shape[1] > ids.shape[1], f"{kind} adapter generated nothing"

    def test_resave_preserves_buffers_and_routing(self, mixed_model, tmp_path):
        """A second save/load round trip on the real artifact keeps the switch
        buffers and the routing unchanged."""
        from granite_switch.hf import GraniteSwitchForCausalLM

        model, _config, tok = mixed_model
        name = next(iter(_template_types(tok)))
        ids = _render_ids(tok, "Is this answerable?", name)
        route_before = _route(model, ids)

        out = tmp_path / "resaved"
        model.save_pretrained(out)
        tok.save_pretrained(out)
        reloaded = GraniteSwitchForCausalLM.from_pretrained(out).eval()
        _assert_buffers_valid(reloaded.model.switch)
        route_after = _route(reloaded, ids)
        assert route_after == route_before, (
            "routing changed after a save/load round trip:\n"
            f"  before: {route_before}\n  after : {route_after}"
        )
