# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch on inputs shaped the way production actually shapes them.

Every other multiswitch test hand-builds token-id lists like
``[TEXT, A_TOK, TEXT, B_TOK]``. Nothing in production does that. Real callers:

  * build the prompt with ``apply_chat_template``, which decides WHERE control
    tokens go from the adapter's technology (per CLAUDE.md: aLoRA tokens land in
    the user message or immediately before the generation prompt; LoRA tokens at
    the sequence start). The switch's correctness depends on that placement, and
    the placement logic is a separate layer with its own rules -- so a template
    that emits a control token somewhere the switch mishandles would pass every
    hand-built test;
  * hold a MULTI-TURN conversation, appending the model's own generated reply and
    then asking again with a different adapter. That is the motivating use case
    (agentic per-step switching) and it reuses the KV cache across turns;
  * send prompts long enough that vLLM/HF split the prefill. The counting anchor
    lives at position 0, so a later chunk can contain no anchor at all -- the same
    structural hazard as the batching bug.

These are the three untested seams closest to real deployment -- exactly the kind
of boundary where code correct in isolation goes wrong the first time something
real drives it.

Uses the module-scoped real composed checkpoint from ``test_multi_switch_e2e``, so
it costs no extra compose. Gated on ``GRANITE_SWITCH_E2E_MODELS=1``.
"""

import os

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("GRANITE_SWITCH_E2E_MODELS") != "1",
        reason="needs a real composed checkpoint; set GRANITE_SWITCH_E2E_MODELS=1",
    ),
]

# Reuse the e2e module's fixture: same session, same composed artifact.
from tests.hf.test_multi_switch_e2e import composed_model  # noqa: F401


@pytest.fixture(scope="module")
def tokenizer(composed_model):  # noqa: F811
    from transformers import AutoTokenizer

    _model, config, _ = composed_model
    # The composed dir is where the config came from; re-derive it the same way
    # the e2e module does rather than reaching into fixture internals.
    from tests.hf.test_multi_switch_e2e import _compose

    return AutoTokenizer.from_pretrained(_compose("multi"))


def _render(tokenizer, question, adapter_name):
    """Render a chat prompt the way a real caller does.

    The template does NOT auto-inject a control token for a plain user turn --
    verified on the composed checkpoint, which rendered
    '<|start_of_role|>user<|end_of_role|>...<|start_of_role|>assistant<|end_of_role|>'
    with no control token at all. The adapter must be requested explicitly.

    The kwarg is ``adapter_name``, not ``intrinsic_name``. See
    ``composer/tokenizer_setup.py::configure_chat_template``, which emits
    ``{%- if adapter_name is defined and adapter_name in adapter_map %}``. Jinja
    treats an unknown kwarg as simply-not-defined, so passing the wrong name
    raises nothing, renders no control token, and every routing assertion
    downstream becomes vacuous -- which is exactly what the GPU run caught.
    Only ``test_template_emits_control_tokens`` stood between that typo and a
    green suite over an all-base sequence.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        adapter_name=adapter_name,
    )


def _control_positions(ids, atoks):
    """Indices in ``ids`` holding any adapter control token."""
    ctrl = set(int(t) for t in atoks)
    return [i for i, t in enumerate(ids) if int(t) in ctrl]


# ── 1. chat template places the control tokens ────────────────────────────────


class TestChatTemplatePlacement:
    """The rendered chat template must produce routing the switch handles.

    Not asserting a specific placement -- that is the template's business and it
    differs by adapter technology. Asserting the CONTRACT: wherever the template
    puts a control token, routing must be base before it and that adapter at and
    after it, and the generation position must end up on a real adapter.
    """

    def test_template_emits_control_tokens(self, composed_model, tokenizer):  # noqa: F811
        """Sanity: the composed template actually injects control tokens.

        If it does not, the tests below would pass vacuously on an all-base
        sequence -- the same trap as a dead switch being trivially batch-invariant.
        """
        _model, config, _ = composed_model
        name = next(iter(config.adapter_names))
        rendered = _render(tokenizer, "What is a hash function?", name)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        pos = _control_positions(ids, config.adapter_token_ids)
        # Report enough to identify WHY on failure. Two runs have now failed here,
        # first with the wrong kwarg name (intrinsic_name) and then with
        # adapter_name -- so the remaining suspect is the NAME not being a key in
        # the template's adapter_map. Print the map keys and the name used rather
        # than guessing a third time.
        import re as _re

        tmpl = tokenizer.chat_template or ""
        map_keys = _re.findall(r"'([^']+)':\s*\{'token'", tmpl)
        assert pos, (
            "the composed chat template emitted NO control token, so every "
            "placement assertion below would be vacuous.\n"
            f"  adapter_name passed : {name!r}\n"
            f"  config.adapter_names: {list(config.adapter_names)}\n"
            f"  template adapter_map keys: {map_keys}\n"
            f"  name in map? {name in map_keys}\n"
            f"  rendered: {rendered[:300]!r}"
        )
        print(f"\n  control tokens at positions {pos} of {len(ids)}")

    def test_routing_follows_template_placement(self, composed_model, tokenizer):  # noqa: F811
        """Routing must be base before the first control token, adapter after."""
        import torch

        model, config, _ = composed_model
        name = next(iter(config.adapter_names))
        rendered = _render(tokenizer, "Explain hashing in one sentence.", name)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        pos = _control_positions(ids, config.adapter_token_ids)
        if not pos:
            pytest.skip("template emitted no control token (see test above)")

        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
        route = model.model._last_adapter_indices[0].tolist()

        first = pos[0]
        assert all(v == 0 for v in route[:first]), (
            f"positions before the first control token ({first}) must be base, "
            f"got {route[:first]}"
        )
        assert route[first] != 0, (
            f"the control token's own position {first} routed to base ({route[first]}); "
            "the adapter must be active at the control token itself"
        )
        # And the last position -- where generation continues from -- must be on
        # an adapter, since that is what the template arranged for.
        assert route[-1] != 0, (
            f"the generation position routed to base; full routing tail: {route[-8:]}"
        )

    def test_template_prompt_generates(self, composed_model, tokenizer):  # noqa: F811
        """A template-built prompt must generate, not just forward."""
        import torch

        model, _config, _ = composed_model
        name = next(iter(_config.adapter_names))
        rendered = _render(tokenizer, "Name one use of a hash function.", name)
        ids = torch.tensor([tokenizer(rendered, add_special_tokens=False)["input_ids"]])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8, do_sample=False)
        # > rather than == : generate() may stop early on EOS.
        assert out.shape[1] > ids.shape[1], (
            f"template prompt generated nothing: {out.shape[1]} <= {ids.shape[1]}"
        )
        text = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        print(f"\n  generated from template prompt: {text.strip()[:120]!r}")


# ── 2. multi-turn with the model's OWN reply appended ─────────────────────────


class TestRealMultiTurn:
    """Generate, append the reply, switch adapter, generate again.

    The existing 'multi-turn' e2e test is one synthetic sequence. This is the real
    shape: turn 1 output becomes turn 2 input, so the second forward sees a prompt
    containing generated tokens and a second control token further along.
    """

    def test_two_turns_different_adapters(self, composed_model, tokenizer):  # noqa: F811
        """Turn 2's adapter must win at the tail, and turn 1's must hold before it."""
        import torch

        model, config, _ = composed_model
        atoks = list(config.adapter_token_ids)
        a, b = atoks[0], atoks[1]

        # Turn 1: prompt + adapter A, generate a real continuation.
        turn1 = tokenizer("Question: what is entropy?", add_special_tokens=False)[
            "input_ids"
        ] + [a]
        ids1 = torch.tensor([turn1])
        with torch.no_grad():
            out1 = model.generate(ids1, max_new_tokens=10, do_sample=False)
        reply1 = out1[0][ids1.shape[1] :].tolist()
        assert reply1, "turn 1 generated nothing"

        # Turn 2: everything so far + a NEW question + adapter B.
        turn2 = (
            turn1
            + reply1
            + tokenizer(" Question: and what is enthalpy?", add_special_tokens=False)[
                "input_ids"
            ]
            + [b]
        )
        ids2 = torch.tensor([turn2])
        with torch.no_grad():
            model(input_ids=ids2)
        route = model.model._last_adapter_indices[0].tolist()

        i_a, i_b = turn2.index(a), turn2.index(b)
        assert route[i_a] != 0 and route[i_b] != 0, (
            f"a control token routed to base: pos {i_a}->{route[i_a]}, "
            f"pos {i_b}->{route[i_b]}"
        )
        assert route[i_b] != route[i_a], (
            f"turn 2's control token ({b}) did not change the adapter: "
            f"both turns routed to {route[i_a]}"
        )
        assert all(v == route[i_b] for v in route[i_b:]), (
            f"turn 2's adapter did not hold to the end: {route[i_b:]}"
        )
        # Latest-wins across the boundary: A holds from i_a until i_b.
        assert all(v == route[i_a] for v in route[i_a:i_b]), (
            f"turn 1's adapter did not hold through its own turn: {route[i_a:i_b]}"
        )

    def test_second_turn_generates(self, composed_model, tokenizer):  # noqa: F811
        """The second turn must also generate -- decode over a longer cache."""
        import torch

        model, config, _ = composed_model
        a, b = list(config.adapter_token_ids)[:2]
        t1 = tokenizer("Briefly: what is a graph?", add_special_tokens=False)[
            "input_ids"
        ] + [a]
        with torch.no_grad():
            o1 = model.generate(torch.tensor([t1]), max_new_tokens=8, do_sample=False)
        t2 = (
            o1[0].tolist()
            + tokenizer(" And a tree?", add_special_tokens=False)["input_ids"]
            + [b]
        )
        with torch.no_grad():
            o2 = model.generate(torch.tensor([t2]), max_new_tokens=8, do_sample=False)
        # generate() may stop early on EOS, so assert it produced SOME new tokens
        # rather than exactly 8. An earlier version asserted == len(t2) + 8 and
        # failed at 29 vs 30 because generation ended on EOS -- a bug in the
        # assertion, not the model.
        assert o2.shape[1] > len(t2), (
            f"second turn generated nothing: {o2.shape[1]} <= {len(t2)}"
        )
        print(
            f"\n  turn2: {tokenizer.decode(o2[0][len(t2) :], skip_special_tokens=True).strip()[:110]!r}"
        )


# ── 3. long prompt: the counting anchor is far from the tail ──────────────────


class TestLongPrompt:
    """A prompt long enough that the anchor is thousands of tokens behind.

    The 1/(1+n) count is recovered from an anchor at position 0. Every other test
    uses prompts of a few dozen tokens. This checks the count still inverts
    exactly when the sequence is long -- and, on backends that chunk the prefill,
    that a chunk without the anchor does not break the count.
    """

    @pytest.mark.parametrize("length", [1024, 4096])
    def test_long_sequence_routing(self, composed_model, length):  # noqa: F811
        """Control tokens placed late in a long prompt still route correctly."""
        import torch

        model, config, _ = composed_model
        a, b = list(config.adapter_token_ids)[:2]
        filler = 50

        # base ... A ... B ... : both controls deep into the sequence.
        i_a = length // 2
        i_b = length - 20
        ids = [filler] * length
        ids[i_a] = a
        ids[i_b] = b

        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
        route = model.model._last_adapter_indices[0].tolist()

        assert all(v == 0 for v in route[:i_a]), (
            f"len={length}: positions before the first control must be base; "
            f"saw {sorted(set(route[:i_a]))}"
        )
        assert all(v == route[i_a] for v in route[i_a:i_b]), (
            f"len={length}: adapter A did not hold from {i_a} to {i_b}; "
            f"saw {sorted(set(route[i_a:i_b]))}"
        )
        assert all(v == route[i_b] for v in route[i_b:]), (
            f"len={length}: adapter B did not hold from {i_b} to the end; "
            f"saw {sorted(set(route[i_b:]))}"
        )
        assert route[i_a] != route[i_b] and route[i_a] != 0, (
            f"len={length}: the two controls did not select distinct adapters "
            f"(A={route[i_a]}, B={route[i_b]})"
        )

    def test_many_transitions_in_a_long_prompt(self, composed_model):  # noqa: F811
        """20 transitions in one long sequence -- stresses the counting head.

        The count is recovered as round(1/signal - 1), so a large n means the
        signal is small and the inversion is tighter. 20 switches is far more than
        any earlier test and the engine claims to support many more.
        """
        import torch

        model, config, _ = composed_model
        atoks = list(config.adapter_token_ids)
        a, b = atoks[0], atoks[1]
        filler = 50

        seq, expected, cur = [], [], 0
        for k in range(20):
            tok = a if k % 2 == 0 else b
            seq.append(tok)
            cur = 1 if tok == a else 2
            expected.append(cur)
            seq.extend([filler] * 30)
            expected.extend([cur] * 30)

        with torch.no_grad():
            model(input_ids=torch.tensor([seq]))
        route = model.model._last_adapter_indices[0].tolist()
        wrong = [
            (i, route[i], expected[i])
            for i in range(len(seq))
            if route[i] != expected[i]
        ]
        assert not wrong, (
            f"{len(wrong)} of {len(seq)} positions mis-routed across 20 transitions; "
            f"first few (index, got, want): {wrong[:6]}"
        )
