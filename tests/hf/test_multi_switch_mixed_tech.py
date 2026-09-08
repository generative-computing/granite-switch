# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch with LoRA and aLoRA adapters in the SAME checkpoint.

Every other real-checkpoint suite composes exactly one technology:

    test_multi_switch_e2e.py     --technology-filter lora
    test_multi_switch_alora.py   --technology-filter alora

Nothing composes with NO filter -- which is the composer's default and the most
likely real deployment: one checkpoint carrying both kinds.

Why that matters. The chat template applies technology-SPECIFIC placement:

  * LoRA  -> control token at the sequence START, plus skip_next_role_marker so the
             following role marker is suppressed (avoids two identical embeddings
             back-to-back after the runtime swap).
  * aLoRA -> control token immediately BEFORE the invocation text, plus a
             first-character drop for the same reason.

In a mixed checkpoint the template's adapter_map holds both types and the routing
shape depends on which adapter you name. A LoRA adapter taking the aLoRA path (or
the reverse) would produce the wrong routing shape while still "working" -- two
layers each correct in isolation and never crossed end to end.

Also covers checkpoint re-save stability (see TestResaveStability): persistent
buffers can silently vanish on load, so a second save/load round trip is worth
asserting rather than assuming.

Heavy: composes a real granite-4.1-3b with NO technology filter. Gated on
GRANITE_SWITCH_E2E_MODELS=1.
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
        reason="composes a real ~3B mixed checkpoint; set GRANITE_SWITCH_E2E_MODELS=1",
    ),
]

BASE_MODEL = "ibm-granite/granite-4.1-3b"
# rag carries aLoRA intrinsics; guardian/core carry LoRA ones. With no filter the
# compose should yield BOTH technologies in one checkpoint.
ADAPTER_REPOS = [
    "ibm-granite/granitelib-rag-r1.0",
    "ibm-granite/granitelib-guardian-r1.0",
]

_E2E_ROOT = Path(os.environ.get("GRANITE_SWITCH_E2E_DIR", "/tmp/granite_switch_e2e"))


def _compose_mixed():
    """Compose (or warm-reuse) a checkpoint with NO technology filter."""
    out_dir = _E2E_ROOT / "multi-mixed"
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
        # deliberately NO --technology-filter
        "--output",
        str(out_dir),
    ]
    print("composing (mixed):", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        pytest.fail(
            f"mixed compose failed (exit {r.returncode})\n"
            f"--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}",
            pytrace=False,
        )
    return out_dir


@pytest.fixture(scope="module")
def mixed_model():
    """(model, config, tokenizer, out_dir) for a mixed-technology checkpoint."""
    from transformers import AutoTokenizer

    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM
    from granite_switch.hf.switch import MultiSwitch

    out_dir = _compose_mixed()
    config = GraniteSwitchConfig.from_pretrained(out_dir)
    tok = AutoTokenizer.from_pretrained(out_dir)
    model = GraniteSwitchForCausalLM.from_pretrained(out_dir).eval()
    assert isinstance(model.model.switch, MultiSwitch)
    print(
        f"\n  mixed checkpoint: {config.num_adapters} adapters: "
        f"{list(getattr(config, 'adapter_names', []) or [])}",
        file=sys.stderr,
    )
    return model, config, tok, out_dir


def _template_types(tokenizer):
    """Parse {adapter_name: technology} out of the template's adapter_map."""
    import re

    tmpl = tokenizer.chat_template or ""
    return dict(
        re.findall(r"'([^']+)':\s*\{'token':\s*'[^']*',\s*'type':\s*'([^']+)'", tmpl)
    )


def _ctrl_positions(ids, atoks):
    ctrl = set(int(t) for t in atoks)
    return [i for i, t in enumerate(ids) if int(t) in ctrl]


def _render(tok, question, adapter_name):
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        adapter_name=adapter_name,
    )


class TestMixedComposition:
    """The unfiltered compose must really contain BOTH technologies."""

    def test_unfiltered_compose_prefers_alora(self, mixed_model):
        """An unfiltered compose is aLoRA-only BY DESIGN, not by accident.

        My original premise here was wrong: I assumed no --technology-filter would
        yield a checkpoint carrying both technologies. It does not, and the reason is
        deliberate. When the granitelib libraries publish an adapter under both an
        `alora/` and a `lora/` subtree, discovery prefers aLoRA:

            adapter_discovery.py:331   if "alora" in technologies: return "alora"
            adapter_discovery.py:77    if tech == "alora" and existing_tech == "lora":
                                           ... replacing lora

        So the real behaviour is: unfiltered -> aLoRA wins for every dual-published
        adapter. That is worth pinning, because a silent flip to LoRA preference
        would change the routing SHAPE of every unfiltered checkpoint
        (adapter-throughout instead of base-then-adapter) without changing any
        adapter name or count.

        A genuinely mixed checkpoint needs libraries whose adapters are published in
        only one technology each; the classes below therefore skip rather than fail
        when a technology is absent.
        """
        _model, _config, tok, _d = mixed_model
        types = _template_types(tok)
        kinds = set(types.values())
        print(f"\n  adapter technologies: {types}", file=sys.stderr)
        assert kinds, f"no adapter types found in the template: {types}"
        assert "alora" in kinds, (
            f"an unfiltered compose of {ADAPTER_REPOS} should prefer aLoRA for "
            f"dual-published adapters; got {kinds}. If discovery's preference "
            "changed, every unfiltered checkpoint's routing shape changed with it."
        )

    def test_one_control_token_per_adapter(self, mixed_model):
        """Token count must still track adapter count across mixed technologies."""
        _model, config, _tok, _d = mixed_model
        assert len(config.adapter_token_ids) == config.num_adapters, (
            f"{len(config.adapter_token_ids)} control tokens for "
            f"{config.num_adapters} adapters"
        )
        assert len(set(config.adapter_token_ids)) == len(config.adapter_token_ids), (
            "duplicate control token ids in a mixed checkpoint"
        )


class TestMixedPlacementAndRouting:
    """Each technology must keep its own placement and routing shape."""

    def test_lora_adapter_routes_from_sequence_start(self, mixed_model):
        """A LoRA-typed adapter puts its token at the start -> adapter throughout."""
        import torch

        model, config, tok, _d = mixed_model
        types = _template_types(tok)
        lora_names = [n for n, t in types.items() if t == "lora"]
        if not lora_names:
            pytest.skip("no lora-typed adapter in this checkpoint")
        ids = tok(
            _render(tok, "Summarize the context.", lora_names[0]),
            add_special_tokens=False,
        )["input_ids"]
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        assert pos, f"no control token rendered for lora adapter {lora_names[0]!r}"
        assert pos[0] == 0, (
            f"a LoRA control token must land at index 0 (sequence start); got {pos[0]}"
        )
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
        route = model.model._last_adapter_indices[0].tolist()
        assert all(v == route[0] for v in route) and route[0] != 0, (
            f"LoRA routing must be one adapter throughout; got {sorted(set(route))}"
        )

    def test_alora_adapter_routes_base_then_adapter(self, mixed_model):
        """An aLoRA-typed adapter in the SAME checkpoint keeps aLoRA's shape."""
        import torch

        model, config, tok, _d = mixed_model
        types = _template_types(tok)
        alora_names = [n for n, t in types.items() if t == "alora"]
        if not alora_names:
            pytest.skip("no alora-typed adapter in this checkpoint")
        ids = tok(
            _render(tok, "Is this answerable from the context?", alora_names[0]),
            add_special_tokens=False,
        )["input_ids"]
        pos = _ctrl_positions(ids, config.adapter_token_ids)
        assert pos, f"no control token rendered for alora adapter {alora_names[0]!r}"
        assert pos[0] > 0, (
            f"an aLoRA control token must NOT be at index 0 (that is LoRA's "
            f"placement); got {pos[0]}"
        )
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))
        route = model.model._last_adapter_indices[0].tolist()
        first = pos[0]
        assert all(v == 0 for v in route[:first]), (
            f"aLoRA must be base before {first}; got {sorted(set(route[:first]))}"
        )
        assert all(v == route[first] for v in route[first:]) and route[first] != 0, (
            f"aLoRA adapter must hold from {first} to the end; "
            f"got {sorted(set(route[first:]))}"
        )

    def test_lora_and_alora_select_different_experts(self, mixed_model):
        """The two technologies must map to distinct expert ids, not collide."""
        import torch

        model, config, tok, _d = mixed_model
        types = _template_types(tok)
        lora = [n for n, t in types.items() if t == "lora"]
        alora = [n for n, t in types.items() if t == "alora"]
        if not lora or not alora:
            pytest.skip(f"need both technologies; got lora={lora} alora={alora}")
        got = {}
        for name in (lora[0], alora[0]):
            ids = tok(
                _render(tok, "Is this answerable?", name), add_special_tokens=False
            )["input_ids"]
            pos = _ctrl_positions(ids, config.adapter_token_ids)
            if not pos:
                continue
            with torch.no_grad():
                model(input_ids=torch.tensor([ids]))
            got[name] = model.model._last_adapter_indices[0].tolist()[pos[0]]
        if len(got) < 2:
            pytest.skip(f"could not render both technologies: {got}")
        assert len(set(got.values())) == 2, (
            f"a LoRA and an aLoRA adapter routed to the SAME expert id: {got}"
        )

    def test_mixed_checkpoint_generates(self, mixed_model):
        """Generation works for both technologies from one checkpoint."""
        import torch

        model, _config, tok, _d = mixed_model
        types = _template_types(tok)
        for kind in ("lora", "alora"):
            names = [n for n, t in types.items() if t == kind]
            if not names:
                continue
            ids = torch.tensor(
                [
                    tok(
                        _render(tok, "Is this answerable?", names[0]),
                        add_special_tokens=False,
                    )["input_ids"]
                ]
            )
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=8, do_sample=False)
            assert out.shape[1] > ids.shape[1], f"{kind} adapter generated nothing"
            print(
                f"\n  {kind} ({names[0]}): "
                f"{tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()[:90]!r}",
                file=sys.stderr,
            )


class TestResaveStability:
    """A composed checkpoint must survive being re-saved and reloaded.

    The codebook and substitute LUT are persistent buffers that can silently
    vanish on ``from_pretrained``. ``test_multi_switch_buffers.py`` covers ONE
    round trip on a synthetic model; this covers a second round trip on the real
    composed artifact, where a re-save could drop or alter a buffer with nobody
    noticing.
    """

    def test_resave_preserves_buffers_and_routing(self, mixed_model, tmp_path):
        import torch

        from granite_switch.hf import GraniteSwitchForCausalLM

        model, config, tok, _d = mixed_model
        types = _template_types(tok)
        name = next(iter(types))
        ids_list = tok(
            _render(tok, "Is this answerable?", name), add_special_tokens=False
        )["input_ids"]
        ids = torch.tensor([ids_list])

        with torch.no_grad():
            model(input_ids=ids)
        route_before = model.model._last_adapter_indices[0].tolist()

        out = tmp_path / "resaved"
        model.save_pretrained(out)
        tok.save_pretrained(out)
        reloaded = GraniteSwitchForCausalLM.from_pretrained(out).eval()
        sw = reloaded.model.switch

        cb = sw.codebook.float()
        assert not torch.all(cb == 0), "re-saved checkpoint lost its codebook"
        norms = cb.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)
        lut = sw.control_to_substitute_lut
        assert lut is not None and not torch.all(lut == 0), (
            "re-saved checkpoint lost its substitute LUT"
        )

        with torch.no_grad():
            reloaded(input_ids=ids)
        route_after = reloaded.model._last_adapter_indices[0].tolist()
        assert route_after == route_before, (
            "routing changed after a save/load round trip:\n"
            f"  before: {route_before}\n  after : {route_after}"
        )
