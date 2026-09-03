# SPDX-License-Identifier: Apache-2.0
"""End-to-end MultiSwitch tests through a REAL composed GraniteSwitch model.

Composes an *actual* granite-4.1-3b checkpoint (MultiSwitch is the only engine)
and runs the full ``GraniteSwitchForCausalLM.forward`` on multi-transition token
sequences, asserting ``model.model._last_adapter_indices`` matches the expected
latest-wins routing.

This proves the full chain persists and rebuilds correctly:

  compose (-> config.json)
      -> from_pretrained -> create_switch -> MultiSwitch
      -> model.forward -> _last_adapter_indices

Heavy: composes a ~3B checkpoint and loads it. Marked slow / requires_model /
gpu, and skipped unless ``GRANITE_SWITCH_E2E_MODELS=1`` is set (composing
downloads the base model + adapters). The module-scoped fixture warm-reuses an
already-composed output dir when its ``config.json`` exists, so re-runs skip the
expensive recompose.

Compose command (mirrors the task runbook)::

  python -m granite_switch.composer.compose_granite_switch \\
      --base-model ibm-granite/granite-4.1-3b \\
      --adapters ibm-granite/granitelib-guardian-r1.0 \\
                 ibm-granite/granitelib-core-r1.0 \\
      --technology-filter lora \\
      --output <dir>
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
        reason="composes a real ~3B checkpoint; set GRANITE_SWITCH_E2E_MODELS=1 to run",
    ),
]

SWITCH_TYPES = ["multi"]

BASE_MODEL = "ibm-granite/granite-4.1-3b"
ADAPTER_REPOS = [
    "ibm-granite/granitelib-guardian-r1.0",
    "ibm-granite/granitelib-core-r1.0",
]

# Where composed checkpoints are cached. Override with GRANITE_SWITCH_E2E_DIR.
_E2E_ROOT = Path(os.environ.get("GRANITE_SWITCH_E2E_DIR", "/tmp/granite_switch_e2e"))


def _compose(switch_type):
    """Compose (or warm-reuse) a real checkpoint for ``switch_type``.

    Returns the output directory. Skips the recompose when config.json already
    exists in the target dir (warm reuse).
    """
    out_dir = _E2E_ROOT / switch_type
    if (out_dir / "config.json").exists():
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "granite_switch.composer.compose_granite_switch",
        "--base-model",
        BASE_MODEL,
        "--adapters",
        *ADAPTER_REPOS,
        "--technology-filter",
        "lora",
        "--output",
        str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"compose failed for {switch_type}:\n"
            f"--- stdout ---\n{result.stdout[-3000:]}\n"
            f"--- stderr ---\n{result.stderr[-3000:]}"
        )
    assert (out_dir / "config.json").exists(), "compose produced no config.json"
    return out_dir


@pytest.fixture(scope="module", params=SWITCH_TYPES, ids=lambda t: t)
def composed_model(request):
    """Module-scoped composed GraniteSwitch model.

    Confirms ``from_pretrained`` rebuilds a real MultiSwitch engine (the only
    engine) from the composed checkpoint.
    """
    from granite_switch.config import GraniteSwitchConfig
    from granite_switch.hf import GraniteSwitchForCausalLM

    switch_type = request.param
    out_dir = _compose(switch_type)

    config = GraniteSwitchConfig.from_pretrained(out_dir)

    model = GraniteSwitchForCausalLM.from_pretrained(out_dir).eval()
    # Confirm from_pretrained rebuilt the coded-memory engine.
    from granite_switch.hf.switch import MultiSwitch

    assert isinstance(model.model.switch, MultiSwitch), (
        f"from_pretrained built {type(model.model.switch).__name__}, "
        f"expected MultiSwitch"
    )
    return model, config, switch_type


def _control_ids(config):
    """Return (A_TOK, B_TOK): control tokens for adapters 1 and 2.

    Real composed configs use the no-base-slot layout: adapter_token_ids[k]
    fires adapter k+1. Two lora adapters give exactly two control tokens.
    """
    atoks = config.adapter_token_ids
    assert len(atoks) >= 2, f"need >= 2 adapters, got {len(atoks)}"
    return atoks[0], atoks[1]  # -> adapter 1, adapter 2


def _run_and_get_indices(model, seq):
    import torch

    input_ids = torch.tensor([seq])
    with torch.no_grad():
        model(input_ids=input_ids)
    return model.model._last_adapter_indices[0].tolist()


# TEXT_TOKEN: any non-control id. 50 is far below the appended control ids.
TEXT = 50


class TestE2EMultiTransition:
    """Full-model routing matches the latest-wins expectation."""

    def test_base_then_A_then_B(self, composed_model):
        """base -> A -> B: [T,A,T,B,T] -> [0,1,1,2,2]."""
        model, config, _ = composed_model
        a, b = _control_ids(config)
        result = _run_and_get_indices(model, [TEXT, a, TEXT, b, TEXT])
        assert result == [0, 1, 1, 2, 2]

    def test_higher_then_lower_A2_then_B1(self, composed_model):
        """A(2) -> B(1) latest-wins: [B,T,A,T] -> [2,2,1,1].

        The case SingleSwitch mis-routes (its softmax averages the two control
        values); the multi engines take the most recent, so this is exact.
        """
        model, config, _ = composed_model
        a, b = _control_ids(config)
        result = _run_and_get_indices(model, [b, TEXT, a, TEXT])
        assert result == [2, 2, 1, 1]

    def test_sticky_single_transition(self, composed_model):
        """One control token persists through a long text tail."""
        model, config, _ = composed_model
        a, _b = _control_ids(config)
        seq = [TEXT, TEXT, a] + [TEXT] * 8
        result = _run_and_get_indices(model, seq)
        assert result[:2] == [0, 0]
        assert all(v == 1 for v in result[2:])

    def test_all_base(self, composed_model):
        """No control tokens -> every position is base (0)."""
        model, _config, _ = composed_model
        result = _run_and_get_indices(model, [TEXT] * 12)
        assert all(v == 0 for v in result)

    def test_control_at_position_zero(self, composed_model):
        """Control token at pos 0 -> adapter active from the start."""
        model, config, _ = composed_model
        a, _b = _control_ids(config)
        result = _run_and_get_indices(model, [a, TEXT, TEXT, TEXT])
        assert all(v == 1 for v in result)

    def test_three_transitions(self, composed_model):
        """A -> B -> A latest-wins across three transitions."""
        model, config, _ = composed_model
        a, b = _control_ids(config)
        result = _run_and_get_indices(model, [TEXT, a, TEXT, b, TEXT, a, TEXT])
        assert result == [0, 1, 1, 2, 2, 1, 1]


class TestE2ERealCheckpointGenerate:
    """Gates that only a REAL composed checkpoint can exercise.

    The routing tests above run a single ``forward`` and read
    ``_last_adapter_indices``. Three separate production bugs lived on paths that
    such a test cannot reach, and each survived the whole suite:

      1. ``codebook`` / ``control_to_substitute_lut`` were registered
         ``persistent=False``, so ``from_pretrained`` returned them as ALL ZEROS
         (a zeroed codebook makes the retrieval softmax uniform, i.e. it averages
         adapters). Only visible after a real save/load.
      2. ``positions`` was never passed to the vLLM switch, so only the first
         request in a flat batch got a counting anchor. Only visible under real
         batching.
      3. The switch built its causal mask as ``q_len x q_len`` while its heads
         attend over the whole KV cache, so every decode step passed a mis-shaped
         bias and ``generate()`` raised. Only visible when actually generating.

    These tests close (1) and (3) on the real checkpoint. (2) is covered by the
    vLLM batching tests.
    """

    def test_real_checkpoint_buffers_are_valid(self, composed_model):
        """The composed checkpoint's switch buffers must load intact.

        Guards bug (1) on the real artifact rather than a synthetic round trip:
        a unit-norm codebook and a LUT that still uses -1 as its
        "not a control token" sentinel.
        """
        import torch

        model, config, _ = composed_model
        sw = model.model.switch

        cb = sw.codebook.float()
        assert not torch.all(cb == 0), (
            "composed checkpoint loaded an ALL-ZERO codebook -- the buffer is not "
            "persistent, so the memory softmax will average adapters instead of "
            "selecting one"
        )
        norms = cb.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)

        lut = sw.control_to_substitute_lut
        assert lut is not None, "composed checkpoint has no substitute LUT"
        assert not torch.all(lut == 0), (
            "substitute LUT loaded ALL ZEROS -- token exchange would rewrite every "
            "token id to 0 (the 'not a control' sentinel is -1)"
        )
        for ctrl, sub in zip(
            config.adapter_token_ids, config.adapter_substitute_token_ids
        ):
            assert int(lut[ctrl]) == int(sub), (
                f"LUT maps control {ctrl} -> {int(lut[ctrl])}, expected {sub}"
            )

    def test_real_checkpoint_generates(self, composed_model):
        """``generate()`` must work on the real checkpoint.

        Guards bug (3). Prefill has ``q_len == kv_len`` so a forward-only test
        cannot see the mask-shape defect; the first decode step is where it fired.
        """
        import torch

        model, config, _ = composed_model
        a, _b = _control_ids(config)
        ids = torch.tensor([[TEXT, TEXT, a, TEXT]])
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=5, do_sample=False)
        assert out.shape[1] == ids.shape[1] + 5, (
            f"generate produced {out.shape[1]} tokens, expected {ids.shape[1] + 5}"
        )

    def test_real_checkpoint_generate_carries_adapter(self, composed_model):
        """The adapter set before generation stays active through decode.

        This is the latest-wins contract applied to generated tokens: a control
        token in the prompt must keep its adapter active while the model produces
        new tokens, otherwise multi-turn serving silently reverts to base.
        """
        import torch

        model, config, _ = composed_model
        _a, b = _control_ids(config)
        ids = torch.tensor([[TEXT, b, TEXT]])
        with torch.no_grad():
            model.generate(ids, max_new_tokens=4, do_sample=False)
        # The final forward is a decode step; routing must still be adapter 2.
        route = model.model._last_adapter_indices[0].tolist()
        assert all(v == 2 for v in route), (
            f"decode routing {route} lost the adapter set by the control token"
        )
