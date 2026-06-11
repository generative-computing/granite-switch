# SPDX-License-Identifier: Apache-2.0
"""End-to-end SingleSwitch tests through the full GraniteSwitchForCausalLM.

Bridges the "isolation gap" called out in issue #107: the bare-switch sharpness
tests in `tests/shared/single_switch_cases.py` exercise SingleSwitch in isolation
with synthetic geometry, but never through `GraniteSwitchForCausalLM.forward()`
with production-like config. These tests run a full model forward and inspect
`model.model._last_adapter_indices` to verify the switch fires correctly when
wired through:

  GraniteSwitchConfig  →  create_switch()  →  SingleSwitch.__init__
                       →  model forward    →  _last_adapter_indices

Parametrized over PRODUCTION_ATTENTION_MULTIPLIERS to catch config-flow
regressions across the realistic multiplier values.

CPU-only. Does not exercise vLLM gain compensation — HF SingleSwitch hardcodes
scaling=1.0 regardless of config. Compensation is tested in the Tier 2 composer
test (see scratch/ISSUE_107_HANDOFF.md for the deferred follow-up design).

Runtime reference (measured on CPU, 2026-04-30):
  seq_len=10_000   → 0.22s / case
  seq_len=32_768   → 1.73s / case
  seq_len=65_536   → 7.37s / case   (marked @pytest.mark.slow)
  seq_len=131_072  → 26.43s / case  (marked @pytest.mark.slow)
"""

import pytest
import torch

pytestmark = pytest.mark.local_fast

from tests.shared.generation_models import DENSE_CFG, make_switch_model
from tests.shared.granite4_constants import (
    MAX_POSITION_EMBEDDINGS,
    PRODUCTION_ATTENTION_MULTIPLIERS,
)
from tests.shared.single_switch_cases import ADAPTER_TOKEN_IDS_LIST, NUM_ADAPTERS

# TEXT_TOKEN matches tests/shared/single_switch_cases.py convention. Any
# non-adapter token ID works — 50 is outside ADAPTER_TOKEN_IDS_LIST (1000+).
TEXT_TOKEN = 50

# Derived ceilings for the test fixture overrides. Both `DENSE_CFG` and the
# granite4 constants are authoritative sources — pick the higher of each so
# the test fixture auto-adjusts if either source is raised later.
#
# TODO: after the granite4_constants.py → granite4_family_constants.py rename
# (see scratch/ISSUE_107_HANDOFF.md §10.2), update the import above; these
# constants continue to work unchanged.
_E2E_MAX_POSITION_EMBEDDINGS = max(
    DENSE_CFG["max_position_embeddings"], MAX_POSITION_EMBEDDINGS,
)
# vocab_size must fit every adapter token ID. ADAPTER_TOKEN_IDS_LIST goes
# up to 1031, so DENSE_CFG's default 256 is too small. Derive from the actual
# token IDs in use rather than hardcoding — auto-tracks if NUM_ADAPTERS grows.
_E2E_VOCAB_SIZE = max(
    DENSE_CFG["vocab_size"], max(ADAPTER_TOKEN_IDS_LIST) + 1,
)


def _build_e2e_overrides(base_cfg, *, num_adapters=NUM_ADAPTERS):
    """Build config overrides for a production-ish E2E test model."""
    adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    return {
        "vocab_size": _E2E_VOCAB_SIZE,
        "max_position_embeddings": _E2E_MAX_POSITION_EMBEDDINGS,
        "num_adapters": num_adapters,
        "adapter_ranks": [8] * num_adapters,
        "adapter_token_ids": ADAPTER_TOKEN_IDS_LIST[:num_adapters],
        "adapter_names": adapter_names,
        "adapter_substitute_token_ids": [1] * num_adapters,
        "num_hidden_layers": len(base_cfg["layer_types"]) + 1,
        "layer_types": ["attention"] + base_cfg["layer_types"],
    }


def _make_e2e_model(base_cfg, overrides):
    """Build a GraniteSwitchForCausalLM and populate adapter_token_ids."""
    model, config = make_switch_model(base_cfg, overrides)
    # Inlined from tests/hf/test_model_forward.py:_set_adapter_token_ids.
    model.model.adapter_token_ids.data = torch.tensor(
        config.adapter_token_ids, dtype=torch.long,
    )
    return model, config


# Function-scoped: each test gets a fresh model (model build is 0.13s, cheap).
# Module scope would save ~19s across the long-context matrix but would require
# auditing that no test mutates model state — not worth it.
@pytest.fixture(
    params=PRODUCTION_ATTENTION_MULTIPLIERS,
    ids=lambda m: f"mult={m}",
)
def e2e_model(request):
    """GraniteSwitchForCausalLM parametrized over attention_multiplier."""
    multiplier = request.param
    base_cfg = {**DENSE_CFG, "attention_multiplier": multiplier}
    overrides = _build_e2e_overrides(base_cfg)
    model, config = _make_e2e_model(base_cfg, overrides)
    return model, config, multiplier


@pytest.fixture
def e2e_model_32adapter():
    """Single-variant fixture for the 32-adapter stress test.

    The adapter-ID rounding math is independent of multiplier, so we don't
    parametrize. Chosen variant: most common production multiplier
    (0.0078125, granite-4.0-h-1b/tiny/small/4.1-8b/30b).
    """
    base_cfg = {**DENSE_CFG, "attention_multiplier": 0.0078125}
    overrides = _build_e2e_overrides(base_cfg)
    model, config = _make_e2e_model(base_cfg, overrides)
    return model, config


def _control_position(seq_len, where):
    """Translate {early, mid, late} → concrete index.

    Mirrors tests/shared/single_switch_cases.py:182-187 so that HF E2E and
    bare-switch sharpness tests sweep the same position semantics.
    """
    return {
        "early": 1,
        "mid": seq_len // 2,
        "late": seq_len - 10,
    }[where]


# ----------------------------------------------------------------------------
# Tier 1 test classes
# ----------------------------------------------------------------------------


class TestE2EBasicAdapterActivation:
    """Minimum viable end-to-end assertion: control token fires the switch."""

    def test_pre_control_is_zero_post_control_matches_adapter(self, e2e_model):
        """End-to-end: a control token at position 2 activates its adapter
        from position 2 onward; positions before it remain at 0 (base).

        Proves the full chain config → create_switch → forward →
        _last_adapter_indices works on a production-ish multiplier.
        """
        model, config, mult = e2e_model
        ctrl_token = config.adapter_token_ids[0]  # adapter_0 → expected index 1
        input_ids = torch.tensor([[10, 20, ctrl_token, 30, 40, 50, 60, 70]])
        with torch.no_grad():
            model(input_ids=input_ids)
        ai = model.model._last_adapter_indices
        assert (ai[:, :2] == 0).all(), f"pre-control should be 0, got {ai[:, :2]}"
        assert (ai[:, 2:] == 1).all(), f"post-control should be 1, got {ai[:, 2:]}"


class TestE2EAllAdaptersRecover:
    """Adapter-ID rounding stress across all 32 supported adapter IDs.

    High adapter IDs (31) are the hardest to recover because the non-control
    mass in the softmax scales with the adapter ID's magnitude. If bf16 math
    drifts, failures appear first at the top of the range.
    """

    @pytest.mark.parametrize("adapter_idx", range(NUM_ADAPTERS))
    def test_each_adapter_recovers(self, e2e_model_32adapter, adapter_idx):
        """Each of the 32 adapter IDs recovers exactly through a full model forward.

        1-indexed by convention (adapter_idx=0 → expected_id=1), per
        src/granite_switch/hf/switch/single.py:135.
        """
        model, config = e2e_model_32adapter
        ctrl_token = config.adapter_token_ids[adapter_idx]
        expected_id = adapter_idx + 1
        input_ids = torch.tensor([[10, 20, ctrl_token, 30, 40, 50, 60, 70]])
        with torch.no_grad():
            model(input_ids=input_ids)
        ai = model.model._last_adapter_indices
        assert (ai[:, :2] == 0).all()
        assert (ai[:, 2:] == expected_id).all(), (
            f"adapter_idx={adapter_idx} (expected {expected_id}): got {ai}"
        )


class TestE2ELongContext:
    """Long-context + position-in-sequence coverage through the full model.

    Bridges the bare-switch sharpness sweep (SingleSwitchContextLengthSweepCases)
    with the full GraniteSwitchForCausalLM.forward() pipeline. The full pipeline
    includes RoPE, hiding-group masking, and decoder layers — each of which has
    its own O(N²) cost component and could theoretically break at extreme
    context lengths while the bare switch still works.

    Control position matters: the pre-control and post-control assertion slices
    differ in size depending on whether the control token is early, mid, or late.
    """

    @pytest.mark.parametrize("control_position", ["early", "mid", "late"])
    @pytest.mark.parametrize("adapter_idx", [0, 15, 31])  # low / mid / high stress
    @pytest.mark.parametrize("seq_len", [
        10_000,
        32_768,
        pytest.param(65_536, marks=pytest.mark.slow),
        pytest.param(131_072, marks=pytest.mark.slow),
    ])
    def test_long_context_e2e(self, e2e_model, seq_len, adapter_idx, control_position):
        """Full model forward at long context with parametrized control position.

        Default CI runs seq_len ∈ {10K, 32K}; `-m slow` adds 65K and 131K.
        """
        model, config, mult = e2e_model
        ctrl_token = config.adapter_token_ids[adapter_idx]
        expected_id = adapter_idx + 1
        ctrl_pos = _control_position(seq_len, control_position)

        seq = [TEXT_TOKEN] * seq_len
        seq[ctrl_pos] = ctrl_token
        input_ids = torch.tensor([seq])

        with torch.no_grad():
            model(input_ids=input_ids)

        ai = model.model._last_adapter_indices[0]
        if ctrl_pos > 0:
            assert (ai[:ctrl_pos] == 0).all(), (
                f"pre-control slice should be all 0; failed at seq_len={seq_len}, "
                f"adapter_idx={adapter_idx}, position={control_position}, "
                f"mult={mult}"
            )
        assert (ai[ctrl_pos:] == expected_id).all(), (
            f"post-control slice should be all {expected_id}; failed at seq_len={seq_len}, "
            f"adapter_idx={adapter_idx}, position={control_position}, "
            f"mult={mult}"
        )
