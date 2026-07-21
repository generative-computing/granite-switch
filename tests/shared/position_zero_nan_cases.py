# SPDX-License-Identifier: Apache-2.0
"""Shared test cases for the position-0 NaN regression (both HF and vLLM).

Provides two mixins that both backend test files inherit:

SDPANaNCases
    Pure-tensor SDPA tests.  No model infrastructure required.  Verifies
    that the Q-side control value of 0.0 (post-fix) produces finite output
    and documents that 1.0 (pre-fix) would produce NaN.

ModelFinitenessCases
    Abstract mixin for model-level NaN regression.  Defines the
    parametrized test methods; each backend provides one hook::

        _assert_no_nan(switch_type, ctrl_pos, seq_len, seed) -> None

    The hook must raise AssertionError (or any exception) if the model
    produces non-finite output for the given input.

    ctrl_pos=0 must always remain in the parametrize list — it is the
    exact scenario that triggered the NaN bug (fixed in dcc5e2b).
"""

import itertools

import pytest
import torch
import torch.nn.functional as F

from tests.shared.gap_equivalence import SWITCH_TYPES

# ════════════════════════════════════════════════════════════════════
# SDPA NaN cases (backend-agnostic)
# ════════════════════════════════════════════════════════════════════


class SDPANaNCases:
    """Verify Q/K control values at position 0 via raw SDPA — no model needed."""

    @staticmethod
    def _sdpa_is_finite(q_ctrl_value: float, seq_len: int = 1) -> bool:
        """Run causal SDPA with a control token at position 0.

        K-ctrl=finfo.min, Q-ctrl=q_ctrl_value for token 0.
        Returns True if all outputs are finite.
        """
        head_dim = 32
        torch.manual_seed(7)
        q = torch.randn(1, 1, seq_len, head_dim + 1)
        k = torch.randn(1, 1, seq_len, head_dim + 1)
        v = torch.randn(1, 1, seq_len, head_dim + 1)

        k[0, 0, 0, head_dim] = torch.finfo(q.dtype).min
        q[0, 0, 0, head_dim] = q_ctrl_value
        v[0, 0, 0, head_dim] = 0.0

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return bool(out.isfinite().all())

    def test_post_fix_q_ctrl_zero_is_finite(self):
        """Post-fix: q_ctrl=0.0 at position 0 produces finite SDPA output."""
        assert self._sdpa_is_finite(q_ctrl_value=0.0)

    def test_post_fix_q_ctrl_zero_is_finite_longer_sequence(self):
        """Post-fix: q_ctrl=0.0 remains finite for seq_len=8."""
        assert self._sdpa_is_finite(q_ctrl_value=0.0, seq_len=8)


# ════════════════════════════════════════════════════════════════════
# Model finiteness cases (abstract — backend provides _assert_no_nan)
# ════════════════════════════════════════════════════════════════════

_CTRL_POS_PARAMS = list(itertools.product(SWITCH_TYPES, [0, 1, 2, 4, 8, 14]))
_CTRL_POS_IDS = [f"{st}-ctrl{p}" for st, p in _CTRL_POS_PARAMS]


class ModelFinitenessCases:
    """Abstract mixin: model output must be finite for all ctrl_pos values.

    Subclasses must implement::

        def _assert_no_nan(self, switch_type, ctrl_pos, seq_len, seed):
            ...  # raise if model produces NaN/Inf

    ctrl_pos=0 is the regression case (NaN bug, fixed in dcc5e2b).
    It must never be removed from the parametrize list below.
    """

    @pytest.mark.parametrize("switch_type", SWITCH_TYPES)
    def test_finite_ctrl_at_position_zero(self, switch_type):
        """Regression: ctrl_pos=0 must produce finite logits (the NaN bug scenario)."""
        self._assert_no_nan(switch_type, ctrl_pos=0, seq_len=8, seed=99)

    @pytest.mark.parametrize(
        "switch_type,ctrl_pos", _CTRL_POS_PARAMS, ids=_CTRL_POS_IDS
    )
    def test_finite_all_ctrl_positions(self, switch_type, ctrl_pos):
        """Model output must be finite across all ctrl_pos values including 0."""
        self._assert_no_nan(
            switch_type, ctrl_pos=ctrl_pos, seq_len=16, seed=42 + ctrl_pos
        )
