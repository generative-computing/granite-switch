# SPDX-License-Identifier: Apache-2.0
"""HF SingleSwitch tests.

Test cases are defined once in tests/shared/single_switch_cases.py.
This file provides:
- HF-specific backend probing and parametrized fixture
- ``_run()`` adapter that bridges mixin → HF SingleSwitch.forward
- TestBatchProcessing (HF-only: vLLM batches externally)
"""

import pytest
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from granite_switch.hf.switch.single import SingleSwitch
from tests.shared.single_switch_cases import (
    ADAPTER_TOKEN_IDS_LIST,
    NUM_ADAPTERS,
    TEXT_TOKEN,
    SingleSwitchAdapterRetrievalCases,
    SingleSwitchContextLengthSweepCases,
    SingleSwitchEdgeCases,
    SingleSwitchGainSensitivityCases,
    SingleSwitchShapeCorrectnessCases,
    SingleSwitchTokenMatchingCases,
)

# ── Config ──────────────────────────────────────────────────────────


class _AttnConfig:
    """Minimal config to select an HF attention backend."""

    def __init__(self, backend="sdpa"):
        self._attn_implementation = backend
        self._pre_quantization_dtype = torch.bfloat16
        self.hidden_size = 128
        self.num_attention_heads = 4  # head_dim = 128/4 = 32


# ── Backend discovery ────────────────────────────────────────────────
#
# Probe each non-eager backend at import time with a small differential-gain
# retrieval call (3 positions, head_dim=32).  Backends that don't work on the
# current platform are skipped in tests.

_NON_EAGER_BACKENDS = sorted(
    name for name in ALL_ATTENTION_FUNCTIONS if "eager" not in name
)


def _probe_single_switch_backend(name):
    """Test whether backend works for SingleSwitch on this platform."""
    try:
        fn = ALL_ATTENTION_FUNCTIONS[name]
    except KeyError:
        return False, "not registered in ALL_ATTENTION_FUNCTIONS"

    try:
        config = _AttnConfig(name)
        module = SingleSwitch(
            num_adapters=4,
            config=config,
            control_token_gain=15.0,
        )
        head_dim = module.head_dim
        gain = 15.0

        # Single active dimension (dim 0) — matches implementation
        q = torch.zeros(1, 1, 3, head_dim, dtype=torch.float32)
        q[:, :, :, 0] = 1.0
        k = torch.zeros(1, 1, 3, head_dim, dtype=torch.float32)
        k[:, :, :, 0] = -gain
        v = torch.zeros(1, 1, 3, head_dim, dtype=torch.float32)

        k[0, 0, 1, 0] = gain
        v[0, 0, 1, 0] = 2.0

        output, _ = fn(
            module,
            q,
            k,
            v,
            None,
            dropout=0.0,
            scaling=1.0,
            sliding_window=None,
        )

        if output.shape != (1, 3, 1, head_dim):
            return False, f"unexpected output shape {output.shape}"

        val_pos0 = output[0, 0, 0, 0].item()
        if abs(val_pos0) > 0.5:
            return False, f"causal masking broken: pos 0 got {val_pos0:.4f}"

        val = output[0, 2, 0, 0].item()
        if abs(round(val) - 2.0) > 0.5:
            return False, f"retrieval broken: pos 2 got {val:.4f}"

        return True, "ok"
    except Exception as e:
        return False, str(e).split("\n")[0]


_BACKEND_PROBE_RESULTS = {
    name: _probe_single_switch_backend(name) for name in _NON_EAGER_BACKENDS
}

_AVAILABLE_BACKENDS = [
    name for name in _NON_EAGER_BACKENDS if _BACKEND_PROBE_RESULTS[name][0]
]


@pytest.fixture(params=_AVAILABLE_BACKENDS)
def backend(request):
    """Parametrized fixture yielding each working non-eager backend name.

    Only backends that passed the import-time probe are included.
    To see which backends were probed and why some were excluded, run:
        python -c "from tests.hf.test_single_switch import _BACKEND_PROBE_RESULTS; \\
                    print({k: v for k, v in _BACKEND_PROBE_RESULTS.items() if not v[0]})"
    """
    return request.param


# ── HF _run adapter ─────────────────────────────────────────────────


def _make_switch(backend="sdpa", num_adapters=NUM_ADAPTERS, control_token_gain=15.0):
    return SingleSwitch(
        num_adapters=num_adapters,
        config=_AttnConfig(backend),
        control_token_gain=control_token_gain,
    )


class _HFSingleSwitchBase:
    """Provides _run() for shared mixin tests.  Uses ``backend`` fixture."""

    @pytest.fixture(autouse=True)
    def _set_backend(self, backend):
        self._backend = backend

    def _run(self, seq, num_adapters=NUM_ADAPTERS, control_token_gain=15.0):
        switch = _make_switch(self._backend, num_adapters, control_token_gain)
        token_ids = torch.tensor(ADAPTER_TOKEN_IDS_LIST[:num_adapters])
        input_ids = torch.tensor([seq])
        # Switch returns (adapter_indices, modified_input_ids); these tests
        # only check adapter selection so we drop the rewritten ids here.
        adapter_indices, _modified = switch.forward(
            input_ids=input_ids,
            adapter_token_ids=token_ids,
        )
        return adapter_indices[0].tolist()


# ── Shared test classes (from mixin) ────────────────────────────────


class TestTokenMatching(_HFSingleSwitchBase, SingleSwitchTokenMatchingCases):
    pass


class TestAdapterRetrieval(_HFSingleSwitchBase, SingleSwitchAdapterRetrievalCases):
    pass


class TestEdgeCases(_HFSingleSwitchBase, SingleSwitchEdgeCases):
    pass


class TestShapeCorrectness(_HFSingleSwitchBase, SingleSwitchShapeCorrectnessCases):
    pass


class TestContextLengthSweep(_HFSingleSwitchBase, SingleSwitchContextLengthSweepCases):
    pass


class TestGainSensitivity(_HFSingleSwitchBase, SingleSwitchGainSensitivityCases):
    pass


# ── HF-only tests ───────────────────────────────────────────────────


class TestBatchProcessing:
    """Batch independence (HF-only: vLLM batches externally)."""

    def test_batch_independence(self, backend):
        switch = _make_switch(backend, num_adapters=4)
        token_ids = torch.tensor(ADAPTER_TOKEN_IDS_LIST[:4])
        input_ids = torch.tensor(
            [
                [
                    TEXT_TOKEN,
                    ADAPTER_TOKEN_IDS_LIST[0],
                    TEXT_TOKEN,
                    TEXT_TOKEN,
                    TEXT_TOKEN,
                ],
                [
                    TEXT_TOKEN,
                    ADAPTER_TOKEN_IDS_LIST[3],
                    TEXT_TOKEN,
                    TEXT_TOKEN,
                    TEXT_TOKEN,
                ],
            ]
        )
        adapter_indices, _modified = switch.forward(
            input_ids=input_ids,
            adapter_token_ids=token_ids,
        )
        assert (adapter_indices[0, 2:] == 1).all()
        assert (adapter_indices[1, 2:] == 4).all()
