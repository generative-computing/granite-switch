# SPDX-License-Identifier: Apache-2.0
"""HF MultiSwitch tests (Kerdock/DG coded-memory engine).

Shared test cases live in ``tests/shared/multi_switch_cases.py``; the coded
engine runs them here. This file provides:

- A mock ``GraniteSwitchConfig``-shaped object with realistic backbone geometry,
  which ``create_switch`` builds into the coded engine.
- An HF-specific attention-backend probe:
  the coded engine's *memory* head honors ``config._attn_implementation``, so we
  probe each non-eager backend once and parametrize over the working ones.
- ``_run(seq, adapter_token_ids)`` that bridges the shared mixins to
  ``switch.forward([batch, seq])``.
- Per-engine assertions on ``num_cache_layers`` (coded=2).
- A codes-exact-retrieval test for the coded engine
  (``recover_count_from_signal`` round-trips n=0,1,50,127), plus memory-gain
  validation and the bf16 ~128-transition ceiling (fp32 exact past 128; a
  bf16-quantized counting signal is exact below 128 and degrades beyond it).

CPU-feasible: the coded engine's counting head forces SDPA (fp32) and its memory
head defaults to SDPA when no non-eager backend is available, so it runs on CPU.
"""

import pytest
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from granite_switch.hf.switch import (
    MultiSwitch,
    create_switch,
)
from granite_switch.hf.switch.codes import (
    KerdockDGCodeGenerator,
    recover_count_from_signal,
)
from tests.shared.multi_switch_cases import (
    A_TOK,
    ATOK_NO_BASE,
    B_TOK,
    NUM_ADAPTERS,
    TEXT_TOKEN,
    MultiSwitchEdgeCases,
    MultiSwitchReturnToBaseCases,
    MultiSwitchShapeCorrectnessCases,
    MultiSwitchStickyCases,
    MultiSwitchTransitionCases,
)

SWITCH_TYPES = ["multi"]
EXPECTED_CACHE_LAYERS = {"multi": 2}


# ── Mock config ─────────────────────────────────────────────────────


class _MockSwitchConfig:
    """Minimal GraniteSwitchConfig-shaped object for create_switch.

    Carries realistic backbone geometry (GQA 4Q/2KV,
    projection_head_dim=64), the token-exchange substitute ids, and the ms_*
    coded-engine params. ``adapter_token_ids`` / ``adapter_substitute_token_ids``
    are set per-run because the layout (no-base-slot vs base-reset) changes the
    length, which the engine keys off.
    """

    def __init__(
        self,
        switch_type,
        adapter_token_ids,
        adapter_substitute_token_ids,
        backend="sdpa",
        num_adapters=NUM_ADAPTERS,
    ):
        self.switch_type = switch_type
        self.num_adapters = num_adapters
        # Backbone geometry (mirrors the vLLM single-switch worker's mock).
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.projection_head_dim = 64
        self.attention_multiplier = 0.125
        self.vocab_size = 2000
        self.hidden_size = 256
        # Token-exchange.
        self.adapter_token_ids = adapter_token_ids
        self.adapter_substitute_token_ids = adapter_substitute_token_ids
        # Switch geometry + gain.
        self.switch_head_dim = 32
        self.control_token_gain = 15.0
        # Coded-engine params.
        self.ms_code_m = 6
        self.ms_code_type = "kerdock"
        self.ms_memory_gain = 28.0
        self.ms_counting_head_dim = 32
        # HF attention backend selection (memory head honors this).
        self._attn_implementation = backend
        self._pre_quantization_dtype = torch.float32


# ── Backend discovery ────────────────────────────────────────────────
#
# The coded engine's memory head dispatches through ALL_ATTENTION_FUNCTIONS
# using config._attn_implementation. Probe each non-eager backend once with a
# real coded forward and parametrize over the ones that work on this platform.
# (The counting head always uses SDPA internally regardless of this setting.)

_NON_EAGER_BACKENDS = sorted(
    name for name in ALL_ATTENTION_FUNCTIONS if "eager" not in name
)


def _probe_coded_backend(name):
    """Return (ok, reason): can the coded memory head use backend ``name`` here?"""
    if name not in ALL_ATTENTION_FUNCTIONS:
        return False, "not registered in ALL_ATTENTION_FUNCTIONS"
    try:
        cfg = _MockSwitchConfig(
            "multi",
            ATOK_NO_BASE,
            [1, 2],
            backend=name,
        )
        switch = create_switch(cfg, layer_idx=0)
        seq = torch.tensor([[TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN]])
        atok = torch.tensor(ATOK_NO_BASE)
        adapter_indices, _ = switch.forward(input_ids=seq, adapter_token_ids=atok)
        got = adapter_indices[0].tolist()
        if got != [0, 1, 1, 2, 2]:
            return False, f"coded routing wrong: {got}"
        return True, "ok"
    except Exception as e:
        return False, str(e).split("\n")[0]


_BACKEND_PROBE_RESULTS = {
    name: _probe_coded_backend(name) for name in _NON_EAGER_BACKENDS
}
_AVAILABLE_BACKENDS = [
    name for name in _NON_EAGER_BACKENDS if _BACKEND_PROBE_RESULTS[name][0]
]


@pytest.fixture(params=_AVAILABLE_BACKENDS or ["sdpa"])
def backend(request):
    """Working non-eager HF attention backend for the coded memory head.

    Falls back to ``sdpa`` if the probe rejected everything (sdpa is the
    coded engine's internal default and is available everywhere torch is).
    To inspect which backends were excluded and why:
        python -c "from tests.hf.test_multi_switch import _BACKEND_PROBE_RESULTS; \\
                    print({k: v for k, v in _BACKEND_PROBE_RESULTS.items() if not v[0]})"
    """
    return request.param


# ── _run adapter (coded engine) ─────────────────────────────────────


class _HFMultiSwitchBase:
    """Provides ``_run()`` for the shared mixins.

    ``switch_type`` is a class attribute set by each concrete Test class.
    ``backend`` (fixture) drives the coded engine's memory-head attention.
    """

    switch_type = None  # overridden per subclass

    @pytest.fixture(autouse=True)
    def _set_backend(self, backend):
        self._backend = backend

    def _run(self, seq, adapter_token_ids):
        # Substitute ids: base-reset layout maps slot0->0; else i+1 per slot.
        if len(adapter_token_ids) == NUM_ADAPTERS + 1:
            sub_ids = list(range(len(adapter_token_ids)))  # [0,1,2]
        else:
            sub_ids = [i + 1 for i in range(len(adapter_token_ids))]  # [1,2]
        cfg = _MockSwitchConfig(
            self.switch_type,
            list(adapter_token_ids),
            sub_ids,
            backend=self._backend,
        )
        switch = create_switch(cfg, layer_idx=0)
        input_ids = torch.tensor([seq])
        atok = torch.tensor(adapter_token_ids)
        adapter_indices, _modified = switch.forward(
            input_ids=input_ids,
            adapter_token_ids=atok,
        )
        return adapter_indices[0].tolist()


# ── Shared test classes ─────────────────────────────────────────────
#
# One concrete class per case-mixin. Keeping them explicit (rather than
# metaprogrammed) means -k selection and failure names stay readable, e.g.
# `TestTransitionsCoded::test_higher_then_lower_A2_then_B1`.


class TestTransitionsCoded(_HFMultiSwitchBase, MultiSwitchTransitionCases):
    switch_type = "multi"


class TestReturnToBaseCoded(_HFMultiSwitchBase, MultiSwitchReturnToBaseCases):
    switch_type = "multi"


class TestStickyCoded(_HFMultiSwitchBase, MultiSwitchStickyCases):
    switch_type = "multi"


class TestEdgeCasesCoded(_HFMultiSwitchBase, MultiSwitchEdgeCases):
    switch_type = "multi"


class TestShapeCoded(_HFMultiSwitchBase, MultiSwitchShapeCorrectnessCases):
    switch_type = "multi"


# ── Engine-specific structural tests ────────────────────────────────


class TestNumCacheLayers:
    """create_switch reports the right cache-slot count for the coded engine."""

    @pytest.mark.parametrize("switch_type", SWITCH_TYPES)
    def test_num_cache_layers(self, switch_type):
        cfg = _MockSwitchConfig(switch_type, ATOK_NO_BASE, [1, 2])
        switch = create_switch(cfg, layer_idx=0)
        assert switch.num_cache_layers == EXPECTED_CACHE_LAYERS[switch_type]

    def test_builds_multiswitch(self):
        """create_switch builds a MultiSwitch (the only engine)."""
        coded = create_switch(_MockSwitchConfig("multi", ATOK_NO_BASE, [1, 2]))
        assert isinstance(coded, MultiSwitch)


class TestCodedLayerIdxSlots:
    """The coded engine reserves counting=layer_idx, memory=layer_idx+1."""

    @pytest.mark.parametrize("layer_idx", [0, 1, 5])
    def test_counting_and_memory_slots(self, layer_idx):
        cfg = _MockSwitchConfig("multi", ATOK_NO_BASE, [1, 2])
        switch = create_switch(cfg, layer_idx=layer_idx)
        assert switch.counting_layer_idx == layer_idx
        assert switch.memory_layer_idx == layer_idx + 1


class TestCodesExactRetrieval:
    """Codes round-trip: recover_count_from_signal inverts 1/(1+n) exactly."""

    @pytest.mark.parametrize("n", [0, 1, 50, 127])
    def test_count_round_trips(self, n):
        g = KerdockDGCodeGenerator(m=6, verbose=False)
        signal = torch.tensor([1.0 / (1.0 + n)])
        recovered = recover_count_from_signal(signal, capacity=g.capacity)
        assert recovered.item() == n

    def test_batched_round_trip(self):
        """A whole batch of counts inverts in one call."""
        g = KerdockDGCodeGenerator(m=6, verbose=False)
        counts = torch.arange(0, 128)
        signal = 1.0 / (1.0 + counts.float())
        recovered = recover_count_from_signal(signal, capacity=g.capacity)
        assert recovered.tolist() == counts.tolist()

    def test_codebook_geometry(self):
        """Kerdock m=6 gives N=64, capacity=2048 (the coded engine relies on it)."""
        g = KerdockDGCodeGenerator(m=6, verbose=False)
        assert g.N == 64
        assert g.capacity == 2048

    # ── memory_gain validation ──────────────────────────────────────────
    # The coded memory head writes key = code(n) * memory_gain, value = expert_id,
    # and queries with code(n); softmax over coded keys should read back the exact
    # expert_id (round + clamp). The gain must be large enough that Kerdock/DG's
    # low mutual coherence resolves the matching address across the whole codebook.
    #
    # A gain sweep (fp32, num_adapters=8, distinct ids written at up to `capacity`
    # addresses) shows: gain<=10 FAILS at high write counts; gain=12 is exact but
    # thin (err~0.11 at n=2048); gain>=16 is exact with near-zero error to full
    # capacity; the default 28.0 has zero error everywhere. So 28.0 is SAFE (not
    # minimal); >=16 is the smallest with solid fp32 margin, and >=20 leaves slack
    # for the vLLM bf16 path. These tests pin that: exact at the default and at 16,
    # and NOT exact at a too-small gain (guards against silently lowering it).
    def _memory_retrieval_exact(self, gain, n_writes, num_adapters=8):
        g = KerdockDGCodeGenerator(m=6, verbose=False)
        cb = g.precompute_codebook(dtype=torch.float32)
        idx = torch.arange(n_writes)
        codes = cb[idx].to(torch.float32)  # queries == unscaled keys
        keys = codes * gain
        vals = (idx % num_adapters).to(torch.float32)  # expert ids at each address
        w = torch.softmax(codes @ keys.t(), dim=1)
        out = (w @ vals).round().clamp(0, num_adapters)
        return bool((out == vals).all())

    @pytest.mark.parametrize("gain", [16.0, 28.0])
    def test_memory_gain_exact_to_capacity(self, gain):
        """Default (28.0) and the validated-minimum (16.0) retrieve exactly at capacity."""
        assert self._memory_retrieval_exact(gain, n_writes=2048)

    def test_memory_gain_too_small_fails(self):
        """A too-small gain (8.0) is NOT exact at high write counts — guards against
        silently lowering ms_memory_gain below the validated floor."""
        assert not self._memory_retrieval_exact(8.0, n_writes=2048)

    # The bf16 counting-signal ceiling (CLAUDE.md gotcha 11) is pinned in
    # tests/unit/test_counting_ceiling.py, which locates the exact first
    # mis-recovery and ties it to Conversation.MAX_RETAINED_CONTROL_TOKENS,
    # and on the real engine in tests/vllm/test_multi_switch.py::TestCountingCeiling.


# ── Regression: real-model attention mask must not change routing ────────────
#
# The counting + memory heads run in fp32 for precision, but the enclosing model
# builds its attention_mask in the MODEL dtype (bf16 in real deployments). An
# earlier version fed that incoming mask straight to the heads' SDPA calls, which
# forced the fp32 attention down to bf16 and corrupted the 1/(1+n) counting —
# routing came out off-by-one (`[T,A,T,B,T]` -> `[0,0,0,1,1]`) on the real model
# while the isolated tests (no mask, or an fp32 mask) passed. The switch now
# builds its own fp32 causal mask internally and ignores the incoming one. These
# tests pin that: routing is identical whether the caller passes no mask, an fp32
# causal mask, or a bf16 causal mask.


class TestAttentionMaskDtypeIndependence:
    """Routing is invariant to the dtype of the model-supplied attention_mask."""

    def _route(self, seq, attention_mask=None):
        cfg = _MockSwitchConfig("multi", ATOK_NO_BASE, [1, 2])
        switch = create_switch(cfg, layer_idx=0)
        kw = dict(
            input_ids=torch.tensor([seq]), adapter_token_ids=torch.tensor(ATOK_NO_BASE)
        )
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask
        ai, _ = switch.forward(**kw)
        return ai[0].tolist()

    @staticmethod
    def _causal_mask(t, dtype):
        neg = torch.finfo(dtype).min
        return (
            torch.triu(torch.full((t, t), neg), diagonal=1).view(1, 1, t, t).to(dtype)
        )

    def test_bf16_mask_does_not_shift_routing(self):
        """A bf16 causal mask (what a bf16 model builds) must NOT change routing —
        this is the exact real-model condition that produced the off-by-one."""
        seq = [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN]
        assert self._route(seq, self._causal_mask(len(seq), torch.bfloat16)) == [
            0,
            1,
            1,
            2,
            2,
        ]

    def test_mask_dtypes_agree(self):
        """No mask, fp32 mask, and bf16 mask all yield the same routing."""
        seq = [B_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN]  # A(2)->B(1): expects [2,2,1,1]
        none_mask = self._route(seq)
        fp32_mask = self._route(seq, self._causal_mask(len(seq), torch.float32))
        bf16_mask = self._route(seq, self._causal_mask(len(seq), torch.bfloat16))
        assert none_mask == fp32_mask == bf16_mask == [2, 2, 1, 1]

    def test_bf16_autocast_stays_correct(self):
        """The coded heads run correctly when the caller is under bf16 autocast
        (the real model's context). The forward disables autocast around both
        heads so their fp32 code(n)*memory_gain softmax cannot be cast to bf16.

        NOTE: on GPU, letting the memory head autocast to bf16 overflows its
        softmax to NaN and routes every token to base — this was a real bug caught
        only by the GPU end-to-end test (tests/hf/test_multi_switch_e2e.py), NOT by
        CPU (CPU autocast does not reproduce the overflow). This CPU test documents
        the intended behavior and guards the no-NaN / correct-routing contract; the
        GPU e2e is the authoritative regression gate for the overflow itself."""
        seq = [B_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN]  # expects [2,2,1,1]
        cfg = _MockSwitchConfig("multi", ATOK_NO_BASE, [1, 2])
        switch = create_switch(cfg, layer_idx=0)
        atok = torch.tensor(ATOK_NO_BASE)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ai, _ = switch.forward(
                input_ids=torch.tensor([seq]), adapter_token_ids=atok
            )
        assert not torch.isnan(ai.float()).any(), (
            "routing produced NaN under bf16 autocast"
        )
        assert ai[0].tolist() == [2, 2, 1, 1]


# ── HF-only: batch independence ─────────────────────────────────────


class TestBatchProcessing:
    """Batch independence for the coded engine (HF batches [batch, seq])."""

    @pytest.mark.parametrize("switch_type", SWITCH_TYPES)
    def test_batch_independence(self, switch_type):
        cfg = _MockSwitchConfig(switch_type, ATOK_NO_BASE, [1, 2])
        switch = create_switch(cfg, layer_idx=0)
        atok = torch.tensor(ATOK_NO_BASE)
        input_ids = torch.tensor(
            [
                [TEXT_TOKEN, A_TOK, TEXT_TOKEN, B_TOK, TEXT_TOKEN],  # 0,1,1,2,2
                [B_TOK, TEXT_TOKEN, A_TOK, TEXT_TOKEN, TEXT_TOKEN],  # 2,2,1,1,1
            ]
        )
        adapter_indices, _ = switch.forward(
            input_ids=input_ids,
            adapter_token_ids=atok,
        )
        assert adapter_indices[0].tolist() == [0, 1, 1, 2, 2]
        assert adapter_indices[1].tolist() == [2, 2, 1, 1, 1]
