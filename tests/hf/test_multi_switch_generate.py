# SPDX-License-Identifier: Apache-2.0
"""MultiSwitch must survive a real ``generate()``, not just a single forward.

Regression test for a decode-only crash. The coded switch built its internal
causal mask as ``q_len x q_len``, but both of its heads attend over the whole KV
cache, so with a cache ``kv_len > q_len``:

    prefill:  q_len=5, kv_len=5  -> mask 5x5  OK   (q_len == kv_len)
    decode:   q_len=1, kv_len=6  -> mask 1x1  WRONG, needs 1x6

On a real composed 3B this surfaced as

    RuntimeError: (*bias): last dimension must be contiguous

from the counting head's SDPA call -- a shape complaint rather than an actual
stride problem (every tensor was contiguous). The fix builds the mask over the
post-cache-update key length, with the causal diagonal offset by the cached
prefix (``diagonal=1 + (kv_len - q_len)``).

Why every existing suite missed it: they all run ONE forward with
``past_key_values=None``, where ``q_len == kv_len`` and the bug cannot appear.
Nothing called ``generate()``. These tests do, plus a direct
cache-growth check on the bare switch so the failure is pinpointed rather than
just observed at the model level.

CPU-only and fast (tiny synthetic geometry).
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from tests.shared.generation_models import DENSE_CFG, make_switch_model
from tests.shared.multi_switch_cases import ATOK_NO_BASE, TEXT_TOKEN

NUM_ADAPTERS = 2


def _overrides(base_cfg):
    return {
        "vocab_size": max(DENSE_CFG["vocab_size"], max(ATOK_NO_BASE) + 1),
        "num_adapters": NUM_ADAPTERS,
        "adapter_ranks": [8] * NUM_ADAPTERS,
        "adapter_token_ids": list(ATOK_NO_BASE),
        "adapter_names": [f"adapter_{i}" for i in range(NUM_ADAPTERS)],
        "adapter_substitute_token_ids": [1, 2],
        "switch_type": "multi",
        # +2 layers: the coded switch owns 2 cache slots (counting + memory).
        "num_hidden_layers": len(base_cfg["layer_types"]) + 2,
        "layer_types": ["attention", "attention"] + base_cfg["layer_types"],
    }


@pytest.fixture
def model():
    base_cfg = dict(DENSE_CFG)
    m, config = make_switch_model(base_cfg, _overrides(base_cfg))
    m.model.adapter_token_ids.data = torch.tensor(
        config.adapter_token_ids, dtype=torch.long
    )
    return m.eval()


A_TOK, B_TOK = ATOK_NO_BASE


def test_generate_runs(model):
    """generate() must not crash -- this is the whole point of the test."""
    ids = torch.tensor([[TEXT_TOKEN, TEXT_TOKEN, A_TOK, TEXT_TOKEN]])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=4, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 4, (
        f"expected {ids.shape[1] + 4} tokens, got {out.shape[1]}"
    )


def test_generate_longer_run(model):
    """More decode steps than the prompt length, so kv_len >> q_len."""
    ids = torch.tensor([[TEXT_TOKEN, A_TOK]])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 8


def test_generate_no_control_token(model):
    """A prompt with no control token must also generate (base routing)."""
    ids = torch.tensor([[TEXT_TOKEN, TEXT_TOKEN, TEXT_TOKEN]])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=4, do_sample=False)
    assert out.shape[1] == ids.shape[1] + 4


def test_switch_accepts_grown_cache_directly():
    """Bare-switch check: q_len=1 against a grown cache must not raise.

    Pinpoints the mask-shape bug at the switch level rather than only observing
    a model-level crash. Feeds the switch a prefill, then a single decode token
    with the same cache, exactly as generate() does.
    """
    from granite_switch.hf.switch import create_switch
    from tests.hf.test_multi_switch import _MockSwitchConfig

    cfg = _MockSwitchConfig("multi", list(ATOK_NO_BASE), [1, 2], backend="sdpa")
    sw = create_switch(cfg, layer_idx=0)
    atok = torch.tensor(ATOK_NO_BASE)
    cache = DynamicCache()

    prefill = torch.tensor([[TEXT_TOKEN, A_TOK, TEXT_TOKEN]])
    ai, _ = sw.forward(
        input_ids=prefill,
        adapter_token_ids=atok,
        past_key_values=cache,
        cache_position=torch.arange(3),
    )
    assert ai.shape == prefill.shape

    # Decode step: one token, cache already holds 3 -> kv_len=4 > q_len=1.
    step = torch.tensor([[TEXT_TOKEN]])
    ai2, _ = sw.forward(
        input_ids=step,
        adapter_token_ids=atok,
        past_key_values=cache,
        cache_position=torch.tensor([3]),
    )
    assert ai2.shape == step.shape
    # The control token fired at prefill position 1, so the decode token must
    # still route to adapter 1 -- the carry has to survive through the cache.
    assert int(ai2[0, 0]) == 1, (
        f"decode step routed to {int(ai2[0, 0])}, expected 1 (carry from prefill)"
    )


def test_switch_mask_spans_kv_length_not_query_length():
    """The mask handed to SDPA must be [.., q_len, KV_len] -- assert it directly.

    Running generate() on CPU does NOT catch the bug: the CPU SDPA kernel happily
    broadcasts a (1,1,1,1) bias over a 1-query-x-6-key attention, while the CUDA
    kernel rejects it ("(*bias): last dimension must be contiguous"). So a
    CPU-only crash test is not a regression test at all -- verified by
    reintroducing the bug and watching all the generate() tests still pass.

    This asserts the invariant instead of relying on a kernel to complain, so it
    fails on CPU the moment the mask is built from q_len again.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from granite_switch.hf.switch import create_switch
    from tests.hf.test_multi_switch import _MockSwitchConfig

    cfg = _MockSwitchConfig("multi", list(ATOK_NO_BASE), [1, 2], backend="sdpa")
    sw = create_switch(cfg, layer_idx=0)
    atok = torch.tensor(ATOK_NO_BASE)
    cache = DynamicCache()

    seen = []
    real = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def spy(module, q, k, v, mask, **kw):
        seen.append(
            (
                tuple(q.shape),
                tuple(k.shape),
                None if mask is None else tuple(mask.shape),
            )
        )
        return real(module, q, k, v, mask, **kw)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = spy
    try:
        sw.forward(
            input_ids=torch.tensor([[TEXT_TOKEN, A_TOK, TEXT_TOKEN]]),
            adapter_token_ids=atok,
            past_key_values=cache,
            cache_position=torch.arange(3),
        )
        seen.clear()  # keep only the decode step
        sw.forward(
            input_ids=torch.tensor([[TEXT_TOKEN]]),
            adapter_token_ids=atok,
            past_key_values=cache,
            cache_position=torch.tensor([3]),
        )
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = real

    assert seen, "no SDPA calls recorded on the decode step"
    for q_shape, k_shape, m_shape in seen:
        q_len, kv_len = q_shape[2], k_shape[2]
        assert kv_len > q_len, (
            f"expected a grown cache on the decode step, got q={q_len} kv={kv_len}"
        )
        assert m_shape == (1, 1, q_len, kv_len), (
            f"mask shape {m_shape} must be (1, 1, q_len={q_len}, kv_len={kv_len}); "
            "a q_len x q_len mask cannot express 1-query-over-N-keys attention and "
            "is rejected by the CUDA SDPA kernel"
        )


def test_generate_routing_carries_adapter(model):
    """After a control token, generated tokens keep routing to that adapter."""
    ids = torch.tensor([[TEXT_TOKEN, B_TOK, TEXT_TOKEN]])
    with torch.no_grad():
        model.generate(ids, max_new_tokens=3, do_sample=False)
    # The last forward is a decode step; its routing must still be adapter 2.
    route = model.model._last_adapter_indices[0].tolist()
    assert all(v == 2 for v in route), (
        f"decode routing {route} lost the adapter set by the control token"
    )
