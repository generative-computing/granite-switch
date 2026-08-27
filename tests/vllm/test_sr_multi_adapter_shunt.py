# SPDX-License-Identifier: Apache-2.0
"""Multi-adapter kernel state of the Shadow-Residual cross-stream shunt.

``WCrossShunt`` is the one module that exists only on the SR path, and until the
vLLM SR decoder stopped rejecting ``num_adapters > 1`` its multi-adapter state was
never built. This pins that state without a GPU: everything
:meth:`WCrossShunt.finalize_weights` computes — the applicability scan, the tier
assignment, the remap table, the packed ``w_ext_cross`` / ``lora_B`` buffers — is
pure torch, and no kernel is launched.

The invariant that matters is **per-adapter routing**: an adapter's shrink rows and
expand columns must be the ones the kernel reaches through its remap-table entry,
whichever slot it was composed into. Note the buffers themselves are *not*
slot-invariant — the SR decoder interface finalizes with one uniform cross rank
(``[cross_stream_rank] * num_adapters``), so all adapters share a tier and pack in
slot order. What must hold is that the remap table permutes with the packing.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm", reason="WCrossShunt imports vLLM's linear layers")

from granite_switch.kernels import SUPPORTED_RANKS
from granite_switch.vllm.decoder.shadow_residual.wcross_shunt import WCrossShunt

HIDDEN = 64  # divisible by the kernel's default BLOCK_N (32)
#: Per-adapter *trained* cross ranks. Distinct and on-tier, so the zero-padding to
#: the checkpoint's single ``cross_stream_rank`` (the max) is exercised too.
TRAINED_RANKS = {0: 16, 1: 32, 2: 64}
CROSS_RANK = max(TRAINED_RANKS.values())
NUM_ADAPTERS = len(TRAINED_RANKS)
DTYPE = torch.float32


def _identity_weights(identity: int, max_rank: int = CROSS_RANK):
    """(lora_A, lora_B) for one adapter, zero-padded from its rank to ``max_rank``.

    Deterministic in ``identity`` so the same adapter yields the same weights
    wherever it is slotted — mirroring how the composer stacks per-adapter
    cross_stream weights and zero-pads the narrower ranks.
    """
    rank = TRAINED_RANKS[identity]
    gen = torch.Generator().manual_seed(500 + identity)
    a = torch.zeros(max_rank, HIDDEN, dtype=DTYPE)
    b = torch.zeros(HIDDEN, max_rank, dtype=DTYPE)
    a[:rank] = torch.randn(rank, HIDDEN, generator=gen, dtype=DTYPE)
    b[:, :rank] = torch.randn(HIDDEN, rank, generator=gen, dtype=DTYPE)
    return a, b


def _make_shunt(slot_identities, zero_identities=(), ranks=None):
    """A finalized shunt whose slot *i* holds adapter ``slot_identities[i]``.

    ``ranks`` defaults to the uniform ``[CROSS_RANK] * n`` that
    :meth:`SRDecoderInterface.finalize_modules` passes; pass a list to exercise the
    module's per-adapter-rank API. ``zero_identities`` get all-zero weights — an SR
    adapter that does not adapt the cross-stream site.
    """
    n = len(slot_identities)
    shunt = WCrossShunt(
        hidden_size=HIDDEN,
        num_adapters=n,
        cross_rank=CROSS_RANK,
        device=torch.device("cpu"),
        dtype=DTYPE,
    )
    with torch.no_grad():
        for slot, identity in enumerate(slot_identities):
            if identity in zero_identities:
                continue
            a, b = _identity_weights(identity)
            shunt.lora_A[slot, 0] = a
            shunt.lora_B[slot, 0] = b
    shunt.finalize_weights([CROSS_RANK] * n if ranks is None else list(ranks))
    return shunt


def _kernel_index(shunt, slot_identities):
    """{adapter identity: kernel-local index} from the shunt's remap table."""
    return {
        identity: int(shunt.remap_table[slot + 1])
        for slot, identity in enumerate(slot_identities)
    }


def _expected_offsets(na):
    """The ``slice_col_r`` row this ``_na`` tier population implies (S=1, no base)."""
    offsets, running = [], 0
    for count, rank in zip(na, SUPPORTED_RANKS):
        offsets.append(running)
        running += count * rank
    return offsets


def test_uniform_cross_rank_packs_one_tier():
    """Every adapter is finalized at the checkpoint's single cross rank.

    Covers the launch geometry in one place: distinct kernel-local slots, the tier
    population, the packed buffer sizes, and the cumulative column offsets.
    """
    ids = [0, 1, 2]
    shunt = _make_shunt(ids)

    assert shunt._num_applicable == NUM_ADAPTERS
    assert int(shunt.remap_table[0]) == 0, "base must stay kernel-local 0"
    assert sorted(_kernel_index(shunt, ids).values()) == [1, 2, 3]

    expected_na = tuple(NUM_ADAPTERS if r == CROSS_RANK else 0 for r in SUPPORTED_RANKS)
    assert shunt._na == expected_na
    # W-less module: no base region, so w_ext_cross is exactly the shrink rows.
    assert shunt.w_ext_cross.shape == (NUM_ADAPTERS * CROSS_RANK, HIDDEN)
    assert shunt._lb_packed.numel() == NUM_ADAPTERS * HIDDEN * CROSS_RANK
    assert shunt._block_cfg_key == (HIDDEN, HIDDEN, NUM_ADAPTERS)
    assert shunt._N == HIDDEN
    assert shunt.slice_col_r.shape == (1, 6)
    assert shunt.slice_col_r[0].tolist() == _expected_offsets(shunt._na)


def test_each_adapters_rows_are_reachable_through_its_remap_entry():
    """The buffers the kernel reads for adapter X are X's weights, from any slot.

    Follows exactly the path the kernel takes — slot -> ``remap_table`` ->
    kernel-local index -> row block in ``w_ext_cross`` (shrink) and column block in
    the packed ``lora_B`` (expand) — for every permutation of three adapters. All
    share one tier here, so kernel-local index ``k`` owns block ``k-1``.
    """
    reference = _make_shunt([0, 1, 2])
    for slot_identities in ([0, 1, 2], [2, 1, 0], [1, 2, 0], [0, 2, 1]):
        shunt = _make_shunt(slot_identities)
        local = _kernel_index(shunt, slot_identities)
        assert sorted(local.values()) == [1, 2, 3], (slot_identities, local)

        lb = shunt._lb_packed.view(NUM_ADAPTERS, HIDDEN, CROSS_RANK)
        for identity, k in local.items():
            want_a, want_b = _identity_weights(identity)
            got_a = shunt.w_ext_cross[(k - 1) * CROSS_RANK : k * CROSS_RANK]
            assert torch.equal(got_a, want_a), (
                f"slots {slot_identities}: adapter {identity} (kernel-local {k}) "
                "reads the wrong shrink rows"
            )
            assert torch.equal(lb[k - 1], want_b), (
                f"slots {slot_identities}: adapter {identity} (kernel-local {k}) "
                "reads the wrong expand columns"
            )

        # Slot order changes what sits where, never the launch geometry.
        assert shunt._na == reference._na
        assert shunt._num_applicable == reference._num_applicable
        assert shunt._block_cfg_key == reference._block_cfg_key
        assert shunt.w_ext_cross.shape == reference.w_ext_cross.shape
        assert torch.equal(shunt.slice_col_r, reference.slice_col_r)


def test_non_contributing_adapter_drops_out_of_the_tiers():
    """An adapter with no cross_stream weights maps to base and takes no tier."""
    ids = [0, 1, 2]
    shunt = _make_shunt(ids, zero_identities=(1,))
    assert shunt._num_applicable == 2
    local = _kernel_index(shunt, ids)
    assert local[1] == 0, "empty adapter must remap to kernel-local 0 (no delta)"
    assert sorted(v for k, v in local.items() if k != 1) == [1, 2]
    assert shunt.w_ext_cross.shape == (2 * CROSS_RANK, HIDDEN)
    assert shunt._block_cfg_key == (HIDDEN, HIDDEN, 2)


def test_no_applicable_adapters_leaves_the_shunt_inert():
    """All-zero cross_stream everywhere: nothing is packed, forward is zero."""
    shunt = _make_shunt([0, 1, 2], zero_identities=(0, 1, 2))
    assert shunt._num_applicable == 0
    assert torch.equal(shunt.remap_table, torch.zeros(4, dtype=torch.long))
    shunt._module_idx = 0
    out = shunt(torch.randn(5, HIDDEN, dtype=DTYPE))
    assert out.shape == (5, HIDDEN)
    assert not out.any(), "an unpopulated shunt must contribute exactly zero"


def test_per_adapter_ranks_split_into_tiers():
    """The module's per-adapter-rank API tiers by rank, ascending.

    Today's SR wiring always finalizes with one uniform rank (see
    :func:`_make_shunt`), so this covers ``finalize_weights``' documented
    per-adapter form rather than a configuration the decoder produces.
    """
    ranks = [TRAINED_RANKS[i] for i in (0, 1, 2)]
    shunt = _make_shunt([0, 1, 2], ranks=ranks)
    assert shunt._num_applicable == NUM_ADAPTERS
    assert shunt._na == tuple(ranks.count(r) for r in SUPPORTED_RANKS)
    assert shunt.w_ext_cross.shape == (sum(ranks), HIDDEN)
    assert shunt.slice_col_r[0].tolist() == _expected_offsets(shunt._na)
    # Ranks ascend with slot here, so tier order and slot order coincide.
    assert _kernel_index(shunt, [0, 1, 2]) == {0: 1, 1: 2, 2: 3}
