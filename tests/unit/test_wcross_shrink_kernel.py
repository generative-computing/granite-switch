# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shrink-only ("W-less") SWITCH kernel — the SR W_cross shunt.

``switch_lora_shrink_expand`` is the one new kernel for Shadow Residual: it writes
``out[m] = (x[m] @ lora_A.T) @ lora_B.T`` for adapter tokens (kernel-local id > 0)
and exactly zero for base tokens (id 0). Unlike ``switch_lora_expand`` there is no
base region in ``x_ext`` (``w_ext_cross`` is built from an empty base ``W``), so the
delta is STORED to a fresh pre-zeroed buffer rather than accumulated in place.

These tests build the packed metadata (``x_ext`` shrink columns, ``lb_packed``,
``tile_slice``, ``slice_col_r``, ``bitmask``) by hand — independently of
``finalize_weights`` — and compare the kernel against a pure-torch reference.
Covered: single tier, mixed rank tiers, the all-base tile (must be exactly zero),
and multiple adapters per tier.

Requires CUDA GPU (Triton kernel).
"""

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE, reason="requires CUDA GPU (Triton kernel)"
)

if _CUDA_AVAILABLE:
    from granite_switch.kernels import (
        BLOCK_M,
        SUPPORTED_RANKS,
        build_w_ext,
        switch_lora_shrink_expand,
    )


# ── helpers ───────────────────────────────────────────────────────────


def _build_packed(adapters, K, N, dev):
    """Build (w_ext_cross, lb_packed, slice_col_r, na, col_of) from adapters.

    ``adapters`` is a list of ``(r, A[r,K], B[N,r])`` ordered by ASCENDING rank
    tier — the kernel numbers kernel-local ids 1.. in that order. Single-slice
    (S=1) shunt layout, matching ``build_w_ext`` and the packed ``lora_B`` layout
    the expand kernel reads.
    """
    # Group lora_A rows by tier for build_w_ext: {rank: [n_r, S=1, r, K]}.
    by_rank = {}
    for r, A, _B in adapters:
        by_rank.setdefault(r, []).append(A)
    lora_A_by_rank = {
        r: torch.stack(mats, dim=0).unsqueeze(1)  # [n_r, 1, r, K]
        for r, mats in by_rank.items()
    }
    w_ext_cross = build_w_ext(
        torch.empty(0, K, device=dev, dtype=adapters[0][1].dtype), lora_A_by_rank
    )  # [sum_r n_r*r, K]

    # na (per-tier counts), slice_col_r base columns (tier block start, S=1).
    na = [sum(1 for r, _, _ in adapters if r == R) for R in SUPPORTED_RANKS]
    slice_col = torch.zeros(1 * 6, dtype=torch.int32, device=dev)
    running = 0
    for k, R in enumerate(SUPPORTED_RANKS):
        slice_col[k] = running
        running += na[k] * R  # S == 1

    # lb_packed: concat over present tiers (ascending) of [n_r, N, r] row-major.
    lb_blocks = []
    for R in SUPPORTED_RANKS:
        tier_B = [B for r, _, B in adapters if r == R]  # each [N, r]
        if tier_B:
            lb_blocks.append(torch.stack(tier_B, dim=0).reshape(-1))  # [n_r*N*r]
    lb_packed = torch.cat(lb_blocks, dim=0).contiguous()

    # Per-adapter shrink-column start (ascending order => cumulative ranks).
    col_of, c = [], 0
    for r, _, _ in adapters:
        col_of.append(c)
        c += r
    return w_ext_cross, lb_packed, slice_col, tuple(na), col_of


def _reference(x_ext, ids, adapters, col_of, N):
    """Pure-torch reference: (x_ext_shrink @ B.T) for id>0, else 0."""
    M = x_ext.shape[0]
    ref = torch.zeros(M, N, dtype=torch.float32, device=x_ext.device)
    for m in range(M):
        idm = int(ids[m].item())
        if idm > 0:
            r, _A, B = adapters[idm - 1]
            c = col_of[idm - 1]
            shrink = x_ext[m, c : c + r].float()  # [r]
            ref[m] = shrink @ B.float().t()  # [N]
    return ref


def _bitmask(ids, M):
    """Per-tile int64 bitmask: bit (id-1) set iff adapter id present in the tile."""
    n_tiles = (M + BLOCK_M - 1) // BLOCK_M
    bm = torch.zeros(n_tiles, dtype=torch.int64, device=ids.device)
    for m in range(M):
        idm = int(ids[m].item())
        if idm > 0:
            bm[m // BLOCK_M] |= 1 << (idm - 1)
    return bm


def _run(adapters, ids, K, N, block_n, dev="cuda", seed=0):
    assert N % block_n == 0, "block_n must divide N"
    torch.manual_seed(seed)
    M = ids.shape[0]
    # Move ids to the device FIRST: _bitmask() builds its tensor on ids.device,
    # and every caller here builds ids on the CPU. Converting only at the kernel
    # call site left the bitmask on the CPU, which Triton rejects with
    # "Pointer argument cannot be accessed from Triton (cpu tensor?)".
    ids = ids.to(dev)
    w_ext_cross, lb_packed, slice_col, na, col_of = _build_packed(adapters, K, N, dev)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    x_ext = (x @ w_ext_cross.t()).contiguous()  # [M, sum_r n_r*r] bf16
    # Independently pin build_w_ext row order: adapter i's shrink columns
    # [col_of[i], col_of[i]+r) must be exactly x @ A_i.T. Without this, both the
    # kernel and the reference index the SAME x_ext columns, so a build_w_ext
    # ordering bug would be masked (they'd agree on wrongly-ordered columns).
    for i, (r_i, A_i, _B_i) in enumerate(adapters):
        c_i = col_of[i]
        torch.testing.assert_close(
            x_ext[:, c_i : c_i + r_i].float(),
            (x @ A_i.t()).float(),
            rtol=2e-2,
            atol=2e-2,
        )
    tile_slice = torch.zeros(
        (N + block_n - 1) // block_n, dtype=torch.int32, device=dev
    )
    bitmask = _bitmask(ids, M)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

    switch_lora_shrink_expand(
        out,
        x_ext,
        ids,
        bitmask,
        lb_packed,
        tile_slice,
        slice_col,
        na,
        1,
        block_n,
        N,
    )
    ref = _reference(x_ext, ids, adapters, col_of, N)
    return out, ref


def _adapter(r, K, N, dev, scale=0.05):
    A = torch.randn(r, K, device=dev, dtype=torch.bfloat16) * scale
    B = torch.randn(N, r, device=dev, dtype=torch.bfloat16) * scale
    return (r, A, B)


# ── tests ─────────────────────────────────────────────────────────────


def test_single_tier_mixed_rows():
    """One tier-16 adapter, mixed base/adapter tokens; base rows exactly zero."""
    dev = "cuda"
    K, N, block_n = 256, 64, 32
    adapters = [_adapter(16, K, N, dev)]
    # 40 tokens: alternate base(0)/adapter(1) — spans >1 row-tile (BLOCK_M=32).
    ids = torch.tensor([0, 1] * 20, dtype=torch.int32)
    out, ref = _run(adapters, ids, K, N, block_n)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
    # base tokens (id 0) must be EXACTLY zero (SR invariant).
    base_rows = (ids == 0).to(dev)
    assert torch.count_nonzero(out[base_rows]) == 0


def test_all_base_tile_is_zero():
    """A tile with no adapter tokens is skipped; pre-zeroed out stays exactly 0."""
    dev = "cuda"
    K, N, block_n = 256, 64, 32
    adapters = [_adapter(16, K, N, dev)]
    ids = torch.zeros(32, dtype=torch.int32)  # one full tile, all base
    out, ref = _run(adapters, ids, K, N, block_n)
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=0)


def test_mixed_rank_tiers():
    """Two adapters across tiers (r=16, r=32); kernel == reference."""
    dev = "cuda"
    K, N, block_n = 512, 128, 32
    adapters = [_adapter(16, K, N, dev), _adapter(32, K, N, dev)]  # ascending
    # ids in {0,1,2}: base, tier-16 adapter, tier-32 adapter.
    ids = torch.tensor([0, 1, 2, 2, 0, 1] * 8, dtype=torch.int32)  # 48 tokens
    out, ref = _run(adapters, ids, K, N, block_n)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_multiple_adapters_same_tier():
    """Two tier-16 adapters (kernel-local ids 1,2); disjoint shrink columns."""
    dev = "cuda"
    K, N, block_n = 256, 64, 32
    adapters = [
        _adapter(16, K, N, dev, scale=0.05),
        _adapter(16, K, N, dev, scale=0.07),
    ]
    ids = torch.tensor([1, 2, 0, 1, 2, 0] * 8, dtype=torch.int32)  # 48 tokens
    out, ref = _run(adapters, ids, K, N, block_n)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
