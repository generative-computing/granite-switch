# SPDX-License-Identifier: Apache-2.0
"""Regression: SWITCH expand kernels must index x_ext with 64-bit offsets.

The expand kernels address x_ext as ``m * stride_xe_m + col``. On granite-4.1-30b
the shared-MLP gate/up row width is 2 * 32768 + shrink columns (~65.5k), so a
forward of ~32.8k tokens pushes ``m * stride_xe_m`` past 2^31. With int32 index
math the offset wrapped and the kernel hit a CUDA illegal memory access (#160).

Each test builds an x_ext whose last row-tile sits past the int32 limit, puts an
adapter on the first and the last tile, and checks both against a pure-torch
reference. Only those two tiles hold data; the rest of x_ext is zero, which keeps
the reference cheap. Needs ~7 GB of free GPU memory; skipped otherwise.

Requires CUDA GPU (Triton kernel).
"""

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _CUDA_AVAILABLE, reason="requires CUDA GPU (Triton kernel)"),
]

if _CUDA_AVAILABLE:
    from granite_switch.kernels import (
        BLOCK_M,
        BLOCK_N,
        switch_lora_expand,
        switch_lora_expand_swiglu,
    )

_INT32_LIMIT = 2**31
_RANK = 16
# granite-4.1-30b intermediate_size: the shape that overflowed in #160.
_H = 32768


def _rows_past_int32(width):
    """Row count (a multiple of BLOCK_M) whose last tile starts past 2^31 elements."""
    first_bad_row = _INT32_LIMIT // width + 1
    return (first_bad_row // BLOCK_M + 2) * BLOCK_M


def _require_free_bytes(n):
    free, _total = torch.cuda.mem_get_info()
    if free < n:
        pytest.skip(
            f"needs {n / 2**30:.1f} GiB free GPU memory, has {free / 2**30:.1f}"
        )


def _active_rows(M, dev):
    """Rows of the first and last row-tile: the only rows that select the adapter."""
    first = torch.arange(0, BLOCK_M, device=dev)
    last = torch.arange(M - BLOCK_M, M, device=dev)
    return torch.cat([first, last])


def _ids_and_bitmask(M, rows, dev):
    ids = torch.zeros(M, dtype=torch.int32, device=dev)
    ids[rows] = 1
    bitmask = torch.zeros((M + BLOCK_M - 1) // BLOCK_M, dtype=torch.int64, device=dev)
    bitmask[rows // BLOCK_M] = 1
    return ids, bitmask


def test_expand_in_place_past_int32_row_offset():
    """switch_lora_expand: one rank-16 adapter, N = 2 * 32768 output columns."""
    dev = "cuda"
    N = 2 * _H
    width = N + _RANK  # base columns + one adapter's shrink columns (S=1)
    M = _rows_past_int32(width)
    assert (M - 1) * width >= _INT32_LIMIT
    _require_free_bytes(M * width * 2 + 2**30)

    torch.manual_seed(0)
    rows = _active_rows(M, dev)
    x_ext = torch.zeros(M, width, device=dev, dtype=torch.bfloat16)
    x_ext[rows] = torch.randn(rows.numel(), width, device=dev, dtype=torch.bfloat16)
    B = torch.randn(N, _RANK, device=dev, dtype=torch.bfloat16) * 0.1
    before = x_ext[rows].float()
    ref = before[:, :N] + before[:, N:] @ B.float().t()

    ids, bitmask = _ids_and_bitmask(M, rows, dev)
    tile_slice = torch.zeros(N // BLOCK_N, dtype=torch.int32, device=dev)
    slice_col = torch.zeros(6, dtype=torch.int32, device=dev)
    slice_col[0] = N  # tier-16 shrink columns start right after the base region
    switch_lora_expand(
        x_ext,
        ids,
        bitmask,
        B.reshape(-1).contiguous(),
        tile_slice,
        slice_col,
        (1, 0, 0, 0, 0, 0),
        1,
        BLOCK_N,
        N,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(x_ext[rows, :N].float(), ref, rtol=2e-2, atol=2e-2)
    # A base row between the two tiles is skipped by the bitmask and stays zero.
    assert torch.count_nonzero(x_ext[M // 2]) == 0


def test_expand_swiglu_past_int32_row_offset():
    """switch_lora_expand_swiglu: the 30B shared-MLP gate/up path, H = 32768."""
    dev = "cuda"
    N = 2 * _H
    width = N + 2 * _RANK  # gate|up base columns + gate and up shrink columns
    M = _rows_past_int32(width)
    assert (M - 1) * width >= _INT32_LIMIT
    _require_free_bytes(M * width * 2 + M * _H * 2 + 2**30)

    torch.manual_seed(0)
    rows = _active_rows(M, dev)
    x_ext = torch.zeros(M, width, device=dev, dtype=torch.bfloat16)
    x_ext[rows] = torch.randn(rows.numel(), width, device=dev, dtype=torch.bfloat16)
    # Packed lora_B for one adapter, [N_total, r]: gate rows [0, H), up rows [H, 2H).
    B = torch.randn(N, _RANK, device=dev, dtype=torch.bfloat16) * 0.1
    xa = x_ext[rows].float()
    gate = xa[:, :_H] + xa[:, N : N + _RANK] @ B[:_H].float().t()
    up = xa[:, _H:N] + xa[:, N + _RANK :] @ B[_H:].float().t()
    ref = torch.nn.functional.silu(gate) * up

    ids, bitmask = _ids_and_bitmask(M, rows, dev)
    # build_w_ext order is tier -> adapter -> slice: gate's shrink block, then up's.
    slice_col = torch.zeros(2 * 6, dtype=torch.int32, device=dev)
    slice_col[0] = N
    slice_col[6] = N + _RANK
    out = torch.empty(M, _H, device=dev, dtype=torch.bfloat16)
    switch_lora_expand_swiglu(
        out,
        x_ext,
        ids,
        bitmask,
        B.reshape(-1).contiguous(),
        slice_col,
        (1, 0, 0, 0, 0, 0),
        2,
        BLOCK_N,
        _H,
        N,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out[rows].float(), ref, rtol=2e-2, atol=3e-2)
    # Zero gate/up on a base row gives silu(0) * 0 == 0.
    assert torch.count_nonzero(out[M // 2]) == 0
