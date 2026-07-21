# SPDX-License-Identifier: Apache-2.0
"""TP-aware LoRA weight slicing tests (inner file — run by test_tp_lora.py).

Tests the tensor parallelism slicing math in SwitchedLoRALinear:
- slice_lora_a_weight / slice_lora_b_weight correctness
- weight_loader attachment and slicing
- Packed module (multi-slice) support

Uses plain nn.Linear as the base layer with manually set TP attributes,
avoiding vLLM parallel layer construction (which requires distributed init).
Integration tests in test_tp_integration.py cover real vLLM TP end-to-end.

Requires CUDA (SwitchedLoRALinear uses device from base_layer.weight).
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import():
    try:
        import granite_switch.vllm.core.lora  # noqa: F401

        return True
    except ImportError:
        return False


_IMPORT_OK = _try_import() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _IMPORT_OK,
    reason="requires CUDA GPU and granite_switch.vllm installed",
)

if _IMPORT_OK:
    from granite_switch.vllm.core.lora import SwitchedLoRALinear

NUM_ADAPTERS = 4
RANK = 8
HIDDEN = 64
OUT = 32


def _make_layer(
    is_column_parallel,
    is_row_parallel,
    tp_size,
    tp_rank,
    num_slices=1,
    output_slices=None,
    reduce_results=False,
):
    """Create a SwitchedLoRALinear wrapping a plain nn.Linear.

    Manually sets TP attributes after construction to avoid needing
    vLLM's distributed init.
    """
    base = nn.Linear(HIDDEN, OUT, bias=False, dtype=torch.bfloat16, device="cuda")

    with (
        patch(
            "granite_switch.vllm.core.lora.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ),
        patch(
            "granite_switch.vllm.core.lora.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ),
    ):
        layer = SwitchedLoRALinear(
            base,
            NUM_ADAPTERS,
            RANK,
            num_slices=num_slices,
            output_slices=output_slices,
        )

    layer._is_column_parallel = is_column_parallel
    layer._is_row_parallel = is_row_parallel
    layer._row_parallel_reduce = is_row_parallel and tp_size > 1 and reduce_results
    return layer


class TestTP1IsNoop:
    """With tp_size=1, slicing methods should return the input unchanged."""

    def test_slice_a_noop(self):
        layer = _make_layer(False, True, tp_size=1, tp_rank=0)
        w = torch.randn(NUM_ADAPTERS, 1, RANK, HIDDEN)
        assert torch.equal(layer.slice_lora_a_weight(w), w)

    def test_slice_b_noop(self):
        layer = _make_layer(True, False, tp_size=1, tp_rank=0)
        w = torch.randn(NUM_ADAPTERS, 1, OUT, RANK)
        assert torch.equal(layer.slice_lora_b_weight(w), w)


class TestColumnParallelSlicing:
    """Column-parallel: lora_A unchanged, lora_B sliced on output dim."""

    def test_lora_a_unchanged(self):
        layer = _make_layer(True, False, tp_size=2, tp_rank=0)
        w = torch.randn(NUM_ADAPTERS, 1, RANK, HIDDEN)
        assert torch.equal(layer.slice_lora_a_weight(w), w)

    def test_lora_b_sliced_rank0(self):
        layer = _make_layer(True, False, tp_size=2, tp_rank=0)
        full_out = OUT * 2
        w = torch.randn(NUM_ADAPTERS, 1, full_out, RANK)
        result = layer.slice_lora_b_weight(w)
        assert result.shape == (NUM_ADAPTERS, 1, OUT, RANK)
        assert torch.equal(result, w[:, :, :OUT, :])

    def test_lora_b_sliced_rank1(self):
        layer = _make_layer(True, False, tp_size=2, tp_rank=1)
        full_out = OUT * 2
        w = torch.randn(NUM_ADAPTERS, 1, full_out, RANK)
        result = layer.slice_lora_b_weight(w)
        assert result.shape == (NUM_ADAPTERS, 1, OUT, RANK)
        assert torch.equal(result, w[:, :, OUT:, :])

    def test_both_ranks_cover_full_weight(self):
        full_out = OUT * 2
        w = torch.randn(NUM_ADAPTERS, 1, full_out, RANK)
        r0 = _make_layer(True, False, tp_size=2, tp_rank=0).slice_lora_b_weight(w)
        r1 = _make_layer(True, False, tp_size=2, tp_rank=1).slice_lora_b_weight(w)
        assert torch.equal(torch.cat([r0, r1], dim=-2), w)


class TestRowParallelSlicing:
    """Row-parallel: lora_A sliced on input dim, lora_B unchanged."""

    def test_lora_b_unchanged(self):
        layer = _make_layer(False, True, tp_size=2, tp_rank=0)
        w = torch.randn(NUM_ADAPTERS, 1, OUT, RANK)
        assert torch.equal(layer.slice_lora_b_weight(w), w)

    def test_lora_a_sliced_rank0(self):
        layer = _make_layer(False, True, tp_size=2, tp_rank=0)
        full_in = HIDDEN * 2
        w = torch.randn(NUM_ADAPTERS, 1, RANK, full_in)
        result = layer.slice_lora_a_weight(w)
        assert result.shape == (NUM_ADAPTERS, 1, RANK, HIDDEN)
        assert torch.equal(result, w[:, :, :, :HIDDEN])

    def test_lora_a_sliced_rank1(self):
        layer = _make_layer(False, True, tp_size=2, tp_rank=1)
        full_in = HIDDEN * 2
        w = torch.randn(NUM_ADAPTERS, 1, RANK, full_in)
        result = layer.slice_lora_a_weight(w)
        assert result.shape == (NUM_ADAPTERS, 1, RANK, HIDDEN)
        assert torch.equal(result, w[:, :, :, HIDDEN:])

    def test_both_ranks_cover_full_weight(self):
        full_in = HIDDEN * 2
        w = torch.randn(NUM_ADAPTERS, 1, RANK, full_in)
        r0 = _make_layer(False, True, tp_size=2, tp_rank=0).slice_lora_a_weight(w)
        r1 = _make_layer(False, True, tp_size=2, tp_rank=1).slice_lora_a_weight(w)
        assert torch.equal(torch.cat([r0, r1], dim=-1), w)


class TestPackedSlicing:
    """Packed module (num_slices > 1): separate LoRA per slice."""

    def test_packed_creates_slices(self):
        layer = _make_layer(
            True,
            False,
            tp_size=2,
            tp_rank=0,
            num_slices=3,
            output_slices=(OUT, OUT, OUT),
        )
        assert layer.num_slices == 3
        assert len(layer.lora_A_slices) == 3
        assert len(layer.lora_B_slices) == 3

    def test_packed_b_slices_each_sliced_correctly(self):
        """Each lora_B slice should be independently sliced on its output dim."""
        slice_sizes = (64, 32, 16)
        layer = _make_layer(
            True,
            False,
            tp_size=2,
            tp_rank=0,
            num_slices=3,
            output_slices=slice_sizes,
        )
        for i, full_out in enumerate(slice_sizes):
            w = torch.randn(NUM_ADAPTERS, 1, full_out, RANK)
            result = layer.slice_lora_b_weight(w, slice_idx=i)
            expected_out = full_out // 2
            assert result.shape == (NUM_ADAPTERS, 1, expected_out, RANK), (
                f"Slice {i}: expected out={expected_out}, got {result.shape[-2]}"
            )
            assert torch.equal(result, w[:, :, :expected_out, :])

    def test_packed_a_slices_unchanged_for_column_parallel(self):
        """Column-parallel: each lora_A slice should be unchanged (input is full)."""
        slice_sizes = (64, 32, 16)
        layer = _make_layer(
            True,
            False,
            tp_size=2,
            tp_rank=0,
            num_slices=3,
            output_slices=slice_sizes,
        )
        for i in range(3):
            w = torch.randn(NUM_ADAPTERS, 1, RANK, HIDDEN)
            result = layer.slice_lora_a_weight(w, slice_idx=i)
            assert torch.equal(result, w), f"Slice {i}: lora_A should be unchanged"

    def test_packed_b_both_ranks_reconstruct(self):
        """Both TP ranks should reconstruct the full weight for each slice."""
        slice_sizes = (64, 32, 16)
        for i, full_out in enumerate(slice_sizes):
            w = torch.randn(NUM_ADAPTERS, 1, full_out, RANK)
            r0 = _make_layer(
                True,
                False,
                tp_size=2,
                tp_rank=0,
                num_slices=3,
                output_slices=slice_sizes,
            ).slice_lora_b_weight(w, slice_idx=i)
            r1 = _make_layer(
                True,
                False,
                tp_size=2,
                tp_rank=1,
                num_slices=3,
                output_slices=slice_sizes,
            ).slice_lora_b_weight(w, slice_idx=i)
            assert torch.equal(torch.cat([r0, r1], dim=-2), w), (
                f"Slice {i}: ranks don't reconstruct full weight"
            )


class TestWeightLoaderAttached:
    """Verify weight_loader is present on all LoRA parameters."""

    def test_single_slice_has_loaders(self):
        layer = _make_layer(True, False, tp_size=2, tp_rank=0)
        assert callable(getattr(layer.lora_A, "weight_loader", None))
        assert callable(getattr(layer.lora_B, "weight_loader", None))

    def test_packed_slices_have_loaders(self):
        layer = _make_layer(
            True,
            False,
            tp_size=2,
            tp_rank=0,
            num_slices=3,
            output_slices=(OUT, OUT, OUT),
        )
        for p in layer.lora_A_slices:
            assert callable(getattr(p, "weight_loader", None))
        for p in layer.lora_B_slices:
            assert callable(getattr(p, "weight_loader", None))

    def test_tp1_loaders_are_identity(self):
        layer = _make_layer(True, False, tp_size=1, tp_rank=0)
        w = torch.randn_like(layer.lora_B.data)
        layer.lora_B.weight_loader(layer.lora_B, w)
        assert torch.equal(layer.lora_B.data, w)


class TestWeightLoaderSlices:
    """Verify weight_loader correctly slices checkpoint weights."""

    def test_column_parallel_b_loader(self):
        layer = _make_layer(True, False, tp_size=2, tp_rank=0)
        sharded_out = layer.lora_B.shape[-2]
        full_out = sharded_out * 2
        full_weight = torch.randn(
            NUM_ADAPTERS, 1, full_out, RANK, device="cuda", dtype=torch.bfloat16
        )
        layer.lora_B.weight_loader(layer.lora_B, full_weight)
        expected = full_weight[:, :, :sharded_out, :]
        assert torch.equal(layer.lora_B.data, expected)

    def test_row_parallel_a_loader(self):
        layer = _make_layer(False, True, tp_size=2, tp_rank=1)
        sharded_in = layer.lora_A.shape[-1]
        full_in = sharded_in * 2
        full_weight = torch.randn(
            NUM_ADAPTERS, 1, RANK, full_in, device="cuda", dtype=torch.bfloat16
        )
        layer.lora_A.weight_loader(layer.lora_A, full_weight)
        expected = full_weight[:, :, :, sharded_in:]
        assert torch.equal(layer.lora_A.data, expected)
