# SPDX-License-Identifier: Apache-2.0
"""HF LoRA layer tests.

Section 1: SwitchedLoRALinear (shared mixins from tests/shared/lora_cases.py)
Section 2: MergedSwitchedLoRALinear (HF-only — fused QKV/gate-up with per-slice LoRA)
Section 3: Input shape handling (2D vs 3D, batched vs single consistency)
"""

import pytest
import torch

from granite_switch.hf.core.lora import SwitchedLoRALinear, MergedSwitchedLoRALinear

from tests.shared.lora_cases import (
    IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK, SEED,
    _seeded_input,
    LoRABasePassthroughCases,
    LoRAAdapterActivationCases,
    LoRABatchIndependenceCases,
    LoRAMathCorrectnessCases,
    LoRAShapeCorrectnessCases,
)

pytestmark = pytest.mark.local_fast


# ════════════════════════════════════════════════════════════════════
# Section 1: SwitchedLoRALinear — shared mixin tests
# ════════════════════════════════════════════════════════════════════

class _HFLoRABase:
    """Provides _make_layer() and _run() for shared mixin tests."""

    def _make_layer(self, in_features, out_features, num_adapters, rank):
        return SwitchedLoRALinear(
            in_features=in_features,
            out_features=out_features,
            num_adapters=num_adapters,
            max_lora_rank=rank,
            bias=True,
        )

    def _run(self, layer, x, adapter_indices):
        return layer.forward(x, adapter_indices)


class TestBasePassthrough(_HFLoRABase, LoRABasePassthroughCases):
    pass


class TestAdapterActivation(_HFLoRABase, LoRAAdapterActivationCases):
    pass


class TestBatchIndependence(_HFLoRABase, LoRABatchIndependenceCases):
    pass


class TestMathCorrectness(_HFLoRABase, LoRAMathCorrectnessCases):
    pass


class TestShapeCorrectness(_HFLoRABase, LoRAShapeCorrectnessCases):
    pass


# ════════════════════════════════════════════════════════════════════
# Section 2: MergedSwitchedLoRALinear — HF-only
# ════════════════════════════════════════════════════════════════════

# Typical QKV slices for a small model: Q=64, K=16, V=16
MERGED_OUTPUT_SLICES = (64, 16, 16)
MERGED_TOTAL_OUT = sum(MERGED_OUTPUT_SLICES)


def _make_merged_layer(
    in_features=IN_FEATURES,
    output_slices=MERGED_OUTPUT_SLICES,
    num_adapters=NUM_ADAPTERS,
    rank=RANK,
):
    return MergedSwitchedLoRALinear(
        in_features=in_features,
        output_slices=output_slices,
        num_adapters=num_adapters,
        max_lora_rank=rank,
        bias=False,
    )


class TestMergedBasePassthrough:
    """All-base → output equals base_layer(x)."""

    def test_all_base_equals_base_layer(self):
        torch.manual_seed(SEED)
        layer = _make_merged_layer()
        x = _seeded_input(2, 5, IN_FEATURES, seed=SEED + 20)
        adapter_indices = torch.zeros(2, 5, dtype=torch.long)

        output = layer.forward(x, adapter_indices)
        expected = layer.base_layer(x)

        torch.testing.assert_close(output, expected)


class TestMergedAdapterActivation:
    """Adapter modifies output in the correct slice region."""

    def test_adapter_modifies_output(self):
        torch.manual_seed(SEED)
        layer = _make_merged_layer()

        # lora_B_slices are zero-initialized; set non-zero weights
        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_B_slices[s].data = torch.randn_like(layer.lora_B_slices[s]) * 0.1

        x = _seeded_input(1, 4, IN_FEATURES, seed=SEED + 21)

        base_indices = torch.zeros(1, 4, dtype=torch.long)
        adapter_indices = torch.ones(1, 4, dtype=torch.long)

        base_output = layer.forward(x, base_indices)
        adapter_output = layer.forward(x, adapter_indices)

        assert not torch.allclose(base_output, adapter_output), \
            "Adapter output should differ from base output"


class TestMergedSliceIndependence:
    """LoRA weights on slice 0 only; slices 1+ output unchanged from base."""

    def test_lora_only_affects_target_slice(self):
        torch.manual_seed(SEED)
        layer = _make_merged_layer()

        # Zero out all LoRA A weights (kills LoRA effect)
        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_A_slices[s].data.zero_()
                layer.lora_B_slices[s].data.zero_()

            # Set non-zero LoRA only on slice 0
            layer.lora_A_slices[0].data[:] = torch.randn_like(layer.lora_A_slices[0])
            layer.lora_B_slices[0].data[:] = torch.randn_like(layer.lora_B_slices[0])

        x = _seeded_input(1, 4, IN_FEATURES, seed=SEED + 22)
        adapter_indices = torch.ones(1, 4, dtype=torch.long)

        output = layer.forward(x, adapter_indices)
        base_output = layer.base_layer(x)

        # Slice 0 should differ (has LoRA)
        slice_0_end = MERGED_OUTPUT_SLICES[0]
        assert not torch.allclose(output[:, :, :slice_0_end], base_output[:, :, :slice_0_end]), \
            "Slice 0 should be modified by LoRA"

        # Slices 1+ should be identical to base (no LoRA)
        torch.testing.assert_close(
            output[:, :, slice_0_end:], base_output[:, :, slice_0_end:],
            msg="Slices 1+ should be unchanged (no LoRA weights)"
        )


class TestMergedMathCorrectness:
    """Known weights → verify per-slice LoRA math."""

    def test_per_slice_lora_math(self):
        torch.manual_seed(SEED)
        layer = _make_merged_layer()

        # Set known weights per slice per adapter
        with torch.no_grad():
            for s in range(layer.num_slices):
                for a in range(NUM_ADAPTERS):
                    out_size = MERGED_OUTPUT_SLICES[s]
                    layer.lora_A_slices[s].data[a, 0] = (
                        torch.eye(RANK, IN_FEATURES) * 0.1 * (s + 1) * (a + 1)
                    )
                    layer.lora_B_slices[s].data[a, 0] = (
                        torch.eye(out_size, RANK) * 0.2 * (s + 1) * (a + 1)
                    )

        x = _seeded_input(1, 3, IN_FEATURES, seed=SEED + 23)

        for adapter_id in range(1, NUM_ADAPTERS + 1):
            adapter_indices = torch.full((1, 3), adapter_id, dtype=torch.long)
            output = layer.forward(x, adapter_indices)
            base_out = layer.base_layer(x)

            # Build expected output slice by slice
            offset = 0
            for s, out_size in enumerate(MERGED_OUTPUT_SLICES):
                tensor_idx = adapter_id - 1
                lora_a = layer.lora_A_slices[s][tensor_idx, 0]
                lora_b = layer.lora_B_slices[s][tensor_idx, 0]
                lora_delta = x @ lora_a.t() @ lora_b.t()
                expected_slice = base_out[:, :, offset:offset + out_size] + lora_delta

                torch.testing.assert_close(
                    output[:, :, offset:offset + out_size], expected_slice,
                    msg=f"Adapter {adapter_id}, slice {s}: math mismatch"
                )
                offset += out_size


class TestMergedBatchIndependence:
    """Batch with different adapters across slices — no cross-talk."""

    def test_batch_different_adapters(self):
        """Each batch row uses a different adapter; verify per-row correctness."""
        torch.manual_seed(SEED)
        layer = _make_merged_layer()

        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_B_slices[s].data = torch.randn_like(layer.lora_B_slices[s]) * 0.1

        batch_size = NUM_ADAPTERS + 1  # base + all adapters
        seq_len = 4

        x = _seeded_input(batch_size, seq_len, IN_FEATURES, seed=SEED + 24)

        # Row 0: base; row i: adapter i
        adapter_indices = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i in range(1, batch_size):
            adapter_indices[i] = i

        batched_output = layer.forward(x, adapter_indices)

        for i in range(batch_size):
            x_single = x[i:i+1]
            idx_single = adapter_indices[i:i+1]
            single_output = layer.forward(x_single, idx_single)

            torch.testing.assert_close(
                batched_output[i], single_output[0],
                msg=f"Row {i} (adapter={adapter_indices[i, 0].item()}) "
                    f"should match single-sequence result"
            )

    def test_batch_mixed_within_sequence(self):
        """Batch where each sequence has intra-sequence adapter switching."""
        torch.manual_seed(SEED)
        layer = _make_merged_layer()

        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_B_slices[s].data = torch.randn_like(layer.lora_B_slices[s]) * 0.1

        x = _seeded_input(3, 5, IN_FEATURES, seed=SEED + 25)

        adapter_indices = torch.tensor([
            [0, 1, 0, 2, 0],
            [2, 0, 1, 1, 0],
            [0, 0, 3, 0, 1],
        ], dtype=torch.long)

        batched_output = layer.forward(x, adapter_indices)

        # Verify every token matches single-token result
        for row in range(3):
            for pos in range(5):
                aid = adapter_indices[row, pos].item()
                x_token = x[row, pos:pos+1].unsqueeze(0)  # (1, 1, features)
                idx_token = torch.tensor([[aid]], dtype=torch.long)
                ref = layer.forward(x_token, idx_token)

                torch.testing.assert_close(
                    batched_output[row, pos], ref[0, 0],
                    msg=f"Row {row}, pos {pos} (adapter={aid}): cross-talk detected"
                )


# ════════════════════════════════════════════════════════════════════
# Section 3: Input shape handling — HF-only
# ════════════════════════════════════════════════════════════════════

class TestInputShapes:
    """2D [num_tokens, features] and 3D [batch, seq, features] both work."""

    def test_2d_input(self):
        """SwitchedLoRALinear accepts 2D input [num_tokens, features]."""
        torch.manual_seed(SEED)
        layer = SwitchedLoRALinear(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x_2d = torch.randn(6, IN_FEATURES)
        indices_1d = torch.tensor([0, 1, 0, 2, 0, 3], dtype=torch.long)

        output = layer.forward(x_2d, indices_1d)
        assert output.shape == (6, OUT_FEATURES)

    def test_3d_input(self):
        """SwitchedLoRALinear accepts 3D input [batch, seq, features]."""
        torch.manual_seed(SEED)
        layer = SwitchedLoRALinear(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x_3d = torch.randn(2, 3, IN_FEATURES)
        indices_2d = torch.tensor([[0, 1, 0], [2, 0, 3]], dtype=torch.long)

        output = layer.forward(x_3d, indices_2d)
        assert output.shape == (2, 3, OUT_FEATURES)

    def test_2d_and_3d_equivalent(self):
        """2D and 3D inputs with same data produce equivalent results."""
        torch.manual_seed(SEED)
        layer = SwitchedLoRALinear(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x_3d = _seeded_input(2, 3, IN_FEATURES, seed=SEED + 30)
        x_2d = x_3d.view(-1, IN_FEATURES)

        indices_2d = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)
        indices_1d = indices_2d.view(-1)

        output_3d = layer.forward(x_3d, indices_2d)
        output_2d = layer.forward(x_2d, indices_1d)

        torch.testing.assert_close(
            output_3d.view(-1, OUT_FEATURES), output_2d,
            msg="2D and 3D inputs should produce equivalent results"
        )

    def test_merged_2d_and_3d_equivalent(self):
        """MergedSwitchedLoRALinear: 2D and 3D inputs produce equivalent results."""
        torch.manual_seed(SEED)
        layer = _make_merged_layer()

        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_B_slices[s].data = torch.randn_like(layer.lora_B_slices[s]) * 0.1

        x_3d = _seeded_input(2, 3, IN_FEATURES, seed=SEED + 31)
        x_2d = x_3d.view(-1, IN_FEATURES)

        indices_2d = torch.tensor([[0, 1, 2], [3, 0, 1]], dtype=torch.long)
        indices_1d = indices_2d.view(-1)

        output_3d = layer.forward(x_3d, indices_2d)
        output_2d = layer.forward(x_2d, indices_1d)

        torch.testing.assert_close(
            output_3d.view(-1, MERGED_TOTAL_OUT), output_2d,
            msg="2D and 3D inputs should produce equivalent results"
        )


class TestBatchedVsSingle:
    """Run N single-sequence forwards, then one batched forward — must be identical."""

    def test_switched_lora_batched_vs_single(self):
        """SwitchedLoRALinear: batched == stacked singles."""
        torch.manual_seed(SEED)
        layer = SwitchedLoRALinear(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        with torch.no_grad():
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        batch_size = 4
        seq_len = 5
        x = _seeded_input(batch_size, seq_len, IN_FEATURES, seed=SEED + 32)

        adapter_indices = torch.tensor([
            [0, 1, 0, 2, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [3, 2, 1, 0, 3],
        ], dtype=torch.long)

        # Batched forward
        batched_output = layer.forward(x, adapter_indices)

        # Single-sequence forwards
        singles = []
        for i in range(batch_size):
            single = layer.forward(x[i:i+1], adapter_indices[i:i+1])
            singles.append(single)
        stacked = torch.cat(singles, dim=0)

        torch.testing.assert_close(
            batched_output, stacked,
            msg="Batched forward should match stacked single-sequence forwards"
        )

    def test_merged_batched_vs_single(self):
        """MergedSwitchedLoRALinear: batched == stacked singles."""
        torch.manual_seed(SEED)
        layer = _make_merged_layer()

        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_B_slices[s].data = torch.randn_like(layer.lora_B_slices[s]) * 0.1

        batch_size = 4
        seq_len = 5
        x = _seeded_input(batch_size, seq_len, IN_FEATURES, seed=SEED + 33)

        adapter_indices = torch.tensor([
            [0, 1, 0, 2, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [3, 2, 1, 0, 3],
        ], dtype=torch.long)

        batched_output = layer.forward(x, adapter_indices)

        singles = []
        for i in range(batch_size):
            single = layer.forward(x[i:i+1], adapter_indices[i:i+1])
            singles.append(single)
        stacked = torch.cat(singles, dim=0)

        torch.testing.assert_close(
            batched_output, stacked,
            msg="Batched forward should match stacked single-sequence forwards"
        )
