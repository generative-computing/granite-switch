# SPDX-License-Identifier: Apache-2.0
"""vLLM LoRA layer tests (inner file — run by test_lora.py in subprocess).

Mirrors the property categories from tests/hf/test_lora.py but uses vLLM-native
interfaces:
- Constructor takes base_layer (nn.Module returning (output, bias) tuple)
- Forward takes (x, meta_args) where meta_args is a 6-tuple from
  CompileFriendlyLoRAKernelMeta
- Input is always 2D [num_tokens, features] (no 3D batch dim)
- Output is (output, bias) tuple
- Requires CUDA (Triton Punica kernels)

Section 1: SwitchedLoRALinear (single slice, num_slices=1)
Section 2: SwitchedLoRALinear with packed modules (num_slices > 1)
"""

from unittest.mock import patch

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_import_vllm():
    try:
        from vllm.lora.ops.triton_ops import lora_expand, lora_shrink  # noqa: F401

        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_import_vllm() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

if _VLLM_AVAILABLE:
    from granite_switch.vllm.core.lora import SwitchedLoRALinear
    from granite_switch.vllm.core.lora_kernel_meta import CompileFriendlyLoRAKernelMeta

# ── Constants ────────────────────────────────────────────────────────

IN_FEATURES = 32
OUT_FEATURES = 16
NUM_ADAPTERS = 4
RANK = 8
SEED = 42

# Typical QKV slices for a small model
PACKED_OUTPUT_SLICES = (64, 16, 16)
PACKED_TOTAL_OUT = sum(PACKED_OUTPUT_SLICES)


# ── Helpers ──────────────────────────────────────────────────────────


class _VLLMBaseLayer(torch.nn.Module):
    """nn.Linear wrapper returning (output, None) like vLLM parallel layers."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x), None


class _VLLMLoRATestBase:
    """Provides CUDA setup, CompileFriendlyLoRAKernelMeta, and helpers."""

    # SwitchedLoRALinear.__init__ calls get_tensor_model_parallel_world_size()
    # and get_tensor_model_parallel_rank() for TP-aware weight sharding.
    # These require an initialized distributed process group, which doesn't
    # exist in unit tests.  Mock them to return tp_size=1, tp_rank=0 (the
    # single-GPU defaults) for the duration of every test.
    @pytest.fixture(autouse=True)
    def setup_cuda(self):
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16  # Punica Triton kernels require bfloat16/float16
        self.lora_meta = CompileFriendlyLoRAKernelMeta(
            num_adapters=NUM_ADAPTERS,
            device=self.device,
            dtype=self.dtype,
        )
        with (
            patch(
                "granite_switch.vllm.core.lora.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "granite_switch.vllm.core.lora.get_tensor_model_parallel_rank",
                return_value=0,
            ),
        ):
            yield

    def _make_layer(self, in_features, out_features, num_adapters, rank):
        """Create a single-slice SwitchedLoRALinear on CUDA."""
        base = _VLLMBaseLayer(in_features, out_features).to(self.device, self.dtype)
        return SwitchedLoRALinear(
            base_layer=base,
            num_adapters=num_adapters,
            max_lora_rank=rank,
        )

    def _make_packed_layer(self, in_features, output_slices, num_adapters, rank):
        """Create a packed (multi-slice) SwitchedLoRALinear on CUDA."""
        total_out = sum(output_slices)
        base = _VLLMBaseLayer(in_features, total_out).to(self.device, self.dtype)
        return SwitchedLoRALinear(
            base_layer=base,
            num_adapters=num_adapters,
            max_lora_rank=rank,
            num_slices=len(output_slices),
            output_slices=output_slices,
        )

    def _run_with_meta(self, layer, x_2d, meta):
        """Forward pass with pre-computed metadata tuple."""
        from granite_switch.vllm.core.lora_kernel_meta import LoRAContext

        ctx = LoRAContext()
        ctx.token_lora_mapping = meta[0]
        ctx.token_indices_sorted = meta[1]
        ctx.num_tokens_per_lora = meta[2]
        ctx.lora_token_start_loc = meta[3]
        ctx.active_lora_ids = meta[4]
        ctx.no_lora_flag_cpu = meta[5]
        ctx.num_active_loras = meta[6]
        layer._lora_ctx = ctx
        result = layer.forward(x_2d)
        layer._lora_ctx = None
        return result

    def _run(self, layer, x_2d, adapter_indices_1d):
        """Forward pass returning (output, bias) tuple."""
        from granite_switch.vllm.core.lora_kernel_meta import LoRAContext

        punica_indices = adapter_indices_1d.to(self.device) - 1
        ctx = LoRAContext()
        self.lora_meta.prepare_and_store(punica_indices, ctx)
        layer._lora_ctx = ctx
        result = layer.forward(x_2d)
        layer._lora_ctx = None
        return result

    def _base_output(self, layer, x_2d):
        """Get base layer output (unpacks tuple)."""
        out, _ = layer.base_layer(x_2d)
        return out


# ════════════════════════════════════════════════════════════════════
# Section 1: SwitchedLoRALinear (single slice, num_slices=1)
# ════════════════════════════════════════════════════════════════════


class TestBasePassthrough(_VLLMLoRATestBase):
    """All-base adapter indices -> output equals base_layer output."""

    def test_all_base_equals_base_layer(self):
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x = torch.randn(10, IN_FEATURES, device=self.device, dtype=self.dtype)
        adapter_indices = torch.zeros(10, dtype=torch.long, device=self.device)

        output, bias = self._run(layer, x, adapter_indices)
        expected = self._base_output(layer, x)

        torch.testing.assert_close(output, expected)

    def test_single_token_base(self):
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x = torch.randn(1, IN_FEATURES, device=self.device, dtype=self.dtype)
        adapter_indices = torch.zeros(1, dtype=torch.long, device=self.device)

        output, _ = self._run(layer, x, adapter_indices)
        expected = self._base_output(layer, x)

        torch.testing.assert_close(output, expected)


class TestAdapterActivation(_VLLMLoRATestBase):
    """Adapter modifies output, different adapters differ, base tokens unchanged."""

    def test_adapter_modifies_output(self):
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        # vLLM zero-inits both lora_A and lora_B; must set both for non-zero delta
        with torch.no_grad():
            layer.lora_A.data = torch.randn_like(layer.lora_A) * 0.1
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = torch.randn(4, IN_FEATURES, device=self.device, dtype=self.dtype)

        base_indices = torch.zeros(4, dtype=torch.long, device=self.device)
        adapter_indices = torch.ones(4, dtype=torch.long, device=self.device)

        base_out, _ = self._run(layer, x, base_indices)
        adapter_out, _ = self._run(layer, x, adapter_indices)

        assert not torch.allclose(base_out, adapter_out), (
            "Adapter output should differ from base output"
        )

    def test_different_adapters_produce_different_outputs(self):
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        with torch.no_grad():
            for a in range(NUM_ADAPTERS):
                layer.lora_A.data[a] = torch.randn_like(layer.lora_A[a]) * 0.1 * (a + 1)
                layer.lora_B.data[a] = torch.randn_like(layer.lora_B[a]) * 0.1 * (a + 1)

        x = torch.randn(4, IN_FEATURES, device=self.device, dtype=self.dtype)

        indices_1 = torch.ones(4, dtype=torch.long, device=self.device)
        indices_2 = torch.full((4,), 2, dtype=torch.long, device=self.device)

        out_1, _ = self._run(layer, x, indices_1)
        out_2, _ = self._run(layer, x, indices_2)

        assert not torch.allclose(out_1, out_2), (
            "Different adapters should produce different outputs"
        )

    def test_base_tokens_unchanged_in_mixed_batch(self):
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)

        with torch.no_grad():
            layer.lora_A.data = torch.randn_like(layer.lora_A) * 0.1
            layer.lora_B.data = torch.randn_like(layer.lora_B) * 0.1

        x = torch.randn(6, IN_FEATURES, device=self.device, dtype=self.dtype)

        mixed_indices = torch.tensor(
            [0, 1, 0, 2, 0, 0], dtype=torch.long, device=self.device
        )
        all_base_indices = torch.zeros(6, dtype=torch.long, device=self.device)

        mixed_out, _ = self._run(layer, x, mixed_indices)
        base_out, _ = self._run(layer, x, all_base_indices)

        for pos in [0, 2, 4, 5]:
            torch.testing.assert_close(
                mixed_out[pos],
                base_out[pos],
                msg=f"Base token at position {pos} should be unchanged",
            )


class TestMathCorrectness(_VLLMLoRATestBase):
    """Known weights -> verify output = base(x) + x @ A^T @ B^T."""

    @pytest.mark.parametrize("num_adapters", [1, 2, 4])
    def test_lora_output_matches_manual_computation(self, num_adapters):
        torch.manual_seed(SEED)
        # Need a fresh lora_meta for non-default num_adapters
        lora_meta = CompileFriendlyLoRAKernelMeta(
            num_adapters=num_adapters,
            device=self.device,
            dtype=self.dtype,
        )
        base = _VLLMBaseLayer(IN_FEATURES, OUT_FEATURES).to(self.device, self.dtype)
        layer = SwitchedLoRALinear(
            base_layer=base,
            num_adapters=num_adapters,
            max_lora_rank=RANK,
        )

        with torch.no_grad():
            for a in range(num_adapters):
                layer.lora_A.data[a, 0] = (
                    torch.eye(RANK, IN_FEATURES, device=self.device, dtype=self.dtype)
                    * 0.1
                    * (a + 1)
                )
                layer.lora_B.data[a, 0] = (
                    torch.eye(OUT_FEATURES, RANK, device=self.device, dtype=self.dtype)
                    * 0.2
                    * (a + 1)
                )

        x = torch.randn(3, IN_FEATURES, device=self.device, dtype=self.dtype)
        torch.manual_seed(SEED + 11)

        for adapter_id in range(1, num_adapters + 1):
            adapter_indices = torch.full(
                (3,), adapter_id, dtype=torch.long, device=self.device
            )
            punica_indices = adapter_indices - 1
            meta = lora_meta.prepare_tensors(punica_indices)
            output, _ = self._run_with_meta(layer, x, meta)

            base_out = self._base_output(layer, x)
            tensor_idx = adapter_id - 1
            lora_a = layer.lora_A[tensor_idx, 0]
            lora_b = layer.lora_B[tensor_idx, 0]
            lora_delta = x @ lora_a.t() @ lora_b.t()
            expected = base_out + lora_delta

            torch.testing.assert_close(
                output,
                expected,
                msg=f"Adapter {adapter_id}: output should match base + x @ A^T @ B^T",
            )


class TestShapeCorrectness(_VLLMLoRATestBase):
    """Output shapes for various token counts."""

    @pytest.mark.parametrize("num_tokens", [1, 5, 10, 20, 100])
    def test_output_shape(self, num_tokens):
        torch.manual_seed(SEED)
        layer = self._make_layer(IN_FEATURES, OUT_FEATURES, NUM_ADAPTERS, RANK)
        x = torch.randn(num_tokens, IN_FEATURES, device=self.device, dtype=self.dtype)
        adapter_indices = torch.zeros(num_tokens, dtype=torch.long, device=self.device)

        output, bias = self._run(layer, x, adapter_indices)
        assert output.shape == (num_tokens, OUT_FEATURES)


# ════════════════════════════════════════════════════════════════════
# Section 2: SwitchedLoRALinear with packed modules (num_slices > 1)
# ════════════════════════════════════════════════════════════════════


class TestPackedBasePassthrough(_VLLMLoRATestBase):
    """All-base -> base output."""

    def test_all_base_equals_base_layer(self):
        torch.manual_seed(SEED)
        layer = self._make_packed_layer(
            IN_FEATURES, PACKED_OUTPUT_SLICES, NUM_ADAPTERS, RANK
        )
        x = torch.randn(10, IN_FEATURES, device=self.device, dtype=self.dtype)
        adapter_indices = torch.zeros(10, dtype=torch.long, device=self.device)

        output, _ = self._run(layer, x, adapter_indices)
        expected = self._base_output(layer, x)

        torch.testing.assert_close(output, expected)


class TestPackedAdapterActivation(_VLLMLoRATestBase):
    """Adapter modifies output in packed layers."""

    def test_adapter_modifies_output(self):
        torch.manual_seed(SEED)
        layer = self._make_packed_layer(
            IN_FEATURES, PACKED_OUTPUT_SLICES, NUM_ADAPTERS, RANK
        )

        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_A_slices[s].data = (
                    torch.randn_like(layer.lora_A_slices[s]) * 0.1
                )
                layer.lora_B_slices[s].data = (
                    torch.randn_like(layer.lora_B_slices[s]) * 0.1
                )

        x = torch.randn(4, IN_FEATURES, device=self.device, dtype=self.dtype)

        base_indices = torch.zeros(4, dtype=torch.long, device=self.device)
        adapter_indices = torch.ones(4, dtype=torch.long, device=self.device)

        base_out, _ = self._run(layer, x, base_indices)
        adapter_out, _ = self._run(layer, x, adapter_indices)

        assert not torch.allclose(base_out, adapter_out), (
            "Adapter output should differ from base output"
        )


class TestPackedSliceIndependence(_VLLMLoRATestBase):
    """LoRA on slice 0 doesn't affect slices 1+."""

    def test_lora_only_affects_target_slice(self):
        torch.manual_seed(SEED)
        layer = self._make_packed_layer(
            IN_FEATURES, PACKED_OUTPUT_SLICES, NUM_ADAPTERS, RANK
        )

        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_A_slices[s].data.zero_()
                layer.lora_B_slices[s].data.zero_()

            # Set non-zero LoRA only on slice 0
            layer.lora_A_slices[0].data[:] = torch.randn_like(layer.lora_A_slices[0])
            layer.lora_B_slices[0].data[:] = torch.randn_like(layer.lora_B_slices[0])

        x = torch.randn(4, IN_FEATURES, device=self.device, dtype=self.dtype)
        adapter_indices = torch.ones(4, dtype=torch.long, device=self.device)

        output, _ = self._run(layer, x, adapter_indices)
        base_output = self._base_output(layer, x)

        # Slice 0 should differ (has LoRA)
        slice_0_end = PACKED_OUTPUT_SLICES[0]
        assert not torch.allclose(
            output[:, :slice_0_end], base_output[:, :slice_0_end]
        ), "Slice 0 should be modified by LoRA"

        # Slices 1+ should be identical to base (no LoRA)
        torch.testing.assert_close(
            output[:, slice_0_end:],
            base_output[:, slice_0_end:],
            msg="Slices 1+ should be unchanged (no LoRA weights)",
        )


class TestPackedMathCorrectness(_VLLMLoRATestBase):
    """Per-slice LoRA math verification."""

    def test_per_slice_lora_math(self):
        torch.manual_seed(SEED)
        layer = self._make_packed_layer(
            IN_FEATURES, PACKED_OUTPUT_SLICES, NUM_ADAPTERS, RANK
        )

        with torch.no_grad():
            for s in range(layer.num_slices):
                for a in range(NUM_ADAPTERS):
                    out_size = PACKED_OUTPUT_SLICES[s]
                    layer.lora_A_slices[s].data[a, 0] = (
                        torch.eye(
                            RANK, IN_FEATURES, device=self.device, dtype=self.dtype
                        )
                        * 0.1
                        * (s + 1)
                        * (a + 1)
                    )
                    layer.lora_B_slices[s].data[a, 0] = (
                        torch.eye(out_size, RANK, device=self.device, dtype=self.dtype)
                        * 0.2
                        * (s + 1)
                        * (a + 1)
                    )

        x = torch.randn(3, IN_FEATURES, device=self.device, dtype=self.dtype)

        for adapter_id in range(1, NUM_ADAPTERS + 1):
            adapter_indices = torch.full(
                (3,), adapter_id, dtype=torch.long, device=self.device
            )
            output, _ = self._run(layer, x, adapter_indices)
            base_out = self._base_output(layer, x)

            offset = 0
            for s, out_size in enumerate(PACKED_OUTPUT_SLICES):
                tensor_idx = adapter_id - 1
                lora_a = layer.lora_A_slices[s][tensor_idx, 0]
                lora_b = layer.lora_B_slices[s][tensor_idx, 0]
                lora_delta = x @ lora_a.t() @ lora_b.t()
                expected_slice = base_out[:, offset : offset + out_size] + lora_delta

                torch.testing.assert_close(
                    output[:, offset : offset + out_size],
                    expected_slice,
                    msg=f"Adapter {adapter_id}, slice {s}: math mismatch",
                )
                offset += out_size


class TestPackedBatchIndependence(_VLLMLoRATestBase):
    """Mixed adapter tokens, no cross-talk."""

    def test_mixed_adapters_no_crosstalk(self):
        torch.manual_seed(SEED)
        layer = self._make_packed_layer(
            IN_FEATURES, PACKED_OUTPUT_SLICES, NUM_ADAPTERS, RANK
        )

        with torch.no_grad():
            for s in range(layer.num_slices):
                layer.lora_B_slices[s].data = (
                    torch.randn_like(layer.lora_B_slices[s]) * 0.1
                )

        # 8 tokens with mixed adapters
        x = torch.randn(8, IN_FEATURES, device=self.device, dtype=self.dtype)
        adapter_indices = torch.tensor(
            [0, 1, 0, 2, 3, 0, 1, 4], dtype=torch.long, device=self.device
        )

        output, _ = self._run(layer, x, adapter_indices)

        # Verify each token independently
        for i in range(8):
            x_single = x[i : i + 1]
            idx_single = adapter_indices[i : i + 1]
            ref, _ = self._run(layer, x_single, idx_single)

            torch.testing.assert_close(
                output[i],
                ref[0],
                msg=f"Token {i} (adapter={adapter_indices[i].item()}): cross-talk detected",
            )
