# SPDX-License-Identifier: Apache-2.0
"""Level-2 unit tests for SwitchedLoRALinear (vLLM fused kernel layer).

Exercises construction, finalize_weights(), and forward-pass correctness
at real Granite model geometries (3B, 8B, 30B) with single-slice and
multi-slice variants.  Mixed adapter ranks are covered.

Requires CUDA GPU (Triton kernel) and vLLM installed.
"""

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_imports():
    try:
        from vllm.model_executor.layers.linear import (  # noqa: F401
            ColumnParallelLinear,
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
        )

        return True
    except ImportError:
        return False


_VLLM_AVAILABLE = _try_imports() if _CUDA_AVAILABLE else False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE or not _VLLM_AVAILABLE,
    reason="requires CUDA GPU and vLLM installed",
)

if _VLLM_AVAILABLE:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    from granite_switch.kernels import BLOCK_N
    from granite_switch.vllm.core.lora import SwitchedLoRALinear
    from granite_switch.vllm.core.lora_kernel_meta import (
        _compute_bitmasks_reference,
    )
    from tests.shared.vllm_distributed import ensure_distributed


# ── Granite model geometries ──────────────────────────────────────────

# (hidden_size, num_heads, num_kv_heads, intermediate_size)
GRANITE_3B = (2560, 40, 8, 8192)
GRANITE_8B = (4096, 32, 8, 12800)
GRANITE_30B = (4096, 32, 8, 32768)


def _head_dim(hidden, heads):
    return hidden // heads


# Single-slice geometries: (K, N, model_label, module_label)
SINGLE_SLICE_GEOMETRIES = []
for label, (hidden, heads, kv_heads, inter) in [
    ("3B", GRANITE_3B),
    ("8B", GRANITE_8B),
    ("30B", GRANITE_30B),
]:
    hd = _head_dim(hidden, heads)
    SINGLE_SLICE_GEOMETRIES.extend(
        [
            (hidden, hidden, f"{label}_o_proj"),
            (inter, hidden, f"{label}_output_linear"),
        ]
    )

# Multi-slice geometries: (K, output_slices, model_label)
MULTI_SLICE_GEOMETRIES = []
for label, (hidden, heads, kv_heads, inter) in [
    ("3B", GRANITE_3B),
    ("8B", GRANITE_8B),
    ("30B", GRANITE_30B),
]:
    hd = _head_dim(hidden, heads)
    q_size = heads * hd
    kv_size = kv_heads * hd
    MULTI_SLICE_GEOMETRIES.extend(
        [
            (hidden, (q_size, kv_size, kv_size), f"{label}_qkv_proj"),
            (hidden, (inter, inter), f"{label}_input_linear"),
        ]
    )

# Adapter rank configurations: (adapter_ranks_list, config_label)
RANK_CONFIGS = [
    ([16, 16], "uniform_16"),
    ([16, 32], "mixed_16_32"),
    ([16, 32, 64], "mixed_16_32_64"),
    # Sub-tier: rank 8 is promoted to 16 and zero-padded. test_mixed_ranks_three_adapters
    # runs the real Triton forward at this config against a pure-torch reference computed at
    # the TRUE rank 8, verifying the zero-pad is numerically exact rather than only documented.
    ([8, 8], "uniform_8"),
]


# ── Helpers ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True, scope="module")
def _init_distributed():
    if not _VLLM_AVAILABLE:
        return
    ensure_distributed()


def _make_single_slice_layer(K, N, num_adapters, max_rank, device):
    """Create a SwitchedLoRALinear wrapping a RowParallelLinear (single-slice)."""
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        base = RowParallelLinear(
            input_size=K,
            output_size=N,
            bias=False,
            params_dtype=torch.bfloat16,
        ).to(device)
        layer = SwitchedLoRALinear(
            base_layer=base,
            num_adapters=num_adapters,
            max_lora_rank=max_rank,
            num_slices=1,
        )
    with torch.no_grad():
        base.weight.data.normal_(0, 0.02)
    return layer


def _make_multi_slice_layer(K, output_slices, num_adapters, max_rank, device):
    """Create a SwitchedLoRALinear wrapping a MergedColumnParallelLinear."""
    N_total = sum(output_slices)
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        base = ColumnParallelLinear(
            input_size=K,
            output_size=N_total,
            bias=False,
            params_dtype=torch.bfloat16,
        ).to(device)
        layer = SwitchedLoRALinear(
            base_layer=base,
            num_adapters=num_adapters,
            max_lora_rank=max_rank,
            num_slices=len(output_slices),
            output_slices=output_slices,
        )
    with torch.no_grad():
        base.weight.data.normal_(0, 0.02)
    return layer


def _fill_lora_weights(layer, adapter_ranks, scale=0.1):
    """Fill checkpoint-format lora_A/B with random data at each adapter's rank."""
    with torch.no_grad():
        if layer.num_slices == 1:
            for i, r in enumerate(adapter_ranks):
                layer.lora_A.data[i, 0, :r, :] = (
                    torch.randn(
                        r, layer.in_features, dtype=layer._dtype, device=layer._device
                    )
                    * scale
                )
                layer.lora_B.data[i, 0, :, :r] = (
                    torch.randn(
                        layer.out_features, r, dtype=layer._dtype, device=layer._device
                    )
                    * scale
                )
        else:
            for i, r in enumerate(adapter_ranks):
                for s in range(layer.num_slices):
                    layer.lora_A_slices[s].data[i, 0, :r, :] = (
                        torch.randn(
                            r,
                            layer.in_features,
                            dtype=layer._dtype,
                            device=layer._device,
                        )
                        * scale
                    )
                    N_s = layer.output_slices[s]
                    layer.lora_B_slices[s].data[i, 0, :, :r] = (
                        torch.randn(N_s, r, dtype=layer._dtype, device=layer._device)
                        * scale
                    )


def _reference_forward(layer, x, adapter_indices, adapter_ranks):
    """Compute reference output using explicit per-token matmul (no kernel).

    For each token m:
      out[m] = x[m] @ W_base.T
      if adapter_indices[m] > 0:
        a = adapter_indices[m] - 1  (0-indexed adapter)
        r = adapter_ranks[a]
        for each slice s:
          shrink = x[m] @ lora_A[a,s,:r,:].T          -> [r]
          out[m, slice_start:slice_end] += shrink @ lora_B[a,s,:,:r].T  -> [N_s]
    """
    M = x.shape[0]
    W_base = layer.base_layer.weight.data  # [N_total, K]

    base_out = x @ W_base.T  # [M, N_total]
    out = base_out.clone()

    S = layer.num_slices
    slice_starts = [0]
    for s in range(S):
        slice_starts.append(slice_starts[-1] + layer.output_slices[s])

    for m in range(M):
        ai = adapter_indices[m].item()
        if ai == 0:
            continue
        a = ai - 1
        r = adapter_ranks[a]
        for s in range(S):
            if S == 1:
                lora_A_s = layer.lora_A.data[a, 0, :r, :]  # [r, K]
                lora_B_s = layer.lora_B.data[a, 0, :, :r]  # [N_s, r]
            else:
                lora_A_s = layer.lora_A_slices[s].data[a, 0, :r, :]
                lora_B_s = layer.lora_B_slices[s].data[a, 0, :, :r]
            shrink = x[m] @ lora_A_s.T  # [r]
            expand = shrink @ lora_B_s.T  # [N_s]
            ns = slice_starts[s]
            ne = slice_starts[s + 1]
            out[m, ns:ne] += expand

    return out


def _wire_single_module_ctx(layer, adapter_indices):
    """Build a LoRAContext for a single finalized module and attach it.

    Mirrors FusedLoRAKernelMeta.prepare_and_store for the one-module case:
    computes the per-tile bitmask AND the kernel-local remapped_indices that
    forward() reads (ctx.remapped_indices[module_idx, :M]). Sets module_idx=0.
    """
    from granite_switch.vllm.core.lora_kernel_meta import LoRAContext

    remap_table_2d = layer.remap_table.unsqueeze(0)  # [1, NA+1]
    bitmask = _compute_bitmasks_reference(adapter_indices, remap_table_2d.T)

    ctx = LoRAContext()
    ctx.adapter_indices = adapter_indices
    ctx.per_module_bitmasks = bitmask
    # [num_modules, M] row-major; module 0's row is a stride-1 view, as in prod.
    ctx.remapped_indices = layer.remap_table[adapter_indices].unsqueeze(0).contiguous()
    layer._lora_ctx = ctx
    layer._module_idx = 0
    return ctx


# ════════════════════════════════════════════════════════════════════════
# 0. Custom-op surface lock
#
# The fused design ships exactly ONE expand op. This guards against the
# research-era scaffolding (superseded variants + launch-cost probes) creeping
# back into the registered surface.
# ════════════════════════════════════════════════════════════════════════


class TestKernelOpSurface:
    def test_only_switch_lora_expand_registered(self):
        # Importing SwitchedLoRALinear (top of module) registers the op.
        assert hasattr(torch.ops.granite_switch, "switch_lora_expand")

    @pytest.mark.parametrize(
        "dead_op",
        [
            "lora_expand",
            "lora_expand_v2",
            "lora_expand_v3",
            "lora_expand_v3_funcadd",
            "lora_expand_v3_rmw",
            "lora_expand_v3_inplace",
            "lora_expand_v4_inplace",
            "lora_expand_v4_inplace_noop",
            "lora_expand_v4_inplace_fullsig_noop",
            "lora_expand_v4_inplace_skiplaunch",
            "noop_expand",
            "rmw_noop_expand",
        ],
    )
    def test_scaffolding_ops_absent(self, dead_op):
        assert not hasattr(torch.ops.granite_switch, dead_op), (
            f"removed scaffolding op torch.ops.granite_switch.{dead_op} is registered again"
        )


# ════════════════════════════════════════════════════════════════════════
# 1. Construction tests
# ════════════════════════════════════════════════════════════════════════


class TestConstruction:
    """Verify checkpoint-format parameter shapes at construction time."""

    @pytest.mark.parametrize("K,N,label", SINGLE_SLICE_GEOMETRIES)
    def test_single_slice_shapes(self, K, N, label):
        device = torch.device("cuda")
        num_adapters = 2
        max_rank = 32
        layer = _make_single_slice_layer(K, N, num_adapters, max_rank, device)

        assert layer.lora_A.shape == (num_adapters, 1, max_rank, K)
        assert layer.lora_B.shape == (num_adapters, 1, N, max_rank)
        assert layer.num_slices == 1
        assert layer.output_slices == (N,)

    @pytest.mark.parametrize("K,output_slices,label", MULTI_SLICE_GEOMETRIES)
    def test_multi_slice_shapes(self, K, output_slices, label):
        device = torch.device("cuda")
        num_adapters = 2
        max_rank = 32
        layer = _make_multi_slice_layer(
            K, output_slices, num_adapters, max_rank, device
        )

        S = len(output_slices)
        assert len(layer.lora_A_slices) == S
        assert len(layer.lora_B_slices) == S
        for s in range(S):
            assert layer.lora_A_slices[s].shape == (num_adapters, 1, max_rank, K)
            assert layer.lora_B_slices[s].shape == (
                num_adapters,
                1,
                output_slices[s],
                max_rank,
            )


# ════════════════════════════════════════════════════════════════════════
# 2. finalize_weights tests
# ════════════════════════════════════════════════════════════════════════


class TestFinalizeWeights:
    """Verify finalize_weights builds correct fused structures."""

    @pytest.mark.parametrize("K,N,label", SINGLE_SLICE_GEOMETRIES)
    @pytest.mark.parametrize("adapter_ranks,rank_label", RANK_CONFIGS[:2])
    def test_single_slice_w_ext_shape(self, K, N, label, adapter_ranks, rank_label):
        device = torch.device("cuda")
        NA = len(adapter_ranks)
        max_rank = max(adapter_ranks)
        layer = _make_single_slice_layer(K, N, NA, max_rank, device)
        _fill_lora_weights(layer, adapter_ranks)
        layer.finalize_weights(adapter_ranks)

        # w_ext rows = N + sum(rank_i * S) for all applicable adapters
        S = 1
        expected_rows = N + sum(r * S for r in adapter_ranks)
        assert layer.w_ext.shape == (expected_rows, K)
        assert layer._finalized

    @pytest.mark.parametrize("K,output_slices,label", MULTI_SLICE_GEOMETRIES)
    @pytest.mark.parametrize("adapter_ranks,rank_label", RANK_CONFIGS[:2])
    def test_multi_slice_w_ext_shape(
        self, K, output_slices, label, adapter_ranks, rank_label
    ):
        device = torch.device("cuda")
        NA = len(adapter_ranks)
        max_rank = max(adapter_ranks)
        layer = _make_multi_slice_layer(K, output_slices, NA, max_rank, device)
        _fill_lora_weights(layer, adapter_ranks)
        layer.finalize_weights(adapter_ranks)

        N_total = sum(output_slices)
        S = len(output_slices)
        expected_rows = N_total + sum(r * S for r in adapter_ranks)
        assert layer.w_ext.shape == (expected_rows, K)

    @pytest.mark.parametrize("K,N,label", SINGLE_SLICE_GEOMETRIES)
    def test_remap_table_structure(self, K, N, label):
        device = torch.device("cuda")
        adapter_ranks = [16, 32]
        NA = len(adapter_ranks)
        max_rank = max(adapter_ranks)
        layer = _make_single_slice_layer(K, N, NA, max_rank, device)
        _fill_lora_weights(layer, adapter_ranks)
        layer.finalize_weights(adapter_ranks)

        # remap_table[0] = 0 (base), others > 0
        assert layer.remap_table[0].item() == 0
        for i in range(NA):
            assert layer.remap_table[i + 1].item() > 0

    def test_non_applicable_adapter_remaps_to_zero(self):
        """Adapter with all-zero lora_A should map to 0 in remap_table."""
        device = torch.device("cuda")
        K, N = 2560, 2560
        adapter_ranks = [16, 16]
        NA = 2
        max_rank = 16
        layer = _make_single_slice_layer(K, N, NA, max_rank, device)

        # Only fill adapter 0, leave adapter 1 as zeros
        with torch.no_grad():
            layer.lora_A.data[0, 0, :16, :] = (
                torch.randn(16, K, dtype=torch.bfloat16, device=device) * 0.1
            )
            layer.lora_B.data[0, 0, :, :16] = (
                torch.randn(N, 16, dtype=torch.bfloat16, device=device) * 0.1
            )

        layer.finalize_weights(adapter_ranks)

        # Adapter 0 (global index 1) should be applicable
        assert layer.remap_table[1].item() > 0
        # Adapter 1 (global index 2) should be non-applicable → 0
        assert layer.remap_table[2].item() == 0

    def test_block_n_alignment_assertion(self):
        """finalize_weights should fail if a slice is not divisible by BLOCK_N."""
        device = torch.device("cuda")
        # Pick a slice size not divisible by BLOCK_N (32)
        bad_slices = (BLOCK_N * 3, BLOCK_N * 2 + 1)  # second slice misaligned
        K = 256
        NA = 1
        max_rank = 16
        layer = _make_multi_slice_layer(K, bad_slices, NA, max_rank, device)
        _fill_lora_weights(layer, [16])

        with pytest.raises(AssertionError, match="block_n=.*must divide"):
            layer.finalize_weights([16])


# ════════════════════════════════════════════════════════════════════════
# 3. Forward correctness tests
# ════════════════════════════════════════════════════════════════════════

# Use smaller geometries for forward tests to keep runtime reasonable.
# These still exercise real alignment constraints (all N_s % BLOCK_N == 0).
FORWARD_SINGLE_GEOMETRIES = [
    (2560, 2560, "3B_o_proj"),
    (4096, 4096, "8B_o_proj"),
]

FORWARD_MULTI_GEOMETRIES = [
    (2560, (2560, 512, 512), "3B_qkv_proj"),
    (4096, (4096, 1024, 1024), "8B_qkv_proj"),
    (4096, (12800, 12800), "8B_input_linear"),
]


class TestForwardCorrectness:
    """Compare fused kernel output against reference per-token matmul."""

    @pytest.mark.parametrize("K,N,label", FORWARD_SINGLE_GEOMETRIES)
    @pytest.mark.parametrize("adapter_ranks,rank_label", RANK_CONFIGS[:2])
    def test_single_slice_matches_reference(
        self, K, N, label, adapter_ranks, rank_label
    ):
        device = torch.device("cuda")
        NA = len(adapter_ranks)
        max_rank = max(adapter_ranks)
        M = 64  # tokens

        layer = _make_single_slice_layer(K, N, NA, max_rank, device)
        _fill_lora_weights(layer, adapter_ranks)

        # Compute reference BEFORE finalize (uses checkpoint-format params)
        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.01

        # Adapter indices: mix of base (0) and adapters (1..NA)
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        for i in range(NA):
            start = (i + 1) * (M // (NA + 1))
            end = (i + 2) * (M // (NA + 1))
            adapter_indices[start:end] = i + 1

        ref = _reference_forward(layer, x, adapter_indices, adapter_ranks)

        # Finalize and run fused forward
        layer.finalize_weights(adapter_ranks)

        _wire_single_module_ctx(layer, adapter_indices)

        fused_out, _ = layer.forward(x)

        torch.testing.assert_close(
            fused_out,
            ref,
            atol=0.25,
            rtol=0.05,
            msg=f"Forward mismatch for {label} ranks={adapter_ranks}",
        )

    @pytest.mark.parametrize("K,output_slices,label", FORWARD_MULTI_GEOMETRIES)
    @pytest.mark.parametrize("adapter_ranks,rank_label", RANK_CONFIGS[:2])
    def test_multi_slice_matches_reference(
        self, K, output_slices, label, adapter_ranks, rank_label
    ):
        device = torch.device("cuda")
        NA = len(adapter_ranks)
        max_rank = max(adapter_ranks)
        M = 64

        layer = _make_multi_slice_layer(K, output_slices, NA, max_rank, device)
        _fill_lora_weights(layer, adapter_ranks)

        torch.manual_seed(42)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.01

        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        for i in range(NA):
            start = (i + 1) * (M // (NA + 1))
            end = (i + 2) * (M // (NA + 1))
            adapter_indices[start:end] = i + 1

        ref = _reference_forward(layer, x, adapter_indices, adapter_ranks)

        layer.finalize_weights(adapter_ranks)

        _wire_single_module_ctx(layer, adapter_indices)

        fused_out, _ = layer.forward(x)

        torch.testing.assert_close(
            fused_out,
            ref,
            atol=0.25,
            rtol=0.05,
            msg=f"Forward mismatch for {label} ranks={adapter_ranks}",
        )

    def test_all_base_tokens_no_lora_contribution(self):
        """When all adapter_indices are 0, output should equal base linear."""
        device = torch.device("cuda")
        K, N = 2560, 2560
        adapter_ranks = [16, 16]
        NA = 2
        max_rank = 16
        M = 32

        layer = _make_single_slice_layer(K, N, NA, max_rank, device)
        _fill_lora_weights(layer, adapter_ranks)
        layer.finalize_weights(adapter_ranks)

        torch.manual_seed(7)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.01
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)

        _wire_single_module_ctx(layer, adapter_indices)

        fused_out, _ = layer.forward(x)
        base_out = x @ layer.base_layer.weight.data.T

        torch.testing.assert_close(fused_out, base_out, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("adapter_ranks,rank_label", RANK_CONFIGS)
    def test_mixed_ranks_three_adapters(self, adapter_ranks, rank_label):
        """Forward correctness with 3 adapters at potentially different ranks."""
        device = torch.device("cuda")
        K, N = 2560, 2560
        NA = len(adapter_ranks)
        max_rank = max(adapter_ranks)
        M = 96

        layer = _make_single_slice_layer(K, N, NA, max_rank, device)
        _fill_lora_weights(layer, adapter_ranks)

        torch.manual_seed(99)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.01

        # Each adapter gets a block of tokens
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        block = M // (NA + 1)
        for i in range(NA):
            adapter_indices[(i + 1) * block : (i + 2) * block] = i + 1

        ref = _reference_forward(layer, x, adapter_indices, adapter_ranks)

        layer.finalize_weights(adapter_ranks)

        _wire_single_module_ctx(layer, adapter_indices)

        fused_out, _ = layer.forward(x)

        torch.testing.assert_close(
            fused_out,
            ref,
            atol=0.25,
            rtol=0.05,
            msg=f"Forward mismatch for mixed ranks {adapter_ranks}",
        )


# ════════════════════════════════════════════════════════════════════════
# 4. Production-realistic forward tests
#
# Models the actual Granite Switch 4.1 adapter configurations:
# 12 adapters with per-adapter module targeting and mixed ranks.
# Two variants: "prefer ALoRA" (deployed config) and "all LoRA".
# ════════════════════════════════════════════════════════════════════════

# Per-adapter module applicability: True = adapter has trained weights for this module
# Columns: (qkv_proj, o_proj, shared_input_linear, shared_output_linear)
#
# "Prefer ALoRA": uses ALoRA variant when available (matches deployed 3B/8B)
_PREFER_ALORA_MODULES = [
    # 0: citations (LoRA, attn only)
    (True, True, False, False),
    # 1: query_rewrite (ALoRA, all linear)
    (True, True, True, True),
    # 2: query_clarification (ALoRA, all linear)
    (True, True, True, True),
    # 3: hallucination_detection (LoRA, all linear)
    (True, True, True, True),
    # 4: answerability (ALoRA, all linear)
    (True, True, True, True),
    # 5: factuality-detection (ALoRA, qkv only)
    (True, False, False, False),
    # 6: policy-guardrails (ALoRA, attn only)
    (True, True, False, False),
    # 7: factuality-correction (ALoRA, qkv only)
    (True, False, False, False),
    # 8: guardian-core (ALoRA, attn only)
    (True, True, False, False),
    # 9: uncertainty (ALoRA, attn only)
    (True, True, False, False),
    # 10: requirement-check (ALoRA, attn only)
    (True, True, False, False),
    # 11: context-attribution (LoRA, qkv only)
    (True, False, False, False),
]

# "All LoRA": always uses LoRA variant
_ALL_LORA_MODULES = [
    # 0: citations (LoRA, attn only)
    (True, True, False, False),
    # 1: query_rewrite (LoRA, all linear)
    (True, True, True, True),
    # 2: query_clarification (LoRA, all linear)
    (True, True, True, True),
    # 3: hallucination_detection (LoRA, all linear)
    (True, True, True, True),
    # 4: answerability (LoRA, all linear)
    (True, True, True, True),
    # 5: factuality-detection (LoRA, all linear)
    (True, True, True, True),
    # 6: policy-guardrails (LoRA, attn only)
    (True, True, False, False),
    # 7: factuality-correction (LoRA, all linear)
    (True, True, True, True),
    # 8: guardian-core (LoRA, all linear)
    (True, True, True, True),
    # 9: uncertainty (LoRA, attn only)
    (True, True, False, False),
    # 10: requirement-check (LoRA, attn only)
    (True, True, False, False),
    # 11: context-attribution (LoRA, qkv only)
    (True, False, False, False),
]

# Ranks per adapter for each variant × model size
_PREFER_ALORA_RANKS = {
    "3B": [16, 32, 32, 16, 16, 32, 16, 32, 16, 32, 16, 16],
    "8B": [16, 32, 32, 16, 16, 32, 16, 32, 16, 32, 16, 16],
    "30B": [16, 32, 32, 16, 16, 32, 16, 32, 16, 32, 16, 32],
}

_ALL_LORA_RANKS = {
    "3B": [16, 32, 32, 16, 16, 32, 16, 32, 32, 32, 64, 16],
    "8B": [16, 32, 32, 16, 16, 32, 16, 32, 32, 32, 64, 16],
    "30B": [16, 32, 32, 16, 16, 32, 16, 32, 32, 32, 64, 32],
}

# Module geometries: (K, N_or_slices, is_multi_slice)
# Index matches column order in applicability tuples:
# 0=qkv_proj, 1=o_proj, 2=shared_input_linear, 3=shared_output_linear
_MODULE_GEOMETRIES = {
    "3B": [
        (2560, (2560, 512, 512), True, "qkv_proj"),
        (2560, 2560, False, "o_proj"),
        (2560, (8192, 8192), True, "shared_input_linear"),
        (8192, 2560, False, "shared_output_linear"),
    ],
    "8B": [
        (4096, (4096, 1024, 1024), True, "qkv_proj"),
        (4096, 4096, False, "o_proj"),
        (4096, (12800, 12800), True, "shared_input_linear"),
        (12800, 4096, False, "shared_output_linear"),
    ],
    "30B": [
        (4096, (4096, 1024, 1024), True, "qkv_proj"),
        (4096, 4096, False, "o_proj"),
        (4096, (32768, 32768), True, "shared_input_linear"),
        (32768, 4096, False, "shared_output_linear"),
    ],
}

# Production configs: (variant_label, modules_table, ranks_dict)
_PRODUCTION_CONFIGS = [
    ("prefer_alora", _PREFER_ALORA_MODULES, _PREFER_ALORA_RANKS),
    ("all_lora", _ALL_LORA_MODULES, _ALL_LORA_RANKS),
]

# ── Large-rank variants ───────────────────────────────────────────────
# Add a 13th adapter at rank 256 (prefer_alora) or 512 (all_lora),
# targeting all modules. Exercises the higher SUPPORTED_RANKS tiers.

_PREFER_ALORA_LARGE_MODULES = [
    *_PREFER_ALORA_MODULES,
    # 12: large-rank adapter (all modules)
    (True, True, True, True),
]

_ALL_LORA_LARGE_MODULES = [
    *_ALL_LORA_MODULES,
    # 12: large-rank adapter (all modules)
    (True, True, True, True),
]

_PREFER_ALORA_LARGE_RANKS = {
    "3B": _PREFER_ALORA_RANKS["3B"] + [256],
    "8B": _PREFER_ALORA_RANKS["8B"] + [256],
    "30B": _PREFER_ALORA_RANKS["30B"] + [256],
}

_ALL_LORA_LARGE_RANKS = {
    "3B": _ALL_LORA_RANKS["3B"] + [512],
    "8B": _ALL_LORA_RANKS["8B"] + [512],
    "30B": _ALL_LORA_RANKS["30B"] + [512],
}

_LARGE_RANK_CONFIGS = [
    ("prefer_alora_r256", _PREFER_ALORA_LARGE_MODULES, _PREFER_ALORA_LARGE_RANKS),
    ("all_lora_r512", _ALL_LORA_LARGE_MODULES, _ALL_LORA_LARGE_RANKS),
]


def _build_production_layer(
    model_size, module_idx, variant_modules, variant_ranks, device
):
    """Build a SwitchedLoRALinear matching a production adapter config.

    Only fills lora_A/B for adapters that are applicable to this module.
    """
    geom = _MODULE_GEOMETRIES[model_size][module_idx]
    K, N_or_slices, is_multi, module_name = geom
    ranks = variant_ranks[model_size]
    NA = len(ranks)
    max_rank = max(ranks)

    if is_multi:
        layer = _make_multi_slice_layer(K, N_or_slices, NA, max_rank, device)
    else:
        layer = _make_single_slice_layer(K, N_or_slices, NA, max_rank, device)

    # Fill only applicable adapters
    with torch.no_grad():
        for i, r in enumerate(ranks):
            if not variant_modules[i][module_idx]:
                continue  # non-applicable: leave as zeros
            if layer.num_slices == 1:
                layer.lora_A.data[i, 0, :r, :] = (
                    torch.randn(r, layer.in_features, dtype=layer._dtype, device=device)
                    * 0.1
                )
                layer.lora_B.data[i, 0, :, :r] = (
                    torch.randn(
                        layer.out_features, r, dtype=layer._dtype, device=device
                    )
                    * 0.1
                )
            else:
                for s in range(layer.num_slices):
                    layer.lora_A_slices[s].data[i, 0, :r, :] = (
                        torch.randn(
                            r, layer.in_features, dtype=layer._dtype, device=device
                        )
                        * 0.1
                    )
                    N_s = layer.output_slices[s]
                    layer.lora_B_slices[s].data[i, 0, :, :r] = (
                        torch.randn(N_s, r, dtype=layer._dtype, device=device) * 0.1
                    )

    return layer, ranks, module_name


# Token activation scenarios: functions that build adapter_indices for M tokens.
# Each returns (adapter_indices, scenario_label).


def _scenario_single_with_base(M, NA):
    """50% base, 50% adapter 0."""
    indices = torch.zeros(M, dtype=torch.long)
    indices[M // 2 :] = 1
    return indices, "single_with_base"


def _scenario_two_adapters_with_base(M, NA):
    """33% base, 33% adapter 0, 33% adapter 1."""
    indices = torch.zeros(M, dtype=torch.long)
    third = M // 3
    indices[third : 2 * third] = 1
    indices[2 * third :] = min(2, NA)
    return indices, "two_adapters_with_base"


def _scenario_dense_four_with_base(M, NA):
    """20% base, then cycle through 4 different adapters."""
    indices = torch.zeros(M, dtype=torch.long)
    active_start = M // 5
    n_active = min(4, NA)
    block = (M - active_start) // n_active
    for i in range(n_active):
        indices[active_start + i * block : active_start + (i + 1) * block] = i + 1
    return indices, "dense_four_with_base"


def _scenario_all_base(M, NA):
    """100% base — full early-exit."""
    return torch.zeros(M, dtype=torch.long), "all_base"


def _scenario_sparse_in_base(M, NA):
    """~90% base, one adapter token every 10 positions."""
    indices = torch.zeros(M, dtype=torch.long)
    for i in range(5, M, 10):
        indices[i] = (i // 10) % NA + 1
    return indices, "sparse_in_base"


def _scenario_all_one_adapter(M, NA):
    """100% adapter 0 — no early-exit anywhere."""
    return torch.ones(M, dtype=torch.long), "all_one_adapter"


def _scenario_many_adapters_with_base(M, NA):
    """Base prefix, then all 12 adapters each get a short block."""
    indices = torch.zeros(M, dtype=torch.long)
    prefix = M // 4  # 25% base prefix
    remaining = M - prefix
    block = remaining // NA
    for i in range(NA):
        start = prefix + i * block
        end = start + block
        indices[start:end] = i + 1
    return indices, "many_adapters_with_base"


def _scenario_only_large_adapter(M, NA):
    """50% base, 50% the last adapter (the large-rank one)."""
    indices = torch.zeros(M, dtype=torch.long)
    indices[M // 2 :] = NA  # last adapter (1-indexed)
    return indices, "only_large_adapter"


def _scenario_random(M, NA, seed):
    """Random activation pattern from a seed.

    Generates a mix of base and adapter tokens with random density and
    adapter selection drawn from all NA adapters.
    """
    rng = torch.Generator().manual_seed(seed)
    # Random base density between 10% and 80%
    base_frac = (torch.rand(1, generator=rng).item() * 0.7) + 0.1
    indices = torch.zeros(M, dtype=torch.long)
    for i in range(M):
        if torch.rand(1, generator=rng).item() > base_frac:
            indices[i] = torch.randint(1, NA + 1, (1,), generator=rng).item()
    return indices, f"random_seed{seed}"


_SCENARIOS = [
    _scenario_single_with_base,
    _scenario_two_adapters_with_base,
    _scenario_dense_four_with_base,
    _scenario_all_base,
    _scenario_sparse_in_base,
    _scenario_all_one_adapter,
    _scenario_many_adapters_with_base,
]

# Extended scenarios for large-rank configs include the large-adapter-only case
_SCENARIOS_LARGE = [*_SCENARIOS, _scenario_only_large_adapter]

_RANDOM_SEEDS = list(range(20))

# Parametrize over: variant × model_size × module × scenario
_PRODUCTION_TEST_CASES = []
for variant_label, variant_modules, variant_ranks in _PRODUCTION_CONFIGS:
    for model_size in ["3B", "8B", "30B"]:
        for module_idx in range(4):
            module_name = _MODULE_GEOMETRIES[model_size][module_idx][3]
            # Hand-crafted scenarios
            for scenario_fn in _SCENARIOS:
                case_id = f"{variant_label}-{model_size}-{module_name}-{scenario_fn.__name__[10:]}"
                _PRODUCTION_TEST_CASES.append(
                    (
                        variant_label,
                        variant_modules,
                        variant_ranks,
                        model_size,
                        module_idx,
                        scenario_fn,
                        case_id,
                    )
                )
            # Randomized scenarios
            for seed in _RANDOM_SEEDS:
                case_id = (
                    f"{variant_label}-{model_size}-{module_name}-random_seed{seed}"
                )
                _PRODUCTION_TEST_CASES.append(
                    (
                        variant_label,
                        variant_modules,
                        variant_ranks,
                        model_size,
                        module_idx,
                        seed,
                        case_id,
                    )
                )


# Large-rank test cases: variant × model_size × module × (scenarios + random)
_LARGE_RANK_TEST_CASES = []
for variant_label, variant_modules, variant_ranks in _LARGE_RANK_CONFIGS:
    for model_size in ["3B", "8B", "30B"]:
        for module_idx in range(4):
            module_name = _MODULE_GEOMETRIES[model_size][module_idx][3]
            # Hand-crafted scenarios (including only_large_adapter)
            for scenario_fn in _SCENARIOS_LARGE:
                case_id = f"{variant_label}-{model_size}-{module_name}-{scenario_fn.__name__[10:]}"
                _LARGE_RANK_TEST_CASES.append(
                    (
                        variant_label,
                        variant_modules,
                        variant_ranks,
                        model_size,
                        module_idx,
                        scenario_fn,
                        case_id,
                    )
                )
            # Randomized scenarios (13 adapters including the large one)
            for seed in _RANDOM_SEEDS:
                case_id = (
                    f"{variant_label}-{model_size}-{module_name}-random_seed{seed}"
                )
                _LARGE_RANK_TEST_CASES.append(
                    (
                        variant_label,
                        variant_modules,
                        variant_ranks,
                        model_size,
                        module_idx,
                        seed,
                        case_id,
                    )
                )


class TestProductionConfigs:
    """Forward correctness with production-realistic 12-adapter configurations."""

    M = 128  # batch token count

    @pytest.fixture(autouse=True)
    def _cleanup_gpu(self):
        yield
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "variant_label,variant_modules,variant_ranks,model_size,module_idx,scenario_or_seed,case_id",
        _PRODUCTION_TEST_CASES,
        ids=[c[-1] for c in _PRODUCTION_TEST_CASES],
    )
    def test_production_forward(
        self,
        variant_label,
        variant_modules,
        variant_ranks,
        model_size,
        module_idx,
        scenario_or_seed,
        case_id,
    ):
        device = torch.device("cuda")
        torch.manual_seed(42)

        layer, ranks, module_name = _build_production_layer(
            model_size,
            module_idx,
            variant_modules,
            variant_ranks,
            device,
        )
        NA = len(ranks)

        x = (
            torch.randn(self.M, layer.in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )
        if callable(scenario_or_seed):
            adapter_indices, _ = scenario_or_seed(self.M, NA)
        else:
            adapter_indices, _ = _scenario_random(self.M, NA, scenario_or_seed)
        adapter_indices = adapter_indices.to(device)

        # Compute reference before finalize
        ref = _reference_forward(layer, x, adapter_indices, ranks)

        # Finalize and run fused
        layer.finalize_weights(ranks)

        _wire_single_module_ctx(layer, adapter_indices)

        fused_out, _ = layer.forward(x)

        # bf16 matmul reduction noise scales with sqrt(K); at K=32768 with
        # rank-32 adapters, max absolute diff can reach ~0.7 in rare elements
        # (2 out of 524K at 30B output_linear). Base-only tokens are exact.
        torch.testing.assert_close(
            fused_out,
            ref,
            atol=0.75,
            rtol=0.05,
            msg=f"Production forward mismatch: {case_id}",
        )


# ════════════════════════════════════════════════════════════════════════
# 5. Large-rank adapter tests
#
# Same 12 production adapters + a 13th at rank 256 or 512.
# Exercises high SUPPORTED_RANKS tiers and wide rank spread.
# ════════════════════════════════════════════════════════════════════════


class TestLargeRankConfigs:
    """Forward correctness with a large-rank (256/512) adapter added."""

    M = 128

    @pytest.fixture(autouse=True)
    def _cleanup_gpu(self):
        yield
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "variant_label,variant_modules,variant_ranks,model_size,module_idx,scenario_or_seed,case_id",
        _LARGE_RANK_TEST_CASES,
        ids=[c[-1] for c in _LARGE_RANK_TEST_CASES],
    )
    def test_large_rank_forward(
        self,
        variant_label,
        variant_modules,
        variant_ranks,
        model_size,
        module_idx,
        scenario_or_seed,
        case_id,
    ):
        device = torch.device("cuda")
        torch.manual_seed(42)

        layer, ranks, module_name = _build_production_layer(
            model_size,
            module_idx,
            variant_modules,
            variant_ranks,
            device,
        )
        NA = len(ranks)

        x = (
            torch.randn(self.M, layer.in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )
        if callable(scenario_or_seed):
            adapter_indices, _ = scenario_or_seed(self.M, NA)
        else:
            adapter_indices, _ = _scenario_random(self.M, NA, scenario_or_seed)
        adapter_indices = adapter_indices.to(device)

        ref = _reference_forward(layer, x, adapter_indices, ranks)

        layer.finalize_weights(ranks)

        _wire_single_module_ctx(layer, adapter_indices)

        fused_out, _ = layer.forward(x)

        # Higher ranks amplify bf16 reduction noise — rank 512 at K=32768
        # can produce occasional diffs up to ~1.5 in extreme outliers.
        torch.testing.assert_close(
            fused_out,
            ref,
            atol=2.0,
            rtol=0.1,
            msg=f"Large-rank forward mismatch: {case_id}",
        )


# ════════════════════════════════════════════════════════════════════════
# 4. Fused gate/up expand + SwiGLU (shared-MLP first projection)
#
# The merged gate/up projection (fuse_swiglu=True) uses a dedicated kernel that
# applies the LoRA delta to gate (slice 0) and up (slice 1) and emits
# silu(gate)*up as a CONTIGUOUS [M, H] in one pass — no strided base_out escapes
# (the kernel reads x_ext by explicit stride), so there is no .contiguous() and
# no separate SiluAndMul. These tests lock numerical correctness against a dense
# silu(gate_dense)*up_dense reference for base-only and mixed-adapter routing,
# multi-token (the regime where the old strided-base + SiluAndMul bug corrupted
# every token but the first). The bitmask gates only the LoRA work; the
# activation store always runs (base-only tiles still emit silu(gate)*up).
# ════════════════════════════════════════════════════════════════════════


def _make_swiglu_layer(K, H, num_adapters, max_rank, device):
    """SwitchedLoRALinear over a 2H-wide column base in fuse_swiglu (gate/up) mode."""
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        base = ColumnParallelLinear(
            input_size=K,
            output_size=2 * H,
            bias=False,
            params_dtype=torch.bfloat16,
        ).to(device)
        layer = SwitchedLoRALinear(
            base_layer=base,
            num_adapters=num_adapters,
            max_lora_rank=max_rank,
            num_slices=2,
            output_slices=(H, H),
            fuse_swiglu=True,
        )
    with torch.no_grad():
        base.weight.data.normal_(0, 0.02)
    return layer


def _silu_mul(gateup, H):
    g = gateup[:, :H].float()
    u = gateup[:, H:].float()
    return (g * torch.sigmoid(g)) * u


class TestFusedSwiGLU:
    @pytest.mark.parametrize("scenario", ["all_base", "mixed"])
    @pytest.mark.parametrize("adapter_ranks,rank_label", RANK_CONFIGS[:2])
    def test_matches_dense_swiglu_reference(self, scenario, adapter_ranks, rank_label):
        device = torch.device("cuda")
        K, H = 256, 256
        NA = len(adapter_ranks)
        M = 64
        layer = _make_swiglu_layer(K, H, NA, max(adapter_ranks), device)
        _fill_lora_weights(layer, adapter_ranks)
        assert layer.fuse_swiglu

        torch.manual_seed(0)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        if scenario == "mixed":
            step = M // (NA + 1)
            for i in range(NA):
                adapter_indices[(i + 1) * step : (i + 2) * step] = i + 1

        # dense reference: corrected gate|up [M, 2H], then silu(gate)*up (fp32).
        # Computed BEFORE finalize (uses checkpoint-format lora params).
        ref_gateup = _reference_forward(layer, x, adapter_indices, adapter_ranks)
        ref = _silu_mul(ref_gateup, H)

        layer.finalize_weights(adapter_ranks)
        _wire_single_module_ctx(layer, adapter_indices)
        out, bias = layer.forward(x)

        assert out.shape == (M, H)
        assert out.is_contiguous()
        assert bias is None
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=0.05,
            rtol=0.05,
            msg=f"fused gate/up+SwiGLU != dense reference ({scenario}, {rank_label})",
        )

    def test_base_only_still_writes_activation(self):
        """All-base routing skips the LoRA dots but must still emit silu(gate)*up
        of the base projection (the always-write contract)."""
        device = torch.device("cuda")
        K, H = 256, 256
        adapter_ranks = [16, 32]
        NA = 2
        M = 40
        layer = _make_swiglu_layer(K, H, NA, max(adapter_ranks), device)
        _fill_lora_weights(layer, adapter_ranks)

        torch.manual_seed(1)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)

        layer.finalize_weights(adapter_ranks)
        _wire_single_module_ctx(layer, adapter_indices)
        out, _ = layer.forward(x)

        gateup = x @ layer.base_layer.weight.data.T  # base only, no LoRA delta
        ref = _silu_mul(gateup, H)
        assert out.is_contiguous()
        torch.testing.assert_close(out.float(), ref, atol=0.02, rtol=0.02)
