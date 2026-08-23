# SPDX-License-Identifier: Apache-2.0
"""Integration tests for FusedLoRAKernelMeta and multi-module bitmask metadata.

Tests the integration layer between FusedLoRAKernelMeta (which computes
per-module bitmasks in one batched call) and multiple SwitchedLoRALinear
modules with divergent remap tables (different per-module applicability).

Also verifies early-exit behavior: when a module's bitmask is all-zero for
a tile (no applicable adapters active), the expand kernel contributes nothing.

Requires CUDA GPU and vLLM installed.
"""

import gc

import pytest
import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _try_imports():
    try:
        from vllm.model_executor.layers.linear import (  # noqa: F401
            ColumnParallelLinear,
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

    from granite_switch.kernels import BLOCK_M
    from granite_switch.vllm.core.lora import SwitchedLoRALinear
    from granite_switch.vllm.core.lora_kernel_meta import (
        FusedLoRAKernelMeta,
        LoRAContext,
        _compute_bitmasks_reference,
    )
    from tests.shared.vllm_distributed import ensure_distributed


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True, scope="module")
def _init_distributed():
    if not _VLLM_AVAILABLE:
        return
    ensure_distributed()


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    yield
    gc.collect()
    torch.cuda.empty_cache()


# ── Helpers ───────────────────────────────────────────────────────────


def _make_layer(K, N, num_adapters, max_rank, device, num_slices=1, output_slices=None):
    """Create a SwitchedLoRALinear with initialized base weights."""
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        if num_slices == 1:
            base = RowParallelLinear(
                input_size=K,
                output_size=N,
                bias=False,
                params_dtype=torch.bfloat16,
            ).to(device)
        else:
            base = ColumnParallelLinear(
                input_size=K,
                output_size=sum(output_slices),
                bias=False,
                params_dtype=torch.bfloat16,
            ).to(device)
        layer = SwitchedLoRALinear(
            base_layer=base,
            num_adapters=num_adapters,
            max_lora_rank=max_rank,
            num_slices=num_slices,
            output_slices=output_slices,
        )
    with torch.no_grad():
        base.weight.data.normal_(0, 0.02)
    return layer


def _fill_selective(layer, adapter_ranks, applicable_mask):
    """Fill lora_A/B only for applicable adapters.

    Args:
        applicable_mask: list of bool, length num_adapters.
    """
    with torch.no_grad():
        for i, r in enumerate(adapter_ranks):
            if not applicable_mask[i]:
                continue
            if layer.num_slices == 1:
                layer.lora_A.data[i, 0, :r, :] = (
                    torch.randn(
                        r, layer.in_features, dtype=layer._dtype, device=layer._device
                    )
                    * 0.1
                )
                layer.lora_B.data[i, 0, :, :r] = (
                    torch.randn(
                        layer.out_features, r, dtype=layer._dtype, device=layer._device
                    )
                    * 0.1
                )
            else:
                for s in range(layer.num_slices):
                    layer.lora_A_slices[s].data[i, 0, :r, :] = (
                        torch.randn(
                            r,
                            layer.in_features,
                            dtype=layer._dtype,
                            device=layer._device,
                        )
                        * 0.1
                    )
                    N_s = layer.output_slices[s]
                    layer.lora_B_slices[s].data[i, 0, :, :r] = (
                        torch.randn(N_s, r, dtype=layer._dtype, device=layer._device)
                        * 0.1
                    )


def _run_layer_forward(layer, x, adapter_indices):
    """Run a finalized layer's forward with proper context setup."""
    remap_table_2d = layer.remap_table.unsqueeze(0)
    bitmask = _compute_bitmasks_reference(adapter_indices, remap_table_2d.T)
    ctx = LoRAContext()
    ctx.adapter_indices = adapter_indices
    ctx.per_module_bitmasks = bitmask
    # Kernel-local indices forward() reads (single module -> row 0); mirrors
    # FusedLoRAKernelMeta.prepare_and_store for the one-module case.
    ctx.remapped_indices = layer.remap_table[adapter_indices].unsqueeze(0).contiguous()
    layer._lora_ctx = ctx
    layer._module_idx = 0
    out, _ = layer.forward(x)
    return out


# ════════════════════════════════════════════════════════════════════════
# 1. Multi-module bitmask metadata
#
# Verifies that FusedLoRAKernelMeta produces correct per-module bitmasks
# when modules have different remap tables (different applicability).
# ════════════════════════════════════════════════════════════════════════

# Production-like module setup: 4 modules with different applicability
# (mimics qkv_proj, o_proj, input_linear, output_linear)
_MULTI_MODULE_CONFIGS = [
    # (K, N, applicable_mask, label)
    # All 12 adapters applicable (qkv_proj — all adapters target it)
    (2560, 2560, [True] * 12, "qkv_all_applicable"),
    # 8 of 12 applicable (o_proj — adapters 5,7,11 are qkv-only)
    (
        2560,
        2560,
        [True, True, True, True, True, False, True, False, True, True, True, False],
        "o_proj_partial",
    ),
    # 4 of 12 applicable (shared_input_linear)
    (
        2560,
        2560,
        [
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        "input_linear_sparse",
    ),
    # 4 of 12 applicable (shared_output_linear — same set)
    (
        2560,
        2560,
        [
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        "output_linear_sparse",
    ),
]


class TestMultiModuleBitmaskMeta:
    """Verify FusedLoRAKernelMeta produces correct per-module bitmasks."""

    def _setup_multi_module(self, device):
        """4 modules with divergent applicability; finalize + register tables."""
        NA = 12
        adapter_ranks = [16, 32, 32, 16, 16, 32, 16, 32, 16, 32, 16, 16]
        max_rank = max(adapter_ranks)

        torch.manual_seed(0)
        modules = []
        for K, N, applicable, label in _MULTI_MODULE_CONFIGS:
            layer = _make_layer(K, N, NA, max_rank, device)
            _fill_selective(layer, adapter_ranks, applicable)
            layer.finalize_weights(adapter_ranks)
            modules.append(layer)

        # Assign module indices and register remap tables
        for idx, m in enumerate(modules):
            m._module_idx = idx

        lora_meta = FusedLoRAKernelMeta(device=device)
        all_remap_tables = torch.stack([m.remap_table for m in modules], dim=0)
        module_cfg_keys = [m._block_cfg_key for m in modules]
        lora_meta.register_remap_tables(all_remap_tables, module_cfg_keys)

        return modules, lora_meta, adapter_ranks

    def test_bitmask_shape(self):
        """per_module_bitmasks has shape [num_modules, num_tiles]."""
        device = torch.device("cuda")
        modules, lora_meta, _ = self._setup_multi_module(device)
        M = 128
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        adapter_indices[64:] = 1

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        num_tiles = (M + BLOCK_M - 1) // BLOCK_M
        assert ctx.per_module_bitmasks.shape == (4, num_tiles)

    def test_all_base_produces_zero_bitmasks(self):
        """When all tokens are base, every module's bitmask should be 0."""
        device = torch.device("cuda")
        modules, lora_meta, _ = self._setup_multi_module(device)
        M = 128
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        assert (ctx.per_module_bitmasks == 0).all()

    def test_non_applicable_adapter_produces_zero_bitmask(self):
        """Activating an adapter non-applicable to a module → bitmask=0 for that module."""
        device = torch.device("cuda")
        modules, lora_meta, _ = self._setup_multi_module(device)
        M = 64

        # Adapter 6 (global index 6, 1-indexed) is only applicable to
        # qkv_all_applicable and o_proj_partial, NOT input_linear or output_linear
        adapter_indices = torch.full((M,), 6, dtype=torch.long, device=device)

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        # Module 0 (qkv): should have nonzero bitmask
        assert (ctx.per_module_bitmasks[0] != 0).any()
        # Module 2 (input_linear): adapter 6 not applicable → all zero
        assert (ctx.per_module_bitmasks[2] == 0).all()
        # Module 3 (output_linear): same
        assert (ctx.per_module_bitmasks[3] == 0).all()

    def test_divergent_bitmasks_across_modules(self):
        """Different modules get different bitmasks for the same adapter_indices."""
        device = torch.device("cuda")
        modules, lora_meta, _ = self._setup_multi_module(device)
        M = 128

        # Mix of adapters: some applicable to all modules, some only to subset
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        # Adapter 2 (applicable to all 4 modules)
        adapter_indices[0:32] = 2
        # Adapter 6 (applicable to modules 0,1 only)
        adapter_indices[32:64] = 6
        # Adapter 12 (context-attribution: applicable to module 0 only — qkv-only)
        adapter_indices[64:96] = 12

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        # Module 0 (qkv): all three adapters are applicable
        assert (ctx.per_module_bitmasks[0] != 0).any()

        # Module 2 (input_linear): only adapter 2 is applicable
        # Tiles for tokens 0-31 should have bits set; tiles for 32-95 should not
        tile_boundary = 32 // BLOCK_M  # tile where adapter 2 ends
        # First tile(s) should be nonzero (adapter 2)
        assert (ctx.per_module_bitmasks[2, :tile_boundary] != 0).all()
        # Later tiles should be zero (adapters 6 and 12 not applicable)
        assert (ctx.per_module_bitmasks[2, tile_boundary:] == 0).all()

    def test_bitmask_matches_individual_computation(self):
        """Batched bitmask via FusedLoRAKernelMeta matches per-module _compute_bitmasks_reference."""
        device = torch.device("cuda")
        modules, lora_meta, _ = self._setup_multi_module(device)
        M = 96

        torch.manual_seed(77)
        adapter_indices = torch.randint(0, 13, (M,), dtype=torch.long, device=device)

        # Batched computation
        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)
        batched = ctx.per_module_bitmasks  # [4, num_tiles]

        # Per-module individual computation
        for mod_idx, module in enumerate(modules):
            remap_2d = module.remap_table.unsqueeze(0)  # [1, NA+1]
            individual = _compute_bitmasks_reference(adapter_indices, remap_2d.T)
            torch.testing.assert_close(
                batched[mod_idx : mod_idx + 1],
                individual,
                msg=f"Bitmask mismatch at module {mod_idx}",
            )

    @pytest.mark.parametrize("M", [1, 31, 32, 33, 63, 64, 65, 128, 256])
    def test_various_token_counts(self, M):
        """Bitmask computation works at various M including non-BLOCK_M-aligned."""
        device = torch.device("cuda")
        modules, lora_meta, _ = self._setup_multi_module(device)

        torch.manual_seed(M)
        adapter_indices = torch.randint(0, 13, (M,), dtype=torch.long, device=device)

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        num_tiles = (M + BLOCK_M - 1) // BLOCK_M
        assert ctx.per_module_bitmasks.shape == (4, num_tiles)

        # Verify against individual per-module computation
        for mod_idx, module in enumerate(modules):
            remap_2d = module.remap_table.unsqueeze(0)
            individual = _compute_bitmasks_reference(adapter_indices, remap_2d.T)
            torch.testing.assert_close(
                ctx.per_module_bitmasks[mod_idx : mod_idx + 1],
                individual,
            )


# ════════════════════════════════════════════════════════════════════════
# 2. Early-exit verification
#
# Confirms that when a module's bitmask is all-zero (no applicable adapters
# active in a tile), the kernel produces output identical to base-only.
# This verifies the optimization path is correct, not just that it exists.
# ════════════════════════════════════════════════════════════════════════


class TestEarlyExit:
    """Verify that zero-bitmask tiles produce base-only output."""

    def _make_partially_applicable_layer(self, device):
        """Layer where only adapters 0,1 are applicable (2,3,4 are not)."""
        K, N = 2560, 2560
        adapter_ranks = [16, 32, 16, 32, 16]
        NA = 5
        max_rank = 32
        applicable = [True, True, False, False, False]

        layer = _make_layer(K, N, NA, max_rank, device)
        _fill_selective(layer, adapter_ranks, applicable)
        layer.finalize_weights(adapter_ranks)
        layer._module_idx = 0
        return layer, adapter_ranks

    def test_non_applicable_adapters_no_contribution(self):
        """Tokens assigned to non-applicable adapters get base-only output."""
        device = torch.device("cuda")
        layer, ranks = self._make_partially_applicable_layer(device)
        M = 64

        torch.manual_seed(42)
        x = (
            torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )

        # All tokens assigned to adapter 3 (non-applicable, global index 4)
        adapter_indices = torch.full((M,), 4, dtype=torch.long, device=device)

        fused_out = _run_layer_forward(layer, x, adapter_indices)
        base_out = x @ layer.base_layer.weight.data.T

        torch.testing.assert_close(fused_out, base_out, atol=1e-5, rtol=1e-5)

    def test_mixed_applicable_and_non_applicable(self):
        """Tiles with only non-applicable adapters match base; tiles with applicable don't."""
        device = torch.device("cuda")
        layer, ranks = self._make_partially_applicable_layer(device)
        M = 128

        torch.manual_seed(42)
        x = (
            torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )

        # First 64 tokens: adapter 3 (non-applicable)
        # Last 64 tokens: adapter 1 (applicable, rank 16)
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        adapter_indices[:64] = 4  # non-applicable
        adapter_indices[64:] = 1  # applicable

        fused_out = _run_layer_forward(layer, x, adapter_indices)
        base_out = x @ layer.base_layer.weight.data.T

        # Non-applicable region: should match base exactly
        torch.testing.assert_close(
            fused_out[:64],
            base_out[:64],
            atol=1e-5,
            rtol=1e-5,
            msg="Non-applicable adapter region should match base",
        )

        # Applicable region: should differ from base
        assert not torch.allclose(fused_out[64:], base_out[64:], atol=0.001), (
            "Applicable adapter region should differ from base"
        )

    def test_all_base_exact_match(self):
        """All-base adapter_indices → output exactly equals base linear."""
        device = torch.device("cuda")
        layer, ranks = self._make_partially_applicable_layer(device)
        M = 64

        torch.manual_seed(42)
        x = (
            torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)

        fused_out = _run_layer_forward(layer, x, adapter_indices)
        base_out = x @ layer.base_layer.weight.data.T

        torch.testing.assert_close(fused_out, base_out, atol=1e-5, rtol=1e-5)

    def test_sparse_non_applicable_in_tile(self):
        """Single non-applicable adapter token in a tile of base tokens → no contamination."""
        device = torch.device("cuda")
        layer, ranks = self._make_partially_applicable_layer(device)
        M = BLOCK_M  # exactly one tile

        torch.manual_seed(42)
        x = (
            torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )

        # One token assigned to non-applicable adapter, rest base
        adapter_indices = torch.zeros(M, dtype=torch.long, device=device)
        adapter_indices[15] = 4  # non-applicable

        fused_out = _run_layer_forward(layer, x, adapter_indices)
        base_out = x @ layer.base_layer.weight.data.T

        # Entire tile should match base (bitmask bit for adapter 4 is 0 in this module)
        torch.testing.assert_close(fused_out, base_out, atol=1e-5, rtol=1e-5)

    def test_applicable_token_in_tile_affects_only_that_token(self):
        """An applicable adapter token in a tile only changes that token's output."""
        device = torch.device("cuda")
        layer, ranks = self._make_partially_applicable_layer(device)
        M = BLOCK_M * 2  # two tiles

        torch.manual_seed(42)
        x = (
            torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )

        # All base
        adapter_indices_base = torch.zeros(M, dtype=torch.long, device=device)
        out_base = _run_layer_forward(layer, x, adapter_indices_base)

        # One token in tile 0 assigned to applicable adapter
        adapter_indices_one = torch.zeros(M, dtype=torch.long, device=device)
        adapter_indices_one[10] = 1  # applicable adapter
        out_one = _run_layer_forward(layer, x, adapter_indices_one)

        # Token 10 should differ
        assert not torch.allclose(out_one[10], out_base[10], atol=0.001)

        # All other tokens in the SAME tile should be unchanged
        # (the expand kernel is per-token, not per-tile in its output)
        for i in range(BLOCK_M):
            if i == 10:
                continue
            torch.testing.assert_close(
                out_one[i],
                out_base[i],
                atol=1e-5,
                rtol=1e-5,
                msg=f"Token {i} in same tile should be unaffected",
            )

        # Second tile entirely unaffected
        torch.testing.assert_close(
            out_one[BLOCK_M:],
            out_base[BLOCK_M:],
            atol=1e-5,
            rtol=1e-5,
            msg="Second tile should be completely unaffected",
        )


# ════════════════════════════════════════════════════════════════════════
# 3. Multi-module forward integration
#
# Simulates the full model pipeline: FusedLoRAKernelMeta computes bitmasks
# once, then each module reads its own row and produces correct output.
# ════════════════════════════════════════════════════════════════════════


class TestMultiModuleForward:
    """End-to-end: kernel meta → per-module bitmask → correct forward per module."""

    def _setup_pipeline(self, device):
        """Build 4 modules mimicking one decoder layer's worth of SwitchedLoRALinear."""
        NA = 12
        adapter_ranks = [16, 32, 32, 16, 16, 32, 16, 32, 16, 32, 16, 16]
        max_rank = max(adapter_ranks)

        # Module applicability patterns (production-like)
        applicability = [
            [True] * 12,  # qkv_proj: all adapters
            [
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                True,
                False,
            ],  # o_proj
            [
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],  # input_linear
            [
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],  # output_linear
        ]

        geometries = [
            (2560, (2560, 512, 512), 3, "qkv_proj"),
            (2560, 2560, 1, "o_proj"),
            (2560, (8192, 8192), 2, "input_linear"),
            (8192, 2560, 1, "output_linear"),
        ]

        torch.manual_seed(0)
        modules = []
        for (K, N_or_slices, num_slices, label), applicable in zip(
            geometries, applicability
        ):
            if num_slices > 1:
                layer = _make_layer(
                    K,
                    None,
                    NA,
                    max_rank,
                    device,
                    num_slices=num_slices,
                    output_slices=N_or_slices,
                )
            else:
                layer = _make_layer(K, N_or_slices, NA, max_rank, device)
            _fill_selective(layer, adapter_ranks, applicable)
            layer.finalize_weights(adapter_ranks)
            modules.append(layer)

        # Wire up kernel meta (as the model does)
        for idx, m in enumerate(modules):
            m._module_idx = idx

        lora_meta = FusedLoRAKernelMeta(device=device)
        all_remap_tables = torch.stack([m.remap_table for m in modules], dim=0)
        module_cfg_keys = [m._block_cfg_key for m in modules]
        lora_meta.register_remap_tables(all_remap_tables, module_cfg_keys)

        return modules, lora_meta, adapter_ranks

    @pytest.mark.parametrize("seed", range(10))
    def test_all_modules_correct_with_shared_context(self, seed):
        """Each module produces correct output when sharing one LoRAContext."""
        device = torch.device("cuda")
        modules, lora_meta, adapter_ranks = self._setup_pipeline(device)
        M = 128

        torch.manual_seed(seed + 100)
        adapter_indices = torch.randint(0, 13, (M,), dtype=torch.long, device=device)

        # Compute bitmasks via kernel meta (single batched call)
        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        # Run each module and compare against its own reference
        for mod_idx, layer in enumerate(modules):
            torch.manual_seed(seed + mod_idx)
            x = (
                torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
                * 0.01
            )

            # Reference: per-token matmul using checkpoint-format weights
            W_base = layer.base_layer.weight.data
            ref = (x @ W_base.T).clone()
            S = layer.num_slices
            slice_starts = [0]
            for s in range(S):
                slice_starts.append(slice_starts[-1] + layer.output_slices[s])

            for m_tok in range(M):
                ai = adapter_indices[m_tok].item()
                if ai == 0:
                    continue
                a = ai - 1
                r = adapter_ranks[a]
                # Check if applicable via remap table
                if layer.remap_table[ai].item() == 0:
                    continue
                for s in range(S):
                    if S == 1:
                        lA = layer.lora_A.data[a, 0, :r, :]
                        lB = layer.lora_B.data[a, 0, :, :r]
                    else:
                        lA = layer.lora_A_slices[s].data[a, 0, :r, :]
                        lB = layer.lora_B_slices[s].data[a, 0, :, :r]
                    shrink = x[m_tok] @ lA.T
                    expand = shrink @ lB.T
                    ns = slice_starts[s]
                    ne = slice_starts[s + 1]
                    ref[m_tok, ns:ne] += expand

            # Fused forward using shared context
            layer._lora_ctx = ctx
            fused_out, _ = layer.forward(x)

            torch.testing.assert_close(
                fused_out,
                ref,
                atol=0.75,
                rtol=0.05,
                msg=f"Module {mod_idx} forward mismatch (seed={seed})",
            )

    def test_pipeline_with_only_sparse_module_adapters(self):
        """Activate adapters that only target input/output_linear (modules 2,3)."""
        device = torch.device("cuda")
        modules, lora_meta, adapter_ranks = self._setup_pipeline(device)
        M = 64

        # Adapter 2 (global index 2) targets all modules
        # but adapter 1 (global index 1) also targets all
        # Adapter 1 (citations) only targets qkv + o_proj (modules 0,1)
        # So let's use ONLY adapter 1 → modules 2,3 should see base-only
        adapter_indices = torch.full((M,), 1, dtype=torch.long, device=device)

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        # Module 2 (input_linear): adapter 1 (citations) is NOT applicable
        # atol=5e-4: fused w_ext GEMM uses different tiling than standalone base GEMM
        torch.manual_seed(55)
        x = (
            torch.randn(M, modules[2].in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )
        modules[2]._lora_ctx = ctx
        out_input, _ = modules[2].forward(x)
        base_input = x @ modules[2].base_layer.weight.data.T
        torch.testing.assert_close(
            out_input,
            base_input,
            atol=5e-4,
            rtol=1e-4,
            msg="input_linear should be base-only for citations adapter",
        )

        # Module 3 (output_linear): same — adapter 1 not applicable
        torch.manual_seed(56)
        x3 = (
            torch.randn(M, modules[3].in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )
        modules[3]._lora_ctx = ctx
        out_output, _ = modules[3].forward(x3)
        base_output = x3 @ modules[3].base_layer.weight.data.T
        torch.testing.assert_close(
            out_output,
            base_output,
            atol=5e-4,
            rtol=1e-4,
            msg="output_linear should be base-only for citations adapter",
        )

        # Module 0 (qkv): adapter 1 IS applicable → should differ from base
        torch.manual_seed(57)
        x0 = (
            torch.randn(M, modules[0].in_features, dtype=torch.bfloat16, device=device)
            * 0.01
        )
        modules[0]._lora_ctx = ctx
        out_qkv, _ = modules[0].forward(x0)
        base_qkv = x0 @ modules[0].base_layer.weight.data.T
        assert not torch.allclose(out_qkv, base_qkv, atol=0.001), (
            "qkv_proj should have LoRA contribution for citations adapter"
        )


# ════════════════════════════════════════════════════════════════════════
# 4. Per-module divergent ranks
#
# Verifies that the same adapter can have different ranks in different
# modules. This is supported by the kernel (each module independently
# builds its own tier structure) but was not previously tested.
# ════════════════════════════════════════════════════════════════════════


class TestPerModuleDivergentRanks:
    """Same adapter, different ranks across modules."""

    def _setup_divergent(self, device):
        """Build 3 modules where adapter 0 has rank 16/32/64 respectively."""
        NA = 4
        K = 2560
        N = 2560

        # Per-module rank lists: adapter 0 varies, others are constant
        module_ranks = [
            [16, 32, 16, 32],  # module 0: adapter 0 has rank 16
            [32, 32, 16, 32],  # module 1: adapter 0 has rank 32
            [64, 32, 16, 32],  # module 2: adapter 0 has rank 64
        ]

        torch.manual_seed(0)
        modules = []
        for mod_idx, ranks in enumerate(module_ranks):
            max_rank = max(ranks)
            layer = _make_layer(K, N, NA, max_rank, device)
            # Fill all adapters as applicable with their module-specific rank
            with torch.no_grad():
                for i, r in enumerate(ranks):
                    layer.lora_A.data[i, 0, :r, :] = (
                        torch.randn(r, K, dtype=layer._dtype, device=device) * 0.1
                    )
                    layer.lora_B.data[i, 0, :, :r] = (
                        torch.randn(N, r, dtype=layer._dtype, device=device) * 0.1
                    )
            layer.finalize_weights(ranks)
            modules.append(layer)

        for idx, m in enumerate(modules):
            m._module_idx = idx

        lora_meta = FusedLoRAKernelMeta(device=device)
        all_remap_tables = torch.stack([m.remap_table for m in modules], dim=0)
        module_cfg_keys = [m._block_cfg_key for m in modules]
        lora_meta.register_remap_tables(all_remap_tables, module_cfg_keys)

        return modules, lora_meta, module_ranks

    def test_different_w_ext_shapes(self):
        """Each module's w_ext has different row count due to different rank tiers."""
        device = torch.device("cuda")
        modules, _, module_ranks = self._setup_divergent(device)

        # Module 0: adapters [16,32,16,32] → tiers: rank16 has 2 adapters, rank32 has 2
        # w_ext rows = N + 2*16 + 2*32 = 2560 + 32 + 64 = 2656
        # Module 2: adapters [64,32,16,32] → tiers: rank16 has 1, rank32 has 2, rank64 has 1
        # w_ext rows = N + 1*16 + 2*32 + 1*64 = 2560 + 16 + 64 + 64 = 2704
        for mod_idx, (layer, ranks) in enumerate(zip(modules, module_ranks)):
            expected_extra = sum(ranks)
            expected_rows = 2560 + expected_extra
            assert layer.w_ext.shape[0] == expected_rows, (
                f"Module {mod_idx}: expected w_ext rows={expected_rows}, "
                f"got {layer.w_ext.shape[0]}"
            )

    def test_remap_tables_differ(self):
        """Each module's remap table reflects its own tier ordering."""
        device = torch.device("cuda")
        modules, _, _ = self._setup_divergent(device)

        # Remap tables should all be non-trivial (all adapters applicable)
        for m in modules:
            assert (m.remap_table[1:] > 0).all(), "All adapters should be applicable"

        # But the actual positions may differ because tier ordering changes
        # when adapter 0's rank changes
        remap_0 = modules[0].remap_table
        remap_2 = modules[2].remap_table
        # Module 0: ranks [16,32,16,32] → sorted by rank: adapters 0,2 (rank16), 1,3 (rank32)
        #   remap: adapter0→1, adapter2→2, adapter1→3, adapter3→4
        # Module 2: ranks [64,32,16,32] → sorted by rank: adapter2 (rank16), adapters1,3 (rank32), adapter0 (rank64)
        #   remap: adapter2→1, adapter1→2, adapter3→3, adapter0→4
        assert remap_0[1].item() != remap_2[1].item(), (
            "Adapter 0 should have different kernel-local positions across modules"
        )

    def test_forward_correctness_with_divergent_ranks(self):
        """Each module produces correct output using its own rank for adapter 0."""
        device = torch.device("cuda")
        modules, lora_meta, module_ranks = self._setup_divergent(device)
        M = 64

        # All tokens use adapter 0 (global index 1)
        adapter_indices = torch.full((M,), 1, dtype=torch.long, device=device)

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        for mod_idx, (layer, ranks) in enumerate(zip(modules, module_ranks)):
            torch.manual_seed(mod_idx + 10)
            x = (
                torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
                * 0.01
            )

            # Reference: base + lora_A[:r] @ lora_B[:, :r]
            r = ranks[0]  # adapter 0's rank in this module
            W_base = layer.base_layer.weight.data
            lA = layer.lora_A.data[0, 0, :r, :]
            lB = layer.lora_B.data[0, 0, :, :r]
            ref = (x @ W_base.T) + (x @ lA.T) @ lB.T

            layer._lora_ctx = ctx
            fused_out, _ = layer.forward(x)

            torch.testing.assert_close(
                fused_out,
                ref,
                atol=0.25,
                rtol=0.05,
                msg=f"Module {mod_idx} (adapter 0 rank={r}) forward mismatch",
            )

    @pytest.mark.parametrize("seed", range(10))
    def test_mixed_adapters_divergent_ranks(self, seed):
        """Random adapter mix with divergent per-module ranks."""
        device = torch.device("cuda")
        modules, lora_meta, module_ranks = self._setup_divergent(device)
        M = 128

        torch.manual_seed(seed + 200)
        adapter_indices = torch.randint(0, 5, (M,), dtype=torch.long, device=device)

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        for mod_idx, (layer, ranks) in enumerate(zip(modules, module_ranks)):
            torch.manual_seed(seed + mod_idx + 300)
            x = (
                torch.randn(M, layer.in_features, dtype=torch.bfloat16, device=device)
                * 0.01
            )

            # Per-token reference
            W_base = layer.base_layer.weight.data
            ref = (x @ W_base.T).clone()
            for m_tok in range(M):
                ai = adapter_indices[m_tok].item()
                if ai == 0:
                    continue
                a = ai - 1
                r = ranks[a]
                lA = layer.lora_A.data[a, 0, :r, :]
                lB = layer.lora_B.data[a, 0, :, :r]
                ref[m_tok] += (x[m_tok] @ lA.T) @ lB.T

            layer._lora_ctx = ctx
            fused_out, _ = layer.forward(x)

            torch.testing.assert_close(
                fused_out,
                ref,
                atol=0.25,
                rtol=0.05,
                msg=f"Module {mod_idx} forward mismatch (seed={seed})",
            )

    def test_adapter_rank_zero_means_non_applicable(self):
        """Rank 0 in one module makes an adapter non-applicable there only."""
        device = torch.device("cuda")
        NA = 3
        K, N = 2560, 2560

        # Module 0: all adapters applicable at [16, 32, 16]
        # Module 1: adapter 0 has rank 0 → non-applicable
        module_ranks = [
            [16, 32, 16],
            [0, 32, 16],
        ]

        torch.manual_seed(0)
        modules = []
        for ranks in module_ranks:
            max_rank = max(ranks) if max(ranks) > 0 else 32
            layer = _make_layer(K, N, NA, max_rank, device)
            with torch.no_grad():
                for i, r in enumerate(ranks):
                    if r == 0:
                        continue
                    layer.lora_A.data[i, 0, :r, :] = (
                        torch.randn(r, K, dtype=layer._dtype, device=device) * 0.1
                    )
                    layer.lora_B.data[i, 0, :, :r] = (
                        torch.randn(N, r, dtype=layer._dtype, device=device) * 0.1
                    )
            layer.finalize_weights(ranks)
            modules.append(layer)

        for idx, m in enumerate(modules):
            m._module_idx = idx

        lora_meta = FusedLoRAKernelMeta(device=device)
        all_remap_tables = torch.stack([m.remap_table for m in modules], dim=0)
        module_cfg_keys = [m._block_cfg_key for m in modules]
        lora_meta.register_remap_tables(all_remap_tables, module_cfg_keys)

        M = 64
        # All tokens: adapter 0 (global index 1)
        adapter_indices = torch.full((M,), 1, dtype=torch.long, device=device)

        ctx = LoRAContext()
        lora_meta.prepare_and_store(adapter_indices, ctx)

        torch.manual_seed(99)
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.01

        # Module 0: adapter 0 IS applicable (rank 16) → LoRA contribution
        modules[0]._lora_ctx = ctx
        out_0, _ = modules[0].forward(x)
        base_0 = x @ modules[0].base_layer.weight.data.T
        assert not torch.allclose(out_0, base_0, atol=0.001), (
            "Module 0 should have LoRA contribution for adapter 0"
        )

        # Module 1: adapter 0 is rank 0 → non-applicable → base only
        modules[1]._lora_ctx = ctx
        out_1, _ = modules[1].forward(x)
        base_1 = x @ modules[1].base_layer.weight.data.T
        torch.testing.assert_close(
            out_1,
            base_1,
            atol=5e-4,
            rtol=1e-4,
            msg="Module 1 should be base-only (adapter 0 rank=0)",
        )
