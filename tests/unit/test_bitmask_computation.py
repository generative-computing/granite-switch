# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-module bitmask computation.

Tests _compute_bitmasks_reference directly with hand-built remap tables,
parametrized over BLOCK_M so the tests remain valid as tiling changes.
"""

import pytest
import torch

import granite_switch.vllm.core.lora_kernel_meta as meta_mod
from granite_switch.vllm.core.lora_kernel_meta import _compute_bitmasks_reference

# Production BLOCK_M is 32 (kernels/switch_lora_kernel.py); the others are synthetic
# probes of the tile arithmetic, so these tests keep holding if the kernel retiles.
_BLOCK_M_VALUES = [1, 8, 16, 32, 64]


def _use_block_m(monkeypatch, value):
    monkeypatch.setattr(meta_mod, "BLOCK_M", value)
    return value


@pytest.fixture(params=_BLOCK_M_VALUES)
def block_m(request, monkeypatch):
    return _use_block_m(monkeypatch, request.param)


@pytest.fixture(params=[b for b in _BLOCK_M_VALUES if b > 1])
def block_m_multi_token(request, monkeypatch):
    """The same sweep minus BLOCK_M=1: tiles that hold more than one token.

    Two tests need a tile wide enough to hold two *different* things at once, and
    at BLOCK_M=1 that is arithmetically impossible rather than merely untested:

    - Splitting a tile between two adapters needs >= 2 tokens. ``M // 2`` is 0 at
      M=1, so the "split" collapses to a single adapter and the expected bitmask
      is unreachable.
    - A partially-filled tile needs ``M % BLOCK_M != 0``, and every M is divisible
      by 1, so there is no padding for the test to find spurious bits in.

    Narrowing the sweep rather than skipping inside the test body, because the two
    read differently in a report: a skip says "might work, nobody checked", when
    the truth is "this configuration does not exist". Asserting the degenerate
    outcome instead would be worse than either -- a green cell pinning arithmetic
    no caller relies on, for a tiling production never uses.
    """
    return _use_block_m(monkeypatch, request.param)


def _make_remap_tables_t(remap_tables):
    """Build remap_tables_t [NA+1, num_modules] from list of per-module remaps.

    Each remap is a list of length NA+1 where remap[0]=0 (base) and
    remap[j] = kernel-local position of global adapter j in that module.
    """
    return torch.tensor(remap_tables, dtype=torch.long).T.contiguous()


class TestSingleAdapterFullTile:
    def test_single_adapter_all_positions(self, block_m):
        # One tile, one adapter at every position → bit 0 set
        M = block_m
        adapter_indices = torch.ones(M, dtype=torch.long)  # all adapter 1

        # Single module, adapter 1 maps to kernel-local 1
        remap_tables_t = _make_remap_tables_t([[0, 1]])  # [NA+1=2, num_modules=1]

        result = _compute_bitmasks_reference(adapter_indices, remap_tables_t)

        assert result.shape == (1, 1)  # [num_modules=1, num_tiles=1]
        assert result[0, 0].item() == 1 << 0


class TestMultipleAdaptersOneTile:
    def test_two_adapters_split(self, block_m_multi_token):
        # One tile: first half adapter 1, second half adapter 2
        M = block_m_multi_token
        half = M // 2
        adapter_indices = torch.cat(
            [
                torch.ones(half, dtype=torch.long),
                torch.full((M - half,), 2, dtype=torch.long),
            ]
        )

        # Single module, both adapters applicable: 1→1, 2→2
        remap_tables_t = _make_remap_tables_t([[0, 1, 2]])

        result = _compute_bitmasks_reference(adapter_indices, remap_tables_t)

        assert result.shape == (1, 1)
        assert result[0, 0].item() == (1 << 0) | (1 << 1)


class TestAllBaseTile:
    def test_all_base(self, block_m):
        # One tile, all tokens are base (adapter_index=0) → bitmask = 0
        M = block_m
        adapter_indices = torch.zeros(M, dtype=torch.long)

        remap_tables_t = _make_remap_tables_t([[0, 1, 2]])

        result = _compute_bitmasks_reference(adapter_indices, remap_tables_t)

        assert result.shape == (1, 1)
        assert result[0, 0].item() == 0


class TestNonApplicableAdapter:
    def test_non_applicable_sets_no_bit(self, block_m):
        # Adapter 2 is non-applicable for this module (remap=0).
        # Tokens assigned to adapter 2 should NOT set any bit.
        M = block_m
        adapter_indices = torch.full((M,), 2, dtype=torch.long)

        # Module: adapter 1→1 (applicable), adapter 2→0 (non-applicable)
        remap_tables_t = _make_remap_tables_t([[0, 1, 0]])

        result = _compute_bitmasks_reference(adapter_indices, remap_tables_t)

        assert result[0, 0].item() == 0


class TestMultiModuleDivergence:
    def test_same_input_different_bitmasks(self, block_m):
        # Two modules with different remap tables.
        # Module 0: adapter 1 applicable (1→1), adapter 2 non-applicable (2→0)
        # Module 1: adapter 1 non-applicable (1→0), adapter 2 applicable (2→1)
        M = block_m
        adapter_indices = torch.ones(M, dtype=torch.long)  # all adapter 1

        remap_tables_t = _make_remap_tables_t(
            [
                [0, 1, 0],  # module 0: adapter 1 → local 1
                [0, 0, 1],  # module 1: adapter 1 → local 0 (non-applicable)
            ]
        )

        result = _compute_bitmasks_reference(adapter_indices, remap_tables_t)

        assert result.shape == (2, 1)
        assert result[0, 0].item() == 1 << 0  # module 0 sees adapter 1
        assert result[1, 0].item() == 0  # module 1 does not


class TestTileBoundary:
    def test_different_patterns_per_tile(self, block_m):
        # Two full tiles: tile 0 has adapter 1, tile 1 has adapter 2.
        adapter_indices = torch.cat(
            [
                torch.ones(block_m, dtype=torch.long),
                torch.full((block_m,), 2, dtype=torch.long),
            ]
        )

        # Single module, both applicable: 1→1, 2→2
        remap_tables_t = _make_remap_tables_t([[0, 1, 2]])

        result = _compute_bitmasks_reference(adapter_indices, remap_tables_t)

        assert result.shape == (1, 2)
        assert result[0, 0].item() == 1 << 0  # tile 0: adapter 1 only
        assert result[0, 1].item() == 1 << 1  # tile 1: adapter 2 only


class TestBitmaskOverflow:
    def test_more_than_64_adapters_raises(self):
        # int64 bitmask can represent at most 64 kernel-local positions.
        # If a module had >64 applicable adapters, the bitmask would silently
        # lose bits (shift >= 64 produces 0). The contract is enforced at
        # registration time: FusedLoRAKernelMeta.register_remap_tables
        # asserts max kernel-local position <= 64. (The lower-level
        # _compute_bitmasks_reference is not the guard site — it trusts its input.)
        from granite_switch.vllm.core.lora_kernel_meta import FusedLoRAKernelMeta

        NA = 65
        # One module, all 65 adapters applicable: remap 1→1, ..., 65→65.
        all_remap_tables = torch.tensor(
            [[0, *list(range(1, NA + 1))]], dtype=torch.long
        )  # [num_modules=1, NA+1]
        module_cfg_keys = [(256, 256, NA)]

        meta = FusedLoRAKernelMeta(torch.device("cpu"))
        with pytest.raises(AssertionError, match="at most 64"):
            meta.register_remap_tables(all_remap_tables, module_cfg_keys)


class TestPaddingNoSpuriousBits:
    def test_partial_tile_padding(self, block_m_multi_token):
        # M = block_m + 1 → two tiles, second tile has 1 real token + padding
        # All tokens use adapter 1 except the single token in tile 2
        # which uses base (0). Padding should not introduce bits.
        adapter_indices = torch.cat(
            [
                torch.ones(block_m_multi_token, dtype=torch.long),  # tile 0: adapter 1
                torch.zeros(1, dtype=torch.long),  # tile 1: base only
            ]
        )

        # adapter 1→1, adapter 2→2
        remap_tables_t = _make_remap_tables_t([[0, 1, 2]])

        result = _compute_bitmasks_reference(adapter_indices, remap_tables_t)

        assert result.shape == (1, 2)
        assert result[0, 0].item() == 1 << 0  # tile 0: adapter 1
        assert result[0, 1].item() == 0  # tile 1: base + padding → 0
