# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the PRODUCTION bitmask path.

Production computes per-module bitmasks with the Triton ``_bitmask_reduce_kernel``
(via ``_compute_bitmasks_triton``, registered as the ``compute_per_module_bitmasks``
custom op) and runs it inside the ``@support_torch_compile`` forward. The
pure-torch ``_compute_bitmasks_reference`` is only a readable spec.

The unit suite (tests/unit/test_bitmask_computation.py) exercises the *reference*
on the rich remap cases. This suite locks the *production* path against it:

  triton kernel  ==  custom op  ==  reference  ==  hand-computed ground truth

over multi-module remap with non-applicable compaction, rank-tier reordering,
multiple adapters/tiles, partial-tile padding and all-base tiles — the cases the
production kernel had never been tested on — plus the centralized facility
(``register_remap_tables`` + ``prepare_and_store``) and the op **under
torch.compile** (the boundary the custom-op registration exists to protect).
"""

import pytest
import torch

import granite_switch.vllm.core.lora_kernel_meta as meta_mod
from granite_switch.vllm.core.lora_kernel_meta import (
    FusedLoRAKernelMeta,
    LoRAContext,
    _compute_bitmasks_reference,
    _compute_bitmasks_triton,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton bitmask kernel requires CUDA"
)

NA = 4  # global adapters

# Three modules with deliberately different remap tables (global adapter id →
# kernel-local position; 0 = non-applicable for that module):
#   mod0: all applicable, identity            [_, 1, 2, 3, 4]
#   mod1: adapters 2 & 4 non-applicable,      [_, 1, 0, 2, 0]  (compaction)
#         applicable 1,3 compacted to 1,2
#   mod2: reordered local positions + one     [_, 3, 1, 2, 0]  (reorder + drop)
#         non-applicable (adapter 4)
REMAP_TABLES = [
    [0, 1, 2, 3, 4],
    [0, 1, 0, 2, 0],
    [0, 3, 1, 2, 0],
]

# Structured 64-token pattern: 8 segments of 8 tokens. At BLOCK_M=8 each segment
# is its own tile (all-base, single-adapter, all-four, sparse, boundary-only);
# other BLOCK_M values straddle these in every combination.
ADAPTER_INDICES = (
    [0] * 8  # all base
    + [1] * 8  # adapter 1 only
    + [2] * 8  # adapter 2 only
    + [3] * 8  # adapter 3 only
    + [4] * 8  # adapter 4 only
    + [1, 2, 3, 4, 1, 2, 3, 4]  # all four
    + [0, 1, 0, 2, 0, 3, 0, 4]  # base interleaved with all four
    + [2, 0, 0, 0, 0, 0, 0, 3]  # only the tile's boundary tokens active
)


@pytest.fixture(autouse=True)
def _register_op():
    """Ensure the compute_per_module_bitmasks custom op is registered."""
    FusedLoRAKernelMeta(torch.device("cuda"))


@pytest.fixture(params=[1, 8, 16, 32, 64])
def block_m(request, monkeypatch):
    # Both the triton launcher and the reference read meta_mod.BLOCK_M; patch it
    # so the equivalence holds across tilings, not just the production 32.
    monkeypatch.setattr(meta_mod, "BLOCK_M", request.param)
    return request.param


def _ground_truth(adapter_indices, remap_tables, block_m):
    """Pure-python per-module bitmasks: [num_modules, num_tiles] int64."""
    M = len(adapter_indices)
    num_tiles = (M + block_m - 1) // block_m
    rows = []
    for remap in remap_tables:
        row = []
        for t in range(num_tiles):
            bm = 0
            for tok in adapter_indices[t * block_m : (t + 1) * block_m]:
                loc = remap[tok]
                if loc > 0:
                    bm |= 1 << (loc - 1)
            row.append(bm)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _tensors():
    ai = torch.tensor(ADAPTER_INDICES, dtype=torch.long, device="cuda")
    remap_t = torch.tensor(REMAP_TABLES, dtype=torch.long).T.contiguous().cuda()
    return ai, remap_t


@cuda_only
def test_triton_matches_reference_and_truth(block_m):
    ai, remap_t = _tensors()
    truth = _ground_truth(ADAPTER_INDICES, REMAP_TABLES, block_m)

    triton_out = _compute_bitmasks_triton(ai, remap_t).cpu()
    ref_out = _compute_bitmasks_reference(ai, remap_t).cpu()

    assert triton_out.shape == truth.shape, (
        f"shape: triton {tuple(triton_out.shape)} vs truth {tuple(truth.shape)}"
    )
    assert torch.equal(triton_out, truth), (
        f"triton != ground truth at BLOCK_M={block_m}\n"
        f"triton:\n{triton_out.tolist()}\ntruth:\n{truth.tolist()}"
    )
    assert torch.equal(ref_out, truth), "reference != ground truth"
    assert torch.equal(triton_out, ref_out), "triton != reference"


@cuda_only
def test_custom_op_matches_truth(block_m):
    """The registered production op (triton body) == ground truth."""
    ai, remap_t = _tensors()
    truth = _ground_truth(ADAPTER_INDICES, REMAP_TABLES, block_m)
    op_out = torch.ops.granite_switch.compute_per_module_bitmasks(ai, remap_t).cpu()
    assert torch.equal(op_out, truth), (
        f"custom op != ground truth at BLOCK_M={block_m}\n"
        f"op:\n{op_out.tolist()}\ntruth:\n{truth.tolist()}"
    )


@cuda_only
def test_facility_end_to_end():
    """register_remap_tables + prepare_and_store produce correct per-module
    bitmasks AND remapped (kernel-local) indices at the production BLOCK_M=32."""
    assert meta_mod.BLOCK_M == 32, "production tiling expected"
    ai, _ = _tensors()
    all_remap = torch.tensor(REMAP_TABLES, dtype=torch.long).cuda()  # [modules, NA+1]

    meta = FusedLoRAKernelMeta(torch.device("cuda"))
    meta.register_remap_tables(all_remap, [(128, 128, NA)] * len(REMAP_TABLES))
    ctx = LoRAContext()
    meta.prepare_and_store(ai, ctx)

    truth = _ground_truth(ADAPTER_INDICES, REMAP_TABLES, 32)
    assert torch.equal(ctx.per_module_bitmasks.cpu(), truth)

    # remapped_indices[m, tok] == REMAP_TABLES[m][adapter_index[tok]]
    remapped_truth = torch.tensor(
        [
            [REMAP_TABLES[m][idx] for idx in ADAPTER_INDICES]
            for m in range(len(REMAP_TABLES))
        ],
        dtype=torch.long,
    )
    assert torch.equal(ctx.remapped_indices.cpu(), remapped_truth)
    assert torch.equal(ctx.adapter_indices.cpu(), torch.tensor(ADAPTER_INDICES))


@cuda_only
def test_custom_op_under_torch_compile():
    """The op must stay correct when called inside a compiled region — this is
    exactly the @support_torch_compile path the custom-op boundary protects."""
    ai, remap_t = _tensors()
    truth = _ground_truth(ADAPTER_INDICES, REMAP_TABLES, meta_mod.BLOCK_M).cuda()

    def fn(a, r):
        return torch.ops.granite_switch.compute_per_module_bitmasks(a, r)

    eager = fn(ai, remap_t)
    compiled = torch.compile(fn, fullgraph=True)(ai, remap_t)

    assert torch.equal(compiled, eager), "compiled op != eager op"
    assert torch.equal(compiled, truth), "compiled op != ground truth"


@cuda_only
def test_random_equivalence(block_m):
    """Randomized triton == reference sweep over fuzzed remap tables.

    Closes the structured test's gaps: M is random (mostly NOT divisible by
    BLOCK_M → exercises the partial-tile tail, where triton masks with
    ``offsets < M`` and the reference pads-and-reshapes), with random NA up to
    the 64-adapter limit, random num_modules, and arbitrary remap tables.
    Compares the two int64 implementations directly (no python ground truth, so
    high-bit wrap stays consistent). Seeds are deterministic for reproducibility.
    """
    TRIALS = 200
    ragged_seen = 0
    highbit_seen = 0
    for trial in range(TRIALS):
        torch.manual_seed(7919 * block_m + trial)
        NA = int(torch.randint(1, 65, (1,)).item())
        num_modules = int(torch.randint(1, 7, (1,)).item())
        M = int(torch.randint(1, 301, (1,)).item())

        ai = torch.randint(0, NA + 1, (M,), dtype=torch.long, device="cuda")
        # kernel-local positions in [0, NA] (0 = non-applicable); base slot is 0.
        remap = torch.randint(0, NA + 1, (num_modules, NA + 1), dtype=torch.long)
        remap[:, 0] = 0
        remap_t = remap.T.contiguous().cuda()

        triton_out = _compute_bitmasks_triton(ai, remap_t)
        ref_out = _compute_bitmasks_reference(ai, remap_t)
        if not torch.equal(triton_out, ref_out):
            mism = (triton_out != ref_out).nonzero()[0]
            raise AssertionError(
                f"triton != reference\n  trial={trial} seed={7919 * block_m + trial} "
                f"block_m={block_m} NA={NA} num_modules={num_modules} M={M}\n"
                f"  first mismatch [mod,tile]={mism.tolist()} "
                f"triton={triton_out[tuple(mism)].item()} ref={ref_out[tuple(mism)].item()}"
            )

        if M % block_m != 0:
            ragged_seen += 1
        if remap.max().item() >= 33:
            highbit_seen += 1

    print(
        f"\n[block_m={block_m}] trials={TRIALS} "
        f"ragged_M={ragged_seen} high_local_pos(>=33)={highbit_seen}"
    )
    if block_m > 1:
        assert ragged_seen > 0, "expected ragged-M (non-tile-aligned) trials"


@cuda_only
def test_high_bit_positions(block_m):
    """Kernel-local positions up to the 64-adapter limit, deterministically.

    Position 64 → bit 63 (the int64 sign bit); reference and triton must wrap
    identically. M=130 is ragged for every BLOCK_M>1.
    """
    NA = 64
    # Single module, adapter j → local position j (so positions span 1..64).
    remap = torch.arange(0, NA + 1, dtype=torch.long).unsqueeze(0)  # [1, NA+1]
    remap_t = remap.T.contiguous().cuda()

    M = 130
    torch.manual_seed(12345)
    ai = torch.randint(0, NA + 1, (M,), dtype=torch.long, device="cuda")
    # Force the extreme positions to appear (incl. 64 → sign bit).
    ai[0], ai[1], ai[2], ai[3] = 1, 32, 63, 64

    triton_out = _compute_bitmasks_triton(ai, remap_t)
    ref_out = _compute_bitmasks_reference(ai, remap_t)
    assert torch.equal(triton_out, ref_out), (
        f"triton != reference at high local positions (block_m={block_m})\n"
        f"triton:\n{triton_out.cpu().tolist()}\nref:\n{ref_out.cpu().tolist()}"
    )
    # The position-64 token sets bit 63 → that tile's mask is a negative int64.
    assert (triton_out < 0).any(), "expected the sign bit (position 64) to be set"
