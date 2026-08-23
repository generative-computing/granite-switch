# SPDX-License-Identifier: Apache-2.0
"""Isolated tensor-parallel correctness test for the fused switch-LoRA layer.

This test deliberately does **not** spin up the vLLM engine (no ``LLM``, no
model load, no process spawn, no NCCL). Instead it exercises the real
``SwitchedLoRALinear`` code path — weight slicing (``slice_lora_a_weight`` /
``slice_lora_b_weight``), ``finalize_weights`` (which builds ``w_ext`` via the
portable ``build_w_ext`` kernel helper), and ``forward`` (the single wide GEMM
plus the portable ``switch_lora_expand`` Triton kernel) — together with the real
``FusedLoRAKernelMeta`` bitmask/remap metadata.

It simulates each TP rank **sequentially on one GPU**, shards the weights exactly
the way vLLM would, runs each rank's ``forward``, then combines the rank outputs
by hand (concat over N for column-parallel, sum over partials for row-parallel)
and compares against the unsharded TP=1 reference. If the fused base+LoRA
sharding math is correct, TP>1 must reproduce TP=1 to within bf16 reduction
noise. A real sharding/permutation bug would produce a gross mismatch, not ~1e-2
noise.

The only vLLM dependencies are *symbols* that ``granite_switch.vllm.core.lora``
imports at module load (the distributed helpers and the linear-layer marker
classes used for ``isinstance``). Those are monkeypatched to lightweight fakes so
no distributed runtime is required.
"""

import pytest
import torch
from torch import nn

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused switch-LoRA kernels require CUDA"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16
RANK = 16  # single supported rank tier for all adapters
NA = 2  # number of adapters
M = 40  # tokens (spans >1 BLOCK_M=32 tile, with a partial tail)
K = 128  # input features (divisible by tp ∈ {2,4})
NSLICE = 128  # per-slice output features (divisible by 32*tp for tp ≤ 4)


# ---------------------------------------------------------------------------
# Fake vLLM base layers (marker classes for isinstance + a .weight/.bias holder)
# ---------------------------------------------------------------------------


class _FakeBase(nn.Module):
    def __init__(self, weight, bias=None, reduce_results=False, skip_bias_add=False):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = (
            nn.Parameter(bias, requires_grad=False) if bias is not None else None
        )
        self.reduce_results = reduce_results
        self.skip_bias_add = skip_bias_add


class _FakeColumn(_FakeBase):
    pass


class _FakeQKV(_FakeBase):
    pass


class _FakeRow(_FakeBase):
    pass


@pytest.fixture
def patched_lora(monkeypatch):
    """Patch the vLLM symbols that lora.py resolves at runtime.

    Returns the lora module so the test can flip the (world_size, rank) the
    layer reads in ``__init__`` before constructing each simulated rank.
    """
    import granite_switch.vllm.core.lora as lora_mod

    monkeypatch.setattr(lora_mod, "ColumnParallelLinear", _FakeColumn)
    monkeypatch.setattr(lora_mod, "MergedColumnParallelLinear", _FakeColumn)
    monkeypatch.setattr(lora_mod, "QKVParallelLinear", _FakeQKV)
    monkeypatch.setattr(lora_mod, "RowParallelLinear", _FakeRow)
    # all-reduce is never exercised here (reduce_results=False); patch to a
    # loud identity so an accidental call is obvious rather than hitting NCCL.
    monkeypatch.setattr(lora_mod, "tensor_model_parallel_all_reduce", lambda t: t)
    return lora_mod


def _set_tp(lora_mod, monkeypatch, world_size, rank):
    monkeypatch.setattr(
        lora_mod, "get_tensor_model_parallel_world_size", lambda: world_size
    )
    monkeypatch.setattr(lora_mod, "get_tensor_model_parallel_rank", lambda: rank)


# ---------------------------------------------------------------------------
# Sharding helpers (mirror vLLM's column-by-N / row-by-K conventions)
# ---------------------------------------------------------------------------


def _shard_dim0(t, rank, tp):
    """Shard a [D, ...] tensor along dim 0 into tp chunks, take `rank`."""
    sh = t.shape[0] // tp
    return t[rank * sh : (rank + 1) * sh]


def _shard_dim_last(t, rank, tp):
    """Shard a [..., D] tensor along the last dim into tp chunks, take `rank`."""
    sh = t.shape[-1] // tp
    return t[..., rank * sh : (rank + 1) * sh]


def _col_shard_concat(slices, rank, tp):
    """Column-parallel base shard: per-slice N-shard, concatenated."""
    return torch.cat([_shard_dim0(s, rank, tp) for s in slices], dim=0)


def _combine_column(outs, sharded_slice_sizes):
    """Reassemble column-parallel rank outputs [M, sum(Ns/tp)] back to full N.

    Each rank output is laid out [slice0_shard | slice1_shard | ...]; the full
    tensor is, per slice, the rank shards concatenated in rank order.
    """
    per_slice = [[] for _ in sharded_slice_sizes]
    for o in outs:
        off = 0
        for i, s in enumerate(sharded_slice_sizes):
            per_slice[i].append(o[:, off : off + s])
            off += s
    return torch.cat([torch.cat(ps, dim=1) for ps in per_slice], dim=1)


# ---------------------------------------------------------------------------
# Kernel-metadata context (real FusedLoRAKernelMeta, single module)
# ---------------------------------------------------------------------------


def _build_ctx(remap_table, adapter_indices):
    from granite_switch.vllm.core.lora_kernel_meta import (
        FusedLoRAKernelMeta,
        LoRAContext,
    )

    meta = FusedLoRAKernelMeta(torch.device(DEVICE))
    # [num_modules=1, NA+1]
    meta.register_remap_tables(remap_table.unsqueeze(0), [(K, NSLICE, NA)])
    ctx = LoRAContext()
    meta.prepare_and_store(adapter_indices, ctx)
    return ctx


def _wire(layer, ctx):
    layer._lora_ctx = ctx
    layer._module_idx = 0


# ---------------------------------------------------------------------------
# Per-case weight generation
# ---------------------------------------------------------------------------


def _make_case(parallelism):
    """Return (n_slices, slice_sizes, base_slices, biases, lora_A, lora_B, x_full).

    base_slices : list of [Ns, K] base weight slices (full, unsharded)
    biases      : list of [Ns] bias slices, or None (row-parallel: no bias)
    lora_A      : list (per slice) of [NA, 1, RANK, K]
    lora_B      : list (per slice) of [NA, 1, Ns, RANK]
    """
    torch.manual_seed(1234 + len(parallelism))
    n_slices = 3 if parallelism == "qkv" else 1

    def randn(*shape):
        return torch.randn(*shape, device=DEVICE, dtype=DTYPE) * 0.05

    base_slices = [randn(NSLICE, K) for _ in range(n_slices)]
    # Column/QKV bias is sharded on N and added once in each rank's own forward
    # (no reduce), so summing/concatenating shards reproduces the full bias — a
    # faithful comparison. This also exercises the finalize_weights bias path
    # (register_buffer slot). Row-parallel bias is deliberately None here: this
    # harness models the all-reduce as an external sum of per-rank partials, so
    # it cannot represent "bias added once, post-reduce"; that path is locked by
    # test_row_parallel_bias_added_after_reduce below.
    if parallelism == "row":
        biases = None
    else:
        biases = [randn(NSLICE) for _ in range(n_slices)]
    lora_A = [randn(NA, 1, RANK, K) for _ in range(n_slices)]
    lora_B = [randn(NA, 1, NSLICE, RANK) for _ in range(n_slices)]
    x_full = randn(M, K)
    return n_slices, base_slices, biases, lora_A, lora_B, x_full


def _build_layer(lora_mod, parallelism, n_slices, base_weight, bias):
    from granite_switch.vllm.core.lora import SwitchedLoRALinear

    if parallelism == "row":
        base = _FakeRow(base_weight, bias=bias, reduce_results=False)
    elif parallelism == "qkv":
        base = _FakeQKV(base_weight, bias=bias)
    else:
        base = _FakeColumn(base_weight, bias=bias)

    output_slices = tuple([NSLICE] * n_slices) if n_slices > 1 else None
    layer = SwitchedLoRALinear(
        base,
        num_adapters=NA,
        max_lora_rank=RANK,
        num_slices=n_slices,
        output_slices=output_slices,
    )
    return layer


def _load_lora(layer, n_slices, lora_A, lora_B):
    """Load FULL lora weights; the layer's weight_loaders shard as configured."""
    if n_slices == 1:
        layer.lora_A.weight_loader(layer.lora_A, lora_A[0])
        layer.lora_B.weight_loader(layer.lora_B, lora_B[0])
    else:
        for i in range(n_slices):
            layer.lora_A_slices[i].weight_loader(layer.lora_A_slices[i], lora_A[i])
            layer.lora_B_slices[i].weight_loader(layer.lora_B_slices[i], lora_B[i])


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@cuda_only
@pytest.mark.parametrize("parallelism", ["column", "qkv", "row"])
@pytest.mark.parametrize("tp", [2, 4])
def test_tp_combine_matches_unsharded(patched_lora, monkeypatch, parallelism, tp):
    lora_mod = patched_lora
    torch.manual_seed(0)

    n_slices, base_slices, biases, lora_A, lora_B, x_full = _make_case(parallelism)
    adapter_ranks = [RANK] * NA

    # Adapter index assignment per token: base / adapter1 / adapter2 / base.
    # Replicated across all TP ranks (as it is in production).
    adapter_indices = torch.zeros(M, dtype=torch.long, device=DEVICE)
    adapter_indices[10:20] = 1
    adapter_indices[20:30] = 2

    # ---- TP=1 reference --------------------------------------------------
    _set_tp(lora_mod, monkeypatch, 1, 0)
    ref_base_weight = torch.cat(base_slices, dim=0)  # [N_total, K]
    ref_bias = torch.cat(biases, dim=0) if biases is not None else None
    ref_layer = _build_layer(lora_mod, parallelism, n_slices, ref_base_weight, ref_bias)
    _load_lora(ref_layer, n_slices, lora_A, lora_B)
    ref_layer.finalize_weights(adapter_ranks)
    ref_ctx = _build_ctx(ref_layer.remap_table, adapter_indices)
    _wire(ref_layer, ref_ctx)
    with torch.no_grad():
        ref_out, _ = ref_layer.forward(x_full)
    ref_out = ref_out.float()

    # ---- TP>1 simulated ranks -------------------------------------------
    rank_outs = []
    remap_tables = []
    for r in range(tp):
        _set_tp(lora_mod, monkeypatch, tp, r)

        if parallelism == "row":
            base_weight = _shard_dim_last(ref_base_weight, r, tp)  # shard K
            x_in = _shard_dim_last(x_full, r, tp)  # shard K
            bias = None
        else:
            base_weight = _col_shard_concat(base_slices, r, tp)  # shard N per slice
            x_in = x_full
            bias = (
                _col_shard_concat([b.unsqueeze(1) for b in biases], r, tp).squeeze(1)
                if biases is not None
                else None
            )

        layer = _build_layer(lora_mod, parallelism, n_slices, base_weight, bias)
        _load_lora(layer, n_slices, lora_A, lora_B)
        layer.finalize_weights(adapter_ranks)
        remap_tables.append(layer.remap_table.clone())

        # remap_table / bitmask are functions of adapter_indices + per-module
        # zero-detection only; for dense adapters they are identical across
        # ranks, so one shared ctx is correct. Lock that invariant.
        ctx = _build_ctx(layer.remap_table, adapter_indices)
        _wire(layer, ctx)
        with torch.no_grad():
            out, _ = layer.forward(x_in)
        rank_outs.append(out.float().clone())

    # Invariant: every rank derived the same remap_table.
    for r in range(1, tp):
        assert torch.equal(remap_tables[0], remap_tables[r]), (
            f"remap_table diverged between rank 0 and rank {r}: "
            f"{remap_tables[0].tolist()} vs {remap_tables[r].tolist()}"
        )

    # ---- Combine ---------------------------------------------------------
    if parallelism == "row":
        combined = torch.stack(rank_outs, dim=0).sum(dim=0)  # sum partials
    else:
        sharded_slice_sizes = tuple([NSLICE // tp] * n_slices)
        combined = _combine_column(rank_outs, sharded_slice_sizes)

    assert combined.shape == ref_out.shape, (
        f"shape mismatch: combined {combined.shape} vs ref {ref_out.shape}"
    )

    max_abs = (combined - ref_out).abs().max().item()
    ref_scale = ref_out.abs().max().item()
    print(
        f"\n[{parallelism} tp={tp}] max_abs_diff={max_abs:.4e} "
        f"ref_scale={ref_scale:.4e} rel={max_abs / max(ref_scale, 1e-9):.4e}"
    )

    # bf16 reduction noise across different K/N splits is ~1e-2 relative; a real
    # sharding/permutation bug is gross. Tolerance catches the latter decisively.
    torch.testing.assert_close(combined, ref_out, rtol=3e-2, atol=3e-2)


@cuda_only
def test_row_parallel_bias_added_after_reduce(patched_lora, monkeypatch):
    """Row-parallel bias must be applied strictly AFTER the all-reduce, once.

    A row-parallel rank holds only a partial sum; bias belongs to the full
    (reduced) output. If bias were folded into the partial before the reduce
    (the original bug), an N-way all-reduce would sum it N times. This test
    spies on the all-reduce input: the bias must NOT be present there, and must
    appear in the output exactly once on top of the reduce result.

    Uses a real reduce_results=True row layer (which also exercises the
    finalize_weights bias path) with the all-reduce patched to identity, so the
    spy sees exactly what production would hand to NCCL.
    """
    lora_mod = patched_lora
    torch.manual_seed(7)

    captured = {}

    def spy_all_reduce(t):
        captured["reduce_in"] = t.clone()
        return t

    monkeypatch.setattr(lora_mod, "tensor_model_parallel_all_reduce", spy_all_reduce)
    # tp_size > 1 + RowParallel + reduce_results=True => _row_parallel_reduce path.
    _set_tp(lora_mod, monkeypatch, 2, 0)

    base_weight = (
        torch.randn(NSLICE, K // 2, device=DEVICE, dtype=DTYPE) * 0.05
    )  # K sharded
    bias = torch.randn(NSLICE, device=DEVICE, dtype=DTYPE)
    from granite_switch.vllm.core.lora import SwitchedLoRALinear

    base = _FakeRow(base_weight, bias=bias, reduce_results=True)
    layer = SwitchedLoRALinear(base, num_adapters=NA, max_lora_rank=RANK, num_slices=1)
    lora_A = [torch.randn(NA, 1, RANK, K, device=DEVICE, dtype=DTYPE) * 0.05]
    lora_B = [torch.randn(NA, 1, NSLICE, RANK, device=DEVICE, dtype=DTYPE) * 0.05]
    _load_lora(layer, 1, lora_A, lora_B)
    layer.finalize_weights([RANK] * NA)  # must not raise (register_buffer slot)

    adapter_indices = torch.zeros(M, dtype=torch.long, device=DEVICE)
    adapter_indices[10:20] = 1
    ctx = _build_ctx(layer.remap_table, adapter_indices)
    _wire(layer, ctx)

    x_in = torch.randn(M, K // 2, device=DEVICE, dtype=DTYPE) * 0.05
    with torch.no_grad():
        out, _ = layer.forward(x_in)

    assert "reduce_in" in captured, "all-reduce was not called on the reduce path"
    # out == reduce_in + bias  =>  out - reduce_in == bias (broadcast over tokens).
    # If bias were folded pre-reduce, reduce_in would already contain it and this
    # difference would be ~0, not the bias.
    delta = out.float() - captured["reduce_in"].float()
    torch.testing.assert_close(
        delta,
        bias.float().expand_as(delta),
        rtol=1e-2,
        atol=1e-2,
    )
