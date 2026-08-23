# SPDX-License-Identifier: Apache-2.0
"""SWITCH — the fused switch-LoRA kernel backend (backend-agnostic Triton).

SWITCH is the fused LoRA kernel backend for ``SwitchedLoRALinear``. The name is
the design:

    SWITCH = Switched-delta · W-ext-fused · In-place · Tiled
             · Conditional-skip · Heterogeneous-rank

  * Switched-delta     — each token selects exactly one adapter's lora_B delta.
  * W-ext-fused        — the base projection and every adapter's lora_A (shrink)
                         are fused into one wide GEMM (x @ w_ext.T): the static,
                         shared substrate, NOT the switched part.
  * In-place           — the switched delta is accumulated in place into the base
                         columns of x_ext (base and shrink columns are disjoint,
                         so the read-modify-write is hazard-free).
  * Tiled              — block-tiled expand (BLOCK_M tokens × BLOCK_N cols per CTA).
  * Conditional-skip   — per-tile bitmask early-exit when no adapter is present.
  * Heterogeneous-rank — per-adapter / per-module rank tiers (16…512).

(Granite Switch is the model/system; SWITCH here is the kernel backend.)

This module is intentionally free of any vLLM or ``torch.library`` dependency
so the kernel can be reused by other inference backends. It exposes:

build_w_ext(W, lora_A_by_rank)
    Build the extended weight matrix once at load time: the base weight rows
    followed by every adapter's ``lora_A`` (shrink) rows, grouped by rank tier.
    A single GEMM ``x @ w_ext.T`` then produces the base projection in the
    leading ``N_total`` columns and all per-adapter shrink vectors after it.

switch_lora_expand(x_ext, adapter_indices, bitmask, lb_packed, tile_slice,
                   slice_col_r, na, S, block_n, N)
    Launch the expand kernel: read each active adapter's shrink columns from
    ``x_ext``, multiply by its packed ``lora_B``, and accumulate the delta in
    place into the base columns ``x_ext[:, :N]``. Base/shrink columns are
    disjoint, so the in-place accumulate is hazard-free.

get_switch_lora_expand_config()
    Triton launch parameters (tiling + warps/stages), in the style of vLLM's
    ``get_lora_op_configs`` — a single source of truth returning a dict.

Supported rank tiers: 16, 32, 64, 128, 256, 512. Tiers absent from a
deployment compile to dead code (``tl.static_range(0)`` emits no instructions).

Adapter index convention
------------------------
The ``adapter_indices`` this kernel reads are **per-module, kernel-local**
indices, not global adapter ids. The caller remaps global ids to each module's
local numbering centrally (per-module ``remap_table``; see
``granite_switch.vllm.core.lora_kernel_meta``) before launch: a global adapter
that does not apply to this module maps to 0, and the applicable ones are
renumbered 1.. within the module. So each module independently numbers only the
adapters that touch it, grouped by ascending rank tier:
    0                            base (no LoRA), or adapter not applicable here
    1 .. n_16                    this module's tier-16 adapters
    n_16+1 .. n_16+n_32          this module's tier-32 adapters
    ...
where (n_16, n_32, ..., n_512) are this module's local per-tier counts. Nothing
requires an adapter to be present in every module or to use the same rank across
modules — the per-module remap is what lets ranks and adapter membership vary.

Block sizes
-----------
BLOCK_M and BLOCK_N are the defaults returned by
``get_switch_lora_expand_config`` (currently 32x32), not immutable — the launcher
takes its tiling from that config rather than hardcoding it. The kernel does NOT
use ``@triton.autotune`` (incompatible with torch.compile). BLOCK_M must match
the bitmask tiling in ``granite_switch.vllm.core.lora_kernel_meta`` (both import
the same constant). ``block_n`` is a per-launch argument bound at finalize time
(it determines the precomputed tile/slice tables); it defaults to BLOCK_N but may
differ, and must divide every output slice so no tile straddles a slice boundary.
"""

import torch
import triton
import triton.language as tl

SUPPORTED_RANKS = (16, 32, 64, 128, 256, 512)


def promote_rank(rank: int) -> int:
    """Round a LoRA rank up to the next rank the kernel supports.

    The tier machinery is fixed at six tiers (``slice_col_r`` is built as
    ``[S, 6]`` and ``_na`` as a 6-tuple), so an off-tier rank cannot simply be
    added to SUPPORTED_RANKS. Callers instead promote the rank and zero-pad the
    adapter's lora_A rows / lora_B columns up to it, which is numerically exact:
    the padded rows contribute nothing to the shrink and their lora_B columns
    contribute nothing to the expand.

    Rank 0 (or negative, meaning "not applicable") is returned unchanged.
    """
    if rank <= 0:
        return rank
    for supported in SUPPORTED_RANKS:
        if supported >= rank:
            return supported
    raise ValueError(
        f"LoRA rank {rank} exceeds the largest supported rank {SUPPORTED_RANKS[-1]}."
    )


# Default tiling/launch constants (returned by get_switch_lora_expand_config).
# Validated across 3B/8B prefill+decode in the block-tuning study; no usable
# gain was found from other values.
BLOCK_M: int = 32
BLOCK_N: int = 32
NUM_WARPS: int = 4
NUM_STAGES: int = 1


def get_switch_lora_expand_config() -> dict:
    """Return the expand kernel's tiling/launch parameters.

    Mirrors vLLM's ``get_lora_op_configs`` convention: one place returns the
    kernel config as a dict, and the launcher passes the values as
    ``tl.constexpr``. Currently fixed defaults; a shape-keyed tuned-config
    lookup can be added here later without touching call sites.
    """
    return {
        "block_m": BLOCK_M,
        "block_n": BLOCK_N,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
    }


# ---------------------------------------------------------------------------
# Load-time: build the extended weight matrix
# ---------------------------------------------------------------------------


def build_w_ext(
    W: torch.Tensor,
    lora_A_by_rank: dict,
) -> torch.Tensor:
    """Build W_ext by stacking W with lora_A rows, ordered by ascending rank.

    Parameters
    ----------
    W              : Tensor [N_total, K]
    lora_A_by_rank : dict {rank: Tensor [n_r, S, rank, K]}
                     S = number of slices (1 for single-slice layers).
                     Rows are appended in tier -> adapter -> slice order:
                     for each tier r, for each adapter a, S*r rows:
                       lora_A[a, 0, :, :]  (slice 0)
                       lora_A[a, 1, :, :]  (slice 1)
                       ...
                     Only ranks in SUPPORTED_RANKS are accepted.

    Returns
    -------
    w_ext : Tensor [N_total + sum_r(n_r * S * r), K]
    """
    parts = [W]
    for r in SUPPORTED_RANKS:
        lA = lora_A_by_rank.get(r)
        if lA is not None and lA.shape[0] > 0:
            n_r, S = lA.shape[0], lA.shape[1]
            # [n_r, S, r, K] -> [n_r * S * r, K] in adapter->slice->row order
            parts.append(lA.reshape(n_r * S * r, lA.shape[3]))
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Expand kernel
# ---------------------------------------------------------------------------


@triton.jit
def _switch_lora_expand_kernel(
    # x_ext  [M, N_total + sum_r(n_r * S * r)] — base output is cols [0, N).
    # Single buffer: shrink cols (>= N) are read; base cols [0, N) are
    # accumulated into IN PLACE. Folding the output into x_ext (disjoint cols,
    # hazard-free) keeps the inductor graph glue-free (no clone + slice_scatter)
    # and saves a pointer arg per launch.
    XExt,
    stride_xe_m,
    stride_xe_n,
    # adapter_indices  [M] — kernel-local (post-remap) ids: 0 = base, 1.. = this
    # module's adapters in ascending-rank order.
    AdapIdx,
    # bitmask  [num_tiles_M] int64 — bit a set iff kernel-local adapter (a+1)
    # is present in the tile (computed from the same post-remap ids).
    Bitmask,
    stride_bm_t,
    # SINGLE packed lora_B: contiguous concat over present tiers of
    # [NA_r, N_total, r] (row-major). Tier r block has strides (N*r, r, 1); its
    # base element offset is cumsum_{r'<r}(NA_r' * N * r'), computed inline below.
    LBPACK,
    M,
    N,
    # per-tile slice lookup
    TileSlice,
    SliceColR,
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Invariant (by design, not checked here): M — the token/row count of x_ext
    # that sets this grid's pid_m range — equals the token count used to build
    # AdapIdx and Bitmask. Granite Switch routes per token: adapter_indices is
    # sized 1:1 to the input tokens, and the decoder preserves that count, so the
    # kernel metadata and the activations always share the same M. The Bitmask
    # load below and the AdapIdx load are therefore unmasked against their own
    # length; if a future change broke the per-token contract (M_fwd > M_prepare),
    # these would read out of bounds.

    # One bit per adapter for this row-tile; bit a set iff adapter (a+1) is
    # present somewhere in the tile. All-zero tile touches no adapter -> skip.
    bitmask = tl.load(Bitmask + pid_m * stride_bm_t).to(tl.int64)
    if bitmask == 0:
        return

    pid_n = tl.program_id(1)
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_range < M
    mask_n = n_range < N

    # This output tile lives in exactly one slice (block_n divides every slice).
    # SliceColR[s] gives, per tier, the first shrink column of that slice's
    # adapters in x_ext — the per-adapter shrink block starts at col_<r> + a*S*r.
    s_idx = tl.load(TileSlice + pid_n).to(tl.int32)
    col_16 = tl.load(SliceColR + s_idx * 6 + 0).to(tl.int32)
    col_32 = tl.load(SliceColR + s_idx * 6 + 1).to(tl.int32)
    col_64 = tl.load(SliceColR + s_idx * 6 + 2).to(tl.int32)
    col_128 = tl.load(SliceColR + s_idx * 6 + 3).to(tl.int32)
    col_256 = tl.load(SliceColR + s_idx * 6 + 4).to(tl.int32)
    col_512 = tl.load(SliceColR + s_idx * 6 + 5).to(tl.int32)

    adap_ids = tl.load(AdapIdx + m_range, mask=mask_m, other=0).to(tl.int32)
    delta = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row_active = adap_ids != 0

    # Kernel-local ids are numbered 1.. across tiers in ascending-rank order, so
    # each tier's ids start after all lower tiers. These offsets convert a
    # tier-local index a to the kernel-local id (and its bit in this module's
    # bitmask).
    id_off_32 = NA_16
    id_off_64 = NA_16 + NA_32
    id_off_128 = NA_16 + NA_32 + NA_64
    id_off_256 = NA_16 + NA_32 + NA_64 + NA_128
    id_off_512 = NA_16 + NA_32 + NA_64 + NA_128 + NA_256

    # Element offset of each tier's block inside the single packed lora_B buffer
    # (tiers concatenated in rank order): cumsum(NA_r * N * r).
    lb_off_16 = 0
    lb_off_32 = NA_16 * N * 16
    lb_off_64 = lb_off_32 + NA_32 * N * 32
    lb_off_128 = lb_off_64 + NA_64 * N * 64
    lb_off_256 = lb_off_128 + NA_128 * N * 128
    lb_off_512 = lb_off_256 + NA_256 * N * 256

    # Per tier, loop over that tier's adapters; the bit test skips adapters not
    # present in this tile. For each active adapter: load its r shrink values
    # from x_ext, load its lora_B block, and accumulate shrink @ lora_B into the
    # output delta. The static_range unrolls at compile time, so absent tiers
    # (NA_r == 0) emit no code. Only the tier-16 loop is annotated; tiers 32..512
    # are the identical pattern with r and the tier offsets substituted.
    r16 = tl.arange(0, 16)
    for a in tl.static_range(NA_16):
        if (bitmask >> a) & 1:  # adapter (a+1) in this tile?
            mask_a = (adap_ids == (a + 1)) & mask_m  # rows that select it
            # shrink: this adapter's r shrink columns for the tile's rows.
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_16 + a * S * 16 + r16[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            # lb: this adapter's lora_B tile [r, BLOCK_N] (rank x out-col);
            # rank is contiguous, each out-col steps by r (layout above).
            lb = tl.load(
                LBPACK
                + lb_off_16
                + a * (N * 16)
                + r16[:, None]
                + n_range[None, :] * 16,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r32 = tl.arange(0, 32)
    for a in tl.static_range(NA_32):
        if (bitmask >> (id_off_32 + a)) & 1:
            mask_a = (adap_ids == (id_off_32 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_32 + a * S * 32 + r32[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_32
                + a * (N * 32)
                + r32[:, None]
                + n_range[None, :] * 32,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r64 = tl.arange(0, 64)
    for a in tl.static_range(NA_64):
        if (bitmask >> (id_off_64 + a)) & 1:
            mask_a = (adap_ids == (id_off_64 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_64 + a * S * 64 + r64[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_64
                + a * (N * 64)
                + r64[:, None]
                + n_range[None, :] * 64,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r128 = tl.arange(0, 128)
    for a in tl.static_range(NA_128):
        if (bitmask >> (id_off_128 + a)) & 1:
            mask_a = (adap_ids == (id_off_128 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_128 + a * S * 128 + r128[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_128
                + a * (N * 128)
                + r128[:, None]
                + n_range[None, :] * 128,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r256 = tl.arange(0, 256)
    for a in tl.static_range(NA_256):
        if (bitmask >> (id_off_256 + a)) & 1:
            mask_a = (adap_ids == (id_off_256 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_256 + a * S * 256 + r256[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_256
                + a * (N * 256)
                + r256[:, None]
                + n_range[None, :] * 256,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r512 = tl.arange(0, 512)
    for a in tl.static_range(NA_512):
        if (bitmask >> (id_off_512 + a)) & 1:
            mask_a = (adap_ids == (id_off_512 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_512 + a * S * 512 + r512[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_512
                + a * (N * 512)
                + r512[:, None]
                + n_range[None, :] * 512,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    # Fold the delta back into the base columns [0, N) of x_ext in place. Only
    # adapter rows are written (row_active); base-only rows already hold the
    # correct projection. Base and shrink columns are disjoint, so this is safe.
    out_ptrs = XExt + m_range[:, None] * stride_xe_m + n_range[None, :] * stride_xe_n
    store_mask = row_active[:, None] & mask_n[None, :]
    existing = tl.load(out_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    tl.store(out_ptrs, (existing + delta).to(tl.bfloat16), mask=store_mask)


def switch_lora_expand(
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    na: tuple,
    S: int,
    block_n: int,
    N: int,
) -> None:
    """Accumulate the LoRA expand delta in place into ``x_ext[:, :N]``.

    Pure-Triton launcher with no torch.library/vLLM dependency, so other
    backends can call it directly. ``na`` is the 6-tuple of per-tier adapter
    counts (n_16, n_32, n_64, n_128, n_256, n_512).
    """
    M = x_ext.shape[0]
    cfg = get_switch_lora_expand_config()
    grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(N, block_n))
    _switch_lora_expand_kernel[grid](
        x_ext,
        x_ext.stride(0),
        x_ext.stride(1),
        adapter_indices,
        bitmask,
        bitmask.stride(0),
        lb_packed,
        M,
        N,
        tile_slice,
        slice_col_r,
        NA_16=na[0],
        NA_32=na[1],
        NA_64=na[2],
        NA_128=na[3],
        NA_256=na[4],
        NA_512=na[5],
        S=S,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=block_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


# ---------------------------------------------------------------------------
# Fused gate/up expand + SwiGLU (shared-MLP first projection ONLY)
# ---------------------------------------------------------------------------
#
# The merged gate/up projection is the one module whose output is consumed by a
# packed-layout kernel (vLLM SiluAndMul). Rather than materialize the corrected
# [M, 2H] gate|up to HBM and re-read it in a separate activation kernel (and pay
# a .contiguous() because the base slice of x_ext is strided), this kernel does
# the whole epilogue in one pass: apply the LoRA delta to the gate (slice 0) and
# up (slice 1) columns, then write silu(gate)*up directly as a CONTIGUOUS [M, H].
#
# It reads x_ext (gate, up, shrink) by explicit stride, so the strided base is a
# non-issue; nothing downstream ever sees it. The bitmask gates only the LoRA
# delta work — the silu(gate)*up store ALWAYS runs (base-only tiles still need
# their activation written).


@triton.jit
def _accum_slice_delta(
    XExt,
    stride_xe_m,
    stride_xe_n,
    LBPACK,
    m_range,
    mask_m,
    adap_ids,
    n_range,
    mask_n,
    bitmask,
    col_16,
    col_32,
    col_64,
    col_128,
    col_256,
    col_512,
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """LoRA delta [BLOCK_M, BLOCK_N] for ONE output slice.

    Same tier structure as the in-place expand kernel, parameterized by the
    slice's shrink-column bases (col_*) and the output-column range (n_range),
    so it is called once for the gate slice and once for the up slice. Caller
    guarantees bitmask != 0.
    """
    delta = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tier-local index -> kernel-local id / bit position (see expand kernel).
    id_off_32 = NA_16
    id_off_64 = NA_16 + NA_32
    id_off_128 = NA_16 + NA_32 + NA_64
    id_off_256 = NA_16 + NA_32 + NA_64 + NA_128
    id_off_512 = NA_16 + NA_32 + NA_64 + NA_128 + NA_256

    # Element offset of each tier's block in the packed lora_B buffer.
    lb_off_16 = 0
    lb_off_32 = NA_16 * N * 16
    lb_off_64 = lb_off_32 + NA_32 * N * 32
    lb_off_128 = lb_off_64 + NA_64 * N * 64
    lb_off_256 = lb_off_128 + NA_128 * N * 128
    lb_off_512 = lb_off_256 + NA_256 * N * 256

    # Same per-tier accumulation as the in-place expand kernel; see there for the
    # annotated tier-16 loop. Tiers 32..512 repeat the pattern.
    r16 = tl.arange(0, 16)
    for a in tl.static_range(NA_16):
        if (bitmask >> a) & 1:
            mask_a = (adap_ids == (a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_16 + a * S * 16 + r16[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_16
                + a * (N * 16)
                + r16[:, None]
                + n_range[None, :] * 16,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r32 = tl.arange(0, 32)
    for a in tl.static_range(NA_32):
        if (bitmask >> (id_off_32 + a)) & 1:
            mask_a = (adap_ids == (id_off_32 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_32 + a * S * 32 + r32[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_32
                + a * (N * 32)
                + r32[:, None]
                + n_range[None, :] * 32,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r64 = tl.arange(0, 64)
    for a in tl.static_range(NA_64):
        if (bitmask >> (id_off_64 + a)) & 1:
            mask_a = (adap_ids == (id_off_64 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_64 + a * S * 64 + r64[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_64
                + a * (N * 64)
                + r64[:, None]
                + n_range[None, :] * 64,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r128 = tl.arange(0, 128)
    for a in tl.static_range(NA_128):
        if (bitmask >> (id_off_128 + a)) & 1:
            mask_a = (adap_ids == (id_off_128 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_128 + a * S * 128 + r128[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_128
                + a * (N * 128)
                + r128[:, None]
                + n_range[None, :] * 128,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r256 = tl.arange(0, 256)
    for a in tl.static_range(NA_256):
        if (bitmask >> (id_off_256 + a)) & 1:
            mask_a = (adap_ids == (id_off_256 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_256 + a * S * 256 + r256[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_256
                + a * (N * 256)
                + r256[:, None]
                + n_range[None, :] * 256,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r512 = tl.arange(0, 512)
    for a in tl.static_range(NA_512):
        if (bitmask >> (id_off_512 + a)) & 1:
            mask_a = (adap_ids == (id_off_512 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_512 + a * S * 512 + r512[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_512
                + a * (N * 512)
                + r512[:, None]
                + n_range[None, :] * 512,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    return delta


@triton.jit
def _switch_lora_expand_swiglu_kernel(
    # x_ext  [M, 2H + shrink]: gate base = cols [0,H), up base = cols [H,2H),
    # shrink cols follow. Read-only here.
    XExt,
    stride_xe_m,
    stride_xe_n,
    # out  [M, H] contiguous: silu(gate)*up
    Out,
    stride_o_m,
    stride_o_n,
    AdapIdx,
    Bitmask,
    stride_bm_t,
    LBPACK,
    M,
    H,
    N,  # H = gate width; N = N_total = 2H = lora_B output-dim count
    SliceColR,  # [S, 6] int32 (S == 2: row 0 = gate, row 1 = up)
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Same by-design invariant as _switch_lora_expand_kernel: M (x_ext rows)
    # equals the token count behind AdapIdx/Bitmask, because routing is per token
    # and the decoder preserves token count. The unmasked Bitmask load relies on
    # it.
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    h_range = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_range < M
    mask_h = h_range < H
    base_mask = mask_m[:, None] & mask_h[None, :]

    gate_n = h_range  # gate output cols [0, H)
    up_n = H + h_range  # up   output cols [H, 2H)

    gate = tl.load(
        XExt + m_range[:, None] * stride_xe_m + gate_n[None, :] * stride_xe_n,
        mask=base_mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        XExt + m_range[:, None] * stride_xe_m + up_n[None, :] * stride_xe_n,
        mask=base_mask,
        other=0.0,
    ).to(tl.float32)

    # LoRA delta only for tiles that touch an adapter; the silu(gate)*up store
    # below always runs (base-only tiles still need their activation written).
    bitmask = tl.load(Bitmask + pid_m * stride_bm_t).to(tl.int64)
    if bitmask != 0:
        adap_ids = tl.load(AdapIdx + m_range, mask=mask_m, other=0).to(tl.int32)
        # Per-tier shrink-column bases for each slice: row 0 = gate, row 1 = up.
        g0 = tl.load(SliceColR + 0 * 6 + 0).to(tl.int32)
        g1 = tl.load(SliceColR + 0 * 6 + 1).to(tl.int32)
        g2 = tl.load(SliceColR + 0 * 6 + 2).to(tl.int32)
        g3 = tl.load(SliceColR + 0 * 6 + 3).to(tl.int32)
        g4 = tl.load(SliceColR + 0 * 6 + 4).to(tl.int32)
        g5 = tl.load(SliceColR + 0 * 6 + 5).to(tl.int32)
        u0 = tl.load(SliceColR + 1 * 6 + 0).to(tl.int32)
        u1 = tl.load(SliceColR + 1 * 6 + 1).to(tl.int32)
        u2 = tl.load(SliceColR + 1 * 6 + 2).to(tl.int32)
        u3 = tl.load(SliceColR + 1 * 6 + 3).to(tl.int32)
        u4 = tl.load(SliceColR + 1 * 6 + 4).to(tl.int32)
        u5 = tl.load(SliceColR + 1 * 6 + 5).to(tl.int32)
        gate += _accum_slice_delta(
            XExt,
            stride_xe_m,
            stride_xe_n,
            LBPACK,
            m_range,
            mask_m,
            adap_ids,
            gate_n,
            mask_h,
            bitmask,
            g0,
            g1,
            g2,
            g3,
            g4,
            g5,
            NA_16,
            NA_32,
            NA_64,
            NA_128,
            NA_256,
            NA_512,
            S,
            N,
            BLOCK_M,
            BLOCK_N,
        )
        up += _accum_slice_delta(
            XExt,
            stride_xe_m,
            stride_xe_n,
            LBPACK,
            m_range,
            mask_m,
            adap_ids,
            up_n,
            mask_h,
            bitmask,
            u0,
            u1,
            u2,
            u3,
            u4,
            u5,
            NA_16,
            NA_32,
            NA_64,
            NA_128,
            NA_256,
            NA_512,
            S,
            N,
            BLOCK_M,
            BLOCK_N,
        )

    # SwiGLU epilogue: silu(gate) * up, written as a contiguous [M, H] tile so
    # nothing downstream sees x_ext's strided base columns.
    out_val = (gate * tl.sigmoid(gate)) * up
    tl.store(
        Out + m_range[:, None] * stride_o_m + h_range[None, :] * stride_o_n,
        out_val.to(Out.dtype.element_ty),
        mask=base_mask,
    )


def switch_lora_expand_swiglu(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    slice_col_r: torch.Tensor,
    na: tuple,
    S: int,
    block_n: int,
    H: int,
    N: int,
) -> None:
    """Fused gate/up LoRA expand + SwiGLU for the shared-MLP first projection.

    Writes ``out[M, H] = silu(gate_corrected) * up_corrected`` where gate/up are
    ``x_ext`` columns [0,H)/[H,2H) plus their LoRA delta. ``S`` must be 2 (gate,
    up), ``N`` is N_total = 2H. ``out`` is a fresh contiguous buffer (mutated).
    """
    M = x_ext.shape[0]
    cfg = get_switch_lora_expand_config()
    grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(H, block_n))
    _switch_lora_expand_swiglu_kernel[grid](
        x_ext,
        x_ext.stride(0),
        x_ext.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        adapter_indices,
        bitmask,
        bitmask.stride(0),
        lb_packed,
        M,
        H,
        N,
        slice_col_r,
        NA_16=na[0],
        NA_32=na[1],
        NA_64=na[2],
        NA_128=na[3],
        NA_256=na[4],
        NA_512=na[5],
        S=S,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=block_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


# ---------------------------------------------------------------------------
# Shrink-only ("W-less") expand — the Shadow-Residual W_cross shunt
# ---------------------------------------------------------------------------
#
# The Shadow-Residual cross-stream link W_cross has NO base weight: its output is
# purely ``(x @ lora_A_cross.T) @ lora_B_cross.T``. There is no base region to
# accumulate into, so the in-place read-modify-write of switch_lora_expand does
# not apply. This kernel writes the delta to a FRESH, PRE-ZEROED ``out[M, N]``:
#
#   * ``x_ext`` holds ONLY shrink columns: ``x @ w_ext_cross.T`` where
#     ``w_ext_cross = build_w_ext(W=<empty [0, K]>, lora_A_cross_by_rank)``. With
#     an empty base the shrink rows start at column 0, so the caller's
#     ``slice_col_r`` bases start at 0 (not N_total).
#   * the read-modify-write epilogue becomes a plain STORE of the delta.
#   * a base token (kernel-local id 0) contributes ``delta == 0`` (its shrink
#     loads are masked to zero), and the launcher pre-zeros ``out``, so a base
#     token — and any tile the bitmask skips entirely — reads back exactly zero.
#     This keeps the cross-stream injection off the base stream: the SR invariant.
#
# Structurally identical to _switch_lora_expand_kernel except for the extra
# ``Out`` pointer and the store epilogue; the tier loops are copied verbatim.


@triton.jit
def _switch_lora_shrink_expand_kernel(
    # x_ext  [M, sum_r(n_r * S * r)] — shrink columns ONLY (no base region).
    XExt,
    stride_xe_m,
    stride_xe_n,
    # out  [M, N] contiguous, PRE-ZEROED — receives delta = shrink @ lora_B.
    Out,
    stride_o_m,
    stride_o_n,
    AdapIdx,
    Bitmask,
    stride_bm_t,
    LBPACK,
    M,
    N,
    TileSlice,
    SliceColR,
    NA_16: tl.constexpr,
    NA_32: tl.constexpr,
    NA_64: tl.constexpr,
    NA_128: tl.constexpr,
    NA_256: tl.constexpr,
    NA_512: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # Same by-design invariant as _switch_lora_expand_kernel: M (x_ext rows)
    # equals the token count behind AdapIdx/Bitmask (per-token routing). Tiles
    # with no adapter are skipped; ``out`` is pre-zeroed so they read back zero.
    bitmask = tl.load(Bitmask + pid_m * stride_bm_t).to(tl.int64)
    if bitmask == 0:
        return

    pid_n = tl.program_id(1)
    m_range = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_range = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_range < M
    mask_n = n_range < N

    # This output tile lives in exactly one slice (block_n divides every slice).
    # For the shunt (S == 1) there is a single slice; SliceColR bases start at 0
    # because w_ext_cross has no base region.
    s_idx = tl.load(TileSlice + pid_n).to(tl.int32)
    col_16 = tl.load(SliceColR + s_idx * 6 + 0).to(tl.int32)
    col_32 = tl.load(SliceColR + s_idx * 6 + 1).to(tl.int32)
    col_64 = tl.load(SliceColR + s_idx * 6 + 2).to(tl.int32)
    col_128 = tl.load(SliceColR + s_idx * 6 + 3).to(tl.int32)
    col_256 = tl.load(SliceColR + s_idx * 6 + 4).to(tl.int32)
    col_512 = tl.load(SliceColR + s_idx * 6 + 5).to(tl.int32)

    adap_ids = tl.load(AdapIdx + m_range, mask=mask_m, other=0).to(tl.int32)
    delta = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    id_off_32 = NA_16
    id_off_64 = NA_16 + NA_32
    id_off_128 = NA_16 + NA_32 + NA_64
    id_off_256 = NA_16 + NA_32 + NA_64 + NA_128
    id_off_512 = NA_16 + NA_32 + NA_64 + NA_128 + NA_256

    lb_off_16 = 0
    lb_off_32 = NA_16 * N * 16
    lb_off_64 = lb_off_32 + NA_32 * N * 32
    lb_off_128 = lb_off_64 + NA_64 * N * 64
    lb_off_256 = lb_off_128 + NA_128 * N * 128
    lb_off_512 = lb_off_256 + NA_256 * N * 256

    r16 = tl.arange(0, 16)
    for a in tl.static_range(NA_16):
        if (bitmask >> a) & 1:  # adapter (a+1) in this tile?
            mask_a = (adap_ids == (a + 1)) & mask_m  # rows that select it
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_16 + a * S * 16 + r16[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_16
                + a * (N * 16)
                + r16[:, None]
                + n_range[None, :] * 16,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r32 = tl.arange(0, 32)
    for a in tl.static_range(NA_32):
        if (bitmask >> (id_off_32 + a)) & 1:
            mask_a = (adap_ids == (id_off_32 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_32 + a * S * 32 + r32[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_32
                + a * (N * 32)
                + r32[:, None]
                + n_range[None, :] * 32,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r64 = tl.arange(0, 64)
    for a in tl.static_range(NA_64):
        if (bitmask >> (id_off_64 + a)) & 1:
            mask_a = (adap_ids == (id_off_64 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_64 + a * S * 64 + r64[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_64
                + a * (N * 64)
                + r64[:, None]
                + n_range[None, :] * 64,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r128 = tl.arange(0, 128)
    for a in tl.static_range(NA_128):
        if (bitmask >> (id_off_128 + a)) & 1:
            mask_a = (adap_ids == (id_off_128 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_128 + a * S * 128 + r128[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_128
                + a * (N * 128)
                + r128[:, None]
                + n_range[None, :] * 128,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r256 = tl.arange(0, 256)
    for a in tl.static_range(NA_256):
        if (bitmask >> (id_off_256 + a)) & 1:
            mask_a = (adap_ids == (id_off_256 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_256 + a * S * 256 + r256[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_256
                + a * (N * 256)
                + r256[:, None]
                + n_range[None, :] * 256,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    r512 = tl.arange(0, 512)
    for a in tl.static_range(NA_512):
        if (bitmask >> (id_off_512 + a)) & 1:
            mask_a = (adap_ids == (id_off_512 + a + 1)) & mask_m
            shrink = tl.load(
                XExt
                + m_range[:, None] * stride_xe_m
                + (col_512 + a * S * 512 + r512[None, :]) * stride_xe_n,
                mask=mask_a[:, None],
                other=0.0,
            ).to(tl.float32)
            lb = tl.load(
                LBPACK
                + lb_off_512
                + a * (N * 512)
                + r512[:, None]
                + n_range[None, :] * 512,
                mask=mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            delta += tl.dot(shrink, lb)

    # Shrink-only: plain STORE of the delta into the fresh ``out`` buffer (no base
    # region to read-modify-write). Base rows have delta == 0 and ``out`` is
    # pre-zeroed, so base tokens stay exactly zero. The store casts to ``Out``'s
    # own dtype so a non-bf16 shunt buffer is honored.
    out_ptrs = Out + m_range[:, None] * stride_o_m + n_range[None, :] * stride_o_n
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, delta.to(Out.dtype.element_ty), mask=store_mask)


def switch_lora_shrink_expand(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    na: tuple,
    S: int,
    block_n: int,
    N: int,
) -> None:
    """Shrink-only ("W-less") expand for the Shadow-Residual W_cross shunt.

    Writes ``out[M, N] = (x @ lora_A_cross.T) @ lora_B_cross.T`` for adapter
    tokens and exactly zero for base tokens (kernel-local id 0). ``x_ext`` holds
    only the shrink columns (``w_ext_cross`` built from an empty base ``W``), so
    ``slice_col_r`` bases start at 0. ``N`` is the shunt output width (= hidden
    size). ``out`` is pre-zeroed here so bitmask-skipped tiles read back zero.
    ``na`` is the 6-tuple of per-tier adapter counts.
    """
    M = x_ext.shape[0]
    out.zero_()
    cfg = get_switch_lora_expand_config()
    grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(N, block_n))
    _switch_lora_shrink_expand_kernel[grid](
        x_ext,
        x_ext.stride(0),
        x_ext.stride(1),
        out,
        out.stride(0),
        out.stride(1),
        adapter_indices,
        bitmask,
        bitmask.stride(0),
        lb_packed,
        M,
        N,
        tile_slice,
        slice_col_r,
        NA_16=na[0],
        NA_32=na[1],
        NA_64=na[2],
        NA_128=na[3],
        NA_256=na[4],
        NA_512=na[5],
        S=S,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=block_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
