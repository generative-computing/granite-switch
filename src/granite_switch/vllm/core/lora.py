# SPDX-License-Identifier: Apache-2.0
"""Fused LoRA layer implementation for Granite Switch (vLLM).

Uses the SWITCH kernel backend: a single GEMM (base + all shrinks) followed by a
Triton expand kernel with bitmask per-tile early exit. See
``granite_switch.kernels.switch_lora_kernel`` for the backend and its naming.
"""

import logging

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from granite_switch.kernels import (
    BLOCK_M,
    BLOCK_N,
    SUPPORTED_RANKS,
    build_w_ext,
    promote_rank,
)
from granite_switch.vllm.core.lora_ops import (
    switch_lora_expand,
    switch_lora_expand_swiglu,
)

logger = logging.getLogger(__name__)


class SwitchedLoRALinear(nn.Module):
    """Fused LoRA linear layer using the switch-LoRA kernel.

    Forward path:
      1. x_ext = x @ w_ext.T   (single GEMM: base output + all shrink vectors)
      2. _lora_expand(...)       (Triton kernel: accumulate LoRA into base_out)

    Weights are stored in checkpoint-compatible format during loading, then
    converted to fused format via finalize_weights().

    Memory layout (post finalize_weights)
    ======================================
    For a layer with S slices (S=1 for o_proj/down_proj, S=2 for gate_up,
    S=3 for QKV), N_total = sum(N_s) output features, K input features,
    and adapters grouped into rank tiers:

    INVARIANTS
    ----------
    1. Each adapter has a rank per module. That rank applies to lora_A
       and lora_B for every slice within that module.  All adapters of
       the same rank (within a module) belong to the same rank tier.
       Different modules may assign different ranks to the same adapter.

    2. An adapter may or may not be applicable to a given module (i.e. have
       non-zero trained weights for it).  Non-applicable adapters contribute
       no rows to w_ext and are remapped to 0 (base) in this module's
       remap_table.

    3. The rank-tier ordering is global (same across all modules). The
       remap_table is per-module (non-applicable adapters are compacted out,
       applicable adapters are sorted by rank). The bitmask is per-module,
       computed from kernel-local (post-remap) adapter indices so that
       bitmask bit a exactly corresponds to kernel-local adapter a+1.
       Column offsets (col_r) into x_ext are also per-module.

    PER-MODULE data (built at finalize_weights, differ across instances)
    --------------------------------------------------------------------
      remap_table   global adapter_id → kernel-local position for this module
                    (0 for non-applicable adapters)
      bitmask       computed per-forward by FusedLoRAKernelMeta from
                    kernel-local indices; exact for this module
      w_ext         only applicable adapters, sorted by rank tier
      lora_B_merged same — only applicable adapters, merged along N_total
      NA_r          count of applicable adapters per rank tier (constexpr
                    in kernel; 0 means the entire tier compiles away)
      col_r         base column offset in x_ext for each rank tier

    w_ext  [N_total + sum_{a applicable to this module}(S * r_a),  K]
    -------------------------------------------------------
      rows 0 .. N_total-1     : W_base (all slices fused, as stored by vLLM)
      tier r0, adapter a0     : lora_A_a0_s0, lora_A_a0_s1, ... (S*r0 rows)
      tier r0, adapter a1     : lora_A_a1_s0, lora_A_a1_s1, ... (S*r0 rows)
      ...
      tier r1, adapter b0     : S*r1 rows
      ...
      (non-applicable adapters contribute no rows)

    x_ext = x @ w_ext.T   [M,  N_total + sum_{a applicable to this module}(S * r_a)]
    -------------------------------------------------------
      cols 0 .. N_total-1     : base outputs (all slices concatenated)
      tier r0, adapter a0     : shrink_s0, shrink_s1, ... each of width r0
      tier r0, adapter a1     : shrink_s0, shrink_s1, ...
      ...

    For token m, adapter a (rank r, per-module tier position pos_a), slice s:
      shrink = x_ext[m, col_r + pos_a*S*r + s*r : col_r + pos_a*S*r + s*r + r]

    lora_B_merged  per rank tier:  [n_r_local,  N_total,  r]
    -------------------------------------------------------
      lora_B_merged[pos_a, 0:N_s0, :]         = lora_B for adapter a, slice 0
      lora_B_merged[pos_a, N_s0:N_s0+N_s1, :] = lora_B for adapter a, slice 1
      ...

    N_s denotes the output feature count of slice s — a property of the base
    layer geometry alone, independent of adapter count or rank.  For a fused
    layer, N_total = sum(N_s) and W_base is the [N_total, K] weight matrix with
    the slice sub-matrices stacked vertically (e.g. for qkv_proj: N_0=q_size,
    N_1=k_size, N_2=v_size).

    Tile (pid_m, pid_n) in the expand kernel covers output columns
    [pid_n*BLOCK_N, (pid_n+1)*BLOCK_N).  For the per-tile slice lookup
    (TileSlice[pid_n]) to be correct, every tile must fall entirely within one
    slice — no tile may straddle a slice boundary.  This requires BLOCK_N to
    divide every N_s exactly (N_s % BLOCK_N == 0 for all s), which ensures that
    every slice boundary is also a tile boundary.  BLOCK_N <= min(N_s) alone is
    not sufficient.  The assert in finalize_weights() enforces this on the local
    (post-TP-shard) slice sizes at load time.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        num_adapters: int,
        max_lora_rank: int,
        num_slices: int = 1,
        output_slices: tuple[int, ...] | None = None,
        fuse_swiglu: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_adapters = num_adapters
        self.max_lora_rank = max_lora_rank
        self.num_slices = num_slices
        # When True (shared-MLP gate/up projection only), forward() fuses the
        # LoRA expand with the SwiGLU activation and returns the activated
        # [M, H] directly — no strided base_out, no separate SiluAndMul. Requires
        # the merged 2-slice (gate, up) layout with equal slice widths.
        self.fuse_swiglu = fuse_swiglu

        if hasattr(base_layer, "weight"):
            in_features = base_layer.weight.shape[1]
            out_features = base_layer.weight.shape[0]
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
        elif hasattr(base_layer, "qweight"):
            in_features = base_layer.input_size
            out_features = base_layer.output_size
            device = base_layer.qweight.device
            dtype = torch.float16
        else:
            raise ValueError(f"Unsupported base layer type: {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self._device = device
        self._dtype = dtype

        # TP config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self._is_column_parallel = isinstance(
            base_layer,
            (ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear),
        )
        self._is_row_parallel = isinstance(base_layer, RowParallelLinear)
        self._row_parallel_reduce = (
            self._is_row_parallel
            and self.tp_size > 1
            and getattr(base_layer, "reduce_results", False)
        )

        # Output slices for packed modules
        if num_slices > 1:
            if output_slices is None:
                raise ValueError("output_slices required for packed modules")
            if self._is_column_parallel and self.tp_size > 1:
                # Assumes each s is divisible by tp_size — enforced by vLLM's
                # column-parallel layer constructors, not re-checked here.
                self.output_slices = tuple(s // self.tp_size for s in output_slices)
            else:
                self.output_slices = output_slices
        else:
            self.output_slices = (out_features,)

        # Checkpoint-format parameters (populated by weight_loader, consumed by finalize_weights)
        if num_slices == 1:
            self.lora_A = nn.Parameter(
                torch.zeros(
                    num_adapters,
                    1,
                    max_lora_rank,
                    in_features,
                    dtype=dtype,
                    device=device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    num_adapters,
                    1,
                    out_features,
                    max_lora_rank,
                    dtype=dtype,
                    device=device,
                )
            )
            self.lora_A.weight_loader = self._make_weight_loader("a")
            self.lora_B.weight_loader = self._make_weight_loader("b")
        else:
            self.lora_A_slices = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(
                            num_adapters,
                            1,
                            max_lora_rank,
                            in_features,
                            dtype=dtype,
                            device=device,
                        )
                    )
                    for _ in range(num_slices)
                ]
            )
            self.lora_B_slices = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(
                            num_adapters,
                            1,
                            output_size,
                            max_lora_rank,
                            dtype=dtype,
                            device=device,
                        )
                    )
                    for output_size in self.output_slices
                ]
            )
            for i, p in enumerate(self.lora_A_slices):
                p.weight_loader = self._make_weight_loader("a", i)
            for i, p in enumerate(self.lora_B_slices):
                p.weight_loader = self._make_weight_loader("b", i)

        # Fused kernel state (populated by finalize_weights)
        self._finalized = False

    # Class-level default so the attribute exists statically (torch.compile sees a
    # stable attribute, not a per-instance add). Wired post-init by GraniteSwitchModel
    # via object.__setattr__ to a single shared LoRAContext.
    _lora_ctx = None

    @property
    def weight(self):
        return self.base_layer.weight

    def slice_lora_a_weight(
        self, full_weight: torch.Tensor, slice_idx: int = 0
    ) -> torch.Tensor:
        if self.tp_size <= 1 or not self._is_row_parallel:
            return full_weight
        full_in = full_weight.shape[-1]
        shard_size = full_in // self.tp_size
        start = self.tp_rank * shard_size
        return full_weight[..., start : start + shard_size]

    def slice_lora_b_weight(
        self, full_weight: torch.Tensor, slice_idx: int = 0
    ) -> torch.Tensor:
        if self.tp_size <= 1 or not self._is_column_parallel:
            return full_weight
        full_out = full_weight.shape[-2]
        shard_size = full_out // self.tp_size
        start = self.tp_rank * shard_size
        return full_weight[..., start : start + shard_size, :]

    def _make_weight_loader(self, ab: str, slice_idx: int = 0):
        slicer = self.slice_lora_a_weight if ab == "a" else self.slice_lora_b_weight

        def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
            sliced = slicer(loaded_weight, slice_idx)
            param.data.copy_(sliced)

        return weight_loader

    def finalize_weights(self, adapter_ranks: list[int], block_n: int | None = None):
        """Convert checkpoint-format LoRA weights to fused kernel format.

        Called once after load_weights(). Builds w_ext, lora_B_merged, and
        the adapter index remap table. Handles both single-slice (S=1) and
        multi-slice (S>1) layers through a unified code path.

        TP assumption: lora_A/lora_B are sharded by even integer division of the
        in/out feature dim across tp_size (see slice_lora_a_weight / slice_lora_b_weight),
        which mirrors how vLLM's parallel linear layers shard the base weight. The
        per-rank shard sizes are therefore exact (no ragged final shard).

        Args:
            adapter_ranks: Rank per adapter for this module (length = num_adapters).
                           Each rank applies to all slices within this module.
                           Currently the same list is passed to all modules (from
                           config.adapter_ranks); per-module differentiation comes
                           from zero-detection only.  The interface accepts a
                           per-module list to support future per-module rank
                           assignment (e.g. adapter 0 rank 16 here but rank 32
                           elsewhere).  Rank 0 or all-zero lora_A rows marks an
                           adapter as non-applicable to this module.
        """
        if self._finalized:
            return

        # block_n determines the precomputed tile/slice tables, so it is bound
        # here at finalize time (not per launch). Defaults to the kernel's
        # BLOCK_N; must divide every output slice so no tile straddles a slice
        # boundary (asserted below).
        if block_n is None:
            block_n = BLOCK_N

        from collections import OrderedDict

        device = self._device
        dtype = self._dtype
        NA = self.num_adapters
        S = self.num_slices

        # Collect lora_A and lora_B checkpoint data for all slices
        if S == 1:
            lora_A_all = [self.lora_A.data]  # [NA, 1, max_rank, K]
            lora_B_all = [self.lora_B.data]  # [NA, 1, N, max_rank]
        else:
            lora_A_all = [p.data for p in self.lora_A_slices]
            lora_B_all = [p.data for p in self.lora_B_slices]

        # Detect coverage: adapters with all-zero lora_A (over their first r rows,
        # any slice) don't cover this module. Computed as ONE batched GPU
        # reduction with a single host sync, not NA*S separate .item() calls.
        # finalize_weights runs on every SwitchedLoRALinear in the model, so the
        # per-element .item() syncs (each a GPU->CPU stall) dominated load time;
        # collapsing them to one .tolist() per module is the bulk of that cost.
        max_rank = lora_A_all[0].shape[2]
        ranks_t = torch.tensor(adapter_ranks, device=device)  # [NA]
        # rank_mask[i, j] = j < r_i  — restricts the check to each adapter's rows.
        rank_mask = torch.arange(max_rank, device=device)[None, :] < ranks_t[:, None]
        applicable_t = torch.zeros(NA, dtype=torch.bool, device=device)
        for s in range(S):
            # [NA, max_rank, K] -> nonzero per (adapter, rank-row) -> [NA, max_rank]
            nz = lora_A_all[s][:, 0, :, :].ne(0).any(dim=-1)
            applicable_t |= (nz & rank_mask).any(dim=1)
        applicable = applicable_t.tolist()  # single sync

        # Build remap_table: global adapter_id (1-based) → kernel-local position
        # Non-applicable adapters map to 0 (base model, no LoRA contribution)
        applicable_adapters = [i for i in range(NA) if applicable[i]]

        # The kernel's tier-based position numbering only knows SUPPORTED_RANKS,
        # but a checkpoint may legitimately carry an off-tier rank — rank 8 is a
        # common LoRA choice, and compose zero-pads every adapter to the model's
        # max_lora_rank, so the whole checkpoint can sit between tiers. Promote
        # each such adapter to the next supported tier and zero-pad its lora_A
        # rows / lora_B columns to match: the padded rows contribute nothing to
        # the shrink and their lora_B columns contribute nothing to the expand,
        # so the promoted adapter is numerically identical to the original.
        #
        # Promotion (rather than adding the rank to SUPPORTED_RANKS) is what the
        # kernel plumbing allows: slice_col_r is built as [S, 6] and _na as a
        # 6-tuple, so the tier count is fixed at six.
        eff_ranks = [promote_rank(r) for r in adapter_ranks]
        pad_to = max((eff_ranks[i] for i in applicable_adapters), default=0)
        if pad_to > max_rank:
            pad = pad_to - max_rank
            # lora_A is [NA, 1, max_rank, K] — pad the rank dim (second to last).
            lora_A_all = [
                torch.nn.functional.pad(a, (0, 0, 0, pad)) for a in lora_A_all
            ]
            # lora_B is [NA, 1, N, max_rank] — pad the rank dim (last).
            lora_B_all = [torch.nn.functional.pad(b, (0, pad)) for b in lora_B_all]
        adapter_ranks = eff_ranks

        rank_order = sorted(applicable_adapters, key=lambda i: adapter_ranks[i])
        remap = torch.zeros(NA + 1, dtype=torch.long, device=device)
        for kernel_idx, orig_idx in enumerate(rank_order):
            remap[orig_idx + 1] = kernel_idx + 1
        self.register_buffer("remap_table", remap, persistent=False)

        # Build rank tiers (applicable adapters only, ascending rank order)
        tiers = OrderedDict()
        for orig_idx in rank_order:
            r = adapter_ranks[orig_idx]
            if r not in tiers:
                tiers[r] = []
            tiers[r].append(orig_idx)

        # Build lora_A_by_rank: {rank: [n_r, S, rank, K]}
        # Layout: adapter outer, slice middle, rank-row inner → tier→adapter→slice order in w_ext
        lora_A_by_rank = {}
        for rank, orig_indices in tiers.items():
            A_list = []
            for oi in orig_indices:
                # [S, rank, K] — all slices for this adapter
                slices = torch.stack(
                    [lora_A_all[s][oi, 0, :rank, :] for s in range(S)], dim=0
                )
                A_list.append(slices)
            lora_A_by_rank[rank] = torch.stack(A_list, dim=0)  # [n_r, S, rank, K]

        # Build w_ext = [W_base | tier_r0_a0_s0, a0_s1..., a1_s0, a1_s1... | tier_r1 ...]
        W_base = self.base_layer.weight.data  # [N_total, K]
        w_ext = build_w_ext(W_base, lora_A_by_rank)
        self.register_buffer("w_ext", w_ext, persistent=False)

        N_total = W_base.shape[0]
        self._N_total = N_total

        # Build lora_B_merged per tier: {rank: [n_r, N_total, rank]}
        # Slices are concatenated along N_total so tiles can access any output col uniformly.
        tier_info = {}
        lora_B_merged = {}
        for rank, orig_indices in tiers.items():
            B_list = []
            for oi in orig_indices:
                # Cat lora_B across slices → [N_total, rank]
                B_adapter = torch.cat(
                    [lora_B_all[s][oi, 0, :, :rank] for s in range(S)], dim=0
                )
                B_list.append(B_adapter)
            lora_B_merged[rank] = torch.stack(B_list, dim=0)  # [n_r, N_total, rank]
            tier_info[rank] = len(orig_indices)

        K = self.in_features
        self._num_applicable = sum(tier_info.values())
        self._block_cfg_key = (K, N_total, self._num_applicable)

        self._block_n = block_n
        assert all(N_s % block_n == 0 for N_s in self.output_slices), (
            f"block_n={block_n} must divide every output slice: {self.output_slices}"
        )

        # Build tile_to_slice[num_tiles_N]: slice index for each output tile
        num_tiles_N = (N_total + block_n - 1) // block_n
        N_cumsum = [0]
        for N_s in self.output_slices:
            N_cumsum.append(N_cumsum[-1] + N_s)

        tile_slice_data = torch.zeros(num_tiles_N, dtype=torch.int32, device=device)
        for t in range(num_tiles_N):
            col_start = t * block_n
            for s in range(S):
                if N_cumsum[s] <= col_start < N_cumsum[s + 1]:
                    tile_slice_data[t] = s
                    break
        self.register_buffer("tile_to_slice", tile_slice_data, persistent=False)

        # Build slice_col_r[S, 6]: for each (slice, tier), the effective base column
        # in x_ext for shrink reads.
        # slice_col_r[s, t] = N_total + sum_{t'<t}(n_t' * S * r_t') + s * r_t
        tier_col_bases = []
        offset = N_total
        for r in SUPPORTED_RANKS:
            tier_col_bases.append(offset)
            n_r = tier_info.get(r, 0)
            offset += n_r * S * r

        slice_col_r_data = torch.tensor(
            [
                [tier_col_bases[t] + s * SUPPORTED_RANKS[t] for t in range(6)]
                for s in range(S)
            ],
            dtype=torch.int32,
            device=device,
        )  # [S, 6]
        self.register_buffer("slice_col_r", slice_col_r_data, persistent=False)

        # Pre-cache expand kernel arguments
        self._precompute_expand_args(
            lora_B_merged, tier_info, N_total, device, dtype, S
        )

        # Fused gate/up + SwiGLU setup (shared-MLP first projection only).
        if self.fuse_swiglu:
            assert S == 2 and self.output_slices[0] == self.output_slices[1], (
                "fuse_swiglu requires the merged 2-slice (gate, up) layout with "
                f"equal widths; got S={S}, output_slices={self.output_slices}"
            )
            # SwiGLU is applied to the post-projection gate/up; a base bias would
            # have to be folded in before silu*mul. Granite gate/up is bias-free.
            assert getattr(self.base_layer, "bias", None) is None, (
                "fuse_swiglu does not support a biased gate/up projection"
            )
            self._H = N_total // 2

        # Bias. Register the buffer slot as None FIRST, then assign through it.
        # Setting `self._fused_bias = None` as a plain attribute up front and
        # then calling register_buffer('_fused_bias', ...) would raise
        # ("attribute already exists"): nn.Module.register_buffer rejects a name
        # that is already a non-buffer attribute. Registering the slot as None
        # and assigning the tensor afterwards routes through buffer machinery.
        self.register_buffer("_fused_bias", None, persistent=False)
        self._output_bias = None
        if getattr(self.base_layer, "bias", None) is not None:
            if not getattr(self.base_layer, "skip_bias_add", False):
                self._fused_bias = self.base_layer.bias.data
            else:
                self._output_bias = self.base_layer.bias

        # Freeze checkpoint-format parameters (data retained for state_dict)
        if S == 1:
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)
        else:
            for p in self.lora_A_slices:
                p.requires_grad_(False)
            for p in self.lora_B_slices:
                p.requires_grad_(False)

        self._finalized = True

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass using fused switch-LoRA kernel."""
        assert self._finalized, "finalize_weights() must be called before forward()"

        x_ext = torch.mm(x, self.w_ext.T)

        # Shared-MLP gate/up: fuse expand + SwiGLU and return the activated
        # [M, H] directly. No strided base_out ever escapes (the kernel reads
        # x_ext by explicit stride), so no .contiguous() and no SiluAndMul.
        if self.fuse_swiglu:
            return self._forward_swiglu(x, x_ext)

        base_out = x_ext[:, : self._N_total]

        ctx = self._lora_ctx
        if (
            ctx is not None
            and ctx.adapter_indices is not None
            and self._num_applicable > 0
        ):
            M = x.shape[0]
            # Kernel-local indices were gathered once for all modules in
            # prepare_and_store(); this is a stride-1 contiguous row-view into
            # ctx.remapped_indices [num_modules, num_tokens] — no per-module
            # gather op, no launch.
            adapter_indices = ctx.remapped_indices[self._module_idx, :M]
            self._run_expand(x_ext, adapter_indices, ctx)

        if self._row_parallel_reduce:
            # base_out is a column-slice of x_ext (row stride N+shrink_cols), so
            # it is non-contiguous; all-reduce's internal .view() requires a
            # packed layout. Copy to contiguous before the reduce. NOTE: this
            # copy is on the TP>1 row-parallel critical path — if it proves
            # costly for large models, revisit fusing the base output into a
            # standalone buffer rather than sharing x_ext.
            base_out = tensor_model_parallel_all_reduce(base_out.contiguous())

        # Bias is added AFTER the all-reduce, never folded into the local partial
        # before it. A row-parallel rank holds only a partial sum; the bias
        # belongs to the full (reduced) output, so adding it pre-reduce would sum
        # it tp_size times. The LoRA delta, by contrast, IS a partial and must go
        # in pre-reduce (above). For TP=1 and column-parallel there is no reduce,
        # so this is just "add the bias once" and its order vs the delta does not
        # matter. (skip_bias_add=True returns the bias via _output_bias instead;
        # the caller applies it, so it is untouched here.)
        if self._fused_bias is not None:
            base_out = base_out + self._fused_bias

        # base_out is a strided view of x_ext (shrink columns make the row stride
        # > N_total). That is fine for every consumer in the Granite stack:
        # qkv -> attention split+RoPE and o/down -> residual add are all
        # stride-safe. The one consumer that assumes packed rows (SiluAndMul on
        # the gate/up output) is handled by the fuse_swiglu path above, which
        # never exposes a strided base. So no .contiguous() is needed here.
        return base_out, self._output_bias

    def _forward_swiglu(self, x: torch.Tensor, x_ext: torch.Tensor):
        """Fused gate/up expand + SwiGLU -> contiguous [M, H] activation."""
        M = x.shape[0]
        H = self._H
        out = torch.empty(M, H, device=x_ext.device, dtype=x_ext.dtype)

        ctx = self._lora_ctx
        if (
            ctx is not None
            and ctx.adapter_indices is not None
            and self._num_applicable > 0
        ):
            adapter_indices = ctx.remapped_indices[self._module_idx, :M]
            bitmask = ctx.per_module_bitmasks[self._module_idx]
        else:
            # No kernel metadata / no applicable adapters: a zero bitmask makes the
            # kernel skip all LoRA work and emit silu(gate)*up of the base only.
            num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M
            adapter_indices = torch.zeros(M, dtype=torch.long, device=x_ext.device)
            bitmask = torch.zeros(num_tiles_m, dtype=torch.int64, device=x_ext.device)

        switch_lora_expand_swiglu(
            out,
            x_ext,
            adapter_indices,
            bitmask,
            self._lb_packed,
            self.slice_col_r,
            self._na[0],
            self._na[1],
            self._na[2],
            self._na[3],
            self._na[4],
            self._na[5],
            self._S,
            self._block_n,
            H,
            self._N_total,
        )
        return out, self._output_bias

    def _precompute_expand_args(
        self, lora_B_merged, tier_info, N_total, device, dtype, S
    ):
        """Cache expand kernel arguments at finalize time."""
        RANKS = SUPPORTED_RANKS
        self._na = tuple(tier_info.get(r, 0) for r in RANKS)
        self._S = S

        # Single packed lora_B buffer — contiguous concat over PRESENT tiers of
        # [NA_r, N_total, r] (row-major). Empty tiers (NA_r == 0) contribute zero
        # elements; the kernel computes each tier's base offset as cumsum(NA_r*N*r)
        # from the NA_* constexprs + N, so this must match exactly that ordering.
        packed_parts = [
            lora_B_merged[r].reshape(-1) for r in RANKS if r in lora_B_merged
        ]
        lb_packed = (
            torch.cat(packed_parts)
            if packed_parts
            else torch.zeros(1, device=device, dtype=dtype)
        )
        self.register_buffer("_lb_packed", lb_packed.contiguous(), persistent=False)

    def _run_expand(self, x_ext, adapter_indices, ctx):
        """Accumulate the LoRA delta in place into x_ext[:, :N] (all slices in one
        launch).

        Whole-buffer in-place accumulate: x_ext is folded as both shrink-read
        source and base-write target (disjoint columns), with a single packed
        lora_B. Mutating x_ext itself (not its [:, :N] view) keeps the inductor
        graph glue-free (no clone + slice_scatter). base_out in forward() already
        aliases x_ext[:, :N_total], so no rebind is needed after this call.
        """
        bitmask = ctx.per_module_bitmasks[self._module_idx]
        switch_lora_expand(
            x_ext,
            adapter_indices,
            bitmask,
            self._lb_packed,
            self.tile_to_slice,
            self.slice_col_r,
            self._na[0],
            self._na[1],
            self._na[2],
            self._na[3],
            self._na[4],
            self._na[5],
            self._S,
            self._block_n,
            self._N_total,
        )
