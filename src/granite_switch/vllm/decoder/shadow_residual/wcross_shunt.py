# SPDX-License-Identifier: Apache-2.0
"""The Shadow-Residual W_cross cross-stream shunt (base -> adapter), SWITCH backend.

``W_cross`` is a "W-less" LoRA module: it has NO base weight. Its output is purely
``(h_base @ lora_A_cross.T) @ lora_B_cross.T`` for adapter tokens and exactly zero
for base tokens. It uses the shrink-only SWITCH kernel
(``switch_lora_shrink_expand``), so — unlike a SwitchedLoRALinear over a zeroed
base — it does NOT pay a full ``[H, H]`` base GEMM; its only matmul is the small
shrink projection ``h_base @ w_ext_cross.T`` (width = sum of applicable cross
ranks).

It reads the SR context's **M-length REAL-id** shunt metadata
(``remapped_indices_shunt`` / ``per_module_bitmasks_shunt``), NOT the 2M
base-half zeros — see :mod:`.kernel_meta` for why (the F9 trap).

Checkpoint layout mirrors a single-slice SwitchedLoRALinear so the composed
``cross_stream.lora_A`` / ``cross_stream.lora_B`` tensors load name-based:
``lora_A [NA, 1, cross_rank, H]``, ``lora_B [NA, 1, H, cross_rank]``.
"""

from collections import OrderedDict

import torch
from torch import nn

from granite_switch.kernels import (
    BLOCK_N,
    SUPPORTED_RANKS,
    build_w_ext,
    promote_rank,
)
from granite_switch.vllm.core.lora_ops import switch_lora_shrink_expand


class WCrossShunt(nn.Module):
    """Shrink-only cross-stream shunt for one SR decoder layer."""

    # Wired post-init by the model to the shared SRLoRAContext.
    _lora_ctx = None

    def __init__(
        self,
        hidden_size: int,
        num_adapters: int,
        cross_rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_adapters = num_adapters
        self.cross_rank = cross_rank
        self._device = device
        self._dtype = dtype

        # Checkpoint-format params (single-slice layout; consumed by finalize).
        self.lora_A = nn.Parameter(
            torch.zeros(
                num_adapters, 1, cross_rank, hidden_size, dtype=dtype, device=device
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                num_adapters, 1, hidden_size, cross_rank, dtype=dtype, device=device
            )
        )
        self._finalized = False

    def finalize_weights(self, adapter_ranks: list[int], block_n: int | None = None):
        """Build the shrink-only fused kernel state from the loaded LoRA weights.

        ``adapter_ranks`` is the per-adapter cross rank (typically
        ``[cross_stream_rank] * num_adapters``). Mirrors
        ``SwitchedLoRALinear.finalize_weights`` for a single slice (S=1,
        N_total = hidden_size) EXCEPT the base region is empty: ``w_ext_cross``
        is built from an empty ``[0, H]`` base, so shrink columns — and therefore
        ``slice_col_r`` — start at 0.
        """
        if self._finalized:
            return
        if block_n is None:
            block_n = BLOCK_N

        device, dtype = self._device, self._dtype
        NA = self.num_adapters
        H = self.hidden_size
        S = 1

        lora_A = self.lora_A.data  # [NA, 1, max_rank, H]
        lora_B = self.lora_B.data  # [NA, 1, H, max_rank]
        max_rank = lora_A.shape[2]

        # Coverage: adapters with all-zero lora_A (over their first r rows) don't
        # apply here. One batched reduction + a single host sync (as in SwitchedLoRALinear).
        ranks_t = torch.tensor(adapter_ranks, device=device)  # [NA]
        rank_mask = torch.arange(max_rank, device=device)[None, :] < ranks_t[:, None]
        nz = lora_A[:, 0, :, :].ne(0).any(dim=-1)  # [NA, max_rank]
        applicable = (nz & rank_mask).any(dim=1).tolist()

        applicable_adapters = [i for i in range(NA) if applicable[i]]

        # Off-tier cross ranks are promoted to the next supported tier and the
        # lora_A rows / lora_B columns zero-padded to match — see
        # ``promote_rank`` and the matching block in
        # ``SwitchedLoRALinear.finalize_weights``.
        eff_ranks = [promote_rank(r) for r in adapter_ranks]
        pad_to = max((eff_ranks[i] for i in applicable_adapters), default=0)
        if pad_to > max_rank:
            pad = pad_to - max_rank
            lora_A = torch.nn.functional.pad(lora_A, (0, 0, 0, pad))  # rank dim -2
            lora_B = torch.nn.functional.pad(lora_B, (0, pad))  # rank dim -1
        adapter_ranks = eff_ranks

        rank_order = sorted(applicable_adapters, key=lambda i: adapter_ranks[i])

        remap = torch.zeros(NA + 1, dtype=torch.long, device=device)
        for kernel_idx, orig_idx in enumerate(rank_order):
            remap[orig_idx + 1] = kernel_idx + 1
        self.register_buffer("remap_table", remap, persistent=False)

        tiers = OrderedDict()
        for orig_idx in rank_order:
            tiers.setdefault(adapter_ranks[orig_idx], []).append(orig_idx)

        # lora_A_by_rank: {rank: [n_r, S=1, rank, H]}
        lora_A_by_rank = {}
        for rank, orig_indices in tiers.items():
            A_list = [
                torch.stack([lora_A[oi, 0, :rank, :]], dim=0)  # [1, rank, H]
                for oi in orig_indices
            ]
            lora_A_by_rank[rank] = torch.stack(A_list, dim=0)  # [n_r, 1, rank, H]

        # w_ext_cross: EMPTY base [0, H] followed by shrink rows only.
        empty_base = torch.empty(0, H, device=device, dtype=dtype)
        w_ext_cross = build_w_ext(empty_base, lora_A_by_rank)  # [sum_r n_r*rank, H]
        self.register_buffer("w_ext_cross", w_ext_cross, persistent=False)

        # lora_B_merged per tier: {rank: [n_r, H, rank]} (single slice, N_total = H).
        tier_info = {}
        lora_B_merged = {}
        for rank, orig_indices in tiers.items():
            B_list = [lora_B[oi, 0, :, :rank] for oi in orig_indices]  # each [H, rank]
            lora_B_merged[rank] = torch.stack(B_list, dim=0)  # [n_r, H, rank]
            tier_info[rank] = len(orig_indices)

        self._num_applicable = sum(tier_info.values())
        self._N = H
        self._block_cfg_key = (H, H, self._num_applicable)

        assert H % block_n == 0, f"block_n={block_n} must divide hidden_size={H}"
        self._block_n = block_n

        # Single slice -> tile_to_slice all zeros; N_total = H.
        num_tiles_N = (H + block_n - 1) // block_n
        self.register_buffer(
            "tile_to_slice",
            torch.zeros(num_tiles_N, dtype=torch.int32, device=device),
            persistent=False,
        )

        # slice_col_r [S=1, 6]: per-tier shrink-col base, starting at 0 (no base region).
        tier_col_bases = []
        offset = 0
        for r in SUPPORTED_RANKS:
            tier_col_bases.append(offset)
            offset += tier_info.get(r, 0) * S * r
        slice_col_r_data = torch.tensor(
            [[tier_col_bases[t] for t in range(6)]],  # s == 0
            dtype=torch.int32,
            device=device,
        )  # [1, 6]
        self.register_buffer("slice_col_r", slice_col_r_data, persistent=False)

        # Packed lora_B: contiguous concat over present tiers of [n_r, H, r] row-major.
        self._na = tuple(tier_info.get(r, 0) for r in SUPPORTED_RANKS)
        self._S = S
        packed_parts = [
            lora_B_merged[r].reshape(-1) for r in SUPPORTED_RANKS if r in lora_B_merged
        ]
        lb_packed = (
            torch.cat(packed_parts)
            if packed_parts
            else torch.zeros(1, device=device, dtype=dtype)
        )
        self.register_buffer("_lb_packed", lb_packed.contiguous(), persistent=False)

        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)
        self._finalized = True

    def forward(self, h_base: torch.Tensor) -> torch.Tensor:
        """Shunt delta for the M base-half rows: ``[M, H]``, zero for base tokens."""
        assert self._finalized, "finalize_weights() must be called before forward()"
        assert hasattr(self, "_module_idx"), (
            "WCrossShunt._module_idx must be assigned by the model's _finalize_fused_lora()"
        )
        M = h_base.shape[0]
        out = torch.empty(M, self._N, device=h_base.device, dtype=h_base.dtype)

        ctx = self._lora_ctx
        if (
            ctx is not None
            and ctx.remapped_indices_shunt is not None
            and self._num_applicable > 0
        ):
            # Keyed by the M-length REAL adapter ids (F9): base tokens (id 0) map to
            # kernel-local 0 -> the shrink kernel emits exactly zero for them.
            adapter_indices = ctx.remapped_indices_shunt[self._module_idx, :M]
            bitmask = ctx.per_module_bitmasks_shunt[self._module_idx]
            x_ext = torch.mm(h_base, self.w_ext_cross.T)  # [M, sum_r n_r*rank]
            # The lora_ops custom op takes NA_16..NA_512 as SEPARATE ints (it
            # unpacks internally to the launcher's `na` tuple), so unpack here —
            # exactly as SwitchedLoRALinear._run_expand does for switch_lora_expand.
            switch_lora_shrink_expand(
                out,
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
                self._N,
            )
        else:
            # No applicable adapters / no kernel meta: the shunt contributes nothing.
            out.zero_()
        return out


__all__ = ["WCrossShunt"]
