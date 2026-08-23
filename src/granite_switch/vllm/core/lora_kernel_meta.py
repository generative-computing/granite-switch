# SPDX-License-Identifier: Apache-2.0
"""Fused LoRA kernel metadata.

A per-module bitmask + remap metadata context for the fused switch-LoRA kernel.

The fused kernel uses a per-tile int64 bitmask for early exit decisions.
Bitmasks are computed per-module from kernel-local (post-remap) adapter
indices so that each module's bitmask exactly matches its kernel-local
adapter numbering — covering both the rank-reordering and non-applicable
adapter cases that would produce wrong results with a shared global bitmask.
"""

import torch
import triton
import triton.language as tl
from torch import nn

from granite_switch.kernels import BLOCK_M

# ---------------------------------------------------------------------------
# Triton bitmask reduction kernel
# ---------------------------------------------------------------------------


@triton.jit
def _or_combine(a, b):
    return a | b


@triton.jit
def _bitmask_reduce_kernel(
    adapter_indices_ptr,
    remap_tables_t_ptr,
    out_ptr,
    M,
    num_modules,
    NA_plus_1,
    BLOCK_M: tl.constexpr,
):
    pid_mod = tl.program_id(0)
    pid_tile = tl.program_id(1)

    offsets = pid_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offsets < M
    indices = tl.load(adapter_indices_ptr + offsets, mask=mask, other=0)
    indices = tl.maximum(indices, tl.zeros_like(indices))
    indices = tl.minimum(indices, NA_plus_1 - 1)

    remap_ptrs = remap_tables_t_ptr + indices * num_modules + pid_mod
    remapped = tl.load(remap_ptrs, mask=mask, other=0)

    is_active = remapped > 0
    bits = is_active.to(tl.int64) << tl.maximum(remapped - 1, tl.zeros_like(remapped))
    bitmask = tl.reduce(bits, axis=0, combine_fn=_or_combine)

    num_tiles = tl.cdiv(M, BLOCK_M)
    out_offset = pid_mod * num_tiles + pid_tile
    tl.store(out_ptr + out_offset, bitmask)


def _compute_bitmasks_triton(
    adapter_indices: torch.Tensor,
    remap_tables_t: torch.Tensor,
) -> torch.Tensor:
    """Production per-module bitmask computation (Triton ``_bitmask_reduce_kernel``).

    This launcher is the body of the ``granite_switch::compute_per_module_bitmasks``
    custom op (registered in ``_ensure_op_registered``), so it is the path that runs
    in deployment — including inside the ``@support_torch_compile`` forward, where the
    custom-op boundary keeps inductor from tracing into the kernel.
    ``_compute_bitmasks_reference`` below is the equivalent pure-torch spec; the two
    must agree bit-for-bit (locked by tests/vllm/test_bitmask_kernel_equivalence).
    """
    M = adapter_indices.shape[0]
    NA_plus_1, num_modules = remap_tables_t.shape
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M

    out = torch.empty(
        num_modules, num_tiles, dtype=torch.int64, device=adapter_indices.device
    )
    grid = (num_modules, num_tiles)
    _bitmask_reduce_kernel[grid](
        adapter_indices,
        remap_tables_t,
        out,
        M,
        num_modules,
        NA_plus_1,
        BLOCK_M=BLOCK_M,
    )
    return out


# ---------------------------------------------------------------------------
# Custom op: compute_per_module_bitmasks
#
# The production Triton launcher (_compute_bitmasks_triton) is registered as this
# custom op so inductor treats it as an opaque node — never traced into, never
# fused through, never autotuned with random data inside the @support_torch_compile
# forward. _compute_bitmasks_reference below is the pure-torch spec kept for tests.
# ---------------------------------------------------------------------------


def _compute_bitmasks_reference(
    adapter_indices: torch.Tensor,
    remap_tables_t: torch.Tensor,
) -> torch.Tensor:
    """Reference (pure-torch) per-module bitmask computation.

    Test-only readable spec for the bitmask contract. Production runs the Triton
    ``_compute_bitmasks_triton`` (registered as the ``compute_per_module_bitmasks``
    custom op); this reference is cross-checked against it bit-for-bit and is not
    wired into the forward path.

    Compute per-module bitmasks from adapter indices and remap tables.

    Args:
        adapter_indices: [M] int64, values in [0, NA].
        remap_tables_t: [NA+1, num_modules] int64.

    Returns:
        per_module_bitmasks: [num_modules, num_tiles] int64.
    """
    M = adapter_indices.shape[0]
    num_modules = remap_tables_t.shape[1]

    # Pad to BLOCK_M boundary so tile-reduction reshape works cleanly.
    pad_m = (-M) % BLOCK_M
    if pad_m:
        adapter_indices = torch.nn.functional.pad(adapter_indices, (0, pad_m), value=0)

    # Clamp for safety (no-op in normal execution, guards against stale data).
    NA = remap_tables_t.shape[0] - 1
    adapter_indices = adapter_indices.clamp(0, NA)

    # Gather kernel-local indices for all modules: [M_padded, num_modules] → T
    all_remapped = remap_tables_t[adapter_indices].T  # [num_modules, M_padded]

    # Per-token per-module bitmask: bit a set iff kernel-local adapter a+1
    # is present at that token position for that module.
    is_active = all_remapped > 0
    bits = is_active.long() << (all_remapped - 1).clamp(min=0)

    # Reduce per-tile: OR all BLOCK_M token-bitmasks within each tile.
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    cols = bits.view(num_modules, num_tiles, BLOCK_M)
    per_module_bitmasks = cols[:, :, 0].clone()
    for i in range(1, BLOCK_M):
        per_module_bitmasks.bitwise_or_(cols[:, :, i])

    return per_module_bitmasks


def _compute_bitmasks_fake(
    adapter_indices: torch.Tensor,
    remap_tables_t: torch.Tensor,
) -> torch.Tensor:
    M = adapter_indices.shape[0]
    num_modules = remap_tables_t.shape[1]
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    return torch.empty(
        num_modules, num_tiles, dtype=torch.int64, device=adapter_indices.device
    )


_op_registered = False


def _ensure_op_registered():
    global _op_registered
    if _op_registered:
        return
    _op_registered = True

    torch.library.custom_op(
        "granite_switch::compute_per_module_bitmasks",
        mutates_args=(),
    )(_compute_bitmasks_triton)

    torch.library.register_fake(
        "granite_switch::compute_per_module_bitmasks",
    )(_compute_bitmasks_fake)


class LoRAContext:
    """Shared per-forward LoRA kernel-metadata context.

    A single instance is created at model level and wired to every
    SwitchedLoRALinear / GraniteLoRAEmbeddedAttention via ``_lora_ctx``.
    Written once per forward in GraniteSwitchModel; read by every layer.
    """

    __slots__ = (
        "adapter_indices",
        "num_tokens",
        "per_module_bitmasks",
        "remapped_indices",
    )

    def __init__(self):
        self.adapter_indices: torch.Tensor | None = None
        self.remapped_indices: torch.Tensor | None = None
        self.per_module_bitmasks: torch.Tensor | None = None
        self.num_tokens: int = 0

    def reset(self):
        self.adapter_indices = None
        self.remapped_indices = None
        self.per_module_bitmasks = None
        self.num_tokens = 0


class FusedLoRAKernelMeta(nn.Module):
    """Compute per-module bitmask metadata for the fused switch-LoRA expand kernel.

    Called once per forward pass. Computes a bitmask for every
    SwitchedLoRALinear module in a single batched operation and stores
    them on the shared LoRAContext.

    The bitmask for module i uses kernel-local (post-remap) adapter indices
    so that bitmask bit a exactly corresponds to kernel-local adapter a+1
    in that module's expand kernel call.  This makes the bitmask exact:
    no false negatives (skipped LoRA) or false positives from non-applicable
    adapters that were remapped to 0 but still had their global bit set.

    register_remap_tables() must be called after finalize_weights() has run
    on all SwitchedLoRALinear instances (i.e. after load_weights()).
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        _ensure_op_registered()

    def register_remap_tables(
        self,
        all_remap_tables: torch.Tensor,
        module_cfg_keys: list,
    ) -> None:
        """Register per-module remap tables after finalize_weights().

        Args:
            all_remap_tables: [num_modules, NA+1] long tensor where row i is
                the remap_table of the i-th SwitchedLoRALinear.  Entry [i, 0]
                = 0 (base), entry [i, j] = kernel-local position of global
                adapter j in module i (0 if non-applicable).
            module_cfg_keys: list of (K, N_total, num_applicable) tuples, one
                per module.  Retained for future use but not consulted at
                forward time (BLOCK_M and BLOCK_N are fixed constants).
        """
        assert all_remap_tables.max() <= 64, (
            f"int64 bitmask supports at most 64 kernel-local adapters per module, "
            f"got {all_remap_tables.max().item()}"
        )

        # Store transposed: [NA+1, num_modules] for row-gather in the custom op.
        self.register_buffer(
            "_remap_tables_t",
            all_remap_tables.T.contiguous(),  # [NA+1, num_modules]
            persistent=False,
        )
        self._module_cfg_keys = module_cfg_keys

    def prepare_and_store(
        self, adapter_indices: torch.Tensor, ctx: LoRAContext
    ) -> None:
        """Compute per-module bitmasks and store on context.

        Args:
            adapter_indices: [num_tokens] with values 0=base, 1..num_adapters.
                             Global adapter indices as returned by SingleSwitch.
            ctx: Shared LoRAContext to populate.
        """
        assert self._remap_tables_t is not None, (
            "register_remap_tables() must be called before prepare_and_store(). "
            "Ensure _finalize_fused_lora() has run."
        )

        M = adapter_indices.shape[0]

        # Triton bitmask launcher, wrapped as the compute_per_module_bitmasks
        # custom op so it stays opaque to inductor inside the compiled forward.
        per_module_bitmasks = torch.ops.granite_switch.compute_per_module_bitmasks(
            adapter_indices,
            self._remap_tables_t,
        )

        # Per-module kernel-local indices, computed once for all modules here so
        # forward() need not re-gather remap_table[adapter_indices] per module.
        # _remap_tables_t is [NA+1, num_modules]; gather + transpose gives a
        # row-major [num_modules, M] buffer, so remapped_indices[module_idx] is a
        # stride-1 contiguous view (no copy, no launch) — exactly what the expand
        # kernel's AdapIdx slot requires.
        ctx.remapped_indices = self._remap_tables_t[adapter_indices].T.contiguous()

        ctx.per_module_bitmasks = per_module_bitmasks  # [num_modules, num_tiles_M]
        ctx.adapter_indices = adapter_indices
        ctx.num_tokens = M
