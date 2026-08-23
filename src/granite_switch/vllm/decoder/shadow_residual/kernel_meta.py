# SPDX-License-Identifier: Apache-2.0
"""Shadow-Residual kernel metadata for the SWITCH kernel.

Two token layouts share one set of per-module remap tables:

* the **2M stacked** projections (qkv / o / gate-up / down) run over
  ``cat([h_base, h_adapt])`` — their adapter-index vector is
  ``cat([0]*M, real)``: the base half is all 0 (pristine base projection, no
  delta, base-only K/V), the adapt half carries the real per-token adapter id.
* the **W_cross shunt** runs over only the ``M`` base-half rows and MUST be keyed
  by the ``M``-length REAL adapter ids. Keying it off the 2M vector's base-half
  zeros would make the shrink-only kernel (id 0 never fires + pre-zeroed output)
  emit identically zero, silently deleting the cross-stream injection. This is
  the "F9" trap; the separate ``*_shunt`` fields exist to avoid it.

``SRFusedLoRAKernelMeta`` computes BOTH layouts in one ``prepare_and_store_sr``
call and stores them on an ``SRLoRAContext``. It reuses the base
``FusedLoRAKernelMeta`` machinery (``register_remap_tables`` + the
``compute_per_module_bitmasks`` custom op) unchanged; only the per-forward
preparation is SR-specific.
"""

import torch

from granite_switch.vllm.core.lora_kernel_meta import (
    FusedLoRAKernelMeta,
    LoRAContext,
)


class SRLoRAContext(LoRAContext):
    """LoRAContext extended with the M-length real-id metadata for the shunt.

    ``remapped_indices`` / ``per_module_bitmasks`` (inherited) hold the **2M**
    metadata consumed by every SwitchedLoRALinear projection. The ``*_shunt``
    fields hold the **M**-length real-id metadata consumed by the WCrossShunt.
    """

    __slots__ = ("per_module_bitmasks_shunt", "remapped_indices_shunt")

    def __init__(self):
        super().__init__()
        self.remapped_indices_shunt: torch.Tensor | None = None
        self.per_module_bitmasks_shunt: torch.Tensor | None = None

    def reset(self):
        super().reset()
        self.remapped_indices_shunt = None
        self.per_module_bitmasks_shunt = None


class SRFusedLoRAKernelMeta(FusedLoRAKernelMeta):
    """Kernel metadata for the dual-stream SR forward.

    ``register_remap_tables`` is inherited unchanged; the module set it is
    given includes both the SwitchedLoRALinear projections and the WCrossShunt
    modules (each with its own ``_module_idx`` / ``remap_table``).
    """

    def prepare_and_store_sr(
        self, adapter_indices: torch.Tensor, ctx: SRLoRAContext
    ) -> None:
        """Compute + store both metadata layouts from the real per-token ids.

        Args:
            adapter_indices: ``[M]`` real ids (0=base, 1..num_adapters), one per
                real input token — as returned by SingleSwitch.
            ctx: shared SRLoRAContext to populate.
        """
        assert self._remap_tables_t is not None, (
            "register_remap_tables() must be called before prepare_and_store_sr(). "
            "Ensure the SR _finalize_fused_lora() has run."
        )

        M = adapter_indices.shape[0]

        # --- 2M stacked layout (projections): base half 0, adapt half real ---
        zeros = torch.zeros_like(adapter_indices)
        ai_2m = torch.cat([zeros, adapter_indices], dim=0)  # [2M]
        ctx.per_module_bitmasks = torch.ops.granite_switch.compute_per_module_bitmasks(
            ai_2m, self._remap_tables_t
        )
        ctx.remapped_indices = self._remap_tables_t[
            ai_2m
        ].T.contiguous()  # [num_modules, 2M]

        # --- M-length real-id layout (shunt): F9-safe (real ids, not zeros) ---
        ctx.per_module_bitmasks_shunt = (
            torch.ops.granite_switch.compute_per_module_bitmasks(
                adapter_indices, self._remap_tables_t
            )
        )
        ctx.remapped_indices_shunt = self._remap_tables_t[
            adapter_indices
        ].T.contiguous()  # [num_modules, M]

        ctx.adapter_indices = adapter_indices  # real ids, for the per-token head select
        ctx.num_tokens = M


__all__ = ["SRFusedLoRAKernelMeta", "SRLoRAContext"]
