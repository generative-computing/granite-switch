# SPDX-License-Identifier: Apache-2.0
"""vLLM custom-op registration for the switch-LoRA expand kernel.

Wraps the backend-agnostic Triton launcher in
``granite_switch.kernels.switch_lora_expand`` as a torch custom op so it is
opaque to torch.compile / CUDA-graph capture. Follows vLLM's own LoRA-op
convention (``vllm/lora/ops/triton_ops/lora_expand_op.py``): a typed op
function + a fake (meta) implementation registered via
``direct_register_custom_op``. Registered under the ``granite_switch::``
namespace via a dedicated ``Library`` so it does not collide with vLLM's ops.
"""

import torch
from torch.library import Library
from vllm.utils.torch_utils import direct_register_custom_op

from granite_switch.kernels import switch_lora_expand as _switch_lora_expand_launch
from granite_switch.kernels import (
    switch_lora_expand_swiglu as _switch_lora_expand_swiglu_launch,
)
from granite_switch.kernels import (
    switch_lora_shrink_expand as _switch_lora_shrink_expand_launch,
)

# Dedicated library so the op lives under torch.ops.granite_switch.*; its
# lifetime must outlive op use, so keep this module-level reference.
_granite_switch_lib = Library("granite_switch", "FRAGMENT")


def _switch_lora_expand(
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    """Accumulate the LoRA expand delta in place into ``x_ext[:, :N]``."""
    _switch_lora_expand_launch(
        x_ext,
        adapter_indices,
        bitmask,
        lb_packed,
        tile_slice,
        slice_col_r,
        (NA_16, NA_32, NA_64, NA_128, NA_256, NA_512),
        S,
        block_n,
        N,
    )


def _switch_lora_expand_fake(
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    return


def _switch_lora_expand_swiglu(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    H: int,
    N: int,
) -> None:
    """Fused gate/up LoRA expand + SwiGLU; writes silu(gate)*up into ``out``."""
    _switch_lora_expand_swiglu_launch(
        out,
        x_ext,
        adapter_indices,
        bitmask,
        lb_packed,
        slice_col_r,
        (NA_16, NA_32, NA_64, NA_128, NA_256, NA_512),
        S,
        block_n,
        H,
        N,
    )


def _switch_lora_expand_swiglu_fake(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    H: int,
    N: int,
) -> None:
    return


def _switch_lora_shrink_expand(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    """Shrink-only ("W-less") expand for the SR W_cross shunt; writes ``out``."""
    _switch_lora_shrink_expand_launch(
        out,
        x_ext,
        adapter_indices,
        bitmask,
        lb_packed,
        tile_slice,
        slice_col_r,
        (NA_16, NA_32, NA_64, NA_128, NA_256, NA_512),
        S,
        block_n,
        N,
    )


def _switch_lora_shrink_expand_fake(
    out: torch.Tensor,
    x_ext: torch.Tensor,
    adapter_indices: torch.Tensor,
    bitmask: torch.Tensor,
    lb_packed: torch.Tensor,
    tile_slice: torch.Tensor,
    slice_col_r: torch.Tensor,
    NA_16: int,
    NA_32: int,
    NA_64: int,
    NA_128: int,
    NA_256: int,
    NA_512: int,
    S: int,
    block_n: int,
    N: int,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="switch_lora_expand",
        op_func=_switch_lora_expand,
        mutates_args=["x_ext"],
        fake_impl=_switch_lora_expand_fake,
        target_lib=_granite_switch_lib,
    )
    switch_lora_expand = torch.ops.granite_switch.switch_lora_expand
except AttributeError:
    switch_lora_expand = _switch_lora_expand

try:
    direct_register_custom_op(
        op_name="switch_lora_expand_swiglu",
        op_func=_switch_lora_expand_swiglu,
        mutates_args=["out"],
        fake_impl=_switch_lora_expand_swiglu_fake,
        target_lib=_granite_switch_lib,
    )
    switch_lora_expand_swiglu = torch.ops.granite_switch.switch_lora_expand_swiglu
except AttributeError:
    switch_lora_expand_swiglu = _switch_lora_expand_swiglu

try:
    direct_register_custom_op(
        op_name="switch_lora_shrink_expand",
        op_func=_switch_lora_shrink_expand,
        mutates_args=["out"],
        fake_impl=_switch_lora_shrink_expand_fake,
        target_lib=_granite_switch_lib,
    )
    switch_lora_shrink_expand = torch.ops.granite_switch.switch_lora_shrink_expand
except AttributeError:
    switch_lora_shrink_expand = _switch_lora_shrink_expand
