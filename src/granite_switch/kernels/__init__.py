# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for Granite Switch (backend-agnostic)."""

from .switch_lora_kernel import (
    BLOCK_M,
    BLOCK_N,
    NUM_STAGES,
    NUM_WARPS,
    SUPPORTED_RANKS,
    build_w_ext,
    get_switch_lora_expand_config,
    promote_rank,
    switch_lora_expand,
    switch_lora_expand_swiglu,
    switch_lora_shrink_expand,
)

__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "NUM_STAGES",
    "NUM_WARPS",
    "SUPPORTED_RANKS",
    "build_w_ext",
    "get_switch_lora_expand_config",
    "promote_rank",
    "switch_lora_expand",
    "switch_lora_expand_swiglu",
    "switch_lora_shrink_expand",
]
