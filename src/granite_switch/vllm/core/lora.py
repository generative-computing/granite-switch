# SPDX-License-Identifier: Apache-2.0
"""LoRA layer implementation for Granite Switch (vLLM).

This module provides SwitchedLoRALinear, a LoRA linear layer that applies different
adapters per token based on precomputed adapter indices. It uses vLLM's optimized
Punica kernels for efficient computation.

IMPORTANT: The prepare_lora_metadata() function in this file is DEPRECATED and
should NOT be used. Instead, use CompileFriendlyLoRAKernelMeta from the core
package, which is designed for torch.compile compatibility.

Architecture:
1. CompileFriendlyLoRAKernelMeta.prepare_tensors() - prepares metadata (compile-friendly)
2. SwitchedLoRALinear - applies LoRA using precomputed metadata

Key features:
1. Per-token adapter selection based on precomputed indices
2. Uses vLLM's optimized Triton kernels (lora_shrink, lora_expand)
3. Works with torch.compile - no data-dependent branching
4. Metadata prepared ONCE per forward pass, shared by all linear layers

Current usage (in granite_switch_model.py):
    from .core import CompileFriendlyLoRAKernelMeta

    # Initialize (once)
    self.lora_meta = CompileFriendlyLoRAKernelMeta(
        num_adapters=num_adapters,
        device=torch.device('cuda'),
        dtype=torch.bfloat16,
    )

    # Prepare metadata (each forward pass)
    punica_indices = adapter_indices - 1  # Convert to Punica convention
    lora_meta_args = self.lora_meta.prepare_tensors(punica_indices)

    # Pass to layers
    output = layer(x, lora_meta_args, ...)
"""

import logging

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

logger = logging.getLogger(__name__)


class SwitchedLoRALinear(nn.Module):
    """LoRA linear layer that applies different adapters per token.

    This layer selects which LoRA adapter to apply for each token based on precomputed
    adapter indices. It uses vLLM's optimized Punica kernels (lora_shrink, lora_expand)
    for efficient per-token adapter application.

    The adapter selection metadata is computed ONCE per forward pass at the model level
    and passed to all linear layers, ensuring efficient batched computation.

    Supports packed modules (e.g., QKV projection with separate Q/K/V LoRA weights).

    Args:
        base_layer: Base linear layer to which LoRA is applied
        num_adapters: Number of LoRA adapters
        max_lora_rank: Maximum rank across all adapters
        num_slices: Number of LoRA slices (1 for standard layers, 3 for QKV, 2 for gate_up)
        output_slices: Tuple of output sizes for each slice (for packed modules)
    """

    def __init__(
        self,
        base_layer: nn.Module,
        num_adapters: int,
        max_lora_rank: int,
        num_slices: int = 1,
        output_slices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_adapters = num_adapters
        self.max_lora_rank = max_lora_rank
        self.num_slices = num_slices

        # NOTE: lora_B weights are PRE-SCALED by (alpha/rank) during model loading.
        # No runtime scaling is needed - we use implicit scaling factor of 1.0.
        # This eliminates data-dependent branching and makes the code torch.compile compatible.
        # num_adapters and max_lora_rank are config metadata, not runtime parameters.

        # Detect layer properties (handles both standard and vLLM parallel layers)
        if hasattr(base_layer, "weight"):
            in_features = base_layer.weight.shape[1]
            out_features = base_layer.weight.shape[0]
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
        elif hasattr(base_layer, "qweight"):
            # Quantized layer
            in_features = base_layer.input_size
            out_features = base_layer.output_size
            device = base_layer.qweight.device
            dtype = torch.float16
        else:
            raise ValueError(f"Unsupported base layer type: {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        # Tensor parallel configuration
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self._is_column_parallel = isinstance(
            base_layer,
            (ColumnParallelLinear, MergedColumnParallelLinear, QKVParallelLinear),
        )
        self._is_row_parallel = isinstance(base_layer, RowParallelLinear)

        # Row-parallel TP > 1: we must add LoRA BEFORE the all-reduce.
        # Our forward() calls quant_method.apply() directly (GEMM only, no
        # all-reduce), adds LoRA in-place, then does the all-reduce on the
        # combined result — matching vLLM's RowParallelLinearWithLoRA pattern.
        self._row_parallel_reduce = (
            self._is_row_parallel
            and self.tp_size > 1
            and getattr(base_layer, "reduce_results", False)
        )

        # For packed modules, we need output_slices.
        # vLLM's QKVParallelLinear.output_sizes and MergedColumnParallelLinear.output_sizes
        # return FULL (unsharded) per-projection sizes. For TP > 1 on column-parallel
        # layers, we must divide by tp_size to get the per-partition sizes that match
        # the actual sharded weight dimensions.
        if num_slices > 1:
            if output_slices is None:
                raise ValueError(
                    "output_slices must be provided for packed modules (num_slices > 1)"
                )
            if len(output_slices) != num_slices:
                raise ValueError(
                    f"output_slices length {len(output_slices)} != num_slices {num_slices}"
                )
            if self._is_column_parallel and self.tp_size > 1:
                self.output_slices = tuple(s // self.tp_size for s in output_slices)
            else:
                self.output_slices = output_slices
        else:
            self.output_slices = (out_features,)

        # LoRA weights: [num_adapters, 1, max_rank, features]
        # Shape matches vLLM's Triton kernel expectations
        # Adapters with smaller ranks are zero-padded to max_rank
        #
        # Structure explanation:
        # - num_adapters: Number of embedded LoRA adapters (selected by switch)
        # - For packed modules (QKV, gate_up): We create a tuple of tensors,
        #   one for each slice (Q/K/V or gate/up)
        # - Each tensor stacks all adapters for that slice
        #
        # This matches vLLM's MergedQKVParallelLinearWithLoRA structure:
        # - Outer: Tuple with n_slices elements (for Q/K/V or gate/up)
        # - Inner: Tensor with shape [num_adapters, 1, max_rank, features]
        #
        # These will be loaded from checkpoint via standard state_dict loading
        if num_slices == 1:
            # Standard case: single LoRA
            self.lora_A = nn.Parameter(
                torch.zeros(
                    self.num_adapters,
                    1,
                    self.max_lora_rank,
                    in_features,
                    dtype=dtype,
                    device=device,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    self.num_adapters,
                    1,
                    out_features,
                    self.max_lora_rank,
                    dtype=dtype,
                    device=device,
                )
            )
            self.lora_A.weight_loader = self._make_weight_loader("a")
            self.lora_B.weight_loader = self._make_weight_loader("b")
        else:
            # Packed module case: separate LoRA for each slice
            # Store as ParameterList to ensure proper parameter registration
            # Matches vLLM's lora_a_stacked / lora_b_stacked structure
            self.lora_A_slices = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(
                            self.num_adapters,
                            1,
                            self.max_lora_rank,
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
                            self.num_adapters,
                            1,
                            output_size,
                            self.max_lora_rank,
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

    # Shared LoRA context reference, wired post-init by GraniteSwitchModel.
    # Plain Python attribute (not nn.Parameter/buffer) — invisible to state_dict.
    _lora_ctx = None

    @property
    def weight(self):
        """Expose base layer weight for upstream module compatibility."""
        return self.base_layer.weight

    def slice_lora_a_weight(
        self,
        full_weight: torch.Tensor,
        slice_idx: int = 0,
    ) -> torch.Tensor:
        """Slice a full (unsharded) lora_A checkpoint weight for this TP rank.

        lora_A shape: [num_adapters, 1, max_rank, in_features]
        Row-parallel: input is sharded → slice last dim.
        Column-parallel: input is full → no-op.
        """
        if self.tp_size <= 1 or not self._is_row_parallel:
            return full_weight
        full_in = full_weight.shape[-1]
        shard_size = full_in // self.tp_size
        start = self.tp_rank * shard_size
        return full_weight[..., start : start + shard_size]

    def slice_lora_b_weight(
        self,
        full_weight: torch.Tensor,
        slice_idx: int = 0,
    ) -> torch.Tensor:
        """Slice a full (unsharded) lora_B checkpoint weight for this TP rank.

        lora_B shape: [num_adapters, 1, out_features, max_rank]
        Column-parallel: output is sharded → slice the out_features dim (dim -2).
        Row-parallel: output is all-reduced → no-op.

        Note: For packed modules (QKV, gate_up), each lora_B_slices[i] is a
        separate parameter containing only that slice's output dimension.
        A simple contiguous split by tp_size is correct — no interleaved
        shard handling needed (unlike a single fused lora_B tensor).
        """
        if self.tp_size <= 1 or not self._is_column_parallel:
            return full_weight
        full_out = full_weight.shape[-2]
        shard_size = full_out // self.tp_size
        start = self.tp_rank * shard_size
        return full_weight[..., start : start + shard_size, :]

    def _make_weight_loader(self, ab: str, slice_idx: int = 0):
        """Create a weight_loader that slices checkpoint LoRA weights for TP."""
        slicer = self.slice_lora_a_weight if ab == "a" else self.slice_lora_b_weight
        base_type = type(self.base_layer).__name__

        def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
            sliced = slicer(loaded_weight, slice_idx)
            logger.debug(
                "TP%d/%d lora_%s slice=%d base=%s param=%s loaded=%s sliced=%s",
                self.tp_rank,
                self.tp_size,
                ab,
                slice_idx,
                base_type,
                list(param.shape),
                list(loaded_weight.shape),
                list(sliced.shape),
            )
            param.data.copy_(sliced)

        return weight_loader

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass — reads LoRA metadata from the shared LoRAContext.

        Args:
            x: [num_tokens, in_features] input tensor

        Returns:
            output: [num_tokens, out_features] with LoRA applied
            bias: Optional bias tensor from base layer
        """
        # Read metadata from shared context (populated once per model forward)
        ctx = self._lora_ctx
        if ctx is not None and ctx.token_lora_mapping is not None:
            meta_args = (
                ctx.token_lora_mapping,
                ctx.token_indices_sorted,
                ctx.num_tokens_per_lora,
                ctx.lora_token_start_loc,
                ctx.active_lora_ids,
                ctx.no_lora_flag_cpu,
                ctx.num_active_loras,
            )
        else:
            meta_args = None

        # --- Base GEMM ---
        # For row-parallel TP > 1, we must add LoRA BEFORE the all-reduce.
        # We call quant_method.apply() directly (GEMM only, no all-reduce),
        # then add LoRA in-place, then all-reduce the combined result.
        # This matches vLLM's own RowParallelLinearWithLoRA pattern.
        # For all other layers, use the normal base_layer forward.
        if self._row_parallel_reduce:
            bias = None if self.base_layer.skip_bias_add else self.base_layer.bias
            output = self.base_layer.quant_method.apply(
                self.base_layer,
                x,
                bias,
            )
            output_bias = (
                self.base_layer.bias if self.base_layer.skip_bias_add else None
            )
        else:
            output, output_bias = self.base_layer(x)

        # --- LoRA computation ---
        num_tokens = x.size(0)
        buffer = torch.empty(
            (self.num_slices, num_tokens, self.max_lora_rank),
            dtype=torch.float32,
            device=x.device,
        )

        if self.num_slices == 1:
            lora_shrink(
                x,
                [self.lora_A],
                buffer,
                *meta_args,
                1.0,
            )
            lora_expand(
                buffer,
                [self.lora_B],
                output,
                *meta_args,
                offset_start=0,
                add_inputs=True,
            )
        else:
            lora_shrink(
                x,
                list(self.lora_A_slices),
                buffer,
                *meta_args,
                1.0,
            )
            lora_expand(
                buffer,
                list(self.lora_B_slices),
                output,
                *meta_args,
                offset_start=0,
                add_inputs=True,
            )

        # Row-parallel TP > 1: all-reduce the combined (base + LoRA) output.
        if self._row_parallel_reduce:
            output = tensor_model_parallel_all_reduce(output)

        return output, output_bias
