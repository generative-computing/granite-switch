# SPDX-License-Identifier: Apache-2.0
"""Torch.compile-friendly LoRA kernel metadata for multi-adapter inference.

This module provides CompileFriendlyLoRAKernelMeta, a reimplementation of vLLM's
LoRAKernelMeta that avoids data-dependent branching to enable torch.compile support.

Reference implementation:
https://github.com/vllm-project/vllm/blob/b9793e6a8c30bc42f35d2a1eac919284aea27f76/vllm/lora/ops/triton_ops/lora_kernel_metadata.py#L13

Key differences from vLLM's LoRAKernelMeta:
1. No data-dependent branching (if no_lora: return)
2. Uses torch.where() and masking instead of early returns
3. All operations are primitive PyTorch ops (torch.compile friendly)
4. Handles zero-adapter case without branching

Architecture:
    Input: adapter_indices [num_tokens] with values in Punica convention:
           -1 = no adapter (base model)
            0 = first adapter
            1 = second adapter
           ...

    Output: Metadata for Punica kernels:
        - token_lora_mapping: [num_tokens] which adapter each token uses
        - token_indices_sorted: [num_tokens] tokens sorted by adapter ID
        - num_tokens_per_lora: [num_adapters+1] token count per adapter
        - lora_token_start_loc: [num_adapters+2] cumulative start indices
        - active_lora_ids: [num_adapters+1] which adapters are active
        - no_lora_flag_cpu: [1] whether all tokens use base model

Usage:
    # Create metadata object (once)
    lora_meta = CompileFriendlyLoRAKernelMeta(
        num_adapters=4,
        device=torch.device("cuda:0"),
        dtype=torch.bfloat16
    )

    # In forward pass (can be inside torch.compile region)
    adapter_indices = ...  # [batch_size] in Punica convention
    meta_args = lora_meta.prepare_tensors(adapter_indices)

    # Use with Punica kernels
    lora_shrink(x, *meta_args, ...)
    lora_expand(output, *meta_args, ...)
"""

import torch
from torch import nn
from torch.library import Library, impl

# Custom op: copy a CUDA bool scalar into a CPU bool[1] mailbox.
_mailbox_lib = Library("compile_friendly_lora_meta", "DEF")
_mailbox_lib.define(
    "set_no_lora_flag_from_gpu_bool(Tensor no_lora_gpu, Tensor(a!) no_lora_flag_cpu) -> ()"
)


@impl(_mailbox_lib, "set_no_lora_flag_from_gpu_bool", "CompositeExplicitAutograd")
def _set_no_lora_flag_from_gpu_bool_impl(
    no_lora_gpu: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
) -> None:
    # no_lora_gpu: CUDA bool scalar (or 0-d/1-d size1)
    if not no_lora_gpu.is_cuda:
        raise RuntimeError("no_lora_gpu must be CUDA")
    if no_lora_gpu.dtype != torch.bool:
        raise RuntimeError("no_lora_gpu must be torch.bool")
    if no_lora_gpu.numel() != 1:
        raise RuntimeError("no_lora_gpu must have numel()==1")

    # no_lora_flag_cpu: CPU bool[1]
    if no_lora_flag_cpu.device.type != "cpu":
        raise RuntimeError("no_lora_flag_cpu must be CPU")
    if no_lora_flag_cpu.dtype != torch.bool:
        raise RuntimeError("no_lora_flag_cpu must be torch.bool")
    if no_lora_flag_cpu.numel() != 1:
        raise RuntimeError("no_lora_flag_cpu must have numel()==1")

    # Copy (this is the single GPU->CPU sync you pay once per forward)
    no_lora_flag_cpu.copy_(no_lora_gpu.to("cpu"))


@impl(_mailbox_lib, "set_no_lora_flag_from_gpu_bool", "Meta")
def _set_no_lora_flag_from_gpu_bool_meta(
    no_lora_gpu: torch.Tensor,
    no_lora_flag_cpu: torch.Tensor,
) -> None:
    # FakeTensor/Meta mode: can't compute values, just validate shape.
    if no_lora_flag_cpu.numel() != 1:
        raise RuntimeError("no_lora_flag_cpu must have numel()==1")
    return


set_no_lora_flag_from_gpu_bool = (
    torch.ops.compile_friendly_lora_meta.set_no_lora_flag_from_gpu_bool
)


# Namespace for this module's custom ops
_cfmeta_lib = Library("cf_lora_meta", "DEF")

# counts: [A+1] (int/long) on CUDA -> start_loc: [A+2] long on CUDA
_cfmeta_lib.define("counts_to_start_loc(Tensor counts) -> Tensor")


@impl(_cfmeta_lib, "counts_to_start_loc", "CompositeExplicitAutograd")
def _counts_to_start_loc_impl(counts: torch.Tensor) -> torch.Tensor:
    """
    counts: [A+1] integer tensor on CUDA (or CPU, but expected CUDA in your pipeline)
    returns: [A+2] int64 tensor on same device
    """
    if counts.numel() < 1:
        raise RuntimeError("counts must have numel() >= 1")

    device = counts.device
    # ensure integer type
    counts_i64 = counts.to(torch.int64)

    # Prefix sum using eager torch.cumsum (NOT Inductor-generated Triton scan)
    prefix = torch.cumsum(counts_i64, dim=0)  # [A+1]

    start_loc = torch.cat(
        [torch.zeros((1,), device=device, dtype=torch.int64), prefix],
        dim=0,
    )  # [A+2]
    return start_loc


@impl(_cfmeta_lib, "counts_to_start_loc", "Meta")
def _counts_to_start_loc_meta(counts: torch.Tensor) -> torch.Tensor:
    # FakeTensor/Meta mode: return correct shape/dtype/device, values unknown.
    a1 = counts.numel()
    return torch.empty((a1 + 1,), device=counts.device, dtype=torch.int64)


counts_to_start_loc = torch.ops.cf_lora_meta.counts_to_start_loc


class LoRAContext:
    """Shared per-forward LoRA metadata, following vLLM's PunicaWrapper pattern.

    A single instance is created at model level and wired to every
    SwitchedLoRALinear / GraniteLoRAEmbeddedAttention via ``_lora_ctx``.
    Written once per forward in GraniteSwitchModel; read by every layer.

    Attributes correspond 1-to-1 to the tuple returned by
    CompileFriendlyLoRAKernelMeta.prepare_tensors().
    """

    __slots__ = (
        "active_lora_ids",
        "lora_token_start_loc",
        "no_lora_flag_cpu",
        "num_active_loras",
        "num_tokens_per_lora",
        "query_group_suppression",
        "token_group_membership",
        "token_indices_sorted",
        "token_lora_mapping",
    )

    def __init__(self):
        self.token_lora_mapping: torch.Tensor | None = None
        self.token_indices_sorted: torch.Tensor | None = None
        self.num_tokens_per_lora: torch.Tensor | None = None
        self.lora_token_start_loc: torch.Tensor | None = None
        self.active_lora_ids: torch.Tensor | None = None
        self.no_lora_flag_cpu: torch.Tensor | None = None
        self.num_active_loras: torch.Tensor | None = None
        self.token_group_membership: torch.Tensor | None = None
        self.query_group_suppression: torch.Tensor | None = None

    def reset(self):
        """Clear all stored metadata (e.g. between forwards)."""
        self.token_lora_mapping = None
        self.token_indices_sorted = None
        self.num_tokens_per_lora = None
        self.lora_token_start_loc = None
        self.active_lora_ids = None
        self.no_lora_flag_cpu = None
        self.num_active_loras = None
        self.token_group_membership = None
        self.query_group_suppression = None


class CompileFriendlyLoRAKernelMeta(nn.Module):
    """Torch.compile-friendly LoRA kernel metadata preparation.

    This class prepares metadata for vLLM's Punica kernels (lora_shrink, lora_expand)
    without using data-dependent branching, making it compatible with torch.compile.

    The metadata tells Punica kernels:
    1. Which adapter each token uses (token_lora_mapping)
    2. What order to process tokens (token_indices_sorted)
    3. How many tokens use each adapter (num_tokens_per_lora)
    4. Where each adapter's tokens start (lora_token_start_loc)
    5. Which adapters are active (active_lora_ids)

    Attributes:
        num_adapters: Number of LoRA adapters
        device: Device to allocate tensors on
        dtype: Data type for weight tensors

    Buffers (pre-allocated):
        adapter_ids_punica: [num_adapters+1] adapter IDs in Punica convention
        no_lora_flag_cpu: [1] CPU tensor for no-lora flag (always False)
    """

    def __init__(
        self,
        num_adapters: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize metadata preparation.

        Args:
            num_adapters: Number of LoRA adapters (not counting base model)
            device: Device to allocate tensors on
            dtype: Data type for weight tensors (default: bfloat16)
        """
        super().__init__()
        self.num_adapters = num_adapters
        self.device = device
        self.dtype = dtype

        # Pre-allocate buffers that don't change between forward passes
        # Register as buffers so they move with the model
        self.register_buffer(
            "adapter_ids_punica",
            torch.arange(-1, num_adapters, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "no_lora_flag_cpu",
            torch.tensor([False], dtype=torch.bool, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "num_active_loras_cpu",
            torch.tensor([num_adapters + 1], dtype=torch.long, device="cpu"),
            persistent=False,
        )

    def prepare_tensors(
        self, adapter_indices: torch.Tensor
    ) -> tuple[
        torch.Tensor,  # token_lora_mapping
        torch.Tensor,  # token_indices_sorted
        torch.Tensor,  # num_tokens_per_lora
        torch.Tensor,  # lora_token_start_loc
        torch.Tensor,  # active_lora_ids
        torch.Tensor,  # no_lora_flag_cpu
        torch.Tensor,  # num_active_loras
    ]:
        """Prepare metadata for Punica kernels without data-dependent branching.

        This method is torch.compile-friendly because it:
        1. Uses only primitive PyTorch operations
        2. Has no data-dependent control flow (no if statements on tensor values)
        3. Always returns tensors with the same shape

        Args:
            adapter_indices: [num_tokens] tensor with adapter IDs in Punica convention:
                           -1 = no adapter (base model only)
                            0 = first adapter
                            1 = second adapter
                           ...

        Returns:
            Tuple of 7 tensors (compatible with vLLM's Punica kernels):
            1. token_lora_mapping: [num_tokens] adapter ID for each token
            2. token_indices_sorted: [num_tokens] token indices sorted by adapter ID
            3. num_tokens_per_lora: [num_adapters+1] count of tokens per adapter
            4. lora_token_start_loc: [num_adapters+2] cumulative start locations
            5. active_lora_ids: [num_adapters+1] adapter IDs (always all adapters)
            6. no_lora_flag_cpu: [1] on CPU, always False
            7. num_active_loras: [1] on CPU, number of active LoRAs

        Note:
            Unlike vLLM's LoRAKernelMeta, this always processes all adapters
            even if some have zero tokens. This trades some efficiency for
            torch.compile compatibility.
        """
        device = adapter_indices.device

        # 1. token_lora_mapping - just pass through (already in Punica convention)
        token_lora_mapping = adapter_indices  # [num_tokens]

        # 2. Count tokens per adapter using one-hot encoding + sum
        # Convert from Punica convention (-1, 0, 1, ...) to indices (0, 1, 2, ...)
        adapter_indices_offset = adapter_indices + 1  # [num_tokens]

        # One-hot encode and sum to get counts
        one_hot = torch.nn.functional.one_hot(
            adapter_indices_offset, num_classes=self.num_adapters + 1
        )  # [num_tokens, num_adapters+1]
        num_tokens_per_lora = one_hot.sum(dim=0)  # [num_adapters+1]

        # 3. Sort token indices by adapter ID
        # argsort is compile-friendly (no data-dependent branching)
        token_indices_sorted = torch.argsort(
            adapter_indices_offset, stable=True
        )  # [num_tokens]

        # 4. Compute cumulative start locations for each adapter
        # This tells kernels where each adapter's tokens begin
        lora_token_start_loc = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(num_tokens_per_lora, dim=0),
            ]
        )  # [num_adapters+2]
        #    lora_token_start_loc = counts_to_start_loc(num_tokens_per_lora)

        # 5. Active adapter IDs - in compile-friendly version, always all adapters
        # Note: Unlike vLLM's version which filters to only active adapters,
        # we always return all adapters for compile compatibility.
        # Punica kernels can handle zero-token adapters efficiently.
        active_lora_ids = self.adapter_ids_punica  # [num_adapters+1]

        #        # 6. No-lora flag (CPU mailbox), computed once per forward without Python branching.
        #        # no_lora_gpu is True iff there are zero adapter tokens (all tokens are base model).
        #        no_lora_gpu = (num_tokens_per_lora[1:].sum() == 0)  # CUDA bool scalar

        #       # Update the CPU mailbox via opaque custom op (single sync per forward).
        #       set_no_lora_flag_from_gpu_bool(no_lora_gpu, self.no_lora_flag_cpu)
        #       no_lora_flag_cpu = self.no_lora_flag_cpu

        # 6. No-lora flag - always False in this implementation
        # The kernels will handle the no-lora case based on token_lora_mapping
        no_lora_flag_cpu = self.no_lora_flag_cpu  # [1] on CPU

        # 7. Number of active LoRAs - always all adapters + base in this implementation
        num_active_loras = self.num_active_loras_cpu  # [1] on CPU

        return (
            token_lora_mapping,
            token_indices_sorted,
            num_tokens_per_lora,
            lora_token_start_loc,
            active_lora_ids,
            no_lora_flag_cpu,
            num_active_loras,
        )

    def prepare_and_store(
        self, adapter_indices: torch.Tensor, ctx: LoRAContext
    ) -> None:
        """Prepare metadata and store directly on the shared LoRAContext.

        This avoids returning a tuple that must be threaded through every
        forward signature, which is the root cause of torch.compile failures
        when the tuple is stored as a mutable attribute.

        Args:
            adapter_indices: [num_tokens] in Punica convention (-1 = base).
            ctx: Shared LoRAContext to populate.
        """
        result = self.prepare_tensors(adapter_indices)
        ctx.token_lora_mapping = result[0]
        ctx.token_indices_sorted = result[1]
        ctx.num_tokens_per_lora = result[2]
        ctx.lora_token_start_loc = result[3]
        ctx.active_lora_ids = result[4]
        ctx.no_lora_flag_cpu = result[5]
        ctx.num_active_loras = result[6]

    def meta_args(self) -> tuple[torch.Tensor, ...]:
        """Get cached metadata arguments.

        Note: This implementation doesn't cache metadata like vLLM's version.
        Each call to prepare_tensors() returns fresh metadata.

        This method is provided for API compatibility but should not be used.
        Instead, use prepare_tensors() directly and pass the result to kernels.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError(
            "CompileFriendlyLoRAKernelMeta doesn't cache metadata. "
            "Use prepare_tensors() directly instead of meta_args()."
        )

    def forward(self, adapter_indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass - same as prepare_tensors().

        This allows using the module in a nn.Sequential or as a regular module.

        Args:
            adapter_indices: [num_tokens] adapter IDs in Punica convention

        Returns:
            Tuple of 7 metadata tensors
        """
        return self.prepare_tensors(adapter_indices)


def create_lora_kernel_meta(
    num_adapters: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> CompileFriendlyLoRAKernelMeta:
    """Factory function to create LoRA kernel metadata preparation.

    Args:
        num_adapters: Number of LoRA adapters
        device: Device to allocate tensors on
        dtype: Data type for weight tensors

    Returns:
        CompileFriendlyLoRAKernelMeta instance
    """
    return CompileFriendlyLoRAKernelMeta(
        num_adapters=num_adapters, device=device, dtype=dtype
    )


# Example usage
if __name__ == "__main__":
    # Create metadata preparation module
    num_adapters = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lora_meta = CompileFriendlyLoRAKernelMeta(num_adapters=num_adapters, device=device)

    # Test with sample adapter indices (Punica convention)
    # -1 = no adapter, 0-3 = adapters
    adapter_indices = torch.tensor([-1, 0, -1, 1, 2, 0], device=device)

    # Prepare metadata (compile-friendly!)
    (
        token_lora_mapping,
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
    ) = lora_meta.prepare_tensors(adapter_indices)

    print("Example metadata preparation:")
    print(f"  Input adapter_indices: {adapter_indices.tolist()}")
    print(f"  token_lora_mapping: {token_lora_mapping.tolist()}")
    print(f"  token_indices_sorted: {token_indices_sorted.tolist()}")
    print(f"  num_tokens_per_lora: {num_tokens_per_lora.tolist()}")
    print(f"  lora_token_start_loc: {lora_token_start_loc.tolist()}")
    print(f"  active_lora_ids: {active_lora_ids.tolist()}")
    print(f"  no_lora_flag_cpu: {no_lora_flag_cpu.tolist()}")
    print(f"  num_active_loras: {num_active_loras.tolist()}")

    # Test with torch.compile (if available)
    if torch.__version__ >= "2.0.0":
        try:
            compiled_meta = torch.compile(lora_meta)
            result = compiled_meta(adapter_indices)
            print("\n✓ torch.compile successful!")
        except Exception as e:
            print(f"\n✗ torch.compile failed: {e}")
