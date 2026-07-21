# SPDX-License-Identifier: Apache-2.0
"""LoRA layer implementations for Granite Switch.

These layers apply conditional LoRA adapters based on per-token adapter indices.
They are used by the Router implementation to apply frozen LoRA adapters
selected by the trainable switch.
"""

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridMLP,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from granite_switch.config import GraniteSwitchConfig


class SwitchedLoRALinear(nn.Module):
    """Linear layer with switched LoRA adapters.

    This layer applies different LoRA adapters to different tokens based on
    per-token adapter indices. Used by Granite Switch to apply frozen LoRA adapters
    selected by the switch layer.

    Supports variable rank/alpha per adapter.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_adapters: Number of LoRA adapters
        max_lora_rank: Maximum rank across all adapters
        bias: Whether to include bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_adapters: int,
        max_lora_rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_adapters = num_adapters
        self.max_lora_rank = max_lora_rank

        # NOTE: lora_B weights are PRE-SCALED by (alpha/rank) during model loading.
        # No runtime scaling is needed - we use implicit scaling factor of 1.0.
        # This eliminates data-dependent branching and makes the code torch.compile compatible.
        # num_adapters and max_lora_rank are config metadata, not runtime parameters.

        # Base linear layer
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # Stacked LoRA weights: [num_adapters, 1, max_lora_rank, features]
        # Index 0 is base model (no LoRA), indices 1+ are adapters
        # Adapters with rank < max_lora_rank are zero-padded
        self.lora_A = nn.Parameter(
            torch.zeros(self.num_adapters, 1, self.max_lora_rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.num_adapters, 1, out_features, self.max_lora_rank)
        )

        # Initialize LoRA weights (will be loaded from pretrained adapters)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        # Context-passing support: stored adapter_indices for use in modules
        # (like mamba) where the caller can't pass adapter_indices explicitly.
        self._adapter_indices: torch.Tensor | None = None

    @property
    def weight(self):
        """Expose base layer weight for upstream module compatibility."""
        return self.base_layer.weight

    @property
    def bias(self):
        """Expose base layer bias for upstream module compatibility."""
        return self.base_layer.bias

    def forward(
        self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with conditional LoRA.

        Args:
            x: Input tensor [batch_size, seq_len, in_features] or [num_tokens, in_features]
            adapter_indices: Adapter index for each token [batch_size, seq_len] or [num_tokens]

        Returns:
            Output tensor with LoRA applied based on adapter_indices
        """
        # Fall back to stored adapter_indices (context-passing pattern)
        if adapter_indices is None:
            adapter_indices = self._adapter_indices

        # Base layer forward
        output = self.base_layer(x)

        # Early exit if all tokens use base model (adapter 0)
        if adapter_indices is None or not torch.any(adapter_indices > 0):
            return output

        # Flatten for token-level processing
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, self.in_features)
            adapter_indices_flat = adapter_indices.view(-1)
            output_flat = output.view(-1, self.out_features)
        else:
            x_flat = x
            adapter_indices_flat = adapter_indices
            output_flat = output

        # Apply LoRA for non-base tokens
        # Iterate over active adapters and apply LoRA to their tokens
        mask = adapter_indices_flat > 0
        if mask.any():
            # Get unique adapter indices that need LoRA
            active_adapters = adapter_indices_flat[mask].unique()

            for adapter_idx in active_adapters:
                # Find tokens using this adapter
                token_mask = adapter_indices_flat == adapter_idx
                token_indices = torch.where(token_mask)[0]

                if len(token_indices) == 0:
                    continue

                # Convert adapter_idx to tensor index (adapter 1 → tensor index 0, etc.)
                tensor_idx = adapter_idx - 1

                # Get LoRA matrices for this adapter
                # Shape: [1, rank, in_features] -> [rank, in_features]
                lora_a = self.lora_A[tensor_idx, 0]  # [rank, in_features]
                lora_b = self.lora_B[tensor_idx, 0]  # [out_features, rank]

                # Get tokens for this adapter
                x_adapter = x_flat[
                    token_indices
                ]  # [num_tokens_with_adapter, in_features]

                # LoRA computation: x @ A^T @ B_prescaled^T
                # lora_B is pre-scaled by (alpha/rank) during model loading
                # Shrink: x @ A^T -> [num_tokens, rank]
                lora_output = torch.matmul(x_adapter, lora_a.t())
                # Expand: buffer @ B_prescaled^T -> [num_tokens, out_features]
                lora_output = torch.matmul(lora_output, lora_b.t())

                # Add to output
                output_flat[token_indices] += lora_output

        # Reshape back if needed
        if len(original_shape) == 3:
            output = output_flat.view(batch_size, seq_len, self.out_features)
        else:
            output = output_flat

        return output


class MergedSwitchedLoRALinear(nn.Module):
    """Linear layer with merged switched LoRA adapters (matches vLLM structure).

    This layer implements fused projections (QKV, gate/up) with separate switched LoRA
    adapters for each slice. The structure EXACTLY matches vLLM's SwitchedLoRALinear
    with num_slices > 1, ensuring identical parameter names for checkpoint compatibility.

    Supports variable rank/alpha per adapter.

    Args:
        in_features: Input dimension (shared by all slices)
        output_slices: Tuple of output dimensions for each slice
                      e.g., (q_size, kv_size, kv_size) for QKV
                      e.g., (intermediate_size, intermediate_size) for gate/up
        num_adapters: Number of LoRA adapters
        max_lora_rank: Maximum rank across all adapters
        bias: Whether to include bias in base layer
    """

    def __init__(
        self,
        in_features: int,
        output_slices: tuple[int, ...],
        num_adapters: int,
        max_lora_rank: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.output_slices = output_slices
        self.num_slices = len(output_slices)
        self.num_adapters = num_adapters
        self.max_lora_rank = max_lora_rank

        # NOTE: lora_B weights are PRE-SCALED by (alpha/rank) during model loading.
        # No runtime scaling is needed - we use implicit scaling factor of 1.0.
        # This eliminates data-dependent branching and makes the code torch.compile compatible.
        # num_adapters and max_lora_rank are config metadata, not runtime parameters.

        # Fused base layer
        total_out_features = sum(output_slices)
        self.base_layer = nn.Linear(in_features, total_out_features, bias=bias)

        # LoRA slices stored as ParameterList (MATCHES vLLM STRUCTURE!)
        # This ensures parameter names are: module.lora_A_slices.0, module.lora_A_slices.1, etc.
        # Shape: [num_adapters, 1, max_lora_rank, features]
        self.lora_A_slices = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(self.num_adapters, 1, self.max_lora_rank, in_features)
                )
                for _ in range(self.num_slices)
            ]
        )
        self.lora_B_slices = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(self.num_adapters, 1, output_size, self.max_lora_rank)
                )
                for output_size in output_slices
            ]
        )

        # Initialize LoRA weights
        for lora_A in self.lora_A_slices:
            nn.init.kaiming_uniform_(lora_A, a=5**0.5)
        for lora_B in self.lora_B_slices:
            nn.init.zeros_(lora_B)

        # Context-passing support: stored adapter_indices for use in modules
        # (like shared_mlp) where the caller can't pass adapter_indices explicitly.
        self._adapter_indices: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward with packed LoRA applied to each slice.

        Args:
            x: Input [batch, seq_len, in_features] or [num_tokens, in_features]
            adapter_indices: Adapter index per token [batch, seq_len] or [num_tokens]

        Returns:
            Output with LoRA applied: [batch, seq_len, total_out_features]
        """
        # Fall back to stored adapter_indices (context-passing pattern)
        if adapter_indices is None:
            adapter_indices = self._adapter_indices

        # Base forward (fused)
        output = self.base_layer(x)

        # Early exit if all tokens use base model (adapter 0)
        if adapter_indices is None or not torch.any(adapter_indices > 0):
            return output

        # Flatten for token-level processing
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, self.in_features)
            adapter_indices_flat = adapter_indices.view(-1)
            output_flat = output.view(-1, sum(self.output_slices))
        else:
            x_flat = x
            adapter_indices_flat = adapter_indices
            output_flat = output

        # Apply LoRA for each slice separately
        offset = 0
        for slice_idx, output_size in enumerate(self.output_slices):
            # Get LoRA weights for this slice
            lora_A = self.lora_A_slices[
                slice_idx
            ]  # [num_adapters, 1, rank, in_features]
            lora_B = self.lora_B_slices[
                slice_idx
            ]  # [num_adapters, 1, output_size, rank]

            # Apply LoRA to this slice's output region
            output_slice = output_flat[:, offset : offset + output_size]
            output_slice = self._apply_lora_to_slice(
                x_flat, output_slice, adapter_indices_flat, lora_A, lora_B
            )
            output_flat[:, offset : offset + output_size] = output_slice

            offset += output_size

        # Reshape back
        if len(original_shape) == 3:
            output = output_flat.view(batch_size, seq_len, sum(self.output_slices))
        else:
            output = output_flat

        return output

    def _apply_lora_to_slice(
        self,
        x: torch.Tensor,
        output_slice: torch.Tensor,
        adapter_indices: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> torch.Tensor:
        """Apply LoRA to a single slice."""
        mask = adapter_indices > 0
        if not mask.any():
            return output_slice

        active_adapters = adapter_indices[mask].unique()

        for adapter_idx in active_adapters:
            token_mask = adapter_indices == adapter_idx
            token_indices = torch.where(token_mask)[0]

            if len(token_indices) == 0:
                continue

            # Convert adapter_idx to tensor index (adapter 1 → tensor index 0, etc.)
            tensor_idx = adapter_idx - 1

            # Get LoRA matrices for this adapter
            lora_a = lora_A[tensor_idx, 0]  # [rank, in_features]
            lora_b = lora_B[tensor_idx, 0]  # [output_size, rank]

            # Get tokens for this adapter
            x_adapter = x[token_indices]

            # LoRA computation: x @ A^T @ B_prescaled^T
            # lora_B is pre-scaled by (alpha/rank) during model loading
            lora_output = torch.matmul(x_adapter, lora_a.t())
            lora_output = torch.matmul(lora_output, lora_b.t())

            # Add to output
            output_slice[token_indices] += lora_output

        return output_slice


class GraniteLoRAEmbeddedAttention(nn.Module):
    """Multi-head attention with conditional LoRA adapters.

    Applies different LoRA adapters to Q, K, V, and O projections based on
    per-token adapter indices from the switch.

    Uses fused QKV projection (matches vLLM structure) for performance.
    """

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config,
            "projection_head_dim",
            self.hidden_size // self.num_heads,
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout
        self.scaling = config.attention_multiplier
        self.is_causal = True  # Required by attention backends

        # Optional QK-norm (Qwen3): per-head RMSNorm on Q and K after
        # projection but before RoPE.  Gated by config.qk_norm (default False).
        self.qk_norm = getattr(config, "qk_norm", False)
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fused QKV projection - conditionally add LoRA based on config
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim

        if "qkv_proj" in config.lora_target_modules:
            # QKV with merged switched LoRA (matches vLLM!)
            num_adapters = config.num_adapters
            max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0
            self.qkv_proj = MergedSwitchedLoRALinear(
                self.hidden_size,
                output_slices=(q_size, kv_size, kv_size),  # Q, K, V
                num_adapters=num_adapters,
                max_lora_rank=max_lora_rank,
                bias=config.attention_bias,
            )
            self.has_qkv_lora = True
        else:
            # No LoRA adapters for QKV - use plain linear layer
            self.qkv_proj = nn.Linear(
                self.hidden_size,
                q_size + 2 * kv_size,  # Q + K + V
                bias=config.attention_bias,
            )
            self.has_qkv_lora = False

        # Output projection - conditionally add LoRA based on config
        if "o_proj" in config.lora_target_modules:
            # O projection with switched LoRA
            num_adapters = config.num_adapters
            max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0
            self.o_proj = SwitchedLoRALinear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                num_adapters,
                max_lora_rank,
                bias=config.attention_bias,
            )
            self.has_o_lora = True
        else:
            # No LoRA adapters for O - use plain linear layer
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=config.attention_bias,
            )
            self.has_o_lora = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter_indices: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        """Forward pass with LoRA and modern Cache API support.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            adapter_indices: Per-token adapter selection [batch, seq_len]
            position_embeddings: Precomputed (cos, sin) for RoPE
            attention_mask: Attention mask
            past_key_values: Cache object for KV caching
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated cache
            cache_position: Token positions for cache indexing

        Returns:
            Tuple of (attention_output, attention_weights, cache)
        """
        bsz, q_len, _ = hidden_states.size()

        # Fused QKV projection - conditionally use LoRA
        if self.has_qkv_lora:
            qkv = self.qkv_proj(hidden_states, adapter_indices)
        else:
            qkv = self.qkv_proj(hidden_states)

        # Split Q, K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        query_states, key_states, value_states = qkv.split(
            [q_size, kv_size, kv_size], dim=-1
        )

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        # QK-norm: per-head RMSNorm before RoPE (Qwen3).
        # Tensors are [batch, seq, heads, head_dim]; RMSNorm normalizes last dim.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply rotary embeddings (precomputed at model level) when present.
        # position_embeddings is None when position_embedding_type == "nope".
        cos, sin = (
            position_embeddings if position_embeddings is not None else (None, None)
        )
        if position_embeddings is not None:
            # Transpose for RoPE: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
            query_states_t = query_states.transpose(1, 2)
            key_states_t = key_states.transpose(1, 2)
            query_states_t, key_states_t = apply_rotary_pos_emb(
                query_states_t, key_states_t, cos, sin
            )
            # Transpose back: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
            query_states = query_states_t.transpose(1, 2)
            key_states = key_states_t.transpose(1, 2)

        # Belief that both cache and attention expect [batch, heads, seq, dim]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states = query_states.transpose(1, 2)

        # KV cache handling with modern Cache API
        if use_cache and past_key_values is not None:
            # Cache will internally handle concatenation with past key/values
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Call HuggingFace attention backend
        # This gets us FlashAttention, SDPA, FlexAttention, etc. for free
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),
        )

        # Reshape and project output - conditionally use LoRA
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.has_o_lora:
            attn_output = self.o_proj(attn_output, adapter_indices)
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # Return Cache object if use_cache, else None
        # Cache object is updated in-place by cache.update() above
        return attn_output, attn_weights, past_key_values if use_cache else None


def replace_shared_mlp_projections_with_lora(
    mlp: "GraniteMoeHybridMLP",
    config: "GraniteSwitchConfig",
) -> tuple:
    """Replace shared MLP input_linear/output_linear with LoRA in-place.

    Returns (has_input_lora, has_output_lora) flags for context-passing.
    """
    num_adapters = config.num_adapters
    max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0
    has_input_lora = False
    has_output_lora = False

    if "shared_input_linear" in config.lora_target_modules:
        old = mlp.input_linear
        mlp.input_linear = MergedSwitchedLoRALinear(
            old.in_features,
            output_slices=(
                config.shared_intermediate_size,
                config.shared_intermediate_size,
            ),
            num_adapters=num_adapters,
            max_lora_rank=max_lora_rank,
            bias=old.bias is not None,
        )
        has_input_lora = True

    if "shared_output_linear" in config.lora_target_modules:
        old = mlp.output_linear
        mlp.output_linear = SwitchedLoRALinear(
            old.in_features,
            old.out_features,
            num_adapters,
            max_lora_rank,
            bias=old.bias is not None,
        )
        has_output_lora = True

    return has_input_lora, has_output_lora
