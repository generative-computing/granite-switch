# SPDX-License-Identifier: Apache-2.0
"""Decoder layer implementations for Granite Switch (vLLM).

This module provides the complete decoder layer hierarchy:
1. GraniteLoRAEmbeddedAttention - Attention with conditional LoRA
2. GraniteSwitchDecoderLayer - Attention decoder layer

The decoder layer uses upstream GraniteMoeSharedMLP for the MLP, with in-place
LoRA replacement on input_linear/output_linear projections.

These layers apply conditional LoRA adapters based on per-token adapter indices
from the switch. They use vLLM's Punica kernels for efficient LoRA computation
with torch.compile-friendly metadata preparation.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope

# Use switched LoRA implementation from core
from .lora import SwitchedLoRALinear

if TYPE_CHECKING:
    pass


class GraniteLoRAEmbeddedAttention(nn.Module):
    """Granite attention with conditional LoRA on QKV and output projections.

    Applies different LoRA adapters to Q, K, V, and O projections based on
    per-token adapter indices from the switch.
    """

    _lora_ctx = None  # Wired post-init by GraniteSwitchModel

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Extract adapter config from model config
        num_adapters = config.num_adapters
        max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0

        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config,
            "projection_head_dim",
            self.hidden_size // self.total_num_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.attention_multiplier

        # QKV projection - conditionally add LoRA based on config
        base_qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        if "qkv_proj" in config.lora_target_modules:
            self.qkv_proj = SwitchedLoRALinear(
                base_qkv_proj,
                num_adapters,
                max_lora_rank,
                num_slices=3,
                output_slices=tuple(base_qkv_proj.output_sizes),
            )
            self.has_qkv_lora = True
        else:
            self.qkv_proj = base_qkv_proj
            self.has_qkv_lora = False

        # Output projection - conditionally add LoRA based on config
        base_o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        if "o_proj" in config.lora_target_modules:
            self.o_proj = SwitchedLoRALinear(base_o_proj, num_adapters, max_lora_rank)
            self.has_o_lora = True
        else:
            self.o_proj = base_o_proj
            self.has_o_lora = False

        # Optional QK-norm (Qwen3): per-head RMSNorm on Q and K after
        # projection but before RoPE.  Gated by config.qk_norm (default False).
        self.qk_norm = getattr(config, "qk_norm", False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary embeddings (only for models with positional encoding)
        if getattr(config, "position_embedding_type", "rope") == "rope":
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
            )
        else:
            self.rotary_emb = None

        # Attention layer — head_dim is the native projection_head_dim.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # SwitchedLoRALinear reads LoRA metadata from the shared LoRAContext.
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-norm: per-head RMSNorm before RoPE (Qwen3).
        # vLLM RMSNorm expects [*, hidden_size], so reshape to per-head vectors.
        if self.qk_norm:
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


def rms_norm_select(
    norm: RMSNorm,
    block_output: torch.Tensor,
    residual: torch.Tensor | None,
    fused: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select between one-arg and two-arg RMSNorm calling conventions.

    Different vLLM model classes use different residual-add-norm patterns.
    Granite/GraniteMoeHybrid add the residual explicitly then call one-arg
    ``norm(x)``.  Llama/Mistral/Qwen2 call two-arg ``norm(x, residual)``
    which fuses the addition into a single CUDA kernel.

    Both produce mathematically identical results, but in bfloat16 the
    fused kernel rounds differently.  To preserve bit-exact equivalence
    between a skinned model and its original, we must match the convention
    used by the original model's vLLM class.

    Args:
        norm: vLLM RMSNorm layer.
        block_output: Output from attention/MLP block.
        residual: Running residual (None on the very first call).
        fused: True → two-arg fused kernel, False → separate add then norm.

    Returns:
        (hidden_states, residual)
    """
    if residual is None:
        # First layer: nothing to add yet.
        residual = block_output
        hidden_states = norm(block_output)
    elif fused:
        hidden_states, residual = norm(block_output, residual)
    else:
        residual = residual + block_output
        hidden_states = norm(residual)
    return hidden_states, residual


def replace_shared_mlp_projections_with_lora(mlp, config):
    """Replace shared MLP input_linear/output_linear with SwitchedLoRALinear in-place.

    Works on vLLM's GraniteMoeSharedMLP. Returns (has_input_lora, has_output_lora).
    """
    num_adapters = config.num_adapters
    max_lora_rank = max(config.adapter_ranks) if config.adapter_ranks else 0
    has_input_lora = False
    has_output_lora = False

    if "shared_input_linear" in config.lora_target_modules:
        base = mlp.input_linear
        mlp.input_linear = SwitchedLoRALinear(
            base,
            num_adapters,
            max_lora_rank,
            num_slices=2,
            output_slices=tuple(base.output_sizes),
        )
        has_input_lora = True

    if "shared_output_linear" in config.lora_target_modules:
        base = mlp.output_linear
        mlp.output_linear = SwitchedLoRALinear(
            base,
            num_adapters,
            max_lora_rank,
        )
        has_output_lora = True

    return has_input_lora, has_output_lora


class GraniteSwitchDecoderLayer(nn.Module):
    """Attention decoder layer with switch-determined adapter selection.

    Supports optional MoE (frozen) alongside shared_mlp when num_local_experts > 0.
    """

    _lora_ctx = None  # Wired post-init by GraniteSwitchModel

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.residual_multiplier = config.residual_multiplier
        self.fused_add_norm = getattr(config, "fused_add_norm", False)
        self.layer_type = "attention"

        self.self_attn = GraniteLoRAEmbeddedAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )

        # MLP section
        self.has_experts = getattr(config, "num_local_experts", 0) > 0
        if self.has_experts:
            from vllm.model_executor.models.granitemoehybrid import GraniteMoeMoE

            self.block_sparse_moe = GraniteMoeMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.block_sparse_moe",
            )

        from vllm.model_executor.models.granitemoehybrid import GraniteMoeSharedMLP

        self.shared_mlp = GraniteMoeSharedMLP(
            config=config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.shared_mlp",
        )
        self._has_shared_input_lora, self._has_shared_output_lora = (
            replace_shared_mlp_projections_with_lora(self.shared_mlp, config)
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Bit-exact compatibility: match the RMSNorm calling convention used
        # by the original model's vLLM class (see rms_norm_select docstring).
        hidden_states, residual = rms_norm_select(
            self.input_layernorm,
            hidden_states,
            residual,
            self.fused_add_norm,
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = hidden_states * self.residual_multiplier

        hidden_states, residual = rms_norm_select(
            self.post_attention_layernorm,
            hidden_states,
            residual,
            self.fused_add_norm,
        )

        if self.has_experts:
            moe_hidden_states = hidden_states.clone()
            moe_hidden_states = self.block_sparse_moe(moe_hidden_states)
            shared_output = self.shared_mlp(hidden_states)
            hidden_states = moe_hidden_states + shared_output
        else:
            hidden_states = self.shared_mlp(hidden_states)

        hidden_states = hidden_states * self.residual_multiplier
        return hidden_states, residual
