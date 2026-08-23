# SPDX-License-Identifier: Apache-2.0
"""Shadow-Residual decoder + attention for the vLLM backend (SWITCH kernel).

Dual-stream design, streams stacked on the token dim (``hs`` is ``[2M, H]``,
base rows ``[:M]`` ++ adapter rows ``[M:]``):

* every projection (qkv / o / gate-up / down) is a :class:`SwitchedLoRALinear`
  run ONCE over the ``[2M, H]`` stack. The base half carries kernel-local id 0
  (via the 2M metadata on the shared :class:`~.kernel_meta.SRLoRAContext`), so it
  gets no delta — a pristine base projection and a base-only K/V. The adapter
  half carries the real id and gets the adapter delta.
* attention is ONE ``Attention(2*num_heads, num_kv_heads)`` call with the base and
  adapter query heads interleaved (even=base, odd=adapter) against the single
  vanilla-sized base-only K/V (from the base half).
* a per-layer :class:`~.wcross_shunt.WCrossShunt` injects ``base -> adapter`` at
  end of layer, keyed by the M-length REAL ids.

Fused switch layout (fused ``qkv_proj`` + fused gate/up with in-kernel SwiGLU) —
the natural SWITCH layout and the perf path. K/V LoRA slices of the fused qkv are
left zero (SR does not adapt K/V), which the loader marks as loaded. TP=1 for v1.
"""

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
from vllm.model_executor.models.granitemoehybrid import GraniteMoeSharedMLP

from granite_switch.vllm.core.lora import SwitchedLoRALinear

from ._sr_ops import deinterleave_heads, interleave_q_heads
from .wcross_shunt import WCrossShunt


def _max_lora_rank(config) -> int:
    return max(config.adapter_ranks) if getattr(config, "adapter_ranks", None) else 0


class ShadowResidualAttention(nn.Module):
    """Doubled-Q / base-only-KV attention for one SR decoder layer (SWITCH kernel)."""

    _lora_ctx = None  # wired post-init by the model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        tp_size = get_tensor_model_parallel_world_size()

        num_adapters = config.num_adapters
        max_lora_rank = _max_lora_rank(config)

        self.hidden_size = config.hidden_size
        # Total vs per-rank (local) head geometry under tensor parallelism, mirroring
        # GraniteLoRAEmbeddedAttention: the parallel linears are built with TOTAL head
        # counts (they shard themselves internally); the doubled-Q attention glue and
        # the qkv split operate on LOCAL counts. SR's stacked [2M, H] streams live on
        # the token axis, orthogonal to TP's head sharding, so both halves shard
        # identically and the base/adapter -> KV mapping is preserved per rank. Both the
        # divisible-KV and replicated-KV (num_key_value_heads < tp_size) regimes are
        # supported; see the doubled-Q head-mapping unit test.
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0, (
            f"num_attention_heads={self.total_num_heads} not divisible by tp_size={tp_size}"
        )
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"num_key_value_heads={self.total_num_kv_heads} not divisible by "
                f"tp_size={tp_size}"
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"tp_size={tp_size} not a multiple of "
                f"num_key_value_heads={self.total_num_kv_heads}"
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "projection_head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.total_q_size = self.total_num_heads * self.head_dim
        self.scaling = config.attention_multiplier
        bias = getattr(config, "attention_bias", False)

        base_qkv = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.qkv_proj = SwitchedLoRALinear(
            base_qkv,
            num_adapters,
            max_lora_rank,
            num_slices=3,
            output_slices=tuple(base_qkv.output_sizes),
        )

        base_o = RowParallelLinear(
            self.total_q_size,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.o_proj = SwitchedLoRALinear(base_o, num_adapters, max_lora_rank)

        self.qk_norm = getattr(config, "qk_norm", False)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if getattr(config, "position_embedding_type", "rope") == "rope":
            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                rope_parameters=config.rope_parameters,
            )
        else:
            self.rotary_emb = None

        # Doubled query heads (base + adapter interleaved) against base-only K/V.
        self.attn = Attention(
            2 * self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
        """``normed``: pre-normed stacked stream ``[2M, H]``; returns ``[2M, H]``."""
        m = normed.shape[0] // 2

        # One fused qkv GEMM+expand over the 2M stack. Base half (id 0) -> base
        # projection; adapter half -> adapted Q (K/V LoRA slices are zero, so the
        # adapter half's K/V equal base K/V — and we take the base half anyway).
        qkv, _ = self.qkv_proj(normed)
        q_2m, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        k_base, v_base = k[:m], v[:m]  # base-only K/V

        if self.qk_norm:
            q_2m = self.q_norm(q_2m.reshape(-1, self.head_dim)).reshape(q_2m.shape)
            k_base = self.k_norm(k_base.reshape(-1, self.head_dim)).reshape(
                k_base.shape
            )

        k_base, v_base = k_base.contiguous(), v_base.contiguous()
        q_dbl = interleave_q_heads(
            q_2m[:m], q_2m[m:], self.num_heads, self.head_dim
        )  # [M, 2*q_size]

        if self.rotary_emb is not None:
            q_dbl, k_base = self.rotary_emb(positions, q_dbl, k_base)

        attn = self.attn(q_dbl, k_base, v_base)  # ONE call, base-only KV
        attn_base, attn_adapt = deinterleave_heads(attn, self.num_heads, self.head_dim)
        attn_stacked = torch.cat([attn_base, attn_adapt], dim=0)  # [2M, q_size]

        o, _ = self.o_proj(attn_stacked)  # base + adapter-only O delta
        return o


class ShadowResidualDecoderLayer(nn.Module):
    """One SR decoder layer (dual-stream, stacked on the token dim, SWITCH kernel).

    Granite non-fused residual convention (materialized): each block runs on
    ``norm(hs)`` and adds ``block * residual_multiplier`` back to ``hs``. The base
    half never receives a delta or shunt, so it stays base-equivalent.
    """

    _lora_ctx = None  # wired post-init by the model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        num_adapters = config.num_adapters
        cross_rank = int(getattr(config, "cross_stream_rank", 0) or 0)
        hidden = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        self.layer_type = "attention"

        self.self_attn = ShadowResidualAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )

        # SR serves dense bases only: this dual-stream decoder builds just the
        # shared MLP and has no routed-expert path, so a MoE base would silently
        # drop every expert. Fail loudly instead of computing wrong output.
        if getattr(config, "num_local_experts", 0) > 0:
            raise NotImplementedError(
                "Shadow-Residual vLLM decoding supports dense bases only; "
                f"num_local_experts={config.num_local_experts} (MoE) is not "
                "supported."
            )

        # Fused shared MLP (gate|up with in-kernel SwiGLU, + down), each wrapped in
        # SwitchedLoRALinear. Runs over the [2M, H] stack; base half gets no delta.
        # Wrapped UNCONDITIONALLY (not gated on config.lora_target_modules, unlike
        # the base helper): an SR checkpoint with MLP LoRA always has a home, and
        # one without just leaves the zero tiers (no delta) — either way correct.
        self.shared_mlp = GraniteMoeSharedMLP(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_mlp",
        )
        max_lora_rank = _max_lora_rank(config)
        base_in = self.shared_mlp.input_linear
        self.shared_mlp.input_linear = SwitchedLoRALinear(
            base_in,
            num_adapters,
            max_lora_rank,
            num_slices=2,
            output_slices=tuple(base_in.output_sizes),
            fuse_swiglu=True,
        )
        # gate/up now applies SwiGLU in-kernel and returns the activated [2M, H];
        # the MLP's own activation becomes a pass-through.
        self.shared_mlp.act_fn = nn.Identity()
        base_out = self.shared_mlp.output_linear
        self.shared_mlp.output_linear = SwitchedLoRALinear(
            base_out, num_adapters, max_lora_rank
        )

        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

        # Cross-stream shunt (base -> adapter), shrink-only kernel, keyed by real ids.
        self.cross_stream = WCrossShunt(
            hidden_size=hidden,
            num_adapters=num_adapters,
            cross_rank=cross_rank,
            device=vllm_config.device_config.device,
            dtype=vllm_config.model_config.dtype,
        )

    def forward(self, positions: torch.Tensor, hs: torch.Tensor) -> torch.Tensor:
        """``hs``: stacked ``[2M, H]`` (base ++ adapter)."""
        m = hs.shape[0] // 2

        normed = self.input_layernorm(hs)
        o = self.self_attn(positions, normed)
        hs = hs + o * self.residual_multiplier

        normed = self.post_attention_layernorm(hs)
        mlp_out = self.shared_mlp(
            normed
        )  # SwitchedLoRALinear inside; base half no delta
        hs = hs + mlp_out * self.residual_multiplier

        # base -> adapter injection over the M base-half rows (adapter-active only).
        cs = self.cross_stream(hs[:m])  # [M, H]
        hs = torch.cat([hs[:m], hs[m:] + cs], dim=0)
        return hs


__all__ = ["ShadowResidualAttention", "ShadowResidualDecoderLayer"]
