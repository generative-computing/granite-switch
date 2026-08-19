# SPDX-License-Identifier: Apache-2.0
"""SingleSwitch using replicated one-hot attention for adapter selection.

This switch uses the backbone's full head geometry (num_attention_heads,
num_key_value_heads, projection_head_dim, attention_multiplier) so that all
attention layers share one FlashAttentionMetadataBuilder configuration.

The same one-hot dim-0 pattern is replicated identically across every head:
- Control tokens: key[:,0]=+effective_gain, query[:,0]=1, value[:,0]=adapter_id
- Other tokens: key[:,0]=-effective_gain, query[:,0]=1, value[:,0]=0

Gain compensation: effective_gain = control_token_gain / attention_multiplier,
so the final logit = attention_multiplier × 1.0 × effective_gain = control_token_gain.

Under TP, each rank constructs its local Q/K/V independently with the same
one-hot pattern — no all-reduce or broadcast needed.
"""

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.attention import Attention


class SingleSwitch(nn.Module):
    """Replicated one-hot attention-based switch for adapter selection.

    Uses the backbone's full head geometry with replicated one-hot dim-0
    pattern across every head. Gain compensation in K preserves softmax
    sharpness under the backbone's attention_multiplier scaling.

    Args:
        num_adapters: Number of LoRA adapters
        vllm_config: vLLM configuration for Attention layer
        control_token_gain: Desired attention logit magnitude (default: 15)
        switch_head_dim: Fallback head_dim for standalone/test mode
        config: GraniteSwitchConfig (provides backbone head geometry)
    """

    def __init__(
        self,
        num_adapters: int,
        vllm_config: VllmConfig | None = None,
        control_token_gain: float = 15.0,
        switch_head_dim: int = 32,
        config=None,
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.control_token_gain = control_token_gain

        if vllm_config.model_config is not None:
            self.dtype = vllm_config.model_config.dtype
        else:
            self.dtype = torch.get_default_dtype()

        if config is not None and hasattr(config, "num_attention_heads"):
            tp_size = get_tensor_model_parallel_world_size()
            self.num_heads = config.num_attention_heads // tp_size
            total_kv = config.num_key_value_heads
            if total_kv >= tp_size:
                self.num_kv_heads = total_kv // tp_size
            else:
                self.num_kv_heads = max(1, total_kv // tp_size)
            self.head_dim = config.projection_head_dim
            self.scaling = config.attention_multiplier
            self.effective_gain = control_token_gain / self.scaling
        else:
            self.num_heads = 1
            self.num_kv_heads = 1
            if switch_head_dim < 32:
                raise ValueError(
                    f"switch_head_dim must be >= 32 for FlashAttention compatibility, got {switch_head_dim}"
                )
            self.head_dim = switch_head_dim
            self.scaling = 1.0
            self.effective_gain = control_token_gain

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix="switch.layers.0",
        )

        # control_to_substitute_lut: [vocab_size_or_higher], -1 at non-control
        # ids and the substitute id at each control slot. The switch performs
        # the runtime token-exchange: it rewrites input_ids in-place so that
        # control-token positions carry the substitute id by the time the
        # decoder embeds them. The decoder is then oblivious — it just calls
        # get_input_embeddings(input_ids) and gets the right result by
        # construction.
        if (
            config is not None
            and getattr(config, "adapter_token_ids", None) is not None
            and getattr(config, "adapter_substitute_token_ids", None) is not None
        ):
            ctrl_ids = config.adapter_token_ids
            sub_ids = config.adapter_substitute_token_ids
            max_ctrl_id = max(ctrl_ids)
            lut_size = max(getattr(config, "vocab_size", 0), max_ctrl_id + 1)
            lut = torch.full((lut_size,), -1, dtype=torch.long)
            for ctrl_id, sub_id in zip(ctrl_ids, sub_ids):
                lut[ctrl_id] = sub_id
            self.register_buffer("control_to_substitute_lut", lut)
        else:
            self.control_to_substitute_lut = None

    @property
    def num_cache_layers(self) -> int:
        """Number of KV cache slots used by this switch (1 Attention layer)."""
        return 1

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adapter indices and rewrite control tokens via the LUT.

        See the HF SingleSwitch docstring for the full rationale. In short:
        the switch performs both adapter selection and token-exchange
        rewrite, so the decoder is agnostic to substitution.

        Args:
            input_ids: Input token IDs [total_tokens] - flattened by vLLM scheduler
            adapter_token_ids: Activating control token IDs [num_adapters]
                              Single token per adapter (no base token slot):
                              - adapter_token_ids[i] = token to activate adapter i+1
                              Output 0 = base (implicit default). SingleSwitch has no mechanism
                              to transition back to base mid-sequence.

        Returns:
            (adapter_indices, modified_input_ids):
              adapter_indices:    [total_tokens] where 0 = base, 1+ = adapters.
              modified_input_ids: [total_tokens] with each control-token id
                                  replaced by its substitute id.
        """
        total_tokens = input_ids.shape[0]
        device = input_ids.device

        # ==================================================================
        # Prepare Q, K, V tensors — replicated one-hot across all heads
        # ==================================================================
        # Every head gets the same one-hot dim-0 pattern. Under TP, each
        # rank's local heads independently compute the correct result.

        q = torch.zeros(
            (total_tokens, self.num_heads, self.head_dim),
            device=device,
            dtype=self.dtype,
        )
        q[:, :, 0] = 1.0

        # Vectorized adapter token matching
        matches = input_ids.unsqueeze(1) == adapter_token_ids.unsqueeze(0)
        is_control = matches.any(dim=1)

        adapter_ids = torch.where(
            is_control,
            matches.long().argmax(dim=1) + 1,
            torch.zeros_like(input_ids, dtype=torch.long),
        )

        # Keys dim 0: ±effective_gain (compensated for attention_multiplier)
        gain_sign = (2.0 * is_control.to(self.dtype) - 1.0) * self.effective_gain

        k = torch.zeros(
            (total_tokens, self.num_kv_heads, self.head_dim),
            device=device,
            dtype=self.dtype,
        )
        k[:, :, 0] = gain_sign.unsqueeze(1)

        v = torch.zeros(
            (total_tokens, self.num_kv_heads, self.head_dim),
            device=device,
            dtype=self.dtype,
        )
        v[:, :, 0] = adapter_ids.to(self.dtype).unsqueeze(1)

        # ==================================================================
        # Call vLLM Attention kernel
        # ==================================================================
        # Output shape: [total_tokens, num_heads * head_dim]
        attn_output = self.attn(q, k, v)

        # ==================================================================
        # Compute adapter indices
        # ==================================================================
        # Head 0, dim 0 is at index 0 of the flattened output.
        # All heads produce the same result; we read from head 0.
        attn_output = attn_output[:, 0]

        # Round to get integer adapter indices
        adapter_indices = torch.round(attn_output).long()

        # Clamp to valid range [0, num_adapters]
        adapter_indices = torch.clamp(adapter_indices, 0, self.num_adapters)

        # Token-exchange rewrite: see the HF switch for the rationale.
        # Skipped only when no LUT was built (no substitute ids configured).
        # No data-dependent gate here — the surrounding decoder is wrapped in
        # @support_torch_compile, which forbids `tensor.any()` branching.
        # `torch.where` runs every step; the cost is one indexed gather and
        # one elementwise select on the flat input.
        modified_input_ids = self.apply_token_exchange(input_ids)

        return adapter_indices, modified_input_ids

    def apply_token_exchange(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Rewrite control-token ids to their substitute ids (pure LUT lookup).

        This is the embedding-side half of the switch — no attention, no KV —
        so it can be reused by the model's ``embed_input_ids`` on the multimodal
        path, where vLLM precomputes ``inputs_embeds`` before ``forward`` runs.
        Returns ``input_ids`` unchanged when no substitute LUT was built.
        """
        if self.control_to_substitute_lut is None:
            return input_ids
        sub_id_per_pos = self.control_to_substitute_lut[input_ids]
        is_control = sub_id_per_pos >= 0
        return torch.where(is_control, sub_id_per_pos, input_ids)
