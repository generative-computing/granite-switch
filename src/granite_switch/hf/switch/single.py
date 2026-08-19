# SPDX-License-Identifier: Apache-2.0
"""SingleSwitch using single-head attention for adapter selection.

This switch uses a single-head attention mechanism with a single active
dimension (dim 0) inside a head_dim-wide vector:
- Control tokens: key[0]=+gain, query[0]=1, value[0]=adapter_id
- Other tokens: key[0]=-gain, query[0]=1, value[0]=0

Uses HuggingFace's attention backends (FlashAttention, SDPA, etc.) for
efficient computation, matching the pattern used in GraniteLoRAEmbeddedAttention.

Uses the modern HuggingFace Cache API for KV caching (required for incremental decoding).
"""

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.granite.modeling_granite import eager_attention_forward


def build_control_to_substitute_lut(config) -> torch.Tensor | None:
    """Derive the control->substitute lookup table from *config*.

    Shape ``[max(vocab_size, max_ctrl_id + 1)]``: ``-1`` at every non-control id
    and the substitute id at each control slot. The ``max`` keeps every control
    id addressable even when ``vocab_size`` lags the tokenizer.

    Returns ``None`` when *config* carries no token-exchange mapping, in which
    case the switch leaves ``input_ids`` untouched.

    Single source of truth for the sizing rule: the table is a pure function of
    ``vocab_size``, ``adapter_token_ids`` and ``adapter_substitute_token_ids``,
    so anything that changes those must re-derive it (see
    :meth:`SingleSwitch.rebuild_control_to_substitute_lut`).
    """
    if config is None:
        return None
    ctrl_ids = getattr(config, "adapter_token_ids", None)
    sub_ids = getattr(config, "adapter_substitute_token_ids", None)
    if not ctrl_ids or not sub_ids:
        return None

    lut_size = max(getattr(config, "vocab_size", 0), max(ctrl_ids) + 1)
    lut = torch.full((lut_size,), -1, dtype=torch.long)
    for ctrl_id, sub_id in zip(ctrl_ids, sub_ids):
        lut[ctrl_id] = sub_id
    return lut


class SingleSwitch(nn.Module):
    """Single-head attention-based switch for adapter selection.

    Uses a single attention head with a single active dimension (dim 0)
    inside a head_dim-wide vector:
    - Control tokens: k[0]=+gain, q[0]=1, v[0]=adapter_id
    - Other tokens: k[0]=-gain, q[0]=1, v[0]=0

    The dot product Q·K is exactly ±gain regardless of head_dim, matching vLLM.

    This computes cumulative sums via causal attention over control tokens.
    Uses HuggingFace's attention backends (same as GraniteLoRAEmbeddedAttention)
    to get FlashAttention, SDPA, etc. for free.

    Uses the modern Cache API for KV caching (required for incremental decoding).
    The switch is assigned layer_idx=-1 to differentiate it from decoder layers.

    Args:
        num_adapters: Number of LoRA adapters
        config: Model configuration (for attention backend selection)
        control_token_gain: Attention gain for control/non-control token separation (default: 15)
        switch_head_dim: Head dimension for switch attention (default: from GraniteSwitchConfig)
    """

    def __init__(
        self,
        num_adapters: int,
        config=None,
        control_token_gain: float = 15.0,
        switch_head_dim: int = 32,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.control_token_gain = control_token_gain
        self.config = config

        # Align with the decoder's native head_dim. (Under token exchange the
        # KV cache no longer carries any expansion, so this is just the
        # base-model projection_head_dim.)
        if config is not None:
            self.head_dim = getattr(
                config,
                "projection_head_dim",
                config.hidden_size // config.num_attention_heads,
            )
        else:
            self.head_dim = switch_head_dim

        self.num_heads = 1
        self.num_key_value_heads = 1
        self.num_key_value_groups = (
            self.num_heads // self.num_key_value_heads
        )  # Should be 1
        self.scaling = 1.0  # No scaling needed for cumsum attention

        # For attention backend compatibility
        self.attention_dropout = 0.0
        self.is_causal = True

        # Layer index for cache - assigned by the model
        # Switch is layer 0, decoder layers are 1 to num_hidden_layers
        self.layer_idx = layer_idx

        # control_to_substitute_lut: [vocab_size_or_higher], -1 at non-control
        # ids and the substitute id at each control slot. The switch performs
        # the runtime token-exchange: it rewrites input_ids in-place so that
        # control-token positions carry the substitute id by the time the
        # decoder embeds them. The decoder is then oblivious — it just calls
        # embed_tokens(input_ids) and gets the right result by construction.
        lut = build_control_to_substitute_lut(config)
        if lut is not None:
            self.register_buffer("control_to_substitute_lut", lut)
        else:
            self.control_to_substitute_lut = None

    def rebuild_control_to_substitute_lut(self, config=None) -> bool:
        """Re-derive the control->substitute table after a vocabulary change.

        ``__init__`` sizes the table from ``config.vocab_size``, so anything that
        grows the vocabulary afterwards — notably
        ``resize_token_embeddings`` when compose adds control and marker tokens —
        leaves the buffer shorter than the config it will be saved alongside.

        That matters because the buffer is persistent. On ``from_pretrained``,
        a stored tensor whose shape disagrees with the freshly-constructed one is
        discarded and the buffer is left as uninitialised memory (there is no
        ``_init_weights`` rule for it), so every id reads as a control id and the
        rewrite sends out-of-range ids into the embedding gather. Re-derive the
        table before saving so the checkpoint and its config agree.

        Returns ``True`` if a table was rebuilt, ``False`` if this switch has no
        token-exchange mapping to rebuild.
        """
        lut = build_control_to_substitute_lut(
            config if config is not None else self.config
        )
        if lut is None:
            return False

        existing = getattr(self, "control_to_substitute_lut", None)
        if existing is not None:
            lut = lut.to(device=existing.device)
            self.control_to_substitute_lut = lut
        else:
            self.register_buffer("control_to_substitute_lut", lut)
        return True

    @property
    def num_cache_layers(self) -> int:
        """Number of cache slots used."""
        return 1

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter_token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adapter indices and rewrite control tokens via the LUT.

        The switch performs both halves of token-exchange:
          1. Adapter selection: read input_ids, detect control tokens via
             input_ids == adapter_token_ids, emit per-token adapter_indices.
          2. Token rewrite: replace each control token's id in input_ids
             with its substitute id (from control_to_substitute_lut).

        Returning the rewritten input_ids means the decoder is oblivious to
        the swap — it simply embeds whatever it's given. There's no
        decoder-side LUT, no per-forward scatter, no clone-guard.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            adapter_token_ids: Activating control token IDs [num_adapters]
                              Single token per adapter (no base token slot):
                              - adapter_token_ids[i] = token to activate adapter i+1
                              Output 0 = base (implicit default). SingleSwitch has no mechanism
                              to transition back to base mid-sequence.
            attention_mask: Optional attention mask [batch, 1, seq_len, seq_len]
            past_key_values: Optional Cache object (shared with model's decoder layers)
            cache_position: Position indices for caching [seq_len]

        Returns:
            (adapter_indices, modified_input_ids):
              adapter_indices:     [batch, seq_len] where 0 = base, 1+ = adapters.
              modified_input_ids:  [batch, seq_len] with each control-token
                                   id replaced by its substitute id.
        """
        bsz, q_len = input_ids.shape
        device = input_ids.device

        # ======================================================================
        # Prepare Q, K, V tensors  (single active dimension: dim 0)
        # ======================================================================
        # Only dim 0 carries signal; remaining dims are zero padding required
        # by the attention backend's head_dim constraint.  This gives
        # Q·K = 1 * (±gain) = ±gain, independent of head_dim.
        query_states = torch.zeros(
            (bsz, self.num_heads, q_len, self.head_dim), device=device
        )
        query_states[:, :, :, 0] = 1.0

        key_states = torch.zeros(
            (bsz, self.num_heads, q_len, self.head_dim), device=device
        )
        key_states[:, :, :, 0] = -self.control_token_gain

        value_states = torch.zeros(
            (bsz, self.num_heads, q_len, self.head_dim), device=device
        )

        # Set keys and values for control tokens
        for adapter_idx in range(self.num_adapters):
            token_id = adapter_token_ids[adapter_idx]
            adapter_id = adapter_idx + 1  # 1-indexed

            mask = input_ids == token_id  # [batch, seq_len]

            # Key dim 0: flip from -gain to +gain
            key_states[:, 0, :, 0][mask] = self.control_token_gain

            # Value dim 0: set adapter_id
            value_states[:, 0, :, 0][mask] = float(adapter_id)

        # ======================================================================
        # KV Cache with modern Cache API (same pattern as GraniteLoRAEmbeddedAttention)
        # ======================================================================
        if past_key_values is not None:
            # Cache will internally handle concatenation with past key/values
            # For switch, we don't have RoPE, so cache_kwargs doesn't include sin/cos
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # ======================================================================
        # Compute attention using HuggingFace backend
        # ======================================================================
        # Call HuggingFace attention backend (same as GraniteLoRAEmbeddedAttention)
        # This gets us FlashAttention, SDPA, FlexAttention, etc. for free
        attention_interface = eager_attention_forward
        if self.config is not None and hasattr(self.config, "_attn_implementation"):
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=None,
        )

        # ======================================================================
        # Compute adapter indices
        # ======================================================================
        # attn_output shape: [batch, seq_len, num_heads, head_dim]
        # num_heads = 1 in this case, and we only care about
        # the first dimension out of those head_dim
        # Extract only first dimension (where adapter_id is stored)
        # Shape: [batch, seq_len, 1, head_dim] -> [batch, seq_len]
        attn_output = attn_output[:, :, 0, 0]  # [batch, seq_len]

        # Round to get integer adapter indices
        adapter_indices = torch.round(attn_output).long()

        # Clamp to valid range [0, num_adapters]
        adapter_indices = torch.clamp(adapter_indices, 0, self.num_adapters)

        # Ensure output shape matches input shape
        assert adapter_indices.shape == input_ids.shape, (
            f"adapter_indices shape {adapter_indices.shape} must match input_ids shape {input_ids.shape}"
        )

        # Token-exchange rewrite: replace each control token's id with its
        # substitute id via the LUT. Done here (rather than in the decoder)
        # so the decoder sees a clean, unified input_ids and never has to
        # know about substitutes. Skipped only when the LUT was not built
        # (no substitute ids configured — e.g. a non-token-exchange test
        # fixture). Kept symmetric with the vLLM switch, which forbids the
        # `tensor.any()` short-circuit under @support_torch_compile.
        if self.control_to_substitute_lut is not None:
            sub_id_per_pos = self.control_to_substitute_lut[input_ids]
            is_control = sub_id_per_pos >= 0
            modified_input_ids = torch.where(is_control, sub_id_per_pos, input_ids)
        else:
            modified_input_ids = input_ids

        return adapter_indices, modified_input_ids
