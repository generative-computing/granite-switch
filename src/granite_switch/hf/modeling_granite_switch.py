# SPDX-License-Identifier: Apache-2.0
"""Granite model with adapter switching for Hugging Face.

This implementation extends the base Granite model with:
1. SingleSwitch for computing per-token adapter indices
2. LoRA-enhanced attention and MLP layers that apply different adapters per token
3. Control token masking to prevent KV cache corruption
"""

import torch
import torch.nn as nn
import transformers
from packaging.version import parse as _parse_version
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridMLP,
    GraniteMoeHybridMoE,
    GraniteMoeHybridPreTrainedModel,
    GraniteMoeHybridRMSNorm,
    GraniteMoeHybridRotaryEmbedding,
)
from transformers.utils import logging

from granite_switch.config import GraniteSwitchConfig

from .core.lora import (
    GraniteLoRAEmbeddedAttention,
    replace_shared_mlp_projections_with_lora,
)
from .switch import create_switch

logger = logging.get_logger(__name__)

# transformers 5.9.0 renamed `input_embeds` -> `inputs_embeds` and dropped the
# unused `cache_position` kwarg in `create_causal_mask`.
_TRANSFORMERS_GE_5_9 = _parse_version(transformers.__version__) >= _parse_version(
    "5.9.0"
)


class GraniteSwitchAttentionDecoderLayer(nn.Module):
    """Attention decoder layer with LoRA and adapter routing.

    Supports optional MoE (frozen) alongside shared_mlp when num_local_experts > 0.
    """

    def __init__(self, config: GraniteSwitchConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        self.layer_type = "attention"

        # Attention with LoRA
        self.self_attn = GraniteLoRAEmbeddedAttention(config, layer_idx)

        # MLP section
        self.has_experts = config.num_local_experts > 0
        if self.has_experts:
            # MoE: frozen router + frozen expert weights (no LoRA)
            self.block_sparse_moe = GraniteMoeHybridMoE(config)

        # Shared MLP: upstream module with LoRA projections replaced in-place
        self.shared_mlp = GraniteMoeHybridMLP(config)
        self._has_shared_input_lora, self._has_shared_output_lora = (
            replace_shared_mlp_projections_with_lora(self.shared_mlp, config)
        )

        # Layer norms
        self.input_layernorm = GraniteMoeHybridRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = GraniteMoeHybridRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _set_shared_mlp_context(self, adapter_indices):
        if self._has_shared_input_lora:
            self.shared_mlp.input_linear._adapter_indices = adapter_indices
        if self._has_shared_output_lora:
            self.shared_mlp.output_linear._adapter_indices = adapter_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adapter_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_values = self.self_attn(
            hidden_states=hidden_states,
            adapter_indices=adapter_indices,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.has_experts:
            moe_output, _router_logits = self.block_sparse_moe(hidden_states)
            self._set_shared_mlp_context(adapter_indices)
            shared_output = self.shared_mlp(hidden_states)
            self._set_shared_mlp_context(None)
            hidden_states = moe_output + shared_output
        else:
            self._set_shared_mlp_context(adapter_indices)
            hidden_states = self.shared_mlp(hidden_states)
            self._set_shared_mlp_context(None)

        hidden_states = residual + hidden_states * self.residual_multiplier

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_values,)
        return outputs


# Backward-compatible alias
GraniteSwitchDecoderLayer = GraniteSwitchAttentionDecoderLayer


class GraniteSwitchPreTrainedModel(GraniteMoeHybridPreTrainedModel):
    """PreTrainedModel base class for GraniteSwitch.

    Inherits from GraniteMoeHybridPreTrainedModel to get weight init for
    all standard PreTrainedModel capabilities.
    """

    config_class = GraniteSwitchConfig
    base_model_prefix = "model"
    _no_split_modules = ["GraniteSwitchAttentionDecoderLayer"]
    _is_stateful = True


class GraniteSwitchModel(GraniteSwitchPreTrainedModel):
    """Granite model with switch-controlled LoRA adapters.

    RoPE is only applied when position_embedding_type == "rope".
    """

    def __init__(self, config: GraniteSwitchConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.embedding_multiplier = config.embedding_multiplier

        # Switch for adapter selection
        if config.num_adapters > 0:
            self.switch = create_switch(config, layer_idx=0)

            # --- Control token buffers ---
            # All values come from config (serialized in config.json).
            # Stored as buffers (not nn.Parameter) so they follow .to(device)
            # without appearing as trainable parameters.
            #
            # adapter_token_ids: Hidden-flavor control tokens, one per adapter.
            #   The switch layer detects these in the input sequence to determine
            #   which adapter to activate. Position in the tensor = adapter index.
            #   These tokens are KV-hidden (masked from attention) so downstream
            #   experts see only clean base-model representations.
            token_ids = config.adapter_token_ids
            if token_ids is not None:
                self.register_buffer(
                    "adapter_token_ids",
                    torch.tensor(token_ids, dtype=torch.long),
                )
            else:
                # Build script hasn't populated yet — zeros placeholder
                self.register_buffer(
                    "adapter_token_ids",
                    torch.zeros(config.num_adapters, dtype=torch.long),
                )

            # Token-exchange LUT lives on the switch module (see hf/switch/
            # single.py); the switch rewrites input_ids in-place during its
            # forward pass, so this model class no longer needs a decoder-
            # side substitute table.

        else:
            self.switch = None
            self.adapter_token_ids = None

        # Decoder layers
        if config.num_adapters > 0:
            layer_offset = self.switch.num_cache_layers
            num_decoder_layers = config.num_hidden_layers - layer_offset
        else:
            num_decoder_layers = config.num_hidden_layers
            layer_offset = 0

        # All layers are attention decoder layers.
        layers = []
        for local_idx in range(num_decoder_layers):
            global_layer_idx = local_idx + layer_offset
            layers.append(GraniteSwitchAttentionDecoderLayer(config, global_layer_idx))
        self.layers = nn.ModuleList(layers)

        # Final norm
        self.norm = GraniteMoeHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings (only if position_embedding_type == "rope")
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == "rope":
            self.rotary_emb = GraniteMoeHybridRotaryEmbedding(config=config)
        else:
            self.rotary_emb = None

        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # Initialize cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # Determine sequence shape and device. With input_ids we get them
        # directly; with pre-supplied inputs_embeds we read from the tensor.
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Causal mask (4D for attention layers). create_causal_mask only
        # uses the embedding tensor for batch/query/dtype inference; we
        # haven't embedded yet (the switch call below may rewrite input_ids
        # first), so pass a stub of the right shape/dtype.
        embed_dtype = self.embed_tokens.weight.dtype
        mask_shape_proxy = (
            inputs_embeds
            if inputs_embeds is not None
            else torch.empty(
                batch_size, seq_length, 1, device=device, dtype=embed_dtype
            )
        )
        mask_kwargs = {
            "config": self.config,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        if _TRANSFORMERS_GE_5_9:
            mask_kwargs["inputs_embeds"] = mask_shape_proxy
        else:
            mask_kwargs["input_embeds"] = mask_shape_proxy
            mask_kwargs["cache_position"] = cache_position
        causal_mask = create_causal_mask(**mask_kwargs)

        # The switch returns adapter_indices alongside modified_input_ids:
        # input_ids with each control token rewritten to its substitute id,
        # so the decoder can embed once without any token-exchange awareness.
        modified_input_ids = input_ids
        if self.switch is not None:
            adapter_indices, modified_input_ids = self.switch(
                input_ids=input_ids,
                adapter_token_ids=self.adapter_token_ids,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
        else:
            adapter_indices = torch.zeros(
                (batch_size, seq_length),
                dtype=torch.long,
                device=device,
            )

        # Embed once, on the (possibly-rewritten) input_ids. The decoder is
        # token-exchange-agnostic — it just embeds whatever the switch
        # passed through.
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(modified_input_ids)
        inputs_embeds = inputs_embeds * self.embedding_multiplier

        # Expose adapter_indices for tests and debugging.
        self._last_adapter_indices = adapter_indices

        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(
                inputs_embeds, position_ids=position_ids
            )

        # Decoder layers
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adapter_indices=adapter_indices,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                if layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GraniteSwitchForCausalLM(GraniteSwitchPreTrainedModel, GenerationMixin):
    """Granite for causal LM with adapter switch.

    Extends GraniteSwitchPreTrainedModel with LM head and generation capabilities.
    """

    config_class = GraniteSwitchConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: GraniteSwitchConfig):
        super().__init__(config)

        self.model = GraniteSwitchModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits / self.config.logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
