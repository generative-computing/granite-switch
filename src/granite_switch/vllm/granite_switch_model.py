# SPDX-License-Identifier: Apache-2.0
"""Granite model with adapter switching for vLLM.

Architecture:
    Input Tokens
        ↓
    Embedding Layer (frozen)
        ↓
    SingleSwitch (adapter selection)
        ↓
    Adapter Indices (per token)
        ↓
    Base Transformer Layers (frozen with frozen LoRA adapters)
        ↓
    Output

The switch detects special tokens and selects the appropriate adapter for each token.
All parameters are frozen - no training needed.
"""

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from granite_switch.config import GraniteSwitchConfig

from .audio.processor import (
    AUDIO_MARKER,
    GraniteSwitchASRDummyInputsBuilder,
    GraniteSwitchASRMultiModalProcessor,
    GraniteSwitchASRProcessingInfo,
)
from .core import (
    CompileFriendlyLoRAKernelMeta,
    GraniteSwitchDecoderLayer,
    LoRAContext,
)
from .core.decoder import GraniteLoRAEmbeddedAttention
from .core.lora import SwitchedLoRALinear
from .switch import create_switch


def _get_intermediate_tensor(
    tensors: IntermediateTensors,
    name: str,
) -> torch.Tensor | None:
    try:
        return tensors[name]
    except KeyError:
        return None


@support_torch_compile
class GraniteSwitchModel(nn.Module):
    """
    Granite transformer with simple attention-based adapter switch.

    The model consists of:
    1. Standard embedding layer
    2. Simple switch (attention-based special token detection)
    3. Base transformer layers with LoRA
    4. LM head

    The switch detects special tokens, selects the appropriate adapter, and
    rewrites each control token's id to its substitute id (token exchange).
    The decoder embeds the rewritten ids and is otherwise oblivious to the
    substitution. Adapter indices are passed as arguments to LoRA layers.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config

        # Validate config type
        if not isinstance(config, GraniteSwitchConfig):
            raise TypeError(
                f"Expected GraniteSwitchConfig, got {type(config).__name__}"
            )

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = 0
        if hasattr(config, "lora_vocab_size"):
            lora_vocab = (
                config.lora_vocab_size if config.lora_vocab_size is not None else 0
            )

        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        # 1. Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        # 2. Switch and adapter configuration
        num_adapters = config.num_adapters
        if num_adapters > 0:
            self.switch = create_switch(config, vllm_config=vllm_config)

            # --- Control token buffers ---
            # All values come from config (serialized in config.json).
            # Stored as plain tensors (not nn.Parameter) so they don't pollute
            # the state_dict and are torch.compile-friendly (no .item() needed).
            # register_buffer makes them follow .to(device) and .cuda() calls.
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
                    torch.zeros(num_adapters, dtype=torch.long),
                )

            # Token-exchange LUT lives on the switch module
            # (see vllm/switch/single.py); the switch rewrites input_ids
            # in-place during its forward pass, so this model class no
            # longer needs a decoder-side substitute table.

            # Initialize compile-friendly LoRA metadata handler
            # This replaces vLLM's LoRAKernelMeta with a torch.compile-compatible version
            # that avoids data-dependent branching
            self.lora_meta = CompileFriendlyLoRAKernelMeta(
                num_adapters=num_adapters,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

        else:
            self.switch = None
            self.adapter_token_ids = None
            self.lora_meta = None

        # 3. Base transformer layers with custom LoRA
        #
        # When adapters are present, config.num_hidden_layers includes a placeholder
        # entry for the switch's KV cache slot (SingleSwitch uses 1 slot for its
        # single attention head). This placeholder exists for HF DynamicCache
        # sizing; vLLM auto-discovers its Attention layers and doesn't need it.
        # We subtract the switch's cache slot count to recover the true number of
        # decoder layers, and use it as an offset into layer_types (whose first
        # entry is an "attention" placeholder for the switch).
        if config.num_adapters > 0:
            layer_offset = self.switch.num_cache_layers
            num_decoder_layers = config.num_hidden_layers - layer_offset
        else:
            layer_offset = 0
            num_decoder_layers = config.num_hidden_layers

        def _make_decoder_layer(prefix: str):
            """Create attention decoder layer."""
            return GraniteSwitchDecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            num_decoder_layers,
            _make_decoder_layer,
            prefix=f"{prefix}.layers",
        )

        # Wire shared LoRAContext to every module that reads per-forward metadata.
        # This follows vLLM's PunicaWrapper pattern: a single shared object
        # populated once per forward, read by all layers that need LoRA metadata
        # or hiding group masks.
        if num_adapters > 0:
            self.lora_ctx = LoRAContext()
            _ctx_types = (
                SwitchedLoRALinear,
                GraniteLoRAEmbeddedAttention,
                GraniteSwitchDecoderLayer,
            )
            for module in self.modules():
                if isinstance(module, _ctx_types):
                    object.__setattr__(module, "_lora_ctx", self.lora_ctx)
        else:
            self.lora_ctx = None

        # 4. RMS Layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # this may be unneccessary given get_input_embeddings
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids."""
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        """Allocate PP profiling buffers for token-leading tensors.

        vLLM slices every IntermediateTensors entry by token count. Keep only
        token-leading metadata here; fixed-size LoRA metadata is recomputed on
        each PP rank from adapter_indices.
        """
        tensors = {
            "hidden_states": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "residual": torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=dtype,
                device=device,
            ),
            "adapter_indices": torch.zeros(
                (batch_size,),
                dtype=torch.long,
                device=device,
            ),
        }

        return IntermediateTensors(tensors)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """
        Forward pass with integrated switch logic.

        The overall class is decorated with @support_torch_compile. The switch computation
        happens inside this method (within the compiled region).

        Args:
            input_ids: Token IDs (num_tokens,)
            positions: Token positions for RoPE (num_tokens,)
            intermediate_tensors: For pipeline parallelism, contains hidden_states
            inputs_embeds: Optional pre-computed embeddings (only used on first rank)

        Returns:
            If last rank: Final hidden states (num_tokens, hidden_size)
            If not last rank: IntermediateTensors with hidden_states
        """
        # ═══════════════════════════════════════════════════════════════
        # COMPILED: Switch + Metadata preparation
        # ═══════════════════════════════════════════════════════════════

        # Step 1: Switch — determine adapter for each token and rewrite
        # control tokens via token-exchange. Only runs on first rank.
        if get_pp_group().is_first_rank:
            if self.switch is not None and input_ids is not None:
                adapter_indices, modified_input_ids = self.switch(
                    input_ids=input_ids,
                    adapter_token_ids=self.adapter_token_ids,
                )
            else:
                # No switch, or the multimodal path (input_ids is None because
                # vLLM pre-merged inputs_embeds): run on base, adapter_id 0.
                if input_ids is not None:
                    num_tokens = input_ids.shape[0]
                    device = input_ids.device
                else:
                    num_tokens = inputs_embeds.shape[0]
                    device = inputs_embeds.device
                adapter_indices = torch.zeros(
                    num_tokens,
                    dtype=torch.long,
                    device=device,
                )
                modified_input_ids = input_ids

            # Step 2: Prepare LoRA metadata ONCE for all decoder layers.
            # Stored on the shared LoRAContext — every SwitchedLoRALinear reads from it.
            if self.lora_meta is not None and self.lora_ctx is not None:
                # Convert to Punica convention: 0=base -> -1=base
                punica_indices = adapter_indices - 1
                self.lora_meta.prepare_and_store(punica_indices, self.lora_ctx)

            # Store metadata in intermediate_tensors for pipeline parallelism.
            if intermediate_tensors is None:
                intermediate_tensors = IntermediateTensors({})
            intermediate_tensors["adapter_indices"] = adapter_indices
        else:
            # Subsequent ranks: recompute fixed-size LoRA metadata from
            # token-leading adapter_indices received through PP.
            if intermediate_tensors is not None:
                adapter_indices = intermediate_tensors["adapter_indices"]
                if self.lora_ctx is not None:
                    punica_indices = adapter_indices - 1
                    self.lora_meta.prepare_and_store(punica_indices, self.lora_ctx)
            else:
                # Fallback: no metadata available (should not happen in normal operation)
                num_tokens = input_ids.shape[0] if input_ids is not None else 0
                adapter_indices = torch.zeros(
                    num_tokens,
                    dtype=torch.long,
                    device=input_ids.device
                    if input_ids is not None
                    else torch.device("cuda"),
                )

        # ═══════════════════════════════════════════════════════════════
        # Get embeddings (or hidden states from previous pipeline stage)
        # ═══════════════════════════════════════════════════════════════
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                # Embed the (possibly-rewritten) input_ids the switch returned.
                # The switch already performed the token-exchange rewrite, so
                # this single lookup produces the correct embeddings for both
                # control positions (substitute id) and content positions.
                hidden_states = self.get_input_embeddings(modified_input_ids)

            hidden_states *= self.config.embedding_multiplier
            residual = None
        else:
            # Non-first rank: get hidden states from intermediate_tensors
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = _get_intermediate_tensor(intermediate_tensors, "residual")

        # Pass through base transformer layers.
        # All per-forward metadata (LoRA + hiding masks) is on the shared LoRAContext.
        # Each layer returns (hidden_states, residual); the residual-add happens
        # inside rms_norm_select using the convention that matches the original
        # model's vLLM class (fused or separate) for bit-exact compatibility.
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        # Final norm: fold in the last residual via rms_norm_select so
        # the same fused/separate convention is used throughout.
        if get_pp_group().is_last_rank:
            from granite_switch.vllm.core.decoder import rms_norm_select

            hidden_states, _ = rms_norm_select(
                self.norm,
                hidden_states,
                residual,
                self.config.fused_add_norm,
            )
            return hidden_states
        else:
            # Non-last rank: return IntermediateTensors with hidden states
            if intermediate_tensors is None:
                intermediate_tensors = IntermediateTensors({})
            intermediate_tensors["hidden_states"] = hidden_states
            intermediate_tensors["residual"] = residual
            return intermediate_tensors


@MULTIMODAL_REGISTRY.register_processor(
    GraniteSwitchASRMultiModalProcessor,
    info=GraniteSwitchASRProcessingInfo,
    dummy_inputs=GraniteSwitchASRDummyInputsBuilder,
)
class GraniteSwitchForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    IsHybrid,
):
    """
    Granite model with switch for causal language modeling.

    This wraps GraniteSwitchModel with an LM head for token prediction.

    Multimodal (audio): the registered ASR processor transcribes audio and
    replaces an ``<|audio|>`` marker with the transcript tokens before the
    decoder runs (see granite_switch.vllm.audio). ``embed_multimodal`` supplies
    the embeddings for those positions — for the alpha, the transcript's own
    text embeddings, which is the seam a future trained audio encoder reuses.
    Audio capability is gated per-checkpoint by ``config.asr_enabled`` (the
    processor reports no audio modality when disabled).
    """

    supports_multimodal = True

    # Without this, the multimodal path passes only inputs_embeds and the switch
    # cannot see control tokens — audio requests would bypass adapters.
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):
        if modality.startswith("audio"):
            return AUDIO_MARKER
        return None

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "input_linear",
        "output_linear",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        # Model with switch inside
        self.model = GraniteSwitchModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = 1.0
        if hasattr(config, "logits_scaling"):
            logit_scale /= config.logits_scaling

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            config.vocab_size,
            logit_scale,
        )
        self.sampler = None  # Will be set by vLLM

    def embed_multimodal(self, **kwargs) -> list:
        """Embeddings for the audio placeholder positions (one tensor per item).

        ALPHA: the transcript token ids produced by the ASR processor are
        embedded with the model's own token table — identical to embedding them
        as ordinary text. Returned UN-scaled; the Granite embedding_multiplier
        is applied later in the forward, so these rows scale consistently with
        normal tokens. A future trained audio encoder swaps in here.
        """
        audio_token_ids = kwargs.get("audio_token_ids")
        if audio_token_ids is None:
            return []
        embeds = self.model.embed_tokens(audio_token_ids)
        num_tokens = kwargs.get("audio_num_tokens")
        if num_tokens is None:
            return [embeds]
        sizes = [int(n) for n in num_tokens]
        return list(torch.split(embeds, sizes))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed token ids; scatter multimodal embeddings into their positions.

        Returns UN-scaled embeddings (the model forward applies the Granite
        embedding_multiplier once over everything).

        Applies the switch's token-exchange rewrite (control -> substitute ids)
        before embedding, so adapter control tokens get their in-distribution
        embeddings exactly as on the text path. Adapter *detection* still runs in
        forward on the raw input_ids (passed because requires_raw_input_tokens).
        """
        ids = input_ids
        switch = getattr(self.model, "switch", None)
        if switch is not None:
            ids = switch.apply_token_exchange(input_ids)
        inputs_embeds = self.model.embed_tokens(ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            mm = multimodal_embeddings
            if isinstance(mm, (list, tuple)):
                mm = torch.cat(list(mm)) if len(mm) else None
            if mm is not None:
                inputs_embeds[is_multimodal] = mm.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass returning hidden states.

        Switch logic now happens inside GraniteSwitchModel.forward(),
        so this is just a simple passthrough.
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states.

        Suppression of control tokens is NOT done here — see issue #14.
        vLLM v1 calls compute_logits on sample-extracted hidden states, which
        are not aligned with per-token adapter_indices from forward().
        Suppression must be implemented at a point where adapter_indices and
        hidden_states share the same token dimension.
        """
        return self.logits_processor(self.lm_head, hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ):
        """Sample next tokens from logits."""
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load model weights from checkpoint.

        Handles two checkpoint formats:

        1. **Composed checkpoints** (from compose_granite_switch.py): parameter names
           match the vLLM model exactly — loaded directly.

        2. **HuggingFace checkpoints** (from save_pretrained): MoE expert weights
           use a stacked format that must be split into per-expert tensors for
           vLLM's FusedMoE layer.

           HF format → vLLM format:
           - block_sparse_moe.input_linear.weight [E, 2*I, H]
             → experts.w13_weight via weight_loader(shard_id="w1"/"w3", expert_id=e)
           - block_sparse_moe.output_linear.weight [E, H, I]
             → experts.w2_weight via weight_loader(shard_id="w2", expert_id=e)
           - block_sparse_moe.router.layer.weight [E, H]
             → block_sparse_moe.gate.weight (direct rename)
        """
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        def _load_direct(name, loaded_weight):
            """Load a weight directly by name."""
            if name.endswith(".bias") and name not in params_dict:
                return
            if is_pp_missing_parameter(name, self):
                return
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        def _load_expert(param_name, loaded_weight, weight_name, shard_id, expert_id):
            """Load a per-expert weight into a FusedMoE packed parameter."""
            if is_pp_missing_parameter(param_name, self):
                return
            if param_name not in params_dict:
                return
            param = params_dict[param_name]
            weight_loader = param.weight_loader
            weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            loaded_params.add(param_name)

        for name, loaded_weight in weights:
            # ── HF stacked MoE: input_linear → per-expert w1/w3 ──
            if name.endswith(".block_sparse_moe.input_linear.weight"):
                for e in range(loaded_weight.size(0)):
                    w1_name = name.replace(
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w1.weight",
                    )
                    w3_name = name.replace(
                        ".block_sparse_moe.input_linear.weight",
                        f".block_sparse_moe.experts.{e}.w3.weight",
                    )
                    w1_param, w3_param = loaded_weight[e].chunk(2, dim=0)
                    _load_expert(
                        name.replace(".input_linear.", ".experts.w13_"),
                        w1_param,
                        w1_name,
                        shard_id="w1",
                        expert_id=e,
                    )
                    _load_expert(
                        name.replace(".input_linear.", ".experts.w13_"),
                        w3_param,
                        w3_name,
                        shard_id="w3",
                        expert_id=e,
                    )
                continue

            # ── HF stacked MoE: output_linear → per-expert w2 ──
            if name.endswith(".block_sparse_moe.output_linear.weight"):
                for e in range(loaded_weight.size(0)):
                    w2_name = name.replace(
                        ".block_sparse_moe.output_linear.weight",
                        f".block_sparse_moe.experts.{e}.w2.weight",
                    )
                    _load_expert(
                        name.replace(".output_linear.", ".experts.w2_"),
                        loaded_weight[e],
                        w2_name,
                        shard_id="w2",
                        expert_id=e,
                    )
                continue

            # ── HF MoE router → gate ──
            if name.endswith(".block_sparse_moe.router.layer.weight"):
                gate_name = name.replace(
                    ".block_sparse_moe.router.layer.weight",
                    ".block_sparse_moe.gate.weight",
                )
                _load_direct(gate_name, loaded_weight)
                continue

            # ── Direct load (built checkpoints + all non-MoE weights) ──
            _load_direct(name, loaded_weight)

        # Report unloaded parameters
        unloaded_params = [name for name in params_dict if name not in loaded_params]
        if unloaded_params:
            print(
                f"Warning: {len(unloaded_params)} parameters were not loaded from checkpoint"
            )
            if len(unloaded_params) <= 10:
                for name in unloaded_params:
                    print(f"  - {name}")
            else:
                for name in unloaded_params[:10]:
                    print(f"  - {name}")
                print(f"  ... and {len(unloaded_params) - 10} more")
