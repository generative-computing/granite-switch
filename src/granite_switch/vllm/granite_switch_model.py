# SPDX-License-Identifier: Apache-2.0
"""Granite model with adapter switching for vLLM.

Architecture:
    Input Tokens
        ↓
    Embedding Layer (frozen)
        ↓
    MultiSwitch (adapter selection)
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
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from granite_switch.config import GraniteSwitchConfig
from granite_switch.token_exchange import apply_token_exchange

from .audio.processor import (
    AUDIO_MARKER,
    GraniteSwitchASRDummyInputsBuilder,
    GraniteSwitchASRMultiModalProcessor,
    GraniteSwitchASRProcessingInfo,
)
from .decoder.interface import select_decoder_interface
from .switch import create_switch

logger = init_logger(__name__)


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

        # Adaptation strategy: confines the LoRA-vs-Shadow-Residual split to the
        # decoder tier. Keyed on config.cross_stream_rank (None -> LoRA, int -> SR).
        # The shared __init__/forward call its hooks so one code path drives both.
        self.decoder_interface = select_decoder_interface(config)

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

            # Fused kernel metadata (bitmask-based). The strategy builds the
            # adaptation's (kernel-meta, ctx) pair: LoRA -> (FusedLoRAKernelMeta,
            # LoRAContext); SR -> (SRFusedLoRAKernelMeta, SRLoRAContext).
            self.lora_meta, self.lora_ctx = self.decoder_interface.make_kernel_meta(
                vllm_config.device_config.device,
            )

        else:
            self.switch = None
            self.adapter_token_ids = None
            self.lora_meta = None
            self.lora_ctx = None

        # 3. Base transformer layers with custom LoRA
        #
        # When adapters are present, config.num_hidden_layers includes placeholder
        # entries for the switch's KV cache slots (MultiSwitch uses 2: a counting
        # slot and a memory slot). These placeholders exist for HF DynamicCache
        # sizing; vLLM auto-discovers its Attention layers and doesn't need them.
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
            """Create one decoder layer for this adaptation."""
            return self.decoder_interface.make_decoder_layer(vllm_config, prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            num_decoder_layers,
            _make_decoder_layer,
            prefix=f"{prefix}.layers",
        )

        # Wire shared LoRAContext to every module that reads per-forward metadata.
        # This follows vLLM's PunicaWrapper pattern: a single shared object
        # populated once per forward, read by all layers that need LoRA metadata
        # or hiding group masks. object.__setattr__ bypasses nn.Module.__setattr__
        # so the context is NOT registered as a submodule/buffer (it carries live
        # per-forward tensors, not parameters) and the attribute stays stable for
        # torch.compile.
        if num_adapters > 0:
            _ctx_types = self.decoder_interface.ctx_wire_types()
            for module in self.modules():
                if isinstance(module, _ctx_types):
                    object.__setattr__(module, "_lora_ctx", self.lora_ctx)

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
            # ``input_ids is not None`` matters on the multimodal path: vLLM can
            # precompute ``inputs_embeds`` and hand the decoder no ids, and the
            # switch reads ids. ``requires_raw_input_tokens`` normally prevents
            # that, so this is a guard rather than an expected branch.
            if self.switch is not None and input_ids is not None:
                # ``positions`` MUST be forwarded. vLLM flattens a batch into a
                # single ``[total_tokens]`` tensor, so a switch cannot infer
                # per-request token offsets on its own. The coded engine derives
                # its ``1/(1+n)`` counting anchor from ``positions == 0``; with a
                # locally-fabricated ``arange(total_tokens)`` only the FIRST
                # request in the batch would contain an anchor, so every other
                # request counts against a missing baseline, recovers a wrong
                # write address ``n``, and retrieves whatever adapter happens to
                # live at that address (arbitrary mis-routing, not a uniform
                # off-by-one).
                adapter_indices, modified_input_ids = self.switch(
                    input_ids=input_ids,
                    adapter_token_ids=self.adapter_token_ids,
                    positions=positions,
                )
            else:
                # No switch, or the multimodal path (input_ids is None because
                # vLLM pre-merged inputs_embeds): run on base, adapter_id 0.
                # Sizing off inputs_embeds in that case is what makes this safe;
                # reading input_ids.device unconditionally would raise instead.
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

            # Prepare kernel metadata ONCE for all decoder layers (adaptation-specific).
            if self.lora_meta is not None and self.lora_ctx is not None:
                self.decoder_interface.prepare_kernel_meta(
                    self.lora_meta, adapter_indices, self.lora_ctx
                )

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
                    self.decoder_interface.prepare_kernel_meta(
                        self.lora_meta, adapter_indices, self.lora_ctx
                    )
            else:
                # Fallback: no metadata available (should not happen in normal operation)
                num_tokens = input_ids.shape[0] if input_ids is not None else 0
                if input_ids is not None:
                    fallback_device = input_ids.device
                elif self.lora_meta is not None:
                    fallback_device = self.lora_meta.device
                else:
                    fallback_device = self.embed_tokens.weight.device
                adapter_indices = torch.zeros(
                    num_tokens,
                    dtype=torch.long,
                    device=fallback_device,
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

        # Pass through base transformer layers via adaptation hooks. The
        # adaptation owns the stack shape: LoRA runs single-stream [M,H] and
        # threads (hidden_states, residual); SR re-stacks to [2M,H] rank-locally
        # in enter_decoder_stack and collapses it in exit_decoder_stack. All
        # per-forward metadata (kernel meta + hiding masks) is on the shared ctx.
        state = self.decoder_interface.enter_decoder_stack(hidden_states, residual)
        for i in range(self.start_layer, self.end_layer):
            state = self.decoder_interface.run_layer(self.layers[i], positions, state)

        if get_pp_group().is_last_rank:
            # Adaptation-specific finalize: LoRA folds the last residual via
            # rms_norm_select (fused/separate convention); SR does the per-token
            # where-merge of its two streams then a plain norm.
            return self.decoder_interface.exit_decoder_stack(
                state, adapter_indices, self.norm, self.config
            )
        else:
            # Non-last rank: ship two token-leading [M,H] tensors so vLLM's
            # per-token IntermediateTensors slice stays correct. LoRA sends
            # (hidden_states, residual); SR overloads residual as its adapter half.
            if intermediate_tensors is None:
                intermediate_tensors = IntermediateTensors({})
            h_out, r_out = self.decoder_interface.to_intermediate(state)
            intermediate_tensors["hidden_states"] = h_out
            intermediate_tensors["residual"] = r_out
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
    # IsHybrid deliberately NOT declared. It is a Protocol requiring
    # get_mamba_state_shape_from_config and get_mamba_state_copy_func, neither of
    # which this model implements or needs: the composer normalizes every layer
    # to attention, so a composed checkpoint has zero mamba layers. The marker
    # arrived by inheritance from GraniteMoeHybridConfig, not by design.
    #
    # Declaring it set ModelConfig._model_info.is_hybrid, and vLLM's escape hatch
    # for exactly this case compares the literal string "attention":
    #     return layer_types is None or not all(
    #         layer == "attention" for layer in layer_types)   # config/model.py
    # transformers 5.16 remaps "attention" -> "full_attention" on load, so the
    # hatch stopped matching and vLLM took the hybrid path, asking for the mamba
    # dtype hook and failing engine init with AttributeError. Nine other vLLM
    # sites gate on is_hybrid (mamba state allocation, block-size alignment,
    # speculative decoding, KV sizing); none should apply here.
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

        Reads the switch's LUT and calls the shared
        :func:`~granite_switch.token_exchange.apply_token_exchange` rather than a
        method on the switch. Every engine drives the rewrite through that one
        function against its own buffer, so this works on all of them without
        asking what kind of switch it got — which is the same reason the rebuild
        is a free function too (a method existed on only one engine and broke
        compose for the other).
        """
        ids = input_ids
        switch = getattr(self.model, "switch", None)
        if switch is not None:
            ids = apply_token_exchange(
                getattr(switch, "control_to_substitute_lut", None), input_ids
            )
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
        """Forward pass returning hidden states."""
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

        No control-token logit suppression is applied here. The intended
        design is that control tokens are freely generatable, so no runtime
        suppression is the target end state. Even if an interim suppression
        were wanted, it could not live here: vLLM v1 calls compute_logits on
        sample-extracted hidden states, which are no longer aligned with the
        per-token adapter_indices computed in forward(). Any suppression would
        have to act where adapter_indices and hidden_states still share the
        same token dimension.
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

        Delegated to the adaptation strategy: LoRA fans HF stacked-MoE tensors
        into per-expert FusedMoE loads (and loads composed checkpoints directly);
        SR does the unfused->fused fuse-at-load mapping and marks the absent
        shared-KV LoRA slices loaded. Each ends by finalizing the fused kernel
        state + registering per-module remap tables (was ``_finalize_fused_lora``).
        """
        return self.model.decoder_interface.load_weights(self, weights)
