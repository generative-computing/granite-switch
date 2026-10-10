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
    IsHybrid,
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

        # Classifier markers share the control-token machinery but select the
        # classifier head independently, preserving the active adapter. The head
        # scores each slot at its read layer (the final post-norm state by default).
        if "classifier" in getattr(config, "adapter_kinds", []):
            from .core import SwitchedClassifierHead

            self.classifier_head = SwitchedClassifierHead(
                hidden_size=config.hidden_size,
                num_classifier_slots=config.num_adapters,
                max_num_labels=config.max_classifier_labels,
            )

            # Packed verdicts assume the ordinary sampler's row selection.
            if vllm_config.speculative_config is not None:
                raise ValueError(
                    "Classifier slots do not support speculative decoding."
                )
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError("Classifier slots do not support microbatching.")

            # Built once: copying a list to CUDA inside forward breaks capture.
            self.register_buffer(
                "classifier_control_token_ids_t",
                torch.tensor(config.classifier_control_token_ids, dtype=torch.long),
                persistent=False,
            )

            is_classifier = [kind == "classifier" for kind in config.adapter_kinds]
            # Index 0 is a sentinel so a 1-based slot id indexes it directly.
            self.register_buffer(
                "classifier_slot_mask_t",
                torch.tensor([False, *is_classifier]),
                persistent=False,
            )

            # One buffer of slot ids per read layer (None = final). Store buffer
            # names, not tensors: Module.to() replaces registered buffers.
            read_layers = config.classifier_read_layers or [None] * num_adapters
            slots_by_layer: dict[int | None, list[int]] = {}
            for slot, (is_clf, layer) in enumerate(
                zip(is_classifier, read_layers), start=1
            ):
                if is_clf:
                    slots_by_layer.setdefault(layer, []).append(slot)
            self._classifier_final_layer_group = None
            self._classifier_layer_groups: dict[int, str] = {}
            for layer, slot_ids in slots_by_layer.items():
                name = f"_classifier_slots_{'final' if layer is None else layer}"
                self.register_buffer(
                    name, torch.tensor(slot_ids, dtype=torch.long), persistent=False
                )
                if layer is None:
                    self._classifier_final_layer_group = name
                else:
                    self._classifier_layer_groups[layer] = name
            # Intermediate scores are not carried across pipeline ranks.
            if (
                self._classifier_layer_groups
                and vllm_config.parallel_config.pipeline_parallel_size > 1
            ):
                raise ValueError(
                    "Intermediate-layer classifier slots do not support "
                    "pipeline parallelism."
                )
        else:
            self.classifier_head = None
            self._classifier_final_layer_group = None
            self._classifier_layer_groups = {}

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

            # Split routing state from classifier probe state. In the default
            # mode classifier slots are removed from the LoRA stream; markers
            # get a separate probe index while the previously selected adapter
            # remains active at that token.
            if self.switch is not None:
                classifier_probe_indices = (
                    self.switch.classifier_indices_from_tokens(input_ids)
                    if input_ids is not None
                    else None
                )
                lora_indices, classifier_indices = self.switch.split_indices(
                    adapter_indices, classifier_probe_indices
                )
            else:
                lora_indices = adapter_indices
                classifier_indices = torch.zeros_like(adapter_indices)

            # Prepare kernel metadata ONCE for all decoder layers. The fused
            # kernels route on lora_indices, including the marker token when
            # coactivation is enabled.
            if self.lora_meta is not None and self.lora_ctx is not None:
                self.decoder_interface.prepare_kernel_meta(
                    self.lora_meta, lora_indices, self.lora_ctx
                )

            # Store the persistent adapter state for pipeline parallelism.
            # Classifier probe indices are reconstructed from this rank's
            # original input_ids and the config-derived marker LUT.
            if intermediate_tensors is None:
                intermediate_tensors = IntermediateTensors({})
            intermediate_tensors["adapter_indices"] = adapter_indices
        else:
            # Subsequent ranks: recompute fixed-size LoRA metadata from
            # token-leading adapter_indices received through PP.
            if intermediate_tensors is not None:
                adapter_indices = intermediate_tensors["adapter_indices"]
                # Recompute both streams from the PP-propagated adapter state
                # and this rank's token ids. The classifier marker LUT is a
                # non-persistent config-derived buffer on every rank.
                if self.switch is not None:
                    classifier_probe_indices = (
                        self.switch.classifier_indices_from_tokens(input_ids)
                        if input_ids is not None
                        else None
                    )
                    lora_indices, classifier_indices = self.switch.split_indices(
                        adapter_indices, classifier_probe_indices
                    )
                else:
                    lora_indices = adapter_indices
                    classifier_indices = torch.zeros_like(adapter_indices)

                if self.lora_ctx is not None:
                    # Reuse this rank's LoRA stream, including the marker token
                    # when coactivation is enabled.
                    self.decoder_interface.prepare_kernel_meta(
                        self.lora_meta, lora_indices, self.lora_ctx
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
                classifier_indices = torch.zeros_like(adapter_indices)

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
        # Classifier slots reading an intermediate layer are scored as soon as
        # that layer runs; only the small score tensor is carried forward.
        if self.classifier_head is not None:
            classifier_scores = torch.zeros(
                classifier_indices.shape[0],
                self.config.max_classifier_labels,
                device=classifier_indices.device,
                dtype=hidden_states.dtype,
            )
        state = self.decoder_interface.enter_decoder_stack(hidden_states, residual)
        for i in range(self.start_layer, self.end_layer):
            state = self.decoder_interface.run_layer(self.layers[i], positions, state)
            group = self._classifier_layer_groups.get(i)
            if group is not None:
                # Layer i's output before the final norm: residual + block output.
                classifier_scores = self._score_group(
                    group,
                    state.residual + state.hidden_states,
                    classifier_indices,
                    classifier_scores,
                )

        if get_pp_group().is_last_rank:
            # Adaptation-specific finalize: LoRA folds the last residual via
            # rms_norm_select (fused/separate convention); SR does the per-token
            # where-merge of its two streams then a plain norm. The returned
            # [M, H] tensor is the final post-norm hidden state (what the LM head
            # consumes).
            hidden_states = self.decoder_interface.exit_decoder_stack(
                state, adapter_indices, self.norm, self.config
            )

            # Pack scores and slot ids onto the hidden states so vLLM's own
            # logits_indices gather selects them; compute_logits unpacks them.
            if self.classifier_head is not None:
                classifier_scores, classifier_slot = self._compute_classifier_outputs(
                    input_ids, hidden_states, classifier_indices, classifier_scores
                )
                hidden_states = torch.cat(
                    [hidden_states, classifier_scores, classifier_slot.unsqueeze(-1)],
                    dim=-1,
                )

            return hidden_states
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

    def _score_group(
        self,
        group: str,
        hidden: torch.Tensor,
        classifier_indices: torch.Tensor,
        classifier_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Write the scores of the slots in buffer ``group``, read from ``hidden``."""
        in_group = torch.isin(classifier_indices, getattr(self, group))
        slot_logits = self.classifier_head(hidden, classifier_indices)
        return torch.where(in_group.unsqueeze(-1), slot_logits, classifier_scores)

    def _compute_classifier_outputs(
        self,
        input_ids: torch.Tensor | None,
        final_hidden_states: torch.Tensor,
        classifier_indices: torch.Tensor,
        classifier_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add final-layer scores and return per-token (scores, slot ids).

        Only marker tokens carry a nonzero slot id. Callers must place one
        classifier marker last in each request; misplaced or duplicate markers
        are not rejected (request-boundary validation is deferred).
        """
        is_marker = (
            torch.isin(input_ids, self.classifier_control_token_ids_t)
            if input_ids is not None
            else torch.zeros_like(classifier_indices, dtype=torch.bool)
        )
        # Exact in bf16: the config limits classifier slot ids to 256.
        classifier_slot = torch.where(
            is_marker, classifier_indices, torch.zeros_like(classifier_indices)
        ).to(final_hidden_states.dtype)
        if self._classifier_final_layer_group is not None:
            classifier_scores = self._score_group(
                self._classifier_final_layer_group,
                final_hidden_states,
                classifier_indices,
                classifier_scores,
            )
        return classifier_scores, classifier_slot


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
        """Forward pass returning hidden states (classifier columns packed on)."""
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

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

        Classifier exit: the classifier request reports its verdict as a
        generated token, like a guardian LoRA. With a classifier head the
        hidden states carry packed (scores, slot id) columns; split them off,
        run the LM head on the real hidden state, then rewrite each classifier
        row: the whole vocab to -inf and the per-label scores onto the label
        words' token ids (``classifier_label_token_ids``). Other rows keep
        their LM logits.
        """
        if self.model.classifier_head is not None:
            hidden_size = self.config.hidden_size
            real_hidden = hidden_states[:, :hidden_size]
            verdict = hidden_states[:, hidden_size:-1]
            slot = hidden_states[:, -1]
        else:
            real_hidden = hidden_states
            verdict = slot = None
        logits = self.logits_processor(self.lm_head, real_hidden)
        return self._apply_classifier_verdict(logits, verdict, slot)

    def _apply_classifier_verdict(
        self,
        logits: torch.Tensor | None,
        verdict: torch.Tensor | None,
        slot: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Rewrite classifier requests' logit rows to emit their label word.

        Rows are classified by the slot mask, not just ``slot > 0``: vLLM's
        profiling sampler run feeds random hidden states, so the packed slot
        column can land on a LoRA slot, which has no label ids.
        """
        if logits is None or verdict is None or slot is None:
            return logits

        slot_mask = self.model.classifier_slot_mask_t
        num_slots = slot_mask.shape[0] - 1
        slot_int = slot.round().long()
        # Out-of-range values land on the sentinel row 0.
        safe_idx = slot_int.clamp(min=0, max=num_slots)
        is_classifier = (slot_int == safe_idx) & slot_mask[safe_idx]
        if not bool(is_classifier.any()):
            return logits

        # Each row writes only its own slot's real label count; padded
        # columns are never read.
        per_slot_label_ids = self.config.classifier_label_token_ids
        rows = is_classifier.nonzero(as_tuple=True)[0]
        logits[rows] = float("-inf")
        for r in rows.tolist():
            row_slot = int(slot_int[r])
            ids = per_slot_label_ids[row_slot - 1]
            label_ids = torch.tensor(ids, dtype=torch.long, device=logits.device)
            logits[r, label_ids] = verdict[r, : len(ids)].to(logits.dtype)
        return logits

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
