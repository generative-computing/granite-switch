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

        # Probe heads: optional alternative for LoRA adapters. A probe
        # slot shares the adapter index/token machinery, but its index is tagged
        # as "probe" (via the switch's adapter_kind_lut). The switch's
        # split_indices() routes those positions out of the LoRA stream (so LoRA
        # no-ops there) and into a probe stream. The head scores each slot
        # at its read layer (the final post-norm state by default).
        if "probe" in getattr(config, "adapter_kinds", []):
            from .core import SwitchedProbeHead

            self.probe_head = SwitchedProbeHead(
                hidden_size=config.hidden_size,
                num_probe_slots=config.num_adapters,
                max_num_labels=config.max_probe_labels,
            )

            # Packed verdicts assume the ordinary sampler's row selection.
            if vllm_config.speculative_config is not None:
                raise ValueError("Probe slots do not support speculative decoding.")
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError("Probe slots do not support microbatching.")

            # Built once: copying a list to CUDA inside forward breaks capture.
            self.register_buffer(
                "probe_control_token_ids_t",
                torch.tensor(config.probe_control_token_ids, dtype=torch.long),
                persistent=False,
            )

            is_probe = [kind == "probe" for kind in config.adapter_kinds]
            # Index 0 is a sentinel so a 1-based slot id indexes it directly.
            self.register_buffer(
                "probe_slot_mask_t",
                torch.tensor([False, *is_probe]),
                persistent=False,
            )

            # One buffer of slot ids per read layer (None = final). Store buffer
            # names, not tensors: Module.to() replaces registered buffers.
            read_layers = config.probe_read_layers or [None] * num_adapters
            slots_by_layer: dict[int | None, list[int]] = {}
            for slot, (is_clf, layer) in enumerate(zip(is_probe, read_layers), start=1):
                if is_clf:
                    slots_by_layer.setdefault(layer, []).append(slot)
            self._probe_final_layer_group = None
            self._probe_layer_groups: dict[int, str] = {}
            for layer, slot_ids in slots_by_layer.items():
                name = f"_probe_slots_{'final' if layer is None else layer}"
                self.register_buffer(
                    name, torch.tensor(slot_ids, dtype=torch.long), persistent=False
                )
                if layer is None:
                    self._probe_final_layer_group = name
                else:
                    self._probe_layer_groups[layer] = name
        else:
            self.probe_head = None
            self._probe_final_layer_group = None
            self._probe_layer_groups = {}

        # 3. Base transformer layers with custom LoRA
        #
        # When adapters are present, config.num_hidden_layers includes placeholder
        # entries for the switch's KV cache slots (MultiSwitch uses 2: a counting
        # slot and a memory slot). These placeholders exist for HF DynamicCache
        # sizing; vLLM auto-discovers its Attention layers and doesn't need them.
        # We subtract the switch's cache slot count to recover the true number of
        # decoder layers.
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
        if self.probe_head is not None:
            tensors["probe_indices"] = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            tensors["probe_scores"] = torch.zeros(
                (batch_size, self.config.max_probe_labels),
                dtype=dtype,
                device=device,
            )

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

        else:
            assert intermediate_tensors is not None
            adapter_indices = intermediate_tensors["adapter_indices"]

        # Split the single adapter_indices stream into a LoRA stream and a
        # probe stream. Probe positions are zeroed in lora_indices (so
        # the LoRA path no-ops there).
        if self.switch is not None:
            lora_indices, probe_indices = self.switch.split_indices(adapter_indices)
        else:
            lora_indices = adapter_indices
            probe_indices = torch.zeros_like(adapter_indices)

        # The verdict is read only at the marker itself, not at tokens that
        # inherit its slot afterwards. Later PP ranks have no input ids, so they
        # receive the marker-only indices from the first rank.
        if self.probe_head is not None:
            if not get_pp_group().is_first_rank:
                probe_indices = intermediate_tensors["probe_indices"]
            elif input_ids is not None:
                is_marker = torch.isin(input_ids, self.probe_control_token_ids_t)
                probe_indices = torch.where(
                    is_marker, probe_indices, torch.zeros_like(probe_indices)
                )

        # Rebuild this rank's fixed-size kernel metadata from token-leading
        # adapter ids. The fused kernel routes on the LoRA stream.
        if self.lora_meta is not None and self.lora_ctx is not None:
            self.decoder_interface.prepare_kernel_meta(
                self.lora_meta, lora_indices, self.lora_ctx
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
        # Probe slots reading an intermediate layer are scored as soon as
        # that layer runs; only the small score tensor is carried forward.
        if self.probe_head is not None:
            if get_pp_group().is_first_rank:
                probe_scores = torch.zeros(
                    probe_indices.shape[0],
                    self.config.max_probe_labels,
                    device=probe_indices.device,
                    dtype=hidden_states.dtype,
                )
            else:
                probe_scores = intermediate_tensors["probe_scores"]
        state = self.decoder_interface.enter_decoder_stack(hidden_states, residual)
        for i in range(self.start_layer, self.end_layer):
            state = self.decoder_interface.run_layer(self.layers[i], positions, state)
            group = self._probe_layer_groups.get(i)
            if group is not None:
                probe_scores = self._score_group(
                    group,
                    self.decoder_interface.layer_output(state, lora_indices),
                    probe_indices,
                    probe_scores,
                )

        if get_pp_group().is_last_rank:
            # Adaptation-specific finalize: LoRA folds the last residual via
            # rms_norm_select (fused/separate convention); SR does the per-token
            # where-merge of its two streams then a plain norm. The returned
            # [M, H] tensor is the final post-norm hidden state (what the LM head
            # consumes).
            hidden_states = self.decoder_interface.exit_decoder_stack(
                state, lora_indices, self.norm, self.config
            )

            # Pack scores and slot ids onto the hidden states so vLLM's own
            # logits_indices gather selects them; compute_logits unpacks them.
            if self.probe_head is not None:
                probe_scores, probe_slot = self._compute_probe_outputs(
                    hidden_states, probe_indices, probe_scores
                )
                hidden_states = torch.cat(
                    [hidden_states, probe_scores, probe_slot.unsqueeze(-1)],
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
            intermediate_tensors["adapter_indices"] = adapter_indices
            if self.probe_head is not None:
                intermediate_tensors["probe_indices"] = probe_indices
                intermediate_tensors["probe_scores"] = probe_scores
            return intermediate_tensors

    def _score_group(
        self,
        group: str,
        hidden: torch.Tensor,
        probe_indices: torch.Tensor,
        probe_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Write the scores of the slots in buffer ``group``, read from ``hidden``."""
        in_group = torch.isin(probe_indices, getattr(self, group))
        slot_logits = self.probe_head(hidden, probe_indices)
        return torch.where(in_group.unsqueeze(-1), slot_logits, probe_scores)

    def _compute_probe_outputs(
        self,
        final_hidden_states: torch.Tensor,
        probe_indices: torch.Tensor,
        probe_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add final-layer scores and return per-token (scores, slot ids).

        ``probe_indices`` is nonzero only at this forward's marker tokens,
        computed on the first PP rank and passed unchanged between stages.
        Callers must place one probe marker last in each request;
        misplaced or duplicate markers are not rejected (request-boundary
        validation is deferred).
        """
        probe_slot = probe_indices.to(final_hidden_states.dtype)
        if self._probe_final_layer_group is not None:
            probe_scores = self._score_group(
                self._probe_final_layer_group,
                final_hidden_states,
                probe_indices,
                probe_scores,
            )
        return probe_scores, probe_slot


@MULTIMODAL_REGISTRY.register_processor(
    GraniteSwitchASRMultiModalProcessor,
    info=GraniteSwitchASRProcessingInfo,
    dummy_inputs=GraniteSwitchASRDummyInputsBuilder,
)
# ``IsHybrid``/``HasInnerState`` are deliberately NOT declared, even though the
# config's parent is ``GraniteMoeHybridConfig``. Declaring them would make vLLM
# size the KV cache for ZERO attention layers, because:
#   1. transformers >=5.16 remaps the bare ``"attention"`` layer type to
#      ``"full_attention"``, and no longer accepts ``"attention"`` at all, so
#      ``layer_types`` is ``["full_attention"] * num_hidden_layers`` (config.py).
#   2. ``ModelConfig.is_hybrid``'s granite-4.0-micro escape hatch compares against
#      the *literal* ``"attention"``, so it no longer fires for us.
#   3. ``GraniteMoeHybridConfig.attribute_map`` aliases ``layers_block_type`` to
#      ``layer_types``, pointing vLLM straight at our all-``full_attention`` list.
#   4. ``get_num_layers_by_block_type`` then counts ``t == "attention"`` and gets 0.
# This worked on vLLM 0.19 + transformers 5.9 only because ``layer_types`` was
# literally ``["attention"]`` back then. The switch model is attention-only with
# no mamba state, so neither interface has anything to contribute anyway.
class GraniteSwitchForCausalLM(
    nn.Module,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
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
        """Forward pass returning hidden states (probe columns packed on)."""
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

        Probe exit: the probe request reports its verdict as a
        generated token, like a guardian LoRA. With a probe head the
        hidden states carry packed (scores, slot id) columns; split them off,
        run the LM head on the real hidden state, then rewrite each probe
        row: the whole vocab to -inf and the per-label scores onto the label
        words' token ids (``probe_label_token_ids``). Other rows keep
        their LM logits.
        """
        if self.model.probe_head is not None:
            hidden_size = self.config.hidden_size
            real_hidden = hidden_states[:, :hidden_size]
            verdict = hidden_states[:, hidden_size:-1]
            slot = hidden_states[:, -1]
        else:
            real_hidden = hidden_states
            verdict = slot = None
        logits = self.logits_processor(self.lm_head, real_hidden)
        return self._apply_probe_verdict(logits, verdict, slot)

    def _apply_probe_verdict(
        self,
        logits: torch.Tensor | None,
        verdict: torch.Tensor | None,
        slot: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Rewrite probe requests' logit rows to emit their label word.

        Rows are classified by the slot mask, not just ``slot > 0``: vLLM's
        profiling sampler run feeds random hidden states, so the packed slot
        column can land on a LoRA slot, which has no label ids.
        """
        if logits is None or verdict is None or slot is None:
            return logits

        slot_mask = self.model.probe_slot_mask_t
        num_slots = slot_mask.shape[0] - 1
        slot_int = slot.round().long()
        # Out-of-range values land on the sentinel row 0.
        safe_idx = slot_int.clamp(min=0, max=num_slots)
        is_probe = (slot_int == safe_idx) & slot_mask[safe_idx]
        if not bool(is_probe.any()):
            return logits

        # Each row writes only its own slot's real label count; padded
        # columns are never read.
        per_slot_label_ids = self.config.probe_label_token_ids
        rows = is_probe.nonzero(as_tuple=True)[0]
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
