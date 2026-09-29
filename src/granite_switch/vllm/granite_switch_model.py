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

        # Classifier heads: optional alternative for LoRA adapters. A classifier
        # slot shares the adapter index/token machinery, but its index is tagged
        # as "classifier" (via the switch's adapter_kind_lut). The switch's
        # split_indices() routes those positions out of the LoRA stream (so LoRA
        # no-ops there) and into a classifier stream, which this head consumes
        # from the final post-norm hidden state.
        if "classifier" in getattr(config, "adapter_kinds", []):
            from .core import SwitchedClassifierHead

            self.classifier_head = SwitchedClassifierHead(
                hidden_size=config.hidden_size,
                num_classifier_slots=config.num_adapters,
                max_num_labels=config.max_classifier_labels,
            )
        else:
            self.classifier_head = None

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

            # Split the single adapter_indices stream into a LoRA stream and a
            # classifier stream. Classifier positions are zeroed in lora_indices
            # (so the LoRA path no-ops there); the classifier head reads them on
            # the last rank via classifier_indices.
            if self.switch is not None:
                lora_indices, classifier_indices = self.switch.split_indices(
                    adapter_indices
                )
            else:
                lora_indices = adapter_indices
                classifier_indices = torch.zeros_like(adapter_indices)

            # Prepare kernel metadata ONCE for all decoder layers
            # (adaptation-specific). The LoRA stream (classifier positions zeroed)
            # is what the fused kernel routes on, so classifier tokens no-op the
            # LoRA path.
            if self.lora_meta is not None and self.lora_ctx is not None:
                self.decoder_interface.prepare_kernel_meta(
                    self.lora_meta, lora_indices, self.lora_ctx
                )

            # Store metadata in intermediate_tensors for pipeline parallelism.
            # Only adapter_indices is propagated; classifier_indices is derived
            # from it on the last rank via split_indices (no new PP tensor).
            if intermediate_tensors is None:
                intermediate_tensors = IntermediateTensors({})
            intermediate_tensors["adapter_indices"] = adapter_indices
        else:
            # Subsequent ranks: recompute fixed-size LoRA metadata from
            # token-leading adapter_indices received through PP.
            if intermediate_tensors is not None:
                adapter_indices = intermediate_tensors["adapter_indices"]
                # Recompute the split from the PP-propagated stream so this
                # rank's Punica metadata sees classifier positions as base.
                if self.switch is not None:
                    lora_indices, classifier_indices = self.switch.split_indices(
                        adapter_indices
                    )
                else:
                    lora_indices = adapter_indices
                    classifier_indices = torch.zeros_like(adapter_indices)

                if self.lora_ctx is not None:
                    # Route on the LoRA stream (classifier positions zeroed) so
                    # classifier tokens no-op the LoRA path on this PP rank too.
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
        state = self.decoder_interface.enter_decoder_stack(hidden_states, residual)
        for i in range(self.start_layer, self.end_layer):
            state = self.decoder_interface.run_layer(self.layers[i], positions, state)

        if get_pp_group().is_last_rank:
            # Adaptation-specific finalize: LoRA folds the last residual via
            # rms_norm_select (fused/separate convention); SR does the per-token
            # where-merge of its two streams then a plain norm. The returned
            # [M, H] tensor is the final post-norm hidden state (what the LM head
            # consumes).
            hidden_states = self.decoder_interface.exit_decoder_stack(
                state, adapter_indices, self.norm, self.config
            )

            if self.lora_ctx is not None:
                self.lora_ctx.classifier_indices = classifier_indices

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

    def _classifier_read_points(
        self,
        classifier_indices: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-request (hidden_state, slot_index) for the classifier verdict.

        The verdict is read at the LAST CONTENT TOKEN, which the classifier
        control token (marker) locates: the chat template emits the marker
        immediately after the last content token, so the read point is
        ``marker - 1``.

        The marker is the LAST classifier control token in a request's slice,
        matched by id in ``input_ids``; ``classifier_indices`` is nonzero from any
        classifier marker onward, so the index alone does not say which slot a
        position belongs to. The read point is ``marker - 1`` and the slot id is
        read AT the marker (``marker - 1`` carries index 0).

        Per request slice ``[start, end)``, three states:

        * marker present with a predecessor in this pass -- the common case.
        * marker opens the slice: its predecessor ended the previous pass, so each
          pass stashes its final hidden row for a single-token look-back.
        * no marker: either still mid-prefill, or the marker was in an earlier
          chunk and that pass's resolved verdict is carried forward, since vLLM
          consumes the verdict only on the pass it samples.

        Returns ``None`` when there is no request metadata (e.g. the startup
        profiling forward), so the caller emits no verdict.
        """
        from vllm.forward_context import get_forward_context

        attn_metadata = get_forward_context().attn_metadata
        # v1 keys this by layer name (a dict), or a list of such dicts under
        # microbatching. They all share query_start_loc, so grab any one.
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0] if attn_metadata else None
        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()), None)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        if query_start_loc is None:
            return None

        qsl = query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()

        # A request's marker is the LAST classifier control token in its slice:
        # ``classifier_indices`` is nonzero from any classifier marker onward, so
        # with several classifier slots the index alone does not say which slot's
        # marker a position belongs to.
        if input_ids is None:
            raise RuntimeError(
                "A classifier request needs input_ids to locate its control token, "
                "but only inputs_embeds was provided. Classifier slots classify a "
                "text prompt; pass input_ids."
            )
        marker_ids = torch.tensor(
            self.config.classifier_control_token_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        is_marker = torch.isin(input_ids, marker_ids)  # [total_tokens]

        stash = getattr(self, "_classifier_prev_pass_tail", None) or {}
        resolved = getattr(self, "_classifier_resolved_verdict", None) or {}
        req_hidden_rows = []
        req_slots = []
        new_stash: dict[int, torch.Tensor] = {}
        new_resolved: dict[int, tuple[torch.Tensor, int]] = {}
        for i in range(len(qsl) - 1):
            start, end = qsl[i], qsl[i + 1]
            hits = is_marker[start:end].nonzero(as_tuple=True)[0]
            if hits.numel() == 0:
                # No marker in this slice, which is either of two states:
                #
                #  * the marker is still ahead (mid-prefill): keep this slice's final
                #    row, keyed by its global end position, in case the marker opens
                #    the next chunk.
                #  * the marker was in an earlier chunk (a rendered prompt continues
                #    past it with the generation prompt): carry that pass's resolved
                #    (row, slot) forward. vLLM consumes the verdict only on the pass
                #    where it samples, which is the final chunk.
                carried = resolved.get(seq_lens[i] - (end - start))
                if carried is not None:
                    row, slot = carried
                    new_resolved[seq_lens[i]] = (row, slot)
                    req_hidden_rows.append(row)
                    req_slots.append(slot)
                    continue
                new_stash[seq_lens[i]] = hidden_states[end - 1]
                req_hidden_rows.append(hidden_states[end - 1])
                req_slots.append(0)
                continue

            # One verdict per request: the exit rewrites a single logit row, so a
            # request naming two different classifier slots has no way to report
            # both. Repeated markers of the SAME slot are fine (a multi-turn prompt
            # carries them from earlier turns); the last one wins.
            distinct = torch.unique(input_ids[start:end][hits])
            if distinct.numel() > 1:
                raise RuntimeError(
                    "A classifier request may name only one classifier slot; "
                    f"request {i} carries control tokens "
                    f"{sorted(int(t) for t in distinct)}. The verdict exit rewrites "
                    "one logit row, so only one slot can report."
                )

            marker = start + int(hits[-1])  # last id match: this slice's marker
            slot = int(classifier_indices[marker])
            req_slots.append(slot)
            if marker > start:
                # Common case: predecessor is in this pass.
                row = hidden_states[marker - 1]
                new_resolved[seq_lens[i]] = (row, slot)
                req_hidden_rows.append(row)
                continue

            # Edge case: marker is the first token of its slice
            marker_global = seq_lens[i] - (end - start)
            prev = stash.get(marker_global)
            if prev is None:
                raise RuntimeError(
                    "Classifier verdict cannot locate its read point: the marker "
                    f"is the first token of request {i}'s chunk (global position "
                    f"{marker_global}), so the last content token (marker - 1) was "
                    "computed in the previous pass, and no stashed hidden row is "
                    "available for it."
                )
            new_resolved[seq_lens[i]] = (prev, slot)
            req_hidden_rows.append(prev)

        self._classifier_prev_pass_tail = new_stash
        self._classifier_resolved_verdict = new_resolved
        return (
            torch.stack(req_hidden_rows, dim=0),
            torch.tensor(req_slots, dtype=torch.long, device=hidden_states.device),
        )


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
        """Forward pass returning hidden states."""
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        self._pending_verdict = None
        classifier_head = getattr(self.model, "classifier_head", None)
        if classifier_head is not None and isinstance(hidden_states, torch.Tensor):
            classifier_indices = getattr(
                self.model.lora_ctx, "classifier_indices", None
            )
            if classifier_indices is not None:
                read = self.model._classifier_read_points(
                    classifier_indices, hidden_states, input_ids
                )
                if read is not None:
                    req_hidden, req_classifier_indices = read
                    verdict = classifier_head(
                        req_hidden, req_classifier_indices
                    )  # [num_reqs, num_labels]
                    self._pending_verdict = (verdict, req_classifier_indices)

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

        Classifier exit: the classifier request reports its verdict as a generated
        token, like a guardian LoRA. For each classifier request we rewrite its logit
        row: set the whole vocab to -inf, then place the per-label scores on each
        label word's token id (``classifier_label_token_ids``) so the sampler
        emits that word. Non-classifier rows keep their normal LM logits.
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return self._apply_classifier_verdict(logits)

    def _apply_classifier_verdict(
        self,
        logits: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Rewrite classifier requests' logit rows to emit their label word."""
        pending = getattr(self, "_pending_verdict", None)
        if logits is None or pending is None:
            return logits
        # Consume it so a same-step prompt_logprobs pass re-applies nothing.
        self._pending_verdict = None
        verdict, req_indices = pending

        per_slot_label_ids = self.config.classifier_label_token_ids

        if verdict.shape[0] != logits.shape[0]:
            raise RuntimeError(
                f"classifier verdict rows ({verdict.shape[0]}) do not match "
                f"logits rows ({logits.shape[0]}); per-request alignment broke."
            )

        is_classifier = req_indices > 0  # [num_reqs]
        if not bool(is_classifier.any()):
            return logits

        # Each request writes only its own slot's label ids and only that slot's
        # real label count (verdict[:, :n]); padded columns are never read.
        rows = is_classifier.nonzero(as_tuple=True)[0]  # [num_classifier_reqs]
        logits[rows] = float("-inf")
        for r in rows.tolist():
            slot = int(req_indices[r])
            ids = per_slot_label_ids[slot - 1]
            n = len(ids)
            label_ids = torch.tensor(ids, dtype=torch.long, device=logits.device)
            logits[r, label_ids] = verdict[r, :n].to(logits.dtype)
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
