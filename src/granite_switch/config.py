# SPDX-License-Identifier: Apache-2.0
"""Configuration for Granite model with adapter switching."""

from transformers import GraniteMoeHybridConfig


class GraniteSwitchConfig(GraniteMoeHybridConfig):
    """Configuration class for GraniteSwitch model.

    Extends the Granite base config with parameters for adapter switching
    using the SingleSwitch mechanism. Control tokens are handled exclusively
    via token exchange: the switch reads ``input_ids``, decides the active
    adapter, and rewrites each control token to its substitute id (from
    ``adapter_substitute_token_ids``) before the decoder embeds the
    sequence. The decoder is unaware of the substitution.

    Args:
        num_adapters (int): Number of LoRA adapters available. Default: 0 (no adapters).
            This counts real LoRA adapters only (not base). Index 0 always means "base / no adapter".
        adapter_token_ids (List[int]): Token IDs for adapter control.
            Length: num_adapters (one token per real adapter). Must be unique.
            adapter_token_ids[i] activates adapter i+1 (1-indexed output).
            Output 0 = base (implicit default, no token needed to return to base).
            NOTE: SingleSwitch cannot transition back to base mid-sequence.
        adapter_substitute_token_ids (List[int]): Token IDs whose embeddings
            replace the control-token embeddings before the decoder runs.
            Length: num_adapters. Required when num_adapters > 0.

        SingleSwitch parameters:
            control_token_gain (float): Attention gain for control/non-control separation. Default: 15.0.
            switch_head_dim (int): Dimension of Q/K/V vectors in switch attention. Default: 32.

        adapter_names (List[str]): Ordered adapter names for name-to-index mapping.
        max_lora_rank (int): Maximum rank across all LoRA adapters (for allocation). Default: 8.
        adapter_ranks (List[int]): Per-adapter ranks. Must have length equal to num_adapters.
        lora_target_modules (List[str]): List of module GROUP names to apply LoRA to.
            Module groups: "qkv_proj", "o_proj", "shared_input_linear", "shared_output_linear".
            Default: all four groups

        Audio (ASR) preprocessing parameters:
            asr_enabled (bool): When True, the vLLM backend registers an audio
                multimodal preprocessor that transcribes audio inputs to text and
                splices the transcript tokens into the prompt before the decoder
                runs. The decoder itself is unchanged and only ever sees text
                tokens. Default: False.
            asr_model_id (Optional[str]): HuggingFace id of the speech-to-text
                model the preprocessor loads. When None and asr_enabled is True,
                the backend falls back to a small built-in default. This makes the
                checkpoint self-describing about which ASR front-end it expects.
            asr_device (str): Device the ASR model runs on. Default "cpu" keeps
                vLLM's GPU KV-cache budget clean; set to a CUDA device to trade GPU
                memory for transcription latency.
            asr_pipeline_kwargs (Optional[dict]): Extra keyword arguments merged
                into the ``transformers.pipeline(...)`` construction call for the
                ASR model (e.g. ``{"chunk_length_s": 15}``). These affect how the
                pipeline is built, so they are baked into the transcriber cache
                key. None means "no extras". Default: None.
            asr_generate_kwargs (Optional[dict]): Default decode-time keyword
                arguments passed to the ASR model on every transcription (e.g.
                ``{"language": "de", "task": "transcribe"}`` for a multilingual
                Whisper). Applied at call time, so a single loaded pipeline can be
                reused; per-request values (via ``mm_processor_kwargs``) override
                these. Ignored by models that do not generate (e.g. CTC). Default:
                None.
            asr_max_audio_clips (int): Maximum number of audio clips accepted in a
                single request. Each clip's transcript is spliced at its own
                ``<|audio|>`` marker. vLLM enforces this as the modality ceiling
                (``--limit-mm-per-prompt`` may lower it, not raise it). For the
                cascade the clips cost no extra KV (transcripts are ordinary text
                tokens bounded by the context window); the ceiling mainly guards
                against one request triggering an unbounded number of synchronous
                ASR transcriptions, and bounds the startup profiling pass. Default:
                32.
            asr_generation_reserve_tokens (int): Tokens held back from the context
                window for the model's generated answer (and prompt overhead) when
                computing how many transcript tokens the audio may occupy. The
                per-request audio budget is roughly
                ``max_model_len - asr_generation_reserve_tokens - prompt_tokens``,
                split across the request's clips. Replaces the old fixed 2048-token
                transcript cap with a context-derived one. Default: 8192.
            asr_chunk_length_s (float): Window length (seconds) our own long-audio
                chunker splits a clip into before transcribing each window. Only
                used for backends that do not self-chunk (see asr_self_chunks); the
                default Whisper backend chunks internally and ignores this. Default:
                30.0.
            asr_chunk_overlap_s (float): Overlap (seconds) between consecutive
                chunker windows, so words straddling a boundary are whole in at
                least one window; the transcript merge de-duplicates the overlap.
                Only used when asr_self_chunks is False. Default: 5.0.
            asr_self_chunks (bool): Whether the ASR backend handles long audio
                itself. True for the Whisper pipeline (its internal
                ``chunk_length_s`` does timestamp-based stitching, higher quality
                than our text-level merge), so our chunker is bypassed. Set False
                for a backend with a fixed input window (e.g. a future speech
                encoder) to route audio through our encoder-agnostic
                split/transcribe/merge chunker. Default: True.
        **kwargs: Additional arguments passed to GraniteConfig.
    """

    model_type = "granite_switch"

    def __init__(
        self,
        num_adapters: int = 0,
        adapter_token_ids: list[int] | None = None,
        adapter_substitute_token_ids: list[int] | None = None,
        # SingleSwitch parameters
        control_token_gain: float = 15.0,
        switch_head_dim: int = 32,
        # Adapter parameters
        adapter_names: list[str] | None = None,
        max_lora_rank: int = 8,
        adapter_ranks: list[int] | None = None,
        lora_target_modules: list[str] | None = None,
        # Audio (ASR) preprocessing parameters
        asr_enabled: bool = False,
        asr_model_id: str | None = None,
        asr_device: str = "cpu",
        asr_pipeline_kwargs: dict | None = None,
        asr_generate_kwargs: dict | None = None,
        asr_max_audio_clips: int = 32,
        asr_generation_reserve_tokens: int = 8192,
        asr_chunk_length_s: float = 30.0,
        asr_chunk_overlap_s: float = 5.0,
        asr_self_chunks: bool = True,
        # vLLM residual-norm convention (for bit-exact skinning equivalence)
        fused_add_norm: bool = False,
        # Parent class defaults (Granite 4 dense configuration)
        num_local_experts: int = 0,
        position_embedding_type: str = "rope",
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        # Compute default layer_types before parent init.
        # layer_types must have length == num_hidden_layers (includes switch layer at
        # index 0 when adapters are present). This ensures DynamicCache pre-allocation
        # matches the global layer indices used by decoder layers.
        if layer_types is None:
            num_hidden_layers = kwargs.get("num_hidden_layers", 32)
            layer_types = ["attention"] * num_hidden_layers

        super().__init__(
            num_local_experts=num_local_experts,
            position_embedding_type=position_embedding_type,
            layer_types=layer_types,
            **kwargs,
        )

        # Default shared_intermediate_size from intermediate_size.
        # All Granite 4 models use shared_mlp naming; for dense models
        # shared_intermediate_size == intermediate_size.
        if self.shared_intermediate_size is None:
            self.shared_intermediate_size = self.intermediate_size

        # Validate num_adapters
        if num_adapters < 0:
            raise ValueError(f"num_adapters must be >= 0, got {num_adapters}")
        self.num_adapters = num_adapters

        # Validate adapter_token_ids if provided
        if num_adapters > 0 and adapter_token_ids is not None:
            if len(adapter_token_ids) != num_adapters:
                raise ValueError(
                    f"adapter_token_ids length ({len(adapter_token_ids)}) must equal "
                    f"num_adapters ({num_adapters})."
                )
            # Token-exchange builds the control→substitute LUT keyed by adapter token id;
            # duplicates would silently collapse to a single slot.
            if len(set(adapter_token_ids)) != len(adapter_token_ids):
                raise ValueError(
                    f"adapter_token_ids must be unique; got {adapter_token_ids}"
                )
        self.adapter_token_ids = adapter_token_ids

        # Validate adapter_substitute_token_ids — required when num_adapters > 0.
        if num_adapters > 0:
            if adapter_substitute_token_ids is None:
                raise ValueError(
                    "adapter_substitute_token_ids is required when num_adapters > 0. "
                    "Every adapter needs a substitute token id whose embedding replaces "
                    "the control-token embedding before the decoder runs."
                )
            if len(adapter_substitute_token_ids) != num_adapters:
                raise ValueError(
                    f"adapter_substitute_token_ids length "
                    f"({len(adapter_substitute_token_ids)}) must equal num_adapters "
                    f"({num_adapters})."
                )
            if any(sid < 0 for sid in adapter_substitute_token_ids):
                raise ValueError(
                    f"adapter_substitute_token_ids must all be >= 0 (real token ids); "
                    f"got {adapter_substitute_token_ids}"
                )
            if adapter_token_ids is None:
                raise ValueError(
                    "adapter_token_ids is required when adapter_substitute_token_ids "
                    "is provided (token-exchange maps control ids to substitute ids)."
                )
        self.adapter_substitute_token_ids = adapter_substitute_token_ids

        # SingleSwitch parameters
        self.control_token_gain = control_token_gain
        self.switch_head_dim = switch_head_dim
        self.fused_add_norm = fused_add_norm

        # Audio (ASR) preprocessing parameters.
        # The decoder is oblivious to audio: when enabled, the vLLM backend's
        # multimodal preprocessor transcribes audio to text and injects the
        # transcript tokens into the prompt before embedding. These fields make
        # the checkpoint self-describing about its ASR front-end.
        self.asr_enabled = asr_enabled
        self.asr_model_id = asr_model_id
        self.asr_device = asr_device
        # Pipeline-construction extras (affect the built pipeline → cache key)
        # and default decode kwargs (applied per call, per-request overridable).
        self.asr_pipeline_kwargs = asr_pipeline_kwargs
        self.asr_generate_kwargs = asr_generate_kwargs
        # Long-audio / multi-clip preprocessing. The transcript token budget is
        # derived from the context window at runtime (max_model_len minus the
        # generation reserve, split across clips) rather than a fixed cap; the
        # chunker settings only apply to backends that do not self-chunk.
        if asr_max_audio_clips < 1:
            raise ValueError(
                f"asr_max_audio_clips must be >= 1, got {asr_max_audio_clips}"
            )
        if asr_generation_reserve_tokens < 0:
            raise ValueError(
                f"asr_generation_reserve_tokens must be >= 0, got "
                f"{asr_generation_reserve_tokens}"
            )
        if asr_chunk_overlap_s >= asr_chunk_length_s:
            raise ValueError(
                f"asr_chunk_overlap_s ({asr_chunk_overlap_s}) must be < "
                f"asr_chunk_length_s ({asr_chunk_length_s})"
            )
        self.asr_max_audio_clips = asr_max_audio_clips
        self.asr_generation_reserve_tokens = asr_generation_reserve_tokens
        self.asr_chunk_length_s = asr_chunk_length_s
        self.asr_chunk_overlap_s = asr_chunk_overlap_s
        self.asr_self_chunks = asr_self_chunks

        # Adapter names
        self.adapter_names = adapter_names

        # Projection head dimension.
        # The QKV projection outputs vectors of size projection_head_dim
        # (= hidden_size / num_attention_heads). The KV cache stores native-
        # head_dim tensors — no expansion under token exchange.
        # We do NOT set head_dim here because HF's RoPE also reads it.
        # Use explicit head_dim from kwargs when available (some models have
        # head_dim != hidden_size // num_attention_heads).
        explicit_head_dim = kwargs.get("head_dim")
        self.projection_head_dim = (
            explicit_head_dim
            if explicit_head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )

        # Validate and store adapter configuration
        if num_adapters > 0:
            if adapter_ranks is None:
                raise ValueError("adapter_ranks must be provided when num_adapters > 0")

            if len(adapter_ranks) != num_adapters:
                raise ValueError(
                    f"adapter_ranks length ({len(adapter_ranks)}) must equal num_adapters ({num_adapters})"
                )

            if max(adapter_ranks) != max_lora_rank:
                raise ValueError(
                    f"max(adapter_ranks)={max(adapter_ranks)} must equal max_lora_rank={max_lora_rank}"
                )

        self.max_lora_rank = max_lora_rank
        self.adapter_ranks = adapter_ranks

        # Default LoRA target module groups.
        # Dynamically determined based on model architecture.
        # Empty when num_adapters == 0 (no LoRA to apply).
        if lora_target_modules is None:
            lora_target_modules = []

            if self.num_adapters > 0:
                # Attention modules (present in all attention layers)
                if any(lt == "attention" for lt in self.layer_types):
                    lora_target_modules.extend(
                        [
                            "qkv_proj",  # Q/K/V fused
                            "o_proj",  # O projection
                        ]
                    )

                # MLP modules: all Granite 4 models use shared_mlp naming
                lora_target_modules.extend(
                    [
                        "shared_input_linear",  # shared_mlp input_linear (fused gate+up)
                        "shared_output_linear",  # shared_mlp output_linear
                    ]
                )

        self.lora_target_modules = lora_target_modules
