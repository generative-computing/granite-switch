# SPDX-License-Identifier: Apache-2.0
"""Configuration for Granite model with adapter switching."""

from transformers import GraniteMoeHybridConfig

# Accepted asr_dtype values. Keep in sync with vllm.audio.asr._ASR_DTYPE_NAMES.
ASR_DTYPES = ("auto", "float16", "bfloat16", "float32")


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

        Audio (ASR) preprocessing parameters (see docs/AUDIO.md):
            asr_enabled (bool): Register the audio preprocessor that transcribes
                audio and splices the transcript into the prompt. Default: False.
            asr_model_id (Optional[str]): HF id of the speech-to-text model. None
                falls back to a small built-in default.
            asr_device (str): Device the ASR model runs on. Default "cpu" keeps
                vLLM's GPU KV-cache budget clean.
            asr_dtype (Optional[str]): Precision the ASR weights load in, one of
                ASR_DTYPES. None/"auto" derives it from asr_device (float16 on
                CUDA). An encoder with BatchNorm layers must set "float32".
                Default: None.
            asr_pipeline_kwargs (Optional[dict]): Extra kwargs merged into the
                ``transformers.pipeline(...)`` construction, e.g.
                ``{"chunk_length_s": 15}``. Baked into the transcriber cache key.
                Default: None.
            asr_generate_kwargs (Optional[dict]): Default decode-time kwargs, e.g.
                ``{"language": "de"}``. Applied per call, so one pipeline is
                reused; per-request ``mm_processor_kwargs`` override them. Ignored
                by non-generative backends. Default: None.
            asr_max_audio_clips (int): Max audio clips per request. Bounds the
                synchronous transcriptions one request can trigger and the startup
                profiling pass; ``--limit-mm-per-prompt`` may lower it, not raise
                it. Default: 32.
            asr_chunk_length_s (float): Chunker window length in seconds. Only
                used when asr_self_chunks is False. Default: 30.0.
            asr_chunk_overlap_s (float): Overlap in seconds between chunker
                windows, de-duplicated by the transcript merge. Only used when
                asr_self_chunks is False. Default: 5.0.
            asr_self_chunks (bool): True when the backend chunks long audio
                itself (Whisper's timestamp stitching beats our text-level merge),
                bypassing our chunker. False routes audio through the
                split/transcribe/merge chunker instead. Default: True.
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
        asr_dtype: str | None = None,
        asr_pipeline_kwargs: dict | None = None,
        asr_generate_kwargs: dict | None = None,
        asr_max_audio_clips: int = 32,
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

        # Audio (ASR) preprocessing. The decoder is oblivious to audio; these
        # fields make the checkpoint self-describing about its ASR front-end.
        self.asr_enabled = asr_enabled
        self.asr_model_id = asr_model_id
        self.asr_device = asr_device
        # Validated here so a typo fails at compose time, not in a vLLM worker.
        if asr_dtype is not None and asr_dtype not in ASR_DTYPES:
            raise ValueError(
                f"asr_dtype must be one of {ASR_DTYPES} or None, got {asr_dtype!r}"
            )
        self.asr_dtype = asr_dtype
        self.asr_pipeline_kwargs = asr_pipeline_kwargs
        self.asr_generate_kwargs = asr_generate_kwargs
        if asr_max_audio_clips < 1:
            raise ValueError(
                f"asr_max_audio_clips must be >= 1, got {asr_max_audio_clips}"
            )
        if asr_chunk_overlap_s >= asr_chunk_length_s:
            raise ValueError(
                f"asr_chunk_overlap_s ({asr_chunk_overlap_s}) must be < "
                f"asr_chunk_length_s ({asr_chunk_length_s})"
            )
        self.asr_max_audio_clips = asr_max_audio_clips
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
