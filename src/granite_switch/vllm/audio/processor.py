# SPDX-License-Identifier: Apache-2.0
"""vLLM multimodal processor for the alpha audio cascade (Path A).

Strategy (see also the design notes in :mod:`granite_switch.vllm.audio`):

* Audio is declared as a multimodal input, so the standard vLLM server accepts
  it and developers keep loading a single model.
* In :meth:`_call_hf_processor` we run ASR and tokenize the transcript.
* :meth:`_get_prompt_updates` replaces the single ``<|audio|>`` marker with the
  **actual transcript token ids** via ``PromptReplacement``. After this the
  transcript tokens are ordinary tokens in ``prompt_token_ids``; the scheduler
  sizes KV for the real, runtime-determined length (no fixed audio window).
* The model's ``embed_multimodal`` then returns ``embed_tokens(transcript_ids)``
  for those placeholder positions — identical to what they'd get as normal text.
  That redundant-but-consistent step is the exact seam the future projection
  model reuses (swap the embedding source for a trained audio encoder).

Modeled on vLLM 0.19.1's ``ultravox.py`` (the reference audio model), confirmed
against that version's API by the scratch probes.

ALPHA SCOPE: audio is answered by the base model. We do not place adapter
control tokens for audio requests, so no token-exchange interaction with the
switch is needed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from transformers import BatchFeature

from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataDict,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    BaseDummyInputsBuilder,
    PromptReplacement,
    PromptUpdate,
)

from .asr import (
    DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
    DEFAULT_ASR_MODEL_ID,
    get_transcriber,
    resolve_generate_kwargs,
)

# The chat-template marker that stands in for an audio clip before replacement.
AUDIO_MARKER = "<|audio|>"

# ASR feature-extractor sample rate the audio is resampled to.
_TARGET_SR = 16_000

# Fallback context length if the served max_model_len cannot be read (should not
# happen in practice; keeps the transcript budget finite either way).
_FALLBACK_CONTEXT_LEN = 8192

# Dummy clip length (seconds) used during vLLM's profiling run.
_DUMMY_AUDIO_SECONDS = 5


class GraniteSwitchASRProcessingInfo(BaseProcessingInfo):
    """Static info vLLM needs about the audio modality."""

    def _asr_enabled(self) -> bool:
        return bool(getattr(self.get_hf_config(), "asr_enabled", False))

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Audio capability is gated per-checkpoint: a non-audio GraniteSwitch
        # reports no modalities, so vLLM never profiles audio or loads ASR.
        if not self._asr_enabled():
            return {}
        # Configurable ceiling on audio clips per request; each clip's transcript
        # is spliced at its own <|audio|> marker. Finite so vLLM can size KV for
        # the worst case (clips x per-clip budget). --limit-mm-per-prompt may
        # lower this, not raise it above the declared ceiling.
        return {"audio": self._asr_max_audio_clips()}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        if not self._asr_enabled():
            return {}
        # Worst-case transcript positions one clip can occupy, used only to size
        # the encoder cache and profiling pass — NOT to bound requests. A clip
        # cannot transcribe to more than the whole context (a longer prompt is
        # rejected by vLLM's standard length check), so the per-clip share of the
        # context window is the honest upper bound. vLLM calls this with count=1.
        count = mm_counts.get("audio", 1) or 1
        return {"audio": max(1, seq_len // count)}

    def get_data_parser(self) -> MultiModalDataParser:
        # Resample incoming audio to the ASR sample rate.
        return MultiModalDataParser(target_sr=_TARGET_SR)

    # --- ASR config resolved from the model's GraniteSwitchConfig ---

    def _asr_model_id(self) -> str:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_model_id", None) or DEFAULT_ASR_MODEL_ID

    def _asr_device(self) -> str:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_device", "cpu") or "cpu"

    def _asr_pipeline_kwargs(self) -> Mapping[str, object]:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_pipeline_kwargs", None) or {}

    def _asr_generate_kwargs(self) -> Mapping[str, object]:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_generate_kwargs", None) or {}

    def _asr_max_audio_clips(self) -> int:
        cfg = self.get_hf_config()
        return int(getattr(cfg, "asr_max_audio_clips", 32) or 32)

    def _asr_self_chunks(self) -> bool:
        cfg = self.get_hf_config()
        return bool(getattr(cfg, "asr_self_chunks", True))

    def _asr_chunk_length_s(self) -> float:
        cfg = self.get_hf_config()
        return float(getattr(cfg, "asr_chunk_length_s", 30.0) or 30.0)

    def _asr_chunk_overlap_s(self) -> float:
        cfg = self.get_hf_config()
        return float(getattr(cfg, "asr_chunk_overlap_s", 5.0) or 0.0)

    def _max_model_len(self) -> int:
        """The served context window, or a safe fallback.

        vLLM passes ``seq_len`` into ``get_mm_max_tokens_per_item`` for profiling,
        but ``_call_hf_processor`` needs the served context at request time to size
        the transcript budget, so read it from the processing context's model
        config (falling back to the checkpoint's max_position_embeddings, then a
        constant).
        """
        model_config = getattr(self.ctx, "model_config", None)
        max_len = getattr(model_config, "max_model_len", None)
        if not max_len:
            max_len = getattr(self.get_hf_config(), "max_position_embeddings", None)
        return int(max_len) if max_len else _FALLBACK_CONTEXT_LEN


class GraniteSwitchASRDummyInputsBuilder(
    BaseDummyInputsBuilder[GraniteSwitchASRProcessingInfo]
):
    """Synthetic inputs for vLLM's startup memory-profiling pass."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_MARKER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Optional[Mapping[str, object]] = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        length = _DUMMY_AUDIO_SECONDS * _TARGET_SR
        audio = torch.zeros(length, dtype=torch.float32).numpy()
        return {"audio": [audio] * num_audios}


class GraniteSwitchASRMultiModalProcessor(
    BaseMultiModalProcessor[GraniteSwitchASRProcessingInfo]
):
    """Runs ASR and splices the transcript tokens into the prompt."""

    def _transcribe(
        self,
        audio,
        generate_kwargs: Optional[Mapping[str, object]] = None,
    ) -> list[int]:
        """Transcribe one audio item to a list of token ids (full transcript).

        The transcript is spliced into the prompt as ordinary text tokens; it is
        never truncated here. A clip whose transcript makes the prompt exceed the
        context is rejected by vLLM's standard prompt-length check.

        Long audio is handled by the backend itself (Whisper) or by our
        encoder-agnostic chunker, per the checkpoint's ``asr_self_chunks`` flag.
        """
        transcriber = get_transcriber(
            model_id=self.info._asr_model_id(),
            device=self.info._asr_device(),
            pipeline_kwargs=self.info._asr_pipeline_kwargs(),
        )
        # The data parser already resampled to _TARGET_SR.
        text = transcriber.transcribe(
            audio,
            sampling_rate=_TARGET_SR,
            generate_kwargs=generate_kwargs or None,
            self_chunks=self.info._asr_self_chunks(),
            chunk_length_s=self.info._asr_chunk_length_s(),
            chunk_overlap_s=self.info._asr_chunk_overlap_s(),
        )
        tokenizer = self.info.get_tokenizer()
        return tokenizer.encode(text, add_special_tokens=False)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        audios = mm_data.get("audios", []) or []

        # Text-only request: just tokenize (mirrors Ultravox's text-only branch).
        if not audios:
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return BatchFeature(dict(input_ids=[input_ids]), tensor_type="pt")

        # Resolve decode kwargs once (config defaults + allowlisted per-request
        # overrides) and apply them to every audio item in this request.
        generate_kwargs = resolve_generate_kwargs(
            self.info._asr_generate_kwargs(),
            mm_kwargs,
            DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
        )

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Transcribe each audio to token ids; concatenate flat with per-item sizes.
        # Transcripts are spliced in full — an oversized request is rejected by
        # vLLM's prompt-length check, not silently truncated here.
        per_item_ids = [self._transcribe(a, generate_kwargs) for a in audios]
        sizes = [len(ids) for ids in per_item_ids]
        flat_ids = [tid for ids in per_item_ids for tid in ids]

        return BatchFeature(
            dict(
                input_ids=[input_ids],
                audio_token_ids=torch.tensor(flat_ids, dtype=torch.long),
                audio_num_tokens=torch.tensor(sizes, dtype=torch.long),
            ),
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_tokens = hf_inputs.get("audio_num_tokens", torch.zeros(0))
        return dict(
            # Flat transcript ids, split back per audio by audio_num_tokens.
            audio_token_ids=MultiModalFieldConfig.flat_from_sizes("audio", num_tokens),
            audio_num_tokens=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        out = out_mm_kwargs.get_data()
        num_tokens = out.get("audio_num_tokens", torch.zeros(0))
        starts = torch.cumsum(num_tokens, dim=0, dtype=torch.long)
        starts = torch.cat([torch.tensor([0], dtype=torch.long), starts])
        all_ids = out.get("audio_token_ids", torch.zeros(0, dtype=torch.long))

        def replacement(item_idx: int):
            s = int(starts[item_idx])
            e = int(starts[item_idx + 1])
            # Replace <|audio|> with the real transcript token ids.
            return [int(t) for t in all_ids[s:e]]

        return [
            PromptReplacement(
                modality="audio",
                target=AUDIO_MARKER,
                replacement=replacement,
            )
        ]
