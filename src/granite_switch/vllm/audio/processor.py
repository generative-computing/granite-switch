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

# Upper bound on transcript tokens per audio item, used for startup memory
# profiling (get_mm_max_tokens_per_item). Generous enough for a long clip; the
# real transcript is usually far shorter. ~ a few minutes of dense speech.
_MAX_TRANSCRIPT_TOKENS = 2048

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
        # One audio clip per request for the alpha.
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        if not self._asr_enabled():
            return {}
        # Bound the transcript length so vLLM can size the KV cache at startup
        # instead of running dummy audio through ASR at max size.
        return {"audio": _MAX_TRANSCRIPT_TOKENS}

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

    def _transcribe(self, audio, generate_kwargs: Optional[Mapping[str, object]] = None) -> list[int]:
        """Transcribe one audio item to a list of token ids."""
        transcriber = get_transcriber(
            model_id=self.info._asr_model_id(),
            device=self.info._asr_device(),
            pipeline_kwargs=self.info._asr_pipeline_kwargs(),
        )
        # The data parser already resampled to _TARGET_SR.
        text = transcriber.transcribe(
            audio, sampling_rate=_TARGET_SR, generate_kwargs=generate_kwargs or None
        )
        tokenizer = self.info.get_tokenizer()
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Guard the profiling bound.
        return ids[:_MAX_TRANSCRIPT_TOKENS]

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

        # Transcribe each audio to token ids; concatenate flat with per-item sizes.
        per_item_ids = [self._transcribe(a, generate_kwargs) for a in audios]
        sizes = [len(ids) for ids in per_item_ids]
        flat_ids = [tid for ids in per_item_ids for tid in ids]

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)

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
