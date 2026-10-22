# SPDX-License-Identifier: Apache-2.0
"""vLLM multimodal processor for the audio cascade.

ASR runs in the HF-processor hook and tokenizes the transcript;
``_get_prompt_updates`` then replaces the ``<|audio|>`` marker with the real
transcript token ids via ``PromptReplacement``, so the scheduler sizes KV for the
runtime-determined length rather than a fixed audio window.

Upstream has reshaped that hook surface three times, and the eras are NOT ordered
by whether a given name exists -- 0.31 brought ``_call_hf_processor`` back with a
different signature after 0.29 deleted it:

===  =========  ==========================================================
era  vLLM       base-class shape
===  =========  ==========================================================
A    0.26-0.28  ``_call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)``
                owns tokenization AND mm data; ``_get_hf_mm_data(mm_items)``
B    0.29-0.30  ``_call_hf_processor`` DELETED; text split out of mm, so
                ``_apply_hf_processor_main(mm_items, hf_processor_mm_kwargs)``
                returns mm tensors only; ``_get_hf_mm_data(mm_items)``
C    0.31-0.32  ``_call_hf_processor(hf_data, hf_kwargs)`` is BACK, but it is
                now called *by* ``_apply_hf_processor_main(mm_items,
                hf_kwargs)`` -- note the renamed kwarg. ``_get_hf_mm_data`` is
                gone, replaced by ``_get_hf_mm_inputs``/``_finalize_hf_mm_data``
===  =========  ==========================================================

So the gate here keys on the *shape* of the base hook, never on a version number
and never on bare ``hasattr``. See ``_base_hook_era``.

Modeled on vLLM 0.19.1's ``ultravox.py``. Audio is answered by the base model —
no adapter control tokens are placed, so the switch is not involved.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence

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
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from .asr import (
    DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
    DEFAULT_ASR_MODEL_ID,
    get_transcriber,
    resolve_driver_device,
    resolve_generate_kwargs,
)

AUDIO_MARKER = "<|audio|>"
_TARGET_SR = 16_000
# Hook-surface eras; see the module docstring. Ordered only for readability --
# nothing compares them, because they are not a version ordering.
_ERA_CALL_HF = "call_hf"  # A: 0.26-0.28
_ERA_MAIN_ONLY = "main_only"  # B: 0.29-0.30
_ERA_CALL_HF_MM = "call_hf_mm"  # C: 0.31+
# Stands in for a transcript with no speech in it. Must tokenize to at least one
# token: vLLM discards a zero-length placeholder and then reports the item as
# missing, so every audio item has to contribute something to the prompt.
_EMPTY_TRANSCRIPT_TEXT = " "
# Keeps the transcript budget finite if max_model_len cannot be read.
_FALLBACK_CONTEXT_LEN = 8192
_DUMMY_AUDIO_SECONDS = 5


def _base_hook_era(base: type) -> str:
    """Which HF-processor contract ``base`` implements, decided by SHAPE.

    ``hasattr(base, "_call_hf_processor")`` cannot answer this. That name is
    present in era A, absent in B, and present again in C with an incompatible
    signature, so an existence check maps A and C onto the same branch and feeds
    era C's call the era-A code path -- which is how ASR breaks silently: the
    transcript is simply never produced and every test stays green.

    The discriminator is the FIRST PARAMETER of ``_call_hf_processor``: era A
    takes ``prompt`` (it owns tokenization), era C takes ``hf_data`` (it is handed
    already-extracted mm data by ``_apply_hf_processor_main``). Era B has no such
    method at all.
    """
    hook = getattr(base, "_call_hf_processor", None)
    if hook is None:
        return _ERA_MAIN_ONLY

    try:
        params = list(inspect.signature(hook).parameters)
    except (TypeError, ValueError):
        # Unsignaturable (C extension, exotic descriptor). Fall back to the other
        # era-C marker: the helper pair that replaced _get_hf_mm_data.
        return _ERA_CALL_HF_MM if hasattr(base, "_get_hf_mm_inputs") else _ERA_CALL_HF

    if "self" in params:
        params.remove("self")
    return _ERA_CALL_HF if params and params[0] == "prompt" else _ERA_CALL_HF_MM


class GraniteSwitchASRProcessingInfo(BaseProcessingInfo):
    """Static info vLLM needs about the audio modality."""

    def _asr_enabled(self) -> bool:
        return bool(getattr(self.get_hf_config(), "asr_enabled", False))

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # No modalities on a non-audio checkpoint, so vLLM never loads ASR.
        if not self._asr_enabled():
            return {}
        # Finite so vLLM can size KV for the worst case. --limit-mm-per-prompt
        # may lower this ceiling, not raise it.
        return {"audio": self._asr_max_audio_clips()}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        if not self._asr_enabled():
            return {}
        # Sizes the encoder cache and profiling pass only — does NOT bound
        # requests (vLLM's prompt-length check does). A clip cannot exceed the
        # whole context, so the per-clip share of it is the honest upper bound.
        count = mm_counts.get("audio", 1) or 1
        return {"audio": max(1, seq_len // count)}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=_TARGET_SR)

    # --- ASR config resolved from the model's GraniteSwitchConfig ---

    def _asr_model_id(self) -> str:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_model_id", None) or DEFAULT_ASR_MODEL_ID

    def _asr_device(self) -> str:
        cfg = self.get_hf_config()
        requested = getattr(cfg, "asr_device", "cuda") or "cuda"
        # The transcriber runs in the vLLM *driver* process (multimodal input
        # preprocessing), not the EngineCore subprocess. Under Exclusive_Process
        # the driver cannot share the device the engine already holds, so a
        # "cuda" request is downgraded to CPU there to avoid cudaErrorDevices-
        # Unavailable. Default compute mode and explicit devices pass through.
        return resolve_driver_device(requested)

    def _asr_dtype(self) -> str | None:
        cfg = self.get_hf_config()
        return getattr(cfg, "asr_dtype", None)

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
        return bool(getattr(cfg, "asr_self_chunks", False))

    def _asr_chunk_length_s(self) -> float:
        cfg = self.get_hf_config()
        return float(getattr(cfg, "asr_chunk_length_s", 120.0) or 120.0)

    def _asr_chunk_overlap_s(self) -> float:
        cfg = self.get_hf_config()
        return float(getattr(cfg, "asr_chunk_overlap_s", 5.0) or 0.0)

    def _max_model_len(self) -> int:
        """The served context window, or a safe fallback.

        ``_call_hf_processor`` needs this at request time to size the transcript
        budget; vLLM's profiling ``seq_len`` is not available there.
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
        mm_options: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        length = _DUMMY_AUDIO_SECONDS * _TARGET_SR
        audio = torch.zeros(length, dtype=torch.float32).numpy()
        return {"audio": [audio] * num_audios}


class GraniteSwitchASRMultiModalProcessor(
    BaseMultiModalProcessor[GraniteSwitchASRProcessingInfo]
):
    """Runs ASR and splices the transcript tokens into the prompt."""

    def _marker_id(self) -> int:
        """Token id of the audio marker, or ``-1`` when it isn't registered.

        ``convert_tokens_to_ids`` answers with the *unk* id for an unknown token
        rather than ``None``, which would silently make every count come out
        wrong, so an unregistered marker is reported as ``-1`` instead.
        """
        tokenizer = self.info.get_tokenizer()
        token_id = tokenizer.convert_tokens_to_ids(AUDIO_MARKER)
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            return -1
        return int(token_id)

    def _count_markers(self, prompt) -> int:
        """Occurrences of the audio marker in a str or token-id prompt."""
        if isinstance(prompt, str):
            return prompt.count(AUDIO_MARKER)
        marker_id = self._marker_id()
        if marker_id < 0:
            return 0
        return sum(1 for token_id in prompt if token_id == marker_id)

    def _validate_marker_count(self, prompt, num_audio_items: int) -> None:
        """Require exactly one audio marker per audio item.

        ``<|audio|>`` is a registered special token, so text a caller types is
        tokenized into the *real* marker. vLLM pairs markers with audio items
        positionally and stops once every item is matched, leaving any extra
        marker in the prompt verbatim — so a spoofed marker silently moves the
        transcript to a caller-chosen position (and, in a multi-clip request,
        shifts every transcript onto the wrong clip) while vLLM's own
        ``_validate_mm_placeholders`` still sees matching counts and passes.

        The reverse case — markers with no audio payload — is the vector behind
        vLLM's own CVE-2026-44222 (GHSA-hpv8-x276-m59f), where models indexing a
        grid from a spoofed placeholder hit an unhandled ``IndexError``.
        """
        num_markers = self._count_markers(prompt)
        if num_markers == num_audio_items:
            return
        raise ValueError(
            f"Prompt contains {num_markers} {AUDIO_MARKER} marker(s) but the "
            f"request carries {num_audio_items} audio item(s); they must match "
            f"exactly. {AUDIO_MARKER} is reserved for audio placement and cannot "
            f"appear in message text."
        )

    def _prompt_and_audio_count(self, *args, **kwargs):
        """Pull (prompt, audio item count) out of whichever ``apply()`` shape.

        vLLM changed ``apply()``'s parameters across the versions this package
        supports: newer builds take a single ``ProcessorInputs`` (carrying
        ``prompt`` and already-parsed ``mm_data_items``), older ones take
        ``(prompt, mm_data, ...)`` with raw mm data. Both are handled here so the
        check does not depend on which is installed.
        """
        inputs = args[0] if args else kwargs.get("inputs")

        # Newer shape: a ProcessorInputs with items already parsed.
        if hasattr(inputs, "prompt") and hasattr(inputs, "mm_data_items"):
            items = inputs.mm_data_items
            count = len(items["audio"]) if "audio" in items else 0
            return inputs.prompt, count

        # Older shape: (prompt, mm_data, ...) with raw mm data to parse.
        prompt = inputs if args else kwargs.get("prompt")
        mm_data = args[1] if len(args) > 1 else kwargs.get("mm_data")
        if prompt is None or mm_data is None:
            raise RuntimeError(
                "Cannot read the prompt and audio items from this vLLM's "
                "MultiModalProcessor.apply() signature, so the audio marker "
                "count cannot be validated. Refusing rather than skipping a "
                "security check silently."
            )
        if not mm_data:
            return prompt, 0
        items = self.info.get_data_parser().parse_mm_data(mm_data)
        count = len(items["audio"]) if "audio" in items else 0
        return prompt, count

    def apply(self, *args, **kwargs):
        """Validate marker/item agreement, then delegate unchanged.

        Enforced here rather than in ``_call_hf_processor`` because that runs with
        only the *cache-missing* items: on a processor-cache hit its item count is
        smaller than the request's, so the comparison would be wrong. ``apply()``
        is the one entry point that always sees the whole request.
        """
        prompt, num_audio_items = self._prompt_and_audio_count(*args, **kwargs)
        self._validate_marker_count(prompt, num_audio_items)
        return super().apply(*args, **kwargs)

    def _transcribe(
        self,
        audio,
        generate_kwargs: Mapping[str, object] | None = None,
    ) -> list[int]:
        """Transcribe one audio item to token ids. Never truncated here — an
        oversized prompt is rejected by vLLM's own length check.

        Guaranteed non-empty: silence, music, non-speech or a clip too short to
        contain a word all transcribe to ``""``, and a zero-length replacement
        would make vLLM drop the placeholder and reject the request. Those clips
        get :data:`_EMPTY_TRANSCRIPT_TEXT` instead, so the model sees an audio
        turn that simply said nothing.
        """
        transcriber = get_transcriber(
            model_id=self.info._asr_model_id(),
            device=self.info._asr_device(),
            pipeline_kwargs=self.info._asr_pipeline_kwargs(),
            dtype=self.info._asr_dtype(),
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
        if not text or not text.strip():
            text = _EMPTY_TRANSCRIPT_TEXT
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            # Only reachable if the tokenizer drops _EMPTY_TRANSCRIPT_TEXT
            # entirely. Fail loudly rather than emit a zero-length placeholder,
            # which surfaces as an opaque "found 0 prompt placeholders".
            raise ValueError(
                f"Tokenizer produced no tokens for {text!r}; cannot build a "
                "prompt placeholder for this audio item. Choose an "
                "_EMPTY_TRANSCRIPT_TEXT this tokenizer encodes to >=1 token."
            )
        self._reject_reserved_ids(ids)
        return ids

    def _reserved_token_ids(self) -> set[int]:
        """Token ids a transcript must never contain.

        The audio marker (a transcript carrying one would mint a phantom
        placeholder) plus every adapter control token (the switch reads raw
        ``input_ids``, so one arriving via the transcript would steer adapter
        selection from audio content).
        """
        reserved: set[int] = set()
        marker_id = self._marker_id()
        if marker_id >= 0:
            reserved.add(marker_id)
        control_ids = getattr(self.info.get_hf_config(), "adapter_token_ids", None)
        for token_id in control_ids or ():
            reserved.add(int(token_id))
        return reserved

    def _reject_reserved_ids(self, ids: Sequence[int]) -> None:
        """Refuse a transcript that tokenized into reserved control tokens.

        ``encode(..., add_special_tokens=False)`` only suppresses *added* BOS/EOS;
        special-token strings already present in the text are still parsed into
        the real ids. So an ASR result containing ``<|audio|>`` or an adapter
        control token would inject genuine control tokens into the prompt.

        Rejected rather than neutralized (which ``split_special_tokens=True``
        would do) so the condition is visible instead of silently rewriting model
        output: a transcript containing these strings means either an attack or a
        badly misbehaving ASR backend, and both are worth surfacing.
        """
        reserved = self._reserved_token_ids()
        if not reserved:
            return
        found = sorted({int(t) for t in ids if int(t) in reserved})
        if not found:
            return
        tokenizer = self.info.get_tokenizer()
        names = [tokenizer.convert_ids_to_tokens(t) for t in found]
        raise ValueError(
            f"Transcript tokenized into reserved control token(s) {names} "
            f"(ids {found}); refusing to splice it into the prompt. Reserved "
            f"tokens must not originate from audio content."
        )

    def _audio_features(
        self,
        audios: Sequence[object],
        mm_kwargs: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        """Transcribe every clip into the out-of-band audio tensors.

        Empty when there is no audio, so the caller can splat it unconditionally.
        """
        if not audios:
            return {}

        # Resolved once, then applied to every audio item in this request.
        generate_kwargs = resolve_generate_kwargs(
            self.info._asr_generate_kwargs(),
            mm_kwargs,
            DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
        )

        # Concatenated flat, with per-item sizes to split them back.
        per_item_ids = [self._transcribe(a, generate_kwargs) for a in audios]
        sizes = [len(ids) for ids in per_item_ids]
        flat_ids = [tid for ids in per_item_ids for tid in ids]

        return dict(
            audio_token_ids=torch.tensor(flat_ids, dtype=torch.long),
            audio_num_tokens=torch.tensor(sizes, dtype=torch.long),
        )

    def _call_hf_processor(self, *args, **kwargs) -> BatchFeature:
        """Transcribe, on the two eras that call this hook (A and C).

        The signature is positional-agnostic because the two eras disagree on it:

        * era A (0.26-0.28) ``(prompt, mm_data, mm_kwargs, tok_kwargs)`` -- this
          hook owns tokenization too, so it must return ``input_ids`` alongside
          the audio tensors;
        * era C (0.31+) ``(hf_data, hf_kwargs)`` -- the base class already pulled
          the mm data out and tokenizes the prompt itself, so returning
          ``input_ids`` here would fight it. Audio tensors ONLY.

        Era B never calls this; see ``_apply_hf_processor_main``.
        """
        era = _base_hook_era(super())

        if era == _ERA_CALL_HF_MM:  # 0.31+
            hf_data = args[0] if args else kwargs["hf_data"]
            hf_kwargs = args[1] if len(args) > 1 else kwargs.get("hf_kwargs", {})
            features = self._audio_features(hf_data.get("audios", []) or [], hf_kwargs)
            return BatchFeature(dict(**features), tensor_type="pt")

        # era A: we own the prompt as well.
        prompt = args[0] if args else kwargs["prompt"]
        mm_data = args[1] if len(args) > 1 else kwargs["mm_data"]
        mm_kwargs = args[2] if len(args) > 2 else kwargs.get("mm_kwargs", {})
        tokenizer = self.info.get_tokenizer()
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        features = self._audio_features(mm_data.get("audios", []) or [], mm_kwargs)
        return BatchFeature(
            dict(input_ids=[input_ids], **features),
            tensor_type="pt",
        )

    def _apply_hf_processor_main(self, **kwargs):
        """Produce the audio tensors on whichever hook this vLLM actually calls.

        Three different contracts hide behind this one name (module docstring):

        * era A (0.26-0.28) -- the base implementation routes through
          ``_call_hf_processor``, so just delegate and let our override above run.
        * era B (0.29-0.30) -- ``_call_hf_processor`` was deleted and this method
          narrowed to ``(mm_items, hf_processor_mm_kwargs) -> BatchFeature``
          returning mm tensors only, with the prompt handled by ``apply()`` via
          ``_postprocess_prompt``. Nothing calls our transcriber unless we do it
          here, so this branch extracts the audio itself.
        * era C (0.31+) -- ``_call_hf_processor`` is back and the base
          implementation calls it again, so delegate as in era A. The kwarg was
          renamed ``hf_processor_mm_kwargs`` -> ``hf_kwargs``, which is why
          delegation forwards ``**kwargs`` untouched rather than naming it.

        Dispatching on ``hasattr(super(), "_call_hf_processor")`` -- as this did
        until 0.31 -- collapses A and C onto one branch, which happens to be
        right, but for era C it was reached for the wrong reason and the era-B
        branch's ``kwargs["hf_processor_mm_kwargs"]`` would ``KeyError`` the
        moment the era were misread. Gate on shape.
        """
        era = _base_hook_era(super())

        if era != _ERA_MAIN_ONLY:  # eras A and C both re-enter _call_hf_processor
            return super()._apply_hf_processor_main(**kwargs)

        mm_items = kwargs["mm_items"]
        hf_processor_mm_kwargs = kwargs["hf_processor_mm_kwargs"]

        valid_mm_items = mm_items.select(
            {k for k, c in mm_items.get_all_counts().items() if c > 0}
        )
        processor_data, passthrough_data = self._get_hf_mm_data(valid_mm_items)
        features = self._audio_features(
            processor_data.get("audios", []) or [],
            hf_processor_mm_kwargs,
        )
        # No input_ids: on 0.29+ the prompt is vLLM's business, not ours.
        return BatchFeature(
            dict(**features, **passthrough_data),
            tensor_type="pt",
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """Always False: ``_call_hf_processor`` leaves the marker in place.

        The base implementation returns True for raw (non-embedding) items,
        which tells vLLM the processor already expanded the placeholder itself —
        so vLLM skips applying our ``PromptReplacement`` and merely *searches*
        the returned prompt for the transcript token ids. They are not there, and
        it raises ``Expected there to be 1 audio prompt placeholders ... found 0``.

        Unlike a real HF processor (e.g. Ultravox's), we tokenize the prompt with
        the ``<|audio|>`` marker untouched and hand the transcript back out of
        band, so the replacement must be applied by vLLM.

        Only the uncached path consults this hook; the cached path already
        hardcodes False, which is why audio works with the default
        ``mm_processor_cache_gb=4`` and breaks under ``--mm-processor-cache-gb 0``.

        Required on vLLM 0.26 - 0.28 and inert on 0.29+, where the base class
        deleted the hook along with the ``is_update_applied`` branch it fed:
        ``_apply_prompt_updates`` now always runs, which is exactly the False
        behaviour asked for here. Kept because one tree has to serve both.
        """
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_tokens = hf_inputs.get("audio_num_tokens", torch.zeros(0))
        return dict(
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
            return [int(t) for t in all_ids[s:e]]

        # TARGET IS A TOKEN-ID LIST, NOT THE MARKER STRING.  vLLM narrowed the
        # accepted target type at 0.29:
        #
        #     0.26 - 0.28   UpdateTarget = PromptSeq | PromptIndex   # str allowed
        #     0.29 - 0.30   UpdateTarget = list[int] | PromptIndex   # str dropped
        #
        # A ``list[int]`` is legal on every one of those lines, so this needs no
        # version branch -- only ``str`` stopped being legal.
        #
        # Passing the string on 0.29+ does not fail loudly where it is given.
        # Token matching simply finds nothing, which sends vLLM into
        # ``_apply_prompt_updates_via_text`` -- a fallback that only exists from
        # 0.29 -- and that calls ``tokenizer.decode(target)`` on the raw string.
        # The resulting error surfaces far away and looks like a tokenizer bug:
        # ``TypeError: Can't extract `str` to `Vec``` from the Rust tokenizer,
        # killing EngineCore during init rather than rejecting the request.
        #
        # ``_marker_id()`` cannot answer -1 here: ``_validate_marker_count``
        # already requires one marker per audio item, and ``_count_markers``
        # reports 0 for an unregistered marker, so a prompt that reached this
        # point with audio items has a registered marker.
        return [
            PromptReplacement(
                modality="audio",
                target=[self._marker_id()],
                replacement=replacement,
            )
        ]
