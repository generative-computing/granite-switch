# SPDX-License-Identifier: Apache-2.0
"""Speech-to-text backend for the alpha audio cascade.

A thin wrapper around a HuggingFace ASR model that turns raw audio into a
transcript string. Kept free of any vLLM import so it can be unit-tested on CPU
and reused by both the multimodal-processor integration and a fallback wrapper.

Design notes:
    * The model is loaded **lazily** on first use and cached per (model_id,
      device) so engine startup stays cheap and a process loads each ASR model
      at most once.
    * Default device is CPU — this keeps vLLM's GPU KV-cache budget clean. Set a
      CUDA device to trade GPU memory for transcription latency.
    * The default model is a small, text-emitting open-source ASR model. The
      alpha intentionally uses a complete audio->text model (encoder+decoder);
      swapping in the Granite Speech 4.1 encoder is future work and only touches
      this module.
    * ``transcribe`` accepts the audio item shapes vLLM hands to multimodal
      processors: a bare array, an ``(array, sampling_rate)`` tuple, a Python
      list of floats, or a torch tensor.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

# Small, CPU-friendly, English ASR model that emits text directly. Used when the
# checkpoint does not name its own (config.asr_model_id is None).
DEFAULT_ASR_MODEL_ID = "distil-whisper/distil-small.en"

_CHUNKING = None


def _load_chunking():
    """Load the pure chunking helpers, memoized.

    Uses the normal relative import in production (running inside the
    ``granite_switch.vllm.audio`` package). Falls back to a direct file-path load
    when this module is imported standalone (the CPU unit tests load ``asr.py`` by
    path to skip the vLLM-importing package ``__init__``).
    """
    global _CHUNKING
    if _CHUNKING is not None:
        return _CHUNKING
    try:
        from . import chunking as _chunking  # normal package import
    except ImportError:
        import importlib.util
        import pathlib

        path = pathlib.Path(__file__).with_name("chunking.py")
        spec = importlib.util.spec_from_file_location("gs_chunking", path)
        _chunking = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_chunking)
    _CHUNKING = _chunking
    return _CHUNKING

# Sample rate expected by Whisper-family feature extractors.
_TARGET_SAMPLE_RATE = 16_000

# Audio item shapes vLLM may pass to a multimodal processor.
AudioInput = Union[
    np.ndarray,
    "list[float]",
    Tuple[np.ndarray, Union[int, float]],
    "object",  # torch.Tensor — typed loosely to avoid importing torch here
]


class ASRTranscriber:
    """Lazily-loaded ASR model wrapper exposing :meth:`transcribe`.

    Instances are cheap to construct; the underlying model is only materialized
    on the first :meth:`transcribe` call (or an explicit :meth:`load`).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_ASR_MODEL_ID,
        device: str = "cpu",
        pipeline_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        # Extra kwargs merged into the pipeline() construction. Because they
        # change the built pipeline, get_transcriber folds them into the cache
        # key so distinct construction options get distinct cached instances.
        self.pipeline_kwargs: Dict[str, Any] = dict(pipeline_kwargs or {})
        self._pipeline = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Materialize the ASR pipeline if it has not been loaded yet."""
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            # Imported lazily so this module stays importable without
            # transformers' heavy audio stack until it is actually used.
            import torch
            from transformers import pipeline

            torch_dtype = (
                torch.float16
                if isinstance(self.device, str) and self.device.startswith("cuda")
                else torch.float32
            )
            # Built-in defaults, then let checkpoint-supplied pipeline_kwargs
            # override any of them (e.g. a different chunk_length_s, or model
            # kwargs a non-Whisper backend needs).
            kwargs: Dict[str, Any] = {
                "task": "automatic-speech-recognition",
                "model": self.model_id,
                "device": self.device,
                "torch_dtype": torch_dtype,
                # Enable internal 30s chunking so audio longer than the model's
                # native window is handled without us re-implementing it.
                "chunk_length_s": 30,
            }
            kwargs.update(self.pipeline_kwargs)
            self._pipeline = pipeline(**kwargs)

    def transcribe(
        self,
        audio: AudioInput,
        sampling_rate: Optional[int] = None,
        generate_kwargs: Optional[Mapping[str, Any]] = None,
        self_chunks: bool = True,
        chunk_length_s: float = 30.0,
        chunk_overlap_s: float = 5.0,
    ) -> str:
        """Transcribe one audio clip to a text string.

        Args:
            audio: The audio samples. Accepts a numpy array, a list of floats, a
                torch tensor, or an ``(array, sampling_rate)`` tuple (vLLM's
                ``AudioItem`` shape).
            sampling_rate: Sample rate of ``audio`` in Hz. Required unless
                ``audio`` is an ``(array, sampling_rate)`` tuple. Audio is
                resampled to 16 kHz when needed.
            generate_kwargs: Decode-time kwargs forwarded to the ASR model for
                this call (e.g. ``{"language": "fr"}``). Applied per call so the
                same loaded pipeline serves many languages. Passed only when
                non-empty, so CTC/non-generative backends are unaffected.
            self_chunks: True when the backend handles long audio itself (the
                Whisper pipeline, via its internal ``chunk_length_s``); the whole
                clip is passed in one call. False routes long audio through our
                encoder-agnostic chunker (:mod:`.chunking`): split into
                overlapping windows, transcribe each, and merge with overlap
                de-duplication.
            chunk_length_s: Window length for our chunker (only used when
                ``self_chunks`` is False).
            chunk_overlap_s: Window overlap for our chunker (only used when
                ``self_chunks`` is False).

        Returns:
            The transcript with surrounding whitespace stripped.
        """
        samples, sr = _coerce_audio(audio, sampling_rate)
        samples = _to_mono_float32(samples)
        samples = _resample(samples, sr, _TARGET_SAMPLE_RATE)

        self.load()

        # Backend chunks internally (Whisper): one call over the whole clip.
        if self_chunks:
            return self._run_pipeline(samples, generate_kwargs)

        # Encoder-agnostic path: split into overlapping windows, transcribe each,
        # then stitch the per-window transcripts (de-duplicating the overlap).
        chunking = _load_chunking()
        segments = chunking.split_waveform(
            samples, _TARGET_SAMPLE_RATE, chunk_length_s, chunk_overlap_s
        )
        texts = [self._run_pipeline(seg, generate_kwargs) for seg in segments]
        return chunking.merge_transcripts(texts).strip()

    def _run_pipeline(
        self,
        samples: np.ndarray,
        generate_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Run the loaded pipeline over an already-resampled mono waveform."""
        # Tell the pipeline the rate so it does not attempt its own resampling.
        call_kwargs: Dict[str, Any] = {}
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = dict(generate_kwargs)
        result = self._pipeline(
            {"raw": samples, "sampling_rate": _TARGET_SAMPLE_RATE},
            **call_kwargs,
        )
        text = result["text"] if isinstance(result, dict) else str(result)
        return text.strip()


# ── Module-level cache + convenience function ────────────────────────────────

# Keyed on (model_id, device, frozen pipeline_kwargs). pipeline_kwargs are in
# the key because they change the constructed pipeline; generate_kwargs are NOT
# — they are applied per transcribe() call, so one cached pipeline serves them
# all (that is what makes per-request language selection cheap).
_TRANSCRIBERS: Dict[Tuple, ASRTranscriber] = {}
_CACHE_LOCK = threading.Lock()


# Decode kwargs a client may set per request. Kept to language/task so a client
# cannot inject arbitrary (potentially expensive or unsafe) generation options;
# everything else is fixed by the checkpoint author in config.asr_generate_kwargs.
DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS = frozenset({"language", "task"})


def resolve_generate_kwargs(
    config_defaults: Optional[Mapping[str, Any]],
    request: Optional[Mapping[str, Any]] = None,
    allowed_keys: "frozenset[str]" = DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
) -> Dict[str, Any]:
    """Merge config-default decode kwargs with allowlisted per-request overrides.

    ``config_defaults`` come from ``config.asr_generate_kwargs``. ``request`` is
    the per-request mapping (vLLM's ``mm_processor_kwargs``); a top-level
    ``language`` or a nested ``asr_generate_kwargs`` object may override the
    defaults, but only keys in ``allowed_keys`` are honored. Request values win
    so one deployed model can serve many languages. Pure and vLLM-free so it can
    be unit-tested on CPU.
    """
    merged: Dict[str, Any] = dict(config_defaults or {})
    if isinstance(request, Mapping):
        nested = request.get("asr_generate_kwargs")
        if isinstance(nested, Mapping):
            merged.update(
                {k: v for k, v in nested.items() if k in allowed_keys}
            )
        language = request.get("language")
        if language is not None:
            merged["language"] = language
    return merged


def _freeze(value: Any) -> Any:
    """Recursively convert dicts/lists into a hashable, order-stable form."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def get_transcriber(
    model_id: Optional[str] = None,
    device: str = "cpu",
    pipeline_kwargs: Optional[Mapping[str, Any]] = None,
) -> ASRTranscriber:
    """Return a process-wide cached :class:`ASRTranscriber`.

    Cached per ``(model_id, device, pipeline_kwargs)``. ``model_id`` of None
    resolves to :data:`DEFAULT_ASR_MODEL_ID`.
    """
    resolved = model_id or DEFAULT_ASR_MODEL_ID
    key = (resolved, device, _freeze(pipeline_kwargs or {}))
    transcriber = _TRANSCRIBERS.get(key)
    if transcriber is None:
        with _CACHE_LOCK:
            transcriber = _TRANSCRIBERS.get(key)
            if transcriber is None:
                transcriber = ASRTranscriber(
                    model_id=resolved,
                    device=device,
                    pipeline_kwargs=pipeline_kwargs,
                )
                _TRANSCRIBERS[key] = transcriber
    return transcriber


def transcribe(
    audio: AudioInput,
    sampling_rate: Optional[int] = None,
    *,
    model_id: Optional[str] = None,
    device: str = "cpu",
    pipeline_kwargs: Optional[Mapping[str, Any]] = None,
    generate_kwargs: Optional[Mapping[str, Any]] = None,
    self_chunks: bool = True,
    chunk_length_s: float = 30.0,
    chunk_overlap_s: float = 5.0,
) -> str:
    """Convenience wrapper: transcribe with the cached transcriber for the args."""
    return get_transcriber(
        model_id=model_id, device=device, pipeline_kwargs=pipeline_kwargs
    ).transcribe(
        audio,
        sampling_rate,
        generate_kwargs=generate_kwargs,
        self_chunks=self_chunks,
        chunk_length_s=chunk_length_s,
        chunk_overlap_s=chunk_overlap_s,
    )


# ── Audio coercion helpers ───────────────────────────────────────────────────


def _coerce_audio(
    audio: AudioInput,
    sampling_rate: Optional[int],
) -> Tuple[np.ndarray, int]:
    """Normalize the various accepted audio shapes to ``(np.ndarray, sr)``."""
    # (array, sampling_rate) tuple — vLLM's AudioItem form.
    if isinstance(audio, tuple):
        if len(audio) != 2:
            raise ValueError(
                f"Tuple audio input must be (array, sampling_rate); got "
                f"length {len(audio)}."
            )
        array, sr = audio
        return _as_numpy(array), int(sr)

    if sampling_rate is None:
        raise ValueError(
            "sampling_rate is required when audio is not an "
            "(array, sampling_rate) tuple."
        )
    return _as_numpy(audio), int(sampling_rate)


def _as_numpy(array: object) -> np.ndarray:
    """Convert a numpy array, list, or torch tensor to a numpy array."""
    if isinstance(array, np.ndarray):
        return array
    # torch.Tensor without importing torch at module load.
    if hasattr(array, "detach") and hasattr(array, "cpu"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    """Downmix to mono and cast to float32."""
    if samples.ndim > 1:
        # Average across channels. Assume the smaller axis is channels.
        channel_axis = int(np.argmin(samples.shape))
        samples = samples.mean(axis=channel_axis)
    return samples.astype(np.float32, copy=False)


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample to ``target_sr`` Hz. No-op when already at the target rate."""
    if orig_sr == target_sr:
        return samples
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised only without librosa
        raise RuntimeError(
            f"Audio sampled at {orig_sr} Hz must be resampled to {target_sr} Hz, "
            "but librosa is not installed. Install the audio extra: "
            "`uv sync --extra audio` (or `pip install librosa`)."
        ) from exc
    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
