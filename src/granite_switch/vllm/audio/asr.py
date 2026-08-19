# SPDX-License-Identifier: Apache-2.0
"""Speech-to-text backend for the audio cascade.

Wraps a HuggingFace ASR pipeline. Free of any vLLM import so it unit-tests on
CPU. The model loads lazily and is cached per (model_id, device, dtype,
pipeline_kwargs), so a process loads each ASR model at most once.

Device defaults to CPU to keep vLLM's GPU KV-cache budget clean; dtype follows
the device unless a checkpoint sets ``asr_dtype``. See docs/AUDIO.md.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, Union

import numpy as np

# Small, CPU-friendly, English ASR model that emits text directly. Used when the
# checkpoint does not name its own (config.asr_model_id is None).
DEFAULT_ASR_MODEL_ID = "distil-whisper/distil-small.en"

ASR_DTYPE_AUTO = "auto"

# Keep in sync with config.ASR_DTYPES.
_ASR_DTYPE_NAMES = frozenset({"float16", "bfloat16", "float32"})
_ASR_DTYPE_ALIASES = {
    "fp16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "float": "float32",
}


def _resolve_torch_dtype(dtype: str | None, device: str) -> Any:
    """Resolve an ``asr_dtype`` name to a ``torch.dtype``.

    None/"auto" derives it from the device: float16 on CUDA, float32 elsewhere
    (CPU float16 is slow and partly unimplemented). Name a dtype explicitly for
    an encoder that cannot run in half precision — BatchNorm raises on a float16
    weight against float32 features rather than promoting.
    """
    import torch

    name = str(dtype or ASR_DTYPE_AUTO).lower()
    name = _ASR_DTYPE_ALIASES.get(name, name)
    if name == ASR_DTYPE_AUTO:
        on_cuda = isinstance(device, str) and device.startswith("cuda")
        return torch.float16 if on_cuda else torch.float32
    if name not in _ASR_DTYPE_NAMES:
        raise ValueError(
            f"Unsupported asr_dtype {dtype!r}. Expected {ASR_DTYPE_AUTO!r} (or "
            f"None) to derive it from the device, or one of: "
            f"{', '.join(sorted(_ASR_DTYPE_NAMES))}."
        )
    return getattr(torch, name)


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
    tuple[np.ndarray, int | float],
    "object",  # torch.Tensor — typed loosely to avoid importing torch here
]


class ASRTranscriber:
    """Lazily-loaded ASR model wrapper exposing :meth:`transcribe`."""

    def __init__(
        self,
        model_id: str = DEFAULT_ASR_MODEL_ID,
        device: str = "cpu",
        pipeline_kwargs: Mapping[str, Any] | None = None,
        dtype: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.pipeline_kwargs: dict[str, Any] = dict(pipeline_kwargs or {})
        self._pipeline = None
        self._load_lock = threading.Lock()

    def load(self) -> None:
        """Materialize the ASR pipeline if it has not been loaded yet."""
        if self._pipeline is not None:
            return
        with self._load_lock:
            if self._pipeline is not None:
                return
            # Lazy: keeps this module importable without transformers' audio stack.
            from transformers import pipeline

            kwargs: dict[str, Any] = {
                "task": "automatic-speech-recognition",
                "model": self.model_id,
                "device": self.device,
                "torch_dtype": _resolve_torch_dtype(self.dtype, self.device),
                "chunk_length_s": 30,
            }
            # pipeline_kwargs last: a checkpoint may override any default above.
            kwargs.update(self.pipeline_kwargs)
            self._pipeline = pipeline(**kwargs)

    def transcribe(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        generate_kwargs: Mapping[str, Any] | None = None,
        self_chunks: bool = True,
        chunk_length_s: float = 30.0,
        chunk_overlap_s: float = 5.0,
    ) -> str:
        """Transcribe one audio clip, stripped. Resampled to 16 kHz as needed.

        ``sampling_rate`` is required unless ``audio`` is an ``(array, rate)``
        tuple. ``generate_kwargs`` is passed only when non-empty, so CTC backends
        are unaffected. ``self_chunks=False`` routes long audio through
        :mod:`.chunking` using ``chunk_length_s``/``chunk_overlap_s``.
        """
        samples, sr = _coerce_audio(audio, sampling_rate)
        samples = _to_mono_float32(samples)
        samples = _resample(samples, sr, _TARGET_SAMPLE_RATE)

        self.load()

        if self_chunks:
            return self._run_pipeline(samples, generate_kwargs)

        chunking = _load_chunking()
        segments = chunking.split_waveform(
            samples, _TARGET_SAMPLE_RATE, chunk_length_s, chunk_overlap_s
        )
        texts = [self._run_pipeline(seg, generate_kwargs) for seg in segments]
        return chunking.merge_transcripts(texts).strip()

    def _run_pipeline(
        self,
        samples: np.ndarray,
        generate_kwargs: Mapping[str, Any] | None = None,
    ) -> str:
        """Run the loaded pipeline over an already-resampled mono waveform."""
        call_kwargs: dict[str, Any] = {}
        if generate_kwargs:
            call_kwargs["generate_kwargs"] = dict(generate_kwargs)
        result = self._pipeline(
            {"raw": samples, "sampling_rate": _TARGET_SAMPLE_RATE},
            **call_kwargs,
        )
        text = result["text"] if isinstance(result, dict) else str(result)
        return text.strip()


# ── Module-level cache + convenience function ────────────────────────────────

# Keyed on what changes the constructed pipeline. generate_kwargs are excluded:
# they apply per transcribe() call, so one cached pipeline serves every language.
_TRANSCRIBERS: dict[tuple, ASRTranscriber] = {}
_CACHE_LOCK = threading.Lock()


# Allowlisted so a client cannot inject arbitrary generation options; everything
# else is fixed by the checkpoint in config.asr_generate_kwargs.
DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS = frozenset({"language", "task"})


def resolve_generate_kwargs(
    config_defaults: Mapping[str, Any] | None,
    request: Mapping[str, Any] | None = None,
    allowed_keys: frozenset[str] = DEFAULT_ALLOWED_REQUEST_GENERATE_KEYS,
) -> dict[str, Any]:
    """Merge config-default decode kwargs with allowlisted per-request overrides.

    ``request`` is vLLM's ``mm_processor_kwargs``: a top-level ``language`` or a
    nested ``asr_generate_kwargs`` may override the defaults, but only for keys
    in ``allowed_keys``. Request values win, so one model serves many languages.
    """
    merged: dict[str, Any] = dict(config_defaults or {})
    if isinstance(request, Mapping):
        nested = request.get("asr_generate_kwargs")
        if isinstance(nested, Mapping):
            merged.update({k: v for k, v in nested.items() if k in allowed_keys})
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
    model_id: str | None = None,
    device: str = "cpu",
    pipeline_kwargs: Mapping[str, Any] | None = None,
    dtype: str | None = None,
) -> ASRTranscriber:
    """Return a process-wide cached :class:`ASRTranscriber`.

    Cached per ``(model_id, device, dtype, pipeline_kwargs)``. ``model_id`` of
    None resolves to :data:`DEFAULT_ASR_MODEL_ID`.
    """
    resolved = model_id or DEFAULT_ASR_MODEL_ID
    key = (resolved, device, dtype, _freeze(pipeline_kwargs or {}))
    transcriber = _TRANSCRIBERS.get(key)
    if transcriber is None:
        with _CACHE_LOCK:
            transcriber = _TRANSCRIBERS.get(key)
            if transcriber is None:
                transcriber = ASRTranscriber(
                    model_id=resolved,
                    device=device,
                    pipeline_kwargs=pipeline_kwargs,
                    dtype=dtype,
                )
                _TRANSCRIBERS[key] = transcriber
    return transcriber


def transcribe(
    audio: AudioInput,
    sampling_rate: int | None = None,
    *,
    model_id: str | None = None,
    device: str = "cpu",
    pipeline_kwargs: Mapping[str, Any] | None = None,
    dtype: str | None = None,
    generate_kwargs: Mapping[str, Any] | None = None,
    self_chunks: bool = True,
    chunk_length_s: float = 30.0,
    chunk_overlap_s: float = 5.0,
) -> str:
    """Convenience wrapper: transcribe with the cached transcriber for the args."""
    return get_transcriber(
        model_id=model_id,
        device=device,
        pipeline_kwargs=pipeline_kwargs,
        dtype=dtype,
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
    sampling_rate: int | None,
) -> tuple[np.ndarray, int]:
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
    """Resample to ``target_sr`` Hz via vLLM's ``AudioResampler``. No-op at target."""
    if orig_sr == target_sr:
        return samples
    # Local import so the module's other helpers still load without vLLM.
    from vllm.multimodal.audio import AudioResampler

    return AudioResampler(target_sr=target_sr).resample(samples, orig_sr=orig_sr)
