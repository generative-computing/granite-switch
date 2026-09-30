# SPDX-License-Identifier: Apache-2.0
"""GPU checks for the default CTC ASR backend (Granite Speech 5.0 TurboCTC).

Exercises the real model through our own :class:`ASRTranscriber`, so it covers
the path a served checkpoint takes rather than transformers in isolation: the
default device, the dtype auto-resolution, the CTC classification that suppresses
pipeline chunking and decode kwargs, and the window that decides whether a clip
reaches the model whole or through our chunker.

Downloads the ~1GB checkpoint on first run. Opt in explicitly:
`pytest -m "audio and requires_model and gpu"`.
"""

import importlib.util
import pathlib
import wave

import numpy as np
import pytest

pytestmark = [
    pytest.mark.audio,
    pytest.mark.slow,
    pytest.mark.requires_model,
    pytest.mark.gpu,
]

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a CUDA GPU", allow_module_level=True)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Load asr.py by path, as the CPU unit tests do: it has no vLLM dependency of its
# own, and the package __init__ above it imports vLLM.
_spec = importlib.util.spec_from_file_location(
    "gs_asr_ctc_gpu", _REPO_ROOT / "src/granite_switch/vllm/audio/asr.py"
)
asr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(asr)

# tests/audio/test1.wav says this, and CTC emits it lowercase and unpunctuated.
_EXPECTED = "what is the capital of israel"


def _test_clip():
    """The committed test waveform as float32 mono, plus its sample rate."""
    with wave.open(str(_REPO_ROOT / "tests/audio/test1.wav")) as handle:
        sample_rate, frames = (
            handle.getframerate(),
            handle.readframes(handle.getnframes()),
        )
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


@pytest.fixture(scope="module")
def transcriber():
    """The default model on CUDA with the default (auto -> bfloat16) dtype."""
    instance = asr.ASRTranscriber(model_id=asr.DEFAULT_ASR_MODEL_ID, device="cuda")
    instance.load()
    return instance


def test_auto_dtype_on_cuda_is_bfloat16():
    assert asr._resolve_torch_dtype(None, "cuda") is torch.bfloat16


def test_backend_is_detected_as_ctc(transcriber):
    # Drives everything else: a CTC pipeline takes neither a chunk window nor
    # decode kwargs, and misclassifying it would surface as a TypeError per call.
    assert transcriber._is_ctc is True


def test_transcribes_correctly_and_ignores_decode_kwargs(transcriber):
    samples, sample_rate = _test_clip()
    # `language` is what a client may send via mm_processor_kwargs. A CTC backend
    # has no generate() to take it, so it must be dropped rather than forwarded.
    text = transcriber.transcribe(
        samples, sampling_rate=sample_rate, generate_kwargs={"language": "fr"}
    )
    assert text == _EXPECTED


def test_float16_override_still_transcribes():
    """float16 is an available override, not a failure mode.

    The default resolves to bfloat16 on CUDA because that is the checkpoint's own
    dtype and it keeps float32's exponent range next to this encoder's BatchNorm
    layers. An earlier version of this test asserted float16 *raised*; measured on
    an A100 (torch 2.10 / transformers 5.16) it does not, so what is worth
    guarding is that the override keeps working and still produces the transcript.
    """
    samples, sample_rate = _test_clip()
    half = asr.ASRTranscriber(
        model_id=asr.DEFAULT_ASR_MODEL_ID, device="cuda", dtype="float16"
    )
    assert half.transcribe(samples, sampling_rate=sample_rate) == _EXPECTED


def _tile_to_seconds(samples, sample_rate, seconds):
    reps = int(np.ceil(seconds * sample_rate / len(samples)))
    return np.tile(samples, reps)[: int(seconds * sample_rate)]


@pytest.mark.parametrize(
    "seconds,expected_calls",
    [
        # At/under the window the clip must reach the model in one piece; past it
        # our chunker splits, since the pipeline's own CTC chunking mis-trims
        # seams for a model that publishes no inputs_to_logits_ratio.
        (int(asr.DEFAULT_CHUNK_LENGTH_S) - 20, 1),
        (int(asr.DEFAULT_CHUNK_LENGTH_S) * 2, None),
    ],
)
def test_window_decides_single_pass_vs_chunked(transcriber, seconds, expected_calls):
    samples, sample_rate = _test_clip()
    audio = _tile_to_seconds(samples, sample_rate, seconds)

    calls: list[float] = []
    inner = transcriber._run_pipeline

    def counting(segment, generate_kwargs=None):
        calls.append(len(segment) / asr._TARGET_SAMPLE_RATE)
        return inner(segment, generate_kwargs)

    transcriber._run_pipeline = counting
    try:
        text = transcriber.transcribe(audio, sampling_rate=sample_rate)
    finally:
        transcriber._run_pipeline = inner

    print(f"{seconds}s -> {len(calls)} call(s) {[round(c) for c in calls]}")
    if expected_calls is None:
        assert len(calls) > 1
        assert max(calls) <= asr.DEFAULT_CHUNK_LENGTH_S + 1
    else:
        assert len(calls) == expected_calls
    # The phrase repeats throughout, so the merge must not collapse it to nothing.
    assert _EXPECTED.split()[-1] in text
