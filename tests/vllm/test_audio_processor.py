# SPDX-License-Identifier: Apache-2.0
"""vLLM-tier tests for the audio ASR multimodal processor.

These exercise the code that lives in ``granite_switch.vllm.audio.processor``
and therefore needs vLLM importable (the base classes come from vLLM). They do
NOT need a GPU or a real ASR model: the transcriber is faked, so what is under
test is our plumbing —

- per-checkpoint gating of the audio modality on ``asr_enabled``,
- the config accessors for ``asr_model_id`` / ``asr_device`` /
  ``asr_pipeline_kwargs`` / ``asr_generate_kwargs``,
- and the Level-2 seam: config-default decode kwargs merged with an allowlisted
  per-request override (vLLM's ``mm_processor_kwargs``) reaching the transcriber.

The pure merge/kwargs logic itself is unit-tested on CPU in
``tests/unit/test_asr.py``; this file checks it is wired through the processor.

Requires vLLM installed. Skipped otherwise (e.g. CPU-only dev machines).
"""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="requires vLLM installed",
)

if _VLLM_AVAILABLE:
    from granite_switch.vllm.audio import processor as proc_mod
    from granite_switch.vllm.audio.processor import (
        GraniteSwitchASRProcessingInfo,
        GraniteSwitchASRMultiModalProcessor,
    )
    from granite_switch.vllm.audio.asr import DEFAULT_ASR_MODEL_ID


def _make_info(*, max_model_len=131072, **cfg_attrs):
    """A ProcessingInfo whose get_hf_config() returns a stub config.

    Bypasses __init__ (which needs a full vLLM InputProcessingContext) — we only
    exercise methods that read the config. A stub ``ctx`` supplies the served
    ``max_model_len`` used to size the transcript budget.
    """
    info = object.__new__(GraniteSwitchASRProcessingInfo)
    cfg = SimpleNamespace(**cfg_attrs)
    info.get_hf_config = lambda: cfg
    info.ctx = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len)
    )
    return info


class TestProcessingInfoGating:
    def test_disabled_reports_no_modalities(self):
        info = _make_info(asr_enabled=False)
        assert info.get_supported_mm_limits() == {}
        assert info.get_mm_max_tokens_per_item(128, {"audio": 1}) == {}

    def test_missing_flag_defaults_disabled(self):
        # A pre-audio checkpoint has no asr_enabled key at all.
        info = _make_info()
        assert info.get_supported_mm_limits() == {}

    def test_enabled_reports_configurable_clip_limit(self):
        # Default ceiling is 32 clips; no longer hard-capped at 1.
        info = _make_info(asr_enabled=True)
        assert info.get_supported_mm_limits() == {"audio": 32}
        info3 = _make_info(asr_enabled=True, asr_max_audio_clips=3)
        assert info3.get_supported_mm_limits() == {"audio": 3}

    def test_max_tokens_per_item_is_context_derived(self):
        # Profiling/encoder-cache hint = per-clip share of the context (seq_len //
        # clip_count), the worst case one clip can occupy. Not a request bound —
        # an oversized transcript is rejected by vLLM's prompt-length check.
        info = _make_info(asr_enabled=True)
        assert info.get_mm_max_tokens_per_item(20000, {"audio": 1}) == {
            "audio": 20000
        }
        assert info.get_mm_max_tokens_per_item(20000, {"audio": 4}) == {
            "audio": 20000 // 4
        }


class TestProcessingInfoAsrAccessors:
    def test_model_id_defaults_when_none(self):
        info = _make_info(asr_enabled=True, asr_model_id=None)
        assert info._asr_model_id() == DEFAULT_ASR_MODEL_ID

    def test_model_id_explicit(self):
        info = _make_info(asr_enabled=True, asr_model_id="openai/whisper-small")
        assert info._asr_model_id() == "openai/whisper-small"

    def test_device_default_cpu(self):
        assert _make_info(asr_enabled=True)._asr_device() == "cpu"

    def test_pipeline_and_generate_kwargs_default_empty(self):
        info = _make_info(asr_enabled=True)
        assert info._asr_pipeline_kwargs() == {}
        assert info._asr_generate_kwargs() == {}

    def test_pipeline_and_generate_kwargs_from_config(self):
        info = _make_info(
            asr_enabled=True,
            asr_pipeline_kwargs={"chunk_length_s": 15},
            asr_generate_kwargs={"language": "de"},
        )
        assert info._asr_pipeline_kwargs() == {"chunk_length_s": 15}
        assert info._asr_generate_kwargs() == {"language": "de"}

    def test_longaudio_accessor_defaults(self):
        info = _make_info(asr_enabled=True)
        assert info._asr_max_audio_clips() == 32
        assert info._asr_self_chunks() is True
        assert info._asr_chunk_length_s() == 30.0
        assert info._asr_chunk_overlap_s() == 5.0

    def test_longaudio_accessors_from_config(self):
        info = _make_info(
            asr_enabled=True,
            asr_max_audio_clips=2,
            asr_self_chunks=False,
            asr_chunk_length_s=20.0,
            asr_chunk_overlap_s=3.0,
        )
        assert info._asr_max_audio_clips() == 2
        assert info._asr_self_chunks() is False
        assert info._asr_chunk_length_s() == 20.0
        assert info._asr_chunk_overlap_s() == 3.0

    def test_max_model_len_from_ctx(self):
        assert _make_info(asr_enabled=True, max_model_len=16000)._max_model_len() == 16000

    def test_max_model_len_falls_back_to_position_embeddings(self):
        info = _make_info(asr_enabled=True, max_position_embeddings=4096)
        info.ctx = SimpleNamespace(model_config=SimpleNamespace(max_model_len=None))
        assert info._max_model_len() == 4096


def _make_processor(info, monkeypatch, capture):
    """A processor whose transcriber is faked; records what it was called with."""
    info.get_tokenizer = lambda: SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [1, 2, 3]
    )
    proc = object.__new__(GraniteSwitchASRMultiModalProcessor)
    proc.info = info

    class FakeTranscriber:
        def transcribe(
            self,
            audio,
            sampling_rate=None,
            generate_kwargs=None,
            self_chunks=True,
            chunk_length_s=30.0,
            chunk_overlap_s=5.0,
        ):
            capture["sampling_rate"] = sampling_rate
            capture["generate_kwargs"] = generate_kwargs
            capture["self_chunks"] = self_chunks
            capture["chunk_length_s"] = chunk_length_s
            capture["chunk_overlap_s"] = chunk_overlap_s
            return "hello world"

    def fake_get_transcriber(model_id=None, device="cpu", pipeline_kwargs=None):
        capture["model_id"] = model_id
        capture["device"] = device
        capture["pipeline_kwargs"] = pipeline_kwargs
        return FakeTranscriber()

    monkeypatch.setattr(proc_mod, "get_transcriber", fake_get_transcriber)
    return proc


class TestTranscribeWiring:
    def test_transcribe_forwards_pipeline_and_generate_kwargs(self, monkeypatch):
        capture = {}
        info = _make_info(
            asr_enabled=True,
            asr_model_id="whisper-x",
            asr_device="cpu",
            asr_pipeline_kwargs={"chunk_length_s": 15},
            asr_generate_kwargs={},
        )
        proc = _make_processor(info, monkeypatch, capture)

        ids = proc._transcribe(
            np.zeros(1600, dtype=np.float32), {"language": "fr"}
        )
        assert ids == [1, 2, 3]
        assert capture["model_id"] == "whisper-x"
        assert capture["pipeline_kwargs"] == {"chunk_length_s": 15}
        assert capture["generate_kwargs"] == {"language": "fr"}
        assert capture["sampling_rate"] == proc_mod._TARGET_SR

    def test_empty_generate_kwargs_becomes_none(self, monkeypatch):
        capture = {}
        info = _make_info(asr_enabled=True, asr_model_id="w", asr_device="cpu")
        proc = _make_processor(info, monkeypatch, capture)
        proc._transcribe(np.zeros(1600, dtype=np.float32), {})
        assert capture["generate_kwargs"] is None


class TestCallHfProcessorMerge:
    """The Level-2 seam end to end (short of a real model): config defaults +
    allowlisted per-request override reach the transcriber."""

    def test_request_language_overrides_config_default(self, monkeypatch):
        capture = {}
        info = _make_info(
            asr_enabled=True,
            asr_model_id="w",
            asr_device="cpu",
            asr_pipeline_kwargs=None,
            asr_generate_kwargs={"task": "transcribe", "language": "de"},
        )
        proc = _make_processor(info, monkeypatch, capture)

        bf = proc._call_hf_processor(
            prompt="<|audio|>",
            mm_data={"audios": [np.zeros(1600, dtype=np.float32)]},
            mm_kwargs={"language": "fr"},   # per-request override
            tok_kwargs={},
        )
        # config 'task' retained, request 'language' wins over config 'de'.
        assert capture["generate_kwargs"] == {"task": "transcribe", "language": "fr"}
        assert "audio_token_ids" in bf

    def test_disallowed_request_key_dropped(self, monkeypatch):
        capture = {}
        info = _make_info(
            asr_enabled=True,
            asr_model_id="w",
            asr_device="cpu",
            asr_pipeline_kwargs=None,
            asr_generate_kwargs={"language": "de"},
        )
        proc = _make_processor(info, monkeypatch, capture)

        proc._call_hf_processor(
            prompt="<|audio|>",
            mm_data={"audios": [np.zeros(1600, dtype=np.float32)]},
            mm_kwargs={"asr_generate_kwargs": {"num_beams": 99, "task": "translate"}},
            tok_kwargs={},
        )
        gk = capture["generate_kwargs"]
        assert gk == {"language": "de", "task": "translate"}
        assert "num_beams" not in gk

    def test_text_only_request_does_not_transcribe(self, monkeypatch):
        capture = {}
        info = _make_info(asr_enabled=True, asr_model_id="w", asr_device="cpu")
        proc = _make_processor(info, monkeypatch, capture)

        bf = proc._call_hf_processor(
            prompt="just text",
            mm_data={},
            mm_kwargs={},
            tok_kwargs={},
        )
        # No audio → transcriber never touched, no audio fields emitted.
        assert capture == {}
        assert "audio_token_ids" not in bf


class TestMultiClipAndBudget:
    """Multiple clips, each transcribed in full and spliced at its marker."""

    def test_two_clips_produce_two_transcripts(self, monkeypatch):
        capture = {}
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor(info, monkeypatch, capture)

        bf = proc._call_hf_processor(
            prompt="<|audio|> and <|audio|>",
            mm_data={"audios": [
                np.zeros(1600, dtype=np.float32),
                np.zeros(1600, dtype=np.float32),
            ]},
            mm_kwargs={},
            tok_kwargs={},
        )
        # Each faked transcript is [1,2,3]; two clips -> per-item sizes [3,3],
        # flat length 6 (spliced at the two markers by _get_prompt_updates).
        assert bf["audio_num_tokens"].tolist() == [3, 3]
        assert len(bf["audio_token_ids"]) == 6

    def test_transcribe_returns_full_transcript(self, monkeypatch):
        # The transcript is never truncated here — it is spliced in full and an
        # oversized prompt is rejected downstream by vLLM's length check.
        capture = {}
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor(info, monkeypatch, capture)
        info.get_tokenizer = lambda: SimpleNamespace(
            encode=lambda text, add_special_tokens=False: [1, 2, 3, 4, 5]
        )
        assert proc._transcribe(np.zeros(1600, dtype=np.float32), {}) == [1, 2, 3, 4, 5]

    def test_self_chunks_and_chunk_params_forwarded(self, monkeypatch):
        capture = {}
        info = _make_info(
            asr_enabled=True,
            asr_model_id="w",
            asr_self_chunks=False,
            asr_chunk_length_s=20.0,
            asr_chunk_overlap_s=3.0,
        )
        proc = _make_processor(info, monkeypatch, capture)
        proc._transcribe(np.zeros(1600, dtype=np.float32), {})
        assert capture["self_chunks"] is False
        assert capture["chunk_length_s"] == 20.0
        assert capture["chunk_overlap_s"] == 3.0
