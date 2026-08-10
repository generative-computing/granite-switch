# SPDX-License-Identifier: Apache-2.0
"""vLLM-tier tests for the audio ASR multimodal processor.

Needs vLLM importable (the base classes come from it) but no GPU and no real ASR
model — the transcriber is faked, so what is under test is the plumbing: modality
gating on ``asr_enabled``, the config accessors, and decode kwargs reaching the
transcriber. The merge logic itself is unit-tested in tests/unit/test_asr.py.
"""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None

pytestmark = [
    pytest.mark.audio,
    pytest.mark.skipif(not _VLLM_AVAILABLE, reason="requires vLLM installed"),
]

if _VLLM_AVAILABLE:
    from granite_switch.vllm.audio import processor as proc_mod
    from granite_switch.vllm.audio.asr import DEFAULT_ASR_MODEL_ID
    from granite_switch.vllm.audio.processor import (
        GraniteSwitchASRMultiModalProcessor,
        GraniteSwitchASRProcessingInfo,
    )


def _make_info(*, max_model_len=131072, **cfg_attrs):
    """A ProcessingInfo whose get_hf_config() returns a stub config.

    Bypasses __init__, which needs a full vLLM InputProcessingContext.
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
        assert info.get_mm_max_tokens_per_item(20000, {"audio": 1}) == {"audio": 20000}
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
        assert (
            _make_info(asr_enabled=True, max_model_len=16000)._max_model_len() == 16000
        )

    def test_max_model_len_falls_back_to_position_embeddings(self):
        info = _make_info(asr_enabled=True, max_position_embeddings=4096)
        info.ctx = SimpleNamespace(model_config=SimpleNamespace(max_model_len=None))
        assert info._max_model_len() == 4096


def _make_processor(info, monkeypatch, capture):
    """A processor whose transcriber is faked; records what it was called with."""
    info.get_tokenizer = lambda: SimpleNamespace(
        encode=lambda text, add_special_tokens=False: [1, 2, 3],
        # No marker registered in this stub, so the reserved-token guard in
        # _transcribe finds nothing to reject.
        convert_tokens_to_ids=lambda token: None,
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

    def fake_get_transcriber(
        model_id=None, device="cpu", pipeline_kwargs=None, dtype=None
    ):
        capture["model_id"] = model_id
        capture["device"] = device
        capture["pipeline_kwargs"] = pipeline_kwargs
        capture["dtype"] = dtype
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

        ids = proc._transcribe(np.zeros(1600, dtype=np.float32), {"language": "fr"})
        assert ids == [1, 2, 3]
        assert capture["model_id"] == "whisper-x"
        assert capture["pipeline_kwargs"] == {"chunk_length_s": 15}
        assert capture["generate_kwargs"] == {"language": "fr"}
        assert capture["sampling_rate"] == proc_mod._TARGET_SR

    def test_dtype_forwarded_from_config(self, monkeypatch):
        capture = {}
        info = _make_info(asr_enabled=True, asr_device="cuda:0", asr_dtype="float32")
        proc = _make_processor(info, monkeypatch, capture)
        proc._transcribe(np.zeros(1600, dtype=np.float32), {})
        assert capture["dtype"] == "float32"

    def test_dtype_defaults_to_none(self, monkeypatch):
        capture = {}
        proc = _make_processor(_make_info(asr_enabled=True), monkeypatch, capture)
        proc._transcribe(np.zeros(1600, dtype=np.float32), {})
        assert capture["dtype"] is None

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
            mm_kwargs={"language": "fr"},  # per-request override
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
            mm_data={
                "audios": [
                    np.zeros(1600, dtype=np.float32),
                    np.zeros(1600, dtype=np.float32),
                ]
            },
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
            encode=lambda text, add_special_tokens=False: [1, 2, 3, 4, 5],
            convert_tokens_to_ids=lambda token: None,
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


def _make_processor_transcribing(info, monkeypatch, text, *, ids_for):
    """Processor whose transcriber returns a fixed ``text``; ``ids_for`` maps it to ids."""
    info.get_tokenizer = lambda: SimpleNamespace(
        encode=lambda t, add_special_tokens=False: ids_for(t),
        convert_tokens_to_ids=lambda token: None,
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
            return text

    monkeypatch.setattr(
        proc_mod,
        "get_transcriber",
        lambda model_id=None, device="cpu", pipeline_kwargs=None, dtype=None: (
            FakeTranscriber()
        ),
    )
    return proc


_BLANK_ID = 5


class TestEmptyAndSilentClip:
    """Silence transcribes to "" -> stands in as _EMPTY_TRANSCRIPT_TEXT.

    A zero-length replacement is not a legal placeholder: vLLM skips zero-length
    content when locating placeholders and then rejects the request with
    ``found 0 prompt placeholders``. So every clip must yield >=1 token, however
    little was said in it.
    """

    def _ids_for(self, text):
        if text == "":
            return []
        if text == proc_mod._EMPTY_TRANSCRIPT_TEXT:
            return [_BLANK_ID]
        return [1, 2, 3]

    def test_empty_transcript_yields_blank_token(self, monkeypatch):
        import torch

        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor_transcribing(
            info, monkeypatch, "", ids_for=self._ids_for
        )

        bf = proc._call_hf_processor(
            prompt="<|audio|>",
            mm_data={"audios": [np.zeros(1600, dtype=np.float32)]},
            mm_kwargs={},
            tok_kwargs={},
        )
        # One token, not zero — the item is still a findable placeholder.
        assert bf["audio_num_tokens"].tolist() == [1]
        assert bf["audio_token_ids"].tolist() == [_BLANK_ID]
        assert bf["audio_token_ids"].dtype == torch.long

    def test_whitespace_only_transcript_also_falls_back(self, monkeypatch):
        # transcribe() strips, so "" is the norm; a backend that does not strip
        # must not slip a whitespace-only transcript past the check.
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor_transcribing(
            info, monkeypatch, "   \n", ids_for=self._ids_for
        )
        assert proc._transcribe(np.zeros(1600, dtype=np.float32), {}) == [_BLANK_ID]

    def test_untokenizable_fallback_raises_clearly(self, monkeypatch):
        # If even the fallback encodes to nothing, fail with our message rather
        # than vLLM's opaque placeholder-count error.
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor_transcribing(
            info, monkeypatch, "", ids_for=lambda text: []
        )
        with pytest.raises(ValueError, match="no tokens"):
            proc._transcribe(np.zeros(1600, dtype=np.float32), {})

    def test_nonempty_transcript_untouched(self, monkeypatch):
        # The fallback must not perturb a clip that did contain speech.
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor_transcribing(
            info, monkeypatch, "words", ids_for=self._ids_for
        )
        assert proc._transcribe(np.zeros(1600, dtype=np.float32), {}) == [1, 2, 3]

    def test_blank_clip_replacement_is_not_empty(self, monkeypatch):
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor_transcribing(
            info, monkeypatch, "", ids_for=self._ids_for
        )

        class _Kwargs:
            def __init__(self, data):
                self._data = data

            def get_data(self):
                return self._data

        import torch

        out = _Kwargs(
            {
                "audio_num_tokens": torch.tensor([1], dtype=torch.long),
                "audio_token_ids": torch.tensor([_BLANK_ID], dtype=torch.long),
            }
        )
        updates = proc._get_prompt_updates(None, {}, out)
        assert len(updates) == 1
        assert updates[0].replacement(0) == [_BLANK_ID]

    def test_mixed_blank_and_nonempty_clips(self, monkeypatch):
        info = _make_info(asr_enabled=True, asr_max_audio_clips=4, asr_model_id="w")
        texts = iter(["", "words"])
        info.get_tokenizer = lambda: SimpleNamespace(
            encode=lambda t, add_special_tokens=False: self._ids_for(t),
            convert_tokens_to_ids=lambda token: None,
        )
        proc = object.__new__(GraniteSwitchASRMultiModalProcessor)
        proc.info = info

        class FakeTranscriber:
            def transcribe(self, audio, **kw):
                return next(texts)

        monkeypatch.setattr(
            proc_mod,
            "get_transcriber",
            lambda model_id=None, device="cpu", pipeline_kwargs=None, dtype=None: (
                FakeTranscriber()
            ),
        )

        bf = proc._call_hf_processor(
            prompt="<|audio|> <|audio|>",
            mm_data={
                "audios": [
                    np.zeros(1600, dtype=np.float32),
                    np.zeros(1600, dtype=np.float32),
                ]
            },
            mm_kwargs={},
            tok_kwargs={},
        )
        # The silent clip contributes 1 token instead of 0, so both items keep a
        # distinct, non-empty span and neither gets dropped.
        assert bf["audio_num_tokens"].tolist() == [1, 3]
        assert bf["audio_token_ids"].tolist() == [_BLANK_ID, 1, 2, 3]


_MARKER_ID = 99
_TRANSCRIPT_IDS = [7, 8]


class _MarkerTokenizer:
    """Marker is one special id, the transcript two; anything else is per-char.

    A class, not a SimpleNamespace: vLLM's ``_seq2tokens`` goes through an
    ``lru_cache`` keyed on the tokenizer, so it has to be hashable.
    """

    def encode(self, text, add_special_tokens=False):
        if text == "hello world":
            return list(_TRANSCRIPT_IDS)
        # Split on the marker the way a fast tokenizer splits on a registered
        # special token; everything else is one id per character.
        ids = []
        for i, part in enumerate(text.split(proc_mod.AUDIO_MARKER)):
            if i:
                ids.append(_MARKER_ID)
            ids.extend(ord(c) for c in part)
        return ids

    def decode(self, ids):
        return "".join(
            proc_mod.AUDIO_MARKER if i == _MARKER_ID else chr(i) for i in ids
        )

    def convert_tokens_to_ids(self, token):
        # A real tokenizer answers with the unk id for an unknown token, not
        # None; there is no unk here, so anything else is simply not a token.
        return _MARKER_ID if token == proc_mod.AUDIO_MARKER else None

    def convert_ids_to_tokens(self, token_id):
        return proc_mod.AUDIO_MARKER if token_id == _MARKER_ID else chr(token_id)


class TestPromptUpdatesAreApplied:
    """The marker must actually become transcript ids on the *uncached* path.

    ``_hf_processor_applies_updates`` is vLLM's "did you expand the placeholder
    yourself?" hook. We do not — we leave ``<|audio|>`` in the prompt and return
    the transcript out of band — so it must report False or vLLM skips our
    ``PromptReplacement`` and then fails to find the transcript it was promised.

    Only the uncached path (``--mm-processor-cache-gb 0``) consults the hook, so
    the default cache setting hid this. These tests drive vLLM's real
    ``_apply_hf_processor_text_mm`` / ``_maybe_apply_prompt_updates`` rather than
    asserting on the override in isolation.
    """

    def _audio_items(self, count=1):
        from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems

        clips = [np.zeros(1600, dtype=np.float32) for _ in range(count)]
        return MultiModalDataItems({"audio": AudioProcessorItems(clips)})

    def _proc(self, monkeypatch, count=1, text="hello world"):
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor_transcribing(
            info, monkeypatch, text, ids_for=lambda t: [7, 8]
        )
        # ids_for is unused: the marker tokenizer does the encoding, so the
        # transcript text maps to ids the same way the real path would.
        tokenizer = _MarkerTokenizer()
        info.get_tokenizer = lambda: tokenizer
        return proc, self._audio_items(count)

    def test_hook_reports_updates_not_applied(self, monkeypatch):
        proc, items = self._proc(monkeypatch)
        assert (
            proc._hf_processor_applies_updates(
                prompt_text=proc_mod.AUDIO_MARKER,
                mm_items=items,
                hf_processor_mm_kwargs={},
                tokenization_kwargs={},
            )
            is False
        )

    def test_uncached_path_reports_updates_not_applied(self, monkeypatch):
        # Real vLLM code: this is the call site that decides whether the
        # replacement runs (processing/processor.py, _apply_hf_processor_text_mm).
        proc, items = self._proc(monkeypatch)
        prompt_ids, _, is_update_applied = proc._apply_hf_processor_text_mm(
            prompt_text=proc_mod.AUDIO_MARKER,
            mm_items=items,
            hf_processor_mm_kwargs={},
            tokenization_kwargs={},
        )
        # Marker still un-expanded, hence False.
        assert prompt_ids == [_MARKER_ID]
        assert is_update_applied is False

    def _apply_uncached(self, proc, items, prompt_text):
        """vLLM's ``apply()`` with cache=None, minus the hashing/cache plumbing."""
        from vllm.multimodal.inputs import MultiModalKwargsItems

        prompt_ids, processed, is_update_applied = proc._apply_hf_processor_text_mm(
            prompt_text=prompt_text,
            mm_items=items,
            hf_processor_mm_kwargs={},
            tokenization_kwargs={},
        )
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            processed, proc._get_mm_fields_config(processed, {})
        )
        updates = proc._get_mm_prompt_updates(items, {}, mm_kwargs)
        return proc._maybe_apply_prompt_updates(
            mm_items=items,
            prompt_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_prompt_updates=updates,
            is_update_applied=is_update_applied,
        )

    def test_marker_becomes_transcript_ids(self, monkeypatch):
        proc, items = self._proc(monkeypatch)
        new_ids, placeholders = self._apply_uncached(proc, items, proc_mod.AUDIO_MARKER)
        # Marker replaced, not merely searched for.
        assert new_ids == _TRANSCRIPT_IDS
        assert _MARKER_ID not in new_ids
        # And the placeholder range points at the transcript.
        (ph,) = placeholders["audio"]
        assert (ph.start_idx, ph.length) == (0, len(_TRANSCRIPT_IDS))

    def test_transcript_spliced_between_surrounding_text(self, monkeypatch):
        proc, items = self._proc(monkeypatch)
        prompt = proc_mod.AUDIO_MARKER + "Q"
        new_ids, placeholders = self._apply_uncached(proc, items, prompt)
        assert new_ids == [*_TRANSCRIPT_IDS, ord("Q")]
        (ph,) = placeholders["audio"]
        assert (ph.start_idx, ph.length) == (0, len(_TRANSCRIPT_IDS))

    def test_silent_clip_still_yields_a_placeholder(self, monkeypatch):
        # The other half of the same invariant: a clip with no speech in it used
        # to produce a zero-length replacement, which vLLM discards and then
        # rejects the request over. The fallback text keeps the span findable.
        proc, items = self._proc(monkeypatch, text="")
        new_ids, placeholders = self._apply_uncached(proc, items, proc_mod.AUDIO_MARKER)
        assert new_ids == [ord(proc_mod._EMPTY_TRANSCRIPT_TEXT)]
        (ph,) = placeholders["audio"]
        assert (ph.start_idx, ph.length) == (0, 1)

    def test_silent_clip_among_speech_clips(self, monkeypatch):
        proc, items = self._proc(monkeypatch, count=2, text="")
        new_ids, placeholders = self._apply_uncached(
            proc, items, proc_mod.AUDIO_MARKER * 2
        )
        assert new_ids == [ord(proc_mod._EMPTY_TRANSCRIPT_TEXT)] * 2
        # Both items keep their own placeholder; neither collapses into the other.
        assert len(placeholders["audio"]) == 2
        assert [p.length for p in placeholders["audio"]] == [1, 1]

    def test_two_clips_each_get_their_own_placeholder(self, monkeypatch):
        proc, items = self._proc(monkeypatch, count=2)
        new_ids, placeholders = self._apply_uncached(
            proc, items, proc_mod.AUDIO_MARKER * 2
        )
        assert new_ids == _TRANSCRIPT_IDS * 2
        # One placeholder per audio item, or _validate_mm_placeholders raises.
        assert len(placeholders["audio"]) == 2


class TestClipCeiling:
    """get_supported_mm_limits publishes asr_max_audio_clips as the clip ceiling."""

    def test_ceiling_equals_configured_max(self):
        assert _make_info(asr_enabled=True).get_supported_mm_limits() == {"audio": 32}
        assert _make_info(
            asr_enabled=True, asr_max_audio_clips=1
        ).get_supported_mm_limits() == {"audio": 1}
        assert _make_info(
            asr_enabled=True, asr_max_audio_clips=8
        ).get_supported_mm_limits() == {"audio": 8}

    def test_per_item_bound_never_exceeds_context(self):
        info = _make_info(asr_enabled=True, asr_max_audio_clips=32)
        seq_len = 4096
        for count in (1, 2, 8, 32):
            bound = info.get_mm_max_tokens_per_item(seq_len, {"audio": count})["audio"]
            assert bound == seq_len // count
            assert bound <= seq_len


_DELEGATED = object()


class TestMarkerSpoofingRejected:
    """The reserved marker must not be mintable from either untrusted side.

    ``<|audio|>`` is a registered special token, so it tokenizes to the real
    marker wherever it appears. vLLM pairs markers with audio items positionally
    and stops once every item is matched, leaving extras in the prompt verbatim,
    so a spoofed marker silently relocates the transcript while vLLM's own
    placeholder validation still sees matching counts and passes.
    """

    def _items(self, count):
        from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems

        clips = [np.zeros(1600, dtype=np.float32) for _ in range(count)]
        return MultiModalDataItems({"audio": AudioProcessorItems(clips)})

    def _proc(self, monkeypatch, *, control_ids=None):
        info = _make_info(
            asr_enabled=True,
            asr_model_id="w",
            adapter_token_ids=list(control_ids or []),
        )
        proc = object.__new__(GraniteSwitchASRMultiModalProcessor)
        proc.info = info
        info.get_tokenizer = lambda: _MarkerTokenizer()
        return proc

    def _apply(self, monkeypatch, proc, prompt, item_count):
        """Drive the real ``apply()`` override with the base call stubbed out.

        Stubbing the base lets the override's own behaviour be observed:
        ``_DELEGATED`` back means validation passed and it handed off.
        """
        from vllm.multimodal.processing import BaseMultiModalProcessor

        monkeypatch.setattr(
            BaseMultiModalProcessor,
            "apply",
            lambda self, *a, **k: _DELEGATED,
            raising=False,
        )
        inputs = SimpleNamespace(prompt=prompt, mm_data_items=self._items(item_count))
        return proc.apply(inputs)

    # ---- the marker injected via user text ----------------------------------

    def test_marker_injected_in_text_is_rejected(self, monkeypatch):
        """A user typing the marker adds a second one for a single clip."""
        proc = self._proc(monkeypatch)
        spoofed = f"{proc_mod.AUDIO_MARKER} hi {proc_mod.AUDIO_MARKER} what was said?"

        with pytest.raises(ValueError, match="marker") as exc:
            self._apply(monkeypatch, proc, spoofed, item_count=1)

        assert "2" in str(exc.value) and "1" in str(exc.value)

    def test_marker_with_no_audio_at_all_is_rejected(self, monkeypatch):
        """Text-only prompt spelling the marker — the shape behind CVE-2026-44222."""
        proc = self._proc(monkeypatch)

        with pytest.raises(ValueError, match="marker"):
            self._apply(monkeypatch, proc, proc_mod.AUDIO_MARKER, item_count=0)

    def test_token_id_prompt_is_counted_too(self, monkeypatch):
        """Callers may pass token ids; the count must not silently read zero."""
        proc = self._proc(monkeypatch)

        with pytest.raises(ValueError, match="marker"):
            self._apply(monkeypatch, proc, [_MARKER_ID, _MARKER_ID], item_count=1)

    def test_matching_counts_are_accepted(self, monkeypatch):
        """Negative control: the guard must not reject legitimate requests."""
        proc = self._proc(monkeypatch)

        assert (
            self._apply(
                monkeypatch,
                proc,
                f"{proc_mod.AUDIO_MARKER} what was said?",
                item_count=1,
            )
            is _DELEGATED
        )

    def test_two_clips_two_markers_accepted(self, monkeypatch):
        proc = self._proc(monkeypatch)
        prompt = proc_mod.AUDIO_MARKER * 2 + " compare them"

        assert self._apply(monkeypatch, proc, prompt, item_count=2) is _DELEGATED

    # ---- the marker injected via the transcript -----------------------------

    def test_marker_injected_via_transcript_is_rejected(self, monkeypatch):
        """ASR output containing the marker must not reach the prompt.

        ``encode(add_special_tokens=False)`` only suppresses *added* BOS/EOS, so a
        marker string inside the transcript still becomes the genuine marker id.
        """
        info = _make_info(asr_enabled=True, asr_model_id="w")
        proc = _make_processor_transcribing(
            info,
            monkeypatch,
            f"and then {proc_mod.AUDIO_MARKER} happened",
            ids_for=lambda t: [1],
        )
        info.get_tokenizer = lambda: _MarkerTokenizer()

        with pytest.raises(ValueError, match="reserved control token"):
            proc._transcribe(np.zeros(1600, dtype=np.float32))

    def test_adapter_control_token_via_transcript_is_rejected(self, monkeypatch):
        """The routing risk: the switch reads raw input_ids.

        A control token arriving from audio content would select an adapter, so
        transcripts carrying one are refused.
        """
        control_id = ord("Z")
        info = _make_info(
            asr_enabled=True, asr_model_id="w", adapter_token_ids=[control_id]
        )
        proc = _make_processor_transcribing(
            info, monkeypatch, "Z", ids_for=lambda t: [control_id]
        )
        info.get_tokenizer = lambda: _MarkerTokenizer()

        with pytest.raises(ValueError, match="reserved control token"):
            proc._transcribe(np.zeros(1600, dtype=np.float32))

    def test_clean_transcript_is_unaffected(self, monkeypatch):
        """Negative control: ordinary transcripts still pass through."""
        info = _make_info(
            asr_enabled=True, asr_model_id="w", adapter_token_ids=[_MARKER_ID + 1]
        )
        proc = _make_processor_transcribing(
            info, monkeypatch, "hello world", ids_for=lambda t: list(_TRANSCRIPT_IDS)
        )
        info.get_tokenizer = lambda: _MarkerTokenizer()

        assert proc._transcribe(np.zeros(1600, dtype=np.float32)) == _TRANSCRIPT_IDS


class TestAudioDurationLimits:
    """Over-long audio must be refused before anything transcribes it.

    ``asr_max_audio_clips`` bounds clip *count* only, and vLLM's prompt-length
    check runs after preprocessing — so without a duration bound a caller can
    have a multi-hour file fully transcribed and only then rejected. The bar here
    is not just "raises" but "raises without the ASR pipeline being touched".
    """

    def _items(self, *durations_s, sr=16_000):
        from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems

        clips = [
            np.zeros(int(seconds * sr), dtype=np.float32) for seconds in durations_s
        ]
        return MultiModalDataItems({"audio": AudioProcessorItems(clips)})

    def _proc(self, monkeypatch, **cfg):
        """Processor whose transcriber records whether it was ever reached."""
        info = _make_info(asr_enabled=True, asr_model_id="w", **cfg)
        info.get_tokenizer = lambda: _MarkerTokenizer()
        proc = object.__new__(GraniteSwitchASRMultiModalProcessor)
        proc.info = info

        calls = []

        def fake_get_transcriber(**kw):
            calls.append(kw)
            raise AssertionError("ASR must not be reached for a rejected request")

        monkeypatch.setattr(proc_mod, "get_transcriber", fake_get_transcriber)

        from vllm.multimodal.processing import BaseMultiModalProcessor

        monkeypatch.setattr(
            BaseMultiModalProcessor,
            "apply",
            lambda self, *a, **k: _DELEGATED,
            raising=False,
        )
        return proc, calls

    def _apply(self, proc, items):
        prompt = proc_mod.AUDIO_MARKER * len(items["audio"])
        return proc.apply(SimpleNamespace(prompt=prompt, mm_data_items=items))

    def test_overlong_clip_rejected_without_transcribing(self, monkeypatch):
        proc, calls = self._proc(monkeypatch, asr_max_audio_seconds_per_clip=60.0)

        with pytest.raises(ValueError, match="per-clip limit"):
            self._apply(proc, self._items(90.0))

        assert calls == [], "ASR was invoked for a request that should be rejected"

    def test_total_across_clips_rejected_without_transcribing(self, monkeypatch):
        proc, calls = self._proc(
            monkeypatch,
            asr_max_audio_seconds_per_clip=60.0,
            asr_max_total_audio_seconds=100.0,
        )

        # Each clip is legal on its own; together they are not.
        with pytest.raises(ValueError, match="total limit"):
            self._apply(proc, self._items(50.0, 50.0, 50.0))

        assert calls == []

    def test_sample_cap_rejected_without_transcribing(self, monkeypatch):
        """The rate-independent backstop fires even when the seconds pass."""
        proc, calls = self._proc(
            monkeypatch,
            asr_max_audio_seconds_per_clip=1000.0,
            asr_max_total_audio_seconds=1000.0,
            asr_max_audio_samples=16_000,  # 1 second's worth
        )

        with pytest.raises(ValueError, match="sample limit"):
            self._apply(proc, self._items(5.0))

        assert calls == []

    def test_error_names_the_offending_size_and_knob(self, monkeypatch):
        proc, _ = self._proc(monkeypatch, asr_max_audio_seconds_per_clip=30.0)

        with pytest.raises(ValueError) as exc:
            self._apply(proc, self._items(45.0))

        message = str(exc.value)
        assert "45.0s" in message and "30.0s" in message
        assert "asr_max_audio_seconds_per_clip" in message

    def test_within_limits_is_accepted(self, monkeypatch):
        """Negative control: legal audio must still get through."""
        proc, _ = self._proc(
            monkeypatch,
            asr_max_audio_seconds_per_clip=60.0,
            asr_max_total_audio_seconds=120.0,
        )

        assert self._apply(proc, self._items(10.0, 20.0)) is _DELEGATED

    def test_boundary_is_inclusive(self, monkeypatch):
        """Exactly at the limit is allowed; the check is > not >=."""
        proc, _ = self._proc(monkeypatch, asr_max_audio_seconds_per_clip=10.0)

        assert self._apply(proc, self._items(10.0)) is _DELEGATED

    def test_defaults_allow_ordinary_clips(self, monkeypatch):
        """A checkpoint with no limits configured keeps working."""
        proc, _ = self._proc(monkeypatch)

        assert self._apply(proc, self._items(5.0)) is _DELEGATED

    def test_text_only_request_unaffected(self, monkeypatch):
        from vllm.multimodal.parse import MultiModalDataItems

        proc, _ = self._proc(monkeypatch)
        empty = MultiModalDataItems({})

        assert proc.apply(SimpleNamespace(prompt="hi", mm_data_items=empty)) is (
            _DELEGATED
        )
