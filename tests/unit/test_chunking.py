# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the encoder-agnostic long-audio chunker.

Pure numpy/stdlib logic (no vLLM), so — like ``test_asr.py`` — the leaf module is
loaded directly by file path to skip the vLLM-importing package ``__init__``.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

_CHUNKING_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src/granite_switch/vllm/audio/chunking.py"
)
_spec = importlib.util.spec_from_file_location("gs_chunking_under_test", _CHUNKING_PATH)
chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chunking)


SR = 16_000


class TestSplitWaveform:
    def test_short_clip_returns_single_segment(self):
        wav = np.zeros(5 * SR, dtype=np.float32)
        segs = chunking.split_waveform(wav, SR, 30.0, 5.0)
        assert len(segs) == 1
        assert len(segs[0]) == len(wav)

    def test_exact_window_is_one_segment(self):
        wav = np.zeros(30 * SR, dtype=np.float32)
        segs = chunking.split_waveform(wav, SR, 30.0, 5.0)
        assert len(segs) == 1

    def test_overlapping_windows_cover_and_step(self):
        # 100s, 30s window, 5s overlap -> step 25s -> starts 0,25,50,75
        wav = np.arange(100 * SR, dtype=np.float32)
        segs = chunking.split_waveform(wav, SR, 30.0, 5.0)
        assert [round(len(s) / SR, 3) for s in segs] == [30.0, 30.0, 30.0, 25.0]
        # Every sample is covered, and consecutive windows overlap by 5s.
        assert segs[0][-1] > segs[1][0]  # overlap: seg0 tail past seg1 head start

    def test_last_segment_is_remainder(self):
        wav = np.zeros(70 * SR, dtype=np.float32)  # 70s -> 30,30,20 (step 25)
        segs = chunking.split_waveform(wav, SR, 30.0, 5.0)
        assert [round(len(s) / SR, 3) for s in segs] == [30.0, 30.0, 20.0]

    def test_no_overlap_tiles(self):
        wav = np.zeros(60 * SR, dtype=np.float32)
        segs = chunking.split_waveform(wav, SR, 30.0, 0.0)
        assert [round(len(s) / SR, 3) for s in segs] == [30.0, 30.0]

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            chunking.split_waveform(np.zeros(SR), SR, 0.0, 0.0)

    def test_overlap_ge_window_raises(self):
        with pytest.raises(ValueError):
            chunking.split_waveform(np.zeros(SR), SR, 30.0, 30.0)


class TestMergeTranscripts:
    def test_empty_list(self):
        assert chunking.merge_transcripts([]) == ""

    def test_single(self):
        assert chunking.merge_transcripts(["hello world"]) == "hello world"

    def test_overlap_deduped(self):
        out = chunking.merge_transcripts(
            ["what is the capital of Israel", "the capital of Israel is Jerusalem"]
        )
        assert out == "what is the capital of Israel is Jerusalem"

    def test_no_overlap_concatenates(self):
        assert chunking.merge_transcripts(["hello world", "foo bar"]) == (
            "hello world foo bar"
        )

    def test_empty_segments_skipped(self):
        assert chunking.merge_transcripts(["a b c", "", "c d e"]) == "a b c d e"

    def test_seam_is_punct_and_case_insensitive(self):
        # "Store." vs "store," and "the" both normalize equal -> 2-word overlap.
        out = chunking.merge_transcripts(["going to the Store.", "the store, i went"])
        assert out == "going to the Store. i went"

    def test_identical_adjacent_collapses(self):
        assert chunking.merge_transcripts(["thank you", "thank you"]) == "thank you"

    def test_partial_overlap_keeps_tail(self):
        out = chunking.merge_transcripts(["a b c d", "c d e f"])
        assert out == "a b c d e f"
