# SPDX-License-Identifier: Apache-2.0
"""The ``--asr-no-speech-marker`` compose option: repeatable, no default."""

import pytest

from granite_switch.composer.compose_granite_switch import _compose_argparser

pytestmark = pytest.mark.audio


def _parse(*extra):
    return _compose_argparser().parse_args(["--adapters", "org/lib", *extra])


class TestNoSpeechMarkerFlag:
    def test_defaults_to_none(self):
        # Model-specific, so compose sets nothing unless asked.
        assert _parse().asr_no_speech_markers is None

    def test_single_marker(self):
        args = _parse("--asr-no-speech-marker", "<|nospeech|>")
        assert args.asr_no_speech_markers == ["<|nospeech|>"]

    def test_repeated_flag_collects_list(self):
        args = _parse(
            "--asr-no-speech-marker",
            "<|nospeech|>",
            "--asr-no-speech-marker",
            "<|nocaptions|>",
        )
        assert args.asr_no_speech_markers == ["<|nospeech|>", "<|nocaptions|>"]

    def test_dash_leading_marker_with_equals(self):
        args = _parse("--asr-no-speech-marker=-silence-")
        assert args.asr_no_speech_markers == ["-silence-"]

    def test_empty_marker_rejected(self):
        with pytest.raises(SystemExit):
            _parse("--asr-no-speech-marker", "")
