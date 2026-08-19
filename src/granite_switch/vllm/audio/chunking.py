# SPDX-License-Identifier: Apache-2.0
"""Encoder-agnostic long-audio chunking for the cascade.

Splits a waveform into overlapping windows and stitches the per-window
transcripts, so a backend with a fixed input window inherits long-audio support.
Windows overlap because a hard cut can split a word; the overlap is therefore
transcribed twice and the merge keeps one copy.

Pure numpy + stdlib (no torch/transformers) so it unit-tests on CPU.
"""

from __future__ import annotations

import re

import numpy as np

# Bounded seam search so the merge stays linear in transcript length.
_MAX_SEAM_WORDS = 60


def split_waveform(
    samples: np.ndarray,
    sr: int,
    window_s: float,
    overlap_s: float,
) -> list[np.ndarray]:
    """Split a 1-D mono waveform into overlapping windows, in order.

    Windows advance by ``window_s - overlap_s``. A clip shorter than one window
    is returned as a single segment.
    """
    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    if not (0 <= overlap_s < window_s):
        raise ValueError(
            f"overlap_s must satisfy 0 <= overlap_s < window_s; got "
            f"overlap_s={overlap_s}, window_s={window_s}"
        )

    window = int(round(window_s * sr))
    step = int(round((window_s - overlap_s) * sr))
    window = max(1, window)
    step = max(1, step)

    n = len(samples)
    if n <= window:
        return [samples]

    segments: list[np.ndarray] = []
    start = 0
    while start < n:
        segments.append(samples[start : start + window])
        if start + window >= n:
            break
        start += step
    return segments


def _norm_word(word: str) -> str:
    """Lowercase and strip surrounding punctuation for seam comparison."""
    return re.sub(r"[^\w]+", "", word).lower()


def _seam_overlap(prev: list[str], nxt: list[str]) -> int:
    """Longest k such that the last k words of ``prev`` match the first k of ``nxt``.

    Punctuation/case-insensitive: ASR renders the overlap slightly differently on
    each side of a seam. Returns 0 when nothing matches.
    """
    max_k = min(len(prev), len(nxt), _MAX_SEAM_WORDS)
    for k in range(max_k, 0, -1):
        a = [_norm_word(w) for w in prev[-k:]]
        b = [_norm_word(w) for w in nxt[:k]]
        if a == b:
            return k
    return 0


def merge_transcripts(transcripts: list[str]) -> str:
    """Concatenate per-window transcripts, de-duplicating the overlap at each seam."""
    merged: list[str] = []
    for text in transcripts:
        words = text.split()
        if not words:
            continue
        if not merged:
            merged = words
            continue
        k = _seam_overlap(merged, words)
        merged.extend(words[k:])
    return " ".join(merged)
