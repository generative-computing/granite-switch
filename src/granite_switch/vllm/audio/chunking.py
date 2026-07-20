# SPDX-License-Identifier: Apache-2.0
"""Encoder-agnostic long-audio chunking for the cascade.

A backend with a fixed input window (e.g. a speech encoder that only accepts a
few seconds at a time) cannot ingest a long clip in one shot. This module splits
a long waveform into overlapping fixed-length windows and stitches the resulting
per-window transcripts back into one string — the same job Whisper's internal
``chunk_length_s`` does, but lifted **above** the backend so *any* transcriber
inherits long-audio support.

Kept pure (numpy + stdlib, no vLLM / torch / transformers import) so it unit-
tests on CPU. The transcriber decides whether to use it via its ``self_chunks``
flag: backends that already chunk internally (Whisper) bypass this entirely.

Why overlap + dedup: cutting on hard boundaries can split a word across two
windows, so consecutive windows overlap and each boundary word is whole in at
least one of them. The overlap region is therefore transcribed twice; the merge
finds the repeated span at each seam and keeps one copy.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

# Cap on how many trailing/leading words we search for a seam overlap. Comfortably
# larger than any plausible word count inside a few seconds of overlap, but bounded
# so the merge stays linear in transcript length.
_MAX_SEAM_WORDS = 60


def split_waveform(
    samples: np.ndarray,
    sr: int,
    window_s: float,
    overlap_s: float,
) -> List[np.ndarray]:
    """Split ``samples`` into overlapping windows of ``window_s`` seconds.

    Consecutive windows advance by ``window_s - overlap_s`` seconds, so each pair
    shares ``overlap_s`` of audio. A clip already shorter than one window is
    returned as a single segment (no copying/splitting).

    Args:
        samples: 1-D mono waveform.
        sr: Sample rate of ``samples`` in Hz.
        window_s: Window length in seconds (must be > 0).
        overlap_s: Overlap between consecutive windows in seconds (0 <= overlap_s
            < window_s).

    Returns:
        A list of 1-D numpy views/arrays, in order.
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

    segments: List[np.ndarray] = []
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


def _seam_overlap(prev: List[str], nxt: List[str]) -> int:
    """Longest k such that the last k words of ``prev`` match the first k of ``nxt``.

    Comparison is punctuation/case-insensitive because ASR often renders the
    overlap region slightly differently on each side of the seam. Returns 0 when
    there is no matching overlap.
    """
    max_k = min(len(prev), len(nxt), _MAX_SEAM_WORDS)
    for k in range(max_k, 0, -1):
        a = [_norm_word(w) for w in prev[-k:]]
        b = [_norm_word(w) for w in nxt[:k]]
        if a == b:
            return k
    return 0


def merge_transcripts(transcripts: List[str]) -> str:
    """Concatenate per-window transcripts, de-duplicating the overlap at each seam.

    For each new window, find the longest word-level overlap between the tail of
    the text so far and the head of the new window, and drop that duplicated span
    from the new window before appending. Empty windows are skipped.
    """
    merged: List[str] = []
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
