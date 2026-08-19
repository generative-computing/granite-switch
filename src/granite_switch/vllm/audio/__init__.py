# SPDX-License-Identifier: Apache-2.0
"""Audio (ASR) preprocessing for the Granite Switch vLLM backend.

Speech-to-text cascade: audio is transcribed and the transcript tokens are
spliced into the prompt, so the decoder only ever sees text. See docs/AUDIO.md.
"""

from .asr import DEFAULT_ASR_MODEL_ID, ASRTranscriber, transcribe

__all__ = [
    "DEFAULT_ASR_MODEL_ID",
    "ASRTranscriber",
    "transcribe",
]
