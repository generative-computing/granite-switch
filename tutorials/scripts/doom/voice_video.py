# SPDX-License-Identifier: Apache-2.0
"""Give a recorded Doom video its voices: speak the transcript record_video.py wrote.

Every line in ``<video>.talk.json`` is spoken by Chatterbox-Turbo at the moment it
was said, the player's in the default voice and the watcher's pitched up, so the
two are told apart; a line starts after the previous one ends. The track is then
muxed into the video.

Runs in an environment with Chatterbox (``pip install chatterbox-tts``), which
pins its own torch and transformers, so not the vLLM one::

    python voice_video.py out/clip.mp4 --out out/clip_voiced.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def pitch_up(wav: np.ndarray, factor: float = 1.15) -> np.ndarray:
    """A second voice from the same model: shorter, higher."""
    n = int(len(wav) / factor)
    x = np.linspace(0, len(wav) - 1, n)
    return np.interp(x, np.arange(len(wav)), wav).astype(np.float32)


def ffmpeg_exe(given: str | None) -> str:
    if given:
        return given
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--talk", type=Path, help="Default: <video>.talk.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gap", type=float, default=0.15, help="Seconds between lines")
    ap.add_argument("--ffmpeg", help="ffmpeg binary (default: imageio-ffmpeg's)")
    args = ap.parse_args()

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    talk = json.loads((args.talk or args.video.with_suffix(".talk.json")).read_text())
    tts = ChatterboxTurboTTS.from_pretrained(device=args.device)
    sr = tts.sr
    dur = talk["frames"] / talk["fps"]
    track = np.zeros(int((dur + 10) * sr), np.float32)
    t_free = 0.0
    for ev in talk["lines"]:
        wav = tts.generate(ev["text"]).squeeze().cpu().numpy().astype(np.float32)
        if ev["who"] == "player":
            wav = pitch_up(wav)
        start = max(ev["t"], t_free)
        i = int(start * sr)
        n = min(len(wav), len(track) - i)
        track[i : i + n] += wav[:n]
        t_free = start + len(wav) / sr + args.gap
        print(f"{start:6.1f}s  {ev['who']:<6} {ev['text']}", flush=True)
    track = np.clip(track[: int(dur * sr)], -1.0, 1.0)
    wav_path = args.out.with_suffix(".wav")
    sf.write(wav_path, track, sr)
    subprocess.run(
        [
            ffmpeg_exe(args.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(args.video),
            "-i",
            str(wav_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(args.out),
        ],
        check=True,
    )
    print(f"wrote {args.out} ({len(talk['lines'])} lines, {dur:.0f} s)")


if __name__ == "__main__":
    main()
