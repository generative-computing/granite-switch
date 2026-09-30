# SPDX-License-Identifier: Apache-2.0
"""Give a recorded Doom video its voices: speak the transcript record_video.py wrote.

Every line in ``<video>.talk.json`` is spoken at the moment it was said, the
player's in the default voice and the watcher's pitched up (or, with
``--partner-voice``, in another model's default voice), so the two are told
apart; a line starts after the previous one ends. The game's own sound
(``<video>.sfx.wav``, from ``record_video.py --sfx``) goes under the voices,
turned down while a line plays. The track is then muxed into the video. Two Chatterbox models: ``standard`` has the expressivity controls
(``--exaggeration``: emotional intensity, 0.5 neutral; ``--cfg-weight``: lower is
slower, more deliberate) and cannot voice sound tags, so they are dropped;
``turbo`` is faster, ignores those controls and voices tags such as [chuckle].

Runs in an environment with Chatterbox (``pip install chatterbox-tts``), which
pins its own torch and transformers, so not the vLLM one::

    python voice_video.py out/clip.mp4 --out out/clip_voiced.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

TAG = re.compile(r"\[[a-z ]+\]\s*")  # [chuckle], [sigh], ...


def pitch_up(wav: np.ndarray, factor: float = 1.15) -> np.ndarray:
    """A second voice from the same model: shorter, higher."""
    n = int(len(wav) / factor)
    x = np.linspace(0, len(wav) - 1, n)
    return np.interp(x, np.arange(len(wav)), wav).astype(np.float32)


def shift(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """The same voice lower (or higher), at the same pace."""
    import librosa

    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=semitones).astype(np.float32)


def game_sound(
    path: Path,
    sr: int,
    n: int,
    spoken: list[tuple[int, int]],
    gain_db: float,
    duck_db: float,
) -> np.ndarray:
    """The game's stereo sound at ``sr``, ``n`` samples, ``gain_db`` down, and
    ``duck_db`` further down (with 0.1 s ramps) wherever a line is spoken."""
    import librosa

    fx, fsr = sf.read(path, dtype="float32", always_2d=True)
    fx = librosa.resample(fx.T, orig_sr=fsr, target_sr=sr).T[:n]
    fx = np.pad(fx, ((0, n - len(fx)), (0, 0)))
    level = np.ones(n, np.float32)
    for a, b in spoken:
        level[a:b] = 10 ** (duck_db / 20)
    ramp = int(0.1 * sr)
    level = np.convolve(level, np.ones(ramp) / ramp, "same").astype(np.float32)
    return fx * (10 ** (gain_db / 20)) * level[:, None]


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
    ap.add_argument("--tts", choices=("standard", "turbo"), default="standard")
    ap.add_argument("--exaggeration", type=float, default=0.8, help="standard only")
    ap.add_argument("--cfg-weight", type=float, default=0.3, help="standard only")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Semitones to shift the player's voice (e.g. -3: deeper, same pace)",
    )
    ap.add_argument(
        "--sfx",
        type=Path,
        help="The game's sound (default: <video>.sfx.wav from record_video.py --sfx, "
        "if there)",
    )
    ap.add_argument("--sfx-gain", type=float, default=-10.0, help="dB")
    ap.add_argument(
        "--duck",
        type=float,
        default=-14.0,
        help="dB on the game sound while a line plays",
    )
    ap.add_argument(
        "--partner-voice",
        choices=("pitched", "standard", "turbo"),
        default="pitched",
        help="The watcher's voice: the player's pitched up, or a model's own default "
        "voice (Chatterbox-Turbo's is an American woman's)",
    )
    args = ap.parse_args()

    def load(kind: str):
        """A Chatterbox model and its generation options."""
        if kind == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            return ChatterboxTurboTTS.from_pretrained(device=args.device), {
                "temperature": args.temperature
            }
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=args.device), {
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg_weight,
            "temperature": args.temperature,
        }

    talk = json.loads((args.talk or args.video.with_suffix(".talk.json")).read_text())
    tts, kw = load(args.tts)
    sr = tts.sr
    partner = None  # (model, options) voicing the watcher, if not the player's
    if args.partner_voice != "pitched":
        partner = (
            (tts, kw) if args.partner_voice == args.tts else load(args.partner_voice)
        )
    dur = talk["frames"] / talk["fps"]
    track = np.zeros(int((dur + 10) * sr), np.float32)
    spoken: list[tuple[int, int]] = []  # sample ranges with a line playing
    t_free = 0.0
    for ev in talk["lines"]:
        text = ev["text"] if args.tts == "turbo" else TAG.sub("", ev["text"]).strip()
        if not text:
            continue
        if ev["who"] == "player" and partner is not None:
            model, opts = partner
            wav = (
                model.generate(text, **opts).squeeze().cpu().numpy().astype(np.float32)
            )
            if model.sr != sr:
                import librosa

                wav = librosa.resample(wav, orig_sr=model.sr, target_sr=sr)
        else:
            wav = tts.generate(text, **kw).squeeze().cpu().numpy().astype(np.float32)
            if ev["who"] == "player":
                wav = pitch_up(wav)
            elif args.pitch:
                wav = shift(wav, sr, args.pitch)
        start = max(ev["t"], t_free)
        if start >= dur:  # lines queued past the end of the video are dropped
            break
        i = int(start * sr)
        if i + len(wav) > len(track):
            track = np.concatenate(
                [track, np.zeros(i + len(wav) - len(track), np.float32)]
            )
        track[i : i + len(wav)] += wav
        spoken.append((i, i + len(wav)))
        t_free = start + len(wav) / sr + args.gap
        print(f"{start:6.1f}s  {ev['who']:<6} {ev['text']}", flush=True)
    track = track[: int(dur * sr)]
    mix = np.stack([track, track], axis=1)
    sfx_path = args.sfx or args.video.with_suffix(".sfx.wav")
    if sfx_path.exists():
        mix += game_sound(sfx_path, sr, len(track), spoken, args.sfx_gain, args.duck)
    wav_path = args.out.with_suffix(".wav")
    sf.write(wav_path, np.clip(mix, -1.0, 1.0), sr)
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
