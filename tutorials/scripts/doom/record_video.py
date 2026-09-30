# SPDX-License-Identifier: Apache-2.0
"""Record the model playing, with a telemetry panel, as an MP4.

A script of segments drives the video. Each segment is either an adapter name
or a free-text instruction (routed by the router adapter), plus seconds of game
time::

    python record_video.py --model models/doom-round0 --out out/granite_doom.mp4 \
        --segment "fighter" 20 \
        --segment "stop fighting and grab health" 20 \
        --segment "collect all the loot" 20

Every action in the video comes from the composed checkpoint, one token per tic,
and so do the critic's danger (every tic) and the weapon plan (every 0.5 s): all
three adapters read the same history in one engine step. The panel shows the
measured wall-clock time of those steps (a 1 s rolling median, so it is
readable) against the 28.6 ms tic. The strip below the game is the full action
distribution as a heatmap, one row per action and one column per tic, filling
and then sliding, with the critic's danger under it. The game runs synchronously
and the video plays at 35 fps, so it is real game speed.

``--policy expert`` records the scripted teacher instead (no GPU needed; its
distribution is one-hot), which is how the layout is checked on a laptop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import (
    ACTION_LABELS,
    AUDIO_HZ,
    DISPLAY_ORDER,
    SHORT_LABELS,
    TIC_HZ,
    TIC_MS,
    WEAPON_NAMES,
    DoomEnv,
)
from expert import BEHAVIORS, PLAN_EVERY_TICS
from history import History, said_entry
from policy import (
    ARMS,
    CRITIC,
    DANGER_LEVELS,
    NARRATOR,
    ROUTER,
    make_policy,
    spoken_entry,
    state_text,
    talk_extra,
)
from talk import SALIENT_EVENTS, brief, sound_tag

W_GAME, H = 640, 480
W_PANEL = 360
BG = (22, 22, 22)
LAYER = (38, 38, 38)
LAYER2 = (57, 57, 57)
TEXT = (244, 244, 244)
TEXT2 = (198, 198, 198)
HELP = (168, 168, 168)
BLUE = (15, 98, 254)
TEAL = (8, 189, 186)
AMBER = (210, 161, 6)
RED = (250, 77, 86)
GREEN = (66, 190, 101)
COLORS = {"fighter": RED, "cautious": TEAL, "collector": AMBER}

HEAT_ROW = 10  # pixels per action row
H_HEAT = 82 + HEAT_ROW * len(DISPLAY_ORDER)  # heatmap strip under the game and panel
HEAT_PX = 3  # pixels per tic
HEAT_GUTTER = 100  # row labels
# Colour position is sqrt(p), so runner-up actions at a few percent stay visible.
STOPS = [(0.0, BG), (0.35, (0, 45, 156)), (0.7, BLUE), (1.0, (200, 228, 255))]


def font(size: int, mono: bool = False):
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
        if mono
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        if mono
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Panel:
    def __init__(self) -> None:
        self.f_title = font(15)
        self.f_big = font(34)
        self.f = font(15)
        self.f_small = font(12)
        self.f_mono = font(12, mono=True)

    def draw(self, img: Image.Image, t: dict) -> None:
        d = ImageDraw.Draw(img)
        x0 = W_GAME
        d.rectangle([x0, 0, x0 + W_PANEL, H], fill=BG)
        x = x0 + 18
        y = 14
        d.text((x, y), "GRANITE SWITCH  ·  DOOM", font=self.f_title, fill=TEXT2)
        y += 26
        d.text((x, y), "one output token per tic", font=self.f_small, fill=HELP)
        y += 26

        # Behavior chip
        c = COLORS.get(t["adapter"], TEXT)
        d.rounded_rectangle([x, y, x + 150, y + 28], radius=14, fill=LAYER2)
        d.ellipse([x + 12, y + 10, x + 20, y + 18], fill=c)
        d.text((x + 28, y + 5), t["adapter"], font=self.f, fill=TEXT)
        d.text(
            (x + 162, y + 6),
            f"{t.get('kind', 'aLoRA')} adapter",
            font=self.f_small,
            fill=HELP,
        )
        y += 38
        if t.get("instruction"):
            d.text((x, y), f"“{t['instruction'][:40]}”", font=self.f_small, fill=TEXT2)
            y += 16
            d.text(
                (x, y),
                f"router -> {t['routed']} ({100 * t['route_p']:.0f}%, {t['route_ms']:.1f} ms)",
                font=self.f_small,
                fill=HELP,
            )
            y += 20
        else:
            y += 36

        # Decision latency
        d.text((x, y), "DECISION  ·  1 s MEDIAN", font=self.f_small, fill=HELP)
        y += 16
        num = f"{t['ms']:.1f}"
        d.text((x, y), num, font=self.f_big, fill=TEXT)
        d.text(
            (x + d.textlength(num, font=self.f_big) + 6, y + 16),
            "ms",
            font=self.f,
            fill=TEXT2,
        )
        d.text(
            (x + 140, y + 4),
            f"run p50 {t['p50']:.1f} ms",
            font=self.f_small,
            fill=TEXT2,
        )
        d.text(
            (x + 140, y + 22),
            f"run p99 {t['p99']:.1f} ms",
            font=self.f_small,
            fill=TEXT2,
        )
        y += 46
        # Tic budget bar: 0..100 ms
        bw = W_PANEL - 36
        d.rectangle([x, y, x + bw, y + 10], fill=LAYER2)
        d.rectangle([x, y, x + int(bw * min(1.0, t["ms"] / 100.0)), y + 10], fill=TEAL)
        tx = x + int(bw * TIC_MS / 100.0)
        d.line([tx, y - 4, tx, y + 14], fill=TEXT, width=1)
        d.text((tx + 4, y + 13), "1 tic 28.6 ms", font=self.f_small, fill=TEXT2)
        d.text((x + bw - 50, y + 13), "100 ms", font=self.f_small, fill=HELP)
        y += 36

        # Action (the full distribution is in the heatmap below)
        d.text((x, y), "ACTION", font=self.f_small, fill=HELP)
        y += 16
        d.text(
            (x, y), ACTION_LABELS.get(t["action"], t["action"]), font=self.f, fill=TEXT
        )
        d.text((x + 200, y + 2), t["action"], font=self.f_mono, fill=HELP)
        y += 26

        # Critic: danger of being hit in the next second
        d.text((x, y), "CRITIC  ·  DANGER NEXT SECOND", font=self.f_small, fill=HELP)
        y += 16
        crit = [t["critic"].get(k, 0.0) for k in DANGER_LEVELS]
        tot = sum(crit) or 1.0
        x1 = x
        for p_, col in zip(crit, (GREEN, AMBER, RED)):
            x2 = x1 + int(bw * p_ / tot)
            if x2 > x1:
                d.rectangle([x1, y, x2, y + 10], fill=col)
            x1 = x2
        y += 14
        d.text(
            (x, y),
            "   ".join(
                f"{k} {100 * c / tot:.0f}%" for k, c in zip(DANGER_LEVELS, crit)
            ),
            font=self.f_small,
            fill=TEXT2,
        )
        y += 22

        # Weapon planner and match
        plan = t.get("plan")
        d.text((x, y), "WEAPON PLANNER  ·  EVERY 0.5 s", font=self.f_small, fill=HELP)
        y += 16
        if plan:
            p_ = plan["probs"].get(str(plan["slot"]), 0.0)
            d.text(
                (x, y),
                f"{WEAPON_NAMES[plan['slot']]} (slot {plan['slot']}, {100 * p_:.0f}%)"
                f"   holding {t['weapon']}",
                font=self.f_small,
                fill=TEXT2,
            )
        y += 22
        s = t["stats"]
        d.text((x, y), "MATCH VS 7 BOTS", font=self.f_small, fill=HELP)
        y += 16
        d.text(
            (x, y),
            f"hp {t['hp']}  frags {s['frags']}  deaths {s['deaths']}  "
            f"rank {s['rank']}  best bot {s['best_bot'][1]}",
            font=self.f_small,
            fill=TEXT2,
        )
        y += 24

        # What the model reads
        d.text((x, y), "MODEL INPUT", font=self.f_small, fill=HELP)
        y += 16
        words, line, lines = t["text"].split(" "), "", []
        for w in words:
            if len(line) + len(w) + 1 > 46:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        for ln in lines:
            if y > H - 40:
                break
            d.text((x, y), ln, font=self.f_mono, fill=TEXT2)
            y += 14
        d.text(
            (x, H - 22), f"tic {t['tick']}  ·  {t['gpu']}", font=self.f_small, fill=HELP
        )


def cmap(p: np.ndarray) -> np.ndarray:
    """Probabilities (any shape) -> RGB uint8 (shape + (3,))."""
    t = np.sqrt(np.clip(p, 0.0, 1.0))
    xs = [s[0] for s in STOPS]
    return np.stack(
        [np.interp(t, xs, [s[1][k] for s in STOPS]) for k in range(3)], axis=-1
    ).astype(np.uint8)


class Heatmap:
    """Per-tic action distribution: fills from the left, then slides."""

    def __init__(self, width: int):
        self.win = (width - HEAT_GUTTER - 18) // HEAT_PX
        # rows: behavior, the actions, danger, fresh-decision mark
        self.buf = np.empty((len(DISPLAY_ORDER) + 3, self.win, 3), np.uint8)
        self.buf[:] = BG
        self.count = 0
        self.f = font(10, mono=True)
        self.f_small = font(11)

    def push(
        self, probs: dict[str, float], adapter: str, decided: bool, critic: dict
    ) -> None:
        col = np.empty((self.buf.shape[0], 3), np.uint8)
        col[0] = COLORS.get(adapter, TEXT)
        col[1:-2] = cmap(np.array([probs.get(a, 0.0) for a in DISPLAY_ORDER]))
        pm, ph = critic.get("mid", 0.0), critic.get("high", 0.0)
        bg, am, rd = np.array(BG), np.array(AMBER), np.array(RED)
        col[-2] = np.clip(bg + pm * (am - bg) + ph * (rd - bg), 0, 255)
        col[-1] = TEXT if decided else BG
        if self.count < self.win:
            self.buf[:, self.count] = col
        else:
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = col
        self.count += 1

    def draw(self, img: Image.Image, y0: int) -> None:
        d = ImageDraw.Draw(img)
        n = len(DISPLAY_ORDER)
        d.text((18, y0 + 8), "ACTION PROBABILITIES", font=self.f_small, fill=HELP)
        d.text(
            (170, y0 + 8),
            "one column per tic, newest at the right  ·  top strip: active behavior  "
            "·  danger: critic P(mid) amber, P(high) red  ·  marks: fresh decisions",
            font=self.f_small,
            fill=HELP,
        )
        gap = np.empty((3, self.win, 3), np.uint8)
        gap[:] = BG
        rows = [np.repeat(self.buf[0:1], 6, axis=0), gap]
        rows += [
            np.repeat(self.buf[i : i + 1], HEAT_ROW, axis=0) for i in range(1, n + 1)
        ]
        rows += [gap, np.repeat(self.buf[n + 1 : n + 2], 8, axis=0)]
        rows += [gap, np.repeat(self.buf[n + 2 : n + 3], 4, axis=0)]
        arr = np.repeat(np.concatenate(rows, axis=0), HEAT_PX, axis=1)
        top = y0 + 28
        img.paste(Image.fromarray(arr), (HEAT_GUTTER, top))
        for i, a in enumerate(DISPLAY_ORDER):
            y = top + 9 + HEAT_ROW * i - 1
            d.text((18, y), SHORT_LABELS[a], font=self.f, fill=TEXT2)
        d.text((18, top + 9 + HEAT_ROW * n + 3), "danger", font=self.f, fill=HELP)
        yb = top + arr.shape[0] + 4
        d.text(
            (HEAT_GUTTER, yb),
            f"-{self.win / TIC_HZ:.0f} s",
            font=self.f_small,
            fill=HELP,
        )
        right = HEAT_GUTTER + self.win * HEAT_PX
        d.text((right - 22, yb), "now", font=self.f_small, fill=HELP)
        # Legend: p on a sqrt scale
        lx, lw = HEAT_GUTTER + 260, 240
        ramp = cmap((np.linspace(0, 1, lw) ** 2)[None, :].repeat(8, axis=0))
        img.paste(Image.fromarray(ramp), (lx, yb + 2))
        d.text((lx - 16, yb), "p", font=self.f_small, fill=HELP)
        for p in (0, 0.05, 0.25, 0.5, 1):
            px = lx + int(lw * p**0.5)
            d.text((px - 6, yb + 11), f"{p:g}", font=self.f_small, fill=HELP)


ACT_ROW = 12  # pixels per model row in the activity map
PURPLE = (165, 110, 255)
MODEL_COLORS = {
    "base": TEXT2,
    **COLORS,
    ARMS: BLUE,
    CRITIC: PURPLE,
    ROUTER: GREEN,
    NARRATOR: TEAL,
}
MODEL_LABELS = {
    "base": "base model",
    ARMS: "weapon plan",
    CRITIC: "critic",
    ROUTER: "router",
    NARRATOR: "narrator",
}


class Activity:
    """Which model runs on each tic: the base model and every adapter in the
    checkpoint, one row each, a column per tic, newest at the right; the labels
    light up while their model runs."""

    def __init__(self, width: int, models: list[str]):
        self.models = models
        self.win = (width - HEAT_GUTTER - 18) // HEAT_PX
        self.buf = np.empty((len(models), self.win, 3), np.uint8)
        self.buf[:] = BG
        self.count = 0
        self.now: set[str] = set()
        self.f = font(10, mono=True)
        self.f_small = font(11)

    @property
    def height(self) -> int:
        h = 52 + ACT_ROW * len(self.models)
        return h + h % 2  # the video codec wants an even frame height

    def push(self, active: set[str]) -> None:
        col = np.array(
            [MODEL_COLORS[m] if m in active else LAYER for m in self.models], np.uint8
        )
        if self.count < self.win:
            self.buf[:, self.count] = col
        else:
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = col
        self.count += 1
        self.now = active

    def draw(self, img: Image.Image, y0: int) -> None:
        d = ImageDraw.Draw(img)
        d.text((18, y0 + 8), "WHICH MODEL RUNS", font=self.f_small, fill=HELP)
        d.text(
            (170, y0 + 8),
            "one column per tic, newest at the right  ·  every adapter reads the base "
            "model's one shared KV cache (aLoRA)",
            font=self.f_small,
            fill=HELP,
        )
        rows = np.repeat(self.buf, ACT_ROW, axis=0)
        rows[ACT_ROW - 2 :: ACT_ROW] = BG  # a thin gap between rows
        arr = np.repeat(rows, HEAT_PX, axis=1)
        top = y0 + 28
        img.paste(Image.fromarray(arr), (HEAT_GUTTER, top))
        for i, m in enumerate(self.models):
            on = m in self.now
            y = top + ACT_ROW * i - 1
            if on:
                d.rectangle([10, y + 2, 14, y + 8], fill=MODEL_COLORS[m])
            d.text(
                (18, y),
                MODEL_LABELS.get(m, m),
                font=self.f,
                fill=MODEL_COLORS[m] if on else HELP,
            )


CAPTION_S = 5.0  # how long a spoken line stays on screen
KINDS = {"alora": "aLoRA", "lora": "LoRA", "sr": "Shadow Residual", "base": "no"}


def draw_captions(img: Image.Image, caps: list[tuple[float, str, str]], now: float):
    """The latest line from the watcher and from the player, over the game view."""
    f = font(16)
    shown = [
        (who, re.sub(r"\[[a-z ]+\]\s*", "", text))  # sound tags are voiced, not shown
        for t, who, text in caps
        if now - t < CAPTION_S
    ][-2:]
    if not shown:
        return
    rows = []
    for who, text in shown:
        words, line = f"{'YOU' if who == 'player' else 'GRANITE'}: {text}".split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 66:
                rows.append((who, line))
                line = w
            else:
                line = f"{line} {w}".strip()
        rows.append((who, line))
    d = ImageDraw.Draw(img)
    y = H - 14 - 22 * len(rows)
    d.rectangle([0, y - 8, W_GAME, H], fill=(10, 12, 16))
    for who, row in rows:
        d.text((14, y), row, font=f, fill=AMBER if who == "player" else TEAL)
        y += 22


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=("vllm", "expert"), default="vllm")
    ap.add_argument("--model", help="Composed checkpoint (for --policy vllm)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--segment",
        nargs=2,
        action="append",
        metavar=("ADAPTER_OR_INSTRUCTION", "SECONDS"),
        required=True,
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--temperature", type=float, default=1.0, help="Style and planner sampling"
    )
    ap.add_argument("--layout", default="log", help="The adapters' prompt layout")
    ap.add_argument("--persona", default="marine", help="How it talks: marine, crime")
    ap.add_argument("--no-hud", action="store_true", help="No status bar (as collect)")
    ap.add_argument("--no-critic", action="store_true", help="Do not ask the critic")
    ap.add_argument("--eager", action="store_true", help="No CUDA graphs")
    ap.add_argument("--cudagraph-mode", default="FULL")
    ap.add_argument("--fa2", action="store_true", help="FlashAttention 2, not 3")
    ap.add_argument(
        "--talk-every", type=float, default=0.0, help="Seconds between spoken lines"
    )
    ap.add_argument(
        "--player",
        action="append",
        default=[],
        metavar="SECONDS:TEXT",
        help="What the watcher says, and when (video seconds); repeatable",
    )
    ap.add_argument(
        "--sfx",
        action="store_true",
        help="Keep the game's sound (<out>.sfx.wav; voice_video.py mixes it in)",
    )
    args = ap.parse_args()

    import imageio.v2 as imageio

    if args.policy == "vllm":
        import torch

        where = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
        pol = make_policy(
            "vllm",
            args.model,
            warmup=100,
            temperature=args.temperature,
            layout=args.layout,
            persona=args.persona,
            enforce_eager=args.eager,
            cudagraph_mode=args.cudagraph_mode,
            attention={"flash_attn_version": 2} if args.fa2 else None,
        )
    else:
        where = "scripted teacher, no model"
        pol = make_policy("expert")
    env = DoomEnv(
        seed=args.seed,
        resolution="640X480",
        hud=not args.no_hud,
        timeout_tics=10**7,
        audio=args.sfx,
    )
    sfx: list[np.ndarray] = []  # the game's sound, one tic per frame
    silent_tic = np.zeros((AUDIO_HZ // TIC_HZ, 2), np.int16)
    obs = env.reset(seed=args.seed)
    hist = History(getattr(pol, "tok", None))
    panel = Panel()
    heat = Heatmap(W_GAME + W_PANEL)
    # Who writes spoken lines: the narrator adapter, or the base model without one.
    talker = getattr(pol, "talker", None) or "base"
    models = ["base", *BEHAVIORS, ARMS, CRITIC, ROUTER]
    act = Activity(W_GAME + W_PANEL, models + ([NARRATOR] if talker != "base" else []))
    talk_until, routed = -1, False  # the frame the line writer runs through
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.out), fps=TIC_HZ, codec="libx264", quality=8, macro_block_size=1
    )
    lat: deque[float] = deque(maxlen=1000)
    last_second: deque[float] = deque(maxlen=TIC_HZ)
    all_ms = []
    script = sorted(
        (float(t), text) for t, text in (x.split(":", 1) for x in args.player)
    )
    caps: list[tuple[float, str, str]] = []  # (video s, who, text)
    frame_i, last_talk, last_reply = 0, -1e9, -1e9
    tally = [0, 0]  # frags, deaths this match: the brief's score
    tag_rng = random.Random(args.seed)
    live, fist = 0, 0  # tics alive, and of those holding the fist
    kind = KINDS.get(getattr(pol, "placement", "base"), "")
    for what, secs in args.segment:
        info: dict = {"instruction": None}
        if what in BEHAVIORS:
            adapter = what
        else:
            r = pol.route(what)
            adapter = r.adapter
            routed = True
            info = {
                "instruction": what,
                "routed": r.adapter,
                "route_p": r.prob,
                "route_ms": r.ms,
            }
        print(f"segment: {what!r} -> {adapter} for {secs}s", flush=True)
        action, critic, plan = "wait", {}, None
        for _ in range(int(float(secs) * TIC_HZ)):
            weapon = None
            active: set[str] = {ROUTER} if routed else set()  # models run this tic
            routed = False
            if obs.dead:  # respawning: nothing to decide
                heat.push({}, adapter, False, critic)
            else:
                want = [adapter] if args.no_critic else [adapter, CRITIC]
                if obs.tick % PLAN_EVERY_TICS == 0:
                    want.append(ARMS)
                # The base model prefills the new state; each adapter reads its KV.
                active |= {"base", *want}
                decs = pol.decide_many(obs, tuple(want), hist)
                d = decs[adapter]
                action = d.action
                critic = decs[CRITIC].probs if CRITIC in decs else {}
                if ARMS in decs:
                    weapon = int(decs[ARMS].action)
                    plan = {"slot": weapon, "probs": decs[ARMS].probs}
                heat.push(d.probs, adapter, True, critic)
                last_second.append(d.ms)
                lat.append(d.ms)
                all_ms.append(d.ms)
                now = frame_i / TIC_HZ
                player = script.pop(0)[1] if script and script[0][0] <= now else None
                # Every --talk-every seconds, or soon after a frag, a death or a
                # new weapon, as the engine (and the narrator's data) does.
                salient = any(e in SALIENT_EVENTS for e in obs.events)
                narrate = (
                    args.talk_every
                    and now - last_reply >= 3.0
                    and (
                        now - last_talk >= args.talk_every
                        or (salient and now - last_talk >= 2.5)
                    )
                )
                if hasattr(pol, "talk") and (player or narrate):
                    state = state_text(obs)
                    b = brief(
                        hist.entries,
                        state,
                        tuple(tally) if args.layout == "chat" else None,
                    )
                    mine = [
                        re.sub(r"\[[^\]]*\]\s*", "", x)
                        for _, w, x in caps
                        if w == "bot"
                    ][-3:]
                    last = " / ".join(f'"{x}"' for x in mine) or None
                    t_talk = time.perf_counter()
                    line = pol.talk([(hist.ids, state)], [b], [player], [last])[0]
                    # Shown as running for as many tics as writing the line took.
                    talk_ms = (time.perf_counter() - t_talk) * 1000
                    talk_until = frame_i + max(1, math.ceil(talk_ms / TIC_MS)) - 1
                    voiced = (
                        sound_tag(b, tag_rng) + line
                    )  # tags: voiced, not in history
                    if args.layout == "chat":
                        hist.append(spoken_entry(state, line, talk_extra(b, player)))
                    else:
                        if player:
                            hist.append(said_entry(obs.tick, "user", player))
                        hist.append(said_entry(obs.tick, "me", line))
                    if player:
                        caps.append((now, "player", player))
                        last_reply = now
                    caps.append((now, "bot", voiced))
                    last_talk = now
                    print(
                        f"t{now:5.1f}  {'[' + player + '] ' if player else ''}-> {line}"
                    )
            if frame_i <= talk_until:
                active.add(talker)
            act.push(active)
            frame = env.frame()
            text, hud = obs.text, (obs.hp, obs.armor, obs.weapon)
            if not obs.dead:
                live += 1
                fist += obs.weapon == "fist"
            hist.observe(obs, action)
            obs = env.step(action, weapon=weapon)
            tally[0] += obs.events.count("frag")
            tally[1] += obs.events.count("died")
            if obs.done:
                obs = env.reset()
                hist.reset()
                tally = [0, 0]
            img = Image.new("RGB", (W_GAME + W_PANEL, H + act.height + H_HEAT), BG)
            img.paste(Image.fromarray(frame), (0, 0))
            a = np.fromiter(lat, dtype=np.float64)
            panel.draw(
                img,
                {
                    **info,
                    "adapter": adapter,
                    "kind": kind,
                    "ms": float(np.median(last_second)),
                    "p50": float(np.percentile(a, 50)),
                    "p99": float(np.percentile(a, 99)),
                    "action": action,
                    "critic": critic,
                    "plan": plan,
                    "stats": env.stats.as_dict(),
                    "hp": hud[0],
                    "armor": hud[1],
                    "weapon": hud[2],
                    "text": text,
                    "tick": obs.tick,
                    "gpu": where,
                },
            )
            draw_captions(img, caps, frame_i / TIC_HZ)
            act.draw(img, H)
            heat.draw(img, H + act.height)
            writer.append_data(np.asarray(img))
            if args.sfx:  # this frame's tic of game sound
                sound = env.audio() if env.audio() is not None else silent_tic
                sfx.append(np.asarray(sound, np.int16).reshape(-1, 2))
            frame_i += 1
    writer.close()
    env.close()
    if args.sfx:
        import wave

        with wave.open(str(args.out.with_suffix(".sfx.wav")), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(AUDIO_HZ)
            w.writeframes(np.concatenate(sfx).tobytes())
    if caps:
        talk = [{"t": round(t, 3), "who": who, "text": text} for t, who, text in caps]
        args.out.with_suffix(".talk.json").write_text(
            json.dumps({"fps": TIC_HZ, "frames": frame_i, "lines": talk}, indent=1)
        )
    a = np.asarray(all_ms)
    st = env.stats.as_dict()
    print(
        f"match: frags {st['frags']} deaths {st['deaths']} rank {st['rank']}; "
        f"holding the fist {100 * fist / max(1, live):.0f}% of live tics"
    )
    print(
        f"wrote {args.out}: {a.size} decisions, p50 {np.percentile(a, 50):.2f} ms, "
        f"p99 {np.percentile(a, 99):.2f} ms, max {a.max():.2f} ms"
    )


if __name__ == "__main__":
    main()
