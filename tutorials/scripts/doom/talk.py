# SPDX-License-Identifier: Apache-2.0
"""The player's own voice: one short spoken line from the base model.

A talk request reads the same prompt every game adapter reads, up to the end of
the state (so it reuses their prefilled KV), then adds one user turn: a brief
of the last few seconds in plain words, and the instruction to speak. The brief
exists because the history is terse (``t12.4 hp 64 face 135 | bot +10 8m | did
cl | frag``): a base model reads a medikit on screen well, but misses a frag
three entries back unless it is told.

The line it speaks goes back into the history as a ``me:`` entry
(:func:`history.said_entry`), so every adapter's next prompt contains it.
"""

from __future__ import annotations

import re

from doom_env import WEAPON_NAMES

BRIEF_ENTRIES = 15  # the last 3 s of history
# Ammo below which a weapon is nearly empty (the BFG spends 40 cells a shot).
LOW_AMMO = {2: 20, 3: 4, 4: 20, 5: 3, 6: 20, 7: 40}
LOW_HP = 35

_ARMS = re.compile(r"\barms ((?:\d:\d+ ?)+)")
_HELD = re.compile(r"\| (\w+) (\d+) \| arms")
_NUM = {k: re.compile(rf"\b{k} (-?\d+)") for k in ("hp", "armor", "hit")}
_FOES = re.compile(r"\bbot ([+-]\d+) (\d+)m")


def _count(n: int, what: str) -> str:
    return f"one {what}" if n == 1 else f"{n} {what}s"


def score(entries: list[str]) -> tuple[int, int]:
    """Frags and deaths in these entries (a whole match's: the live history
    keeps only the last 10 s, so the engine counts its own)."""
    text = "".join(entries)
    return len(re.findall(r"\| frag\b", text)), text.count("| died")


def brief(entries: list[str], state: str, tally: tuple[int, int] | None = None) -> str:
    """What just happened and where things stand, as short plain sentences;
    with ``tally`` (frags, deaths so far), the score too."""
    recent = "".join(entries[-BRIEF_ENTRIES:])
    news = []
    frags = len(re.findall(r"\| frag\b", recent))
    if frags:
        news.append(f"you fragged {_count(frags, 'bot') if frags > 1 else 'a bot'}")
    if "| died" in recent:
        news.append("you got killed")
    if "| respawn" in recent:
        news.append("you respawned with just a pistol")
    got = re.findall(r"\| got (\w+)", recent)
    for k in dict.fromkeys(got):
        news.append({"weapon": "you picked up a new weapon"}.get(k, f"you grabbed {k}"))
    nums = {k: int(m.group(1)) for k, r in _NUM.items() if (m := r.search(state))}
    if nums.get("hit", 0) > 0:
        news.append(f"you took {nums['hit']} damage in the last second")

    now = []
    hp = nums.get("hp")
    if hp is not None:
        now.append(f"health {hp}" + (" (low)" if hp < LOW_HP else ""))
    held = _HELD.search(state)
    arms = _ARMS.search(state)
    owned = {}
    if arms:
        for part in arms.group(1).split():
            slot, ammo = part.split(":")
            owned[int(slot)] = int(ammo)
    if held:
        now.append(f"holding the {held.group(1)} with {held.group(2)} ammo")
    guns = {s: a for s, a in owned.items() if s in LOW_AMMO}
    if guns and all(a < LOW_AMMO[s] for s, a in guns.items()):
        now.append("low on ammo for every gun")
    best = [
        WEAPON_NAMES[s] for s in sorted(guns, reverse=True) if guns[s] >= LOW_AMMO[s]
    ]
    if best and held and held.group(1) != best[0]:
        now.append(f"your best loaded gun is the {best[0]}")
    foes = _FOES.findall(state.split("| see", 1)[-1]) if "| see" in state else []
    if foes:
        b, d = int(foes[0][0]), int(foes[0][1])
        side = "ahead" if abs(b) <= 20 else ("to the left" if b < 0 else "to the right")
        now.append(f"{_count(len(foes), 'bot')} in view, nearest {d} m {side}")
    else:
        now.append("no bot in view")
    parts = []
    if news:
        parts.append("Just now: " + "; ".join(news) + ".")
    parts.append("Right now: " + "; ".join(now) + ".")
    if tally is not None:
        f, d = tally
        parts.append(f"Score so far: {_count(f, 'frag')}, {_count(d, 'death')}.")
    return " ".join(parts)


def sound_tag(brief_text: str, rng) -> str:
    """A sound for Chatterbox-Turbo to voice before a line, chosen from what just
    happened. Left to the model, one opened every line."""
    if "you got killed" in brief_text:
        return "[sigh] " if rng.random() < 0.6 else ""
    if "you fragged" in brief_text:
        return "[chuckle] " if rng.random() < 0.35 else ""
    if "(low)" in brief_text:
        return "[sigh] " if rng.random() < 0.3 else ""
    return ""


# Events the player speaks soon after (engine, record_video), and the same in
# history-entry text.
SALIENT_EVENTS = ("frag", "died", "got weapon")
SALIENT = tuple(f"| {e}" for e in SALIENT_EVENTS)


def match_moments(
    entries: list[str],
    rows: list[dict],
    every_s: float = 4.0,
    gap_s: float = 2.5,
    limit: int = 60,
    with_score: bool = False,
) -> list[dict]:
    """The moments a live game would speak at, in order: every ``every_s``
    seconds, or right after a frag, a death or a new weapon (at least ``gap_s``
    apart), as the engine does. ``rows`` are one match's collect.py rows (tick,
    history length, state); each moment carries its brief (with the score so
    far if ``with_score``) and last log lines."""
    from doom_env import TIC_HZ

    out, last = [], -1e9
    for r in sorted(rows, key=lambda r: r["t"]):
        now, n = r["t"] / TIC_HZ, r["hist_n"]
        if n < 5 or now - last < gap_s:
            continue
        salient = any(k in "".join(entries[max(0, n - 3) : n]) for k in SALIENT)
        if not (now - last >= every_s or salient):
            continue
        out.append(
            {
                "t": r["t"],
                "hist_n": n,
                "state": r["state"],
                "brief": brief(
                    entries[:n], r["state"], score(entries[:n]) if with_score else None
                ),
                "recent": [e.strip() for e in entries[max(0, n - 3) : n]],
            }
        )
        last = now
        if len(out) >= limit:
            break
    return out


def main() -> None:
    import argparse
    import json
    import random
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Spoken-line tools for the Doom demo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("pool", help="Lines from the base model (augmentation pool)")
    pl.add_argument("--model", required=True, help="Any composed checkpoint")
    pl.add_argument("--data", type=Path, required=True, help="A collect.py dir")
    pl.add_argument("--n", type=int, default=3000)
    pl.add_argument("--out", type=Path, required=True)
    mo = sub.add_parser("moments", help="Speaking moments of recorded matches")
    mo.add_argument(
        "--data", type=Path, nargs="+", required=True, help="collect.py dirs"
    )
    mo.add_argument("--style", default="fighter")
    mo.add_argument("--matches", type=int, default=400)
    mo.add_argument("--per-match", type=int, default=60)
    mo.add_argument("--every-s", type=float, default=4.0, help="Talk cadence")
    mo.add_argument("--score", action="store_true", help="Score so far in the brief")
    mo.add_argument("--seed", type=int, default=0)
    mo.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "moments":
        matches = []
        for d in args.data:
            rows_by: dict = {}
            for line in open(d / f"{args.style}.jsonl"):
                r = json.loads(line)
                rows_by.setdefault(r["ep"], []).append(r)
            for line in open(d / f"{args.style}_history.jsonl"):
                h = json.loads(line)
                if h["ep"] in rows_by:
                    matches.append((str(d), h["ep"], h["entries"], rows_by[h["ep"]]))
        random.Random(args.seed).shuffle(matches)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(args.out, "w") as f:
            for d, ep, entries, rows in matches[: args.matches]:
                ms = match_moments(
                    entries,
                    rows,
                    every_s=args.every_s,
                    limit=args.per_match,
                    with_score=args.score,
                )
                f.write(
                    json.dumps(
                        {"data": d, "style": args.style, "ep": ep, "moments": ms}
                    )
                    + "\n"
                )
                n += len(ms)
        print(f"{min(len(matches), args.matches)} matches, {n} moments -> {args.out}")
        return

    from history import History
    from policy import VLLMPolicy

    streams = {}
    for line in open(args.data / "fighter_history.jsonl"):
        h = json.loads(line)
        streams[h["ep"]] = h["entries"]
    rows = [json.loads(x) for x in open(args.data / "fighter.jsonl")]
    rows = [r for r in rows if r["hist_n"] >= 10]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    pol = VLLMPolicy(args.model, max_num_seqs=128, warmup=2)
    games, briefs = [], []
    for r in rows:
        entries = streams[r["ep"]][: r["hist_n"]]
        games.append((History.replay(entries, pol.tok).ids, r["state"]))
        briefs.append(brief(entries, r["state"]))
    lines = []
    for i in range(0, len(games), 256):
        lines += pol.talk(games[i : i + 256], briefs[i : i + 256], temperature=0.9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r, b, t in zip(rows, briefs, lines):
            if t.strip():
                f.write(
                    json.dumps({"ep": r["ep"], "t": r["t"], "brief": b, "line": t})
                    + "\n"
                )
    print(f"{sum(1 for t in lines if t.strip())} lines -> {args.out}")


if __name__ == "__main__":
    main()
