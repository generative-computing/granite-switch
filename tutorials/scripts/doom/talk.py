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


def brief(entries: list[str], state: str) -> str:
    """What just happened and where things stand, as short plain sentences."""
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
    return " ".join(parts)


def main() -> None:
    """Generate a pool of spoken lines from recorded moments, for training the
    game adapters on histories that contain talk (train_alora.py --talk-lines)."""
    import argparse
    import json
    import random
    from pathlib import Path

    from history import History
    from policy import VLLMPolicy

    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    ap.add_argument(
        "--model", required=True, help="Any composed checkpoint (base path)"
    )
    ap.add_argument("--data", type=Path, required=True, help="A collect.py dir")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

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
