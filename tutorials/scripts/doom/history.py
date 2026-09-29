# SPDX-License-Identifier: Apache-2.0
"""Compact game history for the prompt: 5 Hz entries, append-only token ids.

Every adapter reads the same prompt up to its query suffix::

    [system] [history: one entry per 0.2 s, last <= 10 s] [now + current state] <|adapter|>...

The history is base-model text that only grows, so from one tic to the next the
engine's prefix cache holds all of it and the only fresh tokens are the current
state and the suffix. Entries therefore carry absolute match time (``t12.4``,
seconds) rather than an age that would change every tic; the current-state line
starts with ``now t14.2`` so the model can tell how old each entry is.

An entry summarizes the 0.2 s that just ended::

    t12.4 hp 64 face 135 | bot +10 8m, bot -40 20m | did cl | frag

``face`` is the player's heading in whole degrees (the player knows how far it
has turned), so bearings in older entries can be related to the current view.
``did`` is the most frequent action of the span, events (frag, died, respawn,
got health, ...) are appended.

**Window.** The history grows to ``window_s`` seconds, then the oldest half is
dropped. That one tic changes the prompt from the first history token on, so the
engine prefills the kept half again (about 600 tokens); every other tic is
append-only. :attr:`History.resets` counts these.

Each entry is tokenized on its own and the ids are concatenated. Entries start
right after a newline and end with one, so this equals tokenizing the joined
text (``python history.py`` checks it on real play).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from doom_env import TIC_HZ, Observation

ENTRY_TICS = 7  # one entry per 0.2 s: 5 Hz
WINDOW_S = 10.0


def _t(tick: int) -> str:
    return f"t{tick / TIC_HZ:.1f}"


def now_prefix(tick: int) -> str:
    """Start of the current-state line, which the prompt puts after the history."""
    return f"now {_t(tick)} | "


def entry_text(tick: int, obs: Observation, actions: list[str], events: list[str]):
    """One history line for the span ending at ``tick``."""
    ev = "".join(f" | {e}" for e in events)
    if obs.dead:
        return f"{_t(tick)} dead{ev}\n"
    foes = [o for o in obs.seen if o.kind == "enemy"][:2]
    see = ", ".join(f"{o.label} {o.b:+d} {o.d}m" for o in foes) or "clear"
    counts = Counter(actions)
    did = max(reversed(actions), key=counts.__getitem__) if actions else "wait"
    return f"{_t(tick)} hp {obs.hp} face {obs.face} | {see} | did {did}{ev}\n"


def said_entry(tick: int, who: str, text: str) -> str:
    """A spoken line as a history entry: ``me`` (the player's own talk) or
    ``user``. Newlines and quotes are dropped so it stays one entry."""
    line = " ".join(text.replace('"', "").split())
    return f'{_t(tick)} {who}: "{line}"\n'


@dataclass
class History:
    """Per-game history. Feed every tic with :meth:`observe`; read :attr:`ids`.

    ``tokenizer=None`` keeps text only (for collection on machines without the
    tokenizer); :meth:`replay` rebuilds the same history from stored entries.
    """

    tokenizer: object | None = None
    entry_tics: int = ENTRY_TICS
    window_s: float = WINDOW_S
    entries: list[str] = field(default_factory=list)
    entry_ids: list[list[int]] = field(default_factory=list)
    resets: int = 0

    def __post_init__(self) -> None:
        self.max_entries = int(round(self.window_s * TIC_HZ / self.entry_tics))
        self.reset()

    def reset(self) -> None:
        self.entries, self.entry_ids = [], []
        self._actions: list[str] = []
        self._events: list[str] = []
        self._ids: list[int] = []
        self.resets = 0

    @property
    def ids(self) -> list[int]:
        return self._ids

    @property
    def text(self) -> str:
        return "".join(self.entries)

    def observe(self, obs: Observation, action: str | None) -> str | None:
        """Record this tic (the state and the action taken on it).

        Returns the entry text when one was appended this tic, else None.
        """
        if action is not None and not obs.dead:
            self._actions.append(action)
        self._events.extend(obs.events)
        if obs.tick == 0 or obs.tick % self.entry_tics:
            return None
        text = entry_text(obs.tick, obs, self._actions, self._events)
        self._actions, self._events = [], []
        self.append(text)
        return text

    def append(self, text: str) -> bool:
        """Append one entry; returns True if the window was reset first."""
        reset = len(self.entries) >= self.max_entries
        if reset:
            keep = self.max_entries // 2
            self.entries = self.entries[-keep:]
            self.entry_ids = self.entry_ids[-keep:]
            self.resets += 1
        self.entries.append(text)
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            self.entry_ids.append(ids)
            if reset:
                self._ids = [i for e in self.entry_ids for i in e]
            else:
                self._ids = self._ids + ids  # a new list: callers may hold the old
        return reset

    @classmethod
    def replay(cls, entries: list[str], tokenizer=None, **kw) -> History:
        h = cls(tokenizer, **kw)
        for e in entries:
            h.append(e)
        return h


# ── Labels that only the history can answer, or that come from the future ──────
PROBE_WORDS: tuple[str, ...] = ("forward", "left", "right", "back", "wait")
_SIGHT = re.compile(r"face (\d+) \| bot ([+-]\d+) \d+m")


def probe_label(hist: History, obs: Observation) -> str | None:
    """Where, relative to the current heading, was the last enemy in the history?

    Asked only when no enemy is on screen and the state line's own enemy memory
    (5 s) has expired, so the answer is in the history and nowhere else:
    ``forward`` / ``left`` / ``right`` / ``back`` (90-degree sectors), or
    ``wait`` if no enemy appears in the last 10 s.
    """
    if obs.dead or obs.enemy_mem is not None:
        return None
    if any(o.kind == "enemy" for o in obs.seen):
        return None
    for e in reversed(hist.entries):
        m = _SIGHT.search(e)
        if m:
            face, bearing = int(m.group(1)), int(m.group(2))
            rel = (face - bearing - obs.face + 180) % 360 - 180  # + is left
            if abs(rel) <= 45:
                return "forward"
            if 45 < rel <= 135:
                return "left"
            if -135 <= rel < -45:
                return "right"
            return "back"
    return "wait"


CRITIC_TICS = TIC_HZ  # the critic predicts the next second
CRITIC_HIGH_DAMAGE = 30


def critic_labels(taken: list[float], deaths: list[float], ticks: list[int]):
    """Outcome labels for the critic at each of ``ticks``: ``high`` if the player
    dies or takes >= 30 damage within the next second, ``mid`` for any damage,
    else ``low``. ``taken`` and ``deaths`` are the per-tic cumulative counters."""
    out = []
    last = len(taken) - 1
    for t in ticks:
        u = min(last, t + CRITIC_TICS)
        dmg, died = taken[u] - taken[t], deaths[u] > deaths[t]
        out.append(
            "high" if died or dmg >= CRITIC_HIGH_DAMAGE else "mid" if dmg > 0 else "low"
        )
    return out


def main() -> None:
    """Check per-entry tokenization == joined-text tokenization on real play."""
    import argparse

    from doom_env import DoomEnv
    from expert import BEHAVIORS, PLAN_EVERY_TICS, Expert
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--tokenizer", default="ibm-granite/granite-4.1-3b")
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    env = DoomEnv(seed=5, timeout_tics=int(args.seconds * TIC_HZ))
    ex, hist = Expert(), History(tok)
    obs = env.reset(seed=5)
    n_checked, lengths = 0, []
    while not obs.done:
        a = ex.act(obs, BEHAVIORS[0])
        w = ex.weapon(obs) if obs.tick % PLAN_EVERY_TICS == 0 else None
        if hist.observe(obs, None if obs.dead else a) is not None:
            ref = tok.encode(hist.text, add_special_tokens=False)
            assert hist.ids == ref, f"history ids differ at tick {obs.tick}"
            n_checked += 1
            lengths.append(len(hist.ids))
        obs = env.step(a, weapon=w)
    env.close()
    per_entry = [len(e) for e in hist.entry_ids]
    print(hist.text[-600:])
    print(
        f"OK: {n_checked} history states match joined-text tokenization; "
        f"{hist.resets} window resets; history {min(lengths)}-{max(lengths)} tokens; "
        f"entry {min(per_entry)}-{max(per_entry)} tokens "
        f"(mean {sum(per_entry) / len(per_entry):.1f})"
    )


if __name__ == "__main__":
    main()
