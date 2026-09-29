# SPDX-License-Identifier: Apache-2.0
"""Play deathmatches against bots in parallel ViZDoom envs; write jsonl + stats.

A **teacher** labels every visited state: the scripted player (``--teacher
expert``, one-hot labels) or an RL checkpoint (``--teacher runs/rl0/latest.pt``,
soft labels: its whole action distribution). A **driver** chooses the actions:

* ``--policy teacher``: the teacher itself, sampling from its distribution
  (the scripted one adds DART-style noise bursts with ``--dart``)::

      python collect.py --policy teacher --teacher runs/rl0/latest.pt \
          --episodes 40 --timeout-s 120 --out data/d0

* ``--policy vllm``: the student (a composed checkpoint) drives; the teacher
  labels what the student visits (DAgger). All envs step in lockstep, so each
  tic's student decisions (the style adapter every tic, the weapon planner on
  its cadence) are one batched vLLM call::

      python collect.py --policy vllm --model models/doom-d0 \
          --teacher runs/rl0/latest.pt --episodes 30 --out data/d1

* ``--policy random``: the lockstep path with uniform random actions (a stat
  baseline, and a plumbing check without a GPU).

``--stats-only`` with 10-minute matches is the evaluation (``--timeout-s 600``,
the default). Tics while the player is dead need no decision and produce no row.

The takeover test: ``--takeover-s 30`` hands the game from the student to the
teacher after 30 s; ``--mark-s 30`` records frags, deaths and damage at 30 s in
each stats row, so the teacher's play after a student prefix can be compared
with its play after its own first 30 s (``--policy teacher --mark-s 30``). If
the teacher plays as well from states the student leads it into, its labels on
those states can be trusted.

Outputs in ``--out``:

* ``<style>.jsonl``: one row per decided tic: ``state`` (the prompt's current
  state), ``hist_n`` (history entries so far), ``expert`` and ``soft`` (the
  teacher's move label and distribution), ``weapon`` / ``weapon_soft`` on
  planner tics, ``critic`` (outcome in the next second), ``probe`` (history-only
  question, or null), ``act`` (executed).
* ``<style>_history.jsonl``: per match, every history entry in order, so
  training rebuilds each row's windowed history (:class:`history.History`).
* ``stats.jsonl``: one row per match; ``summary.txt``.
* ``videos/<style>_ep<k>.mp4``: the first ``--record`` matches per style.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import (
    ACTIONS,
    BOT_SETS,
    MATCH_TICS,
    RESPAWN_S,
    TIC_HZ,
    WEAPON_SLOTS,
    DoomEnv,
    isolate_workdir,
)
from expert import BEHAVIORS, PLAN_EVERY_TICS, Expert
from history import History, critic_labels, probe_label

RESOLUTION = "640X480"  # one setting everywhere: collection, bench and demo


class _Noise:
    """DART-style noise: bursts of random actions, labels stay the teacher's."""

    def __init__(self, rate: float, rng: random.Random, max_len: int = 6):
        self.rate, self.rng, self.max_len = rate, rng, max_len
        self.left, self.action = 0, "wait"

    def __call__(self, action: str) -> str:
        if self.left == 0 and self.rate > 0 and self.rng.random() < self.rate / 3:
            self.left = self.rng.randint(1, self.max_len)
            self.action = self.rng.choice(ACTIONS)
        if self.left > 0:
            self.left -= 1
            return self.action
        return action


class Teacher:
    """Labels for one game. ``label`` must be called on every live tic (the RL
    teacher's GRU state follows the game)."""

    def __init__(self, spec: str):
        self.soft = spec != "expert"
        if self.soft:
            from rl_teacher import KEEP, RLPolicy

            self.rl, self.keep = RLPolicy(spec), KEEP
        else:
            self.expert = Expert()

    def reset(self) -> None:
        (self.rl if self.soft else self.expert).reset()

    def label(self, obs, style: str) -> tuple[dict[str, float], dict[str, float]]:
        """(move distribution over ACTIONS, weapon distribution over slot digits)."""
        if not self.soft:
            return (
                {self.expert.act(obs, style): 1.0},
                {str(self.expert.weapon(obs, style)): 1.0},
            )
        pm, pw = self.rl.step(obs, style)
        move = {a: float(p) for a, p in zip(ACTIONS, pm)}
        weapon: dict[str, float] = {}
        for i, p in enumerate(pw):
            slot = obs.slot if i == self.keep else WEAPON_SLOTS[i]
            weapon[str(slot)] = weapon.get(str(slot), 0.0) + float(p)
        return move, weapon


def _argmax(d: dict[str, float]) -> str:
    return max(d, key=d.get)


def _sample(d: dict[str, float], rng: random.Random) -> str:
    return rng.choices(list(d), weights=list(d.values()))[0]


def _round(d: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 3) for k, v in d.items() if v >= 0.0005}


def _writer(path: Path):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path), fps=TIC_HZ, codec="libx264", quality=7, macro_block_size=1
    )


def _make_env(task: dict) -> DoomEnv:
    return DoomEnv(
        seed=task["seed"],
        resolution=RESOLUTION,
        hud=True,
        timeout_tics=task["timeout"],
        bots=task["bots"],
        n_bots=task["n_bots"],
        respawn_s=task.get("respawn_s", RESPAWN_S),
        item_rules=task.get("item_rules", "standard"),
    )


class _Match:
    """One match: the env, the teacher, the history and the rows being written."""

    def __init__(self, task: dict):
        self.task, self.style = task, task["behavior"]
        self.env = _make_env(task)
        self.teacher = Teacher(task["teacher"])
        self.rng = random.Random(task["seed"])
        self.noise = _Noise(task["dart"], self.rng)
        self.hist = History()
        self.hist_all: list[str] = []  # every entry, unwindowed, for the stream file
        self.rows: list[dict] = []
        self.taken: list[float] = []
        self.deaths: list[float] = []
        self.video = _writer(Path(task["video"])) if task["video"] else None
        self.obs = self.env.reset(seed=task["seed"])
        self.teacher.reset()
        self.agree = self.decided = 0
        self.mark_tick = task.get("mark")
        self.mark: dict | None = None

    def labels(self):
        """Teacher labels for the current live tic, plus the row skeleton."""
        obs = self.obs
        move, weapon = self.teacher.label(obs, self.style)
        plan = obs.tick % PLAN_EVERY_TICS == 0
        row = {
            "b": self.style,
            "ep": self.task["ep"],
            "t": obs.tick,
            "state": None,  # filled by the caller from policy.state_text
            "hist_n": len(self.hist_all),
            "expert": _argmax(move),
            "soft": _round(move) if self.teacher.soft else None,
            "weapon": _argmax(weapon) if plan else None,
            "weapon_soft": _round(weapon) if plan and self.teacher.soft else None,
            "probe": probe_label(self.hist, obs),
        }
        return move, weapon, plan, row

    def record(self, frame: bool = True) -> None:
        self.taken.append(self.obs.counters["taken"])
        self.deaths.append(self.obs.counters["deaths"])
        if self.mark_tick is not None and self.obs.tick == self.mark_tick:
            s = self.env.stats
            self.mark = {
                "frags": s.frags,
                "deaths": s.deaths,
                "dealt": s.damage_dealt,
                "best_bot": s.best_bot[1],
            }
        if self.video is not None and frame:
            self.video.append_data(self.env.frame())

    def advance(self, action: str, weapon: int | None, row: dict | None) -> None:
        """Apply ``action`` for this tic; history and bookkeeping follow."""
        if row is not None:
            row["act"] = action
            every = self.task.get("row_every", 1)
            # Keep every k-th tic, and every planner tic (the arms adapter's rows).
            if self.task["keep_rows"] and (
                row["t"] % every == 0 or row["weapon"] is not None
            ):
                self.rows.append(row)
        entry = self.hist.observe(self.obs, None if self.obs.dead else action)
        if entry is not None:
            self.hist_all.append(entry)
        self.obs = self.env.step(action, weapon=weapon)

    def finish(self, **extra) -> dict:
        if self.video is not None:
            self.video.close()
        self.env.close()
        ticks = [r["t"] for r in self.rows]
        for r, c in zip(self.rows, critic_labels(self.taken, self.deaths, ticks)):
            r["critic"] = c
        stats = {
            "behavior": self.style,
            "ep": self.task["ep"],
            "seed": self.task["seed"],
            "bots": self.task["bots"],
            "n_bots": self.task["n_bots"],
            "policy": self.task["policy"],
            "teacher": self.task["teacher"],
            **extra,
            **self.env.stats.as_dict(),
        }
        if self.mark is not None:
            s = self.env.stats
            stats["mark"] = self.mark
            stats["after_mark"] = {
                "frags": s.frags - self.mark["frags"],
                "deaths": s.deaths - self.mark["deaths"],
                "dealt": s.damage_dealt - self.mark["dealt"],
                "best_bot": s.best_bot[1] - self.mark["best_bot"],
            }
        history = {"b": self.style, "ep": self.task["ep"], "entries": self.hist_all}
        return {"rows": self.rows, "history": history, "stats": stats}


# ── Teacher-driven path: each worker plays whole matches on its own ─────────────
def _teacher_episode(task: dict) -> dict:
    from policy import state_text

    isolate_workdir()
    m = _Match(task)
    while not m.obs.done:
        m.record()
        if m.obs.dead:
            m.advance("wait", None, None)
            continue
        move, weapon, plan, row = m.labels()
        row["state"] = state_text(m.obs)
        if m.teacher.soft:
            act = _sample(move, m.rng)
            slot = int(_sample(weapon, m.rng)) if plan else None
        else:
            act = m.noise(row["expert"])
            slot = int(row["weapon"]) if plan else None
        m.advance(act, slot, row)
    return m.finish()


# ── Lockstep path: the main process decides for every env each tic ──────────────
def _lockstep_worker(conn, tasks: list[dict]) -> None:
    from policy import state_text

    isolate_workdir()
    for task in tasks:
        m = _Match(task)
        conn.send(("begin",))
        sent = 0  # history entries already sent to the main process
        while not m.obs.done:
            m.record()
            if m.obs.dead:
                m.advance("wait", None, None)
                continue
            move, weapon, plan, row = m.labels()
            row["state"] = state_text(m.obs)
            takeover = task.get("takeover")
            if takeover is not None and m.obs.tick >= takeover:
                # The teacher plays from here, as it does on its own: sampling.
                act = _sample(move, m.rng) if m.teacher.soft else row["expert"]
                slot = None
                if plan:
                    slot = int(
                        _sample(weapon, m.rng) if m.teacher.soft else row["weapon"]
                    )
                row["student"] = None
                m.advance(act, slot, row)
                continue
            new = m.hist_all[sent:]
            sent = len(m.hist_all)
            conn.send(("state", row["state"], new, m.style, plan))
            student, student_slot = conn.recv()
            use_teacher = m.rng.random() < task["beta"]
            act = row["expert"] if use_teacher else student
            slot = None
            if plan:
                slot = (
                    int(row["weapon"])
                    if use_teacher or student_slot is None
                    else student_slot
                )
            m.agree += student == row["expert"]
            m.decided += 1
            row["student"] = student
            m.advance(act, slot, row)
        res = m.finish(agreement=round(m.agree / max(1, m.decided), 4))
        conn.send(("episode", res))
    conn.send(("done",))


def _run_lockstep(tasks: list[dict], workers: int, decide_batch, on_episode) -> None:
    """``decide_batch(keys, states, styles, plans)`` -> [(action, slot|None)]; each
    key's history entries arrive through ``on_entries`` first."""
    ctx = mp.get_context("spawn")
    shards = [tasks[i::workers] for i in range(workers)]
    procs, conns = [], []
    for shard in shards:
        if not shard:
            continue
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_lockstep_worker, args=(child, shard), daemon=True)
        p.start()
        procs.append(p)
        conns.append(parent)
    active = list(conns)
    while active:
        pending = []  # (conn, state, new entries, style, plan)
        for c in list(active):
            while True:
                msg = c.recv()
                if msg[0] == "state":
                    pending.append((c, *msg[1:]))
                    break
                if msg[0] == "begin":
                    decide_batch.begin(c)
                    continue
                if msg[0] == "episode":
                    on_episode(msg[1])
                    continue
                active.remove(c)
                break
        if pending:
            outs = decide_batch(pending)
            for (c, *_), out in zip(pending, outs):
                c.send(out)
    for p in procs:
        p.join()


class _StudentBatch:
    """The student behind the lockstep: one tokenized history per game."""

    def __init__(self, pol):
        from policy import ARMS

        self.pol, self.arms = pol, ARMS
        self.hist: dict[object, History] = {}

    def begin(self, key) -> None:
        self.hist[key] = History(self.pol.tok)

    def __call__(self, pending):
        games, adapters = [], []
        for c, state, new, style, plan in pending:
            h = self.hist[c]
            for e in new:
                h.append(e)
            games.append((h.ids, state))
            adapters.append([style, self.arms] if plan else [style])
        decs = self.pol.decide_games(games, adapters)
        out = []
        for (_, _, _, style, plan), d in zip(pending, decs):
            slot = int(d[self.arms].action) if plan else None
            out.append((d[style].action, slot))
        return out


class _RandomBatch:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def begin(self, key) -> None:
        pass

    def __call__(self, pending):
        return [
            (self.rng.choice(ACTIONS), self.rng.choice(WEAPON_SLOTS) if p[4] else None)
            for p in pending
        ]


# ── Reporting ──────────────────────────────────────────────────────────────────
def summarize(stats: list[dict]) -> str:
    def per_min(key):
        return lambda s: 60 * s[key] / max(1, s["seconds"])

    cols = [
        ("frags", lambda s: s["frags"]),
        ("deaths", lambda s: s["deaths"]),
        ("margin", lambda s: s["margin"]),
        ("top%", lambda s: 100.0 * s["top"]),
        ("rank", lambda s: s["rank"]),
        ("best bot", lambda s: s["best_bot"][1]),
        ("frags/min", per_min("frags")),
        ("dealt/min", per_min("damage_dealt")),
        ("taken/min", per_min("damage_taken")),
        ("pick/min", per_min("pickups")),
        ("threat m", lambda s: s["threat_dist"]),
    ]
    if any("agreement" in s for s in stats):
        cols.append(("agree%", lambda s: 100.0 * s.get("agreement", 0)))
    head = f"{'behavior':<10} {'bots':<8} {'n':>3} " + " ".join(
        f"{c:>13}" for c, _ in cols
    )
    lines = [head, "-" * len(head)]
    groups = sorted(
        {(s["behavior"], s["bots"]) for s in stats},
        key=lambda g: (g[0] not in BEHAVIORS, g),
    )
    for b, bots in groups:
        ss = [s for s in stats if s["behavior"] == b and s["bots"] == bots]
        cells = []
        for _, f in cols:
            vals = [f(s) for s in ss]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{statistics.mean(vals):7.1f} ±{sd:5.1f}")
        lines.append(
            f"{b:<10} {bots:<8} {len(ss):>3} " + " ".join(f"{c:>13}" for c in cells)
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--policy", choices=("teacher", "vllm", "random"), default="teacher"
    )
    ap.add_argument(
        "--teacher", default="expert", help="'expert' or an rl_teacher.py checkpoint"
    )
    ap.add_argument("--model", help="Composed checkpoint (for --policy vllm)")
    ap.add_argument(
        "--layout", default="log", help="The student's prompt layout: log or chat"
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="Student sampling (vllm)"
    )
    ap.add_argument(
        "--behaviors", nargs="+", default=list(BEHAVIORS), choices=BEHAVIORS
    )
    ap.add_argument("--episodes", type=int, default=50, help="Per behavior")
    ap.add_argument(
        "--bots",
        default="default",
        choices=[*sorted(BOT_SETS), "all"],
        help="A bot tier, or 'all' for a random tier per match",
    )
    ap.add_argument("--n-bots", type=int, default=7)
    ap.add_argument(
        "--n-bots-min", type=int, help="Random bot count per match in [min, --n-bots]"
    )
    ap.add_argument(
        "--timeout-min-s",
        type=float,
        help="Random match length per match in [min, --timeout-s]",
    )
    ap.add_argument(
        "--row-every", type=int, default=1, help="Keep every k-th tic's row"
    )
    ap.add_argument(
        "--respawn-s", type=int, default=RESPAWN_S, help="Seconds dead per death"
    )
    ap.add_argument(
        "--item-rules", default="standard", choices=["standard", "classic", "scarce"]
    )
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument(
        "--timeout-s", type=float, default=MATCH_TICS / TIC_HZ, help="Match length"
    )
    ap.add_argument(
        "--dart", type=float, default=0.0, help="Noise-burst rate (scripted teacher)"
    )
    ap.add_argument(
        "--beta", type=float, default=0.0, help="DAgger: P(execute teacher label)"
    )
    ap.add_argument("--record", type=int, default=0, help="MP4s per behavior")
    ap.add_argument(
        "--takeover-s", type=float, help="Teacher plays after this many seconds (vllm)"
    )
    ap.add_argument(
        "--mark-s", type=float, help="Record frags/deaths/damage at this time"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stats-only", action="store_true", help="Do not write per-tic rows"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out = args.out.resolve()  # workers run in their own directories
    args.out.mkdir(parents=True, exist_ok=True)
    teacher = (
        args.teacher if args.teacher == "expert" else str(Path(args.teacher).resolve())
    )
    tasks = []
    for bi, b in enumerate(args.behaviors):
        for ep in range(args.episodes):
            seed = args.seed + 1000 * bi + ep
            rng = random.Random(seed)  # per-match variety, reproducible
            tier = rng.choice(sorted(BOT_SETS)) if args.bots == "all" else args.bots
            n_bots = (
                rng.randint(args.n_bots_min, args.n_bots)
                if args.n_bots_min
                else args.n_bots
            )
            secs = (
                rng.uniform(args.timeout_min_s, args.timeout_s)
                if args.timeout_min_s
                else args.timeout_s
            )
            tasks.append(
                {
                    "behavior": b,
                    "ep": ep,
                    "seed": seed,
                    "timeout": int(secs * TIC_HZ),
                    "bots": tier,
                    "n_bots": n_bots,
                    "row_every": args.row_every,
                    "respawn_s": args.respawn_s,
                    "item_rules": args.item_rules,
                    "teacher": teacher,
                    "dart": args.dart,
                    "beta": args.beta,
                    "policy": args.policy,
                    "takeover": None
                    if args.takeover_s is None
                    else int(args.takeover_s * TIC_HZ),
                    "mark": None if args.mark_s is None else int(args.mark_s * TIC_HZ),
                    "keep_rows": not args.stats_only,
                    "video": str(args.out / "videos" / f"{b}_ep{ep}.mp4")
                    if ep < args.record
                    else None,
                }
            )

    row_files, hist_files = {}, {}
    if not args.stats_only:
        for b in args.behaviors:
            row_files[b] = open(args.out / f"{b}.jsonl", "w")
            hist_files[b] = open(args.out / f"{b}_history.jsonl", "w")
    stats_f = open(args.out / "stats.jsonl", "w")
    all_stats: list[dict] = []
    n_rows = 0

    def on_episode(res: dict) -> None:
        nonlocal n_rows
        s = res["stats"]
        all_stats.append(s)
        stats_f.write(json.dumps(s) + "\n")
        stats_f.flush()
        if res["rows"]:
            f = row_files[s["behavior"]]
            for r in res["rows"]:
                f.write(json.dumps(r) + "\n")
            hist_files[s["behavior"]].write(json.dumps(res["history"]) + "\n")
            n_rows += len(res["rows"])
        print(
            f"[{len(all_stats)}/{len(tasks)}] {s['behavior']:<9} ep{s['ep']:<3} "
            f"frags {s['frags']:>3} deaths {s['deaths']:>3} margin {s['margin']:>+4} "
            f"rank {s['rank']} (best bot {s['best_bot'][0]} {s['best_bot'][1]}) "
            f"pickups {s['pickups']:>3}"
            + (f" agree {100 * s['agreement']:.1f}%" if "agreement" in s else ""),
            flush=True,
        )

    t0 = time.time()
    if args.policy == "teacher":
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            for res in pool.imap_unordered(_teacher_episode, tasks):
                on_episode(res)
    else:
        if args.policy == "vllm":
            from policy import VLLMPolicy

            pol = VLLMPolicy(
                args.model,
                max_num_seqs=max(16, 2 * args.workers),
                warmup=5,
                temperature=args.temperature,
                layout=args.layout,
            )
            batch = _StudentBatch(pol)
        else:
            batch = _RandomBatch(args.seed)
        _run_lockstep(tasks, args.workers, batch, on_episode)

    for f in [*row_files.values(), *hist_files.values()]:
        f.close()
    stats_f.close()
    table = summarize(all_stats)
    (args.out / "summary.txt").write_text(table + "\n")
    print(
        f"\n{len(all_stats)} matches, {n_rows} rows in {time.time() - t0:.0f}s -> {args.out}\n"
    )
    print(table)


if __name__ == "__main__":
    main()
