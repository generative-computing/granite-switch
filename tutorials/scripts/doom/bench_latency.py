# SPDX-License-Identifier: Apache-2.0
"""Per-tic latency of a composed checkpoint on real Doom prompts with history.

A tic's latency is wall clock from "state is available" to "outputs are
available": prompt-id assembly plus one vLLM engine step, which prefills the
fresh tokens and emits one token for every adapter asked that tic.

The prompts come from a trace of scripted deathmatch play (``--seconds`` of it),
replayed in order with its 5 Hz history, so every run sees the same prompts,
including the history window resets (see :mod:`history`). Run once per
checkpoint: the aLoRA one and the same adapters composed as LoRA, whose control
token sits at position 0 so each adapter has its own KV. Scenarios:

* ``reflex``: one game, the fighter every tic. Tics are split into steady ones,
  ones where the history grew (every 0.2 s) and window resets.
* ``multi``: fighter and critic every tic, plus the weapon planner every
  ``PLAN_EVERY_TICS``, in one engine step per tic.
* ``nadapters``: per-tic cost against the number of game adapters asked at once.
* ``switch``: the fighter runs alone; every ``--switch-every`` s the behavior
  switches (fighter <-> cautious). Reports the switch tic against steady ones.
* ``inbatch``: N adapters on a history never seen before, in one step: does
  vLLM share the prefix inside one batch? (Per-request cached tokens.)
* ``kv``: KV blocks one game needs with N adapters, and how many such games fit.
* ``games``: N games deciding in the same engine step, each with its history.
* ``router``: one-token instruction routing.
* ``live``: turbo play with fighter + critic + planner, env in the loop.
* ``talk`` / ``notalk``: N games in real time (a tic every 28.6 ms each), with
  and without each game's base model speaking a line every ``--talk-every`` s.
  How late decisions land against their tic, and how many tics go undecided.

Examples::

    python bench_latency.py --model models/standin --json out/bench_alora.json
    python bench_latency.py --model models/standin-lora --json out/bench_lora.json
    python bench_latency.py --compare out/bench_alora.json out/bench_lora.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import TIC_HZ, TIC_MS, DoomEnv
from expert import BEHAVIORS, PLAN_EVERY_TICS, Expert
from history import History, said_entry
from policy import ARMS, CRITIC, GAME_ADAPTERS, VLLMPolicy, state_text

INSTRUCTIONS = [
    "go kill everything",
    "stop fighting and grab health",
    "pick up all the ammo and armor you can find",
    "play it safe for a while",
    "hunt them down",
]
FIGHTER, CAUTIOUS = BEHAVIORS[0], BEHAVIORS[1]


class GCTimer:
    """Python's cyclic GC pauses (count, total and longest, by generation)."""

    def __init__(self) -> None:
        self.reset()
        gc.callbacks.append(self._cb)

    def reset(self) -> None:
        self.pauses: dict[int, list[float]] = {0: [], 1: [], 2: []}
        self._t0 = 0.0

    def _cb(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
        else:
            self.pauses[info["generation"]].append(
                (time.perf_counter() - self._t0) * 1000
            )

    def summary(self) -> dict:
        return {
            f"gen{g}": {
                "n": len(p),
                "total_ms": round(sum(p), 2),
                "max_ms": round(max(p), 2) if p else 0.0,
            }
            for g, p in self.pauses.items()
        }


def pct(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


# ── Trace ──────────────────────────────────────────────────────────────────────
def make_trace(seconds: float, seed: int) -> list[dict]:
    """Scripted deathmatch play: per tic the state text (None while dead) and the
    history entry appended after deciding it (None on most tics)."""
    env = DoomEnv(
        seed=seed, hud=True, resolution="640X480", timeout_tics=int(seconds * TIC_HZ)
    )
    ex, hist, trace = Expert(), History(), []
    obs = env.reset(seed=seed)
    while not obs.done:
        b = BEHAVIORS[(obs.tick // (10 * TIC_HZ)) % 2]
        a = ex.act(obs, b)
        state = None if obs.dead else state_text(obs)
        entry = hist.observe(obs, a)
        trace.append({"tick": obs.tick, "state": state, "entry": entry})
        w = ex.weapon(obs) if obs.tick % PLAN_EVERY_TICS == 0 else None
        obs = env.step(a, weapon=w)
    env.close()
    return trace


class Replay:
    """Walks the trace, keeping a tokenized history; yields (tic, kind, hist, state)."""

    def __init__(self, trace: list[dict], tokenizer):
        self.trace, self.tok = trace, tokenizer

    def __iter__(self):
        hist = History(self.tok)
        kind = "steady"
        for t in self.trace:
            if t["state"] is not None:
                yield t["tick"], kind, hist.ids, t["state"]
                kind = "steady"
            if t["entry"] is not None:
                kind = "reset" if hist.append(t["entry"]) else "grew"


def split_stats(rows: list[tuple[str, float]]) -> dict:
    out = {"all": pct([ms for _, ms in rows])}
    for k in ("steady", "grew", "reset"):
        out[k] = pct([ms for kind, ms in rows if kind == k])
    out["within_tic"] = round(float(np.mean([ms < TIC_MS for _, ms in rows])), 4)
    return out


# ── Scenarios ──────────────────────────────────────────────────────────────────
def bench_reflex(pol: VLLMPolicy, trace, adapters_at) -> dict:
    """One game; ``adapters_at(tick)`` lists the adapters asked on that tic."""
    rows, fresh, cached, prompt = [], [], [], []
    for tick, kind, hist, state in Replay(trace, pol.tok):
        decs = pol.decide_games([(hist, state)], [adapters_at(tick)])[0]
        d = next(iter(decs.values()))
        rows.append((kind, d.ms))
        for x in decs.values():
            fresh.append(x.fresh_tokens)
            cached.append(x.cached_tokens)
            prompt.append(x.fresh_tokens + x.cached_tokens)
    tot = sum(fresh) + sum(cached)
    return {
        "tic_ms": split_stats(rows),
        "prompt_tokens": pct(prompt),
        "fresh_tokens": pct(fresh),
        "prefix_cache_hit_rate": round(sum(cached) / max(1, tot), 4),
    }


def bench_nadapters(pol: VLLMPolicy, trace, max_n: int) -> dict:
    out = {}
    for n in range(1, max_n + 1):
        names = list(GAME_ADAPTERS[:n])
        rows = []
        for i, (_, kind, hist, state) in enumerate(Replay(trace, pol.tok)):
            if i >= 1500:
                break
            d = pol.decide_games([(hist, state)], [names])[0]
            rows.append((kind, next(iter(d.values())).ms))
        out[str(n)] = split_stats(rows)
        s = out[str(n)]["all"]
        print(
            f"  {n} adapters: p50 {s['p50']:6.2f} ms  p99 {s['p99']:6.2f} ms",
            flush=True,
        )
    return out


def bench_switch(pol: VLLMPolicy, trace, every_s: float) -> dict:
    """Behavior switches every ``every_s`` seconds; the new one last ran then."""
    period = int(every_s * TIC_HZ)
    switch_ms, after_ms, steady = [], [], []
    cur, last_switch = FIGHTER, None
    for tick, _, hist, state in Replay(trace, pol.tok):
        seg = tick // period
        want = FIGHTER if seg % 2 == 0 else CAUTIOUS
        switched = want != cur
        cur = want
        d = pol.decide_games([(hist, state)], [[cur]])[0][cur]
        if switched and seg >= 2:  # both adapters have run before
            switch_ms.append(d.ms)
            last_switch = tick
        elif last_switch is not None and 0 < tick - last_switch <= 3:
            after_ms.append(d.ms)
        else:
            steady.append(d.ms)
    return {
        "every_s": every_s,
        "switch_tic_ms": pct(switch_ms),
        "next_3_tics_ms": pct(after_ms),
        "steady_ms": pct(steady),
    }


def bench_inbatch(pol: VLLMPolicy, trace, reps: int) -> dict:
    """N adapters, one step, on a history no request has seen (a random shuffle
    of real entries): how many prompt tokens does each request find cached?"""
    rng = random.Random(0)
    entries = [t["entry"] for t in trace if t["entry"]]
    states = [t["state"] for t in trace if t["state"]]
    res = {}
    for n in (1, 3, 5):
        names = list(GAME_ADAPTERS[:n])
        per_req, ms = [[] for _ in names], []
        for _ in range(reps):
            h = History.replay(rng.sample(entries, 50), pol.tok)
            d = pol.decide_games([(h.ids, rng.choice(states))], [names])[0]
            for i, a in enumerate(names):
                per_req[i].append(d[a].cached_tokens)
            ms.append(d[names[0]].ms)
            prompt = d[names[0]].fresh_tokens + d[names[0]].cached_tokens
        res[str(n)] = {
            "prompt_tokens": prompt,
            "head_tokens": pol.pb.n_fixed(),
            "cached_tokens_by_request": [round(float(np.mean(c)), 1) for c in per_req],
            "step_ms": pct(ms),
        }
        print(f"  {n} adapters on a new history: {res[str(n)]}", flush=True)
    return res


def bench_kv(pol: VLLMPolicy, trace) -> dict:
    """KV blocks one game holds, by number of adapters, from real prompt lengths."""
    cc = pol.llm.llm_engine.vllm_config.cache_config
    block = cc.block_size
    lens = []
    for i, (_, _, hist, state) in enumerate(Replay(trace, pol.tok)):
        if i % 50 == 0:
            p = pol.pb.game_ids(hist, state, [GAME_ADAPTERS[0]])[0]
            lens.append(len(p))
    n_suffix = len(pol.pb.suffix[GAME_ADAPTERS[0]])
    p95 = int(np.percentile(lens, 95))
    out = {"block_size": block, "gpu_blocks": cc.num_gpu_blocks, "prompt_p95": p95}
    for n in (1, 3, 5):
        if pol.lora:
            blocks = n * math.ceil(p95 / block)
        else:
            blocks = math.ceil((p95 - n_suffix) / block) + n
        out[str(n)] = {
            "blocks_per_game": blocks,
            "games_fit": (cc.num_gpu_blocks or 0) // blocks,
        }
    return out


def bench_games(pol: VLLMPolicy, trace, sizes: list[int], reps: int) -> dict:
    """N games in one step, each at its own point of the trace, fighter only."""
    snaps = [(h, s) for _, _, h, s in Replay(trace, pol.tok)]
    gc.freeze()  # a full GC pass over thousands of snapshots is a ~25 ms pause
    rng = random.Random(0)
    out = {}
    for n in sizes:
        ms = []
        # Walk N games forward together so each keeps its prefix cache warm.
        starts = [rng.randrange(len(snaps) - reps - 1) for _ in range(n)]
        for r in range(reps):
            games = [snaps[s + r] for s in starts]
            d = pol.decide_games(games, [[FIGHTER]] * n)
            ms.append(d[0][FIGHTER].ms)
        st = pct(ms)
        st["games_at_35hz"] = n if st["p99"] < TIC_MS else 0
        out[str(n)] = st
        print(
            f"  {n:>3} games: p50 {st['p50']:6.2f} ms  p99 {st['p99']:6.2f} ms",
            flush=True,
        )
    return out


def bench_talk(
    pol: VLLMPolicy,
    trace,
    sizes: list[int],
    seconds: float,
    talk_every_s: float,
    talk_tokens: int,
    warm_s: float = 2.0,
) -> dict:
    """N games in real time, several of them talking.

    Each game runs its own tic clock (every game's clock offset within the tic)
    and replays the trace from its own starting point. On every tic it asks the
    fighter and critic, and the planner on its cadence. If that tic's decisions
    are not back by the next tic, the game moves on without a new decision
    (it plays a stale one). Every ``talk_every_s`` seconds (0: never) its base
    model speaks ``talk_tokens`` tokens, and the line becomes a history entry, so
    the shared context grows as it would in the live demo. One engine step at a
    time, as vLLM runs anyway.

    Reports, after ``warm_s`` seconds: decision latency from the tic's start,
    the share of decisions later than one tic, the share of tics skipped
    because the last decision was still pending, and how long a spoken line takes.
    """
    from vllm.inputs import TokensPrompt

    engine = pol.llm.llm_engine
    tic_s = 1.0 / TIC_HZ
    rng = random.Random(0)
    out = {}
    for n in sizes:
        pol.llm.reset_prefix_cache()
        t_start = time.perf_counter() + 0.2
        t_warm, t_end = t_start + warm_s, t_start + warm_s + seconds
        games = []
        for i in range(n):
            nxt = t_start + rng.uniform(0, talk_every_s) if talk_every_s else math.inf
            games.append(
                {
                    "off": rng.randrange(len(trace)),
                    "k": 0,
                    "phase": t_start + i * tic_s / n,
                    "hist": History(pol.tok),
                    "state": None,
                    "busy": set(),
                    "talk": None,
                    "next_talk": nxt,
                }
            )
        pending: dict[str, tuple] = {}
        lat, talk_ms, skipped, decided, seq = [], [], 0, 0, 0
        while True:
            now = time.perf_counter()
            if now >= t_end and not pending:
                break
            for gi, g in enumerate(games):
                if now >= t_end:
                    break
                due = g["phase"] + g["k"] * tic_s
                if now < due:
                    continue
                t = trace[(g["off"] + g["k"]) % len(trace)]
                g["k"] += 1
                if t["state"] is not None:
                    g["state"] = t["state"]
                    if g["busy"]:
                        skipped += now >= t_warm
                    else:
                        want = [FIGHTER, CRITIC]
                        if t["tick"] % PLAN_EVERY_TICS == 0:
                            want.append(ARMS)
                        ids = pol.pb.game_ids(g["hist"].ids, t["state"], want)
                        for a, p in zip(want, ids):
                            rid = f"r{seq}"
                            seq += 1
                            engine.add_request(
                                rid, TokensPrompt(prompt_token_ids=p), pol.sp[a]
                            )
                            pending[rid] = ("reflex", gi, due)
                            g["busy"].add(rid)
                        decided += now >= t_warm
                if t["entry"] is not None:
                    g["hist"].append(t["entry"])
                if g["talk"] is None and now >= g["next_talk"] and g["state"]:
                    rid = f"t{seq}"
                    seq += 1
                    sp = pol.talk_params(
                        max_tokens=talk_tokens, ignore_eos=True, stop=None
                    )
                    p = pol.pb.talk_ids(g["hist"].ids, g["state"])
                    engine.add_request(rid, TokensPrompt(prompt_token_ids=p), sp)
                    pending[rid] = ("talk", gi, now)
                    g["talk"], g["talk_tick"] = rid, t["tick"]
                    g["next_talk"] += talk_every_s
            if not pending:
                nxt = min(g["phase"] + g["k"] * tic_s for g in games)
                time.sleep(max(0.0, nxt - time.perf_counter()))
                continue
            for o in engine.step():
                if not o.finished:
                    continue
                what, gi, t0 = pending.pop(o.request_id)
                done = time.perf_counter()
                g = games[gi]
                if what == "reflex":
                    g["busy"].discard(o.request_id)
                    if t0 >= t_warm:
                        lat.append((done - t0) * 1000)
                else:
                    g["talk"] = None
                    g["hist"].append(
                        said_entry(g["talk_tick"], "me", o.outputs[0].text)
                    )
                    if t0 >= t_warm:
                        talk_ms.append((done - t0) * 1000)
        st = pct(lat)
        st["late_share"] = round(float(np.mean([x > TIC_MS for x in lat])), 4)
        st["skipped_tics"] = round(skipped / max(1, skipped + decided), 4)
        st["talk_ms"] = pct(talk_ms)
        out[str(n)] = st
        print(
            f"  {n:>3} games{' talking' if talk_every_s else ''}: decision p50 "
            f"{st['p50']:6.2f} ms  p99 {st['p99']:6.2f} ms  later than a tic "
            f"{100 * st['late_share']:5.1f}%  skipped tics {100 * st['skipped_tics']:5.1f}%"
            + (f"  line p50 {st['talk_ms'].get('p50', 0):6.0f} ms" if talk_ms else ""),
            flush=True,
        )
    return out


def bench_router(pol: VLLMPolicy, n: int) -> dict:
    routes = [pol.route(INSTRUCTIONS[i % len(INSTRUCTIONS)]) for i in range(n)]
    return {"route_ms": pct([r.ms for r in routes])}


def bench_live(pol: VLLMPolicy, seconds: float, seed: int) -> dict:
    """Unthrottled play: the model sets the pace. Fighter + critic every tic,
    planner on its cadence; env time is reported separately."""
    env = DoomEnv(seed=seed, hud=True, resolution="640X480", timeout_tics=10**7)
    hist = History(pol.tok)
    obs = env.reset(seed=seed)
    dec_ms, env_ms, n, t_end = [], [], 0, time.perf_counter() + seconds
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        action, weapon = "wait", None
        if not obs.dead:
            names = [FIGHTER, CRITIC] + (
                [ARMS] if obs.tick % PLAN_EVERY_TICS == 0 else []
            )
            decs = pol.decide_many(obs, tuple(names), hist)
            action = decs[FIGHTER].action
            weapon = int(decs[ARMS].action) if ARMS in decs else None
            dec_ms.append(decs[FIGHTER].ms)
        hist.observe(obs, action)
        t = time.perf_counter()
        obs = env.step(action, weapon=weapon)
        env_ms.append((time.perf_counter() - t) * 1000)
        n += 1
        if obs.done:
            obs = env.reset()
            hist.reset()
    wall = time.perf_counter() - t_start
    env.close()
    return {
        "tic_ms": pct(dec_ms),
        "env_step_ms": pct(env_ms),
        "tics": n,
        "tics_per_s": round(n / wall, 1),
        "x_realtime": round(n / wall / TIC_HZ, 2),
        "history_resets": hist.resets,
    }


# ── Comparison ─────────────────────────────────────────────────────────────────
def compare(a_path: Path, b_path: Path) -> None:
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    la = "LoRA" if a["config"].get("lora") else "aLoRA"
    lb = "LoRA" if b["config"].get("lora") else "aLoRA"
    print(f"{'metric':<52}{la:>12}{lb:>12}")

    def row(name, fa, fb=None):
        fb = fb or fa
        try:
            va, vb = fa(a), fb(b)
        except (KeyError, TypeError):
            return
        print(f"{name:<52}{va:>12}{vb:>12}")

    for scen in ("reflex", "multi"):
        for part in ("all", "steady", "grew", "reset"):
            for q in ("p50", "p99"):
                row(
                    f"{scen} tic ms {part} {q}",
                    lambda r, s=scen, p=part, q=q: r[s]["tic_ms"][p][q],
                )
        row(f"{scen} within one tic", lambda r, s=scen: r[s]["tic_ms"]["within_tic"])
        row(f"{scen} fresh tokens p50", lambda r, s=scen: r[s]["fresh_tokens"]["p50"])
        row(
            f"{scen} prefix-cache hit rate",
            lambda r, s=scen: r[s]["prefix_cache_hit_rate"],
        )
    for n in ("1", "2", "3", "4", "5"):
        row(f"{n} adapters/tic p50 ms", lambda r, n=n: r["nadapters"][n]["all"]["p50"])
        row(f"{n} adapters/tic p99 ms", lambda r, n=n: r["nadapters"][n]["all"]["p99"])
    row("switch tic p50 ms", lambda r: r["switch"]["switch_tic_ms"]["p50"])
    row("switch tic max ms", lambda r: r["switch"]["switch_tic_ms"]["max"])
    row("steady tic p50 ms (switch run)", lambda r: r["switch"]["steady_ms"]["p50"])
    for n in ("1", "3", "5"):
        row(
            f"KV blocks/game, {n} adapters",
            lambda r, n=n: r["kv"][n]["blocks_per_game"],
        )
        row(f"games that fit, {n} adapters", lambda r, n=n: r["kv"][n]["games_fit"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A_JSON", "B_JSON"))
    ap.add_argument("--plain-base", action="store_true", help="No control tokens")
    ap.add_argument("--engine-loop", action="store_true")
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--no-align", action="store_true", help="No block padding")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument(
        "--cudagraph-mode",
        default="FULL",
        choices=("FULL", "FULL_AND_PIECEWISE", "PIECEWISE"),
    )
    ap.add_argument("--seconds", type=float, default=90.0, help="Trace length")
    ap.add_argument("--switch-every", type=float, default=10.0)
    ap.add_argument("--game-sizes", default="1,2,4,8,16")
    ap.add_argument("--live-seconds", type=float, default=20.0)
    ap.add_argument("--talk-sizes", default="1,2,4,8,12,16", help="Games (talk)")
    ap.add_argument("--talk-seconds", type=float, default=20.0)
    ap.add_argument("--talk-every", type=float, default=4.0, help="Seconds per line")
    ap.add_argument("--talk-tokens", type=int, default=20)
    ap.add_argument(
        "--scenarios",
        default="reflex,multi,nadapters,switch,inbatch,kv,games,router,live",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.model:
        raise SystemExit("--model is required")
    scen = set(args.scenarios.split(","))
    sizes = [int(x) for x in args.game_sizes.split(",")]
    talk_sizes = [int(x) for x in args.talk_sizes.split(",")]
    t0 = time.time()
    trace = make_trace(args.seconds, args.seed)
    n_live = sum(t["state"] is not None for t in trace)
    print(
        f"trace: {len(trace)} tics ({n_live} alive) in {time.time() - t0:.1f}s",
        flush=True,
    )

    pol = VLLMPolicy(
        args.model,
        engine_loop=args.engine_loop,
        prefix_caching=not args.no_prefix_cache,
        align=not args.no_align,
        # A talking game can hold four requests at once (three adapters, one line).
        max_num_seqs=max(16, max(sizes) * 2, max(talk_sizes) * 4, len(GAME_ADAPTERS)),
        enforce_eager=args.enforce_eager,
        cudagraph_mode=args.cudagraph_mode,
        base_model=args.plain_base,
        warmup=60,
    )
    import torch
    import vllm

    res: dict = {
        "config": {
            "model": args.model,
            "lora": pol.lora,
            "plain_base": args.plain_base,
            "engine_loop": args.engine_loop,
            "prefix_caching": not args.no_prefix_cache,
            "align": not args.no_align,
            "enforce_eager": args.enforce_eager,
            "cudagraph_mode": args.cudagraph_mode,
            "trace_seconds": args.seconds,
            "gpu": torch.cuda.get_device_name(0),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
        }
    }

    def multi_at(tick: int) -> list[str]:
        return [FIGHTER, CRITIC] + ([ARMS] if tick % PLAN_EVERY_TICS == 0 else [])

    runs = [
        ("reflex", lambda: bench_reflex(pol, trace, lambda t: [FIGHTER])),
        ("multi", lambda: bench_reflex(pol, trace, multi_at)),
        ("nadapters", lambda: bench_nadapters(pol, trace, 5)),
        ("switch", lambda: bench_switch(pol, trace, args.switch_every)),
        ("inbatch", lambda: bench_inbatch(pol, trace, 30)),
        ("kv", lambda: bench_kv(pol, trace)),
        ("games", lambda: bench_games(pol, trace, sizes, reps=300)),
        ("router", lambda: bench_router(pol, 200)),
        (
            "notalk",
            lambda: bench_talk(pol, trace, talk_sizes, args.talk_seconds, 0, 0),
        ),
        (
            "talk",
            lambda: bench_talk(
                pol,
                trace,
                talk_sizes,
                args.talk_seconds,
                args.talk_every,
                args.talk_tokens,
            ),
        ),
        ("live", lambda: bench_live(pol, args.live_seconds, args.seed + 99)),
    ]
    gct = GCTimer()
    for name, fn in runs:
        if name not in scen or (args.plain_base and name in ("switch", "router")):
            continue
        print(f"\n== {name}", flush=True)
        # Scenarios replay the same trace: start each from an empty prefix cache.
        pol.llm.reset_prefix_cache()
        gct.reset()
        res[name] = fn()
        res[name]["python_gc"] = gct.summary()
        print(json.dumps(res[name], indent=1), flush=True)

    for name in ("reflex", "multi"):
        if name in res:
            s = res[name]["tic_ms"]["all"]
            r = res[name]["tic_ms"]["reset"]
            print(
                f"SUMMARY {name}: p50 {s['p50']:.2f} ms p99 {s['p99']:.2f} ms "
                f"(window resets p99 {r.get('p99', float('nan')):.2f} ms); tic budget "
                f"{TIC_MS:.1f} ms; gate p99 < tic: {'PASS' if s['p99'] < TIC_MS else 'FAIL'}"
            )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
