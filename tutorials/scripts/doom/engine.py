# SPDX-License-Identifier: Apache-2.0
"""Real-time Granite Switch play: many games, one AsyncLLM engine, and talk.

Each game is a ViZDoom match in its own process that runs on the wall clock,
one tic every 28.6 ms whether or not the model has answered. On every tic the
game sends its state and waits for that tic's decision only until the tic is
due, then plays the freshest action it has. A decision that arrives late is
applied a tic or more later: its reaction time is real, not simulated.

The server holds each game's tokenized history (the game sends its 5 Hz
entries as text), asks the style adapter, critic and, on its cadence, the
weapon planner for every tic, and lets the base model speak a short line every
few seconds or right after a frag, a death or a new weapon. The line becomes a
``me:`` history entry, so every adapter's next prompt holds it. All requests go
to one vLLM ``AsyncLLM``: its engine core runs in its own process, so a spoken
request (including audio transcribed in the model's ASR cascade) never holds up
a reflex decision.

    python engine.py run --model models/doom-f-alora --games 8 --seconds 120
    # model on a GPU node, games on a CPU node:
    python engine.py serve --model models/doom-f-lora --games 8 --addr-file out/addr
    python engine.py play --games 8 --addr-file out/addr
"""

from __future__ import annotations

import os

# AsyncLLM runs its engine core in a separate process (env.sh sets 0 for LLM).
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

import argparse
import asyncio
import json
import multiprocessing as mp
import random
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from doom_env import TIC_HZ, DoomEnv, isolate_workdir
from expert import BEHAVIORS, PLAN_EVERY_TICS
from history import History, said_entry
from policy import (
    ARMS,
    CRITIC,
    engine_kwargs,
    output_dist,
    prompt_kit,
    state_text,
)
from talk import brief

TIC_S = 1.0 / TIC_HZ
SALIENT = ("frag", "died", "got weapon")  # say something soon after these
MIN_TALK_GAP_S = 2.5


# ── Game side (one process per match) ──────────────────────────────────────────
def game_worker(conn, spec: dict) -> None:
    """Play one match in real time. Messages to the server:
    ``("obs", tick, state | None, entry | None, events)`` every tic (``entry`` is
    the history entry the previous tic produced), then ``("done", stats, lags)``.
    From the server: ``("act", tick, action, slot | None)``."""
    isolate_workdir()
    env = DoomEnv(
        seed=spec["seed"],
        resolution="640X480",  # the students' training data
        timeout_tics=int(spec["seconds"] * TIC_HZ),
        bots=spec["bots"],
        n_bots=spec["n_bots"],
    )
    hist = History()  # text only: the server tokenizes
    obs = env.reset(seed=spec["seed"])
    action, fresh_of, slot, entry = "wait", -1, None, None
    lags: list[int] = []  # per live tic: ticks between the state and the action played
    overruns, step_ms = 0, 5.0
    t_tic = time.perf_counter()
    while not obs.done:
        state = None if obs.dead else state_text(obs)
        conn.send(("obs", obs.tick, state, entry, list(obs.events)))
        deadline = t_tic + TIC_S - (step_ms + 1.0) / 1000
        while state is not None:
            left = deadline - time.perf_counter()
            if left <= 0 or not conn.poll(left):
                break
            _, tick, a, s = conn.recv()
            if tick >= fresh_of:
                action, fresh_of = a, tick
            if s is not None:
                slot = s
            if tick == obs.tick:
                break
        if state is not None:
            lags.append(obs.tick - fresh_of if fresh_of >= 0 else 99)
        entry = hist.observe(obs, None if obs.dead else action)
        t0 = time.perf_counter()
        obs = env.step(action, weapon=slot)
        slot = None
        step_ms = 0.9 * step_ms + 0.1 * (time.perf_counter() - t0) * 1000
        t_tic += TIC_S
        wait = t_tic - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            overruns += 1
    stats = env.stats.as_dict()
    stats["overrun_tics"] = overruns
    env.close()
    conn.send(("done", stats, lags))


# ── Server side ────────────────────────────────────────────────────────────────
class AsyncPolicy:
    """The composed checkpoint behind a vLLM AsyncLLM, with the same prompts,
    vocabularies and engine settings as :class:`policy.VLLMPolicy`."""

    def __init__(self, model: str, *, max_num_seqs: int, gpu_mem: float, temperature):
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.config import CompilationConfig
        from vllm.v1.engine.async_llm import AsyncLLM

        kw = engine_kwargs(
            model, max_num_seqs=max_num_seqs, gpu_memory_utilization=gpu_mem
        )
        kw["compilation_config"] = CompilationConfig(**kw["compilation_config"])
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
        self.tok = AutoTokenizer.from_pretrained(model)
        self.kit = prompt_kit(self.tok, temperature=temperature)
        # Temperature only (top-p autotunes a FlashInfer kernel on first use).
        self.talk_sp = SamplingParams(
            max_tokens=24, temperature=0.8, stop=["\n"], bad_words=["reload"]
        )

    async def _one(self, ids: list[int], sp):
        from vllm.inputs import TokensPrompt

        final = None
        async for out in self.engine.generate(
            TokensPrompt(prompt_token_ids=ids), sp, uuid.uuid4().hex
        ):
            final = out
        return final

    async def decide(self, hist_ids, state: str, adapters: list[str]) -> dict:
        prompts = self.kit.pb.game_ids(hist_ids, state, adapters)
        outs = await asyncio.gather(
            *(self._one(p, self.kit.sp[a]) for a, p in zip(adapters, prompts))
        )
        res = {}
        for a, o in zip(adapters, outs):
            words = self.kit.words[a]
            res[a] = (words[o.outputs[0].token_ids[0]], dict(output_dist(o, words)))
        return res

    async def talk(self, hist_ids, state: str, brief_text: str) -> str:
        o = await self._one(
            self.kit.pb.talk_ids(hist_ids, state, brief_text), self.talk_sp
        )
        return o.outputs[0].text.strip().strip('"').split("\n")[0]

    async def warmup(self) -> None:
        state = (
            "now t1.0 | hp 100 armor 0 | pistol 50 | arms 2:50 | see bot -12 8m | "
            "wall l3 f9 r9 b2 | hit 0 | last forward forward"
        )
        ids = self.tok.encode(
            "t0.2 hp 100 face 90 | bot -12 8m | did forward\n", add_special_tokens=False
        )
        for n in (1, 4, 16):
            await asyncio.gather(
                *(
                    self.decide(ids, state, [BEHAVIORS[0], CRITIC, ARMS])
                    for _ in range(n)
                )
            )
            await asyncio.gather(*(self.talk(ids, state, "") for _ in range(n)))


def nodelay(conn) -> None:
    """Send small messages at once on a TCP connection (no Nagle batching,
    which with delayed ACKs can hold a decision ~40 ms)."""
    import socket

    s = socket.socket(fileno=os.dup(conn.fileno()))
    if s.family in (socket.AF_INET, socket.AF_INET6):
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.close()  # closes the duplicate only


def remote_worker(addr: tuple[str, int], spec: dict) -> None:
    """A game on another node: connect to ``engine.py serve`` and play."""
    from multiprocessing.connection import Client

    conn = Client(addr, authkey=AUTHKEY)
    nodelay(conn)
    conn.send(("hello", spec))
    game_worker(conn, spec)


AUTHKEY = b"granite-switch-doom"


class Game:
    """One match on the server: its history, decisions and talk. With ``conn``
    the match runs elsewhere (``engine.py play``); without, in a local process."""

    def __init__(
        self, gid: int, spec: dict, pol: AsyncPolicy, talk_every: float, conn=None
    ):
        self.proc = None
        if conn is None:
            ctx = mp.get_context("spawn")
            conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=game_worker, args=(child, spec), daemon=True)
        self.conn, self.spec = conn, spec
        self.gid, self.pol, self.talk_every = gid, pol, talk_every
        self.style = spec.get("style", BEHAVIORS[0])
        self.hist = History(pol.tok)
        self.inflight = self.talking = False
        self.lat_ms: list[float] = []
        self.skipped = self.decided = 0
        self.lines: list[tuple[float, str, int]] = []
        self.last_talk = -1e9
        self.next_talk = time.perf_counter() + random.uniform(1, max(1.0, talk_every))
        self.stats: dict = {}
        self.lags: list[int] = []
        self._tasks: set[asyncio.Task] = set()  # keep running tasks referenced

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        inbox: asyncio.Queue = asyncio.Queue()
        loop.add_reader(self.conn.fileno(), lambda: inbox.put_nowait(self.conn.recv()))
        try:
            while True:
                msg = await inbox.get()
                if msg[0] == "done":
                    _, self.stats, self.lags = msg
                    return
                _, tick, state, entry, events = msg
                if entry is not None:
                    self.hist.append(entry)
                if state is None:
                    continue
                if self.inflight:
                    self.skipped += 1
                else:
                    self.inflight = True
                    self.decided += 1
                    self._spawn(self._decide(tick, state))
                now = time.perf_counter()
                due = now >= self.next_talk or any(e in SALIENT for e in events)
                if (
                    self.talk_every
                    and not self.talking
                    and due
                    and now - self.last_talk >= MIN_TALK_GAP_S
                ):
                    self.talking = True
                    self._spawn(self._talk(tick, state))
        finally:
            loop.remove_reader(self.conn.fileno())

    async def _decide(self, tick: int, state: str) -> None:
        t0 = time.perf_counter()
        want = [self.style, CRITIC] + ([ARMS] if tick % PLAN_EVERY_TICS == 0 else [])
        d = await self.pol.decide(self.hist.ids, state, want)
        self.lat_ms.append((time.perf_counter() - t0) * 1000)
        slot = int(d[ARMS][0]) if ARMS in d else None
        try:
            self.conn.send(("act", tick, d[self.style][0], slot))
        except (BrokenPipeError, OSError):
            pass  # the match just ended
        self.inflight = False

    async def _talk(self, tick: int, state: str) -> None:
        t0 = time.perf_counter()
        line = await self.pol.talk(
            self.hist.ids, state, brief(self.hist.entries, state)
        )
        ms = int((time.perf_counter() - t0) * 1000)
        if line:
            self.hist.append(said_entry(tick, "me", line))
            self.lines.append((round(tick / TIC_HZ, 1), line, ms))
        self.last_talk = time.perf_counter()
        self.next_talk = self.last_talk + self.talk_every
        self.talking = False


def pct(xs, q) -> float:
    return round(float(np.percentile(xs, q)), 2) if len(xs) else float("nan")


def match_spec(args, i: int) -> dict:
    return {
        "seed": args.seed + i,
        "seconds": args.seconds,
        "bots": args.bots,
        "n_bots": args.n_bots,
    }


def play(args) -> None:
    """The games' side of ``serve``: wait for its address file, then run every
    match in its own process on this node, connected over TCP."""
    t0 = time.time()
    while not args.addr_file.exists():
        if time.time() - t0 > args.wait_s:
            raise SystemExit(f"no server address in {args.addr_file}")
        time.sleep(2)
    host, port = args.addr_file.read_text().split()
    print(f"playing {args.games} games against {host}:{port}", flush=True)
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=remote_worker, args=((host, int(port)), match_spec(args, i)))
        for i in range(args.games)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print(f"{args.games} games done in {time.time() - t0:.0f}s", flush=True)


async def serve(args) -> dict:
    t0 = time.time()
    pol = AsyncPolicy(
        args.model,
        max_num_seqs=max(16, 5 * args.games),
        gpu_mem=args.gpu_mem,
        temperature=args.temperature,
    )
    await pol.warmup()
    print(f"engine ready in {time.time() - t0:.0f}s ({pol.kit.placement})", flush=True)
    talk = 0 if args.no_talk else args.talk_every
    games = []
    if args.cmd == "serve":
        import socket
        from multiprocessing.connection import Listener

        listener = Listener(("0.0.0.0", args.port), authkey=AUTHKEY)
        host = socket.getfqdn()
        args.addr_file.parent.mkdir(parents=True, exist_ok=True)
        args.addr_file.write_text(f"{host} {listener.address[1]}\n")
        print(
            f"listening on {host}:{listener.address[1]} for {args.games} games",
            flush=True,
        )
        loop = asyncio.get_running_loop()
        for i in range(args.games):
            conn = await loop.run_in_executor(None, listener.accept)
            nodelay(conn)
            _, spec = await loop.run_in_executor(None, conn.recv)
            games.append(Game(i, spec, pol, talk, conn))
        listener.close()
    else:
        for i in range(args.games):
            games.append(Game(i, match_spec(args, i), pol, talk))
        for g in games:
            g.proc.start()
    await asyncio.gather(*(g.run() for g in games))
    pol.engine.shutdown()

    rows = []
    for g in games:
        lags = Counter(g.lags)
        n = max(1, len(g.lags))
        s = g.stats
        rows.append(
            {
                "game": g.gid,
                "seed": g.spec.get("seed"),
                "frags": s.get("frags"),
                "deaths": s.get("deaths"),
                "margin": s.get("margin"),
                "rank": s.get("rank"),
                "decision_ms_p50": pct(g.lat_ms, 50),
                "decision_ms_p99": pct(g.lat_ms, 99),
                "fresh_share": round(lags[0] / n, 4),  # the tic's own decision
                "stale_1_share": round(lags[1] / n, 4),
                "stale_2plus_share": round(
                    sum(v for k, v in lags.items() if k >= 2) / n, 4
                ),
                "skipped_share": round(g.skipped / max(1, g.skipped + g.decided), 4),
                "overrun_tics": s.get("overrun_tics"),
                "lines": g.lines,
            }
        )
    keys = (
        "frags",
        "deaths",
        "margin",
        "decision_ms_p50",
        "decision_ms_p99",
        "fresh_share",
    )
    summary = {
        k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3)
        for k in keys
    }
    summary["top_share"] = round(float(np.mean([r["rank"] == 1 for r in rows])), 3)
    summary["lines_per_min"] = round(
        sum(len(r["lines"]) for r in rows) / (args.games * args.seconds / 60), 2
    )
    return {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "placement": pol.kit.placement,
        "summary": summary,
        "games": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="Model and games on this node")
    serve_p = sub.add_parser("serve", help="Model here; games connect over TCP")
    play_p = sub.add_parser("play", help="Games here, against a `serve` node")
    for p in (run_p, serve_p, play_p):
        p.add_argument("--games", type=int, default=1)
        p.add_argument("--seconds", type=float, default=120.0)
        p.add_argument("--bots", default="default")
        p.add_argument("--n-bots", type=int, default=7)
        p.add_argument("--seed", type=int, default=0)
    for p in (run_p, serve_p):
        p.add_argument("--model", required=True)
        p.add_argument("--talk-every", type=float, default=4.0)
        p.add_argument("--no-talk", action="store_true")
        p.add_argument("--temperature", type=float, default=1.0, help="Style, planner")
        p.add_argument("--gpu-mem", type=float, default=0.5)
        p.add_argument("--json", type=Path)
    for p in (serve_p, play_p):
        p.add_argument(
            "--addr-file", type=Path, required=True, help="Shared file: host port"
        )
    serve_p.add_argument("--port", type=int, default=0, help="0: any free port")
    play_p.add_argument("--wait-s", type=float, default=1800.0)
    args = ap.parse_args()
    if args.cmd == "play":
        play(args)
        return
    res = asyncio.run(serve(args))
    for r in res["games"]:
        print(
            f"game {r['game']}: frags {r['frags']} deaths {r['deaths']} margin "
            f"{r['margin']} rank {r['rank']} | decision p50 {r['decision_ms_p50']} ms "
            f"p99 {r['decision_ms_p99']} ms | own decision in time {100 * r['fresh_share']:.1f}% "
            f"| {len(r['lines'])} lines",
            flush=True,
        )
        for t, line, ms in r["lines"][:4]:
            print(f"    t{t} ({ms} ms): {line}")
    print("SUMMARY", json.dumps(res["summary"]))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
