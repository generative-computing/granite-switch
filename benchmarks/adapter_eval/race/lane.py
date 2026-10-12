# SPDX-License-Identifier: Apache-2.0
"""One lane of the agent race: a fleet of agents through a fixed number of slots, recorded.

The switch benchmark's agent loop runs unchanged (``run_agent`` of
``benchmarks/bench_agent_sim.py``, with its engine, KV gate and prefix-cache
reset): each agent walks its schedule of segments, switching adapter every
``span`` tokens, with tool calls and round trips as real sleeps. Two things
differ from its grid driver, both around that loop:

* Slots instead of waves. The grid releases its agents in waves of
  ``concurrency`` and starts a wave only when the last one has ended; here an
  agent starts the moment any running agent finishes, so ``concurrency`` agents
  run until the fleet runs out.
* A recording engine. Every request the loop issues goes through
  ``RecordingEngine``, which notes its start, first token, tokens and end, and
  what the engine re-prefilled. The gaps between an agent's requests are its
  tool calls and round trips, named from its schedule (``units_of``).

The fleet runs ``--warmup-runs`` times untimed first, as the benchmark's
switching block does, so adapters are loaded and kernels compiled; the prefix
cache is reset before every run.

    python -m benchmarks.adapter_eval.race.lane --lane gs-sr-vllm \\
        --arm shadow-residual --sr-checkpoint <dir> --fleet <grid.jsonl> --out <lane.json>
    python -m benchmarks.adapter_eval.race.lane --lane lora-vllm \\
        --arm native-lora --base-model <id> --lora-root <dir> --fleet <grid.jsonl> --out <lane.json>
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import pathlib
import re
import sys
import time

from benchmarks.bench_agent_sim import (
    _make_engine,
    check_kv_budget,
    kv_capacity,
    run_agent,
)
from benchmarks.bench_switching_grid import reset_prefix_cache

# How often the lane's metrics are sampled, and how often an agent's progress is
# recorded: every PROGRESS_EVERY generated tokens (one character of the page's bar).
METRICS_EVERY = 0.5
PROGRESS_EVERY = 16
# Requests whose time to first token is averaged in the metrics.
TTFT_WINDOW = 8
REQUEST_ID = re.compile(r"-a(\d+)-u(\d+)$")


def units_of(agent: dict, arm: str) -> list[dict]:
    """The units ``run_agent`` issues for ``agent``, in order: its own flattening of the
    schedule, mirrored here so the gaps between requests can be named.

    native-lora issues a request per segment; shadow-residual one per run between
    tool calls. ``tests/unit/test_agent_race.py`` checks this against the requests
    ``run_agent`` actually issues.
    """
    units: list[dict] = []
    if arm == "native-lora":
        for node in agent["nodes"]:
            if node["tool"]:
                units.append(dict(kind="tool"))
                continue
            for d in node["segments_detail"]:
                units.append(
                    dict(
                        kind="gen",
                        tokens=d["tokens"],
                        adapter=d["adapter"],
                        switch=d["switch"],
                    )
                )
        return units
    run = 0
    for node in agent["nodes"]:
        if node["tool"]:
            if run:
                units.append(dict(kind="gen", tokens=run, adapter=None, switch=False))
                run = 0
            units.append(dict(kind="tool"))
        else:
            run += node["span"]
    if run:
        units.append(dict(kind="gen", tokens=run, adapter=None, switch=False))
    return units


def segments_of(agent: dict) -> list[dict]:
    """The agent's segments as its schedule names them, in token order: where each
    ends, its adapter, and whether the adapter changes after it."""
    out, at = [], 0
    for unit in units_of(agent, "native-lora"):
        if unit["kind"] == "gen":
            at += unit["tokens"]
            out.append(dict(end=at, adapter=unit["adapter"], switch=unit["switch"]))
    return out


def gap_after(units: list[dict], ui: int, arm: str) -> str | None:
    """What an agent waits on after its request ``ui``: a round trip (native-lora,
    after a switch), a tool call, both, or nothing."""
    parts = []
    if arm == "native-lora" and units[ui].get("switch"):
        parts.append("rtt")
    if ui + 1 < len(units) and units[ui + 1]["kind"] == "tool":
        parts.append("tool")
    return "+".join(parts) or None


class Recorder:
    """The lane's events, in seconds from the timed run's start, and its metrics."""

    def __init__(self, lane: str, agents: list[dict], arm: str, slots: int) -> None:
        self.lane, self.arm, self.slots = lane, arm, slots
        ordered = sorted(agents, key=lambda a: a["agent_id"])
        # The page's row of each agent: its place in the order agents start.
        self.row = {a["agent_id"]: k for k, a in enumerate(ordered)}
        self.units = {a["agent_id"]: units_of(a, arm) for a in ordered}
        # Each agent's segments, as its schedule names them. Stock vLLM switches
        # adapter at their ends, with a new request. SR's switch is not
        # implemented, and would cost nothing on the GPU (its K/V come from the
        # base model), so its generation runs on: its adapter is shown changing
        # on the same schedule, and the switches counted as for stock vLLM.
        self.segments = {a["agent_id"]: segments_of(a) for a in ordered}
        self.state = {
            a["agent_id"]: dict(
                status="queued", phase=None, tokens=0, shown=0, seg=0, req_end=0
            )
            for a in ordered
        }
        self.switches = self.requests = 0
        self.events: list[dict] = []
        self.t0 = time.perf_counter()
        self.started_at: dict[tuple[int, int], float] = {}
        self.ttfts: collections.deque = collections.deque(maxlen=TTFT_WINDOW)
        self.generated = 0
        self.prompt_tokens = self.cached_tokens = self.recomputed_tokens = 0
        # An agent's first request prefills its own prompt, which nothing could
        # have cached; every later one sends that context again, and what the
        # cache did not hold of it is the re-prefill.
        self.first_seen: set[int] = set()
        self.first_prefill = 0
        self.resent_tokens = self.resent_cached = 0
        self._last = (0.0, 0)  # (time, generated) at the previous sample

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def emit(self, ev: str, agent_id: int | None = None, **fields) -> None:
        event = {"t": round(self.now(), 4), "ev": ev, "srv": self.lane}
        if agent_id is not None:
            event["conv"] = self.row[agent_id]
        self.events.append({**event, **fields})

    def agent_start(self, agent_id: int, slot: int) -> None:
        self.state[agent_id]["status"] = "running"
        self.emit("conv_start", agent_id, agent=agent_id, slot=slot)
        units = self.units[agent_id]
        if units and units[0]["kind"] == "tool":
            self._phase(agent_id, "tool")

    def _phase(self, agent_id: int, phase: str | None, **fields) -> None:
        self.state[agent_id]["phase"] = phase
        if phase:
            self.emit("step_start", agent_id, step=phase, **fields)

    def _shown_adapter(self, agent_id: int) -> str | None:
        """The adapter the schedule gives the agent's next token."""
        segs, seg = self.segments[agent_id], self.state[agent_id]["seg"]
        return segs[seg]["adapter"] if seg < len(segs) else None

    def _switch(self, agent_id: int, adapter: str | None, to: str | None) -> None:
        self.switches += 1
        self.emit("switch", agent_id, adapter=adapter, to=to)

    def request_start(
        self,
        agent_id: int,
        ui: int,
        adapter: str | None,
        prompt_tokens: int,
        n_tokens: int = 0,
    ) -> None:
        st = self.state[agent_id]
        st["req_end"] = st["tokens"] + n_tokens
        if self.arm == "shadow-residual":
            adapter = self._shown_adapter(agent_id)
        self.started_at[agent_id, ui] = self.now()
        self._phase(
            agent_id, "prefill", unit=ui, adapter=adapter, prompt_tokens=prompt_tokens
        )

    def first_token(self, agent_id: int, ui: int) -> None:
        ttft = self.now() - self.started_at.pop((agent_id, ui), self.now())
        self.ttfts.append(ttft)
        self._phase(agent_id, "decode", unit=ui, ttft=round(ttft, 4))

    def tokens(self, agent_id: int, delta: int) -> None:
        st = self.state[agent_id]
        st["tokens"] += delta
        self.generated += delta
        if st["tokens"] // PROGRESS_EVERY > st["shown"] // PROGRESS_EVERY:
            st["shown"] = st["tokens"]
            self.emit("progress", agent_id, tokens=st["tokens"])
        if self.arm != "shadow-residual":
            return
        segs = self.segments[agent_id]
        while st["seg"] < len(segs) and st["tokens"] >= segs[st["seg"]]["end"]:
            ended = segs[st["seg"]]
            st["seg"] += 1
            if ended["switch"]:
                self._switch(agent_id, ended["adapter"], self._shown_adapter(agent_id))
            if st["seg"] < len(segs) and st["tokens"] < st["req_end"]:
                self._phase(agent_id, "decode", adapter=self._shown_adapter(agent_id))

    def request_end(
        self, agent_id: int, ui: int, prompt_tokens: int, cached: int | None
    ) -> None:
        st = self.state[agent_id]
        if st["shown"] != st["tokens"]:
            st["shown"] = st["tokens"]
            self.emit("progress", agent_id, tokens=st["tokens"])
        first = agent_id not in self.first_seen
        self.first_seen.add(agent_id)
        if cached is not None:
            self.prompt_tokens += prompt_tokens
            self.cached_tokens += cached
            self.recomputed_tokens += prompt_tokens - cached
            if first:
                self.first_prefill += prompt_tokens - cached
            else:
                self.resent_tokens += prompt_tokens
                self.resent_cached += cached
        self.requests += 1
        self.emit(
            "req_done", agent_id, unit=ui, prompt_tokens=prompt_tokens, cached=cached
        )
        units = self.units[agent_id]
        if self.arm == "native-lora" and units[ui]["switch"]:
            nxt = next((u for u in units[ui + 1 :] if u["kind"] == "gen"), None)
            self._switch(agent_id, units[ui]["adapter"], nxt and nxt["adapter"])
        self._phase(agent_id, gap_after(units, ui, self.arm))

    def agent_done(self, agent_id: int, elapsed: float) -> None:
        self.state[agent_id].update(status="done", phase=None)
        self.emit("conv_done", agent_id, agent=agent_id, wall_time=round(elapsed, 4))

    def snapshot(self) -> dict:
        """The lane's metrics now: what its agents are doing, and the engine's work."""
        t, generated = self.now(), self.generated
        last_t, last_generated = self._last
        self._last = (t, generated)
        running = [s for s in self.state.values() if s["status"] == "running"]

        def doing(*phases):
            return sum(1 for s in running if (s["phase"] or "") in phases)

        return {
            "running": len(running),
            "queued": sum(1 for s in self.state.values() if s["status"] == "queued"),
            "done": sum(1 for s in self.state.values() if s["status"] == "done"),
            "prefill": doing("prefill"),
            "decode": doing("decode"),
            "waiting": doing("rtt", "tool", "rtt+tool"),
            "tok_s": round((generated - last_generated) / (t - last_t), 1)
            if t > last_t
            else 0.0,
            "ttft_avg": round(sum(self.ttfts) / len(self.ttfts), 4)
            if self.ttfts
            else None,
            "switches": self.switches,
            "requests": self.requests,
            "first_prefill": self.first_prefill,
            "reprefilled": self.resent_tokens - self.resent_cached,
            "context_hit": round(self.resent_cached / self.resent_tokens, 4)
            if self.resent_tokens
            else None,
            "generated": generated,
        }

    async def sample(self, every: float) -> None:
        while True:
            await asyncio.sleep(every)
            self.emit("metrics", m=self.snapshot())


class RecordingEngine:
    """The engine, with every request the agent loop issues recorded on its way through."""

    def __init__(self, engine, recorder: Recorder, adapter_of) -> None:
        self._engine = engine
        self._rec = recorder
        self._adapter_of = adapter_of
        self.adapters_run: set[str] = set()

    def generate(self, prompt, params, request_id, **kw):
        found = REQUEST_ID.search(request_id)
        if not found:
            raise ValueError(f"unexpected request id {request_id!r}")
        agent_id, ui = int(found[1]), int(found[2])
        ids = prompt["prompt_token_ids"]
        lora = kw.get("lora_request")
        adapter = lora.lora_name if lora is not None else self._adapter_of(ids)
        if adapter:
            self.adapters_run.add(adapter)
        stream = self._engine.generate(prompt, params, request_id, **kw)
        return self._stream(
            stream, agent_id, ui, adapter, len(ids), getattr(params, "max_tokens", 0)
        )

    async def _stream(self, stream, agent_id, ui, adapter, prompt_tokens, n_tokens):
        rec = self._rec
        rec.request_start(agent_id, ui, adapter, prompt_tokens, n_tokens)
        seen, out = 0, None
        async for out in stream:
            got = len(out.outputs[0].token_ids)
            if got > seen:
                if seen == 0:
                    rec.first_token(agent_id, ui)
                rec.tokens(agent_id, got - seen)
                seen = got
            yield out
        cached = getattr(out, "num_cached_tokens", None) if out is not None else None
        rec.request_end(agent_id, ui, prompt_tokens, cached)

    def __getattr__(self, name):
        return getattr(self._engine, name)


async def run_slots(
    engine, flavour, agents, arm, args, lora_for, slots: int, recorder=None
) -> list[dict]:
    """Run ``agents`` in id order through ``slots`` slots: each slot takes the next
    agent as soon as its own has finished. Returns ``run_agent``'s rows."""
    pending = collections.deque(sorted(agents, key=lambda a: a["agent_id"]))
    rows: list[dict] = []

    async def slot(k: int) -> None:
        while pending:
            agent = pending.popleft()
            if recorder is not None:
                recorder.agent_start(agent["agent_id"], k)
            await run_agent(engine, flavour, agent, arm, args, lora_for, rows)
            if recorder is not None:
                row = next(r for r in rows if r["agent_id"] == agent["agent_id"])
                recorder.agent_done(agent["agent_id"], row["elapsed_s"])

    await asyncio.gather(*(slot(k) for k in range(min(slots, len(pending)))))
    return rows


async def reset_cache(engine) -> None:
    ok = reset_prefix_cache(engine)
    if asyncio.iscoroutine(ok):
        ok = await ok or True
    if not ok:
        raise SystemExit("FATAL: could not reset the prefix cache between runs")


async def race_lane(engine, flavour, agents, arm, args, lora_for, adapter_of) -> dict:
    """The lane's warm-up runs, then its recorded run: its events and its agents' rows."""
    warmups = []
    for _ in range(args.warmup_runs):
        await reset_cache(engine)
        t = time.perf_counter()
        await run_slots(engine, flavour, agents, arm, args, lora_for, args.concurrency)
        warmups.append(round(time.perf_counter() - t, 2))
    await reset_cache(engine)
    recorder = Recorder(args.lane, agents, arm, args.concurrency)
    proxy = RecordingEngine(engine, recorder, adapter_of)
    recorder.start()
    sampler = asyncio.create_task(recorder.sample(args.metrics_every))
    try:
        rows = await run_slots(
            proxy, flavour, agents, arm, args, lora_for, args.concurrency, recorder
        )
    finally:
        sampler.cancel()
    wall = recorder.now()
    recorder.emit("metrics", m=recorder.snapshot())
    return {
        "lane": args.lane,
        "arm": arm,
        "concurrency": args.concurrency,
        "agents": len(agents),
        "wall_s": round(wall, 4),
        "warmup_wall_s": warmups,
        "generated": recorder.generated,
        "prompt_tokens": recorder.prompt_tokens,
        "recomputed_tokens": recorder.recomputed_tokens,
        "cached_tokens": recorder.cached_tokens,
        "switches": recorder.switches,
        "requests": recorder.requests,
        # The adapters the engine actually ran: for SR, the one every agent kept.
        "adapters_run": sorted(proxy.adapters_run),
        "first_prefill_tokens": recorder.first_prefill,
        "resent_tokens": recorder.resent_tokens,
        "resent_cached_tokens": recorder.resent_cached,
        "reprefilled_tokens": recorder.resent_tokens - recorder.resent_cached,
        "rows": sorted(rows, key=lambda r: r["agent_id"]),
        "events": recorder.events,
    }


def lora_resolver(root: pathlib.Path, needed: set[int]):
    """``lora_for`` for native-lora, as the switch benchmark's grid driver resolves its
    adapters: ``adapter_NN`` folders under ``root``, one LoRARequest per name."""
    paths: dict[int, str] = {}
    for d in sorted(root.glob("adapter_*")):
        m = re.fullmatch(r"adapter_(\d+)", d.name)
        if not m:
            continue
        cfgs = (
            sorted(d.glob("*/*/adapter_config.json"))
            or sorted(d.glob("*/adapter_config.json"))
            or sorted(d.glob("adapter_config.json"))
        )
        if cfgs:
            paths[int(m.group(1))] = str(cfgs[0].parent)
    missing = sorted(needed - set(paths))
    if missing:
        raise SystemExit(
            f"FATAL: the fleet uses {len(needed)} adapters, {len(missing)} have no "
            f"PEFT folder under {root}: {missing[:10]}"
        )
    cache: dict[str, object] = {}

    def lora_for(name):
        if name in (None, "base"):
            return None
        if name not in cache:
            from vllm.lora.request import LoRARequest

            idx = int(name[2:])
            cache[name] = LoRARequest(name, idx + 1, paths[idx])
        return cache[name]

    return lora_for


def sr_adapters(checkpoint: pathlib.Path) -> dict[int, str]:
    """A composed checkpoint's adapters by control token, named as the fleet names them."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    names = cfg.get("adapter_names") or []
    ids = cfg.get("adapter_token_ids") or []
    if not ids or len(names) != len(ids):
        raise SystemExit(f"FATAL: {checkpoint}/config.json names no control tokens")
    short = {}
    for name, token in zip(names, ids, strict=True):
        m = re.fullmatch(r"adapter_(\d+)", name)
        short[token] = f"ad{int(m.group(1))}" if m else name
    return short


def load_fleet(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def adapters_used(agents: list[dict]) -> set[int]:
    return {
        int(d["adapter"][2:])
        for a in agents
        for n in a["nodes"]
        for d in n["segments_detail"]
        if d["adapter"] not in (None, "base")
    }


def environment() -> dict:
    """The GPU and library versions the lane ran on."""
    out = {}
    try:
        import torch

        out["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
        out["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import vllm

        out["vllm"] = vllm.__version__
    except Exception:
        pass
    return out


async def amain(args) -> None:
    agents = load_fleet(pathlib.Path(args.fleet))
    longest = max(a["prompt_tokens"] + a["decode_tokens"] for a in agents)
    if longest >= args.max_model_len:
        raise SystemExit(
            f"FATAL: an agent needs {longest:,} tokens of context but --max-model-len "
            f"is {args.max_model_len:,}"
        )
    if args.arm == "shadow-residual":
        checkpoint = pathlib.Path(args.sr_checkpoint)
        names = sr_adapters(checkpoint)
        if not args.sr_control_id:
            ids = list(names)
            args.sr_control_id = ids[args.sr_adapter_index % len(ids)]
        model, lora_for = str(checkpoint), (lambda name: None)

        def adapter_of(ids):
            return names.get(ids[0], "?")
    else:
        model = args.base_model
        lora_for = lora_resolver(pathlib.Path(args.lora_root), adapters_used(agents))

        def adapter_of(ids):
            return None

    engine, flavour = _make_engine(model, args)
    cap = kv_capacity(engine)
    print(
        f"KV capacity: {cap:,} tokens" if cap else "KV capacity: unreadable",
        file=sys.stderr,
    )
    # At most `concurrency` agents run at once: the largest of them bound the KV.
    need = sorted(agents, key=lambda a: a["prompt_tokens"] + a["decode_tokens"])
    check_kv_budget(
        engine,
        need[-args.concurrency :],
        args.kv_fraction,
        args.allow_kv_overcommit,
    )
    report = await race_lane(
        engine, flavour, agents, args.arm, args, lora_for, adapter_of
    )
    report["kv_capacity"] = cap
    report.update(environment())
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report))
    print(
        f"[race] {args.lane}: {len(agents)} agents through {args.concurrency} slots in "
        f"{report['wall_s']:.1f}s (warm-up {report['warmup_wall_s']}), "
        f"first prefill {report['first_prefill_tokens']:,} tokens, "
        f"re-prefilled {report['reprefilled_tokens']:,}",
        file=sys.stderr,
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lane", required=True, help="the lane's name on the page")
    ap.add_argument("--arm", required=True, choices=["native-lora", "shadow-residual"])
    ap.add_argument("--fleet", required=True, help="the workload: a grid file's agents")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--metrics-every", type=float, default=METRICS_EVERY)
    # The switch benchmark's engine and agent settings, with its grid driver's defaults.
    ap.add_argument("--base-model", default="")
    ap.add_argument("--sr-checkpoint", default="")
    ap.add_argument("--lora-root", default="")
    ap.add_argument("--sr-control-id", type=int, default=0)
    ap.add_argument("--sr-adapter-index", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--max-loras", type=int, default=32)
    ap.add_argument("--max-cpu-loras", type=int, default=64)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=30000)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--kv-fraction", type=float, default=0.85)
    ap.add_argument("--allow-kv-overcommit", action="store_true")
    ap.add_argument("--sleep-dist", choices=("log", "uniform"), default="uniform")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    if args.arm == "shadow-residual" and not args.sr_checkpoint:
        ap.error("--arm shadow-residual needs --sr-checkpoint")
    if args.arm == "native-lora" and not (args.base_model and args.lora_root):
        ap.error("--arm native-lora needs --base-model and --lora-root")
    asyncio.run(amain(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
