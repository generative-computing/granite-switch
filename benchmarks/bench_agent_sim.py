#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copied from the switch benchmark (granite-switch-staging, branch
# feature/switch-benchmark, commit 5a51e57) and kept unchanged, so this benchmark
# measures switching exactly as it does. Update it only by copying it again.
"""Multi-agent ReAct simulation: native-lora against shadow-residual, one wave, one arm.

Design and rationale: docs/AGENT_SIM_DESIGN.html. Fleet: scratch/gen_agent_configs.py.

WHAT THIS DRIVER IS

An OPEN-LOOP race. `--concurrent` agents are released together as coroutines against one in-process
AsyncLLMEngine, and each agent walks its own ReAct schedule from the fleet file. Nothing is batched by
this driver -- the engine's continuous batching decides what runs together, which is the point: while
one agent sleeps on a tool call or a network round trip, the GPU serves the others. That interaction is
the whole reason to run a fleet rather than a fixed batch, and it only exists because the sleeps are
real.

THE TWO ARMS

  native-lora       A NEW REQUEST per segment. vLLM binds a LoRA per request (LoRARequest), so this arm
                    cannot change adapter inside a generation -- closing and re-issuing is the only
                    option, and it is an architectural constraint rather than a handicap imposed here.
                    Each new request re-prefills its prompt: the original prompt plus everything
                    generated so far. With prefix caching on, whether that costs anything depends on
                    the adapter: vLLM keys its prefix cache on the LoRA name, so a FIRST visit to an
                    adapter recomputes the whole prefix while a REVISIT recomputes only what grew since
                    that adapter's own previous turn.

  shadow-residual   One request per inter-tool RUN. The control token sits in the prompt
                    (ARMS["gs-sr-vllm"]["prompt_control"]) and one adapter serves the whole task, so
                    adapter changes cost nothing and need no boundary. It still stops at tool nodes,
                    exactly like native, so the unavoidable boundaries are identical between the arms
                    and the measured difference is the ADAPTER-SWITCH boundaries alone.

                    SCOPE: this is the floor SR's cost model predicts, not a measured switch. It never
                    switches. An implemented SR switch would add routing work this arm does not do.

WHY EACH AGENT GETS ITS OWN PROMPT CONTENT

Not just its own prompt LENGTH. Prefix caching is content-addressed, so if two agents shared filler the
first would populate the cache and the rest would hit it -- handing the re-prefilling arm back most of
the cost being measured. bench_switch_emission.py carries the same warning for the same reason ("ONE
FILLER PER BATCH SLOT, distinct"), where a shared filler returned (B-1)/B of it. Filler here is seeded
per agent_id, so it is distinct across agents AND identical across the two arms.

THE STOPWATCH

Elapsed wall clock is recorded when an agent's cumulative generated tokens cross each decile of its
total. Output is streamed token-by-token rather than awaited per request, because for
shadow-residual a decile can fall INSIDE a request -- it generates a whole inter-tool run in one call.
The fleet generator guarantees a span boundary exactly on every decile, so native's marks land on
request boundaries anyway; streaming just makes both arms measure the same way.

UNTESTED WITHOUT A GPU. The engine import is version-sensitive (v0 AsyncLLMEngine vs v1 AsyncLLM);
_make_engine handles both and fails loudly rather than silently degrading.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import pathlib
import random
import re
import sys
import time

# --- RTT categories: EFFECTIVE column of rtt_table.tsv, milliseconds -----------------------------
# Effective, not raw: an LLM call over a network pays serialization and HTTP, and cat 6 pays a gateway.
# Log-uniform within the range -- a 100-500ms range spans 5x, and uniform sampling would overweight the
# slow end.
RTT_MS = {
    1: (0.2, 1.0),
    2: (1.0, 5.0),
    3: (2.0, 10.0),
    4: (5.0, 30.0),
    5: (50.0, 200.0),
    6: (100.0, 500.0),
}
TOOL_MS = {
    "calculator": (1.0, 10.0),
    "sql_lookup": (5.0, 50.0),
    "rag_no_rerank": (20.0, 80.0),
}


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    """A draw that is uniform in log space, so each octave of the range is equally likely."""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def draw_ms(rng: random.Random, lo: float, hi: float, dist: str) -> float:
    """A latency draw, log-uniform or uniform over [lo, hi].

    Which one barely matters and it is worth knowing why. TOOL sleeps are arm-neutral -- both arms stop at
    the same tool boundaries and sleep the same drawn sequence -- so the distribution cancels in the ratio.
    The switch round trip is native-only, and there uniform over 2-10 ms gives a mean of 6 ms against
    log-uniform's 4.7, so uniform is marginally conservative for shadow-residual. In the agent run the
    sleeps were 0.04% of native's elapsed time either way.
    """
    return rng.uniform(lo, hi) if dist == "uniform" else log_uniform(rng, lo, hi)


def filler_ids(agent_id: int, n: int, vocab_lo: int, vocab_hi: int) -> list[int]:
    """`n` token ids unique to this agent.

    Seeded on agent_id alone, so the two arms replay byte-identical prompts while no two agents share a
    prefix. Deliberately NOT text: tokenizing a paragraph would give a length this driver cannot control,
    and the forced decode lengths depend on exact counts.
    """
    rng = random.Random(0xA6E17 ^ agent_id)
    return [rng.randrange(vocab_lo, vocab_hi) for _ in range(n)]


class Stopwatch:
    """Records elapsed time as an agent's cumulative generated tokens cross each decile."""

    def __init__(self, total: int, marks: int, t0: float, on_mark=None) -> None:
        self.t0 = t0
        self.total = total
        self.marks = [total * (i + 1) // marks for i in range(marks)]
        self.hit: dict[int, float] = {}
        self.on_mark = on_mark

    def observe(self, cumulative: int) -> None:
        # Marks are ascending; record every one now satisfied. A single streamed chunk can cross more
        # than one mark when spans are short, so this loops rather than checking only the next.
        for m in self.marks:
            if m not in self.hit and cumulative >= m:
                self.hit[m] = time.perf_counter() - self.t0
                if self.on_mark is not None:
                    self.on_mark(m, self.hit[m])

    def as_dict(self, total: int) -> dict[str, float]:
        return {
            f"p{round(100 * m / total)}": round(self.hit[m], 4)
            for m in self.marks
            if m in self.hit
        }


def kv_capacity(engine) -> int | None:
    """Tokens of GPU KV cache this engine actually allocated, or None if it cannot be read.

    Introspection, not a constant: the 667,632-token figure this study sizes fleets against was measured
    with a TWELVE-adapter checkpoint, and adapter weights come out of the same budget, so it shrinks as M
    grows. The attribute path has moved between vLLM versions and between the v0 and v1 engines, so
    several are tried and None is returned rather than guessing.
    """
    for path in (
        ("llm_engine", "cache_config"),
        ("engine", "cache_config"),
        ("cache_config",),
        ("vllm_config", "cache_config"),
        ("engine_core", "cache_config"),
    ):
        obj = engine
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        blocks = getattr(obj, "num_gpu_blocks", None)
        size = getattr(obj, "block_size", None)
        if blocks and size:
            return int(blocks) * int(size)
    return None


def check_kv_budget(engine, wave, fraction: float, allow_overcommit: bool) -> None:
    """Refuse to run if the wave's peak KV demand would push the engine into preemption.

    Past roughly `fraction` of capacity vLLM PREEMPTS AND SWAPS rather than failing, so an over-committed
    wave does not announce itself -- it returns timings that are inflated by an amount nobody can
    attribute. That is the one failure mode this design cannot detect from its own output, which is why it
    is checked before any work is done.

    If capacity cannot be read, this warns and proceeds: blocking a run on an introspection failure would
    be worse than relying on the preemption warnings the engine logs, which the job's watcher greps for.
    """
    need = sum(a["prompt_tokens"] + a["decode_tokens"] for a in wave)
    cap = kv_capacity(engine)
    if cap is None:
        print(
            f"WARNING: could not read KV capacity from this engine; wave needs {need:,} tokens. "
            f"Proceeding -- watch the log for preemption warnings, which are the fallback signal.",
            file=sys.stderr,
        )
        return
    print(
        f"KV: wave needs {need:,} tokens of a measured {cap:,} ({need / cap:.0%}); "
        f"the 12-adapter reference was 667,632",
        file=sys.stderr,
    )
    if need > fraction * cap:
        msg = (
            f"FATAL: wave peak {need:,} tokens is {need / cap:.0%} of the {cap:,} this engine "
            f"allocated, above the {fraction:.0%} at which vLLM preempts and swaps rather than "
            f"failing. Every timing would be inflated silently. Raise --waves in "
            f"gen_agent_configs.py and regenerate the fleet, or pass --allow-kv-overcommit "
            f"to measure anyway knowing the numbers are not comparable."
        )
        if not allow_overcommit:
            raise SystemExit(msg)
        print(msg.replace("FATAL", "OVERCOMMITTED (allowed)"), file=sys.stderr)


def _make_engine(model: str, args) -> tuple[object, str]:
    """Build an in-process async engine, tolerating the v0/v1 split.

    Returns (engine, flavour). Fails loudly: a driver that silently fell back to a synchronous engine
    would serialise the agents and measure nothing the design asks about.
    """
    from vllm import AsyncEngineArgs

    ea = AsyncEngineArgs(
        model=model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,  # ON: realistic, and conservative for shadow-residual
        enable_lora=args.arm == "native-lora",
        max_loras=args.max_loras,
        max_cpu_loras=args.max_cpu_loras,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,  # keep stats: the prefix-cache hit rate is a result here
    )
    try:
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(ea), "v1"
    except Exception:
        pass
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        return AsyncLLMEngine.from_engine_args(ea), "v0"
    except Exception as exc:
        raise SystemExit(
            f"FATAL: no usable async engine in this vLLM build: {exc}"
        ) from exc


async def run_request(
    engine,
    flavour,
    prompt_ids,
    n_tokens,
    req_id,
    lora_request,
    on_tokens,
    counters=None,
):
    """Generate exactly `n_tokens`, streaming so the stopwatch sees each step.

    min_tokens == max_tokens: the schedule's span lengths are the experiment, so a sequence that stopped
    early would silently shorten the task. Detokenization is off -- nothing here reads text.

    Also accumulates what the ENGINE says it re-prefilled, into `counters`. The fleet PREDICTS native's
    recompute from the name-keyed cache rule (92x the useful output at M=64); num_cached_tokens is vLLM's
    own accounting of what it did not have to recompute, so the two can be compared instead of the
    prediction being taken on trust. Without it the figure shows a consequence with no evidence of the
    cause -- and the whole claim is about the cause. `num_cached_tokens` may be absent on older builds, in
    which case nothing is recorded rather than a zero being invented, since a zero here would read as
    "the cache never helped", which is the shape of the result being claimed.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(
        temperature=0.0, min_tokens=n_tokens, max_tokens=n_tokens, detokenize=False
    )
    kw = {"lora_request": lora_request} if lora_request is not None else {}
    seen = 0
    async for out in engine.generate(
        TokensPrompt(prompt_token_ids=prompt_ids), sp, req_id, **kw
    ):
        got = len(out.outputs[0].token_ids)
        if got > seen:
            on_tokens(got - seen)
            seen = got
    if seen != n_tokens:
        raise RuntimeError(
            f"{req_id}: generated {seen} of {n_tokens} -- forced length not honoured"
        )
    if counters is not None:
        cached = getattr(out, "num_cached_tokens", None)
        if cached is None:
            counters["cache_counter_absent"] = 1
        else:
            counters["prompt_tokens_submitted"] += len(prompt_ids)
            counters["prefill_cached"] += int(cached)
            counters["prefill_recomputed"] += len(prompt_ids) - int(cached)
    return list(out.outputs[0].token_ids)


async def run_agent(engine, flavour, agent, arm, args, lora_for, results):
    """Walk one agent's ReAct schedule, recording its decile timings."""
    rng = random.Random(0xC0FFEE ^ agent["agent_id"])
    cat = agent["rtt_category"]
    prompt = filler_ids(
        agent["agent_id"], agent["prompt_tokens"], args.vocab_lo, args.vocab_hi
    )
    if arm == "shadow-residual":
        # prompt_control: the adapter's control token leads the prompt and never changes.
        prompt = [args.sr_control_id, *prompt]
    total = agent["decode_tokens"]

    generated: list[int] = []
    cum = 0
    t0 = time.perf_counter()

    # Progress on every DECILE CROSSING, not only on completion. All 20 agents in a wave are released
    # together and finish within minutes of each other, so completion-only logging reads "0 finished" for
    # over an hour and then everything at once -- which is the same blindness it was added to remove. Ten
    # lines per agent, in the streaming path but only on a crossing, so it cannot perturb the schedule.
    def mark_reached(tokens, elapsed):
        print(
            f"  agent {agent['agent_id']:3d}  {100 * tokens // total:3d}%  {elapsed / 60:7.2f} min",
            file=sys.stderr,
            flush=True,
        )

    watch = Stopwatch(total, args.deciles, t0, on_mark=mark_reached)
    counters = dict(
        requests=0,
        switch_sleeps=0.0,
        tool_sleeps=0.0,
        switches=0,
        tool_calls=0,
        # What the engine re-prefilled, from its own accounting rather than from the fleet's prediction.
        prompt_tokens_submitted=0,
        prefill_cached=0,
        prefill_recomputed=0,
    )

    def on_tokens(delta):
        nonlocal cum
        cum += delta
        watch.observe(cum)

    # Flatten the schedule into the units each arm actually issues.
    #   native          one request per SEGMENT (it cannot switch inside a generation)
    #   shadow-residual one request per inter-tool RUN (it can, and only tools force a boundary)
    units: list[dict] = []
    if arm == "native-lora":
        for node in agent["nodes"]:
            if node["tool"]:
                units.append(dict(kind="tool", node=node))
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
    else:
        run = 0
        for node in agent["nodes"]:
            if node["tool"]:
                if run:
                    units.append(
                        dict(kind="gen", tokens=run, adapter=None, switch=False)
                    )
                    run = 0
                units.append(dict(kind="tool", node=node))
            else:
                run += node["span"]
        if run:
            units.append(dict(kind="gen", tokens=run, adapter=None, switch=False))

    tool_i = 0
    for ui, unit in enumerate(units):
        if unit["kind"] == "tool":
            # A tool call: a boundary and a latency, paid IDENTICALLY by both arms. The tool is drawn
            # from this agent's toolbox by the fleet file, so both arms sleep the same sequence.
            idx = (
                agent["tool_draws"][tool_i] if tool_i < len(agent["tool_draws"]) else 0
            )
            tool_i += 1
            lo, hi = TOOL_MS[agent["toolbox"][idx]["category"]]
            ms = draw_ms(rng, lo, hi, args.sleep_dist)
            counters["tool_sleeps"] += ms / 1000.0
            counters["tool_calls"] += 1
            await asyncio.sleep(ms / 1000.0)
            continue

        lora = lora_for(unit["adapter"]) if arm == "native-lora" else None
        generated += await run_request(
            engine,
            flavour,
            prompt + generated,
            unit["tokens"],
            f"{arm}-a{agent['agent_id']}-u{ui}",
            lora,
            on_tokens,
            counters,
        )
        counters["requests"] += 1

        # The round trip back to software. REAL sleep, not added afterwards: the GPU must be free for
        # other agents while this one waits, or the race is arithmetic instead of a race.
        # shadow-residual pays this only at tool nodes -- it has no other boundaries.
        if arm == "native-lora" and unit["switch"]:
            ms = draw_ms(rng, *RTT_MS[cat], args.sleep_dist)
            counters["switch_sleeps"] += ms / 1000.0
            counters["switches"] += 1
            await asyncio.sleep(ms / 1000.0)

    results.append(
        dict(
            arm=arm,
            agent_id=agent["agent_id"],
            wave=agent["wave"],
            rtt_category=cat,
            cycle=agent["cycle"],
            prompt_tokens=agent["prompt_tokens"],
            decode_tokens=total,
            n_switches_planned=agent["n_switches"],
            n_segments=agent["n_segments"],
            elapsed_s=round(time.perf_counter() - t0, 4),
            deciles=watch.as_dict(total),
            **{
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in counters.items()
            },
        )
    )
    # Progress, to stderr. A wave takes tens of minutes and produced NO output until every agent had
    # finished, so a slow run and a hung one looked identical from outside -- the only way to tell them
    # apart was nvidia-smi inside the pod. Prints on completion rather than periodically, so it cannot
    # perturb the schedule being measured.
    print(
        f"  agent {agent['agent_id']:3d} done in {results[-1]['elapsed_s']:8.1f}s "
        f"(prompt {agent['prompt_tokens']:6,}, {agent['n_segments']:4d} segments) "
        f"-- {len(results)} finished",
        file=sys.stderr,
        flush=True,
    )


async def amain(args):
    # Plain or gzipped: the versioned fleet ships compressed (5.8 MB of node detail, 386 KB gzipped)
    # while a locally regenerated one is plain, and the driver should not care which it was handed.
    fp = pathlib.Path(args.fleet)
    if not fp.exists():
        raise SystemExit(f"FATAL: no fleet file at {fp}")
    opener = gzip.open if fp.suffix == ".gz" else open
    with opener(fp, "rt") as fh:
        fleet = [json.loads(x) for x in fh if x.strip()]
    wave = [a for a in fleet if a["wave"] == args.wave]
    if not wave:
        raise SystemExit(f"FATAL: no agents with wave=={args.wave} in {args.fleet}")
    longest = max(a["prompt_tokens"] + a["decode_tokens"] for a in wave)
    if longest >= args.max_model_len:
        raise SystemExit(
            f"FATAL: agent needs {longest:,} tokens of context but --max-model-len is "
            f"{args.max_model_len:,}. A sequence that cannot reach its forced length would be "
            f"truncated, and a truncated task is not the task."
        )

    model = args.base_model if args.arm == "native-lora" else args.sr_checkpoint

    # Read the control token from the CHECKPOINT rather than trusting a flag. A wrong id is the worst
    # kind of failure here: the run completes, the timings look plausible, and the arm silently decoded
    # on base the whole time. config.json carries adapter_token_ids for a granite-switch checkpoint.
    if args.arm == "shadow-residual" and args.sr_control_id <= 0:
        cfg = json.loads((pathlib.Path(model) / "config.json").read_text())
        ids = cfg.get("adapter_token_ids") or []
        if not ids:
            raise SystemExit(
                f"FATAL: {model}/config.json has no adapter_token_ids, so no control token can be "
                f"placed in the prompt and this arm would decode on base. Pass --sr-control-id "
                f"explicitly only if you know the id."
            )
        args.sr_control_id = ids[args.sr_adapter_index % len(ids)]
        print(
            f"  sr control token: id={args.sr_control_id} "
            f"(adapter index {args.sr_adapter_index} of {len(ids)})",
            file=sys.stderr,
        )
    engine, flavour = _make_engine(model, args)
    print(f"engine: {flavour}  arm={args.arm}  model={model}", file=sys.stderr)
    # Checked HERE rather than in a separate gate job: the engine already exists, so this costs nothing,
    # where a standalone gate paid a second full model load to learn the same number.
    check_kv_budget(engine, wave, args.kv_fraction, args.allow_kv_overcommit)

    # Resolve every adapter the wave references to a real PEFT directory, UP FRONT.
    #
    # The fleet names adapters unpadded ("ad0".."ad99"); compose_switch_repro.sh writes them padded and
    # nested ("adapter_00/granite-4.1-3b/lora"). Templating the path would have missed on both counts,
    # and it would have missed on the FIRST request rather than at startup -- by which time the other
    # arm may already have run and the log would read as a half-successful comparison. Resolving here
    # also catches a fleet referencing more adapters than were composed, which is the failure behind the
    # ~77% re-prefill artefact in SWITCH_MULTI_TURN.md 5.0: the arms rotated over different-sized pools,
    # which looked like a measurement fault and was not.
    lora_paths: dict[int, str] = {}
    if args.arm == "native-lora":
        root = pathlib.Path(args.lora_root)
        if not root.is_dir():
            raise SystemExit(f"FATAL: --lora-root {root} is not a directory")
        for d in sorted(root.glob("adapter_*")):
            m = re.fullmatch(r"adapter_(\d+)", d.name)
            if not m:
                continue
            # nested layout first (what the composer writes), then flat, for a hand-built fleet
            cfg = (
                sorted(d.glob("*/*/adapter_config.json"))
                or sorted(d.glob("*/adapter_config.json"))
                or sorted(d.glob("adapter_config.json"))
            )
            if cfg:
                lora_paths[int(m.group(1))] = str(cfg[0].parent)
        needed = {
            int(dd["adapter"][2:])
            for a in wave
            for n in a["nodes"]
            for dd in n["segments_detail"]
            if dd["adapter"] not in (None, "base")
        }
        missing = sorted(needed - set(lora_paths))
        if missing:
            raise SystemExit(
                f"FATAL: the wave references {len(needed)} adapters but {len(missing)} have no PEFT "
                f"directory under {root}: {missing[:10]}{' ...' if len(missing) > 10 else ''}. "
                f"Found {len(lora_paths)} on disk. Both arms must rotate over the same pool."
            )
        print(
            f"resolved {len(needed)} adapters under {root} "
            f"(e.g. ad{min(needed)} -> {lora_paths[min(needed)]})"
        )
        # max_loras is how many adapters vLLM keeps ON THE GPU, and BELOW THE POOL SIZE IS THE RIGHT
        # SETTING -- which is the opposite of what it looks like.
        #
        # Measured, wave 0 at M=64, same fleet, same schedules, compared per agent: raising max_loras from
        # 32 to 64 made ALL TWENTY agents slower, mean +7.2% (range +6.0% to +7.9%). Holding 32 more
        # adapters resident cost 7.16% of the KV cache (786,304 -> 730,016 tokens) and native slowed by
        # 7.2%. A 1:1 correspondence, and the reason is that native's time is dominated by re-prefill while
        # KV size is what limits how much prefix survives to be reused. Adapter residency is worth less
        # than the cache it displaces.
        #
        # So GPU-side eviction is NOT the confound it appeared to be. What matters is that max_loras stays
        # at or above the number of adapters concurrently in flight -- one per agent, so ~20 here -- or
        # requests would queue for a slot. 32 clears that with margin. Below ~20 this warning would be
        # worth heeding; above the pool size it is the wrong direction.
        if args.max_loras < 20:
            print(
                f"WARNING: --max-loras {args.max_loras} may be below the number of adapters in flight "
                f"(up to one per concurrent agent), so requests could queue for a slot -- a cost that is "
                f"real but is NOT re-prefill. Note the opposite is also a mistake: raising max_loras to "
                f"the full pool of {len(needed)} was measured 7.2% SLOWER, because the KV it displaces "
                f"matters more than adapter residency.",
                file=sys.stderr,
                flush=True,
            )

    # One LoRARequest per adapter name, created once. The int id must be stable across the run or vLLM
    # treats the same adapter as a new one and the prefix cache stops recognising it -- which would
    # quietly turn every revisit into a first visit and inflate exactly what this measures.
    lora_cache: dict[str, object] = {}

    def lora_for(name):
        if name is None or name == "base":
            return None  # unbound request: base, with its own prefix-cache namespace
        if name not in lora_cache:
            from vllm.lora.request import LoRARequest

            idx = int(name[2:])
            # The NAME stays the fleet's ("ad7"), because vLLM keys its prefix cache on it and the
            # whole measurement is which visits hit that cache. Only the path is resolved.
            lora_cache[name] = LoRARequest(name, idx + 1, lora_paths[idx])
        return lora_cache[name]

    results: list[dict] = []
    t0 = time.perf_counter()
    # Released together: this is the race.
    await asyncio.gather(
        *(
            run_agent(engine, flavour, a, args.arm, args, lora_for, results)
            for a in wave
        )
    )
    makespan = time.perf_counter() - t0

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in sorted(results, key=lambda r: r["agent_id"]):
            fh.write(json.dumps(r) + "\n")
    el = sorted(r["elapsed_s"] for r in results)
    # Nearest rank, matching agent_sim_summary.py. int(0.95 * len) indexed one past that -- at n=20
    # it returned the MAXIMUM and called it a p95, so this line and the summary script disagreed on
    # the same shard (5756.3s here against 5727.2s there). Two p95s for one dataset is worse than
    # either convention.
    p95 = el[min(len(el) - 1, int(round(0.95 * (len(el) - 1))))]
    print(
        f"wrote {out}  agents={len(results)}  makespan={makespan:.1f}s  "
        f"median={el[len(el) // 2]:.1f}s  p95={p95:.1f}s",
        file=sys.stderr,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=["native-lora", "shadow-residual"])
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument(
        "--fleet", default="docs/switch_data_agent_sim/agent_fleet_100.jsonl.gz"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="", help="native-lora: the base checkpoint")
    ap.add_argument(
        "--sr-checkpoint", default="", help="shadow-residual: the SR checkpoint"
    )
    ap.add_argument(
        "--lora-root", default="", help="native-lora: directory of PEFT adapter dirs"
    )
    ap.add_argument(
        "--sr-control-id",
        type=int,
        default=0,
        help="token id of the SR adapter's control token, which leads the prompt and never "
        "changes (ARMS['gs-sr-vllm'] uses prompt_control). Left at 0 it is READ from "
        "the checkpoint's adapter_token_ids, which is safer than passing it: a wrong "
        "id decodes on base and still looks like a successful run.",
    )
    ap.add_argument(
        "--sr-adapter-index",
        type=int,
        default=0,
        help="which of the checkpoint's adapters this arm uses for the whole task",
    )
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--max-loras", type=int, default=32)
    ap.add_argument(
        "--max-cpu-loras",
        type=int,
        default=128,
        help="hold the whole pool on CPU. Too low and native pages adapters from disk, "
        "adding a cost that is real but is NOT re-prefill and would confound it.",
    )
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=30000)
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument(
        "--sleep-dist",
        choices=("log", "uniform"),
        default="log",
        help="how tool and round-trip latencies are drawn within their ranges. Defaults to log, "
        "which is what the agent-runtime study used; the switching grid passes uniform.",
    )
    ap.add_argument(
        "--kv-fraction",
        type=float,
        default=0.85,
        help="refuse to run if the wave's peak KV demand exceeds this fraction of what the engine "
        "actually allocated. Above it vLLM preempts and swaps rather than failing, which inflates "
        "every timing without announcing itself.",
    )
    ap.add_argument(
        "--allow-kv-overcommit",
        action="store_true",
        help="run anyway when over the KV fraction, accepting that the timings are not comparable.",
    )
    args = ap.parse_args()

    if args.arm == "native-lora" and not (args.base_model and args.lora_root):
        raise SystemExit("FATAL: native-lora needs --base-model and --lora-root")
    if args.arm == "shadow-residual" and not args.sr_checkpoint:
        raise SystemExit("FATAL: shadow-residual needs --sr-checkpoint")

    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
