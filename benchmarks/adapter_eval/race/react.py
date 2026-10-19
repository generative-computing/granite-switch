# SPDX-License-Identifier: Apache-2.0
"""ReAct agents for the agent race, built with the switch benchmark's own generator.

The paper's agents (``benchmarks/gen_agent_configs.py``, copied unchanged): each
walks a ReAct cycle of typed steps, drawn from the generator's own tables:

============ ============== ================ =====================================
step         median tokens  adapter switch   inside the step
============ ============== ================ =====================================
think        120            35% of the time  1 + Poisson(2.0) segments, a switch each
prepare_call 10             95%              one segment
act          0 (tool call)                   a tool's latency, slept for real
observe      35             80%              1 + Poisson(0.5) segments
reflect      90             70%              1 + Poisson(1.5) segments
============ ============== ================ =====================================

over three cycles (classic5, toolheavy10, shortadapters), choosing adapters
novelty-first over a pool of 64. Three settings are the race's, not the paper's:

* **Lengths:** 128, 256, 512 or 1,024 tokens, 16 agents each, as the race's mixed
  workload. The generator closes every tenth of an agent's tokens on a step
  boundary, for the paper's stopwatch; tenths of 128 tokens are not whole, so here
  an agent is one chunk, closed by its last step. Every other step keeps its
  drawn length.
* **Prompts:** the switching workload's ladder (256 to 16,384 tokens, shuffled
  with its seed), so the agents' contexts are those of the other races. The
  generator sizes its own for 20 agents at a time, at 10,000 tokens each.
* **Network distance:** the same datacenter for every agent (the generator's
  category 3, a round trip of 2 to 10 ms), as in the other races. The generator
  draws a mix, from the same process to another region.

Nothing else is changed: the steps, their lengths, switches and segments, the
cycles, the tools and the adapter choice are the generator's, from its own
functions with its own seed.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

GENERATOR = "benchmarks/gen_agent_configs.py"
GRID = "benchmarks/gen_switching_grid.py"
LENGTHS = (128, 256, 512, 1024)
# The generator's own seed, and the switching workload's, for its prompt ladder.
SEED = 20260917
LADDER_SEED = 20260921
# Same cluster / same datacenter: the generator's category 3.
RTT_CATEGORY = 3


def _load(root: Path, rel: str):
    spec = importlib.util.spec_from_file_location(Path(rel).stem, root / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(
    root: Path,
    agents: int = 64,
    lengths: tuple[int, ...] = LENGTHS,
    adapters: int = 64,
    seed: int = SEED,
    cycles: tuple[str, ...] | None = None,
) -> list[dict]:
    """``agents`` ReAct agents, an equal share at each of ``lengths``, shuffled so
    the lengths interleave. Both lanes read the one list, so an agent's steps,
    prompt and schedule are the same in both. Each agent walks one of ``cycles``
    (the generator's names; all of its cycles when not given), drawn at random."""
    gen, grid = _load(root, GENERATOR), _load(root, GRID)
    if agents % len(lengths):
        raise SystemExit(f"FATAL: {agents} agents do not split evenly over {lengths}")
    cycles = tuple(cycles or gen.CYCLES)
    unknown = sorted(set(cycles) - set(gen.CYCLES))
    if unknown:
        raise SystemExit(f"FATAL: no ReAct cycle {unknown}; have {sorted(gen.CYCLES)}")
    if adapters > 64:
        raise SystemExit(
            f"FATAL: --adapters {adapters} exceeds the 64 the SWITCH kernel holds"
        )
    rng = random.Random(seed)
    ladder = grid.ladder()
    prompts = (ladder * -(-agents // len(ladder)))[:agents]
    random.Random(LADDER_SEED ^ 0x5EED).shuffle(prompts)
    order = [n for n in lengths for _ in range(agents // len(lengths))]
    random.Random(seed).shuffle(order)
    return [
        _agent(gen, rng, i, decode, prompts[i], adapters, cycles)
        for i, decode in enumerate(order)
    ]


def _agent(
    gen, rng, agent_id: int, decode: int, prompt: int, adapters: int, cycles
) -> dict:
    """One agent, as the generator's ``main`` builds one, with the race's length,
    prompt and network distance. The adapter choice is the generator's loop."""
    cycle_name = rng.choice(list(cycles))
    nodes = gen.build_spans(rng, decode, gen.CYCLES[cycle_name], deciles=1)
    toolbox = [
        dict(name=f"t{t:02d}", category=rng.choice(sorted(gen.TOOLS)))
        for t in range(gen.TOOLS_PER_AGENT)
    ]
    tool_draws = [rng.randrange(gen.TOOLS_PER_AGENT) for n in nodes if n["tool"]]
    seen: dict[str, int] = {}
    cur, cum, switches = "base", 0, 0
    for node in nodes:
        if node["tool"]:
            node["segments_detail"] = []
            continue
        detail = []
        for si, seg in enumerate(node.get("segments", [node["span"]])):
            is_switch = node["switch"] if si == 0 else True
            if is_switch:
                unused = [j for j in range(adapters) if f"ad{j}" not in seen]
                if unused and rng.random() < 0.9:
                    cur = f"ad{rng.choice(unused)}"
                elif seen:
                    cur = min(seen, key=seen.get)
                else:
                    cur = f"ad{rng.randrange(adapters)}"
                switches += 1
                first = cur not in seen
                recompute = prompt + cum if first else cum - seen[cur]
            else:
                first, recompute = False, 0
            detail.append(
                dict(
                    adapter=cur,
                    tokens=seg,
                    switch=is_switch,
                    first_visit=first,
                    native_recompute=recompute,
                )
            )
            cum += seg
            seen[cur] = cum
        node["segments_detail"] = detail
    segs = [d for n in nodes for d in n["segments_detail"]]
    return dict(
        agent_id=agent_id,
        wave=0,
        n_segments=len(segs),
        n_first_visits=sum(1 for d in segs if d["first_visit"]),
        native_recompute_tokens=sum(d["native_recompute"] for d in segs),
        rtt_category=RTT_CATEGORY,
        toolbox=toolbox,
        tool_draws=tool_draws,
        cycle=cycle_name,
        prompt_tokens=prompt,
        decode_tokens=decode,
        n_nodes=len(nodes),
        n_switches=switches,
        n_tool_nodes=sum(1 for n in nodes if n["tool"]),
        effective_span=round(decode / max(1, switches), 1),
        nodes=nodes,
    )


def write(agents: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(a) + "\n" for a in agents))
    return out
