# SPDX-License-Identifier: Apache-2.0
"""Put a recorded agent race into its page, ``docs/agent_race.html``.

The page lives on main, not on this branch: give a main checkout's copy. Runs
locally, on the log of the pod that ran ``run_race.py``::

    python -m benchmarks.adapter_eval.race.build_page <pod.log> \
        --page <main checkout>/docs/agent_race.html

The page then goes to main in a PR of its own.

The page holds one race per configuration (agents per lane, agents at a time,
tokens per agent), chosen on the page. The race is read from between the log's
markers and written into the page's ``const RACE_EVENTS_EMBEDDED = {...}; //
<<RACE_EVENTS>>`` line under its configuration, replacing a race of the same one
and keeping the others, so the page
opens straight from disk, as the ALORA-vs-LoRA race page does. A race that
carries an absolute path anywhere is refused: the page is public, the run's
storage is not.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import lane
from .run_race import decode_block

EMBED = re.compile(r"^const RACE_EVENTS_EMBEDDED = (.*); // <<RACE_EVENTS>>$", re.M)


def private_strings(value, where: str = "race") -> list[str]:
    """Where ``value`` holds a string that looks like a path on the run's machine."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in private_strings(v, f"{where}.{k}")]
    if isinstance(value, list):
        return [
            p for i, v in enumerate(value) for p in private_strings(v, f"{where}[{i}]")
        ]
    if isinstance(value, str) and value.startswith("/"):
        return [where]
    return []


def config_key(race: dict) -> str:
    """A race's configuration: agents per lane, agents at a time and tokens per
    agent (``mix`` for a mixed workload), e.g. ``64-8-512``; ``-react`` for the
    paper's ReAct agents, e.g. ``64-16-mix-react``."""
    meta = race["metadata"]
    work = meta.get("workload", {})
    tokens = "mix" if work.get("decode_mix") else work.get("decode_tokens", 512)
    kind = "-react" if work.get("agents") == "react" else ""
    return f"{meta['runs']}-{meta.get('concurrency', 8)}-{tokens}{kind}"


def _order(key: str) -> tuple:
    agents, slots, tokens, *kind = key.split("-")
    return (
        bool(kind),
        int(agents),
        int(slots),
        tokens == "mix",
        int(tokens) if tokens.isdigit() else 0,
    )


def races_in(page: str) -> dict[str, dict]:
    """The races ``page`` holds, by configuration, whatever key they were stored
    under. A page from before there was a choice holds one race."""
    found = EMBED.search(page)
    if not found:
        raise ValueError("the page has no RACE_EVENTS_EMBEDDED line")
    data = json.loads(found.group(1))
    if not data:
        return {}
    stored = [data] if "metadata" in data else list(data.values())
    return {config_key(race): race for race in stored}


def embed(page: str, races: dict[str, dict] | None) -> str:
    """``page`` with ``races`` (agents per lane: race) as its data; ``None`` or
    ``{}`` empties it."""
    if not EMBED.search(page):
        raise ValueError("the page has no RACE_EVENTS_EMBEDDED line")
    ordered = dict(sorted((races or {}).items(), key=lambda kv: _order(kv[0]))) or None
    line = (
        f"const RACE_EVENTS_EMBEDDED = {json.dumps(ordered, separators=(',', ':'))};"
        " // <<RACE_EVENTS>>"
    )
    return EMBED.sub(lambda _: line, page, count=1)


def add_race(page: str, race: dict) -> str:
    """``page`` with ``race`` as its race of that configuration, the others kept."""
    races = races_in(page)
    races[config_key(race)] = race
    return embed(page, races)


def add_switches(race: dict, fleet: list[dict]) -> dict:
    """``race``, recorded before the recorder kept switches (2026-10-12), with them
    added as it now records them: stock vLLM's where a request ended at an
    adapter change, and SR's adapter shown on the agent's schedule at the
    recorded token marks. ``fleet`` is the race's workload, regenerated: it must
    match the recorded agents and stock vLLM's adapters, or this refuses.

    Exact: a switch falls every ``span`` tokens, a multiple of the recorder's
    progress step, so the delta that crosses it was recorded with its time.
    Recorded times and counts are kept; events are only added, and SR's shown
    adapters set.
    """
    ordered = sorted(fleet, key=lambda a: a["agent_id"])
    if any(e["ev"] == "switch" for e in race["events"]):
        raise ValueError("the race already has its switches")
    for name, rows in race["agents"].items():
        recorded = [(r["agent_id"], r["prompt_tokens"]) for r in rows]
        if recorded != [(a["agent_id"], a["prompt_tokens"]) for a in ordered]:
            raise ValueError(f"{name}'s agents are not this fleet's")
    arms = {name: meta["arm"] for name, meta in race["metadata"]["lanes"].items()}
    segs = [lane.segments_of(a) for a in ordered]
    units = {
        arm: [lane.units_of(a, arm) for a in ordered] for arm in set(arms.values())
    }
    for name, arm in arms.items():
        if arm != "native-lora":
            continue
        ran = [[] for _ in ordered]
        for e in race["events"]:
            if e["srv"] == name and e["ev"] == "step_start" and e["step"] == "prefill":
                ran[e["conv"]].append(e["adapter"])
        if ran != [[s["adapter"] for s in agent] for agent in segs]:
            raise ValueError(f"{name}'s adapters are not this fleet's schedule")

    state: dict[tuple[str, int], dict] = {}
    run: dict[str, set] = {name: set() for name in arms}
    out: list[dict] = []
    for e in race["events"]:
        e = dict(e)
        out.append(e)
        if "conv" not in e:
            continue
        name, k = e["srv"], e["conv"]
        arm, st = arms[name], state.setdefault((name, k), dict(tokens=0, seg=0, end=0))
        mine = segs[k]

        def emit(ev, **fields):
            out.append({"t": e["t"], "ev": ev, "srv": name, "conv": k, **fields})

        def shown():
            return mine[st["seg"]]["adapter"] if st["seg"] < len(mine) else None

        if e["ev"] == "step_start" and e["step"] == "prefill":
            run[name].add(e.get("adapter"))
            if arm == "shadow-residual":
                st["end"] = st["tokens"] + units[arm][k][e["unit"]]["tokens"]
                e["adapter"] = shown()
        elif e["ev"] == "progress" and arm == "shadow-residual":
            st["tokens"] = e["tokens"]
            while st["seg"] < len(mine) and st["tokens"] >= mine[st["seg"]]["end"]:
                ended = mine[st["seg"]]
                st["seg"] += 1
                if ended["switch"]:
                    emit("switch", adapter=ended["adapter"], to=shown())
                if st["seg"] < len(mine) and st["tokens"] < st["end"]:
                    emit("step_start", step="decode", adapter=shown())
        elif e["ev"] == "req_done" and arm == "native-lora":
            todo = units[arm][k]
            if todo[e["unit"]]["switch"]:
                nxt = next(
                    (u for u in todo[e["unit"] + 1 :] if u["kind"] == "gen"), None
                )
                emit(
                    "switch",
                    adapter=todo[e["unit"]]["adapter"],
                    to=nxt and nxt["adapter"],
                )

    upgraded = {
        **race,
        "events": out,
        "metadata": json.loads(json.dumps(race["metadata"])),
    }
    meta = upgraded["metadata"]
    meta["workload"].setdefault("min_agents", len(ordered))
    meta["switches_added"] = (
        "from the schedule, at the recorded token marks (build_page)"
    )
    for name, info in meta["lanes"].items():
        mine = [e for e in out if e["srv"] == name]
        info["switches"] = sum(1 for e in mine if e["ev"] == "switch")
        info["requests"] = sum(1 for e in mine if e["ev"] == "req_done")
        info["adapters_run"] = sorted(a for a in run[name] if a)
    return upgraded


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", type=Path, nargs="?", help="the race pod's saved log")
    p.add_argument(
        "--page",
        type=Path,
        required=True,
        help="the page: docs/agent_race.html in a main checkout",
    )
    p.add_argument(
        "--add-switches",
        type=Path,
        metavar="FLEET",
        help="instead of a log: add the switches to the page's race recorded without "
        "them, on this regenerated workload (gen_switching_grid.py)",
    )
    args = p.parse_args(argv)
    if args.add_switches:
        page = args.page.read_text()
        fleet = lane.load_fleet(args.add_switches)
        races = races_in(page)
        lacking = [
            k
            for k, r in races.items()
            if r["metadata"]["runs"] == len(fleet)
            and not any(e["ev"] == "switch" for e in r["events"])
        ]
        if len(lacking) != 1:
            raise SystemExit(f"expected one {len(fleet)}-agent race without switches")
        key = lacking[0]
        races[key] = add_switches(races[key], fleet)
        args.page.write_text(embed(page, races))
        lanes = races[key]["metadata"]["lanes"]
        print(
            f"added the switches to the {key}-agent race: "
            + ", ".join(
                f"{k} {v['switches']} over {v['requests']} requests"
                for k, v in lanes.items()
            )
        )
        return 0
    if args.log is None:
        p.error("give the race pod's log, or --add-switches")
    race = decode_block(args.log.read_text(errors="replace"))
    leaks = private_strings(race)
    if leaks:
        raise SystemExit(f"refusing to publish: absolute paths at {leaks[:5]}")
    page = add_race(args.page.read_text(), race)
    args.page.write_text(page)
    meta = race["metadata"]
    walls = ", ".join(f"{k} {v['wall_s']:.1f}s" for k, v in meta["lanes"].items())
    held = ", ".join(races_in(page))
    print(
        f"wrote {args.page}: the {config_key(race)} race, {len(race['events'])} "
        f"events; {walls}. The page holds: {held}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
