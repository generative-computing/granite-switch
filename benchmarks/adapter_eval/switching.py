# SPDX-License-Identifier: Apache-2.0
"""Agents switching adapters, measured by the switch benchmark's own scripts.

One cell of its grid_concurrency figure (``benchmarks/gen_switching_grid.py``,
``bench_switching_grid.py``, ``bench_agent_sim.py``, copied unchanged): agents
released together in waves, each generating ``decode_tokens`` and switching
adapter every ``span`` tokens, with tool calls and their latencies at its rate,
over its synthetic adapters (``make_synth_fleet.py``, composed and checked as
its ``compose_switch_repro.sh`` does, ``verify_composed.py``). This module only
builds those scripts' command lines and reads their per-agent rows back.

Per technology, on ``adapters`` synthetic adapters of that flavor:

======== ===================================== ==================================
tech     granite-switch                        PEFT (vLLM): stock vLLM
======== ===================================== ==================================
lora     a checkpoint of the LoRA adapters,    its ``native-lora`` arm: a new
         switching by control token            request per segment, with a
         (``gs_switch.py``)                    LoRARequest
alora    the same, with the aLoRA adapters     the same, with the aLoRA adapters
                                               without invocation tokens
sr       its ``shadow-residual`` arm: one      none: stock vLLM has no SR
         adapter for the task (SR switches
         without a re-prefill)
======== ===================================== ==================================

A cell's value is its figure's: the p95 time for an agent to complete, by
nearest rank over the cell's agents.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .common import error

FLEET_SCRIPT = "benchmarks/make_synth_fleet.py"
VERIFY_SCRIPT = "benchmarks/verify_composed.py"
GRID_SCRIPT = "benchmarks/gen_switching_grid.py"
DRIVER = "benchmarks/bench_switching_grid.py"
GS_DRIVER = "benchmarks.adapter_eval.gs_switch"
# Each technology's synthetic flavor, the leaf folder it builds, and the
# composer's --technology for it (SR is detected from its weights).
FLAVORS = {
    "lora": ("lora", "lora", "lora"),
    "alora": ("alora", "alora", "alora"),
    "sr": ("sr", "alora", "alora"),
}
# The engines each technology runs on: stock vLLM has no SR.
ENGINES = {"lora": ("gs", "native"), "alora": ("gs", "native"), "sr": ("gs",)}


def names(n: int) -> list[str]:
    """The synthetic fleet's adapter folders, as its builder names them."""
    return [f"adapter_{k:02d}" for k in range(n)]


def fleet_command(
    base: str, out: Path, tech: str, settings: dict, target: str
) -> list[str]:
    flavor = FLAVORS[tech][0]
    return [
        FLEET_SCRIPT,
        "--base", base,
        "--output", str(out),
        "--flavor", flavor,
        "--num-adapters", str(settings["adapters"]),
        "--rank", str(settings["rank"]),
        "--target-model", target,
    ]  # fmt: skip


def leaves(fleet: Path, tech: str, target: str, n: int) -> list[Path]:
    """The fleet's adapter folders, in adapter order."""
    leaf = FLAVORS[tech][1]
    return [fleet / name / target / leaf for name in names(n)]


def compose_args(
    base: str, fleet: Path, tech: str, target: str, n: int, out: Path
) -> list[str]:
    """The composer's arguments, as that benchmark's compose_one passes them."""
    return [
        "--base-model", base,
        "--adapters", str(fleet),
        "--target-model", target,
        "--technology", FLAVORS[tech][2],
        "--include-adapters", *names(n),
        "--output", str(out),
    ]  # fmt: skip


def verify_command(
    checkpoint: Path, tech: str, settings: dict, base_vocab: int
) -> list[str]:
    """That benchmark's check of a composed checkpoint: a compose that drops weights
    without an error would otherwise give a fast-looking arm."""
    n, rank = settings["adapters"], settings["rank"]
    return [
        VERIFY_SCRIPT,
        str(checkpoint),
        "--expect-adapters", str(n),
        "--expect-rank", str(rank),
        "--expect-alpha", str(rank),
        "--base-vocab", str(base_vocab),
        "--expect-names", ",".join(names(n)),
        *(["--expect-kv"] if tech != "sr" else []),
    ]  # fmt: skip


def grid_command(settings: dict, out: Path) -> list[str]:
    return [
        GRID_SCRIPT,
        "--concurrency", str(settings["concurrency"]),
        "--span", str(settings["span"]),
        "--decode", str(settings["decode_tokens"]),
        "--adapters", str(settings["adapters"]),
        "--min-agents", str(settings["min_agents"]),
        "--out", str(out),
    ]  # fmt: skip


def cell(settings: dict) -> str:
    return f"{settings['concurrency']}:{settings['span']}"


def arm_command(
    tech: str, engine: str, model: str, fleet_dir: Path, native_root: Path | None,
    settings: dict, out: Path,
) -> list[str]:  # fmt: skip
    """One engine's run: its driver for stock vLLM and SR, ``gs_switch`` for
    granite-switch LoRA and aLoRA."""
    common = [
        "--cells",
        cell(settings),
        "--fleet-dir",
        str(fleet_dir),
        "--out",
        str(out),
    ]
    if engine == "native":
        return [
            DRIVER, "--arm", "native-lora", "--base-model", model,
            "--lora-root", str(native_root), "--lora-rank", str(settings["rank"]),
            *common,
        ]  # fmt: skip
    if tech == "sr":
        return [DRIVER, "--arm", "shadow-residual", "--sr-checkpoint", model, *common]
    return ["-m", GS_DRIVER, "--checkpoint", model, *common]


def p95(values: list[float]) -> float:
    """95th percentile by nearest rank, as its figure takes it."""
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def entry(rows: list[dict], settings: dict) -> dict:
    """A cell's entry from the driver's per-agent rows, or an error."""
    if not rows:
        return error("the driver wrote no agents")
    elapsed = [r["elapsed_s"] for r in rows]

    def total(key):
        return sum(r.get(key, 0) for r in rows)

    return {
        "p95_s": round(p95(elapsed), 2),
        "median_s": round(statistics.median(elapsed), 2),
        "agents": len(rows),
        "waves": len({r.get("wave") for r in rows}),
        "elapsed_s": [round(e, 2) for e in sorted(elapsed)],
        "switches": total("switches"),
        "tool_calls": total("tool_calls"),
        "prefill_recomputed": total("prefill_recomputed"),
        "prefill_cached": total("prefill_cached"),
        "arm": rows[0].get("arm"),
        **settings,
    }


def read_entry(out: Path, settings: dict) -> dict:
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    return entry(rows, settings)
