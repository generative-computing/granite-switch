#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copied from the switch benchmark (granite-switch-staging, branch
# feature/switch-benchmark, commit 5a51e57) and kept unchanged, so this benchmark
# measures switching exactly as it does. Update it only by copying it again.
"""Sweep the concurrency x switching-frequency grid for one arm, inside a single engine load.

A companion to bench_agent_sim.py, which runs ONE cell (one wave of one fleet) per invocation. That is
right for the agent-runtime study, where a cell is the whole experiment; it is wrong here, where there are
49 cells per arm and 49 engine loads would add two and a half hours of startup to a grid whose compute is
eight. This driver loads the engine once and walks the cells.

It imports the primitives rather than re-implementing them, so both studies issue requests, sleep, and
time their deciles through exactly the same code. What differs is only the loop around them.

TWO THINGS THIS DRIVER HAS TO GET RIGHT THAT THE SINGLE-CELL ONE DOES NOT:

  THE PREFIX CACHE MUST BE RESET BETWEEN CELLS. Cells share an engine, so without a reset each cell
  inherits the previous cell's warm cache. The measured hit rate would then drift upward through the
  sweep and the last cell of a job would look better than the first for no reason but its position --
  and since cells are grouped by span, that drift would masquerade as a span effect.

  A CELL CAN HOLD SEVERAL WAVES. At low concurrency a cell runs extra waves to reach --min-agents, and
  each wave must be released on its own: gathering all of a cell's agents at once would make the
  concurrency the agent count rather than the cell's concurrency, which is the independent variable.
"""

import argparse
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmarks.bench_agent_sim import (
    _make_engine,
    check_kv_budget,
    kv_capacity,
    run_agent,
)


def parse_cells(spec):
    """ "1:32,2:32" -> [(1, 32), (2, 32)], preserving order so a job's cells run as written."""
    cells = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        c, s = part.split(":")
        cells.append((int(c), int(s)))
    if not cells:
        raise SystemExit("FATAL: --cells parsed to nothing")
    return cells


def reset_prefix_cache(engine) -> bool:
    """Drop the prefix cache, returning whether it could be done.

    Tried across the paths vLLM has used for it. Returns False rather than raising so the caller can
    decide: for this sweep a silent failure would let cache state leak between cells, so the caller
    treats False as fatal unless explicitly allowed.
    """
    for target in (
        engine,
        getattr(engine, "engine", None),
        getattr(engine, "llm_engine", None),
    ):
        if target is None:
            continue
        fn = getattr(target, "reset_prefix_cache", None)
        if callable(fn):
            out = fn()
            if asyncio.iscoroutine(out):
                return out  # awaited by the caller
            return True
    return False


async def run_cell(engine, flavour, fleet, args, lora_for, cell):
    """One cell: its waves in sequence, each wave's agents released together."""
    conc, span = cell
    rows: list[dict] = []
    waves = sorted({a["wave"] for a in fleet})
    t0 = time.perf_counter()
    for w in waves:
        wave = [a for a in fleet if a["wave"] == w]
        await asyncio.gather(
            *(
                run_agent(engine, flavour, a, args.arm, args, lora_for, rows)
                for a in wave
            )
        )
    for r in rows:
        r["concurrency"] = conc
        r["span"] = span
        r["cell_seconds"] = round(time.perf_counter() - t0, 4)
    return rows


async def amain(args):
    cells = parse_cells(args.cells)
    fleet_dir = pathlib.Path(args.fleet_dir)
    fleets = {}
    for conc, span in cells:
        f = fleet_dir / f"grid_c{conc}_s{span}.jsonl"
        if not f.exists():
            raise SystemExit(f"FATAL: no fleet for cell C={conc} span={span} at {f}")
        fleets[(conc, span)] = [
            json.loads(x) for x in f.read_text().splitlines() if x.strip()
        ]
    print(
        f"{len(cells)} cell(s): {', '.join(f'C{c}/s{s}' for c, s in cells)}",
        file=sys.stderr,
    )

    model = args.sr_checkpoint if args.arm == "shadow-residual" else args.base_model
    if not model:
        raise SystemExit(f"FATAL: no checkpoint given for arm {args.arm}")
    if args.arm == "shadow-residual" and not args.sr_control_id:
        cfg = json.loads((pathlib.Path(model) / "config.json").read_text())
        ids = cfg.get("adapter_token_ids") or []
        if not ids:
            raise SystemExit(f"FATAL: {model}/config.json has no adapter_token_ids")
        args.sr_control_id = ids[args.sr_adapter_index % len(ids)]
        print(f"sr control token id={args.sr_control_id}", file=sys.stderr)

    # The engine is sized for the WHOLE job: max_model_len must hold the largest prompt plus its decode
    # across every cell, or a cell late in the sweep fails after hours of work.
    longest = max(
        a["prompt_tokens"] + a["decode_tokens"] for f in fleets.values() for a in f
    )
    if longest >= args.max_model_len:
        raise SystemExit(
            f"FATAL: a cell needs {longest:,} tokens of context but --max-model-len is "
            f"{args.max_model_len:,}"
        )
    engine, flavour = _make_engine(model, args)
    print(f"engine: {flavour}  arm={args.arm}  model={model}", file=sys.stderr)
    cap = kv_capacity(engine)
    print(
        f"KV capacity: {cap:,} tokens" if cap else "KV capacity: unreadable",
        file=sys.stderr,
    )
    # Every cell is checked before ANY work is done: discovering at cell 40 that its wave overcommits KV
    # would mean the run had already spent hours producing preempted, silently inflated timings.
    for (conc, span), f in fleets.items():
        for w in sorted({a["wave"] for a in f}):
            check_kv_budget(
                engine,
                [a for a in f if a["wave"] == w],
                args.kv_fraction,
                args.allow_kv_overcommit,
            )

    lora_paths: dict[int, str] = {}
    if args.arm == "native-lora":
        import re

        root = pathlib.Path(args.lora_root)
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
                lora_paths[int(m.group(1))] = str(cfgs[0].parent)
        needed = {
            int(dd["adapter"][2:])
            for f in fleets.values()
            for a in f
            for n in a["nodes"]
            for dd in n["segments_detail"]
            if dd["adapter"] not in (None, "base")
        }
        missing = sorted(needed - set(lora_paths))
        if missing:
            raise SystemExit(
                f"FATAL: the grid references {len(needed)} adapters, {len(missing)} have no PEFT "
                f"directory under {root}: {missing[:10]}"
            )
        print(f"resolved {len(needed)} adapters under {root}", file=sys.stderr)

    lora_cache: dict[str, object] = {}

    def lora_for(name):
        if name is None or name == "base":
            return None
        if name not in lora_cache:
            from vllm.lora.request import LoRARequest

            idx = int(name[2:])
            lora_cache[name] = LoRARequest(name, idx + 1, lora_paths[idx])
        return lora_cache[name]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as fh:
        for i, cell in enumerate(cells):
            ok = reset_prefix_cache(engine)
            if asyncio.iscoroutine(ok):
                ok = await ok or True
            if not ok and not args.allow_warm_cache:
                raise SystemExit(
                    "FATAL: could not reset the prefix cache between cells. Cells share an engine, so "
                    "each would inherit the previous cell's warm cache, the measured hit rate would drift "
                    "upward through the sweep, and because cells are grouped by span that drift would "
                    "masquerade as a span effect. Pass --allow-warm-cache to measure anyway."
                )
            rows = await run_cell(engine, flavour, fleets[cell], args, lora_for, cell)
            for r in sorted(rows, key=lambda r: r["agent_id"]):
                fh.write(json.dumps(r) + "\n")
                written += 1
            fh.flush()
            el = sorted(r["elapsed_s"] for r in rows)
            p95 = el[min(len(el) - 1, int(round(0.95 * (len(el) - 1))))]
            print(
                f"  [{i + 1}/{len(cells)}] C={cell[0]:<3} span={cell[1]:<4} {len(rows):3d} agents  "
                f"median={el[len(el) // 2]:8.1f}s  p95={p95:8.1f}s",
                file=sys.stderr,
                flush=True,
            )
    print(f"wrote {out}  {written} rows across {len(cells)} cell(s)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, choices=["native-lora", "shadow-residual"])
    ap.add_argument(
        "--cells", required=True, help='e.g. "1:32,2:32,4:32"  (concurrency:span)'
    )
    ap.add_argument("--fleet-dir", required=True)
    ap.add_argument("--out", required=True)
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
    ap.add_argument(
        "--sleep-dist",
        choices=("log", "uniform"),
        default="uniform",
        help="uniform here, unlike the agent study: the grid fixes one deployment distance, so a "
        "log-uniform draw within it buys nothing.",
    )
    ap.add_argument(
        "--allow-warm-cache",
        action="store_true",
        help="run even if the prefix cache cannot be reset between cells, accepting that later cells "
        "inherit earlier ones' cache.",
    )
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
