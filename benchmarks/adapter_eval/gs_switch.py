# SPDX-License-Identifier: Apache-2.0
"""The switching experiment's granite-switch LoRA and aLoRA arm.

The switch benchmark (``benchmarks/bench_switching_grid.py`` and
``bench_agent_sim.py``, copied unchanged) has a granite-switch arm only for SR,
and it never switches: one adapter serves the whole task, which SR's shared base
K/V makes free. A granite-switch LoRA or aLoRA switch is not free, so this arm
switches them the way its native-lora arm switches stock vLLM: a new request at
every segment, re-prefilling the context, with the same round trip after each
switch. The one difference is how a request names its adapter: by that
adapter's control token, first in the prompt, rather than by a LoRARequest. With
prefix caching on, a revisit reuses what that adapter computed before, as
stock vLLM's cache keyed on the LoRA does.

Everything else is that benchmark's code: its cell loop (``run_cell``), agent
loop (``run_agent``, driven as its native-lora arm), engine, gates, prefix-cache
reset and timing. ``ControlTokenEngine`` is the whole of the change.

    python -m benchmarks.adapter_eval.gs_switch --checkpoint <dir> \\
        --cells 8:32 --fleet-dir <dir> --out <rows.jsonl>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

from benchmarks.bench_agent_sim import _make_engine, check_kv_budget, kv_capacity
from benchmarks.bench_switching_grid import parse_cells, reset_prefix_cache, run_cell


class ControlToken:
    """Stands in for a LoRARequest: the adapter's control token."""

    def __init__(self, name: str, token_id: int) -> None:
        self.lora_name = name
        self.control_id = token_id


class ControlTokenEngine:
    """The engine, with each request's adapter put first in its prompt as a
    control token instead of passed as a LoRARequest."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def generate(self, prompt, params, request_id, lora_request=None):
        from vllm.inputs import TokensPrompt

        ids = list(prompt["prompt_token_ids"])
        if lora_request is not None:
            ids = [lora_request.control_id, *ids]
        return self._engine.generate(
            TokensPrompt(prompt_token_ids=ids), params, request_id
        )

    def __getattr__(self, name):
        return getattr(self._engine, name)


def control_tokens(checkpoint: pathlib.Path) -> dict[str, int]:
    """Each fleet adapter (``adN``) by its control token, from the checkpoint's own
    config rather than a flag: a wrong id would silently decode on another adapter."""
    cfg = json.loads((checkpoint / "config.json").read_text())
    names = cfg.get("adapter_names") or []
    ids = cfg.get("adapter_token_ids") or []
    if not names or len(names) != len(ids):
        raise SystemExit(f"FATAL: {checkpoint}/config.json names no control tokens")
    by_name = dict(zip(names, ids, strict=True))
    out = {}
    for k in range(len(names)):
        token = by_name.get(f"adapter_{k:02d}")
        if token is None:
            raise SystemExit(f"FATAL: the checkpoint has no adapter_{k:02d}")
        out[f"ad{k}"] = token
    return out


async def amain(args) -> None:
    cells = parse_cells(args.cells)
    fleets = {}
    for conc, span in cells:
        f = pathlib.Path(args.fleet_dir) / f"grid_c{conc}_s{span}.jsonl"
        if not f.exists():
            raise SystemExit(f"FATAL: no fleet for cell C={conc} span={span} at {f}")
        fleets[(conc, span)] = [
            json.loads(x) for x in f.read_text().splitlines() if x.strip()
        ]
    checkpoint = pathlib.Path(args.checkpoint)
    tokens = control_tokens(checkpoint)
    needed = {
        dd["adapter"]
        for f in fleets.values()
        for a in f
        for n in a["nodes"]
        for dd in n["segments_detail"]
        if dd["adapter"] not in (None, "base")
    }
    missing = sorted(needed - set(tokens))
    if missing:
        raise SystemExit(f"FATAL: the checkpoint lacks adapters {missing[:10]}")

    # Its engine without vLLM's LoRA (a granite-switch checkpoint holds its
    # adapters), then driven as its native-lora arm: a request per segment.
    engine, flavour = _make_engine(
        str(checkpoint), argparse.Namespace(**{**vars(args), "arm": "shadow-residual"})
    )
    cap = kv_capacity(engine)
    print(
        f"KV capacity: {cap:,} tokens" if cap else "KV capacity: unreadable",
        file=sys.stderr,
    )
    for f in fleets.values():
        for w in sorted({a["wave"] for a in f}):
            check_kv_budget(
                engine,
                [a for a in f if a["wave"] == w],
                args.kv_fraction,
                args.allow_kv_overcommit,
            )
    proxy = ControlTokenEngine(engine)
    run_args = argparse.Namespace(**{**vars(args), "arm": "native-lora"})

    def lora_for(name):
        return None if name in (None, "base") else ControlToken(name, tokens[name])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for cell in cells:
            ok = reset_prefix_cache(engine)
            if asyncio.iscoroutine(ok):
                ok = await ok or True
            if not ok:
                raise SystemExit(
                    "FATAL: could not reset the prefix cache between cells"
                )
            rows = await run_cell(
                proxy, flavour, fleets[cell], run_args, lora_for, cell
            )
            for r in sorted(rows, key=lambda r: r["agent_id"]):
                r["arm"] = "gs-lora-switch"
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            el = sorted(r["elapsed_s"] for r in rows)
            print(
                f"  C={cell[0]} span={cell[1]}: {len(rows)} agents, slowest {el[-1]:.1f}s",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cells", required=True, help='e.g. "8:32"  (concurrency:span)')
    ap.add_argument("--fleet-dir", required=True)
    ap.add_argument("--out", required=True)
    # That benchmark's engine and agent settings, with its defaults.
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
    asyncio.run(amain(ap.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
