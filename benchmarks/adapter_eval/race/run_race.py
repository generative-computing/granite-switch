# SPDX-License-Identifier: Apache-2.0
"""Run the agent race on a pod: both lanes on one GPU, one after the other.

Through the adapter benchmark's script mode, which installs the commit and ships
the harness::

    benchmarks/adapter_eval/vela/submit.sh script <ref> \\
        benchmarks/adapter_eval/race/run_race.py [--model <id>]

With the adapter benchmark's own pieces, as its switching block uses them:

1. The workload: the switching block's cell (``switching`` in adapters.yaml)
   from the switch benchmark's generator (``gen_switching_grid.py``): agents
   with prompts of their own, ``decode_tokens`` generated each, a switch every
   ``span``, tool calls at its rate.
2. Its synthetic adapters, LoRA and SR, from the benchmark's per-model cache.
3. The SR checkpoint: the SR adapters composed by the commit's composer and
   checked by ``verify_composed.py``.
4. The lanes, each in a process of its own with fresh compile caches (``lane.py``):
   gs-sr-vllm, granite-switch SR (the switch benchmark's shadow-residual arm),
   then lora-vllm, stock vLLM serving the LoRA adapters (its native-lora arm).
5. The race: both lanes' events, each timed from its own start, saved in the
   run's folder and printed into the log, gzipped and base64-encoded between
   ``RACE_BEGIN`` and ``RACE_END``, for ``build_page.py``.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import json
import os
import sys
from pathlib import Path

from benchmarks.adapter_eval import switching
from benchmarks.adapter_eval.common import load_spec
from benchmarks.adapter_eval.run_benchmark import (
    COMPOSER_MODULE,
    base_model_dir,
    clear_compile_caches,
    commit_info,
    generation_prompt_anchor,
    run_streamed,
    synthetic_fleet,
)

RACE_BEGIN = "=== AGENT_RACE_BEGIN ==="
RACE_END = "=== AGENT_RACE_END ==="
LANE_MODULE = "benchmarks.adapter_eval.race.lane"
# The lanes, in the order they run: (name on the page, the switch benchmark's arm,
# synthetic flavor).
LANES = (
    ("gs-sr-vllm", "shadow-residual", "sr"),
    ("lora-vllm", "native-lora", "lora"),
)
TAG = "race"


def harness_root() -> Path:
    """The harness checkout: run_race.py itself runs from a copy (extra/script.py)."""
    return Path(switching.__file__).resolve().parents[2]


def lane_command(
    lane: str, arm: str, grid_file: Path, out: Path, settings: dict, *,
    base_model: str = "", lora_root: Path | None = None, checkpoint: Path | None = None,
) -> list[str]:  # fmt: skip
    cmd = [
        "-m", LANE_MODULE,
        "--lane", lane,
        "--arm", arm,
        "--fleet", str(grid_file),
        "--out", str(out),
        "--concurrency", str(settings["concurrency"]),
        "--warmup-runs", str(settings["warmup_runs"]),
    ]  # fmt: skip
    if arm == "shadow-residual":
        return [*cmd, "--sr-checkpoint", str(checkpoint)]
    return [
        *cmd, "--base-model", base_model, "--lora-root", str(lora_root),
        "--lora-rank", str(settings["rank"]),
    ]  # fmt: skip


def race_record(spec, settings: dict, commit: dict, lanes: list[dict]) -> dict:
    """Both lanes' reports as one race: its metadata, its events in time order, and
    each lane's agents."""
    walls = {lane["lane"]: lane["wall_s"] for lane in lanes}
    keep = (
        "arm", "wall_s", "warmup_wall_s", "generated", "prompt_tokens",
        "recomputed_tokens", "cached_tokens", "first_prefill_tokens", "resent_tokens",
        "resent_cached_tokens", "reprefilled_tokens", "kv_capacity", "gpu", "torch",
        "vllm",
    )  # fmt: skip
    return {
        "metadata": {
            "mode": "sequential",
            "runs": max(lane["agents"] for lane in lanes),
            "concurrency": settings["concurrency"],
            "timestamp": dt.datetime.now(dt.UTC).isoformat(),
            "race_wall": max(walls.values()),
            "model": spec.model_id,
            "base_model": spec.base_model,
            "commit": commit,
            "workload": {
                k: settings[k]
                for k in ("decode_tokens", "span", "adapters", "rank", "warmup_runs")
            },
            "lanes": {lane["lane"]: {k: lane.get(k) for k in keep} for lane in lanes},
        },
        "race_wall": max(walls.values()),
        "servers": [lane["lane"] for lane in lanes],
        "events": sorted(
            (e for lane in lanes for e in lane["events"]), key=lambda e: e["t"]
        ),
        "agents": {lane["lane"]: lane["rows"] for lane in lanes},
    }


def encode_block(race: dict) -> str:
    """The race between the markers: gzipped JSON in base64 lines, which a pod log
    carries whole, unlike one very long line."""
    data = base64.b64encode(gzip.compress(json.dumps(race).encode())).decode()
    lines = [data[i : i + 76] for i in range(0, len(data), 76)]
    return "\n".join([RACE_BEGIN, *lines, RACE_END])


def decode_block(text: str) -> dict:
    begin = text.index(RACE_BEGIN) + len(RACE_BEGIN)
    end = text.index(RACE_END, begin)
    data = "".join(line.strip() for line in text[begin:end].splitlines())
    return json.loads(gzip.decompress(base64.b64decode(data)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="an adapters.yaml model id")
    p.add_argument(
        "--work-dir",
        default=os.environ.get("ADAPTER_BENCH_WORK_DIR", "race-work"),
        help="the run's folder (default: the one script mode gives the run)",
    )
    p.add_argument(
        "--model-dir",
        default="/workspace/models" if Path("/workspace").is_dir() else None,
        help="where to compose the SR checkpoint (default: the pod's local disk)",
    )
    p.add_argument(
        "--switching-cache",
        default=None,
        help="the synthetic adapters' cache (default: the benchmark's, per model)",
    )
    args = p.parse_args(argv)

    spec = load_spec(model=args.model)
    settings = spec.switching.settings()
    python, root = sys.executable, harness_root()
    work = Path(args.work_dir).resolve() / "race"
    work.mkdir(parents=True, exist_ok=True)
    model_root = Path(args.model_dir or work / "models")
    base_model = os.environ.get("ADAPTER_BENCH_BASE_MODEL") or spec.base_model
    cache = Path(
        args.switching_cache
        or Path(os.environ["ADAPTER_BENCH_WORK_ROOT"]) / "synthetic" / spec.model_id
    )
    print(
        f"[{TAG}] {spec.model_id}: {settings['min_agents']} agents per lane, "
        f"{settings['concurrency']} at a time",
        flush=True,
    )

    grid_file = work / f"grid_c{settings['concurrency']}_s{settings['span']}.jsonl"
    if run_streamed([python, *switching.grid_command(settings, grid_file)], root, TAG)[
        0
    ]:
        raise SystemExit("FATAL: workload generation failed")
    base_dir = base_model_dir(base_model)
    vocab = json.loads((base_dir / "config.json").read_text())["vocab_size"]
    anchor = generation_prompt_anchor(
        base_model, spec.model.prompt_for("sr")["chat_template_kwargs"]
    )
    fleets = {}
    for tech in ("lora", "sr"):
        fleets[tech] = synthetic_fleet(
            python, root, cache, str(base_dir), tech, settings, spec.model_id,
            anchor if tech == "sr" else None,
        )  # fmt: skip
        if fleets[tech] is None:
            raise SystemExit(f"FATAL: the synthetic {tech} adapters failed")

    checkpoint = model_root / "race_sr"
    compose = [
        python, "-m", COMPOSER_MODULE,
        *switching.compose_args(
            base_model, fleets["sr"], "sr", spec.model_id, settings["adapters"], checkpoint
        ),
    ]  # fmt: skip
    verify = [python, *switching.verify_command(checkpoint, "sr", settings, vocab)]
    if run_streamed(compose, root, TAG)[0] or run_streamed(verify, root, TAG)[0]:
        raise SystemExit("FATAL: the SR checkpoint failed to compose or verify")

    reports = []
    for lane, arm, tech in LANES:
        clear_compile_caches()
        out = work / f"{lane}.json"
        out.unlink(missing_ok=True)
        cmd = lane_command(
            lane, arm, grid_file, out, settings,
            base_model=base_model, lora_root=fleets[tech], checkpoint=checkpoint,
        )  # fmt: skip
        if run_streamed([python, *cmd], root, TAG)[0] or not out.is_file():
            raise SystemExit(f"FATAL: lane {lane} failed")
        reports.append(json.loads(out.read_text()))

    race = race_record(
        spec, settings, commit_info(Path(os.environ["ADAPTER_BENCH_REPO"])), reports
    )
    (work / "race.json").write_text(json.dumps(race))
    for lane, meta in race["metadata"]["lanes"].items():
        print(
            f"[{TAG}] {lane}: all agents done in {meta['wall_s']:.1f}s, "
            f"first prefill {meta['first_prefill_tokens']:,} tokens, "
            f"re-prefilled {meta['reprefilled_tokens']:,}",
            flush=True,
        )
    print(encode_block(race), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
