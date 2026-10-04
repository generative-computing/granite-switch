# SPDX-License-Identifier: Apache-2.0
"""Score saved answers again with the current scorers, without generating.

When only an intrinsic's scoring changes (its ``score_version`` in
``adapters.yaml``), the answers every run saved stay valid, so they are
scored again instead of generated again. Runs on a pod with the storage volume
mounted, from the harness checkout::

    python -m benchmarks.adapter_eval.rescore --targets targets.json \\
        --work-root <runs root> --bench-root <staged adapters + eval> [--model <id>]

``targets.json`` lists the cells to score again (``publish.py
rescore-targets``). Each names whether it is a commit's row or the reference
columns, the commit, the intrinsic and column, and the run that produced it:
its ``finished`` time and, for runs that recorded it, its folder (``run_ts``).
A run's answers stay where it left them::

    <work root>/<model>/<sha12>/<run_ts>/predictions/<intrinsic>/<tech>.jsonl
    <work root>/reference/<model>/<run_ts>/predictions/<intrinsic>/<column>.jsonl

Runs from before there were several models have no ``<model>`` level (the
first model's); a run without ``run_ts`` is found by its ``finished`` time.

Prints the rescore block (``common.py``) that ``publish.py merge-rescore``
reads from the pod log: each cell's new metrics, or why it could not be scored
again. Standard library plus ``yaml``.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from . import staged
from .common import RESCORE_BEGIN, RESCORE_END, format_block, load_spec
from .run_benchmark import now, score_cell
from .scorers import ScorerUnavailable

# The record each kind of run writes into its folder, with its ``finished``.
RECORDS = {"row": "results.json", "reference": "reference.json"}


def run_bases(work_root: Path, entry: dict, spec) -> list[Path]:
    """The folders holding the target's runs, one per run, newest layout first."""
    model = spec.model_id
    first = model == spec.models[0].id
    if entry["kind"] == "row":
        sha12 = entry["sha"][:12]
        return [work_root / model / sha12] + ([work_root / sha12] if first else [])
    reference = work_root / "reference"
    return [reference / model] + ([reference] if first else [])


def find_run_dir(work_root: Path, entry: dict, spec) -> Path | None:
    """The folder of the run that produced a target cell, or None."""
    record = RECORDS[entry["kind"]]
    bases = run_bases(work_root, entry, spec)
    if entry.get("run_ts"):
        for base in bases:
            if (base / entry["run_ts"] / record).is_file():
                return base / entry["run_ts"]
    for base in bases:
        for folder in sorted(base.glob("*"), reverse=True):
            try:
                finished = json.loads((folder / record).read_text())["run"]["finished"]
            except (OSError, ValueError, KeyError):
                continue
            if finished == entry["finished"]:
                return folder
    return None


def rescore(entry: dict, work_root: Path, bench_root: Path, spec, stamp: str) -> dict:
    """One target's new cell, or the reason it has none."""
    intrinsic = spec.intrinsic(entry["intrinsic"])
    column = entry["column"]
    result = {
        k: entry[k]
        for k in ("kind", "sha", "finished", "intrinsic", "column")
        if k in entry
    }
    run_dir = find_run_dir(work_root, entry, spec)
    answers = (
        run_dir / "predictions" / intrinsic.id / f"{column}.jsonl" if run_dir else None
    )
    if answers is None or not answers.is_file():
        result["error"] = "saved answers not found"
        return result
    try:
        metrics = score_cell(
            intrinsic.scorer,
            answers,
            staged.eval_path(bench_root, intrinsic.id).parent,
            run_dir / "scores" / intrinsic.id / f"{column}.rescore-{stamp}.json",
        )
    except ScorerUnavailable as e:
        result["error"] = f"scorer unavailable ({e})"
        return result
    except Exception as e:
        traceback.print_exc()
        result["error"] = f"scoring failed ({type(e).__name__})"
        return result
    result["cell"] = {**metrics, "score_version": intrinsic.score_version}
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--work-root", required=True, type=Path)
    p.add_argument("--bench-root", required=True, type=Path)
    p.add_argument(
        "--model", default=None, help="adapters.yaml model id (default: the first)"
    )
    args = p.parse_args(argv)

    started = now()
    spec = load_spec(model=args.model)
    staged.check_bench_root(args.bench_root, spec)
    stamp = started.replace(":", "")
    results = []
    for entry in json.loads(args.targets.read_text()):
        result = rescore(entry, args.work_root, args.bench_root, spec, stamp)
        name = f"{entry['kind']} {entry.get('sha', '')[:8]} {entry['intrinsic']}/{entry['column']}"
        print(f"[rescore] {name}: {result.get('error', 'ok')}", flush=True)
        results.append(result)
    block = {
        "model": spec.model_id,
        "bench_version": spec.bench_version,
        "cells": results,
        "run": {
            "started": started,
            "finished": now(),
            "harness_sha": os.environ.get("ADAPTER_BENCH_HARNESS_SHA"),
            "harness_dirty": os.environ.get("ADAPTER_BENCH_HARNESS_DIRTY") == "1",
        },
    }
    print(format_block(block, RESCORE_BEGIN, RESCORE_END), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
