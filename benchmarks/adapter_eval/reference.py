# SPDX-License-Identifier: Apache-2.0
"""Reference columns: the staged checkpoints without granite-switch, and the base.

Runs on a GPU pod in its own virtualenv (pinned torch, transformers and peft;
see ``vela/pod_entry.sh``), from the harness checkout::

    cd <harness> && <refenv>/bin/python -m benchmarks.adapter_eval.reference \\
        --bench-root <staged adapters + eval> --work-dir <outputs> \\
        [--limit 20] [--only answerability,guardian_core/sr]

Four columns per intrinsic (``common.REFERENCE_COLUMNS``):

* ``lora``, ``alora``, ``sr``: the checkpoint the granite-switch run composes,
  loaded with HF + PEFT instead. SR runs on the shadow-residual repo's model
  code, shipped to the pod at a pinned commit.
* ``base``: the base model with no adapter.

None of them depends on the granite-switch commit, so they are computed once
per ``(bench_version, reference_version)``; ``publish.py`` shows them on every
row of that benchmark version.

Steps:

1. Find the staged cells (``staged.py``), as ``run_benchmark.py`` does.
2. Convert copies where PEFT needs it: an SR checkpoint's invocation tokens
   are dropped (``staged.peft_sr_copy``) and MLP weights saved without their
   ``mlp.`` level are renamed (``staged.mlp_key_copy``). A checkpoint with a
   LoRA weight that still names no base-model weight is an error cell.
3. Generate every cell in its own process on one GPU (``hf_generate.py``), as
   many at once as there are GPUs, the longest cells first.
4. Score every cell and print the reference block (``common.py``) that
   ``publish.py`` reads from the pod log.

Everything the run produced (jobs, predictions, full score reports) is kept
under ``--work-dir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

from . import staged
from .common import (
    BASE_COLUMN,
    REFERENCE_COLUMNS,
    REFERENCE_LIBRARIES,
    error,
    format_reference_block,
    load_spec,
    skipped,
)
from .run_benchmark import base_model_dir, now, print_table, score_cell
from .scorers import ScorerUnavailable

GENERATE_MODULE = "benchmarks.adapter_eval.hf_generate"


def parse_only(text: str | None, spec) -> dict[str, tuple[str, ...]] | None:
    """``--only`` as ``{intrinsic: columns}`` in spec order; None runs everything.

    Entries are comma-separated, each an intrinsic (all four columns) or
    ``intrinsic/column``, e.g. ``answerability,guardian_core/sr``.
    """
    wanted: dict[str, set[str]] = {}
    for entry in (e.strip() for e in (text or "").split(",")):
        if not entry:
            continue
        intrinsic, _, column = entry.partition("/")
        spec.select([intrinsic])  # raises on an unknown intrinsic
        if column and column not in REFERENCE_COLUMNS:
            raise ValueError(
                f"unknown column {column!r}; known: {list(REFERENCE_COLUMNS)}"
            )
        wanted.setdefault(intrinsic, set()).update(
            [column] if column else REFERENCE_COLUMNS
        )
    if not wanted:
        return None
    return {
        i.id: tuple(c for c in REFERENCE_COLUMNS if c in wanted[i.id])
        for i in spec.intrinsics
        if i.id in wanted
    }


def gpu_ids(count: int | None) -> list[str]:
    """The GPUs to spread cells over: ``CUDA_VISIBLE_DEVICES``, else every GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [g.strip() for g in visible.split(",") if g.strip()]
    else:
        import torch

        ids = [str(i) for i in range(torch.cuda.device_count())]
    ids = ids[:count] if count else ids
    if not ids:
        raise SystemExit("no GPUs to run the reference cells on")
    return ids


def versions() -> dict[str, str | None]:
    out = {}
    for name in REFERENCE_LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def sr_code_available() -> bool:
    """Whether the shadow-residual model code was shipped (is on ``PYTHONPATH``).

    Checked without importing it: the import happens in the cell's own
    process, where a failure fails only that cell.
    """
    spec = importlib.util.find_spec("shadow_residual")
    return spec is not None and any(
        (Path(p) / "shadow_residual" / "build.py").is_file()
        for p in spec.submodule_search_locations or ()
    )


def count_rows(path: Path, limit: int | None) -> int:
    with path.open() as f:
        n = sum(1 for line in f if line.strip())
    return n if limit is None else min(n, limit)


def job_status(job: dict, rc: int) -> dict:
    status_path = Path(job["status_path"])
    if rc != 0 or not status_path.is_file():
        print(f"[generate] {job['key']}: process exited with {rc}", flush=True)
        return {"ok": False, "reason": "generation failed"}
    return json.loads(status_path.read_text())


def run_jobs(
    jobs: list[dict],
    gpus: list[str],
    python: str,
    harness_root: Path,
    module: str = GENERATE_MODULE,
    poll: float = 10.0,
) -> dict[str, dict]:
    """Run each job in its own process on one GPU, as many at once as GPUs.

    Jobs start in list order, each on the next free GPU. Returns every job's
    status (``hf_generate.py``) by key.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(harness_root), env.get("PYTHONPATH")) if p
    )
    pending = list(jobs)
    free = list(gpus)
    running: dict[str, tuple[subprocess.Popen, dict]] = {}
    statuses: dict[str, dict] = {}
    try:
        while pending or running:
            while pending and free:
                job, gpu = pending.pop(0), free.pop(0)
                status_path = Path(job["status_path"])
                status_path.parent.mkdir(parents=True, exist_ok=True)
                status_path.unlink(missing_ok=True)
                job_path = status_path.parent / f"{job['column']}.job.json"
                job_path.write_text(json.dumps(job, indent=2))
                print(f"[generate] gpu {gpu}: {job['key']}", flush=True)
                running[gpu] = (
                    subprocess.Popen(
                        [python, "-m", module, str(job_path)],
                        env={**env, "CUDA_VISIBLE_DEVICES": gpu},
                    ),
                    job,
                )
            time.sleep(poll)
            for gpu, (proc, job) in list(running.items()):
                rc = proc.poll()
                if rc is not None:
                    del running[gpu]
                    free.append(gpu)
                    statuses[job["key"]] = job_status(job, rc)
    finally:
        for proc, _ in running.values():
            proc.kill()
    return statuses


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bench-root", required=True, type=Path)
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="where converted adapter copies go (default: <work-dir>/models)",
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="local copy of the adapters.yaml base model (default: download it)",
    )
    p.add_argument("--limit", type=int, default=None, help="rows per eval set")
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated intrinsic ids or intrinsic/column pairs, "
        "e.g. answerability,guardian_core/sr",
    )
    p.add_argument("--gpus", type=int, default=None, help="default: every GPU")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument(
        "--token-budget",
        type=int,
        default=262144,
        help="padded prompt plus new tokens per batch",
    )
    p.add_argument("--max-batch", type=int, default=64, help="rows per batch")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--results-out", type=Path, default=None)
    args = p.parse_args(argv)

    started = now()
    spec = load_spec()
    only = parse_only(args.only, spec)
    work = args.work_dir.resolve()
    copies = (args.model_dir or work / "models").resolve() / "reference"
    harness_root = Path(__file__).resolve().parents[2]
    base_model = args.base_model or spec.base_model
    print(
        f"[reference] bench_version {spec.bench_version}, "
        f"reference_version {spec.reference_version}",
        flush=True,
    )

    discovered = {
        (c.intrinsic, c.tech): c
        for c in staged.discover(args.bench_root, spec, list(only) if only else None)
    }
    has_sr_code = sr_code_available()
    cells: dict[str, dict[str, dict]] = {}
    runnable: list[tuple[str, str, Path | None, Path]] = []
    for intrinsic in spec.select(list(only) if only else None):
        cells[intrinsic.id] = {}
        for column in only[intrinsic.id] if only else REFERENCE_COLUMNS:
            if column == BASE_COLUMN:
                adapter = None
                ev = staged.eval_path(args.bench_root, intrinsic.id)
                reason = None if ev.is_file() else "eval set not staged"
            else:
                c = discovered[intrinsic.id, column]
                adapter, ev, reason = c.adapter_dir, c.eval_path, c.skip_reason
            if reason:
                cells[intrinsic.id][column] = skipped(reason)
            elif column == "sr" and not has_sr_code:
                reason = "SR model code not available"
                cells[intrinsic.id][column] = error(reason)
            else:
                cells[intrinsic.id][column] = error("not run")
                runnable.append((intrinsic.id, column, adapter, ev))
            print(f"[stage] {intrinsic.id}/{column}: {reason or 'ok'}")

    base_dir = base_model_dir(base_model) if runnable else None
    run_meta: dict = {"sr_invocation_dropped": [], "mlp_keys_renamed": []}
    base_modules: set[str] | None = None
    jobs = []
    for intrinsic_id, column, adapter, ev in runnable:
        key = f"{intrinsic_id}/{column}"
        if adapter is not None:
            dest = copies / intrinsic_id / column
            try:
                if column == "sr":
                    change = staged.peft_sr_copy(adapter, dest)
                    if change:
                        adapter = dest
                        run_meta["sr_invocation_dropped"].append(intrinsic_id)
                        print(f"[convert] {key}: {json.dumps(change)}")
                change = staged.mlp_key_copy(adapter, dest)
                if change:
                    adapter = dest
                    run_meta["mlp_keys_renamed"].append(key)
                    print(f"[convert] {key}: {json.dumps(change)}")
                if base_modules is None:
                    base_modules = staged.model_modules(base_dir)
                missing = staged.modules_missing_from_base(adapter, base_modules)
            except Exception as e:
                traceback.print_exc()
                cells[intrinsic_id][column] = error(
                    f"adapter check failed ({type(e).__name__})"
                )
                continue
            if missing:
                print(
                    f"[check] {key}: {len(missing)} LoRA modules with no base "
                    f"weight, e.g. {missing[:3]}"
                )
                cells[intrinsic_id][column] = error(
                    "adapter weights name modules the base model lacks"
                )
                continue
        jobs.append(
            {
                "key": key,
                "column": column,
                "base_model": str(base_dir),
                "adapter_dir": str(adapter) if adapter else None,
                "eval_path": str(ev),
                "out_path": str(
                    work / "predictions" / intrinsic_id / f"{column}.jsonl"
                ),
                "status_path": str(
                    work / "generate" / intrinsic_id / f"{column}.status.json"
                ),
                "limit": args.limit,
                "max_new_tokens": spec.intrinsic(intrinsic_id).max_new_tokens,
                "max_model_len": args.max_model_len,
                "token_budget": args.token_budget,
                "max_batch": args.max_batch,
            }
        )
    # Longest first, so the cells still running at the end are short ones.
    jobs.sort(
        key=lambda j: count_rows(Path(j["eval_path"]), args.limit)
        * j["max_new_tokens"],
        reverse=True,
    )

    gpus = gpu_ids(args.gpus) if jobs else []
    statuses = run_jobs(jobs, gpus, args.python, harness_root) if jobs else {}
    gpu = None
    for job in jobs:
        intrinsic_id, column = job["key"].split("/")
        st = statuses.get(job["key"], {"ok": False, "reason": "generation failed"})
        if not st.get("ok"):
            cells[intrinsic_id][column] = error(st.get("reason", "generation failed"))
            continue
        gpu = gpu or st.get("gpu")
        try:
            metrics = score_cell(
                spec.intrinsic(intrinsic_id).scorer,
                Path(job["out_path"]),
                Path(job["eval_path"]).parent,
                work / "scores" / intrinsic_id / f"{column}.json",
            )
        except ScorerUnavailable as e:
            cells[intrinsic_id][column] = skipped(str(e))
            continue
        except Exception as e:
            traceback.print_exc()
            cells[intrinsic_id][column] = error(f"scoring failed ({type(e).__name__})")
            continue
        cell = dict(metrics)
        cell["truncated"] = st.get("truncated", 0)
        if st.get("too_long"):
            cell["too_long"] = st["too_long"]
        cells[intrinsic_id][column] = cell

    reference = {
        "reference_version": spec.reference_version,
        "bench_version": spec.bench_version,
        "base_model": spec.base_model,
        "cells": cells,
        "run": {
            "started": started,
            "finished": now(),
            "limit": args.limit,
            "only": [f"{i}/{c}" for i, cols in only.items() for c in cols]
            if only
            else None,
            "max_model_len": args.max_model_len,
            "token_budget": args.token_budget,
            "max_batch": args.max_batch,
            "base_model_local_copy": base_model != spec.base_model,
            "harness_sha": os.environ.get("ADAPTER_BENCH_HARNESS_SHA"),
            "harness_dirty": os.environ.get("ADAPTER_BENCH_HARNESS_DIRTY") == "1",
            "sr_ref": os.environ.get("ADAPTER_BENCH_SR_REF"),
            "gpu": gpu,
            "gpus": len(gpus),
            **versions(),
            **run_meta,
        },
    }
    print_table(cells, spec)
    out = args.results_out or work / "reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reference, indent=2, sort_keys=True))
    print(format_reference_block(reference), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
