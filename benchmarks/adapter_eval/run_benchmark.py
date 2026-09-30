# SPDX-License-Identifier: Apache-2.0
"""Benchmark one granite-switch commit: compose, generate, score, report.

Runs on the GPU pod with the benchmarked commit's virtualenv, from the
harness checkout (not the commit's), so the harness is the same for every
commit::

    cd <harness> && <commit>/.venv/bin/python -m benchmarks.adapter_eval.run_benchmark \\
        --repo-dir <commit checkout> --bench-root <staged adapters + eval> \\
        --work-dir <outputs> [--model granite-4.2-3b] [--limit 20] \\
        [--only answerability,guardian_core]

Steps:

1. Find the staged cells (``staged.py``) of the model's bench root; a
   missing or malformed checkpoint becomes a skipped cell.
2. Compose twice with the commit's composer CLI: all LoRA + aLoRA adapters
   into one checkpoint, all SR adapters into another. SR is a
   whole-checkpoint dual-stream mode and cannot be mixed with the others.
   SR checkpoints that turn on at invocation tokens are first converted to
   the composer's single-anchor form (``staged.sr_anchor_copy``), and MLP
   weights saved without their ``mlp.`` level are renamed
   (``staged.mlp_key_copy``). A checkpoint with a LoRA weight that still
   names no base-model weight is an error cell: the composer would drop it
   without a word.
3. Generate greedily for every cell, one subprocess per composed checkpoint.
4. Score every cell and print the results block (``common.py``) that
   ``publish.py`` reads from the pod log. It also records the library
   versions and each staged checkpoint's fingerprint
   (``staged.fingerprint``), for the page's details box.

Everything the run produced (manifests, predictions, full score reports) is
kept under ``--work-dir``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path

from . import staged
from .common import (
    COMPOSE_GROUPS,
    adapter_name,
    compose_group,
    error,
    format_results_block,
    is_scored,
    load_spec,
    skipped,
)
from .scorers import ScoreContext, ScorerUnavailable, get_scorer

COMPOSER_MODULE = "granite_switch.composer.compose_granite_switch"


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def commit_info(repo_dir: Path) -> dict:
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "-1", "--format=%H%x00%cI%x00%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sha, date, subject = out.split("\x00", 2)
    return {"sha": sha, "date": date, "subject": subject}


def composer_flags(python: str, repo_dir: Path) -> set[str]:
    """Flags this commit's composer accepts, read from its ``--help``."""
    res = subprocess.run(
        [python, "-m", COMPOSER_MODULE, "--help"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    return {tok.strip(",[]") for tok in res.stdout.split() if tok.startswith("--")}


def compose(
    python: str,
    repo_dir: Path,
    manifest: dict,
    manifest_path: Path,
    base_model: str,
    out_dir: Path,
    flags: set[str],
) -> bool:
    import yaml

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True))
    cmd = [
        python,
        "-m",
        COMPOSER_MODULE,
        "--adapters",
        str(manifest_path),
        "--base-model",
        base_model,
        "--output",
        str(out_dir),
    ]
    # Staged checkpoints carry no io.yaml; newer composers refuse that
    # unless told to synthesize one.
    if "--create-ioyaml" in flags:
        cmd.append("--create-ioyaml")
    print(f"[compose] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=repo_dir).returncode == 0


def compose_manifest(
    cells: list[staged.StagedCell], paths: dict[tuple[str, str], Path]
) -> dict:
    """The composer's adapter manifest for one compose group.

    ``paths`` maps (intrinsic, technology) to the folder to compose: the
    staged checkpoint or its converted copy. One intrinsic has both a LoRA
    and an aLoRA cell in the same group, so the technology is part of the key.
    """
    return {
        adapter_name(c.intrinsic, c.tech): {
            "path": str(paths[c.intrinsic, c.tech]),
            # SR is detected by the composer from the weights; the manifest
            # type only chooses between aLoRA and LoRA placement.
            "type": "alora" if c.tech == "alora" else "lora",
        }
        for c in cells
    }


def generation_prompt_anchor(base_model: str) -> tuple[str, int]:
    """The last token of the base chat template's generation prompt, as (text, id).

    This is where the composer puts an SR adapter's control token.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    messages = [{"role": "user", "content": "Hello"}]
    without, with_prompt = (
        tokenizer.apply_chat_template(messages, add_generation_prompt=g, tokenize=False)
        for g in (False, True)
    )
    if not (with_prompt.startswith(without) and len(with_prompt) > len(without)):
        raise ValueError("the chat template adds no generation prompt at its end")
    anchor_id = tokenizer.encode(with_prompt, add_special_tokens=False)[-1]
    return tokenizer.decode([anchor_id]), anchor_id


def versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Installed versions of ``names``; None for one that is not installed."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def base_model_dir(base_model: str) -> Path:
    if Path(base_model).is_dir():
        return Path(base_model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base_model))


def generate(python: str, harness_root: Path, jobs_path: Path, spec: dict) -> dict:
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(spec, indent=2))
    env = dict(os.environ)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(harness_root), env.get("PYTHONPATH")) if p
    )
    cmd = [python, "-m", "benchmarks.adapter_eval.generate", str(jobs_path)]
    print(f"[generate] {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    status_path = Path(spec["status_path"])
    if rc != 0 or not status_path.is_file():
        return {"failed": f"generation process exited with {rc}"}
    return json.loads(status_path.read_text())


def score_cell(scorer_name: str, pred_path: Path, eval_dir: Path, out_path: Path):
    rows = staged.read_jsonl(pred_path)
    result = get_scorer(scorer_name)(
        rows, ScoreContext(eval_dir=eval_dir, env=dict(os.environ))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"metrics": result.metrics, "details": result.details}, indent=2)
    )
    return result.metrics


def print_table(cells: dict, spec) -> None:
    print("\n[results]")
    for intrinsic_id, by_tech in cells.items():
        headline = spec.intrinsic(intrinsic_id).headline
        parts = []
        for tech_id, cell in by_tech.items():
            if is_scored(cell):
                parts.append(f"{tech_id}={cell.get(headline, float('nan')):.4f}")
            else:
                parts.append(f"{tech_id}=({cell.get('skipped') or cell.get('error')})")
        print(f"  {intrinsic_id:<24} " + "  ".join(parts))
    print(flush=True)


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
        help="where composed checkpoints and converted adapter copies go "
        "(default: <work-dir>/models); local disk is much faster than a COS mount",
    )
    p.add_argument("--repo-dir", required=True, type=Path, help="commit checkout")
    p.add_argument(
        "--model", default=None, help="adapters.yaml model id (default: the first)"
    )
    p.add_argument(
        "--base-model",
        default=None,
        help="local copy of the model's base model (default: download it)",
    )
    p.add_argument("--limit", type=int, default=None, help="rows per eval set")
    p.add_argument("--only", default=None, help="comma-separated intrinsic ids")
    p.add_argument("--results-out", type=Path, default=None)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--prefix-caching", action="store_true")
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="engine token budget per step (default: vLLM's)",
    )
    p.add_argument(
        "--no-chunked-prefill",
        action="store_true",
        help="never split a prompt across engine steps; the token budget "
        "defaults to --max-model-len",
    )
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args(argv)

    started = now()
    spec = load_spec(model=args.model)
    staged.check_bench_root(args.bench_root, spec)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    work = args.work_dir.resolve()
    model_root = (args.model_dir or work / "models").resolve()
    copies = model_root / "adapters"
    harness_root = Path(__file__).resolve().parents[2]
    base_model = args.base_model or spec.base_model
    commit = commit_info(args.repo_dir)
    print(f"[bench] commit {commit['sha']} — {commit['subject']}", flush=True)
    print(f"[bench] model {spec.model_id} ({spec.base_model})", flush=True)

    cells: dict[str, dict[str, dict]] = {}
    runnable: dict[str, list[staged.StagedCell]] = {g: [] for g in COMPOSE_GROUPS}
    for cell in staged.discover(args.bench_root, spec, only):
        cells.setdefault(cell.intrinsic, {})
        if cell.skip_reason:
            cells[cell.intrinsic][cell.tech] = skipped(cell.skip_reason)
        else:
            runnable[compose_group(cell.tech)].append(cell)
            cells[cell.intrinsic][cell.tech] = error("not run")
        print(f"[stage] {cell.intrinsic}/{cell.tech}: {cell.skip_reason or 'ok'}")

    flags = composer_flags(args.python, args.repo_dir)
    anchor = None
    token_budget = args.max_num_batched_tokens or (
        args.max_model_len if args.no_chunked_prefill else None
    )
    run_meta: dict = {
        "vllm": None,
        "gpu": None,
        "scheduler": None,
        "sr_anchor_converted": [],
        "mlp_keys_renamed": [],
        # The versions the commit's lockfile installed.
        **versions(("torch", "transformers")),
        "adapters": {
            f"{c.intrinsic}/{c.tech}": staged.fingerprint(c.adapter_dir)
            for group_cells in runnable.values()
            for c in group_cells
        },
    }
    base_modules: set[str] | None = None
    for group, group_cells in runnable.items():
        paths = {(c.intrinsic, c.tech): c.adapter_dir for c in group_cells}
        for c in [c for c in group_cells if c.tech == "sr"]:
            dest = copies / c.intrinsic / c.tech
            try:
                anchor = anchor or generation_prompt_anchor(base_model)
                change = staged.sr_anchor_copy(c.adapter_dir, dest, anchor)
            except Exception as e:
                traceback.print_exc()
                cells[c.intrinsic][c.tech] = error(
                    f"anchor conversion failed ({type(e).__name__})"
                )
                group_cells.remove(c)
                continue
            if change:
                paths[c.intrinsic, c.tech] = dest
                run_meta["sr_anchor_converted"].append(c.intrinsic)
                print(f"[convert] {c.intrinsic}/sr: {json.dumps(change)}")
        for c in list(group_cells):
            key = (c.intrinsic, c.tech)
            dest = copies / c.intrinsic / c.tech
            try:
                change = staged.mlp_key_copy(paths[key], dest)
                if change:
                    paths[key] = dest
                    run_meta["mlp_keys_renamed"].append(f"{c.intrinsic}/{c.tech}")
                    print(f"[convert] {c.intrinsic}/{c.tech}: {json.dumps(change)}")
                if base_modules is None:
                    base_modules = staged.model_modules(base_model_dir(base_model))
                missing = staged.modules_missing_from_base(paths[key], base_modules)
            except Exception as e:
                traceback.print_exc()
                cells[c.intrinsic][c.tech] = error(
                    f"adapter check failed ({type(e).__name__})"
                )
                group_cells.remove(c)
                continue
            if missing:
                print(
                    f"[check] {c.intrinsic}/{c.tech}: {len(missing)} LoRA modules "
                    f"with no base weight, e.g. {missing[:3]}"
                )
                cells[c.intrinsic][c.tech] = error(
                    "adapter weights name modules the base model lacks"
                )
                group_cells.remove(c)
        if not group_cells:
            continue
        model_dir = model_root / group
        if not compose(
            args.python,
            args.repo_dir,
            compose_manifest(group_cells, paths),
            work / "manifests" / f"{group}.yaml",
            base_model,
            model_dir,
            flags,
        ):
            for c in group_cells:
                cells[c.intrinsic][c.tech] = error("compose failed")
            continue

        jobs = []
        for c in group_cells:
            intrinsic = spec.intrinsic(c.intrinsic)
            jobs.append(
                {
                    "key": f"{c.intrinsic}/{c.tech}",
                    "adapter_name": adapter_name(c.intrinsic, c.tech),
                    "eval_path": str(c.eval_path),
                    "out_path": str(
                        work / "predictions" / c.intrinsic / f"{c.tech}.jsonl"
                    ),
                    "limit": args.limit,
                    "max_new_tokens": intrinsic.max_new_tokens,
                }
            )
        status = generate(
            args.python,
            harness_root,
            work / "generate" / f"{group}.jobs.json",
            {
                "model_dir": str(model_dir),
                "status_path": str(work / "generate" / f"{group}.status.json"),
                "llm": {
                    "max_model_len": args.max_model_len,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "enforce_eager": args.enforce_eager,
                    "enable_prefix_caching": args.prefix_caching,
                    "enable_chunked_prefill": not args.no_chunked_prefill,
                    "max_num_batched_tokens": token_budget,
                },
                "jobs": jobs,
            },
        )
        if "failed" in status:
            for c in group_cells:
                cells[c.intrinsic][c.tech] = error("generation failed")
            continue
        run_meta["vllm"] = run_meta["vllm"] or status.get("vllm")
        run_meta["gpu"] = run_meta["gpu"] or status.get("gpu")
        run_meta["scheduler"] = run_meta["scheduler"] or status.get("scheduler")

        for c, job in zip(group_cells, jobs, strict=True):
            st = status["jobs"].get(
                job["key"], {"ok": False, "reason": "not generated"}
            )
            if not st.get("ok") or "n_generated" not in st:
                cells[c.intrinsic][c.tech] = error(
                    st.get("reason", "generation failed")
                )
                continue
            try:
                metrics = score_cell(
                    spec.intrinsic(c.intrinsic).scorer,
                    Path(job["out_path"]),
                    c.eval_path.parent,
                    work / "scores" / c.intrinsic / f"{c.tech}.json",
                )
            except ScorerUnavailable as e:
                cells[c.intrinsic][c.tech] = skipped(str(e))
                continue
            except Exception as e:
                traceback.print_exc()
                cells[c.intrinsic][c.tech] = error(
                    f"scoring failed ({type(e).__name__})"
                )
                continue
            cell = dict(metrics)
            cell["truncated"] = st.get("truncated", 0)
            if st.get("too_long"):
                cell["too_long"] = st["too_long"]
            cells[c.intrinsic][c.tech] = cell

    results = {
        "model": spec.model_id,
        "bench_version": spec.bench_version,
        "base_model": spec.base_model,
        "commit": commit,
        "cells": cells,
        "run": {
            "started": started,
            "finished": now(),
            "limit": args.limit,
            "only": only,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
            "prefix_caching": args.prefix_caching,
            "base_model_local_copy": base_model != spec.base_model,
            "harness_sha": os.environ.get("ADAPTER_BENCH_HARNESS_SHA"),
            "harness_dirty": os.environ.get("ADAPTER_BENCH_HARNESS_DIRTY") == "1",
            **run_meta,
        },
    }
    print_table(cells, spec)
    out = args.results_out or work / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(format_results_block(results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
