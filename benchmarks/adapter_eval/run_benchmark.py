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
import collections
import datetime as dt
import functools
import getpass
import json
import os
import shutil
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path

from . import staged
from . import throughput as switch_bench
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
# The --only entry that measures decode throughput (with intrinsics, or alone).
THROUGHPUT = "throughput"


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


def generation_prompt_anchor(
    base_model: str, template_kwargs: dict | None = None
) -> tuple[str, int]:
    """The last token of the base chat template's generation prompt, as (text, id).

    The composer places an SR adapter's control token from it. Rendered with
    the SR cells' chat-template options (Granite 4.2: reasoning off), as
    their prompts are.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    messages = [{"role": "user", "content": "Hello"}]
    without, with_prompt = (
        tokenizer.apply_chat_template(
            messages, add_generation_prompt=g, tokenize=False, **(template_kwargs or {})
        )
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
    # vLLM 0.26 samples with FlashInfer, which compiles its kernel on first use
    # and needs the ninja build tool, absent from the pod image. Earlier vLLM
    # sampled natively, as this keeps it; greedy decoding picks the same tokens.
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
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


def clear_compile_caches() -> None:
    """Remove vLLM's, Triton's, FlashInfer's and TorchInductor's caches.

    As the switch benchmark's sweep does before each engine (its
    ``clear_caches``), so a warm autotune cache cannot favour the engine that
    runs later. The folders follow the cache variables, which this pod sets.
    """
    home = Path.home()
    xdg = Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache")
    try:
        user = getpass.getuser()
    except Exception:  # no passwd entry, as in some containers
        user = "user"
    for folder in (
        Path(os.environ.get("VLLM_CACHE_ROOT") or xdg / "vllm"),
        Path(os.environ.get("TRITON_CACHE_DIR") or home / ".triton" / "cache"),
        xdg / "flashinfer",
        Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR") or f"/tmp/torchinductor_{user}"),
    ):
        shutil.rmtree(folder, ignore_errors=True)


def time_engine(python: str, harness_root: Path, spec: dict) -> dict:
    """One engine's throughput entry from the switch benchmark's driver, or an error.

    ``spec``: the technology, the engine, the model, the adapter folders, the
    settings and the dump file (``throughput.command``).
    """
    dump = Path(spec["dump"])
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.unlink(missing_ok=True)
    cmd = [
        python,
        str(harness_root / switch_bench.DRIVER),
        *switch_bench.command(
            spec["technology"],
            spec["engine"],
            spec["model"],
            spec["paths"],
            spec["settings"],
            dump,
        ),
    ]
    env = dict(os.environ)
    # Unlike generation, vLLM keeps its engine in its own process, as a server
    # and that benchmark do; its sweep also samples natively (see generate()).
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print(f"[throughput] {' '.join(cmd)}", flush=True)
    tail: collections.deque[str] = collections.deque(maxlen=50)
    with subprocess.Popen(
        cmd,
        cwd=harness_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        for line in proc.stdout:  # into the pod log as it comes
            print(line, end="", flush=True)
            tail.append(line.rstrip())
    if proc.returncode != 0 or not dump.is_file():
        return error(switch_bench.failure(list(tail)))
    return switch_bench.read_entry(dump, spec["settings"])


def throughput_fleets(
    bench_root: Path, spec, base_model: str, copies: Path, sr_anchor
) -> tuple[dict[str, list[dict]], list[str]]:
    """Per technology, the adapters both throughput engines serve.

    Every staged adapter of the model, whatever ``--only`` picked, so a
    technology's fleet is the same on every run. Each is converted as for the
    accuracy run (SR anchor, MLP names), and both engines get that folder, as
    that benchmark gives both its arms the same files; stock vLLM gets an aLoRA
    without its invocation tokens, which would keep it off after a one-token
    prompt (``staged.peft_sr_copy``). An adapter whose conversion fails, or that
    names modules the base model lacks, is left out and named in the second
    value. ``sr_anchor`` returns the SR anchor.
    """
    fleets: dict[str, list[dict]] = {t.id: [] for t in spec.technologies}
    left_out: list[str] = []
    base_modules: set[str] | None = None
    for cell in staged.discover(bench_root, spec, None):
        if cell.skip_reason:
            continue
        key = f"{cell.intrinsic}/{cell.tech}"
        dest = copies / "throughput" / cell.intrinsic / cell.tech
        native = copies / "native" / cell.intrinsic / cell.tech
        path = cell.adapter_dir
        try:
            if cell.tech == "sr" and staged.sr_anchor_copy(path, dest, sr_anchor()):
                path = dest
            if staged.mlp_key_copy(path, dest):
                path = dest
            if base_modules is None:
                base_modules = staged.model_modules(base_model_dir(base_model))
            if staged.modules_missing_from_base(path, base_modules):
                raise ValueError("it names modules the base model lacks")
            if not (cell.tech == "alora" and staged.peft_sr_copy(path, native)):
                native = path
        except Exception as e:
            print(f"[throughput] {key}: left out ({e})", flush=True)
            left_out.append(key)
            continue
        fleets[cell.tech].append({"cell": cell, "compose": path, "native": native})
    return fleets, left_out


def run_throughput(
    args, spec, harness_root: Path, work: Path, model_root: Path, base_model: str,
    flags: set[str], fleets: dict[str, list[dict]], composed: dict[str, set[str]],
) -> dict:  # fmt: skip
    """Per technology, granite-switch's decode throughput and stock vLLM's.

    The switch benchmark's comparison: one technology's adapters composed into
    a checkpoint of their own, against the same adapters on stock vLLM, the two
    engines one after the other on this GPU, which one goes first alternating
    by technology. ``composed`` names the adapters of each checkpoint the
    accuracy run composed; one with exactly a fleet's adapters is reused.
    """
    out: dict[str, dict] = {}
    for k, tech in enumerate(spec.technologies):
        fleet = fleets[tech.id]
        if not fleet:
            out[tech.id] = skipped("no adapter staged")
            continue
        cells = [f["cell"] for f in fleet]
        names = {adapter_name(c.intrinsic, tech.id) for c in cells}
        group = compose_group(tech.id)
        if COMPOSE_GROUPS[group] == (tech.id,) and composed.get(group) == names:
            checkpoint, ready = model_root / group, True
        else:
            checkpoint = model_root / f"throughput_{tech.id}"
            ready = compose(
                args.python,
                args.repo_dir,
                compose_manifest(
                    cells, {(f["cell"].intrinsic, tech.id): f["compose"] for f in fleet}
                ),
                work / "manifests" / f"throughput_{tech.id}.yaml",
                base_model,
                checkpoint,
                flags,
            )
        entries = {}
        for engine in ("gs", "native") if k % 2 == 0 else ("native", "gs"):
            if engine == "gs" and not ready:
                entries[engine] = error("compose failed")
                continue
            clear_compile_caches()
            entries[engine] = time_engine(
                args.python,
                harness_root,
                {
                    "technology": tech.id,
                    "engine": engine,
                    "model": str(checkpoint if engine == "gs" else base_model),
                    "paths": [str(f["native"]) for f in fleet],
                    "settings": spec.throughput.settings(),
                    "dump": str(work / "throughput" / f"{tech.id}_{engine}.jsonl"),
                },
            )
        out[tech.id] = {"gs": entries["gs"], "native": entries["native"]}
        print(f"[throughput] {tech.id}: {json.dumps(speeds(out[tech.id]))}", flush=True)
    return out


def speeds(entry: dict) -> dict:
    """An entry's tokens/s per engine, for the log."""
    return {e: entry[e].get("tokens_per_s", entry[e].get("error")) for e in entry}


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
    measure_speed = only is None or THROUGHPUT in only
    picked = None if only is None else [o for o in only if o != THROUGHPUT]
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
    for cell in staged.discover(args.bench_root, spec, picked) if picked != [] else []:
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
    composed: dict[str, set[str]] = {}
    for group, group_cells in runnable.items():
        paths = {(c.intrinsic, c.tech): c.adapter_dir for c in group_cells}
        for c in [c for c in group_cells if c.tech == "sr"]:
            dest = copies / c.intrinsic / c.tech
            try:
                anchor = anchor or generation_prompt_anchor(
                    base_model, spec.model.prompt_for("sr")["chat_template_kwargs"]
                )
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
        composed[group] = {adapter_name(c.intrinsic, c.tech) for c in group_cells}

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
                    **spec.model.prompt_for(c.tech),
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
            cell["score_version"] = spec.intrinsic(c.intrinsic).score_version
            cell["truncated"] = st.get("truncated", 0)
            if st.get("too_long"):
                cell["too_long"] = st["too_long"]
            cells[c.intrinsic][c.tech] = cell

    throughput = None
    if measure_speed:

        @functools.cache
        def sr_anchor():
            return anchor or generation_prompt_anchor(
                base_model, spec.model.prompt_for("sr")["chat_template_kwargs"]
            )

        fleets, run_meta["throughput_left_out"] = throughput_fleets(
            args.bench_root, spec, base_model, copies, sr_anchor
        )
        throughput = run_throughput(
            args, spec, harness_root, work, model_root, base_model, flags, fleets,
            composed,
        )  # fmt: skip

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
            # The run's folder under the work root, where its answers stay.
            "run_ts": os.environ.get("RUN_TS"),
            "harness_dirty": os.environ.get("ADAPTER_BENCH_HARNESS_DIRTY") == "1",
            "throughput": {
                **spec.throughput.settings(),
                **switch_bench.PINNED,
                "driver": switch_bench.DRIVER,
            },
            **run_meta,
        },
    }
    if throughput is not None:
        results["throughput"] = throughput
    print_table(cells, spec)
    out = args.results_out or work / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(format_results_block(results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
