# SPDX-License-Identifier: Apache-2.0
"""Find trained adapters and eval sets, and copy the chosen ones into place.

Runs on a pod with the adapter storage mounted. Two steps, with a human
choice in between::

    # 1. List every adapter checkpoint and eval file under the source roots,
    #    and draft a selection (one checkpoint per cell).
    python -m benchmarks.adapter_eval.stage discover \\
        --adapter-root [<intrinsic>=]<dir> [...] --eval-root [<intrinsic>=]<dir> [...]

    # 2. Copy the reviewed selection into the bench root.
    python -m benchmarks.adapter_eval.stage apply \\
        --selection selection.json --bench-root <dir> [--judge-prompt <file>]

Both take ``--model`` (default: the first in ``adapters.yaml``): the draft
picks checkpoints trained on that base model, and a bench root holds one
model's cells only. ``apply`` records the model in the root and refuses a
root staged for another.

``discover`` prints its findings between markers so they can be read back
from the pod log. Nothing is written by it. A root given as
``<intrinsic>=<dir>`` names the intrinsic of everything under it whose path
names none. A run's own ``predictions.jsonl`` (its eval rows plus the
model's outputs) is listed as an eval candidate: it holds exactly the rows the
run was scored on.

``apply`` copies rather than links, so retraining a source run cannot change
the benchmark underneath it. An eval file that is a predictions file is
copied without the model's outputs. Each staged folder gets a ``provenance.json``
with the source path and file checksums. Existing cells are kept unless
``--replace`` is given; replacing a staged checkpoint changes the numbers, so
it goes with a ``bench_version`` bump.

The selection file::

    {"adapters": {"<intrinsic>": {"<technology>": "<checkpoint dir>"}},
     "eval": {"<intrinsic>": "<file.jsonl>"}}
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
from pathlib import Path

from . import staged
from .common import load_spec
from .scorers.query_rewrite import PROMPT_FILE

DISCOVERY_BEGIN = "=== ADAPTER_BENCH_DISCOVERY_BEGIN ==="
DISCOVERY_END = "=== ADAPTER_BENCH_DISCOVERY_END ==="
STAGE_BEGIN = "=== ADAPTER_BENCH_STAGE_BEGIN ==="
STAGE_END = "=== ADAPTER_BENCH_STAGE_END ==="

# What each benchmarked technology was trained with, as read from the
# weights. Used only for the draft selection; the config names break ties.
EXPECTED = {
    "lora": {"rank": 16, "modules": "qkvo+mlp", "names": ("lora_qkvo_mlp_r16",)},
    "alora": {"rank": 32, "modules": "qkvo+mlp", "names": ("alora_qkvo_mlp_r32",)},
    "sr": {
        "rank": 32,
        "cross_rank": 32,
        "modules": "qo+mlp",
        "shared_kv": True,
        "names": ("sr_qo_mlp_r32_c32_sharedkv",),
    },
}
ATTENTION = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "o_proj": "o",
    "qkv_proj": "qkv",
}
MLP = {
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
    "input_linear",
    "output_linear",
}

# Path keywords that name an intrinsic. Long keys match anywhere in the path;
# short keys must be a whole ``/``- or ``_``-separated token.
INTRINSIC_KEYWORDS = {
    "answerability": (("answerab",), ()),
    "hallucination_detection": (("hallucination",), ("hd",)),
    "query_rewrite": (("query_rewrite", "rewrite"), ("qr",)),
    "query_clarification": (("clarification", "clarif"), ("qc",)),
    "guardian_core": (("guardian", "ood_safety"), ()),
    "requirement_check": (("requirement",), ("req",)),
}

SKIP_DIRS = {".git", "__pycache__", "wandb", ".cache", "node_modules"}
# Intermediate trainer checkpoints. The final adapter sits beside them, and
# skipping them keeps a walk over the storage mount short.
CHECKPOINT_DIR = re.compile(r"^checkpoint[_-]\d+$")
PREDICTIONS_FILE = "predictions.jsonl"
# Fields a predictions file adds to its eval rows.
PREDICTION_KEYS = ("generated_content",)
PROGRESS_EVERY = 500
SCORE_NAME = re.compile(r"(score|metric|result|eval)", re.IGNORECASE)
MAX_SCORE_FILE = 1 << 20


def _norm(path: Path) -> str:
    return str(path).lower().replace("-", "_")


def guess_intrinsic(path: Path) -> str | None:
    text = _norm(path)
    tokens = set(re.split(r"[/_.]", text))
    hits = [
        intrinsic
        for intrinsic, (long_keys, short_keys) in INTRINSIC_KEYWORDS.items()
        if any(k in text for k in long_keys) or tokens & set(short_keys)
    ]
    return hits[0] if len(hits) == 1 else None


def guess_config(path: Path) -> str | None:
    text = _norm(path)
    for tech, exp in EXPECTED.items():
        # "alora_..." must not count as "lora_...".
        if any(re.search(rf"(?<![a-z]){name}", text) for name in exp["names"]):
            return tech
    return None


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size))


def weight_summary(weights: Path) -> dict:
    """Adapter rank, cross-stream rank and adapted modules, from the tensors."""
    rank = cross = None
    names = set()
    for key, meta in safetensors_header(weights).items():
        if key == "__metadata__" or "lora_A" not in key:
            continue
        r = meta["shape"][0]
        if ".cross_stream." in key:
            cross = r
        else:
            rank = r
            names.add(key.split(".lora_A")[0].rsplit(".", 1)[-1])
    return {"rank": rank, "cross_rank": cross, "modules": module_shape(names)}


def module_shape(names: set[str]) -> str | None:
    """{"q_proj", "o_proj", "up_proj", "down_proj"} -> "qo+mlp"."""
    letters = "".join(ATTENTION[n] for n in names if n in ATTENTION)
    parts = ["".join(c for c in "qkvo" if c in letters)] if letters else []
    if names & MLP:
        parts.append("mlp")
    parts += sorted(names - ATTENTION.keys() - MLP)
    return "+".join(parts) or None


def lora_ranks(weights: Path) -> tuple[int | None, int | None]:
    """(adapter rank, cross-stream rank) read from the tensor shapes."""
    summary = weight_summary(weights)
    return summary["rank"], summary["cross_rank"]


def base_model_matches(name: str | None, expected: str) -> bool | None:
    """Whether a checkpoint was trained on ``expected``, compared by model name.

    Local paths and hub cache folders count, e.g.
    ``.../models--ibm-granite--granite-4.1-3b/snapshots/x``. None when the
    checkpoint does not say.
    """
    if not name:
        return None
    want = expected.rstrip("/").rsplit("/", 1)[-1].lower()
    return want in re.split(r"/|--", name.lower())


def parse_root(text: str) -> tuple[Path, str | None]:
    """``"<intrinsic>=<dir>"`` or ``"<dir>"`` -> (dir, intrinsic or None)."""
    hint, sep, path = text.partition("=")
    if sep and hint in INTRINSIC_KEYWORDS:
        return Path(path), hint
    return Path(text), None


def flat_numbers(obj, prefix: str = "", depth: int = 3, out=None) -> dict:
    out = {} if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, int | float):
        out[prefix or "value"] = obj
    elif isinstance(obj, dict) and depth > 0:
        for k, v in obj.items():
            flat_numbers(v, f"{prefix}.{k}" if prefix else str(k), depth - 1, out)
    return out


def nearby_scores(adapter_dir: Path) -> list[dict]:
    """Numeric metrics from score-like JSON files next to a checkpoint."""
    found = []
    for folder in (adapter_dir, adapter_dir.parent, adapter_dir.parent.parent):
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        for f in entries:
            if (
                f.suffix != ".json"
                or f.name == staged.CONFIG_FILE
                or not SCORE_NAME.search(f.name)
                or f.stat().st_size > MAX_SCORE_FILE
            ):
                continue
            try:
                numbers = flat_numbers(json.loads(f.read_text()))
            except (OSError, ValueError):
                continue
            if numbers:
                found.append(
                    {"file": str(f), "metrics": dict(list(numbers.items())[:40])}
                )
    return found


def walk(root: Path, max_depth: int):
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not CHECKPOINT_DIR.match(d)
        )
        if len(here.parts) - root_depth >= max_depth:
            dirnames[:] = []
        yield here, filenames


def mtime(path: Path) -> str:
    ts = path.stat().st_mtime
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def adapter_candidate(folder: Path, root: Path, hint: str | None = None) -> dict:
    # Guesses read only the part below the search root, so the mount's own
    # path cannot match a keyword.
    rel = folder.relative_to(root)
    tech, reason = staged.detect_technology(folder)
    weights = folder / staged.WEIGHTS_FILE
    cand = {
        "path": str(folder),
        "technology": tech,
        "invalid": reason,
        "intrinsic_guess": guess_intrinsic(rel) or hint,
        "config_guess": guess_config(rel),
        "checkpoint": bool(re.search(r"(^|/)checkpoint[_-]\d+", rel.as_posix())),
        "mtime": mtime(weights),
        "size_mb": round(weights.stat().st_size / 2**20, 1),
    }
    if tech == "sr":
        # Only the run's name records the K/V mode (see staged.SHARED_KV).
        cand["shared_kv"] = staged.says_shared_kv(rel.as_posix())
    try:
        config = json.loads((folder / staged.CONFIG_FILE).read_text())
        cand["base_model"] = config.get("base_model_name_or_path")
        cand["target_modules"] = config.get("target_modules")
        cand["alpha"] = config.get("lora_alpha")
        # How the adapter turns on: aLoRA invocation tokens or an SR anchor.
        cand["activation"] = (
            "invocation"
            if config.get("alora_invocation_tokens")
            else "anchor"
            if config.get("last_context_token")
            else None
        )
    except (OSError, ValueError):
        pass
    try:
        cand.update(weight_summary(weights))
    except (OSError, ValueError, KeyError, struct.error):
        pass
    cand["scores"] = nearby_scores(folder)
    predictions = folder / PREDICTIONS_FILE
    if predictions.is_file():
        cand["predictions"] = {
            **eval_candidate(predictions, root, cand["intrinsic_guess"]),
            "run": str(folder),
        }
    return cand


def eval_candidate(path: Path, root: Path, hint: str | None = None) -> dict:
    rows = 0
    first = None
    row_hashes = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except ValueError:
                row = {}
            if first is None:
                first = dict(row) if isinstance(row, dict) else row
            if isinstance(row, dict):
                for k in PREDICTION_KEYS:
                    row.pop(k, None)
            row_hashes.append(hashlib.sha256(json.dumps(row, sort_keys=True).encode()))
    keys = sorted(first) if isinstance(first, dict) else []
    # Order-free, and blind to model outputs: two runs scored on the same rows
    # have the same hash.
    rows_hash = hashlib.sha256()
    for h in sorted(h.hexdigest() for h in row_hashes):
        rows_hash.update(h.encode())
    return {
        "path": str(path),
        "rows": rows,
        "rows_hash": rows_hash.hexdigest()[:16],
        "keys": keys,
        "usable": "messages" in keys and "ground_truth" in keys,
        "intrinsic_guess": guess_intrinsic(path.relative_to(root)) or hint,
        "mtime": mtime(path),
        "size_mb": round(path.stat().st_size / 2**20, 1),
    }


def probe_judge() -> str:
    """Whether the query-rewrite judge answers from this pod."""
    from .scorers.query_rewrite import DEFAULT_JUDGE_MODEL, Judge

    url = os.environ.get("ADAPTER_BENCH_JUDGE_URL")
    key = os.environ.get("RITS_API_KEY")
    if not url or not key:
        return "not configured"
    model = os.environ.get("ADAPTER_BENCH_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
    try:
        Judge(url, key, model, "").complete("Reply with {}")
    except Exception as e:
        return f"unreachable ({type(e).__name__})"
    return "ok"


def fits_cell(a: dict, intrinsic_id: str, tech_id: str, base_model: str) -> bool:
    """Whether a checkpoint has the technology, shape, K/V mode and base model of a cell."""
    exp = EXPECTED[tech_id]
    return (
        a["technology"] == tech_id
        and a["intrinsic_guess"] == intrinsic_id
        and a.get("rank") == exp["rank"]
        and a.get("cross_rank") == exp.get("cross_rank")
        and a.get("modules") == exp["modules"]
        and a.get("shared_kv") == exp.get("shared_kv")
        and base_model_matches(a.get("base_model"), base_model) is not False
        and not a["checkpoint"]
    )


def draft_selection(spec, adapters: list[dict], evals: list[dict]) -> dict:
    """One checkpoint per cell, and one eval set per intrinsic.

    Among the checkpoints that fit a cell, the one whose path names the
    expected configuration wins, then the newest. The eval set is the first
    picked run's own predictions (in technology order), so the benchmark
    scores the rows the internal number was scored on. A note flags runs of
    one intrinsic that were scored on different rows.
    """
    selection = {"adapters": {}, "eval": {}, "notes": {}}
    for intrinsic in spec.intrinsics:
        picked = []
        for tech in spec.technologies:
            fits = [
                a
                for a in adapters
                if fits_cell(a, intrinsic.id, tech.id, spec.base_model)
            ]
            if fits:
                best = max(
                    fits,
                    key=lambda a: (
                        a["config_guess"] == tech.id,
                        base_model_matches(a.get("base_model"), spec.base_model)
                        is True,
                        a["mtime"],
                    ),
                )
                selection["adapters"].setdefault(intrinsic.id, {})[tech.id] = best[
                    "path"
                ]
                picked.append(best)
        preds = [
            a["predictions"] for a in picked if a.get("predictions", {}).get("usable")
        ]
        if preds:
            selection["eval"][intrinsic.id] = preds[0]["path"]
            if len({p["rows_hash"] for p in preds}) > 1:
                selection["notes"][intrinsic.id] = (
                    "the picked runs were scored on different rows: "
                    + ", ".join(f"{p['rows']} rows in {p['run']}" for p in preds)
                )
            continue
        fits = [
            e for e in evals if e["usable"] and e["intrinsic_guess"] == intrinsic.id
        ]
        if fits:
            named = [e for e in fits if re.search(r"(eval|test)", Path(e["path"]).name)]
            selection["eval"][intrinsic.id] = max(
                named or fits, key=lambda e: e["mtime"]
            )["path"]
    return selection


def scan(roots: list[str], max_depth: int, visit) -> None:
    """Call ``visit(folder, files, root, hint)`` for every folder, with progress."""
    for text in roots:
        root, hint = parse_root(text)
        start = time.monotonic()
        folders = 0
        for folder, files in walk(root, max_depth):
            folders += 1
            if folders % PROGRESS_EVERY == 0:
                print(f"[discover] {root}: {folders} folders...", flush=True)
            visit(folder, files, root, hint)
        took = time.monotonic() - start
        print(f"[discover] {root}: {folders} folders in {took:.0f}s", flush=True)


def short_base(name: str | None) -> str:
    return re.split(r"/|--", name.rstrip("/"))[-1] if name else "?"


def cmd_discover(args) -> int:
    spec = load_spec(model=args.model)
    adapters, evals = [], []

    unweighted = []

    def visit_adapter(folder, files, root, hint):
        if staged.CONFIG_FILE in files and staged.WEIGHTS_FILE in files:
            adapters.append(adapter_candidate(folder, root, hint))
        elif PREDICTIONS_FILE in files or folder.name == "checkpoints":
            # A scored run without final weights: list what it holds instead.
            unweighted.append(
                {"path": str(folder), "entries": sorted(os.listdir(folder))}
            )

    def visit_eval(folder, files, root, hint):
        for name in sorted(files):
            if name.endswith(".jsonl"):
                evals.append(eval_candidate(folder / name, root, hint))

    scan(args.adapter_root, args.max_depth, visit_adapter)
    scan(args.eval_root, args.max_depth, visit_eval)
    seen = {e["path"] for e in evals}
    evals += [
        a["predictions"]
        for a in adapters
        if "predictions" in a and a["predictions"]["path"] not in seen
    ]

    print(f"\n[discover] {len(adapters)} adapter checkpoints, {len(evals)} jsonl files")
    for a in sorted(adapters, key=lambda a: a["path"]):
        if a["checkpoint"]:
            continue
        preds = a.get("predictions")
        print(
            f"  {a['technology'] or 'INVALID':<6} r={a.get('rank')!s:<4} "
            f"x={a.get('cross_rank')!s:<4} {a.get('modules') or '?':<10} "
            f"{short_base(a.get('base_model')):<16} "
            f"{a['intrinsic_guess'] or '?':<24} "
            + ("shared_kv " if a.get("shared_kv") else "")
            + f"preds={preds['rows'] if preds else '-'!s:<6} "
            f"{a['path']}" + (f"  [{a['invalid']}]" if a["invalid"] else "")
        )
    for e in sorted(evals, key=lambda e: e["path"]):
        mark = "ok " if e["usable"] else "-- "
        print(
            f"  {mark} rows={e['rows']:<6} {e['intrinsic_guess'] or '?':<24} {e['path']}"
        )
    for u in unweighted:
        print(f"  no weights: {u['path']}: {' '.join(u['entries'][:20])}")
    judge = probe_judge()
    print(f"[discover] judge: {judge}")

    report = {
        "adapters": adapters,
        "evals": evals,
        "unweighted": unweighted,
        "judge": judge,
        "draft_selection": draft_selection(spec, adapters, evals),
    }
    print(DISCOVERY_BEGIN)
    print(json.dumps(report, sort_keys=True))
    print(DISCOVERY_END, flush=True)
    return 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_into(
    files: dict[str, Path], dest: Path, provenance: dict, replace: bool
) -> dict:
    """Copy ``{name: source}`` into ``dest`` via a temporary folder, then swap."""
    if dest.exists() and not replace:
        raise FileExistsError(f"{dest} already staged (use --replace)")
    tmp = dest.with_name(dest.name + ".staging")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    checksums = {}
    for name, src in files.items():
        target = tmp / name
        shutil.copyfile(src, target)
        checksums[target.name] = sha256(target)
        if checksums[target.name] != sha256(src):
            raise OSError(f"checksum mismatch copying {src}")
    provenance = {**provenance, "files": checksums, "staged_at": now()}
    (tmp / staged.PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return provenance


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_eval_rows(path: Path) -> int:
    n = 0
    for row in staged.read_jsonl(path):
        missing = {"messages", "ground_truth"} - set(row)
        if missing:
            raise ValueError(f"{path}: row {n} lacks {sorted(missing)}")
        n += 1
    if not n:
        raise ValueError(f"{path}: empty")
    return n


def without_predictions(src: Path, tmp_dir: Path) -> tuple[Path, list[str]]:
    """``src``, or a copy in ``tmp_dir`` without the model-output fields."""
    rows = staged.read_jsonl(src)
    found = sorted({k for row in rows for k in PREDICTION_KEYS if k in row})
    if not found:
        return src, []
    clean = tmp_dir / staged.EVAL_FILE
    staged.write_jsonl(
        clean, [{k: v for k, v in r.items() if k not in found} for r in rows]
    )
    return clean, found


def check_judge_prompt(path: Path) -> None:
    slots = dict.fromkeys(
        (
            "previous_question",
            "previous_answer",
            "current_question",
            "golden_rewritten_question",
            "rewritten_question",
        ),
        "x",
    )
    path.read_text().format(**slots)


def cmd_apply(args) -> int:
    spec = load_spec(model=args.model)
    selection = json.loads(args.selection.read_text())
    root = args.bench_root
    report = {"model": spec.model_id, "adapters": {}, "eval": {}}
    failures = 0
    staged.check_bench_root(root, spec)
    root.mkdir(parents=True, exist_ok=True)
    (root / staged.MODEL_FILE).write_text(
        json.dumps({"model": spec.model_id, "name": spec.base_model}, indent=2)
    )

    for intrinsic_id, by_tech in selection.get("adapters", {}).items():
        spec.intrinsic(intrinsic_id)
        for tech_id, src in by_tech.items():
            src = Path(src)
            key = f"{intrinsic_id}/{tech_id}"
            try:
                reason = staged.check_adapter(src, tech_id)
                if reason:
                    raise ValueError(reason)
                extra = {}
                if tech_id == "sr":
                    if not staged.says_shared_kv(str(src)):
                        raise ValueError(
                            "not a shared-K/V run: its path does not say sharedkv"
                        )
                    extra["shared_kv"] = True
                names = [staged.CONFIG_FILE, staged.WEIGHTS_FILE, "io.yaml"]
                files = {n: src / n for n in names if (src / n).is_file()}
                rank, cross = lora_ranks(src / staged.WEIGHTS_FILE)
                prov = copy_into(
                    files,
                    staged.adapter_dir(root, intrinsic_id, tech_id),
                    {
                        "source": str(src),
                        "technology": tech_id,
                        "rank": rank,
                        "cross_rank": cross,
                        **extra,
                    },
                    args.replace,
                )
                report["adapters"][key] = prov
                print(f"[stage] adapter {key}: ok (r={rank})")
            except Exception as e:
                failures += 1
                report["adapters"][key] = {"error": f"{type(e).__name__}: {e}"}
                print(f"[stage] adapter {key}: FAILED {e}")

    for intrinsic_id, src in selection.get("eval", {}).items():
        spec.intrinsic(intrinsic_id)
        src = Path(src)
        try:
            rows = check_eval_rows(src)
            dest = staged.eval_path(root, intrinsic_id).parent
            with tempfile.TemporaryDirectory() as tmp:
                clean, removed = without_predictions(src, Path(tmp))
                files = {staged.EVAL_FILE: clean}
                if intrinsic_id == "query_rewrite" and args.judge_prompt:
                    check_judge_prompt(args.judge_prompt)
                    files[PROMPT_FILE] = args.judge_prompt
                prov = copy_into(
                    files,
                    dest,
                    {
                        "source": str(src),
                        "source_sha256": sha256(src),
                        "rows": rows,
                        "removed_fields": removed,
                    },
                    args.replace,
                )
            report["eval"][intrinsic_id] = prov
            print(f"[stage] eval {intrinsic_id}: ok ({rows} rows)")
        except Exception as e:
            failures += 1
            report["eval"][intrinsic_id] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[stage] eval {intrinsic_id}: FAILED {e}")

    print("\n[stage] bench root now holds:")
    for cell in staged.discover(root, spec):
        print(f"  {cell.intrinsic}/{cell.tech}: {cell.skip_reason or 'ok'}")
    print(STAGE_BEGIN)
    print(json.dumps(report, sort_keys=True))
    print(STAGE_END, flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    root_help = "a folder to search, or <intrinsic>=<folder>"
    d.add_argument("--adapter-root", action="append", default=[], help=root_help)
    d.add_argument("--eval-root", action="append", default=[], help=root_help)
    d.add_argument("--max-depth", type=int, default=8)
    d.add_argument("--model", default=None, help="draft for this model")
    a = sub.add_parser("apply")
    a.add_argument("--model", default=None, help="the model the root is for")
    a.add_argument("--selection", type=Path, required=True)
    a.add_argument("--bench-root", type=Path, required=True)
    a.add_argument("--judge-prompt", type=Path, default=None)
    a.add_argument("--replace", action="store_true")
    args = p.parse_args(argv)
    return cmd_discover(args) if args.cmd == "discover" else cmd_apply(args)


if __name__ == "__main__":
    sys.exit(main())
