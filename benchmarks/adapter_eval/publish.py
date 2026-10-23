# SPDX-License-Identifier: Apache-2.0
"""Cache check, results merge and page render for the adapter benchmark.

Runs locally, not on the pod::

    python -m benchmarks.adapter_eval.publish check <sha> [--model <id>]  # exit 0 = hit
    python -m benchmarks.adapter_eval.publish check-reference [--model <id>]
    python -m benchmarks.adapter_eval.publish extract <pod.log> --out results.json
    python -m benchmarks.adapter_eval.publish extract <pod.log> --kind discovery ...
    python -m benchmarks.adapter_eval.publish merge results.json
    python -m benchmarks.adapter_eval.publish merge-reference reference.json
    python -m benchmarks.adapter_eval.publish render
    python -m benchmarks.adapter_eval.publish summary <sha> [--model <id>]  # PR comment
    python -m benchmarks.adapter_eval.publish rescore-targets --out targets.json
    python -m benchmarks.adapter_eval.publish merge-rescore rescore.json
    python -m benchmarks.adapter_eval.publish check-release <revision>  # exit 0 = hit
    python -m benchmarks.adapter_eval.publish check-release-reference <revision>
    python -m benchmarks.adapter_eval.publish merge-release results.json
    python -m benchmarks.adapter_eval.publish merge-release-reference reference.json

The page data (``docs/benchmarks/data.json``) holds one row per (commit,
model), each row being the results block ``run_benchmark.py`` printed. A
commit is a cache hit for a model when its row has the model's current
``bench_version`` and every (intrinsic, technology) cell is present and not an
error. Skipped cells (nothing staged) do count as done: staging a new adapter
is a ``bench_version`` bump.

The reference columns (``reference.py``) do not depend on the commit, so the
page data holds them once per model, under ``references``, and the page
repeats them on every row of their model and ``bench_version``. They are a
cache hit when they have the model's current ``bench_version``, the current
``reference_version``, and every (intrinsic, column) cell present and not an
error.

Blocks and data from before there were several models name only the base
model; they are read as the model with that base model.

A Hugging Face release (``releases`` in adapters.yaml) has a row of its own
under ``releases``, named by its Hub revision, with its reference columns
inside it. It is a cache hit when the row has the current ``release_version``
and every cell, its throughput and its switching, each measured with the
current settings, and its reference the current versions too.

A scored cell records its intrinsic's ``score_version``. When that version is
bumped (only the scoring changed), the cell's saved answers are scored again
(``rescore-targets``, ``rescore.py``, ``merge-rescore``) instead of generated
again; a cell without a ``score_version`` was scored with version 1.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from . import stage
from .common import (
    BASE_COLUMN,
    BEGIN_MARKER,
    CLASSIFIER,
    END_MARKER,
    NO_CLASSIFIER,
    REFERENCE_BEGIN,
    REFERENCE_COLUMNS,
    REFERENCE_END,
    REFERENCE_LIBRARIES,
    RELEASE_BEGIN,
    RELEASE_END,
    RELEASE_REFERENCE_BEGIN,
    RELEASE_REFERENCE_END,
    RESCORE_BEGIN,
    RESCORE_END,
    Spec,
    extract_block,
    is_error,
    is_scored,
    iter_cells,
    load_spec,
    served_label,
)
from .throughput import ARMS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "docs" / "benchmarks" / "data.json"
PAGE_PATH = REPO_ROOT / "docs" / "benchmarks" / "index.html"
COMMIT_URL = "https://github.com/generative-computing/granite-switch/commit/{sha}"
# A granite-switch version on PyPI, for a release served from there.
PYPI_URL = "https://pypi.org/project/granite-switch/{version}/"
PAGE_URL = "https://generative-computing.github.io/granite-switch/benchmarks/"
# rescore-targets' exit status when no cell needs scoring again (a crash is 1).
NOTHING_TO_RESCORE = 3

BLOCKS = {
    "results": (BEGIN_MARKER, END_MARKER),
    "reference": (REFERENCE_BEGIN, REFERENCE_END),
    "rescore": (RESCORE_BEGIN, RESCORE_END),
    "discovery": (stage.DISCOVERY_BEGIN, stage.DISCOVERY_END),
    "stage": (stage.STAGE_BEGIN, stage.STAGE_END),
    "release": (RELEASE_BEGIN, RELEASE_END),
    "release-reference": (RELEASE_REFERENCE_BEGIN, RELEASE_REFERENCE_END),
}
# Column-group labels on the page.
ENGINE_LABEL = "granite-switch (vLLM)"
REFERENCE_LABEL = "HF + PEFT"
BASE_LABEL = "Base"
RATIO_LABEL = "Gain ratio"
NATIVE_LABEL = "PEFT (vLLM)"
SPEEDUP_LABEL = "Speedup"
# A row's throughput per technology: granite-switch's and stock vLLM's.
ENGINES = {"gs": ENGINE_LABEL, "native": NATIVE_LABEL}
# The --only entries of the throughput runs (run_benchmark.py).
THROUGHPUT = "throughput"
SWITCHING = "switching"
# The page's two sections.
TASK_LABEL = "Task Quality"
SERVING_LABEL = "Serving Quality - Throughput"


def decode_label(spec: Spec) -> str:
    t = spec.throughput
    return f"Decode {t.generated_tokens} tokens, no switching, batch {t.batch}"


def switching_label(spec: Spec) -> str:
    sw = spec.switching
    return (
        f"Decode {sw.decode_tokens} tokens, switch size {sw.span}, "
        f"concurrency {sw.concurrency}"
    )


# --- data ------------------------------------------------------------------


def load_data(path: Path, spec: Spec) -> dict:
    """The page data, in the current layout (see the module docstring)."""
    data = json.loads(path.read_text()) if path.is_file() else {}
    data.setdefault("rows", [])
    references = data.setdefault("references", {})
    single = data.pop("reference", None)  # the one-model layout
    if single is not None:
        single.setdefault("model", spec.model_of(single))
        references.setdefault(single["model"], single)
    for row in data["rows"]:
        row.setdefault("model", spec.model_of(row))
    return data


def save_data(path: Path, data: dict, spec: Spec) -> None:
    data["spec"] = spec.public()
    data["rows"].sort(key=lambda r: r["commit"]["date"], reverse=True)
    for rows in data.get("releases", {}).values():
        rows.sort(key=lambda r: r["release"]["date"], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def find_row(data: dict, sha: str, model_id: str) -> dict | None:
    matches = [
        r
        for r in data["rows"]
        if r["model"] == model_id and r["commit"]["sha"].startswith(sha)
    ]
    if len(matches) > 1:
        raise ValueError(f"{sha!r} matches {len(matches)} rows; use the full sha")
    return matches[0] if matches else None


def tokens_per_s(row: dict, tech_id: str, engine: str, spec: Spec) -> float | None:
    """A row's decode throughput for one technology and engine, measured with
    the current settings, or None."""
    measured = ((row.get("throughput") or {}).get(tech_id) or {}).get(engine)
    if not isinstance(measured, dict) or not spec.throughput.matches(measured):
        return None
    value = measured.get("tokens_per_s")
    return value if isinstance(value, int | float) else None


def has_engine(tech_id: str, engine: str) -> bool:
    """Whether the technology runs on the engine: stock vLLM has no SR."""
    return (tech_id, engine) in ARMS


def throughput_done(row: dict, spec: Spec) -> bool:
    """Whether every technology's throughput is measured, or has no adapter.
    True for a model that does not measure it (``serving`` in adapters.yaml)."""
    if not spec.model.measures(THROUGHPUT):
        return True
    for tech in spec.serving_technologies:
        entry = (row.get("throughput") or {}).get(tech.id)
        if entry is None or is_error(entry):
            return False
        if "skipped" not in entry and any(
            tokens_per_s(row, tech.id, e, spec) is None
            for e in ENGINES
            if has_engine(tech.id, e)
        ):
            return False
    return True


def p95_seconds(row: dict, tech_id: str, engine: str, spec: Spec) -> float | None:
    """A row's p95 time for agents switching adapters to complete, for one
    technology and engine, measured with the current settings, or None."""
    measured = ((row.get(SWITCHING) or {}).get(tech_id) or {}).get(engine)
    if not isinstance(measured, dict) or not spec.switching.matches(measured):
        return None
    value = measured.get("p95_s")
    return value if isinstance(value, int | float) else None


def switching_done(row: dict, spec: Spec) -> bool:
    """Whether every technology's switching run is measured, on every engine it
    has. True for a model that does not measure it."""
    if not spec.model.measures(SWITCHING):
        return True
    for tech in spec.serving_technologies:
        entry = (row.get(SWITCHING) or {}).get(tech.id)
        if entry is None or is_error(entry):
            return False
        if "skipped" not in entry and any(
            p95_seconds(row, tech.id, e, spec) is None
            for e in ENGINES
            if has_engine(tech.id, e)
        ):
            return False
    return True


def switching_speedup(row: dict, tech_id: str, spec: Spec) -> float | None:
    """How many times sooner granite-switch's agents finish than stock vLLM's."""
    if not has_engine(tech_id, "native"):
        return None
    gs, native = (p95_seconds(row, tech_id, e, spec) for e in ENGINES)
    return None if not gs or native is None else native / gs


def has_cell(intrinsic, column: str) -> bool:
    """Whether an intrinsic has a cell in a column: the classifier column only
    for an intrinsic with a classifier head (adapters.yaml)."""
    return column != CLASSIFIER or intrinsic.classifier is not None


def missing_cells(row: dict, spec: Spec) -> list[str]:
    """What keeps ``row`` from being a cache hit: ``intrinsic/tech`` cells,
    ``throughput`` and ``switching``."""
    out = []
    for intrinsic in spec.intrinsics:
        for tech in spec.technologies:
            if not has_cell(intrinsic, tech.id):
                continue
            cell = row["cells"].get(intrinsic.id, {}).get(tech.id)
            if cell is None or is_error(cell):
                out.append(f"{intrinsic.id}/{tech.id}")
    if not throughput_done(row, spec):
        out.append(THROUGHPUT)
    if not switching_done(row, spec):
        out.append(SWITCHING)
    return out


def cache_hit(row: dict | None, spec: Spec) -> bool:
    return (
        row is not None
        and row.get("bench_version") == spec.bench_version
        and not missing_cells(row, spec)
    )


def merge(data: dict, results: dict, spec: Spec) -> dict:
    """Put a run's results into ``data`` and return the stored row.

    A full run replaces the commit's row for its model. An ``--only`` run
    replaces just those intrinsics, or ``intrinsic/technology`` cells, and the
    throughput if it ran it, inside the existing row of the same
    ``bench_version``.
    """
    spec = spec.for_model(spec.model_of(results))
    refuse_private(spec)
    results["model"] = spec.model_id
    run = results["run"]
    if run.get("limit") is not None:
        raise ValueError("refusing to publish a --limit run")
    if results["bench_version"] != spec.bench_version:
        raise ValueError(
            f"results are bench_version {results['bench_version']}, "
            f"adapters.yaml has {spec.bench_version} for {spec.model_id}"
        )
    sha = results["commit"]["sha"]
    old = find_row(data, sha, spec.model_id)
    if run.get("only") and old and old.get("bench_version") == spec.bench_version:
        for picked in run["only"]:
            if picked in (THROUGHPUT, SWITCHING):
                if picked in results:  # absent when the model does not measure it
                    old[picked] = results[picked]
            elif "/" in picked:
                intrinsic_id, tech_id = picked.split("/", 1)
                new = results["cells"].get(intrinsic_id, {}).get(tech_id)
                if new is not None:
                    old["cells"].setdefault(intrinsic_id, {})[tech_id] = new
            else:
                old["cells"][picked] = results["cells"][picked]
        old.setdefault("updates", []).append(run)
        return old
    if old:
        data["rows"].remove(old)
    data["rows"].append(results)
    return results


def refuse_private(spec: Spec) -> None:
    """A private model's results are kept off the page (common.PRIVATE_MODELS_ENV)."""
    if spec.model.private:
        raise ValueError(f"refusing to publish {spec.model_id}: a private model")


def reference_current(reference: dict | None, spec: Spec) -> bool:
    """Whether reference columns were computed for the current adapters.yaml."""
    return (
        reference is not None
        and reference.get("bench_version") == spec.bench_version
        and reference.get("reference_version") == spec.reference_version
    )


def reference_missing(reference: dict, spec: Spec) -> list[str]:
    """Reference cells still to run, as ``intrinsic/column``."""
    out = []
    for intrinsic in spec.intrinsics:
        for column in REFERENCE_COLUMNS:
            if not has_cell(intrinsic, column):
                continue
            cell = reference["cells"].get(intrinsic.id, {}).get(column)
            if cell is None or is_error(cell):
                out.append(f"{intrinsic.id}/{column}")
    return out


def reference_hit(reference: dict | None, spec: Spec) -> bool:
    return reference_current(reference, spec) and not reference_missing(reference, spec)


def only_arg(keys: list[str]) -> str:
    """The ``reference.py --only`` value that runs exactly ``keys``."""
    by_intrinsic: dict[str, list[str]] = {}
    for key in keys:
        intrinsic_id, column = key.split("/")
        by_intrinsic.setdefault(intrinsic_id, []).append(column)
    return ",".join(
        intrinsic_id
        if set(columns) == set(REFERENCE_COLUMNS)
        else ",".join(f"{intrinsic_id}/{c}" for c in columns)
        for intrinsic_id, columns in by_intrinsic.items()
    )


def merge_reference(data: dict, reference: dict, spec: Spec) -> dict:
    """Put a reference run into ``data`` and return the stored reference.

    A full run replaces its model's stored reference. An ``--only`` run
    replaces just its cells inside a stored reference of the same versions.
    """
    spec = spec.for_model(spec.model_of(reference))
    refuse_private(spec)
    reference["model"] = spec.model_id
    run = reference["run"]
    if run.get("limit") is not None:
        raise ValueError("refusing to publish a --limit run")
    if not reference_current(reference, spec):
        raise ValueError(
            f"reference is bench_version {reference.get('bench_version')}, "
            f"reference_version {reference.get('reference_version')}; adapters.yaml "
            f"has {spec.bench_version}, {spec.reference_version} for {spec.model_id}"
        )
    old = data["references"].get(spec.model_id)
    if run.get("only") and reference_current(old, spec):
        for key in run["only"]:
            intrinsic_id, column = key.split("/")
            old["cells"].setdefault(intrinsic_id, {})[column] = reference["cells"][
                intrinsic_id
            ][column]
        old.setdefault("updates", []).append(run)
        return old
    data["references"][spec.model_id] = reference
    return reference


# --- releases ----------------------------------------------------------------
# A Hugging Face release (``releases`` in adapters.yaml) has a row of its own,
# under ``releases`` by model, named by its Hub revision: its run as published
# (``run_benchmark.py --release``) and, as its ``reference``, the library
# adapters it was composed from under HF + PEFT, with the base model
# (``reference.py --release``). Each is served by its granite-switch version
# from PyPI (``serve_pypi``). A release with ``reference_from`` has no
# reference of its own: its row shows that release's.

RELEASE_URL = "https://huggingface.co/{repo}/commit/{revision}"
HUB_URL = "https://huggingface.co/{repo}/tree/{revision}"


def release_rows(data: dict, model_id: str) -> list[dict]:
    return data.get("releases", {}).get(model_id, [])


def find_release(data: dict, release) -> dict | None:
    return next(
        (
            r
            for r in release_rows(data, release.model)
            if r["release"]["revision"] == release.revision
        ),
        None,
    )


def release_speed(row: dict, block: str, engine: str, spec: Spec) -> float | None:
    """A release's tokens/s (throughput) or p95 seconds (switching) on one
    engine, measured with the current settings, or None. Its switching cell is
    over its own adapters, so their count and rank are its own."""
    measured = (row.get(block) or {}).get(engine)
    if not isinstance(measured, dict):
        return None
    if block == THROUGHPUT:
        current, key = spec.throughput.matches(measured), "tokens_per_s"
    else:
        own = ("adapters", "rank")
        current = all(
            measured.get(k) == v
            for k, v in spec.switching.settings().items()
            if k not in own
        )
        key = "p95_s"
    value = measured.get(key)
    return value if current and isinstance(value, int | float) else None


def _stale(cell: dict, intrinsic) -> bool:
    return is_scored(cell) and cell.get("score_version", 1) < intrinsic.score_version


def release_technologies(spec: Spec):
    """The technologies a release can hold: a published checkpoint has no
    classifier head, so that column is not in any release."""
    return tuple(t for t in spec.technologies if t.id != CLASSIFIER)


def release_missing(row: dict | None, spec: Spec) -> list[str]:
    """What keeps a release's row from being a cache hit: ``intrinsic/tech``
    cells (one scored with an older score_version runs again; releases are not
    rescored), ``throughput`` and ``switching``."""
    row = row or {}
    out = []
    for intrinsic in spec.intrinsics:
        for tech in release_technologies(spec):
            cell = row.get("cells", {}).get(intrinsic.id, {}).get(tech.id)
            if cell is None or is_error(cell) or _stale(cell, intrinsic):
                out.append(f"{intrinsic.id}/{tech.id}")
    for block in (THROUGHPUT, SWITCHING):
        if any(release_speed(row, block, e, spec) is None for e in ENGINES):
            out.append(block)
    return out


def release_reference_current(reference: dict | None, spec: Spec) -> bool:
    return (
        reference is not None
        and reference.get("reference_version") == spec.reference_version
        and reference.get("release_version") == spec.release_version
    )


def release_reference_missing(row: dict | None, spec: Spec) -> list[str]:
    """A release's reference cells still to run, as ``intrinsic/column``."""
    reference = (row or {}).get("reference")
    current = release_reference_current(reference, spec)
    out = []
    for intrinsic in spec.intrinsics:
        for column in REFERENCE_COLUMNS:
            if column == CLASSIFIER:  # not in any release (release_technologies)
                continue
            cell = (
                reference["cells"].get(intrinsic.id, {}).get(column)
                if current
                else None
            )
            if cell is None or is_error(cell) or _stale(cell, intrinsic):
                out.append(f"{intrinsic.id}/{column}")
    return out


def release_hit(row: dict | None, spec: Spec) -> bool:
    """Whether a release's row is complete, served by its PyPI version."""
    if row is None:
        return False
    release = spec.release(row["release"]["revision"])
    return (
        row.get("release_version") == spec.release_version
        and (row.get("served_by") or {}).get("pypi") == release.serve_pypi
        and not release_missing(row, spec)
    )


def release_reference_hit(row: dict | None, spec: Spec) -> bool:
    return row is not None and not release_reference_missing(row, spec)


def reference_mismatch(
    row: dict, reference: dict | None, partial: bool = False
) -> list[str]:
    """The adapters (``intrinsic/tech``) a reference run used whose checksum
    differs in a release's run, or is missing from it: what keeps the release
    from showing that run's columns (``reference_from``). A ``partial`` run
    (``--only``) is compared on what it ran."""
    ran = (row.get("run") or {}).get("adapters") or {}
    used = ((reference or {}).get("run") or {}).get("adapters") or {}
    return sorted(
        key
        for key, f in used.items()
        if not (partial and key not in ran)
        and (ran.get(key) or {}).get("weights_sha256") != f.get("weights_sha256")
    )


def _release_holder(block: dict, spec: Spec):
    """The release a block is of, and its model's definition."""
    release = spec.release(block["release"]["revision"])
    spec = spec.for_model(release.model)
    refuse_private(spec)
    if block["run"].get("limit") is not None:
        raise ValueError("refusing to publish a --limit run")
    return release, spec


def merge_release(data: dict, results: dict, spec: Spec) -> dict:
    """Put a release's run into ``data`` and return its row.

    A full run replaces the row's own cells and speed, and keeps its reference;
    an ``--only`` run replaces just what it ran, in a row of the same
    ``release_version``. A release that shows another's reference
    (``reference_from``) must have run the adapters that reference ran.
    """
    release, spec = _release_holder(results, spec)
    run = results["run"]
    if results.get("release_version") != spec.release_version:
        raise ValueError(
            f"results are release_version {results.get('release_version')}, "
            f"adapters.yaml has {spec.release_version}"
        )
    served = results["served_by"]
    if served.get("pypi") != release.serve_pypi:
        raise ValueError(
            f"served by {served.get('pypi') or 'a commit'}, not by the PyPI version "
            f"adapters.yaml says serves it ({release.served})"
        )
    old = find_release(data, release)
    if release.reference_from:
        source = find_release(data, spec.release(release.reference_from))
        differ = reference_mismatch(
            results, (source or {}).get("reference"), partial=bool(run.get("only"))
        )
        if differ:
            raise ValueError(
                f"release {release.key} shows release {release.reference_from[:8]}'s "
                f"reference columns, but its {', '.join(differ)} differ from what "
                "that reference ran: drop reference_from and run its own"
            )
    rows = data.setdefault("releases", {}).setdefault(spec.model_id, [])
    if (
        run.get("only")
        and old
        and old.get("release_version") == spec.release_version
        and "served_by" in old
    ):
        for picked in run["only"]:
            if picked in (THROUGHPUT, SWITCHING):
                if picked in results:
                    old[picked] = results[picked]
            else:
                old["cells"][picked] = results["cells"][picked]
        old.setdefault("updates", []).append(run)
        return old
    if old:
        rows.remove(old)
        if old.get("reference"):
            results["reference"] = old["reference"]
    rows.append(results)
    return results


def merge_release_reference(data: dict, reference: dict, spec: Spec) -> dict:
    """Put a release's reference run into its row, made if it has none yet;
    returns the stored reference. An ``--only`` run replaces just its cells in
    a stored reference of the same versions."""
    release, spec = _release_holder(reference, spec)
    row = find_release(data, release)
    if release.reference_from:
        raise ValueError(
            f"release {release.key} shows release {release.reference_from[:8]}'s "
            "reference columns (reference_from in adapters.yaml): it has none of its own"
        )
    for other in spec.releases_of(spec.model_id):
        shown = (
            find_release(data, other)
            if other.reference_from == release.revision
            else None
        )
        if (
            shown
            and "run" in shown
            and (differ := reference_mismatch(shown, reference))
        ):
            raise ValueError(
                f"release {other.key} shows this reference, but its "
                f"{', '.join(differ)} differ from what it ran"
            )
    if not release_reference_current(reference, spec):
        raise ValueError(
            f"reference is reference_version {reference.get('reference_version')}, "
            f"release_version {reference.get('release_version')}; adapters.yaml "
            f"has {spec.reference_version}, {spec.release_version}"
        )
    run = reference["run"]
    if row is None:
        row = {
            "model": spec.model_id,
            "release_version": spec.release_version,
            "release": reference["release"],
            "cells": {},
        }
        data.setdefault("releases", {}).setdefault(spec.model_id, []).append(row)
    old = row.get("reference")
    if run.get("only") and release_reference_current(old, spec):
        for key in run["only"]:
            intrinsic_id, column = key.split("/")
            old["cells"].setdefault(intrinsic_id, {})[column] = reference["cells"][
                intrinsic_id
            ][column]
        old.setdefault("updates", []).append(run)
        return old
    row["reference"] = {k: v for k, v in reference.items() if k != "release"}
    return row["reference"]


# --- rescoring ---------------------------------------------------------------


def cell_run(holder: dict, intrinsic_id: str, column: str) -> dict:
    """The run that produced a stored cell of a row or reference.

    That is the last ``--only`` run that covered the cell, else the holder's
    own run.
    """
    for update in reversed(holder.get("updates", [])):
        only = update.get("only") or []
        if intrinsic_id in only or f"{intrinsic_id}/{column}" in only:
            return update
    return holder["run"]


def rescore_targets(
    data: dict, spec: Spec, only: list[str] | None = None
) -> list[dict]:
    """The model's scored cells to score again, as ``rescore.py`` targets.

    A cell is a target when it was scored with an older ``score_version`` than
    its intrinsic's, or, with ``only``, whenever its intrinsic is listed. Only
    rows and references of the model's current ``bench_version`` count: older
    ones are kept as they were.
    """
    holders = [
        ("row", r)
        for r in data["rows"]
        if r["model"] == spec.model_id and r.get("bench_version") == spec.bench_version
    ]
    reference = data["references"].get(spec.model_id)
    if reference_current(reference, spec):
        holders.append(("reference", reference))
    targets = []
    for kind, holder in holders:
        for intrinsic in spec.intrinsics:
            for column, cell in sorted(holder["cells"].get(intrinsic.id, {}).items()):
                stale = cell.get("score_version", 1) < intrinsic.score_version
                if not is_scored(cell) or not (
                    stale or (only and intrinsic.id in only)
                ):
                    continue
                run = cell_run(holder, intrinsic.id, column)
                target = {
                    "kind": kind,
                    "intrinsic": intrinsic.id,
                    "column": column,
                    "finished": run["finished"],
                }
                if run.get("run_ts"):
                    target["run_ts"] = run["run_ts"]
                if kind == "row":
                    target["sha"] = holder["commit"]["sha"]
                targets.append(target)
    return targets


def merge_rescore(data: dict, block: dict, spec: Spec) -> dict[str, list[str]]:
    """Put a rescore run's cells into ``data``; returns the merged and skipped.

    A cell is replaced only while it still comes from the run that was scored
    again; it keeps the generation counts (``truncated``, ``too_long``) of
    the cell it replaces.
    """
    spec = spec.for_model(spec.model_of(block))
    refuse_private(spec)
    merged: list[str] = []
    skipped: list[str] = []
    touched: dict[int, tuple[dict, list[str]]] = {}
    for entry in block["cells"]:
        name = f"{entry['kind']} {entry.get('sha', '')[:8]} {entry['intrinsic']}/{entry['column']}"
        if "cell" not in entry:
            skipped.append(f"{name}: {entry.get('error', 'not scored')}")
            continue
        if entry["kind"] == "row":
            holder = find_row(data, entry["sha"], spec.model_id)
        else:
            holder = data["references"].get(spec.model_id)
        if holder is None or (
            cell_run(holder, entry["intrinsic"], entry["column"])["finished"]
            != entry["finished"]
        ):
            skipped.append(f"{name}: the stored cell comes from another run now")
            continue
        by_column = holder["cells"].setdefault(entry["intrinsic"], {})
        old = by_column.get(entry["column"], {})
        cell = dict(entry["cell"])
        cell.update({k: old[k] for k in ("truncated", "too_long") if k in old})
        by_column[entry["column"]] = cell
        merged.append(name)
        touched.setdefault(id(holder), (holder, []))[1].append(
            f"{entry['intrinsic']}/{entry['column']}"
        )
    for holder, cells in touched.values():
        holder.setdefault("updates", []).append(
            {"rescored": cells, "finished": block["run"]["finished"]}
        )
    return {"merged": merged, "skipped": skipped}


# --- page ------------------------------------------------------------------

# Column groups of one intrinsic, in page order: key -> label.
GROUP_LABELS = {
    "gs": ENGINE_LABEL,
    "ref": REFERENCE_LABEL,
    "base": BASE_LABEL,
    "ratio": RATIO_LABEL,
}
# The page's filters: its two sections; then, within Task Quality, each
# intrinsic and the column groups above.
SECTION_LABELS = {"task": TASK_LABEL, "thr": "Serving Quality"}
# A gain ratio divides by the adapter's gain over the base under HF + PEFT;
# below this gain the ratio mostly measures noise, so it is not shown.
MIN_GAIN = 0.01

PAGE_STYLE = """
:root { color-scheme: light; }
body { font: 14px/1.45 system-ui, sans-serif; margin: 2em; color: #1b1b1b;
       background: #fff; }
h1 { margin-bottom: .2em; }
h2 { font-size: 18px; margin: 1.2em 0 .3em; }
h2 small { font-weight: 400; color: #666; margin-left: .4em; }
h3 { font-size: 15px; margin: 1.6em 0 .3em; }
code { font-size: 13px; }
.lede, .note { color: #444; max-width: 62em; }
.flow { margin: 1.2em 0; padding: .9em 1.1em; border: 1px solid #e3e3e3;
        border-radius: 8px; max-width: 74em; background: #fcfcfc; }
.flow .lane { display: flex; align-items: center; gap: .55em; flex-wrap: wrap;
              margin: .45em 0; }
@media (min-width: 1000px) { .flow .lane { flex-wrap: nowrap; } }
.flow .tag { flex: 0 0 8em; color: #666; font-size: 12px;
             text-transform: uppercase; letter-spacing: .04em; }
.flow .step { border: 1px solid; border-radius: 6px; padding: .45em .75em;
              flex: 0 1 15em; min-width: 9em; }
.flow .step b { display: block; }
.flow .step span { color: #555; font-size: 12.5px; }
.flow .arrow { color: #888; font-size: 18px; }
.flow .result { font-weight: 600; font-size: 13px; padding: .3em .7em;
                border: 1px dashed; border-radius: 999px; white-space: nowrap; }
.flow .gs { background: #f3f3f3; border-color: #c4c4c4; }
.flow .ref { background: #e6ecf4; border-color: #b4c3d8; }
.flow figcaption { margin-top: .5em; color: #333; }
.tabs, .controls { display: none; }
.js .tabs { display: flex; gap: .4em; margin: 1.2em 0 .6em; flex-wrap: wrap; }
.tabs button { font: inherit; padding: .4em 1em; border: 1px solid #c4c4c4;
               border-radius: 999px; background: #fff; cursor: pointer; }
.tabs button[aria-selected="true"] { background: #1b1b1b; border-color: #1b1b1b;
                                     color: #fff; }
.js .controls { display: grid; grid-template-columns: max-content 1fr;
                gap: .3em 1em; align-items: baseline; margin-bottom: .6em;
                color: #444; }
.controls .name { color: #666; }
.controls .boxes { display: flex; gap: .4em 1em; flex-wrap: wrap; }
.controls label { cursor: pointer; white-space: nowrap; }
.controls .off { opacity: .45; }
.controls .off label { cursor: default; }
.js section.model { display: none; }
.js section.model.active { display: block; }
.wrap { overflow-x: auto; }
table.bench { border-collapse: collapse; }
.bench th, .bench td { border: 1px solid #d0d0d0; padding: 4px 8px; }
.bench th { background: #f3f3f3; font-weight: 600; }
.bench th.grp-ref, .bench th.grp-base { background: #e6ecf4; }
.bench td.grp-ref, .bench td.grp-base { background: #f5f8fc; }
.bench th.grp-ratio { background: #efece3; }
.bench td.grp-ratio { background: #fbfaf6; }
.bench th.grp-thr { background: #e5efe7; }
.bench th.section { font-size: 15px; letter-spacing: .02em; }
.bench td.grp-thr { background: #f6faf7; }
.bench .start { border-left: 2px solid #b0b0b0; }
.bench .i-start { border-left: 2px solid #6b6b6b; }
.bench .commit { position: sticky; left: 0; z-index: 1; background: #fff;
                 white-space: nowrap; box-shadow: 1px 0 0 #d0d0d0; }
.bench th.commit { background: #f3f3f3; z-index: 2; }
.bench .commit .date { color: #777; font-size: 12px; }
.bench td.subject { max-width: 16em; overflow: hidden; text-overflow: ellipsis;
                    white-space: nowrap; }
.bench td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bench td.best { font-weight: 700; }
.bench td.skip { color: #9a9a9a; text-align: center; }
.bench td.err { color: #b3261e; text-align: center; }
.bench tr.old td { color: #8a8a8a; }
table.hide-gs .grp-gs, table.hide-ref .grp-ref, table.hide-base .grp-base,
table.hide-ratio .grp-ratio, table.hide-thr .grp-thr { display: none; }
button.info { width: 1.45em; height: 1.45em; padding: 0; margin-left: 5px;
              font: italic 600 12px/1 Georgia, serif; vertical-align: 1px;
              border: 1px solid #b8b8b8; border-radius: 999px; background: #fff;
              color: #555; cursor: pointer; }
button.info:hover, button.info:focus { border-color: #555; color: #1b1b1b; }
.tip { position: fixed; z-index: 10; max-width: 34em; padding: .7em .9em;
       max-height: calc(100vh - 16px); overflow: auto; box-sizing: border-box;
       background: #fff; border: 1px solid #bdbdbd; border-radius: 8px;
       box-shadow: 0 6px 24px rgba(0, 0, 0, .15); font-size: 13px; }
.tip h4 { margin: .5em 0 .15em; font-size: 12px; text-transform: uppercase;
          letter-spacing: .04em; color: #666; }
.tip h4:first-child { margin-top: 0; }
.tip dl { display: grid; grid-template-columns: auto 1fr; gap: .1em .8em; margin: 0; }
.tip dt { color: #666; }
.tip dd { margin: 0; }
.tip table { border-collapse: collapse; margin-top: .2em; }
.tip th, .tip td { padding: 1px 8px 1px 0; text-align: left; font-weight: 400; }
.tip th { color: #666; }
.tip p { margin: .2em 0 0; color: #666; }
.notes { margin-top: 2em; }
.notes li { margin: .2em 0; }
"""

PAGE_SCRIPT = """
(() => {
  document.documentElement.classList.add("js");
  const tabs = [...document.querySelectorAll(".tabs button")];
  const sections = [...document.querySelectorAll("section.model")];
  function show(id) {
    if (!sections.some(s => s.dataset.model === id)) id = sections[0].dataset.model;
    sections.forEach(s => s.classList.toggle("active", s.dataset.model === id));
    tabs.forEach(t => t.setAttribute("aria-selected", String(t.dataset.model === id)));
    if (decodeURIComponent(location.hash.slice(1)) !== id) {
      history.replaceState(null, "", "#" + id);
    }
  }
  tabs.forEach(t => t.addEventListener("click", () => show(t.dataset.model)));
  addEventListener("hashchange", () => show(decodeURIComponent(location.hash.slice(1))));
  show(decodeURIComponent(location.hash.slice(1)));

  // Filters: data-section (task, thr), data-intrinsic and data-group (the
  // column groups of Task Quality).
  const boxes = [...document.querySelectorAll(".controls input")];
  const kind = k => boxes.filter(b => k in b.dataset);
  const on = k => kind(k).filter(b => b.checked).map(b => b.dataset[k]);
  function showsSomething() {
    const sections = on("section");
    return sections.includes("thr") || (sections.includes("task") &&
      on("intrinsic").length > 0 && on("group").length > 0);
  }
  function filter(event) {
    if (event && !showsSomething()) event.target.checked = true;
    const task = on("section").includes("task");
    const serving = on("section").includes("thr");
    const groups = on("group"), intrinsics = on("intrinsic");
    document.querySelectorAll(".controls .task-only").forEach(e => {
      e.classList.toggle("off", !task);
      e.querySelectorAll("input").forEach(i => { i.disabled = !task; });
    });
    document.querySelectorAll("table.bench").forEach(t => {
      t.classList.toggle("hide-thr", !serving);
      kind("group").forEach(b =>
        t.classList.toggle("hide-" + b.dataset.group, !task || !b.checked));
      kind("intrinsic").forEach(b =>
        t.classList.toggle("hide-in-" + b.dataset.intrinsic, !b.checked));
      // Headers span what is left under them.
      let total = 0;
      t.querySelectorAll("th.i").forEach(th => {
        const span = groups.reduce((n, g) => n + Number(th.dataset[g] || 0), 0);
        const shown = task && span > 0 && intrinsics.includes(th.dataset.intrinsic);
        th.colSpan = Math.max(span, 1);
        th.hidden = !shown;
        if (shown) total += span;
      });
      const head = t.querySelector("th.task-head");
      if (head) { head.colSpan = Math.max(total, 1); head.hidden = total === 0; }
    });
  }
  boxes.forEach(b => b.addEventListener("change", filter));
  filter();  // a reload may restore unchecked boxes

  const tip = document.getElementById("tip");
  let pinned = null, timer = null;
  function open(button) {
    clearTimeout(timer);
    tip.innerHTML = button.nextElementSibling.innerHTML;
    tip.hidden = false;
    // Beside the button, which sits in the table's first column.
    const r = button.getBoundingClientRect();
    const left = Math.min(r.right + 8, innerWidth - tip.offsetWidth - 8);
    const top = Math.min(r.top - 12, innerHeight - tip.offsetHeight - 8);
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  function close() { if (!pinned) tip.hidden = true; }
  function later() { timer = setTimeout(close, 200); }
  document.querySelectorAll("button.info").forEach(b => {
    b.removeAttribute("title");  // the box replaces the plain tooltip
    b.addEventListener("mouseenter", () => open(b));
    b.addEventListener("focus", () => open(b));
    b.addEventListener("mouseleave", later);
    b.addEventListener("blur", later);
    b.addEventListener("click", e => {
      e.stopPropagation();
      pinned = pinned === b ? null : b;
      open(b);
    });
  });
  tip.addEventListener("mouseenter", () => clearTimeout(timer));
  tip.addEventListener("mouseleave", later);
  document.addEventListener("click", e => {
    if (!tip.contains(e.target)) { pinned = null; tip.hidden = true; }
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") { pinned = null; tip.hidden = true; }
  });
  addEventListener("scroll", () => pinned ? open(pinned) : close(), true);
})();
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _join(*names: str) -> str:
    return " ".join(n for n in names if n)


def _classes(*names: str) -> str:
    joined = _join(*names)
    return f' class="{joined}"' if joined else ""


def _headline(cell: dict | None, headline: str) -> float | None:
    """A cell's headline value, or None when it has none."""
    if cell is None or not is_scored(cell):
        return None
    value = cell.get(headline)
    return value if isinstance(value, int | float) else None


def _cell_title(cell: dict) -> str:
    return ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(cell.items())
        if isinstance(v, int | float)
    )


def _cell_html(
    cell: dict | None,
    headline: str,
    best: float | None,
    extra: str = "",
    missing: str = "not run",
) -> str:
    """One ``<td>``; ``missing`` is the hover text when there is no cell."""
    if cell is None:
        cls, title, text = "skip", missing, "·"
    elif "skipped" in cell:
        cls, title, text = "skip", cell["skipped"], "—"
    elif is_error(cell):
        cls, title, text = "err", cell["error"], "error"
    elif _headline(cell, headline) is None:
        cls, title, text = "err", f"no {headline} metric", "?"
    else:
        value = cell[headline]
        cls = "num best" if value == best else "num"
        title, text = _cell_title(cell), f"{value * 100:.1f}"
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def _speed_title(measured: dict) -> str:
    runs = measured.get("runs_s") or []
    parts = [
        f"{measured['tokens_per_s']:,.0f} tokens/s",
        f"batch {measured['batch']} of {measured['prompt_tokens']}-token prompts, "
        f"{measured['generated_tokens']} tokens generated each",
    ]
    if measured.get("adapters"):
        parts.append(f"{measured['adapters']} adapters, one per request")
    parts.append(
        f"median of {len(runs)} timed runs, {measured['median_s']:.2f} s"
        + (f" (from {min(runs):.2f} to {max(runs):.2f} s)" if runs else "")
    )
    if measured.get("gpu"):
        parts.append(measured["gpu"])
    return "; ".join(parts)


def _throughput_html(
    row: dict, tech_id: str, engine: str, extra: str, spec: Spec
) -> str:
    """One engine's throughput ``<td>`` for one technology."""
    entry = (row.get("throughput") or {}).get(tech_id)
    measured = (entry or {}).get(engine)
    if not has_engine(tech_id, engine):
        cls, title, text = "skip", "stock vLLM has no SR implementation", "—"
    elif row.get("throughput") is None:
        cls, title, text = "skip", "not measured: the run predates throughput", "·"
    elif entry is None:
        cls, title, text = "skip", "not measured", "·"
    elif "skipped" in entry:
        cls, title, text = "skip", entry["skipped"], "—"
    elif not isinstance(measured, dict):
        cls, title, text = "skip", "not measured", "·"
    elif is_error(measured):
        cls, title, text = "err", measured["error"], "error"
    elif tokens_per_s(row, tech_id, engine, spec) is None:
        cls, title, text = "skip", "measured with other settings", "·"
    else:
        cls, title = "num", _speed_title(measured)
        text = f"{measured['tokens_per_s']:,.0f}"
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def speedup(row: dict, tech_id: str, spec: Spec) -> float | None:
    """granite-switch's decode throughput over stock vLLM's, from one run."""
    if not has_engine(tech_id, "native"):
        return None
    gs, native = (tokens_per_s(row, tech_id, e, spec) for e in ENGINES)
    return None if gs is None or native is None else gs / native


def percent_faster(ratio: float) -> str:
    """A speedup as a whole percentage: 1.34 is +34%."""
    return f"{(ratio - 1) * 100:+.0f}%"


def _speedup_html(row: dict, tech_id: str, extra: str, spec: Spec) -> str:
    entry = (row.get("throughput") or {}).get(tech_id) or {}
    ratio = speedup(row, tech_id, spec)
    if not has_engine(tech_id, "native"):
        cls, title, text = "skip", "no stock-vLLM SR to compare with", "—"
    elif "skipped" in entry:
        cls, title, text = "skip", entry["skipped"], "—"
    elif ratio is None:
        cls, title, text = "skip", "needs both engines' throughput", "·"
    else:
        gs, native = (tokens_per_s(row, tech_id, e, spec) for e in ENGINES)
        cls, text = "num", percent_faster(ratio)
        title = (
            f"{ratio:.2f}x: {ENGINE_LABEL} {gs:,.0f} tokens/s, {NATIVE_LABEL} "
            f"{native:,.0f} tokens/s; the same GPU, one after the other"
        )
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def _switching_title(measured: dict, adapters: str | None = None) -> str:
    """``adapters`` names the agents' adapters; the synthetic fleet's by default."""
    parts = [
        f"p95 {measured['p95_s']:.1f} s, median {measured['median_s']:.1f} s, "
        f"over {measured['agents']} agents in {measured['waves']} waves",
        f"{measured['switches']} adapter switches and {measured['tool_calls']} tool "
        f"calls in all; {measured['prefill_recomputed']:,} prompt tokens re-prefilled",
        adapters
        or f"{measured['adapters']} synthetic adapters of rank {measured['rank']}",
    ]
    if warmups := measured.get("warmup_p95_s"):
        parts.append(
            "untimed warm-up first: p95 " + ", ".join(f"{w:.1f} s" for w in warmups)
        )
    return "; ".join(parts)


def _switching_html(
    row: dict, tech_id: str, engine: str, extra: str, spec: Spec
) -> str:
    """One engine's switching ``<td>`` for one technology: its p95 seconds."""
    entry = (row.get(SWITCHING) or {}).get(tech_id)
    measured = (entry or {}).get(engine)
    if not has_engine(tech_id, engine):
        cls, title, text = "skip", "stock vLLM has no SR implementation", "—"
    elif row.get(SWITCHING) is None:
        cls, title, text = "skip", "not measured: the run predates this experiment", "·"
    elif entry is None or not isinstance(measured, dict):
        cls, title, text = "skip", "not measured", "·"
    elif is_error(measured):
        cls, title, text = "err", measured["error"], "error"
    elif p95_seconds(row, tech_id, engine, spec) is None:
        cls, title, text = "skip", "measured with other settings", "·"
    else:
        cls, title = "num", _switching_title(measured)
        text = f"{measured['p95_s']:,.0f}"
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def _switching_speedup_html(row: dict, tech_id: str, extra: str, spec: Spec) -> str:
    ratio = switching_speedup(row, tech_id, spec)
    if not has_engine(tech_id, "native"):
        cls, title, text = "skip", "no stock-vLLM SR to compare with", "—"
    elif ratio is None:
        cls, title, text = "skip", "needs both engines' times", "·"
    else:
        gs, native = (p95_seconds(row, tech_id, e, spec) for e in ENGINES)
        cls, text = "num", percent_faster(ratio)
        title = (
            f"{ratio:.2f}x sooner: {ENGINE_LABEL} {gs:.1f} s, {NATIVE_LABEL} "
            f"{native:.1f} s at p95; the same GPU, one after the other"
        )
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def gain_ratio(gs: float, peft: float, base: float) -> float | None:
    """(granite-switch - base) / (HF + PEFT - base); None below ``MIN_GAIN``."""
    gain = peft - base
    return None if gain < MIN_GAIN else (gs - base) / gain


def _ratio_html(
    gs: float | None,
    peft: float | None,
    base: float | None,
    extra: str,
    missing: str,
) -> str:
    if gs is None or peft is None or base is None:
        cls, title, text = "skip", missing, "·"
    else:
        ratio = gain_ratio(gs, peft, base)
        title = (
            f"over {BASE_LABEL}: {ENGINE_LABEL} {(gs - base) * 100:+.1f} points, "
            f"{REFERENCE_LABEL} {(peft - base) * 100:+.1f}"
        )
        if ratio is None:
            cls, text = "skip", "n/a"
            title += (
                f"; a gain under {MIN_GAIN * 100:.0f} point is too small to divide by"
            )
        else:
            cls, text = "num", f"{ratio:.2f}"
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def _reference_gap(reference: dict | None, row: dict, spec: Spec) -> str | None:
    """Why ``row`` shows no reference columns, or None when it shows them."""
    if reference is None:
        return "reference not computed yet"
    if reference.get("reference_version") != spec.reference_version:
        return "reference out of date"
    if row.get("bench_version") != reference.get("bench_version"):
        return "no reference for this benchmark version"
    return None


def _libraries(run: dict, names: tuple[str, ...]) -> str:
    return ", ".join(f"{name} {run[name]}" for name in names if run.get(name))


def _reference_note(reference: dict | None, spec: Spec) -> str:
    if reference is None or not reference_current(reference, spec):
        return (
            f"The {REFERENCE_LABEL} and {BASE_LABEL} columns are not computed "
            f"yet for benchmark v{spec.bench_version}."
        )
    run = reference["run"]
    return (
        f"The {REFERENCE_LABEL} and {BASE_LABEL} columns do not depend on the "
        f"commit: computed once, on {run['finished'][:10]} with "
        f"{_libraries(run, REFERENCE_LIBRARIES)}, and shown on every row of "
        f"benchmark v{reference['bench_version']}."
    )


def _adapter_fingerprints(row: dict, reference: dict | None) -> dict:
    """The row's staged-checkpoint fingerprints, else its reference's.

    A bench root is fixed per benchmark version, so a reference of the same
    version used the same checkpoints.
    """
    found = row["run"].get("adapters")
    if (
        not found
        and reference
        and reference.get("bench_version") == row.get("bench_version")
    ):
        found = reference["run"].get("adapters")
    return found or {}


def _ranks(fingerprint: dict) -> str:
    text = f"r={fingerprint['rank']}" if fingerprint.get("rank") else "r=?"
    if fingerprint.get("cross_rank"):
        text += f", cross r={fingerprint['cross_rank']}"
    return text


def _adapters_line(fingerprints: dict, spec: Spec) -> str:
    """``LoRA r=16 · SR r=32, cross r=32``: the ranks of each staged technology."""
    parts = []
    for tech in spec.technologies:
        ranks = sorted(
            {_ranks(f) for k, f in fingerprints.items() if k.endswith(f"/{tech.id}")}
        )
        if ranks:
            parts.append(f"{tech.label} {' or '.join(ranks)}")
    return " · ".join(parts)


def _classifiers_line(row: dict) -> str:
    """Whether the row's composed checkpoints held classifier slots (a
    classifier head in an adapter slot), over every run that composed for it:
    its own and its ``--only`` updates (a re-score composes nothing).

    A run from before they were recorded composed from the same manifests,
    which never ask for one.
    """
    runs = [row["run"], *(u for u in row.get("updates", []) if u.get("only"))]
    most: dict[str, int] = {}
    for run in runs:
        for name, n in run.get("classifier_slots", {}).items():
            most[name] = max(n, most.get(name, 0))
    held = [f"{n} in {name}" for name, n in sorted(most.items()) if n]
    if held:
        return f"Classifier slots: {', '.join(held)}."
    unrecorded = sum("classifier_slots" not in run for run in runs)
    if not unrecorded:
        return "Classifier slots: none; every composed checkpoint holds adapters only."
    runs_text = "the run predates" if unrecorded == len(runs) else "some runs predate"
    return f"Classifier slots: none requested; {runs_text} recording them."


def _throughput_settings(run: dict) -> str:
    t = run.get("throughput")
    if not t:
        return ""
    return (
        f"batch {t['batch']}, {t['generated_tokens']} tokens generated, median of "
        f"{t['timed_runs']} runs after {t['warmup_runs']} warm-up"
    )


def _items(pairs) -> str:
    """A details list of (label, HTML) pairs, leaving out the empty ones."""
    body = "".join(
        f"<dt>{_esc(k)}</dt><dd>{v}</dd>" for k, v in pairs if v not in (None, "")
    )
    return f"<dl>{body}</dl>"


def _gpu(run: dict) -> str:
    """The GPU a run's engines ran on, and how many each spread over."""
    gpu = _esc(run.get("gpu") or "")
    gpus = run.get("tensor_parallel_size") or 1
    return (
        f"{gpu} &times; {gpus}, the model split over them" if gpu and gpus > 1 else gpu
    )


def _code(run: dict) -> str:
    """The benchmark's own code that ran, as a link to its commit."""
    sha = run.get("harness_sha")
    if not sha:
        return "not recorded"
    text = (
        f'<a href="{_esc(COMMIT_URL.format(sha=sha))}"><code>{_esc(sha[:8])}</code></a>'
    )
    return text + (" with local changes" if run.get("harness_dirty") else "")


def _details(row: dict, reference: dict | None, gap: str | None, spec: Spec) -> str:
    """The run-details box of a row, as HTML."""
    items, code = _items, _code
    commit, run = row["commit"], row["run"]
    versions = (
        _esc(_libraries(run, ("vllm", "torch", "transformers"))) or "not recorded"
    )
    out = [
        "<h4>Commit</h4>",
        items(
            [
                ("sha", f"<code>{_esc(commit['sha'])}</code>"),
                ("date", _esc(commit["date"][:16].replace("T", " "))),
                ("subject", _esc(commit["subject"])),
            ]
        ),
        f"<h4>{_esc(ENGINE_LABEL)} run</h4>",
        items(
            [
                ("finished", _esc(run.get("finished", "")[:16].replace("T", " "))),
                ("libraries", versions),
                ("GPU", _gpu(run)),
                (
                    "context",
                    f"{run['max_model_len']:,} tokens"
                    if run.get("max_model_len")
                    else "",
                ),
                ("CUDA graphs", "off" if run.get("enforce_eager") else "on"),
                ("throughput", _esc(_throughput_settings(run))),
                ("benchmark", f"v{row.get('bench_version')}"),
                ("benchmark code", code(run)),
            ]
        ),
    ]
    out.append("<h4>Adapters</h4>")
    fingerprints = _adapter_fingerprints(row, reference)
    if fingerprints:
        head = "".join(f"<th>{_esc(t.label)}</th>" for t in spec.technologies)
        body = []
        for intrinsic in spec.intrinsics:
            cells = []
            for tech in spec.technologies:
                f = fingerprints.get(f"{intrinsic.id}/{tech.id}")
                weights = (f or {}).get("weights_sha256")
                cells.append(
                    f"<td>{_esc(_ranks(f))} <code>{_esc(weights[:8])}</code></td>"
                    if f and weights
                    else f"<td>{_esc(_ranks(f)) if f else '—'}</td>"
                )
            body.append(f"<tr><th>{_esc(intrinsic.name)}</th>{''.join(cells)}</tr>")
        staged = sorted(
            {f["staged_at"][:10] for f in fingerprints.values() if f.get("staged_at")}
        )
        out += [
            f"<table><tr><th></th>{head}</tr>{''.join(body)}</table>",
            "<p>Each checkpoint's rank and the first 8 characters of its weights'"
            " SHA-256 checksum, which change if the checkpoint is replaced"
            + (f". Staged {_esc(', '.join(staged))}." if staged else ".")
            + "</p>",
        ]
    out.append(f"<p>{_esc(_classifiers_line(row))}</p>")
    out.append(f"<h4>{_esc(REFERENCE_LABEL)} and {_esc(BASE_LABEL)}</h4>")
    if gap:
        out.append(f"<p>{_esc(gap)}</p>")
    else:
        ref_run = reference["run"]
        sr_ref = ref_run.get("sr_ref")
        instructions = ref_run.get("base_instructions_sha256")
        out.append(
            items(
                [
                    (
                        "finished",
                        _esc(ref_run.get("finished", "")[:16].replace("T", " ")),
                    ),
                    ("libraries", _esc(_libraries(ref_run, REFERENCE_LIBRARIES))),
                    (
                        "SR model code",
                        f"<code>{_esc(sr_ref[:8])}</code>, the commit of the "
                        "standalone SR implementation"
                        if sr_ref
                        else "",
                    ),
                    (
                        "Base prompt",
                        f"instructions <code>{_esc(instructions[:8])}</code> (checksum)"
                        if instructions
                        else "",
                    ),
                    ("GPU", _esc(ref_run.get("gpu") or "")),
                    ("version", f"reference v{reference.get('reference_version')}"),
                    ("benchmark code", code(ref_run)),
                ]
            )
        )
    return "".join(out)


def _details_text(row: dict) -> str:
    """A plain-text summary of the details box, for browsers without scripts."""
    run = row["run"]
    parts = [
        _libraries(run, ("vllm", "torch", "transformers")),
        run.get("gpu"),
        f"benchmark v{row.get('bench_version')}",
    ]
    return "; ".join(p for p in parts if p)


def _task_head(head: list[list[str]], spec: Spec) -> None:
    """The Task Quality headers under the section's: per intrinsic, its name,
    its column groups and their technologies (rows 2 to 4 of ``head``)."""
    techs = spec.technologies
    n = len(techs)
    sizes = {"gs": n, "ref": n, "base": 1, "ratio": n}
    width = sum(sizes.values())
    data_attrs = " ".join(f'data-{g}="{size}"' for g, size in sizes.items())
    for intrinsic in spec.intrinsics:
        mark = f"in-{intrinsic.id}"  # what its filter hides
        head[1].append(
            f'<th class="i i-start {mark}" colspan="{width}" {data_attrs} '
            f'data-intrinsic="{_esc(intrinsic.id)}">'
            f"{_esc(intrinsic.name)}<br>"
            f"<small>{_esc(intrinsic.headline_label)}</small></th>"
        )
        for group, label in GROUP_LABELS.items():
            span = 'rowspan="2"' if group == "base" else f'colspan="{sizes[group]}"'
            start = "start i-start" if group == "gs" else "start"
            head[2].append(
                f'<th class="grp-{group} {start} {mark}" {span}>{_esc(label)}</th>'
            )
            if group == "base":
                continue
            for k, tech in enumerate(techs):
                first = start if k == 0 else ""
                head[3].append(
                    f"<th{_classes(f'grp-{group}', first, mark)}>{_esc(tech.label)}</th>"
                )


def _task_cells(
    gs_cells: dict, ref_cells: dict | None, gap: str | None, spec: Spec
) -> list[str]:
    """One row's Task Quality cells: per intrinsic, granite-switch's, HF + PEFT's,
    the base model's and the gain ratios. ``ref_cells`` is None, with ``gap``
    saying why, when the row has no reference columns."""
    techs = spec.technologies
    cells = []
    for intrinsic in spec.intrinsics:
        headline = intrinsic.headline
        mark = f"in-{intrinsic.id}"
        by_tech = gs_cells.get(intrinsic.id, {})
        by_column = {} if ref_cells is None else ref_cells.get(intrinsic.id, {})
        # An intrinsic with no classifier head says so in that column, on
        # every row, those from before the column existed too.
        none = {
            t.id: {"skipped": NO_CLASSIFIER}
            for t in techs
            if not has_cell(intrinsic, t.id)
        }
        gs = [none.get(t.id) or by_tech.get(t.id) for t in techs]
        ref = [none.get(t.id) or by_column.get(t.id) for t in techs]
        base = by_column.get(BASE_COLUMN)
        scored = [
            v
            for v in (_headline(c, headline) for c in [*gs, *ref, base])
            if v is not None
        ]
        best = max(scored) if len(scored) > 1 else None
        ref_missing = gap or "not run"
        for k, cell in enumerate(gs):
            start = "start i-start" if k == 0 else ""
            cells.append(_cell_html(cell, headline, best, _join("grp-gs", start, mark)))
        for k, cell in enumerate(ref):
            start = "start" if k == 0 else ""
            cells.append(
                _cell_html(
                    cell, headline, best, _join("grp-ref", start, mark), ref_missing
                )
            )
        cells.append(
            _cell_html(base, headline, best, _join("grp-base start", mark), ref_missing)
        )
        for k in range(len(techs)):
            start = "start" if k == 0 else ""
            # No adapter for this technology: the ratio cell says so too.
            skip = next((c for c in (gs[k], ref[k]) if c and "skipped" in c), None)
            if skip:
                extra = _join("grp-ratio", start, mark)
                cells.append(_cell_html(skip, headline, None, extra))
                continue
            cells.append(
                _ratio_html(
                    _headline(gs[k], headline),
                    _headline(ref[k], headline),
                    _headline(base, headline),
                    _join("grp-ratio", start, mark),
                    gap
                    or f"needs the {ENGINE_LABEL}, {REFERENCE_LABEL} and "
                    f"{BASE_LABEL} scores",
                )
            )
    return cells


def _model_table(rows: list[dict], reference: dict | None, spec: Spec) -> str:
    n = len(spec.technologies)
    # The serving blocks the model measures; each is 3 groups of a column per
    # technology they measure (not a classifier: it answers in one token).
    blocks = [b for b in (THROUGHPUT, SWITCHING) if spec.model.measures(b)]
    techs = spec.serving_technologies
    serving_width = 3 * len(techs) * len(blocks)
    sizes = {"gs": n, "ref": n, "base": 1, "ratio": n}
    width = sum(sizes.values())
    count = len(spec.intrinsics)
    head = [
        [
            '<th class="commit" rowspan="4">Commit</th>',
            '<th rowspan="4">Subject</th>',
            f'<th class="task-head section i-start" colspan="{width * count}">'
            f"{_esc(TASK_LABEL)}</th>",
        ]
        + (
            [
                f'<th class="grp-thr section i-start" colspan="{serving_width}">'
                f"{_esc(SERVING_LABEL)}</th>"
            ]
            if blocks
            else []
        ),
        [],
        [],
        [],
    ]
    _task_head(head, spec)
    # The serving blocks, per technology rather than per intrinsic.
    labels = {
        THROUGHPUT: (decode_label(spec), "tokens/s"),
        SWITCHING: (switching_label(spec), "p95 seconds to complete"),
    }
    for block, unit in (labels[b] for b in blocks):
        head[1].append(
            f'<th class="grp-thr i-start" colspan="{3 * len(techs)}">{_esc(block)}<br>'
            f"<small>{_esc(unit)}</small></th>"
        )
        for label in (*ENGINES.values(), SPEEDUP_LABEL):
            head[2].append(
                f'<th class="grp-thr start" colspan="{len(techs)}">{_esc(label)}</th>'
            )
            for k, tech in enumerate(techs):
                first = "start" if k == 0 else ""
                head[3].append(
                    f"<th{_classes('grp-thr', first)}>{_esc(tech.label)}</th>"
                )

    body = []
    for row in rows:
        commit = row["commit"]
        sha = commit["sha"]
        gap = _reference_gap(reference, row, spec)
        cells = [
            '<td class="commit">'
            f'<a href="{_esc(COMMIT_URL.format(sha=sha))}"><code>'
            f"{_esc(sha[:8])}</code></a>"
            f'<button class="info" type="button" aria-label="Run details" '
            f'title="{_esc(_details_text(row))}">i</button>'
            f'<div class="details" hidden>{_details(row, reference, gap, spec)}</div>'
            f'<div class="date">{_esc(commit["date"][:10])}</div></td>',
            f'<td class="subject" title="{_esc(commit["subject"])}">'
            f"{_esc(commit['subject'])}</td>",
        ]
        cells += _task_cells(
            row["cells"], None if gap else reference["cells"], gap, spec
        )
        if THROUGHPUT in blocks:
            for e, engine in enumerate(ENGINES):
                for k, tech in enumerate(techs):
                    start = ("start i-start" if e == 0 else "start") if k == 0 else ""
                    extra = f"grp-thr {start}".strip()
                    cells.append(_throughput_html(row, tech.id, engine, extra, spec))
            for k, tech in enumerate(techs):
                extra = "grp-thr start" if k == 0 else "grp-thr"
                cells.append(_speedup_html(row, tech.id, extra, spec))
        if SWITCHING in blocks:
            for e, engine in enumerate(ENGINES):
                for k, tech in enumerate(techs):
                    start = ("start i-start" if e == 0 else "start") if k == 0 else ""
                    extra = f"grp-thr {start}".strip()
                    cells.append(_switching_html(row, tech.id, engine, extra, spec))
            for k, tech in enumerate(techs):
                extra = "grp-thr start" if k == 0 else "grp-thr"
                cells.append(_switching_speedup_html(row, tech.id, extra, spec))
        old = row.get("bench_version") != spec.bench_version
        body.append(f"<tr{_classes('old' if old else '')}>" + "".join(cells) + "</tr>")
    if not body:
        body.append(
            f'<tr><td class="commit" colspan="2">no commit benchmarked yet</td>'
            f'<td colspan="{width * len(spec.intrinsics) + serving_width}"></td></tr>'
        )
    return (
        '<div class="wrap"><table class="bench"><thead>'
        + "".join(f"<tr>{''.join(r)}</tr>" for r in head)
        + "</thead><tbody>\n"
        + "\n".join(body)
        + "\n</tbody></table></div>"
    )


RELEASES_HEADING = "Hugging Face releases"
RELEASES_NOTE = (
    "Composed checkpoints published on Hugging Face, a row per release, named by "
    "its Hub revision. Each runs as published, served by the granite-switch "
    f"version from PyPI that loads it; {REFERENCE_LABEL} runs the library adapters "
    "it was composed from. "
    "Its speed is over all of its own adapters, against stock vLLM serving the "
    "same library adapters, so it does not compare with the table above."
)


def _release_gap(reference: dict | None, spec: Spec) -> str | None:
    """Why a release's row shows no reference columns, or None when it shows them."""
    if reference is None:
        return "reference not computed yet"
    if not release_reference_current(reference, spec):
        return "reference out of date"
    return None


def _release_speed_html(
    row: dict, block: str, engine: str, extra: str, spec: Spec
) -> str:
    """One engine's throughput or switching ``<td>`` of a release."""
    measured = (row.get(block) or {}).get(engine)
    value = release_speed(row, block, engine, spec)
    if not isinstance(measured, dict):
        cls, title, text = "skip", "not measured", "·"
    elif is_error(measured):
        cls, title, text = "err", measured["error"], "error"
    elif value is None:
        cls, title, text = "skip", "measured with other settings", "·"
    elif block == THROUGHPUT:
        cls, title, text = "num", _speed_title(measured), f"{value:,.0f}"
    else:
        own = f"the release's {measured['adapters']} adapters, rank up to {measured['rank']}"
        cls, title, text = "num", _switching_title(measured, own), f"{value:,.0f}"
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def _release_speedup_html(row: dict, block: str, extra: str, spec: Spec) -> str:
    gs, native = (release_speed(row, block, e, spec) for e in ENGINES)
    if not gs or native is None:
        cls, title, text = "skip", "needs both engines' numbers", "·"
    else:
        # More tokens per second, or fewer seconds, is faster.
        ratio = gs / native if block == THROUGHPUT else native / gs
        unit = "tokens/s" if block == THROUGHPUT else "s at p95"
        cls, text = "num", percent_faster(ratio)
        title = (
            f"{ratio:.2f}x: {ENGINE_LABEL} {gs:,.1f} {unit}, {NATIVE_LABEL} "
            f"{native:,.1f} {unit}; the same GPU, one after the other"
        )
    return f'<td{_classes(cls, extra)} title="{_esc(title)}">{text}</td>'


def _shown_reference(row: dict, release, references: dict) -> dict | None:
    """The reference a release's row shows: its own, or that of the release
    its ``reference_from`` names (``references``: the model's by revision)."""
    if release is not None and release.reference_from:
        return references.get(release.reference_from)
    return row.get("reference")


def _served_short(row: dict) -> str:
    """What served a release's row, in a few words, under its revision."""
    served = row.get("served_by")
    return f"PyPI {served['pypi']}" if served else "not run yet"


def _release_details(row: dict, release, spec: Spec, reference: dict | None) -> str:
    """The details box of a release's row, with the reference it shows, as HTML."""
    rel, run, served = row["release"], row.get("run") or {}, row.get("served_by")
    out = [
        "<h4>Release</h4>",
        _items(
            [
                (
                    "repository",
                    f'<a href="{_esc(HUB_URL.format(**rel))}">'
                    f"<code>{_esc(rel['repo'])}</code></a>",
                ),
                ("revision", f"<code>{_esc(rel['revision'])}</code>"),
                ("date", _esc(rel["date"][:16].replace("T", " "))),
                ("commit", _esc(rel.get("title") or "")),
                (
                    "composed from",
                    ", ".join(
                        f"<code>{_esc(lib)}</code> <code>{_esc(rev[:8])}</code>"
                        for lib, rev in release.sources.items()
                    )
                    if release
                    else "",
                ),
                (
                    "base model",
                    f"<code>{_esc(spec.base_model)}</code> "
                    f"<code>{_esc(release.base_revision[:8])}</code>"
                    if release
                    else "",
                ),
            ]
        ),
        f"<h4>{_esc(ENGINE_LABEL)} run</h4>",
    ]
    if served is None:
        out.append("<p>not run yet</p>")
    else:
        version = served["pypi"]
        served_html = (
            f'<a href="{_esc(PYPI_URL.format(version=version))}">'
            f"<code>granite-switch {_esc(version)}</code></a> from PyPI"
        )
        out.append(
            _items(
                [
                    ("served by", served_html),
                    ("finished", _esc(run.get("finished", "")[:16].replace("T", " "))),
                    (
                        "libraries",
                        _esc(_libraries(run, ("vllm", "torch", "transformers")))
                        or "not recorded",
                    ),
                    ("GPU", _gpu(run)),
                    (
                        "context",
                        f"{run['max_model_len']:,} tokens"
                        if run.get("max_model_len")
                        else "",
                    ),
                    ("CUDA graphs", "off" if run.get("enforce_eager") else "on"),
                    ("throughput", _esc(_throughput_settings(run))),
                    ("benchmark", f"releases v{row.get('release_version')}"),
                    ("benchmark code", _code(run)),
                ]
            )
        )
    fingerprints = run.get("adapters") or {}
    if fingerprints and release:
        body = []
        for intrinsic in spec.intrinsics:
            name = release.adapters.get(intrinsic.id)
            found = [
                (tech, f)
                for tech in spec.technologies
                if (f := fingerprints.get(f"{intrinsic.id}/{tech.id}"))
            ]
            for tech, f in found:
                weights = f.get("weights_sha256") or ""
                body.append(
                    f"<tr><th>{_esc(intrinsic.name)}</th><td><code>{_esc(name)}</code>"
                    f" {_esc(tech.label)} {_esc(_ranks(f))} <code>{_esc(weights[:8])}</code>"
                    "</td></tr>"
                )
        out += [
            "<h4>Adapters</h4>",
            f"<table>{''.join(body)}</table>",
            "<p>The release's name for each, its rank and the first 8 characters of "
            "its library checkpoint's SHA-256 checksum.</p>",
        ]
    out.append(f"<h4>{_esc(REFERENCE_LABEL)} and {_esc(BASE_LABEL)}</h4>")
    if release is not None and release.reference_from:
        source = spec.release(release.reference_from)
        out.append(
            f'<p>Those of release <a href="{_esc(RELEASE_URL.format(**vars(source)))}">'
            f"<code>{_esc(source.key)}</code></a>: the same adapters, on the same "
            "base.</p>"
        )
    gap = _release_gap(reference, spec)
    if gap:
        out.append(f"<p>{_esc(gap)}</p>")
    else:
        ref_run = reference["run"]
        instructions = ref_run.get("base_instructions_sha256")
        out.append(
            _items(
                [
                    (
                        "finished",
                        _esc(ref_run.get("finished", "")[:16].replace("T", " ")),
                    ),
                    ("libraries", _esc(_libraries(ref_run, REFERENCE_LIBRARIES))),
                    (
                        "Base prompt",
                        f"instructions <code>{_esc(instructions[:8])}</code> (checksum)"
                        if instructions
                        else "",
                    ),
                    ("GPU", _esc(ref_run.get("gpu") or "")),
                    ("version", f"reference v{reference.get('reference_version')}"),
                    ("benchmark code", _code(ref_run)),
                ]
            )
        )
    return "".join(out)


def _release_details_text(row: dict) -> str:
    """A plain-text summary of a release's details box, for browsers without scripts."""
    run, served = row.get("run") or {}, row.get("served_by")
    parts = [
        f"served by {served_label(served)}" if served else "not run yet",
        _libraries(run, ("vllm", "torch", "transformers")),
        run.get("gpu"),
    ]
    return "; ".join(p for p in parts if p)


def _release_table(rows: list[dict], spec: Spec) -> str:
    """A model's Hugging Face releases: the Task Quality columns as in its main
    table, then each release's speed over its own adapters, per engine."""
    width = 3 * len(spec.technologies) + 1
    count = len(spec.intrinsics)
    head = [
        [
            '<th class="commit" rowspan="4">Release</th>',
            '<th rowspan="4">Repository</th>',
            f'<th class="task-head section i-start" colspan="{width * count}">'
            f"{_esc(TASK_LABEL)}</th>",
            f'<th class="grp-thr section i-start" colspan="6">{_esc(SERVING_LABEL)}</th>',
        ],
        [],
        [],
        [],
    ]
    _task_head(head, spec)
    for label, unit in (
        (decode_label(spec), "tokens/s"),
        (switching_label(spec), "p95 seconds to complete"),
    ):
        head[1].append(
            f'<th class="grp-thr i-start" colspan="3">{_esc(label)}<br>'
            f"<small>{_esc(unit)}</small></th>"
        )
        for engine_label in (*ENGINES.values(), SPEEDUP_LABEL):
            head[2].append(
                f'<th class="grp-thr start" rowspan="2">{_esc(engine_label)}</th>'
            )
    by_revision = {r.revision: r for r in spec.releases_of(spec.model_id)}
    references = {
        r["release"]["revision"]: r["reference"] for r in rows if r.get("reference")
    }
    body = []
    for row in rows:
        rel = row["release"]
        release = by_revision.get(rel["revision"])
        reference = _shown_reference(row, release, references)
        gap = _release_gap(reference, spec)
        cells = [
            '<td class="commit">'
            f'<a href="{_esc(RELEASE_URL.format(**rel))}"><code>'
            f"{_esc(rel['revision'][:8])}</code></a>"
            f'<button class="info" type="button" aria-label="Release details" '
            f'title="{_esc(_release_details_text(row))}">i</button>'
            '<div class="details" hidden>'
            f"{_release_details(row, release, spec, reference)}</div>"
            f'<div class="date">{_esc(rel["date"][:10])}</div>'
            f'<div class="date">{_esc(_served_short(row))}</div></td>',
            f'<td class="subject" title="{_esc(rel["repo"])}">'
            f'<a href="{_esc(HUB_URL.format(**rel))}">'
            f"{_esc(rel['repo'].rsplit('/', 1)[-1])}</a></td>",
        ]
        cells += _task_cells(
            row.get("cells", {}), None if gap else reference["cells"], gap, spec
        )
        for block in (THROUGHPUT, SWITCHING):
            for e, engine in enumerate(ENGINES):
                extra = "grp-thr start i-start" if e == 0 else "grp-thr start"
                cells.append(_release_speed_html(row, block, engine, extra, spec))
            cells.append(_release_speedup_html(row, block, "grp-thr start", spec))
        old = row.get("release_version") != spec.release_version
        body.append(f"<tr{_classes('old' if old else '')}>" + "".join(cells) + "</tr>")
    if not body:
        body.append(
            '<tr><td class="commit" colspan="2">no Hugging Face release yet</td>'
            f'<td colspan="{width * count + 6}"></td></tr>'
        )
    return (
        f"<h3>{_esc(RELEASES_HEADING)}</h3>"
        f'<p class="note">{_esc(RELEASES_NOTE)}</p>'
        '<div class="wrap"><table class="bench"><thead>'
        + "".join(f"<tr>{''.join(r)}</tr>" for r in head)
        + "</thead><tbody>\n"
        + "\n".join(body)
        + "\n</tbody></table></div>"
    )


FLOW = f"""<figure class="flow">
<div class="lane"><div class="tag">Every commit</div>
<div class="step gs"><b>1. Pick adapters</b><span>trained LoRA, aLoRA and SR
checkpoints and classifier heads, staged once per base model</span></div><div class="arrow">&rarr;</div>
<div class="step gs"><b>2. Compose</b><span>the commit's composer builds one
checkpoint holding every adapter</span></div><div class="arrow">&rarr;</div>
<div class="step gs"><b>3. Evaluate</b><span>its vLLM backend answers each eval
set and a scorer grades the answers; decoding is timed against stock vLLM</span></div>
<div class="arrow">&rarr;</div>
<div class="result gs">{ENGINE_LABEL}</div></div>
<div class="lane"><div class="tag">Once per model</div>
<div class="step ref"><b>Same adapters, no granite-switch</b><span>Hugging Face
transformers + PEFT</span></div><div class="arrow">&rarr;</div>
<div class="result ref">{REFERENCE_LABEL}</div>
<div class="step ref"><b>Base model alone</b><span>no adapter; told the
answer format</span></div>
<div class="arrow">&rarr;</div><div class="result ref">{BASE_LABEL}</div></div>
<figcaption><b>{RATIO_LABEL}</b> = ({ENGINE_LABEL} &minus; {BASE_LABEL}) &divide;
({REFERENCE_LABEL} &minus; {BASE_LABEL}). At 1.00, granite-switch keeps all of an
adapter's gain over the base model.</figcaption>
</figure>"""


def render(data: dict, spec: Spec) -> str:
    sections, tabs = [], []
    for m in spec.public_models:
        model_spec = spec.for_model(m.id)
        rows = [r for r in data["rows"] if r["model"] == m.id]
        reference = data["references"].get(m.id)
        fingerprints = next(
            (f for f in (_adapter_fingerprints(r, reference) for r in rows) if f),
            (reference or {}).get("run", {}).get("adapters", {}),
        )
        adapters = _adapters_line(fingerprints, model_spec)
        tabs.append(
            f'<button type="button" role="tab" data-model="{_esc(m.id)}">'
            f"{_esc(m.label)}</button>"
        )
        sections.append(
            f'<section class="model" data-model="{_esc(m.id)}">'
            f"<h2>{_esc(m.label)} <small><code>{_esc(m.name)}</code></small></h2>"
            f'<p class="note">{_esc(_reference_note(reference, model_spec))}'
            + (f" Staged adapters: {_esc(adapters)}." if adapters else "")
            + "</p>"
            + _model_table(rows, reference, model_spec)
            + _release_table(release_rows(data, m.id), model_spec)
            + "</section>"
        )
    # Every model's intrinsics, in spec order: the filters apply to every tab.
    intrinsics = {
        i.id: i for m in spec.public_models for i in spec.for_model(m.id).intrinsics
    }

    def boxes(kind: str, items) -> str:
        return "".join(
            f'<label><input type="checkbox" data-{kind}="{_esc(key)}" checked> '
            f"{_esc(label)}</label>"
            for key, label in items
        )

    filters = (
        '<span class="name">Show</span>'
        f'<span class="boxes">{boxes("section", SECTION_LABELS.items())}</span>'
        '<span class="name task-only">Adapters</span><span class="boxes task-only">'
        + boxes("intrinsic", ((i.id, i.name) for i in intrinsics.values()))
        + '</span><span class="name task-only">Columns</span>'
        f'<span class="boxes task-only">{boxes("group", GROUP_LABELS.items())}</span>'
    )
    hide_intrinsics = "".join(
        f"table.hide-in-{i} .in-{i} {{ display: none; }}\n" for i in intrinsics
    )
    t, sw = spec.throughput, spec.switching
    tech_notes = "".join(
        f"<li><b>{_esc(t.label)}</b>: {_esc(t.source)}</li>" for t in spec.technologies
    )
    return f"""<!DOCTYPE html>
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Generated by benchmarks/adapter_eval/publish.py; do not edit. -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Granite Switch adapter benchmark</title>
<style>{PAGE_STYLE}{hide_intrinsics}</style>
</head>
<body>
<h1>Granite Switch adapter benchmark</h1>
<p class="lede">How well trained intrinsic adapters work through each
granite-switch commit, next to the same adapters without granite-switch and
the base model alone. Greedy decoding. Accuracy is in percent, decode
throughput in tokens per second, agents' time to finish in seconds.</p>
{FLOW}
<nav class="tabs" role="tablist" aria-label="Base model">{"".join(tabs)}</nav>
<div class="controls">{filters}</div>
{chr(10).join(sections)}
<section class="notes">
<h2>Reading the table</h2>
<ul class="note">
<li>Each intrinsic shows its headline metric. The best of its nine accuracy
columns is <b>bold</b>. Hover a cell for all its metrics, or for why it is
empty.</li>
<li><b>{_esc(ENGINE_LABEL)}</b>: the adapters composed into the base model by
each commit's composer, and served by its vLLM backend.</li>
<li><b>{_esc(REFERENCE_LABEL)}</b>: the same adapter checkpoints without
granite-switch, loaded with Hugging Face transformers and PEFT. SR runs on a
standalone Hugging Face implementation of the Shadow Residual model. A
classifier head runs on the Hugging Face base model's final hidden state.</li>
<li><b>{_esc(BASE_LABEL)}</b>: the base model with no adapter. A final
instruction tells it the answer format each scorer expects, which only the
adapters were trained on.</li>
<li><b>{_esc(RATIO_LABEL)}</b>: the share of an adapter's gain over the base
model that granite-switch keeps. <b>n/a</b>: the adapter gains under
{MIN_GAIN * 100:.0f} point under {_esc(REFERENCE_LABEL)}, too little to
divide by.</li>
<li><b>—</b> no adapter or eval set for this cell; <b>·</b> not run;
<b>error</b> the run failed. Greyed rows used an older benchmark version.</li>
<li><b>Classifier</b>: a linear head on the base model's final hidden state at
the end of the conversation, trained on the frozen base model, composed as a
classifier slot (granite-switch PR #134, which now calls it an activation
probe) beside the LoRA and aLoRA adapters.
Its answer is one label word, the first generated token; yes / no, read as
answerable / unanswerable for answerability. Only some intrinsics have one; it
is not timed, as it decodes nothing.</li>
<li><b>{_esc(TASK_LABEL)}</b> is how well the adapters do their tasks;
<b>{_esc(SERVING_LABEL)}</b> how fast they are served, per technology rather
than per intrinsic, both engines on the same GPU one after the other in each
commit's run, measured by the switch benchmark's own scripts.</li>
<li><b>{_esc(decode_label(spec))}</b>: tokens per second while decoding, as in
the switch benchmark's batch-decode figure but at one batch size: {t.batch}
one-token prompts, each request on one of the technology's adapters (all of the
model's, drawn at random, the same draw for both engines), exactly
{t.generated_tokens} tokens generated each, the median of {t.timed_runs} timed
runs after {t.warmup_runs} warm-up runs, no prefix caching.</li>
<li><b>{_esc(switching_label(spec))}</b>: the p95 time for an agent to
finish, in seconds, as in its grid-concurrency figure but at one cell:
{sw.concurrency} agents at a time ({sw.min_agents} in all), each generating
{sw.decode_tokens} tokens over a prompt of its own, switching adapter every
{sw.span} tokens over {sw.adapters} synthetic adapters, with tool calls between
runs, timed after {sw.warmup_runs} untimed run{"" if sw.warmup_runs == 1 else "s"}
of the same cell in the same engine, so adapters are loaded and kernels
compiled, as for all but the first cell of that figure. Stock vLLM re-issues
the request at every switch, re-prefilling the
context (cached for an adapter it saw before), and so does granite-switch LoRA
and aLoRA, by control token; granite-switch SR switches without a re-prefill,
so its agents run uninterrupted but for the tool calls.</li>
<li><b>{_esc(ENGINE_LABEL)}</b> (throughput): the technology's adapters composed
into a checkpoint of their own; a request's prompt is its adapter's control
token.</li>
<li><b>{_esc(NATIVE_LABEL)}</b>: the same adapter checkpoints served by stock
vLLM's multi-LoRA support, without granite-switch (the switch benchmark's
native-lora arm, lora-vllm in its figures). An aLoRA is served active from the
first token, the way it decodes once on. Stock vLLM has no SR implementation,
so SR has no number here (<b>—</b>).</li>
<li><b>{_esc(SPEEDUP_LABEL)}</b>: how much faster {_esc(ENGINE_LABEL)} is
than {_esc(NATIVE_LABEL)}, in whole percent: +34% is 1.34 times the tokens per
second, or agents finishing in 1/1.34 of the time; a negative value is slower,
-13% being 0.87 times the tokens per second, or agents taking 1/0.87 of the
time. None for SR.</li>
<li>The <b>i</b> next to a commit opens its run details: library versions,
GPU, the adapter checkpoints and the reference run.</li>
<li>Granite 4.2 prompts turn reasoning off and carry their documents as tool
messages, the way its adapters were trained.</li>
</ul>
<p class="note">Adapters:</p>
<ul class="note">{tech_notes}</ul>
</section>
<div id="tip" class="tip" role="tooltip" hidden></div>
<script>{PAGE_SCRIPT}</script>
</body>
</html>
"""


# --- PR comment -------------------------------------------------------------


def _plain(cell: dict | None, headline: str) -> str:
    """A cell as plain text: its value, or why it has none."""
    if cell is None:
        return "·"
    if "skipped" in cell:
        return "—"
    if is_error(cell):
        return "error"
    value = _headline(cell, headline)
    return "?" if value is None else f"{value * 100:.1f}"


def summary(
    data: dict, spec: Spec, sha: str, outcome: str, run_url: str | None = None
) -> str:
    """The PR comment of a ``/benchmark`` run, as Markdown.

    It starts with a marker naming the model, so a later run of the same
    model edits this comment instead of adding one.
    """
    marker = f"<!-- adapter-benchmark:{spec.model_id} -->"
    title = f"### Adapter benchmark: {spec.model.label}"
    run = f" · [run]({run_url})" if run_url else ""
    row = find_row(data, sha, spec.model_id)
    if outcome == "failed" or row is None:
        return (
            f"{marker}\n{title}\n\nThe run for `{sha[:8]}` failed, so nothing "
            f"was published{run}."
        )
    reference = data["references"].get(spec.model_id)
    gap = _reference_gap(reference, row, spec)
    techs = spec.technologies
    labels = " / ".join(t.label for t in techs)
    cached = " (cached result)" if outcome == "cached" else ""
    lines = [
        marker,
        title,
        "",
        f"Commit `{sha[:8]}`{cached} · [results page]({PAGE_URL}#{spec.model_id}){run}",
        "",
        f"| Intrinsic | {ENGINE_LABEL} {labels} | {REFERENCE_LABEL} {labels} "
        f"| {BASE_LABEL} | {RATIO_LABEL} {labels} |",
        "|---|---|---|---|---|",
    ]
    for intrinsic in spec.intrinsics:
        headline = intrinsic.headline
        # No classifier head: "—" in that column, as on the page.
        none = {
            t.id: {"skipped": NO_CLASSIFIER}
            for t in techs
            if not has_cell(intrinsic, t.id)
        }
        gs = {**row["cells"].get(intrinsic.id, {}), **none}
        ref = {**({} if gap else reference["cells"].get(intrinsic.id, {})), **none}
        ratios = []
        for t in techs:
            if t.id in none:
                ratios.append("—")
                continue
            values = [
                _headline(gs.get(t.id), headline),
                _headline(ref.get(t.id), headline),
                _headline(ref.get(BASE_COLUMN), headline),
            ]
            ratio = None if None in values else gain_ratio(*values)
            ratios.append(
                "·" if None in values else "n/a" if ratio is None else f"{ratio:.2f}"
            )
        lines.append(
            f"| {intrinsic.name} "
            f"| {' / '.join(_plain(gs.get(t.id), headline) for t in techs)} "
            f"| {' / '.join(_plain(ref.get(t.id), headline) for t in techs)} "
            f"| {_plain(ref.get(BASE_COLUMN), headline)} "
            f"| {' / '.join(ratios)} |"
        )
    if gap:
        lines += ["", f"No {REFERENCE_LABEL} and {BASE_LABEL} columns yet: {gap}."]

    # The speed lines: only the technologies the serving blocks measure.
    serving = spec.serving_technologies
    serving_labels = " / ".join(t.label for t in serving)

    def speeds(engine: str) -> str:
        values = [
            "—"
            if not has_engine(t.id, engine)
            else tokens_per_s(row, t.id, engine, spec)
            for t in serving
        ]
        return " / ".join(
            "·" if v is None else v if isinstance(v, str) else f"{v:,.0f}"
            for v in values
        )

    def seconds(engine: str) -> str:
        return " / ".join(
            "—"
            if not has_engine(t.id, engine)
            else "·"
            if (v := p95_seconds(row, t.id, engine, spec)) is None
            else f"{v:,.0f}"
            for t in serving
        )

    def percents(ratios: list[float | None]) -> str:
        return " / ".join(
            "—"
            if not has_engine(t.id, "native")
            else "·"
            if r is None
            else percent_faster(r)
            for t, r in zip(serving, ratios, strict=True)
        )

    if spec.model.measures(THROUGHPUT):
        lines += [
            "",
            f"{decode_label(spec)}, tokens/s, {serving_labels}: {ENGINE_LABEL} {speeds('gs')}; "
            f"{NATIVE_LABEL} {speeds('native')}; {SPEEDUP_LABEL.lower()} "
            + percents([speedup(row, t.id, spec) for t in serving]),
        ]
    if spec.model.measures(SWITCHING):
        lines += [
            "",
            f"{switching_label(spec)}, p95 seconds to complete, {serving_labels}: "
            f"{ENGINE_LABEL} {seconds('gs')}; {NATIVE_LABEL} {seconds('native')}; "
            f"{SPEEDUP_LABEL.lower()} "
            + percents([switching_speedup(row, t.id, spec) for t in serving]),
        ]
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, default=DATA_PATH)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="exit 0 and print the row on a cache hit")
    c.add_argument("sha")
    c.add_argument("--model", default=None, help="default: the first model")
    cr = sub.add_parser(
        "check-reference", help="exit 0 and print the reference columns on a cache hit"
    )
    cr.add_argument("--model", default=None, help="default: the first model")
    e = sub.add_parser("extract", help="pull a JSON block out of a pod log")
    e.add_argument("log", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--kind", choices=sorted(BLOCKS), default="results")
    m = sub.add_parser("merge", help="add a run's results to the page data")
    m.add_argument("results", type=Path)
    mr = sub.add_parser("merge-reference", help="add a reference run to the page data")
    mr.add_argument("reference", type=Path)
    r = sub.add_parser("render", help="write the page from the page data")
    r.add_argument("--out", type=Path, default=PAGE_PATH)
    s = sub.add_parser("summary", help="print a /benchmark run's PR comment")
    s.add_argument("sha")
    s.add_argument("--model", default=None, help="default: the first model")
    s.add_argument("--outcome", choices=("ran", "cached", "failed"), default="ran")
    s.add_argument("--run-url", default=None)
    rt = sub.add_parser(
        "rescore-targets",
        help=f"list the cells to score again; exit {NOTHING_TO_RESCORE} if none",
    )
    rt.add_argument("--model", default=None, help="default: the first model")
    rt.add_argument(
        "--only", default=None, help="comma-separated intrinsics to score again anyway"
    )
    rt.add_argument("--out", type=Path, required=True)
    ms = sub.add_parser("merge-rescore", help="add a rescore run to the page data")
    ms.add_argument("rescore", type=Path)
    for name, what in (
        ("check-release", "its row"),
        ("check-release-reference", "its reference columns"),
    ):
        cl = sub.add_parser(name, help=f"exit 0 and print {what} on a cache hit")
        cl.add_argument("release", help="its revision (8+ characters)")
    mrl = sub.add_parser("merge-release", help="add a release's run to the page data")
    mrl.add_argument("results", type=Path)
    mrr = sub.add_parser(
        "merge-release-reference", help="add a release's reference run to the page data"
    )
    mrr.add_argument("reference", type=Path)
    args = p.parse_args(argv)

    spec = load_spec()
    if args.cmd == "extract":
        block = extract_block(args.log.read_text(errors="replace"), *BLOCKS[args.kind])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")
        if args.kind in ("results", "reference", "release", "release-reference"):
            bad = [f"{i}/{t}" for i, t, c in iter_cells(block["cells"]) if is_error(c)]
            name = {
                "results": lambda: block["commit"]["sha"][:8],
                "reference": lambda: "reference",
                "release": lambda: f"release {block['release']['revision'][:8]}",
                "release-reference": lambda: (
                    f"release {block['release']['revision'][:8]} reference"
                ),
            }[args.kind]()
            print(f"extracted {name}; error cells: {', '.join(bad) or 'none'}")
        else:
            print(f"extracted {args.kind} block to {args.out}")
        return 0

    data = load_data(args.data, spec)
    if args.cmd in ("check", "check-reference", "summary", "rescore-targets"):
        spec = spec.for_model(args.model)
    if args.cmd == "check":
        row = find_row(data, args.sha, spec.model_id)
        if cache_hit(row, spec):
            print(json.dumps(row, indent=2, sort_keys=True))
            return 0
        if row is None:
            print(f"cache miss: no {spec.model_id} row for {args.sha}")
        elif row.get("bench_version") != spec.bench_version:
            print(f"cache miss: row is bench_version {row.get('bench_version')}")
        else:
            print(f"cache miss: cells to run: {', '.join(missing_cells(row, spec))}")
        return 1
    if args.cmd == "check-reference":
        reference = data["references"].get(spec.model_id)
        if reference_hit(reference, spec):
            print(json.dumps(reference, indent=2, sort_keys=True))
            return 0
        if reference is None:
            print(f"cache miss: no reference columns for {spec.model_id}")
        elif not reference_current(reference, spec):
            print(
                f"cache miss: reference is bench_version {reference.get('bench_version')}, "
                f"reference_version {reference.get('reference_version')}"
            )
        else:
            missing = reference_missing(reference, spec)
            print(
                f"cache miss: cells to run: {', '.join(missing)} "
                f"(just these: --only {only_arg(missing)})"
            )
        return 1
    if args.cmd == "merge":
        results = json.loads(args.results.read_text())
        row = merge(data, results, spec)
        save_data(args.data, data, spec)
        print(f"merged {row['commit']['sha'][:8]} ({row['model']}) into {args.data}")
        return 0
    if args.cmd == "merge-reference":
        stored = merge_reference(data, json.loads(args.reference.read_text()), spec)
        save_data(args.data, data, spec)
        print(f"merged the {stored['model']} reference columns into {args.data}")
        return 0
    if args.cmd == "render":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(data, spec))
        print(f"wrote {args.out} ({len(data['rows'])} rows)")
        return 0
    if args.cmd == "summary":
        print(summary(data, spec, args.sha, args.outcome, args.run_url))
        return 0
    if args.cmd == "rescore-targets":
        only = [i.strip() for i in args.only.split(",")] if args.only else None
        if only:
            spec.select(only)  # raises on an unknown intrinsic
        targets = rescore_targets(data, spec, only)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(targets, indent=2) + "\n")
        print(f"{len(targets)} {spec.model_id} cells to score again")
        return 0 if targets else NOTHING_TO_RESCORE
    if args.cmd == "merge-rescore":
        outcome = merge_rescore(data, json.loads(args.rescore.read_text()), spec)
        save_data(args.data, data, spec)
        print(f"rescored {len(outcome['merged'])} cells into {args.data}")
        for line in outcome["skipped"]:
            print(f"  skipped {line}")
        return 0
    if args.cmd in ("check-release", "check-release-reference"):
        release = spec.release(args.release)
        spec = spec.for_model(release.model)
        row = find_release(data, release)
        if args.cmd == "check-release":
            hit, missing = release_hit(row, spec), release_missing(row, spec)
        else:
            if release.reference_from:
                print(
                    f"release {release.key} shows release "
                    f"{release.reference_from[:8]}'s reference columns"
                )
                row = find_release(data, spec.reference_release(release))
            hit, missing = (
                release_reference_hit(row, spec),
                release_reference_missing(row, spec),
            )
        if hit:
            print(json.dumps(row, indent=2, sort_keys=True))
            return 0
        if row is None:
            print(f"cache miss: no row for release {release.key}")
        elif (
            args.cmd == "check-release"
            and row.get("release_version") != spec.release_version
        ):
            print(f"cache miss: row is release_version {row.get('release_version')}")
        else:
            print(
                f"cache miss: to run: {', '.join(missing) or 'its granite-switch run'}"
            )
        return 1
    if args.cmd == "merge-release":
        row = merge_release(data, json.loads(args.results.read_text()), spec)
        save_data(args.data, data, spec)
        print(f"merged release {row['release']['revision'][:8]} into {args.data}")
        return 0
    if args.cmd == "merge-release-reference":
        block = json.loads(args.reference.read_text())
        merge_release_reference(data, block, spec)
        save_data(args.data, data, spec)
        print(
            f"merged release {block['release']['revision'][:8]}'s reference into {args.data}"
        )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
