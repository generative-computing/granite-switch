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
    END_MARKER,
    REFERENCE_BEGIN,
    REFERENCE_COLUMNS,
    REFERENCE_END,
    REFERENCE_LIBRARIES,
    Spec,
    extract_block,
    is_error,
    is_scored,
    iter_cells,
    load_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "docs" / "benchmarks" / "data.json"
PAGE_PATH = REPO_ROOT / "docs" / "benchmarks" / "index.html"
COMMIT_URL = "https://github.com/generative-computing/granite-switch/commit/{sha}"
BLOCKS = {
    "results": (BEGIN_MARKER, END_MARKER),
    "reference": (REFERENCE_BEGIN, REFERENCE_END),
    "discovery": (stage.DISCOVERY_BEGIN, stage.DISCOVERY_END),
    "stage": (stage.STAGE_BEGIN, stage.STAGE_END),
}
# Column-group labels on the page.
ENGINE_LABEL = "granite-switch (vLLM)"
REFERENCE_LABEL = "HF + PEFT"
BASE_LABEL = "Base"


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


def missing_cells(row: dict, spec: Spec) -> list[str]:
    """Cells that keep ``row`` from being a cache hit, as ``intrinsic/tech``."""
    out = []
    for intrinsic in spec.intrinsics:
        for tech in spec.technologies:
            cell = row["cells"].get(intrinsic.id, {}).get(tech.id)
            if cell is None or is_error(cell):
                out.append(f"{intrinsic.id}/{tech.id}")
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
    replaces just those intrinsics inside the existing row of the same
    ``bench_version``.
    """
    spec = spec.for_model(spec.model_of(results))
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
        for intrinsic_id in run["only"]:
            old["cells"][intrinsic_id] = results["cells"][intrinsic_id]
        old.setdefault("updates", []).append(run)
        return old
    if old:
        data["rows"].remove(old)
    data["rows"].append(results)
    return results


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


# --- page ------------------------------------------------------------------

PAGE_STYLE = """
:root { color-scheme: light; }
body { font: 14px/1.4 system-ui, sans-serif; margin: 2em; color: #1b1b1b;
       background: #fff; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; }
th, td { border: 1px solid #d0d0d0; padding: 4px 8px; }
th { background: #f3f3f3; font-weight: 600; }
th.ref { background: #e6ecf4; }
td.ref { background: #f5f8fc; }
th.g, td.g { border-left: 2px solid #8a8a8a; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.best { font-weight: 700; }
td.skip { color: #9a9a9a; text-align: center; }
td.err { color: #b3261e; text-align: center; }
td.date { white-space: nowrap; }
td.subject { max-width: 20em; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }
tr.old td { color: #8a8a8a; }
code { font-size: 13px; }
.note { color: #555; max-width: 60em; }
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _classes(*names: str) -> str:
    joined = " ".join(n for n in names if n)
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


def _reference_gap(reference: dict | None, row: dict, spec: Spec) -> str | None:
    """Why ``row`` shows no reference columns, or None when it shows them."""
    if reference is None:
        return "reference not computed yet"
    if reference.get("reference_version") != spec.reference_version:
        return "reference out of date"
    if row.get("bench_version") != reference.get("bench_version"):
        return "no reference for this benchmark version"
    return None


def _reference_note(reference: dict | None, spec: Spec) -> str:
    note = (
        f"The {REFERENCE_LABEL} and {BASE_LABEL} columns do not depend on the "
        "commit: they are computed once per benchmark version and repeated on "
        "every row of it."
    )
    if (
        reference is None
        or reference.get("reference_version") != spec.reference_version
    ):
        return f"{note} They are not computed yet."
    run = reference["run"]
    libraries = ", ".join(
        f"{name} {run[name]}" for name in REFERENCE_LIBRARIES if run.get(name)
    )
    return (
        f"{note} Shown: benchmark v{reference['bench_version']}, computed "
        f"{run['finished'][:10]} with {libraries}."
    )


def render(data: dict, spec: Spec) -> str:
    techs = spec.technologies
    ref_columns = [t.id for t in techs] + [BASE_COLUMN]
    reference = data["references"].get(spec.model_id)
    head = [
        [f'<th rowspan="3">{h}</th>' for h in ("Commit", "Date", "Subject")],
        [],
        [],
    ]
    for intrinsic in spec.intrinsics:
        head[0].append(
            f'<th class="g" colspan="{len(techs) + len(ref_columns)}">'
            f"{_esc(intrinsic.name)}<br>"
            f"<small>{_esc(intrinsic.headline_label)}</small></th>"
        )
        head[1].append(
            f'<th class="g" colspan="{len(techs)}">{_esc(ENGINE_LABEL)}</th>'
        )
        head[1].append(
            f'<th class="ref" colspan="{len(techs)}">{_esc(REFERENCE_LABEL)}</th>'
        )
        head[1].append(f'<th class="ref" rowspan="2">{_esc(BASE_LABEL)}</th>')
        head[2].extend(
            f"<th{_classes('g' if k == 0 else '')}>{_esc(t.label)}</th>"
            for k, t in enumerate(techs)
        )
        head[2].extend(f'<th class="ref">{_esc(t.label)}</th>' for t in techs)

    body = []
    for row in [r for r in data["rows"] if r["model"] == spec.model_id]:
        commit = row["commit"]
        sha = commit["sha"]
        old = row.get("bench_version") != spec.bench_version
        gap = _reference_gap(reference, row, spec)
        cells = [
            f'<td><a href="{_esc(COMMIT_URL.format(sha=sha))}"><code>'
            f"{_esc(sha[:8])}</code></a></td>",
            f'<td class="date">{_esc(commit["date"][:10])}</td>',
            f'<td class="subject" title="{_esc(commit["subject"])}">'
            f"{_esc(commit['subject'])}</td>",
        ]
        for intrinsic in spec.intrinsics:
            by_tech = row["cells"].get(intrinsic.id, {})
            by_column = {} if gap else reference["cells"].get(intrinsic.id, {})
            group = [by_tech.get(t.id) for t in techs]
            group += [by_column.get(c) for c in ref_columns]
            scored = [
                v
                for v in (_headline(c, intrinsic.headline) for c in group)
                if v is not None
            ]
            best = max(scored) if len(scored) > 1 else None
            for k, cell in enumerate(group):
                if k < len(techs):
                    extra, missing = ("g" if k == 0 else ""), "not run"
                else:
                    extra, missing = "ref", gap or "not run"
                cells.append(_cell_html(cell, intrinsic.headline, best, extra, missing))
        cls = ' class="old"' if old else ""
        body.append(f"<tr{cls}>" + "".join(cells) + "</tr>")

    tech_notes = "".join(
        f"<li><b>{_esc(t.label)}</b>: {_esc(t.source)}</li>" for t in techs
    )
    n_columns = len(techs) + len(ref_columns)
    return f"""<!DOCTYPE html>
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Generated by benchmarks/adapter_eval/publish.py; do not edit. -->
<html lang="en">
<head>
<meta charset="utf-8">
<title>Granite Switch adapter benchmark</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
<h1>Granite Switch adapter benchmark</h1>
<p class="note">Accuracy of trained intrinsic adapters on
<code>{_esc(spec.base_model)}</code> (greedy decoding). Values are
percentages; the best of an adapter's {n_columns} columns is bold. Hover a
cell for all its metrics, or for the reason it is empty.</p>
<ul class="note">
<li><b>{_esc(ENGINE_LABEL)}</b>: the adapters composed into the base model
with each commit's composer and run with its vLLM backend.</li>
<li><b>{_esc(REFERENCE_LABEL)}</b>: the same adapter checkpoints without
granite-switch, loaded with Hugging Face transformers and PEFT. SR runs on a
standalone Hugging Face implementation of the Shadow Residual model.</li>
<li><b>{_esc(BASE_LABEL)}</b>: the base model with no adapter, with Hugging
Face transformers.</li>
</ul>
<p class="note">{_esc(_reference_note(reference, spec))}</p>
<p class="note">Adapters:</p>
<ul class="note">{tech_notes}</ul>
<p class="note"><b>—</b> no adapter or eval set for this cell;
<b>·</b> not run; <b>error</b> the run failed (hover for the step). Greyed
rows used an older benchmark version (current: v{spec.bench_version}).</p>
<div class="wrap">
<table>
<thead>
<tr>{"".join(head[0])}</tr>
<tr>{"".join(head[1])}</tr>
<tr>{"".join(head[2])}</tr>
</thead>
<tbody>
{chr(10).join(body)}
</tbody>
</table>
</div>
</body>
</html>
"""


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
    args = p.parse_args(argv)

    spec = load_spec()
    if args.cmd == "extract":
        block = extract_block(args.log.read_text(errors="replace"), *BLOCKS[args.kind])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")
        if args.kind in ("results", "reference"):
            bad = [f"{i}/{t}" for i, t, c in iter_cells(block["cells"]) if is_error(c)]
            name = block["commit"]["sha"][:8] if args.kind == "results" else "reference"
            print(f"extracted {name}; error cells: {', '.join(bad) or 'none'}")
        else:
            print(f"extracted {args.kind} block to {args.out}")
        return 0

    data = load_data(args.data, spec)
    if args.cmd in ("check", "check-reference"):
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
