# SPDX-License-Identifier: Apache-2.0
"""Cache check, results merge and page render for the adapter benchmark.

Runs locally, not on the pod::

    python -m benchmarks.adapter_eval.publish check <sha>        # exit 0 = hit
    python -m benchmarks.adapter_eval.publish extract <pod.log> --out results.json
    python -m benchmarks.adapter_eval.publish extract <pod.log> --kind discovery ...
    python -m benchmarks.adapter_eval.publish merge results.json
    python -m benchmarks.adapter_eval.publish render

The page data (``docs/benchmarks/data.json``) holds one row per commit, each
row being the results block ``run_benchmark.py`` printed. A commit is a cache
hit when its row has the current ``bench_version`` and every
(intrinsic, technology) cell is present and not an error. Skipped cells
(nothing staged) do count as done: staging a new adapter is a
``bench_version`` bump.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from . import stage
from .common import (
    BEGIN_MARKER,
    END_MARKER,
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
    "discovery": (stage.DISCOVERY_BEGIN, stage.DISCOVERY_END),
    "stage": (stage.STAGE_BEGIN, stage.STAGE_END),
}


# --- data ------------------------------------------------------------------


def load_data(path: Path) -> dict:
    if not path.is_file():
        return {"rows": []}
    return json.loads(path.read_text())


def save_data(path: Path, data: dict, spec: Spec) -> None:
    data["spec"] = spec.public()
    data["rows"].sort(key=lambda r: r["commit"]["date"], reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def find_row(data: dict, sha: str) -> dict | None:
    matches = [r for r in data["rows"] if r["commit"]["sha"].startswith(sha)]
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

    A full run replaces the commit's row. An ``--only`` run replaces just
    those intrinsics inside the existing row of the same ``bench_version``.
    """
    run = results["run"]
    if run.get("limit") is not None:
        raise ValueError("refusing to publish a --limit run")
    if results["bench_version"] != spec.bench_version:
        raise ValueError(
            f"results are bench_version {results['bench_version']}, "
            f"adapters.yaml is {spec.bench_version}"
        )
    sha = results["commit"]["sha"]
    old = find_row(data, sha)
    if run.get("only") and old and old.get("bench_version") == spec.bench_version:
        for intrinsic_id in run["only"]:
            old["cells"][intrinsic_id] = results["cells"][intrinsic_id]
        old.setdefault("updates", []).append(run)
        return old
    if old:
        data["rows"].remove(old)
    data["rows"].append(results)
    return results


# --- page ------------------------------------------------------------------

PAGE_STYLE = """
:root { color-scheme: light; }
body { font: 14px/1.4 system-ui, sans-serif; margin: 2em; color: #1b1b1b;
       background: #fff; }
table { border-collapse: collapse; }
th, td { border: 1px solid #d0d0d0; padding: 4px 8px; }
th { background: #f3f3f3; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.best { font-weight: 700; }
td.skip { color: #9a9a9a; text-align: center; }
td.err { color: #b3261e; text-align: center; }
td.date { white-space: nowrap; }
td.subject { max-width: 28em; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }
tr.old td { color: #8a8a8a; }
code { font-size: 13px; }
.note { color: #555; max-width: 60em; }
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _cell_title(cell: dict) -> str:
    return ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(cell.items())
        if isinstance(v, int | float)
    )


def _cell_html(cell: dict | None, headline: str, best: float | None) -> str:
    if cell is None:
        return '<td class="skip" title="not run">·</td>'
    if "skipped" in cell:
        return f'<td class="skip" title="{_esc(cell["skipped"])}">—</td>'
    if is_error(cell):
        return f'<td class="err" title="{_esc(cell["error"])}">error</td>'
    value = cell.get(headline)
    if not isinstance(value, int | float):
        return f'<td class="err" title="no {_esc(headline)} metric">?</td>'
    cls = "num best" if best is not None and value == best else "num"
    return f'<td class="{cls}" title="{_esc(_cell_title(cell))}">{value * 100:.1f}</td>'


def render(data: dict, spec: Spec) -> str:
    techs = spec.technologies
    head_top = ['<th rowspan="2">Commit</th>', '<th rowspan="2">Date</th>']
    head_top.append('<th rowspan="2">Subject</th>')
    head_sub = []
    for intrinsic in spec.intrinsics:
        head_top.append(
            f'<th colspan="{len(techs)}">{_esc(intrinsic.name)}<br>'
            f"<small>{_esc(intrinsic.headline_label)}</small></th>"
        )
        head_sub.extend(f"<th>{_esc(t.label)}</th>" for t in techs)

    body = []
    for row in data["rows"]:
        commit = row["commit"]
        sha = commit["sha"]
        old = row.get("bench_version") != spec.bench_version
        cells = [
            f'<td><a href="{_esc(COMMIT_URL.format(sha=sha))}"><code>'
            f"{_esc(sha[:8])}</code></a></td>",
            f'<td class="date">{_esc(commit["date"][:10])}</td>',
            f'<td class="subject" title="{_esc(commit["subject"])}">'
            f"{_esc(commit['subject'])}</td>",
        ]
        for intrinsic in spec.intrinsics:
            by_tech = row["cells"].get(intrinsic.id, {})
            scored = [
                c[intrinsic.headline]
                for c in by_tech.values()
                if is_scored(c) and isinstance(c.get(intrinsic.headline), int | float)
            ]
            best = max(scored) if len(scored) > 1 else None
            cells.extend(
                _cell_html(by_tech.get(t.id), intrinsic.headline, best) for t in techs
            )
        cls = ' class="old"' if old else ""
        body.append(f"<tr{cls}>" + "".join(cells) + "</tr>")

    tech_notes = "".join(
        f"<li><b>{_esc(t.label)}</b>: {_esc(t.source)}</li>" for t in techs
    )
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
<p class="note">Accuracy of trained intrinsic adapters after they are composed
into <code>{_esc(spec.base_model)}</code> with each commit's composer and run
with its vLLM backend (greedy decoding). Values are percentages; the best
technology per adapter is bold. Hover a cell for all its metrics, or for the
reason it is empty.</p>
<ul class="note">{tech_notes}</ul>
<p class="note"><b>—</b> no adapter or eval set for this cell;
<b>·</b> not run for this commit; <b>error</b> the run failed (hover for the
step). Greyed rows used an older benchmark version (current:
v{spec.bench_version}).</p>
<table>
<thead>
<tr>{"".join(head_top)}</tr>
<tr>{"".join(head_sub)}</tr>
</thead>
<tbody>
{chr(10).join(body)}
</tbody>
</table>
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
    e = sub.add_parser("extract", help="pull a JSON block out of a pod log")
    e.add_argument("log", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--kind", choices=sorted(BLOCKS), default="results")
    m = sub.add_parser("merge", help="add a run's results to the page data")
    m.add_argument("results", type=Path)
    r = sub.add_parser("render", help="write the page from the page data")
    r.add_argument("--out", type=Path, default=PAGE_PATH)
    args = p.parse_args(argv)

    spec = load_spec()
    if args.cmd == "extract":
        block = extract_block(args.log.read_text(errors="replace"), *BLOCKS[args.kind])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(block, indent=2, sort_keys=True) + "\n")
        if args.kind == "results":
            bad = [f"{i}/{t}" for i, t, c in iter_cells(block["cells"]) if is_error(c)]
            sha = block["commit"]["sha"][:8]
            print(f"extracted {sha}; error cells: {', '.join(bad) or 'none'}")
        else:
            print(f"extracted {args.kind} block to {args.out}")
        return 0

    data = load_data(args.data)
    if args.cmd == "check":
        row = find_row(data, args.sha)
        if cache_hit(row, spec):
            print(json.dumps(row, indent=2, sort_keys=True))
            return 0
        if row is None:
            print(f"cache miss: no row for {args.sha}")
        elif row.get("bench_version") != spec.bench_version:
            print(f"cache miss: row is bench_version {row.get('bench_version')}")
        else:
            print(f"cache miss: cells to run: {', '.join(missing_cells(row, spec))}")
        return 1
    if args.cmd == "merge":
        results = json.loads(args.results.read_text())
        row = merge(data, results, spec)
        save_data(args.data, data, spec)
        print(f"merged {row['commit']['sha'][:8]} into {args.data}")
        return 0
    if args.cmd == "render":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(data, spec))
        print(f"wrote {args.out} ({len(data['rows'])} rows)")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
