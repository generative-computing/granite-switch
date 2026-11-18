# SPDX-License-Identifier: Apache-2.0
"""The marker filters that keep the release tier out of every other test run.

A `-m` on the pytest command line replaces the one in `addopts` instead of adding
to it, so the exclusions have to be stated in each place: `addopts` for plain
`pytest` runs, and every `-m` in the CI workflows. Dropping one fails nothing on
its own: the run quietly includes the whole release tier, downloading real models
(a merge once lost `not release` from `addopts` and doubled the GPU runtime).
"""

import re
import shlex
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
MARK_EXPR_RE = re.compile(r"""-m\s+(["'])(.+?)\1""")


def _pytest_ini() -> dict:
    with open(REPO / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["pytest"]["ini_options"]


def _workflow_mark_expressions() -> list[tuple[str, str]]:
    found = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for match in MARK_EXPR_RE.finditer(line):
                found.append((f"{path.name}:{lineno}", match.group(2)))
    return found


def _selects_release(expr: str) -> bool:
    return "release" in expr.replace("not release", "")


def test_addopts_excludes_deep_and_release():
    addopts = _pytest_ini()["addopts"]
    args = shlex.split(addopts) if isinstance(addopts, str) else list(addopts)
    assert "-m" in args, f"addopts has no -m filter: {addopts!r}"
    expr = args[args.index("-m") + 1]
    assert "not deep" in expr and "not release" in expr, (
        f"addopts -m {expr!r} must exclude both deep and release; without them a "
        "plain `pytest` run includes the deep suite and the whole release tier"
    )


def test_release_marker_is_registered():
    markers = _pytest_ini()["markers"]
    assert any(m.split(":", 1)[0].strip() == "release" for m in markers)


def test_workflows_have_mark_expressions():
    # Guards the check below against passing vacuously if the regex stops matching.
    assert _workflow_mark_expressions()


@pytest.mark.parametrize(
    "where,expr", _workflow_mark_expressions(), ids=lambda v: str(v)
)
def test_workflow_mark_expression_keeps_the_exclusions(where, expr):
    assert "not deep" in expr, f"{where}: -m {expr!r} must exclude deep"
    if not _selects_release(expr):
        assert "not release" in expr, (
            f"{where}: -m {expr!r} replaces addopts, so it must exclude release "
            "itself, or the run includes the whole release tier"
        )
