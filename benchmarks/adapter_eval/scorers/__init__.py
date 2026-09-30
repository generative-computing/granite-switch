# SPDX-License-Identifier: Apache-2.0
"""Per-intrinsic scorers.

Each scorer takes prediction rows — an eval row plus the model output under
``generated_content`` — and returns a ``ScoreResult``:

* ``metrics``: flat ``{name: number}``, published on the results page. Values
  are fractions in [0, 1] except counts. Always includes ``n``.
* ``details``: the full report (per-class, per-dataset, ...), saved next to
  the predictions on COS but never published.

The parsing and metric rules are ports of the internal
``scripts/score_*_predictions.py`` scorers, so numbers are comparable with the
internal results page. They are kept exact on purpose — change them only
together with ``BENCH_VERSION``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path


class ScorerUnavailable(Exception):
    """The cell cannot be scored in this environment.

    The message is published as the cell's skip reason, so it must not name
    paths, hosts or secrets.
    """


@dataclass
class ScoreContext:
    """What a scorer may need beyond the rows."""

    eval_dir: Path
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass
class ScoreResult:
    metrics: dict[str, float | int]
    details: dict


Scorer = Callable[[list[dict], ScoreContext], ScoreResult]


def get_scorer(name: str) -> Scorer:
    from . import (
        answerability,
        guardian,
        hallucination_detection,
        query_clarification,
        query_rewrite,
        requirements,
    )

    scorers: dict[str, Scorer] = {
        "answerability": answerability.score,
        "hallucination_detection": hallucination_detection.score,
        "guardian": guardian.score,
        "query_clarification": query_clarification.score,
        "query_rewrite": query_rewrite.score,
        "requirements": requirements.score,
    }
    if name not in scorers:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(scorers)}")
    return scorers[name]


def require(row: dict, key: str) -> object:
    if key not in row:
        raise ValueError(f"prediction row missing {key!r}")
    return row[key]
