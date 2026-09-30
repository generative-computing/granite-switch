# SPDX-License-Identifier: Apache-2.0
"""Query clarification: answer "CLEAR" or ask a clarifying question.

Port of the internal ``score_query_clarification_predictions.py``:

* ``qc_type == "ambiguous"``: correct if the parsed output is not "CLEAR".
* otherwise: correct if the parsed output is "CLEAR".

The headline is overall accuracy; per-``qc_category`` accuracies are also
reported. The generation-quality judge is a separate pass and is not run.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from . import ScoreContext, ScoreResult

CATEGORY_KEYS = (
    ("underspecified", "underspecified_accuracy"),
    ("clear_random", "clear_random_accuracy"),
    ("clear_hard", "clear_hard_accuracy"),
)


def parse_clarification(raw: str | None) -> str:
    """'{"clarification": "CLEAR"}' -> CLEAR; falls back to the raw text."""
    if raw is None:
        return ""
    s = raw.strip()
    m = re.search(r'\{\s*"clarification"\s*:\s*"([^"]*)"\s*\}', s)
    if m:
        return m.group(1)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "clarification" in obj:
            return str(obj["clarification"])
        if isinstance(obj, str):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    parts = s.split('"clarification":', 1)
    if len(parts) > 1:
        q = re.search(r'"([^"]*)"', parts[1])
        if q:
            return q.group(1)
    return s


def is_correct(qc_type: str | None, generated: str | None) -> bool:
    g = parse_clarification(generated).strip().upper()
    if qc_type == "ambiguous":
        return g != "CLEAR"
    return g == "CLEAR"


def score(rows: list[dict], ctx: ScoreContext) -> ScoreResult:
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    for r in rows:
        ok = is_correct(r.get("qc_type"), r.get("generated_content", ""))
        cat = r.get("qc_category", "unknown")
        by_cat[cat][1] += 1
        overall[1] += 1
        if ok:
            by_cat[cat][0] += 1
            overall[0] += 1

    def acc(pair: list[int]) -> float:
        return pair[0] / pair[1] if pair[1] else 0.0

    metrics: dict = {"overall_accuracy": acc(overall)}
    for cat, key in CATEGORY_KEYS:
        if cat in by_cat:
            metrics[key] = acc(by_cat[cat])
    metrics["n"] = overall[1]
    details = {
        "overall_correct": overall[0],
        "overall_total": overall[1],
        "by_category": {
            cat: {"accuracy": acc(v), "correct": v[0], "total": v[1]}
            for cat, v in sorted(by_cat.items())
        },
    }
    return ScoreResult(metrics=metrics, details=details)
