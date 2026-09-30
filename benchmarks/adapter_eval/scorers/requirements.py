# SPDX-License-Identifier: Apache-2.0
"""Requirement check: does a response meet a requirement, yes or no.

Port of the internal ``score_requirements_predictions.py``. The model must
emit exactly ``{"score": "yes"}`` or ``{"score": "no"}``; anything else is
``invalid`` and counts as wrong.

The internal results page headlines balanced accuracy (the mean of the yes
and no recalls), so that is reported alongside the scorer's own accuracy and
weighted F1.
"""

from __future__ import annotations

import json

from . import ScoreContext, ScoreResult, require
from ._metrics import confusion_matrix, confusion_report

LABELS = ["no", "yes", "invalid"]


def parse_prediction(raw: str | None) -> tuple[str, str | None]:
    """Return ``(label, failure_reason)``; the reason is None on success."""
    text = (raw or "").strip()
    if not text:
        return "invalid", "empty"
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return "invalid", "parse_error"
    if not isinstance(obj, dict):
        return "invalid", "not_object"
    if "score" not in obj:
        return "invalid", "missing_score"
    value = obj["score"]
    if not isinstance(value, str):
        return "invalid", "bad_score_value"
    value = value.strip().lower()
    if value in ("yes", "no"):
        return value, None
    return "invalid", "bad_score_value"


def classify_ref(label) -> str:
    # Some eval sets store the gold answer as the target JSON, not the bare
    # label; accept both.
    if isinstance(label, str) and label.strip().startswith("{"):
        parsed, _ = parse_prediction(label)
        return parsed
    s = (label or "").strip().lower()
    return s if s in ("yes", "no") else "invalid"


def score(rows: list[dict], ctx: ScoreContext) -> ScoreResult:
    refs, preds = [], []
    failures: dict[str, int] = {}
    for r in rows:
        refs.append(classify_ref(require(r, "ground_truth")))
        pred, fail = parse_prediction(require(r, "generated_content"))
        if fail is not None:
            failures[fail] = failures.get(fail, 0) + 1
        preds.append(pred)

    report = confusion_report(confusion_matrix(refs, preds, LABELS), LABELS)
    present = [i for i, lbl in enumerate(LABELS[:2]) if report["support"][i]]
    balanced = (
        sum(report["recall"][i] for i in present) / len(present) if present else 0.0
    )
    metrics = {
        "balanced_accuracy": balanced,
        "accuracy": report["accuracy"],
        "weighted_f1": report["weighted_f1"],
        "yes_f1": report["f1"][1],
        "no_f1": report["f1"][0],
        "invalid": sum(1 for p in preds if p == "invalid"),
        "n": report["total"],
    }
    return ScoreResult(
        metrics=metrics, details={"report": report, "parse_failures": failures}
    )
