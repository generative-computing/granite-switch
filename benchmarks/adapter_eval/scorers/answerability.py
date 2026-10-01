# SPDX-License-Identifier: Apache-2.0
"""Answerability: unanswerable / answerable classification.

Port of the internal ``score_answerability_predictions.py``, unfiltered
("mixed validation set") view, with one difference: a label counts without
its double quotes too. The adapters answer ``"answerable"``; the base model,
told to answer that way, answers ``answerable``, and the strict form scored
all of those as ``others`` (benchmark v2). The headline is accuracy.
"""

from __future__ import annotations

import re

from . import ScoreContext, ScoreResult, require
from ._metrics import confusion_matrix, confusion_report

LABELS = ["unanswerable", "answerable", "others"]


def classify(label: str) -> str:
    # "unanswerable" first: "answerable" is a substring of it.
    s = label.lower()
    if "unanswerable" in s:
        return "unanswerable"
    if "answerable" in s:
        return "answerable"
    return "others"


def normalize_prediction(raw: str) -> str:
    """First whitespace token: its first double-quoted span, else the token.

    '"unanswerable"<|end_of_text|>' -> unanswerable
    'unanswerable'  (no quotes)     -> unanswerable
    'It is answerable'              -> others (the label must come first)
    """
    s = raw.strip()
    if not s:
        return classify("")
    first = s.split()[0]
    matches = re.findall(r'"(.*?)"', first)
    return classify(matches[0] if matches else first)


def score(rows: list[dict], ctx: ScoreContext) -> ScoreResult:
    refs = [classify(require(r, "ground_truth")) for r in rows]
    preds = [normalize_prediction(require(r, "generated_content")) for r in rows]
    report = confusion_report(confusion_matrix(refs, preds, LABELS), LABELS)
    metrics = {
        "accuracy": report["accuracy"],
        "weighted_f1": report["weighted_f1"],
        "unanswerable_f1": report["f1"][0],
        "answerable_f1": report["f1"][1],
        "n": report["total"],
    }
    return ScoreResult(metrics=metrics, details={"overall": report})
