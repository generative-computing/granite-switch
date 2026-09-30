# SPDX-License-Identifier: Apache-2.0
"""Hallucination detection: per-sentence faithful / unfaithful judgments.

Port of the internal ``score_hallucination_predictions.py``. The model emits a
JSON array of ``{"r": <sentence id>, "f": <label>}``; the reference in
``ground_truth`` has the same shape. Rows are grouped by
``hallucination_dataset`` and scored at sentence and response level.

The internal scorer's own aggregate is the mean over datasets of the
response-level macro metrics. The internal results page headlines an
accuracy, so the sentence and response accuracies are reported too, both as
a mean over datasets and pooled over all rows.
"""

from __future__ import annotations

import json
from collections import defaultdict

from . import ScoreContext, ScoreResult, require
from ._metrics import accuracy, sklearn_macro, sklearn_prf

POSITIVE = "faithful"
NEGATIVE = "unfaithful"


def normalize_label(label) -> str:
    if label == "unfaithful" or label == "partial":
        return NEGATIVE
    return POSITIVE


def parse_generated(raw: str | None) -> list | None:
    """Text between the first '[' and the last ']', if it is a JSON list."""
    if raw is None:
        return None
    text = raw.strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        if isinstance(obj, list):
            return obj
    return None


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    per = sklearn_prf(y_true, y_pred, [POSITIVE, NEGATIVE])
    macro_p, macro_r, macro_f1 = sklearn_macro(y_true, y_pred)

    def block(m: dict) -> dict:
        return {
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "support": m["support"],
        }

    return {
        "accuracy": round(accuracy(y_true, y_pred), 4),
        POSITIVE: block(per[0]),
        NEGATIVE: block(per[1]),
        "macro": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
    }


def _sentence_labels(records: list[dict]) -> tuple[list, list, int, int]:
    y_true, y_pred = [], []
    skipped = parse_failures = 0
    for rec in records:
        generated = rec["generated"]
        if generated is None:
            parse_failures += 1
            continue
        ref_by_r = {item["r"]: item["f"] for item in rec["reference"]}
        gen_by_r = {
            item["r"]: item["f"]
            for item in generated
            if isinstance(item, dict) and "r" in item and "f" in item
        }
        for r_idx, ref_label in ref_by_r.items():
            if r_idx not in gen_by_r:
                skipped += 1
                continue
            y_true.append(normalize_label(ref_label))
            y_pred.append(normalize_label(gen_by_r[r_idx]))
    return y_true, y_pred, skipped, parse_failures


def _response_labels(records: list[dict]) -> tuple[list, list, int]:
    y_true, y_pred = [], []
    parse_failures = 0
    for rec in records:
        generated = rec["generated"]
        if generated is None or len(generated) == 0:
            parse_failures += 1
            continue
        ref = [normalize_label(item["f"]) for item in rec["reference"]]
        gen = [
            normalize_label(item["f"])
            for item in generated
            if isinstance(item, dict) and "f" in item
        ]
        y_true.append(NEGATIVE if NEGATIVE in ref else POSITIVE)
        y_pred.append(NEGATIVE if NEGATIVE in gen else POSITIVE)
    return y_true, y_pred, parse_failures


def evaluate_sentence_level(records: list[dict]) -> dict | None:
    y_true, y_pred, skipped, parse_failures = _sentence_labels(records)
    if not y_true:
        return None
    m = compute_metrics(y_true, y_pred)
    m.update(
        total_instances=len(records),
        total_sentences=len(y_true),
        parse_failures=parse_failures,
        skipped_sentences=skipped,
    )
    return m


def evaluate_response_level(records: list[dict]) -> dict | None:
    y_true, y_pred, parse_failures = _response_labels(records)
    if not y_true:
        return None
    m = compute_metrics(y_true, y_pred)
    m.update(total_responses=len(y_true), parse_failures=parse_failures)
    return m


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score(rows: list[dict], ctx: ScoreContext) -> ScoreResult:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        reference = require(row, "ground_truth")
        if reference is None:
            raise ValueError("prediction row has a null ground_truth")
        by_dataset[row.get("hallucination_dataset", "unknown")].append(
            {
                "reference": reference,
                "generated": parse_generated(row.get("generated_content")),
            }
        )

    per_dataset: dict[str, dict] = {}
    for name in sorted(by_dataset):
        levels = {}
        sent = evaluate_sentence_level(by_dataset[name])
        if sent:
            levels["sentence_level"] = sent
        resp = evaluate_response_level(by_dataset[name])
        if resp:
            levels["response_level"] = resp
        per_dataset[name] = levels

    def level_mean(level: str, *path: str) -> float | None:
        vals = []
        for levels in per_dataset.values():
            if level in levels:
                v = levels[level]
                for key in path:
                    v = v[key]
                vals.append(v)
        return _mean(vals)

    all_records = [rec for recs in by_dataset.values() for rec in recs]
    s_true, s_pred, _, _ = _sentence_labels(all_records)
    r_true, r_pred, _ = _response_labels(all_records)
    parse_failures = sum(1 for rec in all_records if rec["generated"] is None)

    candidates = {
        "response_macro_f1_mean": level_mean("response_level", "macro", "f1"),
        "response_accuracy_mean": level_mean("response_level", "accuracy"),
        "sentence_accuracy_mean": level_mean("sentence_level", "accuracy"),
        "sentence_macro_f1_mean": level_mean("sentence_level", "macro", "f1"),
        "sentence_accuracy_pooled": accuracy(s_true, s_pred) if s_true else None,
        "response_accuracy_pooled": accuracy(r_true, r_pred) if r_true else None,
    }
    metrics = {k: v for k, v in candidates.items() if v is not None}
    metrics["parse_failures"] = parse_failures
    metrics["n"] = len(rows)
    details = {
        "aggregate_response_level": {
            "avg_macro_precision": level_mean("response_level", "macro", "precision"),
            "avg_macro_recall": level_mean("response_level", "macro", "recall"),
            "avg_macro_f1": candidates["response_macro_f1_mean"],
            "num_datasets": sum(1 for lv in per_dataset.values() if lv),
        },
        "per_dataset": per_dataset,
    }
    return ScoreResult(metrics=metrics, details=details)
