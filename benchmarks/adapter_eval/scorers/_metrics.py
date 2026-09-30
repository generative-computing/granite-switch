# SPDX-License-Identifier: Apache-2.0
"""Classification metrics shared by the scorers.

The harness runs inside the benchmarked commit's virtualenv, which has no
scikit-learn, so the two metric conventions the internal scorers use are
re-implemented here:

* ``confusion_report`` — the answerability / requirement-check convention:
  a fixed label set, ``max(x, 1e-10)`` denominators, support-weighted F1.
* ``sklearn_prf`` / ``sklearn_macro`` / ``accuracy`` — the behaviour of
  ``sklearn.metrics.precision_recall_fscore_support(zero_division=0)`` and
  ``accuracy_score`` that the hallucination and guardian scorers call.
"""

from __future__ import annotations

from collections.abc import Sequence


def confusion_matrix(
    refs: Sequence[str], preds: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """``[ref][pred]`` counts in ``labels`` order."""
    idx = {lbl: i for i, lbl in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for r, p in zip(refs, preds, strict=True):
        m[idx[r]][idx[p]] += 1
    return m


def confusion_report(cm: list[list[int]], labels: Sequence[str]) -> dict:
    """Per-class precision/recall/F1, accuracy and support-weighted F1."""
    n = len(labels)
    total = sum(map(sum, cm))
    diag = sum(cm[i][i] for i in range(n))
    support = [sum(row) for row in cm]
    precision, recall, f1 = [0.0] * n, [0.0] * n, [0.0] * n
    for i in range(n):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = support[i] - tp
        precision[i] = tp / max(tp + fp, 1e-10)
        recall[i] = tp / max(tp + fn, 1e-10)
        f1[i] = 2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1e-10)
    return {
        "labels": list(labels),
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": diag / total if total else 0.0,
        "weighted_f1": (
            sum(f1[i] * support[i] for i in range(n)) / total if total else 0.0
        ),
        "total": total,
        "confusion_matrix": cm,
    }


def _prf_one(y_true: Sequence, y_pred: Sequence, label) -> tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
    pred_pos = sum(1 for p in y_pred if p == label)
    true_pos = sum(1 for t in y_true if t == label)
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / true_pos if true_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sklearn_prf(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> list[dict]:
    """Per-label P/R/F1/support, as ``average=None, zero_division=0``."""
    out = []
    for label in labels:
        p, r, f = _prf_one(y_true, y_pred, label)
        out.append(
            {
                "precision": p,
                "recall": r,
                "f1": f,
                "support": sum(1 for t in y_true if t == label),
            }
        )
    return out


def sklearn_macro(y_true: Sequence, y_pred: Sequence) -> tuple[float, float, float]:
    """Macro P/R/F1 over the labels present in either list.

    Matches sklearn's ``average="macro"`` with no ``labels=`` argument, which
    averages over the sorted union of ``y_true`` and ``y_pred``.
    """
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0, 0.0, 0.0
    rows = [_prf_one(y_true, y_pred, lbl) for lbl in labels]
    n = len(rows)
    return (
        sum(r[0] for r in rows) / n,
        sum(r[1] for r in rows) / n,
        sum(r[2] for r in rows) / n,
    )


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def balanced_accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Mean recall over the classes present in ``y_true``."""
    classes = sorted(set(y_true))
    if not classes:
        return 0.0
    return sum(_prf_one(y_true, y_pred, c)[1] for c in classes) / len(classes)
