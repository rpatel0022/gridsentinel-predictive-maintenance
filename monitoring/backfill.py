"""Delayed-label backfill — true performance, computed once labels finally arrive.

In real predictive maintenance the "did it actually fail?" label lands weeks after
the model served its prediction (a maintenance report is filed). So at serving time
we monitor only leading indicators (drift, anomaly-score shape); we cannot know real
precision/recall yet. This job closes that gap: when the failure reports for a past
window arrive, it re-scores that window's *stored predictions* against the now-known
labels and records the model's **true** performance.

Most portfolios skip this entirely — handling the delayed-label problem honestly is
the senior tell the plan calls out. The core (:func:`backfill_performance`) is pure
(pandas only, so it runs in lean CI); ROC-AUC is added when scikit-learn is present.
"""

from __future__ import annotations

import json
import os

import numpy as np
from pipelines.labels import FAILURE_EVENTS, make_labels


def backfill_performance(
    timestamps,
    scores,
    threshold: float,
    events=FAILURE_EVENTS,
    *,
    warn_hours: float = 2.0,
    at: str = "",
) -> dict:
    """True performance over a now-labelled window of stored predictions.

    Args:
        timestamps: Decision times of the stored predictions.
        scores: The anomaly scores the model emitted at serving time.
        threshold: The alert threshold in force then (alert if score ≥ threshold).
        events: The failure events whose reports have now arrived.
        warn_hours: Lead-in for the detection label (matches serving).
        at: Optional ISO timestamp stamped onto the record.

    Returns:
        Confusion counts, precision, recall (and ROC-AUC if both classes present
        and scikit-learn is installed).
    """
    labels = make_labels(timestamps, warn_hours=warn_hours, label_active=True, events=events)
    labels = labels.to_numpy()
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    result = {
        "at": at,
        "n": int(len(labels)),
        "n_positives": int((labels == 1).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }
    if len(set(labels.tolist())) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            result["roc_auc"] = float(roc_auc_score(labels, scores))
        except ImportError:
            pass
    return result


def record_backfill(result: dict, path: str) -> str:
    """Append a backfill result to a JSONL history (the true-performance log)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(result) + "\n")
    return path
