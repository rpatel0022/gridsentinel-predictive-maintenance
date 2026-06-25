"""Tests for the delayed-label backfill (pandas core — runs in lean CI)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from monitoring.backfill import backfill_performance, record_backfill  # noqa: E402
from pipelines.labels import FAILURE_EVENTS  # noqa: E402

EV = FAILURE_EVENTS[0]  # starts 2020-04-18 00:00, ends 23:59


def _series():
    # 3 windows: well before (negative), 1h before (positive lead-in), mid-failure (positive).
    ts = pd.Series(
        [
            EV.start - pd.Timedelta(days=5),
            EV.start - pd.Timedelta(hours=1),
            EV.start + pd.Timedelta(hours=2),
        ]
    )
    return ts


def test_backfill_perfect_predictions():
    ts = _series()
    scores = [0.0, 1.0, 1.0]  # low for the negative, high for the two positives
    r = backfill_performance(ts, scores, threshold=0.5, events=(EV,))
    assert r["n"] == 3 and r["n_positives"] == 2
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 0
    assert r["recall"] == 1.0 and r["precision"] == 1.0
    assert r["roc_auc"] == 1.0


def test_backfill_counts_misses_and_false_alarms():
    ts = _series()
    scores = [1.0, 0.0, 0.0]  # alarms on the negative, misses both positives
    r = backfill_performance(ts, scores, threshold=0.5, events=(EV,))
    assert r["fp"] == 1 and r["fn"] == 2 and r["tp"] == 0
    assert r["recall"] == 0.0


def test_record_backfill_appends_jsonl(tmp_path):
    import json

    path = tmp_path / "perf" / "history.jsonl"
    record_backfill({"recall": 0.9, "at": "t1"}, str(path))
    record_backfill({"recall": 0.8, "at": "t2"}, str(path))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["at"] == "t2"


def test_threshold_changes_outcome():
    ts = _series()
    scores = np.array([0.0, 0.6, 0.6])
    lenient = backfill_performance(ts, scores, threshold=0.5, events=(EV,))
    strict = backfill_performance(ts, scores, threshold=0.9, events=(EV,))
    assert lenient["recall"] == 1.0
    assert strict["recall"] == 0.0  # nothing clears 0.9
