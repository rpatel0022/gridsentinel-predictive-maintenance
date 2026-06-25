"""Tests for the unsupervised anomaly detector (Phase 2).

Tiny synthetic frames only — they exercise the detector's plumbing, never feed a
real model. Skipped where scikit-learn isn't installed (the ``modeling`` extra), so
the lean CI stays green.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from pipelines.anomaly import (  # noqa: E402
    anomaly_score,
    build_detector,
    evaluate,
    lead_times,
    normal_training_mask,
)
from pipelines.features import WINDOW_START  # noqa: E402
from pipelines.labels import FAILURE_EVENTS  # noqa: E402

FIRST = FAILURE_EVENTS[0].start  # 2020-04-18 00:00


def test_normal_mask_is_pre_first_failure():
    ts = pd.Series([FIRST - pd.Timedelta(days=1), FIRST, FIRST + pd.Timedelta(days=1)])
    mask = normal_training_mask(ts)
    assert mask.tolist() == [True, False, False]


def test_injected_outlier_scores_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(200, 3))
    det = build_detector().fit(normal)
    outlier = np.array([[50.0, 50.0, 50.0]])
    assert anomaly_score(det, outlier)[0] > np.median(anomaly_score(det, normal))


def test_lead_times_measures_pre_failure_crossing():
    ts = pd.Series(
        [
            FIRST - pd.Timedelta(hours=3),
            FIRST - pd.Timedelta(hours=1),
            FIRST + pd.Timedelta(hours=1),
        ]
    )
    scores = np.array([0.0, 1.0, 1.0])  # crosses 0.5 one hour before the failure
    lt = lead_times(ts, scores, threshold=0.5, events=(FAILURE_EVENTS[0],))
    assert lt[0]["lead_hours"] == pytest.approx(1.0)
    assert lt[0]["flagged_during"] is True


def test_lead_times_reports_no_pre_alert():
    ts = pd.Series([FIRST - pd.Timedelta(hours=1)])
    scores = np.array([0.0])  # never crosses
    lt = lead_times(ts, scores, threshold=0.5, events=(FAILURE_EVENTS[0],))
    assert lt[0]["lead_hours"] is None


def test_evaluate_separates_injected_failure():
    end = FAILURE_EVENTS[0].end
    # Train: normal windows before the failure. Test must contain BOTH classes:
    # anomalous in-failure windows (positive) and post-failure normal windows
    # (negative), so ROC-AUC is defined.
    pre = pd.date_range(
        FIRST - pd.Timedelta(hours=20), FIRST - pd.Timedelta(minutes=10), freq="10min"
    )
    during = pd.date_range(FIRST, end - pd.Timedelta(minutes=10), freq="10min")
    post = pd.date_range(end + pd.Timedelta(hours=1), end + pd.Timedelta(hours=11), freq="10min")
    starts = list(pre) + list(during) + list(post)
    n = len(starts)
    rng = np.random.default_rng(1)
    f1 = rng.normal(0, 0.1, n)
    f2 = rng.normal(0, 0.1, n)
    # Spike features only during the failure interval.
    in_failure = np.array([FIRST <= t <= end for t in starts])
    f1[in_failure] += 8.0
    feats = pd.DataFrame({WINDOW_START: starts, "TP2_mean": f1, "Motor_current_std": f2})
    res = evaluate(feats, warn_hours=2.0)
    assert res["roc_auc"] > 0.8
    assert res["test_positives"] > 0
