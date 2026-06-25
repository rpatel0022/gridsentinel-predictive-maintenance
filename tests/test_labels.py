"""Tests for the MetroPT-3 failure labelling, derived from the real report table."""

import pytest

pd = pytest.importorskip("pandas")

from pipelines.labels import DROP, FAILURE_EVENTS, FailureEvent, make_labels  # noqa: E402


def test_four_real_events():
    assert len(FAILURE_EVENTS) == 4
    assert all(ev.kind == "air_leak" for ev in FAILURE_EVENTS)


def test_warning_window_is_positive():
    ev = FAILURE_EVENTS[0]  # starts 2020-04-18 00:00
    ts = pd.Series([ev.start - pd.Timedelta(hours=2)])
    assert make_labels(ts, warn_hours=24).iloc[0] == 1


def test_outside_window_is_negative():
    ev = FAILURE_EVENTS[0]
    ts = pd.Series([ev.start - pd.Timedelta(days=10)])
    assert make_labels(ts, warn_hours=24).iloc[0] == 0


def test_inside_failure_positive_in_detection_mode():
    ev = FAILURE_EVENTS[0]
    ts = pd.Series([ev.start + pd.Timedelta(hours=1)])
    assert make_labels(ts, warn_hours=24, label_active=True).iloc[0] == 1


def test_inside_failure_dropped_in_predict_mode():
    ev = FAILURE_EVENTS[0]
    ts = pd.Series([ev.start + pd.Timedelta(hours=1)])
    assert make_labels(ts, warn_hours=24, label_active=False).iloc[0] == DROP


def test_horizon_controls_positive_span():
    ev = FAILURE_EVENTS[0]
    t = pd.Series([ev.start - pd.Timedelta(hours=10)])
    assert make_labels(t, warn_hours=4).iloc[0] == 0  # outside a 4h horizon
    assert make_labels(t, warn_hours=24).iloc[0] == 1  # inside a 24h horizon


def test_custom_event_list():
    ev = FailureEvent(pd.Timestamp("2021-01-02 00:00"), pd.Timestamp("2021-01-02 06:00"), "x", "y")
    ts = pd.Series(
        [
            pd.Timestamp("2021-01-01 23:00"),  # 1h before -> positive
            pd.Timestamp("2021-01-02 03:00"),  # inside -> drop
            pd.Timestamp("2020-12-01 00:00"),  # far before -> negative
        ]
    )
    out = make_labels(ts, warn_hours=24, label_active=False, events=(ev,))
    assert out.tolist() == [1, DROP, 0]
    # In detection mode the in-failure row becomes positive instead of dropped.
    out_detect = make_labels(ts, warn_hours=24, label_active=True, events=(ev,))
    assert out_detect.tolist() == [1, 1, 0]
