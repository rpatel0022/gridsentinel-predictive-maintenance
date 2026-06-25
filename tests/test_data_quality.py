"""Tests for the MetroPT-3 data-quality spike module.

Tiny hand-built frames only — they exercise the profiler, never feed a model, so
the "100% real data" mandate is intact.
"""

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandera")

from pipelines.data_quality import profile  # noqa: E402
from pipelines.metropt3_schema import ANALOG_RANGES, ANALOG_SENSORS, DIGITAL_SIGNALS  # noqa: E402


def _frame(n: int = 5) -> "pd.DataFrame":
    data = {"timestamp": pd.date_range("2026-01-01", periods=n, freq="min")}
    for col in ANALOG_SENSORS:
        data[col] = [1.5] * n
    for col in DIGITAL_SIGNALS:
        data[col] = [0] * n
    return pd.DataFrame(data)


def test_profile_clean_frame():
    rep = profile(_frame(5))
    assert rep["rows"] == 5
    assert rep["null_counts"] == {}
    assert rep["timestamp_monotonic"] is True
    assert rep["duplicate_timestamps"] == 0
    for c in ANALOG_SENSORS:
        assert rep["analog"][c]["out_of_bound_rows"] == 0


def test_profile_flags_out_of_bound():
    df = _frame(5)
    sensor = ANALOG_SENSORS[0]
    df.loc[0, sensor] = ANALOG_RANGES[sensor][1] + 1000.0
    rep = profile(df)
    assert rep["analog"][sensor]["out_of_bound_rows"] == 1


def test_profile_detects_duplicate_timestamps():
    df = _frame(3)
    df.loc[2, "timestamp"] = df.loc[1, "timestamp"]
    rep = profile(df)
    assert rep["duplicate_timestamps"] == 1
