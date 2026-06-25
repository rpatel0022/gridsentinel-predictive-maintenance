"""Tests for the windowed feature engineering."""

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandera")

from pipelines.features import WINDOW_START, build_windowed_features, feature_columns  # noqa: E402
from pipelines.metropt3_schema import ANALOG_SENSORS, DIGITAL_SIGNALS  # noqa: E402


def _raw(n: int = 120, freq_s: int = 10) -> "pd.DataFrame":
    data = {"timestamp": pd.date_range("2020-02-01", periods=n, freq=f"{freq_s}s")}
    for c in ANALOG_SENSORS:
        data[c] = [1.5] * n
    for c in DIGITAL_SIGNALS:
        data[c] = [1] * n
    return pd.DataFrame(data)


def test_aggregates_into_windows():
    df = _raw(n=120, freq_s=10)  # 1200s = two 10-min windows
    feats = build_windowed_features(df, freq="10min")
    assert len(feats) == 2
    assert WINDOW_START in feats.columns


def test_feature_columns_present():
    feats = build_windowed_features(_raw(), freq="10min")
    cols = feature_columns(feats)
    # 7 analog x {mean,std,min,max} + 8 digital x {mean}
    assert len(cols) == len(ANALOG_SENSORS) * 4 + len(DIGITAL_SIGNALS)
    assert "TP2_mean" in cols and "COMP_mean" in cols
    assert WINDOW_START not in cols


def test_digital_mean_is_duty_fraction():
    df = _raw(n=60, freq_s=10)  # 600s = one 10-min window, COMP all 1
    feats = build_windowed_features(df, freq="10min")
    assert feats["COMP_mean"].iloc[0] == pytest.approx(1.0)


def test_partial_window_dropped_by_min_rows():
    df = _raw(n=63, freq_s=10)  # 1 full window (60 rows) + 3 trailing rows
    feats = build_windowed_features(df, freq="10min", min_rows=6)
    assert len(feats) == 1


def test_requires_timestamp():
    with pytest.raises(ValueError):
        build_windowed_features(pd.DataFrame({"x": [1, 2]}))
