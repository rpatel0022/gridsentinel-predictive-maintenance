"""Tests for the MetroPT-3 data-validation contract.

The fixtures here are tiny, hand-built frames used *only* to exercise the
validator — they are not training data and never feed a model, so the project's
"100% real data" mandate is intact.
"""

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pandera")

from pandera.errors import SchemaError, SchemaErrors  # noqa: E402
from pipelines.metropt3_schema import (  # noqa: E402
    ANALOG_SENSORS,
    DIGITAL_SIGNALS,
    build_schema,
    validate,
)


def _valid_frame(n: int = 3) -> "pd.DataFrame":
    """A minimal well-formed MetroPT-3-shaped frame."""
    data = {"timestamp": pd.date_range("2026-01-01", periods=n, freq="min")}
    for col in ANALOG_SENSORS:
        data[col] = [1.5] * n
    for col in DIGITAL_SIGNALS:
        data[col] = [0] * n
    return pd.DataFrame(data)


def test_valid_frame_passes():
    out = validate(_valid_frame())
    assert list(out.columns)[: 1 + len(ANALOG_SENSORS) + len(DIGITAL_SIGNALS)]
    assert len(out) == 3


def test_extra_columns_allowed():
    df = _valid_frame()
    df["gpsLong"] = [-8.6] * len(df)  # incidental column tolerated (strict=False)
    validate(df)


def test_digital_signal_must_be_binary():
    df = _valid_frame()
    df.loc[0, DIGITAL_SIGNALS[0]] = 2  # not in {0, 1}
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(df)


def test_null_analog_rejected():
    df = _valid_frame()
    df.loc[0, ANALOG_SENSORS[0]] = None
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(df)


def test_unsorted_timestamps_rejected():
    df = _valid_frame()
    # Reverse time order — must be caught before any temporal split (leakage guard).
    df["timestamp"] = df["timestamp"][::-1].to_numpy()
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(df)


def test_missing_required_column_rejected():
    df = _valid_frame().drop(columns=[ANALOG_SENSORS[0]])
    with pytest.raises((SchemaError, SchemaErrors)):
        validate(df)


def test_schema_builds():
    schema = build_schema()
    assert "timestamp" in schema.columns
