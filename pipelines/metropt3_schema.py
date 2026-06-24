"""Data-validation contract for the MetroPT-3 dataset.

MetroPT-3 (UCI #791 / Zenodo 7766691) is the real Air Production Unit (compressor)
telemetry from Porto metro trains: seven analog sensor channels plus eight digital
signals, sampled over time, with real failure events from maintenance reports.

This module encodes the *expected shape* of that data as a pandera schema so every
row that enters the feature pipeline is checked for schema drift, out-of-range
values, nulls, and — critically — time ordering (shuffling before a temporal split
is the classic leakage bug). Validating inputs is Upgrade 2 from PLAN.md: a real
data-validation suite is the cleanest differentiator most portfolios skip.

Column names/dtypes follow the published dataset documentation; ``strict=False``
lets incidental columns (e.g. GPS, row index) pass. **Physical ranges are marked
TBC** and get tightened against the real distribution during the Phase 0
data-quality spike — the contract is intentionally permissive until then so it
fails on genuine corruption, not on assumptions.
"""

from __future__ import annotations

import pandera.pandas as pa

TIMESTAMP = "timestamp"

# Seven analog sensor channels (pressures, temperature, motor current).
ANALOG_SENSORS: tuple[str, ...] = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
)

# Eight digital signals — each is binary {0, 1}.
DIGITAL_SIGNALS: tuple[str, ...] = (
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)


def build_schema() -> pa.DataFrameSchema:
    """Build the MetroPT-3 validation schema.

    Returns:
        A pandera ``DataFrameSchema`` requiring the timestamp, the seven analog
        sensors (non-null floats), and the eight binary digital signals, with the
        timestamp checked for ascending order. Extra columns are allowed.
    """
    columns: dict[str, pa.Column] = {
        TIMESTAMP: pa.Column(
            "datetime64[ns]",
            coerce=True,
            nullable=False,
            checks=pa.Check(
                lambda s: bool(s.is_monotonic_increasing),
                error="timestamps must be sorted ascending (no shuffling before temporal CV)",
            ),
        ),
    }
    for name in ANALOG_SENSORS:
        # Floats, no nulls. Physical bounds are TBC pending the data-quality spike.
        columns[name] = pa.Column(float, coerce=True, nullable=False)
    for name in DIGITAL_SIGNALS:
        columns[name] = pa.Column(
            float,
            coerce=True,
            nullable=False,
            checks=pa.Check.isin([0, 1], error=f"{name} must be binary {{0, 1}}"),
        )
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


def validate(df, *, lazy: bool = True):
    """Validate a MetroPT-3 dataframe against :func:`build_schema`.

    Args:
        df: The dataframe to validate.
        lazy: If True (default), collect *all* schema errors before raising, so a
            data-quality report shows every problem at once rather than the first.

    Returns:
        The validated (and dtype-coerced) dataframe.

    Raises:
        pandera.errors.SchemaError / SchemaErrors: If validation fails.
    """
    return build_schema().validate(df, lazy=lazy)
