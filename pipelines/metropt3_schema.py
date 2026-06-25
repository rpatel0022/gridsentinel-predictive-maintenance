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
lets incidental columns (e.g. GPS, row index) pass. **Physical ranges were
tightened against the real distribution** during the Phase 0 data-quality spike
(the full 1,516,948-row UCI #791 CSV, 2020-02-01 → 2020-09-01): zero nulls,
strictly-monotonic non-duplicated timestamps, every digital signal exactly
``{0, 1}``. The analog bounds in :data:`ANALOG_RANGES` sit a physical margin
beyond the observed extremes — wide enough to admit legitimate new readings
(e.g. a colder winter oil temperature than the Feb–Sep sample saw), tight enough
to reject genuine corruption (a stuck ``9999`` line, a sign-flipped channel).
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

# Physical bounds (inclusive) for the analog channels, derived from the Phase 0
# data-quality spike. Pressures are in bar, oil temperature in °C, motor current
# in A. Each bound brackets the observed min/max with a deliberate margin so the
# check fails on corruption, not on plausible unseen operating points. Observed
# extremes over the full dataset are shown for provenance.
#                            (low,   high)   # observed [min, max]
ANALOG_RANGES: dict[str, tuple[float, float]] = {
    "TP2": (-1.0, 12.0),  # [-0.032, 10.676]
    "TP3": (-1.0, 12.0),  # [ 0.730, 10.302]
    "H1": (-1.0, 12.0),  # [-0.036, 10.288]
    "DV_pressure": (-1.0, 12.0),  # [-0.032,  9.844]
    "Reservoirs": (-1.0, 12.0),  # [ 0.712, 10.300]
    "Oil_temperature": (-10.0, 120.0),  # [15.400, 89.050]
    "Motor_current": (-1.0, 12.0),  # [ 0.020,  9.295]
}

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
        # Floats, no nulls, within the spike-derived physical bounds.
        low, high = ANALOG_RANGES[name]
        columns[name] = pa.Column(
            float,
            coerce=True,
            nullable=False,
            checks=pa.Check.in_range(
                low,
                high,
                error=f"{name} outside physical range [{low}, {high}]",
            ),
        )
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
