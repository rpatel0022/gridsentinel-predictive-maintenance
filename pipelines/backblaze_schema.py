"""Validation contract for the Backblaze Drive Stats *lifetime* table.

This is the fleet-scale companion to MetroPT-3 (ADR-0001). The source is a real,
public per-drive lifetime summary derived from Backblaze's quarterly Drive Stats —
418k drives, 161 models, ~24k real failures — fetched from a GitHub mirror (the
direct Backblaze host is network-blocked here). One row per drive:

* ``serial_number`` — drive id (may be blank for a few rows; tolerated).
* ``capacity_tb`` — drive capacity in TB.
* ``model`` — drive model string (the dominant reliability signal).
* ``min_date`` / ``max_date`` — first- and last-observed dates (``max >= min``).
* ``failed`` — 1 if the drive failed by ``max_date``, else 0 (right-censored).

Unlike MetroPT this is **reliability/lifetime data, not sensor telemetry** — so it
drives survival/failure-rate analysis, not the SMART-sensor pipeline (see
``docs/backblaze_results.md``).
"""

from __future__ import annotations

import pandera.pandas as pa

REQUIRED = ("capacity_tb", "model", "min_date", "max_date", "failed")


def build_schema() -> pa.DataFrameSchema:
    """Build the Backblaze lifetime-table schema."""
    return pa.DataFrameSchema(
        {
            # 0 is Backblaze's "unknown capacity" marker (real, not corruption); allow it.
            "capacity_tb": pa.Column(float, coerce=True, checks=pa.Check.ge(0)),
            "model": pa.Column(str, coerce=True, nullable=False),
            "min_date": pa.Column("datetime64[ns]", coerce=True, nullable=False),
            "max_date": pa.Column("datetime64[ns]", coerce=True, nullable=False),
            "failed": pa.Column(
                int,
                coerce=True,
                checks=pa.Check.isin([0, 1], error="failed must be binary {0, 1}"),
            ),
        },
        checks=pa.Check(
            lambda df: (df["max_date"] >= df["min_date"]).all(),
            error="max_date must be >= min_date (no negative lifetimes)",
        ),
        strict=False,
        coerce=True,
    )


def validate(df, *, lazy: bool = True):
    """Validate a Backblaze lifetime dataframe against :func:`build_schema`."""
    return build_schema().validate(df, lazy=lazy)
