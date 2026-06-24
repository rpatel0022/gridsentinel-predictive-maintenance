"""MetroPT-3 data-quality spike — reproduces the profile behind the schema bounds.

This is the Phase 0 data-quality spike referenced in ADR-0001: load the real
MetroPT-3 CSV, validate it against :mod:`pipelines.metropt3_schema`, and profile
the analog channels so the physical ranges in that schema have provenance instead
of being guessed. It is a diagnostic, not part of the serving path — run it once
when the dataset (or the schema's ``ANALOG_RANGES``) changes.

Usage::

    python -m pipelines.data_quality "MetroPT3(AirCompressor).csv"

The dataset itself is never committed (see ``.gitignore``); fetch it from
UCI #791 / Zenodo 7766691 first. Requires the ``pipelines`` extra (pandas/pandera).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from pipelines.metropt3_schema import ANALOG_RANGES, ANALOG_SENSORS, DIGITAL_SIGNALS, validate


def profile(df: pd.DataFrame) -> dict:
    """Compute the data-quality profile used to set/justify the schema bounds.

    Args:
        df: A MetroPT-3 dataframe with a parsed ``timestamp`` column.

    Returns:
        A dict with row count, null counts, timestamp ordering/span/duplicates,
        the observed value set of each digital signal, and per-analog
        min/max/quantiles plus the count of rows outside the schema's bound.
    """
    rep: dict = {"rows": int(len(df))}
    signal_cols = list(ANALOG_SENSORS) + list(DIGITAL_SIGNALS)
    rep["null_counts"] = {k: int(v) for k, v in df[signal_cols].isna().sum().items() if v}
    rep["timestamp_monotonic"] = bool(df["timestamp"].is_monotonic_increasing)
    rep["timestamp_span"] = [str(df["timestamp"].min()), str(df["timestamp"].max())]
    rep["duplicate_timestamps"] = int(df["timestamp"].duplicated().sum())
    rep["digital_value_sets"] = {
        c: sorted(df[c].dropna().unique().tolist()) for c in DIGITAL_SIGNALS
    }
    analog: dict = {}
    for c in ANALOG_SENSORS:
        s = df[c]
        low, high = ANALOG_RANGES[c]
        analog[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "p01": float(s.quantile(0.0001)),
            "p1": float(s.quantile(0.01)),
            "p50": float(s.quantile(0.50)),
            "p99": float(s.quantile(0.99)),
            "p9999": float(s.quantile(0.9999)),
            "bound": [low, high],
            "out_of_bound_rows": int(((s < low) | (s > high)).sum()),
        }
    rep["analog"] = analog
    return rep


def load(csv_path: str) -> pd.DataFrame:
    """Load the MetroPT-3 CSV with the timestamp parsed to datetime."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load, validate, and print the profile.

    Returns a non-zero exit code if schema validation fails, so the spike can
    gate a pipeline.
    """
    parser = argparse.ArgumentParser(description="MetroPT-3 data-quality spike")
    parser.add_argument("csv", help="path to MetroPT3(AirCompressor).csv")
    args = parser.parse_args(argv)

    df = load(args.csv)
    rep = profile(df)

    print(f"rows={rep['rows']:,}  null_cols={rep['null_counts'] or 'none'}")
    print(
        f"timestamp: monotonic={rep['timestamp_monotonic']} "
        f"duplicates={rep['duplicate_timestamps']} span={rep['timestamp_span']}"
    )
    non_binary = {k: v for k, v in rep["digital_value_sets"].items() if set(v) - {0.0, 1.0}}
    print(f"digital signals non-binary: {non_binary or 'none'}")
    print("\nanalog channels (observed [min, max] vs schema bound, out-of-bound rows):")
    for c, m in rep["analog"].items():
        print(
            f"  {c:16s} [{m['min']:8.3f}, {m['max']:8.3f}]  "
            f"bound={m['bound']}  out_of_bound={m['out_of_bound_rows']}"
        )

    try:
        validate(df, lazy=True)
    except Exception as exc:  # noqa: BLE001 - surface any schema failure to the CLI
        print(f"\nSCHEMA VALIDATION: FAIL -> {type(exc).__name__}: {str(exc)[:400]}")
        return 1
    print(f"\nSCHEMA VALIDATION: PASS on all {rep['rows']:,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
