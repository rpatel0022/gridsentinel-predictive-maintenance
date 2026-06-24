"""Windowed feature engineering for the MetroPT-3 baseline.

The raw CSV is ~10 s cadence (1.5M rows). For the Phase 1 supervised baseline we
aggregate it into **non-overlapping fixed-width windows** (default 10 min) and
summarise each window per channel. Non-overlapping windows are a deliberate
leakage-safety choice: overlapping/rolling windows share rows, so a naive
train/test split would leak — fixed tiles keep each row in exactly one window, and
the temporal CV splitter then keeps whole windows on one side of the boundary.

Per window we emit, for each analog channel, mean/std/min/max (an air leak shows up
as drift and rising variance in the pressure and duty-cycle signals), and for each
digital channel the mean (its duty fraction over the window). The window's *start*
timestamp is kept as the decision time the labeller and CV splitter use.
"""

from __future__ import annotations

import pandas as pd

from pipelines.metropt3_schema import ANALOG_SENSORS, DIGITAL_SIGNALS

WINDOW_START = "window_start"


def build_windowed_features(
    df: pd.DataFrame,
    *,
    freq: str = "10min",
    min_rows: int = 6,
) -> pd.DataFrame:
    """Aggregate raw MetroPT-3 rows into per-window summary features.

    Args:
        df: Raw frame with a ``timestamp`` column plus the analog/digital signals.
        freq: Pandas offset alias for the window width (e.g. ``"10min"``).
        min_rows: Drop windows with fewer than this many raw rows (the partial
            tail window, or gaps from sensor outages).

    Returns:
        One row per window, sorted by time, with a :data:`WINDOW_START` column
        (window start timestamp) and the aggregated feature columns. The frame is
        ready for :func:`pipelines.labels.make_labels` on ``WINDOW_START``.
    """
    if "timestamp" not in df.columns:
        raise ValueError("df must have a 'timestamp' column")
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values("timestamp")

    grouper = pd.Grouper(key="timestamp", freq=freq)
    agg: dict[str, list[str]] = {c: ["mean", "std", "min", "max"] for c in ANALOG_SENSORS}
    agg.update({c: ["mean"] for c in DIGITAL_SIGNALS})

    grouped = work.groupby(grouper)
    feats = grouped.agg(agg)
    feats.columns = [f"{col}_{stat}" for col, stat in feats.columns]
    counts = grouped.size()

    feats = feats[counts >= min_rows].copy()
    # std is NaN for single-row windows; min_rows>=2 avoids it, but guard anyway.
    feats = feats.fillna(0.0)
    feats.index.name = WINDOW_START
    feats = feats.reset_index().sort_values(WINDOW_START).reset_index(drop=True)
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input columns of a windowed-feature frame (everything but time)."""
    return [c for c in df.columns if c != WINDOW_START]
