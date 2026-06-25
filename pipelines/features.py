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

# Per-analog summary stats, in the order they appear in the feature vector. Kept as
# a module constant so training (`build_windowed_features`) and serving
# (`aggregate_window`) produce features in the *identical* order — a feature-skew
# bug between train and serve is one of the classic ways a model silently rots.
ANALOG_STATS: tuple[str, ...] = ("mean", "std", "min", "max")


def feature_names() -> list[str]:
    """Canonical ordered feature-column names (analog stats, then digital duties)."""
    names = [f"{c}_{stat}" for c in ANALOG_SENSORS for stat in ANALOG_STATS]
    names += [f"{c}_mean" for c in DIGITAL_SIGNALS]
    return names


def aggregate_window(df: pd.DataFrame) -> dict[str, float]:
    """Summarise one window's raw rows into the feature dict the model consumes.

    Same statistics as :func:`build_windowed_features` for a single window, so a
    request to the serving API yields exactly the features the model trained on.
    ``std`` is NaN for a single row; we fill 0.0 to match the training pipeline.
    """
    feats: dict[str, float] = {}
    for c in ANALOG_SENSORS:
        s = df[c].astype(float)
        feats[f"{c}_mean"] = float(s.mean())
        std = s.std()  # ddof=1, matching pandas .agg(["std"]) in training
        feats[f"{c}_std"] = float(std) if pd.notna(std) else 0.0
        feats[f"{c}_min"] = float(s.min())
        feats[f"{c}_max"] = float(s.max())
    for c in DIGITAL_SIGNALS:
        feats[f"{c}_mean"] = float(df[c].astype(float).mean())
    return feats


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
