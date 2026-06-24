"""Failure labels for MetroPT-3, derived from the company's maintenance reports.

The MetroPT-3 CSV ships **unlabeled**; the ground-truth failures live in the
dataset description's failure table (four "air leak / high stress" events). This
module is the single source of truth for those events and turns them into a
maintenance-decision target: at time ``t``, should we dispatch maintenance?

Two framings, selected by ``label_active`` — both grounded only in the real report
table (no synthetic or injected faults, per the 100%-real-data rule):

* **Detection** (``label_active=True``, the Phase 1 baseline default): positive =
  the ``warn``-hour lead-in *plus* the active failure interval ``[start, end]``.
  Asks "is a leak developing or active now?" — this is where the real signal lives
  (see ``docs/phase1_baseline_results.md``).
* **Predict-ahead** (``label_active=False``): positive = the lead-in
  ``[start - warn, start)`` only; the in-failure interval is *dropped* (the
  compressor is already in a known-bad state, so scoring there would be trivial).
  Asks "can you warn before it starts?" — the Phase 1 spike found **no reliable
  long-horizon signal** here across the four failures under strict temporal CV, an
  honest finding that motivates the Phase 2 fleet data and sequence/anomaly models.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FailureEvent:
    """One real failure from the MetroPT-3 maintenance-report table."""

    start: pd.Timestamp
    end: pd.Timestamp
    kind: str
    severity: str


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# The four real failures (UCI #791 "Failure Information" table). Times are local to
# the dataset; all four are air leaks flagged "high stress".
FAILURE_EVENTS: tuple[FailureEvent, ...] = (
    FailureEvent(_ts("2020-04-18 00:00"), _ts("2020-04-18 23:59"), "air_leak", "high_stress"),
    FailureEvent(_ts("2020-05-29 23:30"), _ts("2020-05-30 06:00"), "air_leak", "high_stress"),
    FailureEvent(_ts("2020-06-05 10:00"), _ts("2020-06-07 14:30"), "air_leak", "high_stress"),
    FailureEvent(_ts("2020-07-15 14:30"), _ts("2020-07-15 19:00"), "air_leak", "high_stress"),
)

# Sentinel for "drop this row" (inside a failure interval, in predict-ahead mode).
DROP = -1


def make_labels(
    timestamps: pd.Series,
    *,
    warn_hours: float = 2.0,
    label_active: bool = True,
    events: tuple[FailureEvent, ...] = FAILURE_EVENTS,
) -> pd.Series:
    """Label each timestamp for the maintenance-decision target.

    Args:
        timestamps: Decision times (e.g. feature-window start times).
        warn_hours: Length of the pre-failure warning lead-in. A row is positive if
            a failure *starts* within this many hours after it.
        label_active: If True (detection), rows inside a failure interval are also
            positive. If False (predict-ahead), they are dropped instead.
        events: The failure events to label against (defaults to the real four).

    Returns:
        An integer Series aligned to ``timestamps``: ``1`` for a warning/active
        row, :data:`DROP` (``-1``) for an in-failure row when ``label_active`` is
        False, else ``0``.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    warn = pd.Timedelta(hours=warn_hours)
    labels = pd.Series(0, index=ts.index, dtype=int)
    for ev in events:
        warning = (ts >= ev.start - warn) & (ts < ev.start)
        labels[warning] = 1
    for ev in events:
        in_failure = (ts >= ev.start) & (ts <= ev.end)
        labels[in_failure] = 1 if label_active else DROP
    return labels
