"""Prometheus metrics for the serving service — two-tier observability (Phase 4).

The plan calls for *system* metrics (is the service healthy/fast?) **and** *model*
metrics (is the model behaving?). Because failure labels lag by weeks, we cannot
compute live accuracy; instead we monitor **leading indicators** — the
anomaly-score distribution and the alert rate — which shift *before* ground truth
arrives. A drift in those is the early signal Phase 4's monitor acts on.

Collectors live at module scope (one global registry), so repeated ``create_app``
calls — e.g. across tests — reuse the same series instead of re-registering.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# --- Model metrics (leading indicators) ---
PREDICTIONS = Counter(
    "gridsentinel_predictions_total",
    "Windows scored, labelled by whether they alerted.",
    ["alert"],
)
ANOMALY_SCORE = Histogram(
    "gridsentinel_anomaly_score",
    "Distribution of anomaly scores served (a drift in this shape precedes label drift).",
    buckets=(-0.3, -0.2, -0.1, -0.05, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)

# --- System metrics ---
REQUEST_LATENCY = Histogram(
    "gridsentinel_request_latency_seconds",
    "Request latency by endpoint.",
    ["endpoint"],
)
VALIDATION_ERRORS = Counter(
    "gridsentinel_validation_errors_total",
    "Requests rejected by schema validation (bad/out-of-range telemetry).",
)


def record_prediction(anomaly_score: float, alert: bool) -> None:
    """Record one served prediction (model metrics)."""
    PREDICTIONS.labels(alert=str(alert).lower()).inc()
    ANOMALY_SCORE.observe(anomaly_score)
