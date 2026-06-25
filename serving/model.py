"""Model artifact + scoring core for the GridSentinel serving service.

A *bundle* is everything the API needs to turn a window of raw sensor readings into
a decision: the fitted anomaly-detector pipeline, the operating threshold, the exact
feature order, and provenance (version, train time, offline metrics). Bundling the
threshold and feature order *with* the model — not hard-coding them in the service —
is what makes a model swap a one-file change and keeps train/serve features aligned.

The scoring core here is framework-free (no FastAPI), so it is unit-testable on its
own and reused unchanged by the API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import joblib
import pandas as pd
from pipelines.anomaly import anomaly_score
from pipelines.features import aggregate_window, feature_names

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "models/anomaly_detector.joblib")


@dataclass
class ModelBundle:
    """A deployable model: pipeline + threshold + feature order + provenance."""

    detector: object  # fitted sklearn Pipeline (scaler + IsolationForest)
    threshold: float
    features: list[str]
    version: str
    trained_at: str = ""
    metrics: dict = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, path: str = DEFAULT_MODEL_PATH) -> str:
    """Persist a bundle to ``path`` (creating parent dirs). Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Load a bundle from ``path``.

    Raises:
        FileNotFoundError: If no artifact exists there (the service should report
            unhealthy rather than pretend to serve).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"no model artifact at {path}; build one first")
    bundle = joblib.load(path)
    # Serve single-threaded *per request*: an Isolation Forest with n_jobs=-1 spawns
    # all-core parallelism per call, so under concurrent load the threads
    # oversubscribe the CPU and tail latency explodes (caught by the load test).
    # Concurrency belongs at the request level, not inside one inference.
    try:
        bundle.detector.named_steps["iforest"].n_jobs = 1
    except (AttributeError, KeyError):
        pass
    return bundle


def score_features(bundle: ModelBundle, feats: dict[str, float]) -> tuple[float, bool]:
    """Score a single feature dict → ``(anomaly_score, alert)``.

    Raises:
        KeyError: If a feature the model expects is missing.
    """
    missing = [f for f in bundle.features if f not in feats]
    if missing:
        raise KeyError(f"missing features: {missing}")
    row = pd.DataFrame([[feats[f] for f in bundle.features]], columns=bundle.features)
    score = float(anomaly_score(bundle.detector, row.to_numpy())[0])
    return score, bool(score >= bundle.threshold)


def predict_window(bundle: ModelBundle, readings: pd.DataFrame) -> dict:
    """Aggregate a window of raw readings and score it.

    Args:
        bundle: The loaded model bundle.
        readings: Raw rows for one window (the 7 analog + 8 digital signals).

    Returns:
        A dict with the anomaly score, the alert decision, the threshold, the number
        of samples aggregated, and the model version.
    """
    feats = aggregate_window(readings)
    score, alert = score_features(bundle, feats)
    return {
        "anomaly_score": score,
        "alert": alert,
        "threshold": bundle.threshold,
        "n_samples": int(len(readings)),
        "model_version": bundle.version,
    }


def expected_features() -> list[str]:
    """The canonical feature order a freshly-built bundle should carry."""
    return feature_names()
