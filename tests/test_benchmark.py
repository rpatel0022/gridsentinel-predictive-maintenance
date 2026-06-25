"""Tests for the edge benchmark helpers (sklearn-guarded)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pipelines.anomaly import build_detector  # noqa: E402
from serving.benchmark import artifact_size_kb, inference_latency_ms  # noqa: E402


def _fitted(n_estimators=20):
    rng = np.random.default_rng(0)
    return build_detector(n_estimators=n_estimators).fit(rng.normal(0, 1, (200, 5)))


def test_artifact_size_positive():
    assert artifact_size_kb(_fitted()) > 0


def test_smaller_ensemble_is_smaller():
    assert artifact_size_kb(_fitted(10)) < artifact_size_kb(_fitted(100))


def test_latency_returns_percentiles():
    row = np.zeros((1, 5))
    out = inference_latency_ms(_fitted(), row, n_iters=50)
    assert set(out) == {"p50_ms", "p99_ms", "mean_ms"}
    assert out["p99_ms"] >= out["p50_ms"] > 0
