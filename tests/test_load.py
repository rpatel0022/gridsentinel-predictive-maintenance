"""Test for the serving load-test harness (sklearn/httpx-guarded)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from pipelines.anomaly import anomaly_score, build_detector  # noqa: E402
from pipelines.features import aggregate_window, feature_names  # noqa: E402
from pipelines.metropt3_schema import ANALOG_SENSORS, DIGITAL_SIGNALS  # noqa: E402
from serving.app import create_app  # noqa: E402
from serving.load_test import run_load_test  # noqa: E402
from serving.model import ModelBundle  # noqa: E402


def _bundle() -> ModelBundle:
    feats = feature_names()
    reading = {c: 5.0 for c in ANALOG_SENSORS} | {c: 1.0 for c in DIGITAL_SIGNALS}
    rows = [aggregate_window(pd.DataFrame([reading] * 6)) for _ in range(40)]
    X = pd.DataFrame(rows)[feats].to_numpy()
    det = build_detector(n_estimators=20).fit(X)
    thr = float(np.quantile(anomaly_score(det, X), 0.99))
    return ModelBundle(det, thr, feats, "load-v1")


def test_load_test_runs_clean():
    app = create_app(_bundle())
    reading = {c: 5.0 for c in ANALOG_SENSORS} | {c: 1.0 for c in DIGITAL_SIGNALS}
    payload = {"readings": [reading] * 6}
    stats = run_load_test(app, payload, n_requests=20, concurrency=4)
    assert stats["n_requests"] == 20
    assert stats["errors"] == 0
    assert stats["throughput_rps"] > 0
    assert stats["p99_ms"] >= stats["p50_ms"] > 0
