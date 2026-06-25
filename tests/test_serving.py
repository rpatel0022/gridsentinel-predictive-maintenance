"""Tests for the FastAPI serving service (Phase 3).

A tiny in-memory bundle is fitted on synthetic windows — no real data, no saved
artifact. Skipped where the serving/modeling deps aren't installed, so lean CI
stays green.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from pipelines.anomaly import anomaly_score, build_detector  # noqa: E402
from pipelines.features import aggregate_window, feature_names  # noqa: E402
from pipelines.metropt3_schema import ANALOG_SENSORS, DIGITAL_SIGNALS  # noqa: E402
from serving.app import create_app  # noqa: E402
from serving.model import ModelBundle, predict_window, score_features  # noqa: E402


def _reading(level: float = 5.0) -> dict:
    """One in-range reading: analog channels at ``level`` (~mid-range), digitals 1."""
    r = {c: level for c in ANALOG_SENSORS}
    r.update({c: 1.0 for c in DIGITAL_SIGNALS})
    return r


def _window(level: float = 5.0, n: int = 6, jitter: float = 0.05) -> list[dict]:
    rng = np.random.default_rng(int(level * 100))
    out = []
    for _ in range(n):
        r = _reading(level)
        for c in ANALOG_SENSORS:
            r[c] += float(rng.normal(0, jitter))
        out.append(r)
    return out


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    feats = feature_names()
    # Fit on many "normal" windows around level 5.0 so the serving feature semantics
    # (aggregate_window) match training.
    rows = [aggregate_window(pd.DataFrame(_window(level=5.0))) for _ in range(120)]
    X = pd.DataFrame(rows)[feats].to_numpy()
    det = build_detector().fit(X)
    threshold = float(np.quantile(anomaly_score(det, X), 0.99))
    return ModelBundle(detector=det, threshold=threshold, features=feats, version="test-v1")


@pytest.fixture()
def client(bundle) -> TestClient:
    return TestClient(create_app(bundle))


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == "test-v1"


def test_health_degraded_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    body = TestClient(create_app(None)).get("/health").json()
    assert body["status"] == "degraded" and body["model_loaded"] is False


def test_predict_happy_path(client):
    resp = client.post("/predict", json={"readings": _window(level=5.0, n=6)})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"anomaly_score", "alert", "threshold", "n_samples", "model_version"}
    assert body["n_samples"] == 6
    assert isinstance(body["alert"], bool)


def test_predict_alerts_on_anomalous_window(client):
    # A window far from the trained-normal level (but still in physical range).
    resp = client.post("/predict", json={"readings": _window(level=10.0, n=6)})
    assert resp.status_code == 200
    assert resp.json()["alert"] is True


def test_out_of_range_rejected(client):
    bad = _reading()
    bad["TP2"] = 999.0  # outside [-1, 12]
    resp = client.post("/predict", json={"readings": [bad]})
    assert resp.status_code == 422
    assert "physical range" in resp.text


def test_non_binary_digital_rejected(client):
    bad = _reading()
    bad["COMP"] = 2.0
    resp = client.post("/predict", json={"readings": [bad]})
    assert resp.status_code == 422
    assert "binary" in resp.text


def test_empty_readings_rejected(client):
    assert client.post("/predict", json={"readings": []}).status_code == 422


def test_missing_field_rejected(client):
    bad = _reading()
    del bad["Motor_current"]
    assert client.post("/predict", json={"readings": [bad]}).status_code == 422


def test_score_features_missing_key_raises(bundle):
    with pytest.raises(KeyError):
        score_features(bundle, {"TP2_mean": 1.0})


def test_predict_window_core(bundle):
    out = predict_window(bundle, pd.DataFrame(_window(level=5.0, n=4)))
    assert out["n_samples"] == 4 and out["model_version"] == "test-v1"
    assert isinstance(out["alert"], bool)


def test_metrics_endpoint_exposes_prometheus(client):
    body = client.get("/metrics").text
    assert "gridsentinel_predictions_total" in body
    assert "gridsentinel_anomaly_score" in body


def test_prediction_increments_metric(client):
    from serving.metrics import PREDICTIONS

    before = sum(
        s.value
        for m in PREDICTIONS.collect()
        for s in m.samples
        if s.name == "gridsentinel_predictions_total"
    )
    client.post("/predict", json={"readings": _window(level=5.0, n=6)})
    after = sum(
        s.value
        for m in PREDICTIONS.collect()
        for s in m.samples
        if s.name == "gridsentinel_predictions_total"
    )
    assert after == before + 1


def test_validation_error_counted(client):
    from serving.metrics import VALIDATION_ERRORS

    before = VALIDATION_ERRORS._value.get()
    bad = _reading()
    bad["TP2"] = 999.0
    client.post("/predict", json={"readings": [bad]})
    assert VALIDATION_ERRORS._value.get() == before + 1
