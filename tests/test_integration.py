"""End-to-end integration test: features → train → registry → serve → predict.

Exercises the whole stack together on synthetic data (no real dataset, no committed
artifact), so a break in the wiring between modules is caught even though each unit
test passes. Skipped unless the full serving/modeling stack is installed.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("joblib")

from fastapi.testclient import TestClient  # noqa: E402
from pipelines.anomaly import anomaly_score, build_detector  # noqa: E402
from pipelines.features import build_windowed_features, feature_columns  # noqa: E402
from pipelines.metropt3_schema import ANALOG_SENSORS, DIGITAL_SIGNALS  # noqa: E402
from serving.app import create_app  # noqa: E402
from serving.model import ModelBundle  # noqa: E402
from serving.registry import PRODUCTION, ModelRegistry  # noqa: E402


def _raw(n_rows: int) -> "pd.DataFrame":
    """Synthetic raw 10s telemetry — normal operation around mid-range."""
    rng = np.random.default_rng(0)
    data = {"timestamp": pd.date_range("2020-02-01", periods=n_rows, freq="10s")}
    for c in ANALOG_SENSORS:
        data[c] = 5.0 + rng.normal(0, 0.05, n_rows)
    for c in DIGITAL_SIGNALS:
        data[c] = 1.0
    return pd.DataFrame(data)


def test_end_to_end_features_train_register_serve(tmp_path):
    # 1. raw telemetry -> windowed features
    feats = build_windowed_features(_raw(6 * 60 * 3), freq="10min")  # ~3h of 10s rows
    cols = feature_columns(feats)
    X = feats[cols].to_numpy()
    assert len(feats) >= 3

    # 2. train the detector + set a label-free threshold
    detector = build_detector(n_estimators=50).fit(X)
    threshold = float(np.quantile(anomaly_score(detector, X), 0.99))
    bundle = ModelBundle(detector, threshold, cols, "it-v1", metrics={"roc_auc": 0.9})

    # 3. register + promote through the registry
    reg = ModelRegistry(str(tmp_path / "registry"))
    reg.register(bundle, "t0")
    reg.promote("it-v1", "t1")
    assert reg.stage_version(PRODUCTION) == "it-v1"

    # 4. serve the production model and score a fresh window of raw readings
    client = TestClient(create_app(reg.load(PRODUCTION)))
    assert client.get("/health").json()["model_version"] == "it-v1"

    reading = {c: 5.0 for c in ANALOG_SENSORS} | {c: 1.0 for c in DIGITAL_SIGNALS}
    resp = client.post("/predict", json={"readings": [reading] * 6})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "it-v1"
    assert body["n_samples"] == 6
    assert isinstance(body["alert"], bool)
    assert "anomaly_score" in body
