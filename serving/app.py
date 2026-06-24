"""FastAPI inference service for GridSentinel.

Exposes the Phase 2 anomaly detector behind a schema-validated HTTP API. The
request body is a window of raw sensor readings; pydantic validates every reading
against the *same* physical contract the training data is held to
(`pipelines.metropt3_schema`), so malformed or out-of-range telemetry is rejected at
the door (HTTP 422) rather than silently scored. The model bundle carries its own
threshold and feature order, so swapping models never touches this file.

Run locally::

    MODEL_PATH=models/anomaly_detector.joblib uvicorn serving.app:app --reload
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pipelines.metropt3_schema import ANALOG_RANGES, ANALOG_SENSORS, DIGITAL_SIGNALS
from pydantic import BaseModel, Field, model_validator

from serving.model import ModelBundle, load_bundle, predict_window


class SensorReading(BaseModel):
    """One raw sample: the 7 analog + 8 digital signals, range-validated."""

    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float
    COMP: float
    DV_eletric: float
    Towers: float
    MPG: float
    LPS: float
    Pressure_switch: float
    Oil_level: float
    Caudal_impulses: float

    @model_validator(mode="after")
    def _within_contract(self) -> SensorReading:
        for name in ANALOG_SENSORS:
            low, high = ANALOG_RANGES[name]
            value = getattr(self, name)
            if not (low <= value <= high):
                raise ValueError(f"{name}={value} outside physical range [{low}, {high}]")
        for name in DIGITAL_SIGNALS:
            value = getattr(self, name)
            if value not in (0.0, 1.0):
                raise ValueError(f"{name}={value} must be binary {{0, 1}}")
        return self


class PredictRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, description="one window of samples")


class PredictResponse(BaseModel):
    anomaly_score: float
    alert: bool
    threshold: float
    n_samples: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


def create_app(bundle: ModelBundle | None = None) -> FastAPI:
    """Build the app. A bundle may be injected (tests); otherwise it loads lazily
    from ``MODEL_PATH`` so the service starts even before an artifact exists."""
    app = FastAPI(title="GridSentinel", version="0.1.0")
    app.state.bundle = bundle

    def get_bundle() -> ModelBundle:
        if app.state.bundle is None:
            try:
                app.state.bundle = load_bundle()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return app.state.bundle

    @app.get("/")
    def root() -> dict:
        return {"service": "GridSentinel", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        bundle = app.state.bundle
        if bundle is None:
            try:
                bundle = load_bundle()
                app.state.bundle = bundle
            except FileNotFoundError:
                return HealthResponse(status="degraded", model_loaded=False)
        return HealthResponse(status="ok", model_loaded=True, model_version=bundle.version)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        bundle = get_bundle()
        readings = pd.DataFrame([r.model_dump() for r in req.readings])
        result = predict_window(bundle, readings)
        return PredictResponse(**result)

    return app


app = create_app()
