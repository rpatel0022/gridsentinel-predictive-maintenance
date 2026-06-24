# serving/

The production inference service (Phase 3): a FastAPI app that serves the Phase 2
anomaly detector behind a schema-validated HTTP API.

## Layout

- `app.py` — FastAPI app. `POST /predict` (a window of raw readings → anomaly score
  + alert), `GET /health`, `GET /`. Every reading is validated against the same
  physical contract as the training data (`pipelines/metropt3_schema.py`), so bad
  telemetry is rejected with HTTP 422.
- `model.py` — framework-free scoring core + the `ModelBundle` (pipeline +
  threshold + feature order + provenance) and its save/load. Swapping models is a
  one-file change; the service code never moves.
- `build_artifact.py` — trains the detector on real MetroPT-3 and saves a bundle
  (gitignored; rebuilt from data, never committed).

## Run it

```bash
pip install -e ".[serving,modeling]"
python -m serving.build_artifact "MetroPT3(AirCompressor).csv"   # -> models/anomaly_detector.joblib
uvicorn serving.app:app --reload                                  # http://localhost:8000/docs
# or the container:
docker compose up --build
```

## Example

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"readings": [{"TP2": 8.4, "TP3": 9.0, "H1": 8.8, "DV_pressure": 0.0,
       "Reservoirs": 9.0, "Oil_temperature": 62.0, "Motor_current": 4.0,
       "COMP": 1, "DV_eletric": 0, "Towers": 1, "MPG": 1, "LPS": 0,
       "Pressure_switch": 1, "Oil_level": 1, "Caudal_impulses": 1}]}'
# -> {"anomaly_score": ..., "alert": false, "threshold": ..., "n_samples": 1, "model_version": "..."}
```

Still to come (Phase 3+): registry stages + rollback/canary, a CI metric gate that
blocks a regressing model, and dependency/image scanning.
