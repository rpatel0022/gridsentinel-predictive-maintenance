# GridSentinel — developer task runner. `make help` lists targets.
# DATA points at the real MetroPT-3 CSV (fetched, never committed).
DATA ?= MetroPT3(AirCompressor).csv

.PHONY: help install test lint fmt quality data-quality train anomaly gate \
        drift artifact serve loadtest edge docker clean

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install all extras (dev, pipelines, modeling, serving)
	pip install -e ".[dev,pipelines,modeling,serving]"

test:  ## Run the test suite
	pytest -q

lint:  ## Lint + format check (matches CI)
	ruff check . && ruff format --check .

fmt:  ## Auto-format
	ruff format .

quality: lint test  ## Everything CI's quality job runs

data-quality:  ## Validate + profile the real dataset
	python -m pipelines.data_quality "$(DATA)"

train:  ## Train the RF/XGBoost baselines (MLflow)
	python -m pipelines.train_baseline "$(DATA)"

anomaly:  ## Train + evaluate the anomaly detector (MLflow)
	python -m pipelines.anomaly "$(DATA)"

lstm:  ## Train + evaluate the LSTM sequence model (needs the dl extra)
	python -m pipelines.lstm_model "$(DATA)"

backblaze-data:  ## Fetch the Backblaze fleet lifetime CSV (from a GitHub mirror)
	mkdir -p data && curl -sSL -o data/backblaze_drive_dates.csv \
	  https://raw.githubusercontent.com/zachmayer/backblaze_analysis/master/results/drive_dates.csv

backblaze: backblaze-data  ## Fleet-reliability model on 418k drives / 24k failures
	python -m pipelines.backblaze data/backblaze_drive_dates.csv

gate:  ## Run the metric gate on the real data
	python -m pipelines.metric_gate --csv "$(DATA)"

artifact:  ## Build the serving model bundle -> models/
	python -m serving.build_artifact "$(DATA)"

edge:  ## Edge size/latency/accuracy benchmark
	python -m serving.benchmark "$(DATA)"

serve: artifact  ## Build the artifact and run the API locally
	uvicorn serving.app:app --reload

loadtest:  ## Load-test the serving API (needs models/ built)
	python -m serving.load_test

status:  ## Operational status: live model, thresholds, audit trail
	python -m serving.status

dashboard-assets: backblaze-data  ## Precompute small dashboard assets from real data
	python -m reports.precompute --metropt "$(DATA)" --backblaze data/backblaze_drive_dates.csv

dashboard:  ## Launch the interactive Streamlit results dashboard
	streamlit run reports/app.py

dashboard-static:  ## Build the static PNG + interactive HTML dashboards
	python -m reports.dashboard --metropt "$(DATA)" --backblaze data/backblaze_drive_dates.csv --format both --out docs/dashboard.png

selfheal:  ## Run one retrain -> gate -> promote/keep cycle
	python -m monitoring.self_heal "$(DATA)"

retrain-if-drift:  ## Retrain only if the live feed has drifted (needs EIA_API_KEY)
	python -m monitoring.drift_trigger "$(DATA)" --ref-start 2026-01-08T00 --ref-end 2026-01-15T00

drift:  ## Drift on the live EIA feed (needs EIA_API_KEY)
	python -m monitoring.eia_drift --ref-start 2026-01-08T00 --ref-end 2026-01-15T00

docker:  ## Bring up API + Prometheus + Grafana
	docker compose up --build

clean:  ## Remove local model/mlflow artifacts
	rm -rf models mlruns mlflow.db .pytest_cache .ruff_cache
