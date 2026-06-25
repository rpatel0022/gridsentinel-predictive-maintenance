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

drift:  ## Drift on the live EIA feed (needs EIA_API_KEY)
	python -m monitoring.eia_drift --ref-start 2026-01-08T00 --ref-end 2026-01-15T00

docker:  ## Bring up API + Prometheus + Grafana
	docker compose up --build

clean:  ## Remove local model/mlflow artifacts
	rm -rf models mlruns mlflow.db .pytest_cache .ruff_cache
