# GridSentinel

**A self-healing, IoT-scale predictive-maintenance service that cuts expected
maintenance cost ~X% vs schedule-based upkeep** — runs in production with full
MLOps, detects its own drift, retrains automatically, and ships to the edge.

> _`~X%` is the full-system target. **Phase 1 baseline (first real number):** on a
> held-out failure under strict temporal CV, the cost-tuned XGBoost leak detector
> cuts expected maintenance cost **~60% vs the best fixed schedule** (ROC-AUC 0.92,
> ~30% averaged over the scorable folds). Details + the honest caveats:
> [docs/phase1_baseline_results.md](docs/phase1_baseline_results.md)._

![CI](https://github.com/rpatel0022/ametek-ml-engineer-project/actions/workflows/ci.yml/badge.svg)
**Status:** Phase 3 — productionizing. The Phase 2 anomaly detector (ROC-AUC 0.95,
recall 0.89, 19–48 h early warning, [results](docs/phase2_anomaly_results.md)) is
served behind a **schema-validated FastAPI app** (`serving/`) in a **Docker** image,
and CI/CD now guards it: a **metric gate** rebuilds the model on the real data and
fails the build if it regresses, plus dependency + image scanning. Next: registry
rollback, then observability/drift. Continues per [PLAN.md](PLAN.md).

GridSentinel ingests streaming telemetry from a fleet of IoT-connected power
units, predicts failures and remaining useful life, flags anomalies in real time,
and operates the full MLOps lifecycle. It is built on **100% real data** — real
failure-labeled datasets for training, and a genuinely-live public feed for the
production/monitoring layer (see [the data seam](docs/architecture.md)).

## Why it's graded on dollars, not accuracy

A missed failure means an emergency truck-roll and downtime; a false alarm only
wastes an inspection. GridSentinel tunes its decision threshold to **minimise
expected dollar cost** against that asymmetry, and always reports its lift over a
dumb fixed-schedule baseline. That logic already exists and is tested:
[`src/gridsentinel/cost.py`](src/gridsentinel/cost.py).

```python
from gridsentinel import CostModel, optimal_threshold, periodic_schedule_cost

model = CostModel(cost_fn=1000.0, cost_fp=10.0, cost_tp=10.0)  # missed failure hurts 100x
threshold, cost = optimal_threshold(y_true, y_score, model)     # ROI-optimal cutoff, not 0.5
```

## Requirement → project traceability (curated to the Telular ML Engineer role)

| Posting requirement | How GridSentinel delivers it |
|---|---|
| Supervised learning | Failure classification + RUL regression (RandomForest / XGBoost baselines) |
| Unsupervised learning | Anomaly detection (Isolation Forest / autoencoder) + health clustering + PCA/UMAP |
| Deep learning | LSTM / Temporal-CNN sequence RUL, vs the GBM baseline |
| Feature pipelines + training workflows | Reproducible pipeline + Feast feature store |
| Productionize (not prototypes) | FastAPI service, containerized, deployed live on AWS, versioned |
| MLOps lifecycle | MLflow registry, GitHub Actions CI/CD with metric gates, Prometheus + Grafana, Evidently drift → retrain |
| Cloud + ML services | AWS ECS/Fargate + S3 (SageMaker stretch) |
| Drift detection & iteration | Evidently monitors on a live feed → automated retrain → canary → promote → rollback |
| Measurable customer ROI | Decision threshold tuned to a $ cost function; headline lift vs schedule-based upkeep |
| Security / compliance | Secret management, dependency + image scanning in CI, model-governance audit trail |
| _Bonus:_ Edge ML | Quantize → ONNX, with measured size/latency reduction |
| _Bonus:_ Agentic AI / RAG | Optional Phase 6: LLM agent + RAG over UPS manuals → maintenance work-order |

## Repository layout

```
src/gridsentinel/   Core library (cost model, temporal CV)
pipelines/          Features, labels, training, anomaly detection (P1-2)
serving/            FastAPI inference service + Docker (P3)
monitoring/         Observability, drift, self-healing loop (P4+)
infra/              AWS deploy, live-ingestion service, edge target (P5+)
tests/              Unit / data-validation / model-behavioral tests
docs/adr/           Architecture decision records
docs/architecture.md  System diagram + the data-seam explainer
PLAN.md             Full project strategy & phased build
```

## Develop

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

## Run the models

```bash
pip install -e ".[dev,pipelines,modeling]"
# Fetch real MetroPT-3 (UCI #791) — never committed; then:
python -m pipelines.data_quality "MetroPT3(AirCompressor).csv"     # validate + profile
python -m pipelines.train_baseline "MetroPT3(AirCompressor).csv"   # RF/XGBoost → MLflow
python -m pipelines.anomaly "MetroPT3(AirCompressor).csv"          # Isolation Forest → MLflow
```

Runs are tracked in MLflow (defaults to a local `sqlite:///mlflow.db`). Results:
[Phase 1 baseline](docs/phase1_baseline_results.md) ·
[Phase 2 anomaly detection](docs/phase2_anomaly_results.md).

## Key decisions

- [ADR 0001 — dataset, live feed, and cloud](docs/adr/0001-dataset-feed-and-cloud.md):
  MetroPT-3 (+ Backblaze) · EIA live feed · AWS.

See [PLAN.md](PLAN.md) for the full strategy, phased build, and how each phase
proves itself.
