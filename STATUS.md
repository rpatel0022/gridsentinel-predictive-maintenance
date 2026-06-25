# Project Status & Resume Guide

_Last updated: 2026-06-24. This file lets any new session pick up exactly where we
left off. Start here, then read `PLAN.md`._

## One-line summary
Building **GridSentinel** — a production-grade predictive-maintenance ML system —
as a portfolio project to land the AMETEK Telular ML Engineer role. Foundation +
validated real data + live EIA feed + **Phase 1 supervised baseline** + **Phase 2
anomaly detector** + **Phase 3 FastAPI serving service (Dockerized)** are in; next
is the CI metric gate + registry rollback, then observability.

## Decisions locked (see `docs/adr/0001-dataset-feed-and-cloud.md`)
- **Dataset:** MetroPT-3 (primary) + Backblaze (fleet-scale companion, Phase 2)
- **Live feed:** EIA Open Data API v2 (hourly US electricity demand)
- **Cloud:** AWS (ECS/Fargate + S3)

## Done so far ✅
- Repo scaffold (`src/ pipelines/ serving/ monitoring/ infra/ tests/ docs/`)
- **Cost model** (`src/gridsentinel/cost.py`) — expected-$ cost, cost-optimal
  decision threshold, naive + fixed-schedule baselines. The ROI core.
- **MetroPT-3 validation contract** (`pipelines/metropt3_schema.py`) — pandera
  schema, 7 analog + 8 binary signals (matches the dataset's "15 features"),
  timestamp-ordering / leakage guard. **Physical ranges tightened** from TBC to
  spike-derived bounds — passes on all 1,516,948 real rows.
- **Data-quality spike** (`pipelines/data_quality.py` + `docs/data_quality_metropt3.md`)
  — reproducible profiler; provenance for the schema bounds.
- **EIA connector** (`pipelines/connectors.py`) — request builder + IO, key from
  env. **Verified live** (24 hourly PJM demand rows from API v2).
- **CI** green (lint + format + 32 tests). `docs/architecture.md` + ADR + README.

## BLOCKER — RESOLVED ✅
Network policy was widened; `archive.ics.uci.edu`, `zenodo.org`, and `api.eia.gov`
are all reachable now. MetroPT-3 was downloaded (218 MB, UCI #791) and validated;
the EIA feed was smoke-tested with a live key. Nothing currently blocked.

## Done this session ✅
1. Pulled MetroPT-3, validated against `pipelines/metropt3_schema.py` — clean
   (0 nulls, monotonic timestamps, digitals all `{0,1}`); 15-feature schema matched
   the real data exactly, so ADR-0001 stands unchanged (revisit trigger not hit).
2. Ran the data-quality spike and tightened the TBC analog ranges to physical
   bounds (see `docs/data_quality_metropt3.md`).
3. Smoke-tested `EIAConnector.fetch_demand()` live — 24 hourly rows returned.

## Phase 1 done ✅ (this session)
- Windowed feature pipeline (`pipelines/features.py`), real failure labels from the
  report table (`pipelines/labels.py`), temporal CV with embargo (`gridsentinel/cv.py`).
- RF + XGBoost baselines, cost-tuned threshold (train-tuned, frozen for test),
  MLflow tracking (`pipelines/train_baseline.py`).
- **First real ROI:** cost-tuned XGBoost leak detector cuts expected cost ~60% vs
  the best fixed schedule on the held-out June failure (ROC-AUC 0.92; ~30% mean over
  2 scorable folds). Full results + honest caveats: `docs/phase1_baseline_results.md`.

## Phase 2 in progress 🔵 (this session)
- **Unsupervised anomaly detection** (`pipelines/anomaly.py`): Isolation Forest fit
  on the failure-free baseline period (zero failure labels), evaluated vs the 4 real
  failures. **ROC-AUC 0.95, recall 0.89** at a label-free threshold, and **19–48 h
  early warning** on 3 of 4 failures — the lead time supervised prediction couldn't
  find. Results: `docs/phase2_anomaly_results.md`.
- **Calibration was tested and rejected:** isotonic/sigmoid calibration +
  validation-slice thresholding all *worsened* ROI (precision collapse); the real
  bottleneck is the 4-failure scarcity, not calibration. (See the results doc.)

## Phase 3 in progress 🔵 (this session)
- **FastAPI inference service** (`serving/`): `POST /predict` (window of raw
  readings → anomaly score + alert), `/health`, `/`. Pydantic validates every
  reading against the Phase 0 physical contract → bad telemetry rejected with 422.
- **Model bundle** (`serving/model.py`): pipeline + threshold + feature order +
  provenance; save/load; framework-free scoring core. Train/serve features share
  one aggregation (`pipelines/features.aggregate_window`) — no feature skew.
- **Docker**: `Dockerfile` (non-root, healthcheck) + `docker-compose.yml`.
- **CI/CD metric gate** (`pipelines/metric_gate.py` + `.github/workflows/model-eval.yml`):
  rebuilds the model on real data and fails the build if ROC-AUC/PR-AUC/recall drop
  below committed floors (gate logic unit-tested in lean CI; passes on real data at
  0.95/0.39/0.89 vs floors 0.90/0.30/0.75). Adds **pip-audit** dependency scan (CI)
  + **Trivy** image scan (model-eval).

## Phase 4 in progress 🔵 (autonomous loop)
- **Prometheus instrumentation** (`serving/metrics.py` + `/metrics` endpoint): two-
  tier observability — *model* metrics (prediction/alert counts, anomaly-score
  distribution = the leading indicators, since labels lag) + *system* metrics
  (per-endpoint latency, validation-error count). Verified live on real data.

## Next steps (in order)
1. **[Phase 4]** **Evidently** drift report on the live EIA feed (feature/prediction
   drift) → a drift signal that can trigger retraining.
2. **[Phase 4]** **Grafana** dashboards + Prometheus scrape in docker-compose.
3. **[Phase 4]** Drift → **automated retrain → canary/promote**, alert thresholds +
   runbook.
4. **[Phase 3]** MLflow **registry stages + rollback/canary** wiring.
5. **[later]** Sequence models (LSTM/TCN) + Backblaze fleet data for scale.

## How to resume
- Branch: `claude/refine-plan-md-6swc1n` (this is also PR #1).
- Setup: `pip install -e ".[dev,pipelines]"` then `pytest` (should be all green).
- EIA key: obtained — add as `EIA_API_KEY` in the environment.
