# Project Status & Resume Guide

_Last updated: 2026-06-24. This file lets any new session pick up exactly where we
left off. Start here, then read `PLAN.md`._

> **Session paused (user's 5-hour window).** Phases 0–5 built & tested end-to-end
> (117 tests, CI-green). `make help` lists one-command workflows. Remaining items are
> environment-blocked (Backblaze data access; PyTorch/DL framework) or optional.

## One-line summary
**GridSentinel** — a production-grade predictive-maintenance ML system, built as a
portfolio project to land the AMETEK Telular ML Engineer role. Phases 0–5 (data
validation → supervised + anomaly + neural models → cost-tuned serving → CI gate →
observability → self-healing retrain/rollback → edge/load benchmarks) are built and
tested end-to-end; remaining items (Backblaze fleet data, real LSTM/TCN) are blocked
by the sandbox environment, not by design.

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
- **Drift detection** (`monitoring/drift.py`): PSI + two-sample KS, overall verdict.
  Dependency-light, deterministic, tested. `monitoring/eia_drift.py` runs it on the
  **live EIA feed** — verified real seasonal drift (PJM winter vs now: PSI 0.46,
  KS p 0.009, drift detected). This is the retrain-trigger signal.
- **Grafana + Prometheus stack** (`monitoring/prometheus/`, `monitoring/grafana/`,
  `docker-compose.yml`): `docker compose up` → API + Prometheus (scrapes `/metrics`)
  + Grafana with an auto-provisioned two-tier dashboard. Config validated by tests.
- **Model registry** (`serving/registry.py`): file-based stages (production/
  candidate), **promote + rollback**, and an **audit trail** (governance). Pure
  stage logic unit-tested in lean CI + a joblib round-trip. (Covers the Phase 3
  registry/rollback item; the self-heal loop promotes through it.)
- **Self-heal orchestrator** (`monitoring/self_heal.py`): drift → retrain candidate
  → metric-gate → promote **only if** it clears the gate and doesn't regress vs
  production, else keep current. Pure decision unit-tested; **demoed end-to-end on
  real data** (register → promote → audit). On-call **runbook** at `docs/runbook.md`
  (alert thresholds + retrain/rollback procedures). *Phase 4 self-healing loop done.*

## Phase 5 / docs in progress 🔵 (autonomous loop)
- **ADRs** for the real forks: `0002` anomaly-as-primary (+ calibration rejected),
  `0003` serving + file-registry stack, `0004` in-house PSI/KS drift.
- **Model card** (`docs/model_card.md`) — intended use, real metrics, honest limits,
  governance. **Google ML Test Score** self-assessment (`docs/ml_test_score.md`):
  **4.5** (min of the four sections), every point backed by a named artifact.
- README now has an **artifacts index** linking results/ADRs/governance docs.

- **Edge benchmark** (`serving/benchmark.py` + `docs/edge_benchmark.md`): measured
  size/latency/accuracy across ensemble sizes on real data. 300→50 trees = **5.9×
  smaller, ~4× faster p99 (24→6 ms), same ROC-AUC (~0.95)**, recall cost 0.89→0.75.
  Cloud-vs-edge recommendation included.

- **Delayed-label backfill** (`monitoring/backfill.py`): once real failure reports
  arrive, re-scores the stored predictions for that window against the now-known
  labels and logs true precision/recall/ROC-AUC (JSONL history). Honest handling of
  lagging labels — pure core unit-tested.

- **AWS deploy notes** (`infra/aws/`): Fargate `task-definition.json` (port 8000,
  `/health` check, `EIA_API_KEY` via SSM secret, CloudWatch logs) + a deploy guide
  (ECR/ECS/S3 model bundle, never-in-git) + a **cost note** (~$35–40/mo, ~$18 scaled
  down). Validated by tests (structure + no-leaked-secret). Documented target, not
  applied from the repo.

- **End-to-end integration test** (`tests/test_integration.py`): features → train →
  registry → serve → predict on synthetic data — catches cross-module wiring breaks
  that unit tests miss. Closes the ML Test Score infra gap (3.3 → infra 6.0; final
  still 4.5, bounded by the Data/Model sections).

- **Load test** (`serving/load_test.py` + `docs/load_test_results.md`): per-request
  **p99 31 ms (< 50 ms SLO)**. Found + fixed `n_jobs=-1` thread oversubscription, and
  documented the GIL-bound ~40 rps/process → horizontal-scaling capacity model.

- **Architecture doc refreshed** (`docs/architecture.md`) to the **built** system
  (✅/◑/○ markers) — accurate models, in-house drift, file registry, edge benchmark,
  backfill, CI gate/SLO.

- **Neural sequence baseline** (`pipelines/sequence_model.py`): MLP over stacked
  temporal windows (1h context) — **ROC-AUC 0.93, PR-AUC 0.51, ROI 32%**, competitive
  with XGBoost (best supervised PR-AUC) but not beating the anomaly detector. True
  LSTM/TCN deferred — the PyTorch CPU wheel index is proxy-blocked in-sandbox.
  ([results](docs/sequence_model_results.md))
- **Model-behavioral tests** (`tests/test_model_behavior.py`): CheckList-style —
  minimum-functionality, directional-expectation, order-invariance, noise-stability.
  (Writing them surfaced that Isolation-Forest scores aren't monotone in the tail —
  encoded honestly rather than asserting a false expectation.)

## Next steps (blocked / follow-on — need a different environment)
1. **[data]** Backblaze fleet dataset — download is 403/JS-gated in-sandbox; deferred
   (not faked — the 100%-real-data rule). Needs data access.
2. **[ML]** True LSTM/TCN — PyTorch CPU index proxy-blocked; needs a DL framework.
3. **[polish]** Per-stage thresholds in the registry; optional Phase 6 GenAI/RAG.

> **Core system (Phases 0–5) is complete and tested (113 tests).** The two biggest
> remaining items (Backblaze, real LSTM) are both blocked by the sandbox environment
> (data access / DL framework), not by design — they're ready to resume elsewhere.

> Phases 0–5 are built and tested end-to-end (108 tests). Remaining items are larger
> follow-on efforts (Backblaze needs data access; LSTM is label-limited), not quick
> wins — the core system is complete.

## How to resume
- Branch: `claude/refine-plan-md-6swc1n` (this is also PR #1).
- Setup: `pip install -e ".[dev,pipelines]"` then `pytest` (should be all green).
- EIA key: obtained — add as `EIA_API_KEY` in the environment.
