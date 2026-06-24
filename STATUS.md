# Project Status & Resume Guide

_Last updated: 2026-06-24. This file lets any new session pick up exactly where we
left off. Start here, then read `PLAN.md`._

## One-line summary
Building **GridSentinel** — a production-grade predictive-maintenance ML system —
as a portfolio project to land the AMETEK Telular ML Engineer role. Foundation +
validated real data + live EIA feed are in, and the **Phase 1 baseline** now
produces a first real ROI number; next step is Phase 2 (calibration, sequence
models, fleet data).

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

## Next steps (in order)
1. **[Phase 2]** Probability **calibration** (the train-tuned threshold doesn't
   transfer for RF / on the small July failure — see results doc); reliability curve.
2. **[Phase 2]** **Sequence models** (LSTM / Temporal-CNN) over the raw stream +
   unsupervised **anomaly detection** scored against the same real failures.
3. **[Phase 2]** Layer in **Backblaze** Drive Stats — fleet scale + many failures
   (fixes the "only 2 scorable folds" limitation), register best model in MLflow.

## How to resume
- Branch: `claude/refine-plan-md-6swc1n` (this is also PR #1).
- Setup: `pip install -e ".[dev,pipelines]"` then `pytest` (should be all green).
- EIA key: obtained — add as `EIA_API_KEY` in the environment.
