# Project Status & Resume Guide

_Last updated: 2026-06-24. This file lets any new session pick up exactly where we
left off. Start here, then read `PLAN.md`._

## One-line summary
Building **GridSentinel** — a production-grade predictive-maintenance ML system —
as a portfolio project to land the AMETEK Telular ML Engineer role. Foundation is
built and tested; next step is pulling the real dataset and training baselines.

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
  timestamp-ordering / leakage guard. **Physical ranges still marked TBC.**
- **EIA connector** (`pipelines/connectors.py`) — request builder + IO, key from env.
- **CI** green (lint + format + 20 tests). `docs/architecture.md` + ADR + README.

## BLOCKER 🚧 — why we paused
This environment's **network policy blocks** `archive.ics.uci.edu`, `zenodo.org`,
and `api.eia.gov` (proxy returns 403). So the real data can't be downloaded from
the current session. **To unblock:** widen the environment's network policy to
allow those 3 hosts, then start a NEW session.

## Next steps (in order)
1. **[needs network]** Download MetroPT-3, validate against
   `pipelines/metropt3_schema.py`, tighten the TBC ranges, update ADR-0001 if the
   real schema differs.
2. **[needs network + EIA key]** Smoke-test `EIAConnector.fetch_demand()` live.
   Set the key as env var `EIA_API_KEY` (already obtained).
3. **[offline OK]** Phase 1 modeling: temporal/grouped CV splitter, RF/XGBoost
   baselines, MLflow tracking, cost-tuned threshold vs the schedule baseline →
   produces the first real **ROI %** to replace `~X%` in the README.

## How to resume
- Branch: `claude/refine-plan-md-6swc1n` (this is also PR #1).
- Setup: `pip install -e ".[dev,pipelines]"` then `pytest` (should be all green).
- EIA key: obtained — add as `EIA_API_KEY` in the environment.
