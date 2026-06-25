# monitoring/

Observability and self-healing (Phase 4).

## In place

- `drift.py` — data-drift detection (PSI + two-sample KS) with an overall verdict.
  Dependency-light and deterministic; this is the trigger logic for retraining.
- `eia_drift.py` — runs that drift check on the **live EIA demand feed**. Verified
  real, naturally-occurring drift: PJM demand, a winter reference vs the current
  week → **PSI 0.46, KS p 0.009, drift detected** (seasonal, not injected).
- Serving-side Prometheus metrics live in `serving/metrics.py` (model + system
  metrics exposed at `/metrics`).

## To come

Prometheus + Grafana dashboards (compose), the delayed-label backfill job, and the
drift → retrain → canary → promote → rollback loop, plus the on-call runbook.
