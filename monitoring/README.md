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
- `prometheus/prometheus.yml` + `grafana/` — the full local stack. `docker compose
  up` brings up the API, Prometheus (scrapes `/metrics`), and Grafana with the
  **auto-provisioned "GridSentinel" dashboard** (Grafana :3000, Prometheus :9090).
  The dashboard shows both tiers: prediction/alert rate, anomaly-score p50/p99
  (model leading indicators), request-latency p99, and validation-error rate.

## To come

The delayed-label backfill job and the drift → retrain → canary → promote →
rollback loop, plus the on-call runbook.
