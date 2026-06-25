# GridSentinel — on-call runbook

One page: what alerts fire, the threshold, and what to do. Metrics come from the
service's `/metrics` (Prometheus → Grafana); drift comes from `monitoring/drift.py`.

## Alerts & thresholds

| Alert | Signal | Threshold | First response |
|---|---|---|---|
| **Data drift** | PSI / KS on the live feed (`monitoring/eia_drift.py`) | PSI > 0.2 **or** KS p < 0.05 | Confirm it's real (weather/season vs. a feed change), then trigger a retrain cycle (below). |
| **Anomaly-alert surge** | `gridsentinel_predictions_total{alert="true"}` rate | > 3× the trailing-24h baseline | Check the fleet — a real fault wave vs. drift. Cross-check the anomaly-score p99 panel. |
| **Latency SLO breach** | `gridsentinel_request_latency_seconds` p99 | p99 > 0.5 s for 5 min | Scale the API / check the model size; if a recent promotion caused it, **roll back**. |
| **Bad-telemetry surge** | `gridsentinel_validation_errors_total` rate | > 1% of requests | Upstream/device problem — the model is fine; chase the sender. Schema is doing its job. |
| **Model regression** | candidate fails the metric gate in a retrain cycle | any gate floor missed | Auto-loop keeps the current model (no promotion). Page to investigate the data. |

## Why "leading indicators", not accuracy

True failure labels lag by weeks, so we **cannot** alert on live accuracy. We alert
on what moves *first* — input/prediction drift and the anomaly-score distribution —
and backfill true performance when labels arrive (the delayed-label backfill job).

## Retrain → promote (self-healing)

The loop is automated (`monitoring/self_heal.run`) but runnable by hand:

```bash
python -c "from monitoring.self_heal import run; \
  print(run('MetroPT3(AirCompressor).csv', 'models/registry', 'manual-$(date -u +%FT%TZ)'))"
```

It retrains a **candidate**, runs the metric gate, and promotes it **only if** it
clears the gate and doesn't regress on ROC-AUC vs the current production model.
Every transition is in the registry audit log.

## Rollback (a bad model reached production)

```python
from serving.registry import ModelRegistry
ModelRegistry("models/registry").rollback("incident-<id>")  # restores the prior production version
```

Then confirm `/health` reports the restored `model_version` and the latency/alert
panels recover. Record the incident in the audit trail (the rollback is logged).

## Check live state

```bash
python -m serving.status              # production model, its metrics, threshold, recent audit
```

To tune an alert threshold without a redeploy (e.g. raise recall on a noisy feed),
set a per-stage override — it's audit-logged and applied on next model load:

```python
from serving.registry import ModelRegistry
ModelRegistry("models/registry").set_threshold("production", 0.012, "incident-<id>")
```

## Escalation

If drift is real *and* a fresh retrain can't clear the gate, the data has shifted
beyond the current feature set — escalate to model owners (consider new features /
the fleet-scale dataset), keep the last-good model serving, and watch the
delayed-label backfill for true impact.
