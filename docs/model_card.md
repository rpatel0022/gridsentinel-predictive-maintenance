# Model card — GridSentinel anomaly detector

## Model details

- **Model:** Isolation Forest (300 trees) over 36 windowed features (per-analog
  mean/std/min/max + per-digital duty fraction), standardized. `serving/model.py`.
- **Version scheme:** `iforest-<UTC timestamp>`; the live version is in the registry
  `production` stage and surfaced at `/health`.
- **Date:** 2026-06. Built as a portfolio system.
- **Companion model:** a supervised XGBoost detector (`pipelines/train_baseline.py`)
  kept for precise, labelled-signature detection.

## Intended use

- **Use:** flag developing/active anomalies (e.g. air leaks) in compressor / UPS-like
  IoT telemetry as **decision support** for maintenance dispatch — an early warning,
  not an autonomous shutdown.
- **Out of scope:** safety-critical automated control; domains unlike compressed-air /
  power equipment; any use where a missed or false alert isn't human-reviewed.

## Training & evaluation data

- **Source:** MetroPT-3 (UCI #791) — real Porto-metro compressor telemetry, 100% real,
  never synthetic. Failure labels from the company's maintenance reports (4 air leaks).
- **Train (unsupervised):** the 9,386 windows *before the first reported failure*
  (verified failure-free) — no labels used to fit.
- **Eval:** the 16,098 later windows containing all four real failures.

## Metrics (strict temporal evaluation)

| Metric | Anomaly detector | Supervised XGBoost |
|---|---|---|
| ROC-AUC | 0.95 | 0.92 |
| PR-AUC | 0.39 | 0.44 |
| Recall @ operating point | 0.89 | 0.33 |
| Precision @ operating point | 0.40 | 0.38 |

**Early warning** (anomaly detector): first alert 19.5 h / 0.2 h / 47.8 h / 47.8 h
before the four failures. Operating threshold = 99th percentile of anomaly score over
healthy operation (label-free).

## Limitations & honest caveats

- **Only four real failures** → small evaluation; the supervised model has just 2
  scorable temporal folds. Treat absolute numbers as indicative.
- **One fast failure (2020-05-29)** gives only ~0.2 h lead — short-onset events warn
  late.
- **Precision ~0.40** at high recall: expect false alarms; pair with the asymmetric
  cost model (a false alarm ≪ a missed failure) and human review.
- **No reliable many-hours-ahead *prediction*** — value is in detecting the developing
  anomaly, not forecasting a day out (see `docs/phase1_baseline_results.md`).
- **Single device, single domain.** Cross-device/fleet generalization is unproven
  until the Backblaze fleet data lands.
- **Calibration** was tested and didn't help at this data scale (ADR-0002).

## Ethical / operational considerations

- **No PII:** industrial sensor signals only.
- **Delayed ground truth:** true failure labels lag weeks; the system alerts on
  leading indicators now and backfills true performance later (`docs/runbook.md`).
- **Governance:** every model promotion/rollback is recorded in the registry audit
  trail; secrets (EIA key) are env-only, never committed.

## Maintenance

Retrains via the self-healing loop (`monitoring/self_heal.py`): a candidate is
promoted only if it clears the metric gate and doesn't regress vs production; drift on
the live feed (`monitoring/drift.py`) is the trigger. Rollback is one registry call.
