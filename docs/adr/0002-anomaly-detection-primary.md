# ADR 0002 — Unsupervised anomaly detection as the primary model

- **Status:** Accepted
- **Date:** 2026-06-25
- **Deciders:** Rushi Patel

## Context

MetroPT-3 ships **four** real failures (air leaks). Phase 1 built supervised
RF/XGBoost baselines under strict temporal CV and hit the wall that few failures
forces: only 2 of 5 folds are scorable, and *predict-ahead* (warn 24 h before a
failure) collapsed to ROC-AUC ≈ 0.4 with negative ROI — telemetry does not forecast
an air leak a day out across so few events. The detection framing (active failure +
a short lead-in) did carry signal (XGBoost ROC-AUC 0.92), but the operating-point
recall didn't transfer well to the smallest failure.

I hypothesised **probability calibration** would fix the threshold transfer and
tested it directly.

## Decision

Make the **unsupervised Isolation-Forest anomaly detector the primary model**,
trained on the verified failure-free period and evaluated against the real
failures. Keep the supervised detector as a complementary, precise signal. Use the
**detection** label framing (not predict-ahead) where labels are needed.

## Rationale

- Anomaly detection needs **zero failure labels to train**, so it is immune to the
  four-failure scarcity that caps the supervised approach.
- On the real failures it beats the baseline *and* delivers what Phase 1 couldn't:
  ROC-AUC **0.95**, recall **0.89** at a label-free threshold, and **19–48 h early
  warning** on three of four failures.
- It produces the leading-indicator signal the Phase 4 drift/health monitor needs.

## Alternatives considered

- **Probability calibration (isotonic/sigmoid) + validation-slice thresholds** —
  tested and **rejected**: every variant *worsened* ROI (precision collapsed to
  ~0.05), and carving a calibration slice steals one of the four failures from
  training. The bottleneck is failure count, not calibration.
- **Supervised model as primary** — strong PR-AUC on the one large failure, but
  brittle across folds and no early warning; kept as a complement, not the lead.
- **Sequence models (LSTM/TCN)** — deferred: same label scarcity, heavier.

## Consequences

- The production/monitoring layer keys off the anomaly score (no labels required).
- Reported metrics are honest about the small evaluation (2 scorable folds; one
  fast failure with short lead time).
- **Revisit trigger:** once Backblaze fleet data lands (many failures across many
  devices), re-evaluate whether a supervised or sequence model should lead.
