# Phase 2 — unsupervised anomaly detection (results)

_Reproduce: `python -m pipelines.anomaly "MetroPT3(AirCompressor).csv"`. Tracked in
MLflow under `gridsentinel-phase2-anomaly`._

## Why anomaly detection (and why calibration was the wrong fix first)

Phase 1 flagged a threshold-transfer gap and I hypothesised **probability
calibration** would fix it. I tested that directly — isotonic and sigmoid
calibration, plus validation-slice thresholding — and **all of them made ROI
worse** (recall rose but precision collapsed to ~0.05; carving out a calibration
slice also stole one of the four failures from training, leaving only one scorable
fold). The real bottleneck isn't calibration, it's **having only four failures**.

So the right move is a method that doesn't depend on failure labels at all.
**Isolation Forest** learns *normal* compressor behaviour (abundant) and flags
deviation — it needs **zero failure labels to train**.

## Setup

- Same 10-min windowed features as Phase 1 (`pipelines/features.py`).
- **Train (unsupervised):** the 9,386 windows *before the first reported failure*
  (Feb 1 → Apr 18) — a verified failure-free baseline. The report table is used only
  to locate this clean period, never to label training rows.
- **Evaluate:** the 16,098 later windows (Apr–Sep) containing all four failures.
- **Operating threshold:** the 99th percentile of anomaly score over the healthy
  training period — a label-free alert line.

## Result

| Metric | Anomaly detector (Phase 2) | Supervised XGBoost (Phase 1) |
|---|---|---|
| ROC-AUC | **0.95** | 0.92 |
| PR-AUC | 0.39 | 0.44 |
| Recall @ operating point | **0.89** | 0.33 |
| Precision @ operating point | 0.40 | 0.38 |
| Failure labels needed to train | **0** | many (scarce) |

**Early warning** (first alert before each real failure, 48 h look-back):

| Failure | First alert before onset |
|---|---|
| 2020-04-18 | 19.5 h |
| 2020-05-29 | 0.2 h |
| 2020-06-05 | 47.8 h |
| 2020-07-15 | 47.8 h |

## Why this matters

- **It gives the early warning Phase 1 couldn't.** Supervised *predict-ahead*
  collapsed to ROC-AUC ≈ 0.4; the anomaly score instead rises **hours to ~2 days
  before** three of the four failures. That is the leading-indicator signal a
  drift/health monitor needs (Phase 4).
- **It is robust to label scarcity.** Performance depends on a clean baseline, not
  on how many failures exist — the opposite of the supervised model's weakness.
- **Honest caveat — the May 29 event.** Only 0.2 h of lead there: that failure
  develops fast (a 6.5 h interval starting late at night), so the leading indicator
  is short. Worth a per-failure-type view once Backblaze adds more events.

## How the two models combine

They are complementary, not competing: the **supervised** detector is precise once a
known failure signature is present; the **unsupervised** detector gives recall and
lead time without labels and catches novel deviations. Phase 4 fuses them — the
anomaly score becomes a monitored leading indicator that can trigger retraining, and
the supervised model arbitrates the alert.
