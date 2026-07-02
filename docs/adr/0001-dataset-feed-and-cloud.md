# ADR 0001 — Primary dataset, live feed, and cloud target

- **Status:** Accepted
- **Date:** 2026-06-24

## Context

GridSentinel needs three foundational choices locked before Phase 0 build work,
because each one shapes everything downstream:

1. a **failure-labeled dataset** to train and evaluate the predictive-maintenance
   models on;
2. a **genuinely-live public feed** to drive the production / monitoring /
   retraining layer (real, continuously-arriving data — no synthetic streams);
3. a **cloud target** for the live deployment (shapes the Phase 5 serving stack).

Constraints: a hard "100% real data, no synthetic" rule, and a target problem
domain spanning **power/battery equipment** (UPS-style units) and **cellular IoT
fleet monitoring** — so the data and feed choices should land in that world.

## Decision

1. **Primary dataset: MetroPT-3**, with **Backblaze Drive Stats** as a
   fleet-scale companion introduced in Phase 2. NASA Li-ion Battery is kept as an
   optional UPS-themed RUL showcase.
2. **Primary live feed: EIA Open Data API v2** (hourly US electricity demand /
   generation per balancing authority).
3. **Cloud: AWS** (ECS/Fargate + S3; SageMaker as a stretch).

## Rationale

**Dataset — MetroPT-3 primary + Backblaze companion.** MetroPT-3 is the only
candidate that is a genuine *industrial IoT predictive-maintenance* dataset with
real failure events drawn from maintenance reports, in a compressor /
power-equipment domain that fits the target problem. It supports all three ML
modes in scope — failure classification, anomaly detection, and RUL — so Phase 1
can ship quickly. It cannot, however, carry an
"IoT-scale fleet" claim or a real retraining cadence, so Backblaze is layered in
at Phase 2: 344k devices (fleet scale), real failures with severe class
imbalance, and a new batch every quarter (authentic ongoing data to retrain on).
The two are different domains, so they stay **separate model tracks** — which
strengthens, rather than muddies, the "handles heterogeneous IoT fleets" story.

**Live feed — EIA.** Power-domain, lands squarely in the target world, free with
an API key, and exhibits natural weather/season/time-of-day
drift — so the drift monitor catches *real* non-stationarity rather than noise we
injected. (Note the deliberate data seam: EIA has no failure labels; it drives
the production/monitoring layer with absent/delayed ground truth, while the
labeled datasets train the models. See [architecture.md](../architecture.md) →
the data seam.)

**Cloud — AWS.** The most broadly-adopted cloud for ML services (ECS/Fargate, S3,
SageMaker), and its free tier covers the live-demo footprint.

## Alternatives considered

- **Dataset: Backblaze as primary** — strongest fleet/scale story, but hard
  drives are off-domain for the power-equipment framing and the data volume slows
  Phase 1.
- **Dataset: NASA Battery as primary** — most battery-thematic, but small lab
  experiments with no fleet or streaming narrative.
- **Live feed: Sensor.Community** — most authentic physical-device drift, but
  air-quality domain is off-thesis; kept as a possible second feed.
- **Cloud: Azure** — plausible, but weaker breadth of ML services for this demo
  and no offsetting advantage.

## Consequences

- Phase 0 must include a **MetroPT-3 data-quality spike** before any modeling —
  if the data is unusable, this ADR gets superseded, not patched.
- Two model tracks (MetroPT-3, Backblaze) means two feature pipelines; acceptable
  given AI-assisted parallel development, but it is the main scope risk to watch.
- AWS choice commits Phase 5 to ECS/Fargate + S3; revisit only if free-tier
  limits block the live URL.
- **Revisit trigger:** MetroPT-3 quality failure, or EIA API access/rate limits
  proving impractical for a continuous feed.
