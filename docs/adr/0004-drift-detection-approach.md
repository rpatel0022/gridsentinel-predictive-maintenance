# ADR 0004 — Drift detection: in-house PSI + KS over a framework

- **Status:** Accepted
- **Date:** 2026-06-25
- **Deciders:** Rushi Patel

## Context

The self-healing loop needs a drift signal on the live EIA feed to trigger
retraining. The plan named Evidently. Evidently is a capable reporting tool but its
API moves fast across versions and pulls a large dependency tree, which is a
maintenance and CI-stability risk for what is, at its core, a small trigger.

## Decision

Implement the drift **trigger** in-house: **PSI (Population Stability Index)** plus a
**two-sample KS test**, with a combined per-feature verdict (`monitoring/drift.py`).
A feature is flagged when either fires. Keep the logic framework-free so a richer
Evidently *report* can sit on top later without changing the trigger.

## Rationale

- **The trigger must be boring and reliable.** PSI and KS are standard, well
  understood, and ~50 lines of deterministic numpy/scipy — trivially unit-tested,
  no version churn, no CI weight.
- **It already proves itself on real data:** PJM demand, winter vs the current week
  → PSI 0.46, KS p 0.009, drift detected — genuine seasonal non-stationarity, not
  injected noise.
- Separating *trigger* (here) from *reporting* (optional Evidently) keeps the
  retraining decision auditable and dependency-light.

## Alternatives considered

- **Evidently** — nice HTML drift reports, but heavy and API-unstable; better as an
  optional reporting layer on top of this trigger than as the trigger itself.
- **Single test only (KS or PSI)** — KS gets over-sensitive at large N; PSI needs
  binning. Using both, flag-on-either, is more robust than either alone.

## Consequences

- The drift trigger has no heavyweight dependency and stays green in lean CI.
- We get a number (PSI) and a significance (KS p), not a rich visual report — add
  Evidently for the latter if a stakeholder wants dashboards.
- **Revisit trigger:** if we need multivariate/embedding drift or polished reports
  for non-engineers, layer Evidently (or similar) on top.
