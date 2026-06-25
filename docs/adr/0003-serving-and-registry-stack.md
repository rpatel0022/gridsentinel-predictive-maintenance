# ADR 0003 — Serving stack: FastAPI + a file-based registry

- **Status:** Accepted
- **Date:** 2026-06-25

## Context

Phase 3 needed to turn the model into a deployed, operable service with schema
validation, versioning, promotion, and rollback — the "shipped, not a prototype"
bar — on a part-time timeline and a free-tier footprint. MLflow is already in the
stack for experiment tracking.

## Decision

Serve with **FastAPI** (pydantic request validation reusing the Phase 0 physical
contract), package the model as a self-describing **joblib bundle** (pipeline +
threshold + feature order + provenance), and manage stages/rollback with a
**file-based registry** (`registry.json` + bundle files + an audit log) rather than
standing up the MLflow Model Registry server.

## Rationale

- **One contract, enforced once.** The serving request is validated against the
  same `metropt3_schema` ranges the training data is held to, so bad telemetry is
  rejected (422) by the same rule everywhere — no schema drift between train/serve.
- **The bundle carries its own threshold + feature order**, so a model swap never
  touches service code and train/serve features can't skew.
- **The file registry is enough** for stages, promote, rollback, and a governance
  audit trail, with zero extra infrastructure — its transition logic is pure and
  unit-tested. MLflow's registry adds a server/DB to run and secure for the same
  outcome at this scale.

## Alternatives considered

- **MLflow Model Registry server** — richer UI/stages, but a service to host and
  secure; revisit if multiple consumers or a team workflow need it.
- **BentoML / KServe** — heavier serving frameworks; overkill for one model on a
  free tier.
- **Validate features only (not raw readings)** — rejected: accepting raw readings
  and validating them is the realistic IoT contract and exercises the schema.

## Consequences

- Trivial local/edge deploy (`docker compose up`); the artifact is mounted, never
  baked into the image.
- The registry is single-node (file-backed); concurrent writers aren't supported.
- **Revisit trigger:** a second model/consumer, a team promotion workflow, or a
  cloud deploy that wants a managed registry → move to MLflow Registry or SageMaker.
