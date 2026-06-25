# infra/

Deployment and infrastructure (Phase 5).

- `aws/task-definition.json` — ECS Fargate task for the inference API (port 8000,
  `/health` healthcheck, `EIA_API_KEY` from SSM, CloudWatch logs).
- `aws/README.md` — deploy notes: ECR/ECS/S3/SSM architecture, the deploy commands,
  and a rough monthly **cost note** (~$35–40/mo, or ~$18 scaled down). The model
  bundle ships via S3 (gated in CI), never committed and never blindly baked.

These are documented targets (ADR-0001 chose AWS); not applied from this repo. The
local equivalent is `docker compose up` (API + Prometheus + Grafana).
