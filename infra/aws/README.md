# AWS deploy — ECS/Fargate + S3

Deploy notes for the GridSentinel inference service. This is the documented target
(ADR-0001: AWS); it has not been applied from this repo. `task-definition.json` is a
ready-to-register Fargate task; fill the `<ACCOUNT_ID>` / `<REGION>` placeholders.

## Architecture

```
        ┌─────────────┐   pull image    ┌────────────────────┐
  ECR ──┤ gridsentinel│◄────────────────┤ ECS Fargate service │── ALB :443 ──► clients
        └─────────────┘                 │  task: api :8000    │
        S3 (model bundles) ───sync────► │  /models/*.joblib   │
        SSM Parameter Store ──secret──► │  EIA_API_KEY        │
        CloudWatch Logs  ◄──────────────┤  awslogs            │
```

- **ECR** holds the image built by CI (the `Dockerfile`).
- **S3** holds the gated model bundle. The model is **never in git and never baked
  blindly** — CI runs the metric gate (`model-eval` workflow), uploads the passing
  bundle to `s3://gridsentinel-models/`, and the task syncs it to `/models` on start
  (task role grants read on that bucket). New model → new gate → new S3 object →
  rolling task restart picks it up.
- **SSM Parameter Store (SecureString)** holds `EIA_API_KEY`; injected as a secret —
  **zero keys in git or the image** (Upgrade 6).
- **CloudWatch** for logs; Prometheus scrapes the task's `/metrics` (sidecar or an
  AMP agent) into the Grafana dashboard from `monitoring/`.

## Deploy

```bash
# 1. Build & push the image (CI does this on a gated build)
aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR"
docker build -t "$ECR/gridsentinel:$GIT_SHA" . && docker push "$ECR/gridsentinel:$GIT_SHA"

# 2. Upload the gated model bundle
aws s3 cp models/anomaly_detector.joblib s3://gridsentinel-models/anomaly_detector.joblib

# 3. Register the task and roll the service
aws ecs register-task-definition --cli-input-json file://infra/aws/task-definition.json
aws ecs update-service --cluster gridsentinel --service api --force-new-deployment
```

Rollback is a redeploy of the prior task-definition revision (and `registry.rollback`
for the model itself — see `docs/runbook.md`).

## Cost note (rough, us-east-1)

| Item | Sizing | ~Monthly |
|---|---|---|
| Fargate task | 0.5 vCPU / 1 GB, 1 task 24×7 | ~$18 |
| ALB | 1, low traffic | ~$16 |
| S3 + ECR | a few GB | < $1 |
| CloudWatch logs | low volume | ~$1–3 |
| **Total** | single-task demo | **~$35–40/mo** |

Scale-to-zero (stop the service when idle) or an ALB-less single task with a public
IP drops this toward ~$18. SageMaker endpoints were considered (ADR-0001) but cost
more for a single small model than a Fargate task; revisit if managed autoscaling or
multi-model hosting is needed.
