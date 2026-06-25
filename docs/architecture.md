# GridSentinel — Architecture

> Reflects the **built** system (Phases 0–5). ✅ built · ◑ documented (not deployed
> from this repo) · ○ deferred follow-on.

```mermaid
flowchart LR
    subgraph Data["Data sources (100% real)"]
        DS1["MetroPT-3 ✅<br/>failure-labeled (4 real failures)"]
        DS2["Backblaze ○<br/>fleet-scale (needs data access)"]
        LF["EIA live feed ✅<br/>hourly grid demand"]
    end

    subgraph Train["Training & registry"]
        FP["Feature pipeline ✅<br/>+ pandera validation"]
        ML["Models ✅: RF/XGB (supervised)<br/>· Isolation-Forest anomaly · LSTM ○"]
        REG["MLflow tracking ✅<br/>+ file registry: stages/rollback/audit ✅"]
    end

    subgraph Serve["Production"]
        API["FastAPI service ✅<br/>pydantic schema validation · /metrics"]
        COST["Cost-optimised<br/>decision threshold ✅"]
    end

    subgraph Ops["Observability & self-healing"]
        MON["Prometheus + Grafana ✅<br/>system + model metrics"]
        DRIFT["PSI/KS drift ✅<br/>on the live feed"]
        RETRAIN["Drift → retrain → metric-gate →<br/>promote / keep · rollback ✅"]
        BF["Delayed-label backfill ✅<br/>true perf when labels land"]
    end

    DS1 --> FP
    DS2 -.-> FP
    FP --> ML --> REG --> API
    COST --> API
    LF --> API
    API --> MON
    LF --> DRIFT
    API --> DRIFT
    DRIFT --> RETRAIN --> REG
    API -.-> BF --> MON

    API -->|deploy notes| AWS["AWS ECS/Fargate + S3 ◑"]
    REG -->|size/latency benchmark| EDGE["Edge: 50-tree model ✅<br/>5.9× smaller, 4× faster"]
```

CI/CD guards the loop: a **metric gate** (`pipelines/metric_gate.py`) rebuilds the
model on real data and fails the build on regression; **pip-audit** + **Trivy** scan
deps and image; the **load test** holds a per-request **p99 31 ms** SLO.

## The data seam (read this before asking "where are the live labels?")

Two data tiers, two jobs — deliberately not one source pretending to be both:

- **Failure-labeled data** (MetroPT-3; Backblaze when accessible) **trains and
  evaluates** the models. That is where ground-truth failures live.
- **The live EIA feed drives production/monitoring/retraining.** It has *no* failure
  labels, so the served model runs against it as a continuously-monitored stream with
  **absent/delayed ground truth**. The drift monitor watches leading indicators now;
  the **delayed-label backfill** (`monitoring/backfill.py`) computes true performance
  if and when labels arrive. This is the realistic shape of operating ML in the field
  — see [ADR 0001](adr/0001-dataset-feed-and-cloud.md) and PLAN.md.
