# GridSentinel — Architecture

> Phase 0 sketch. Components fill in as phases land; this diagram is the contract
> the build works toward. Boxes marked _(Pn)_ arrive in that phase.

```mermaid
flowchart LR
    subgraph Data["Data sources (100% real)"]
        DS1["MetroPT-3<br/>failure-labeled (P1)"]
        DS2["Backblaze<br/>fleet-scale (P2)"]
        LF["EIA live feed<br/>hourly grid demand (P0)"]
    end

    subgraph Train["Training & registry"]
        FP["Feature pipeline<br/>+ data-validation tests (P1)"]
        ML["Models: RF/XGB · LSTM ·<br/>anomaly detection (P1–P2)"]
        REG["MLflow<br/>tracking + registry (P1)"]
    end

    subgraph Serve["Production"]
        API["FastAPI service<br/>pydantic schemas (P3)"]
        COST["Cost-optimised<br/>decision threshold (P0 core)"]
    end

    subgraph Ops["Observability & self-healing"]
        MON["Prometheus + Grafana<br/>system + model metrics (P4)"]
        DRIFT["Evidently drift<br/>on leading indicators (P4)"]
        RETRAIN["Drift → retrain →<br/>canary → promote → rollback (P4)"]
    end

    DS1 --> FP
    DS2 --> FP
    FP --> ML --> REG --> API
    COST --> API
    LF --> API
    API --> MON
    LF --> DRIFT
    API --> DRIFT
    DRIFT --> RETRAIN --> REG

    API -->|deployed| AWS["AWS ECS/Fargate + S3 (P5)"]
    REG -->|quantized| EDGE["ONNX edge target (P5)"]
```

## The data seam (read this before asking "where are the live labels?")

Two data tiers, two jobs — deliberately not one source pretending to be both:

- **Failure-labeled datasets** (MetroPT-3, Backblaze) **train and evaluate** the
  models. That is where ground-truth failures live.
- **The live EIA feed drives production/monitoring/retraining.** It has *no*
  failure labels, so the served model runs against it as a continuously-monitored
  stream with **absent/delayed ground truth**; true performance is backfilled if
  and when labels arrive. This is the realistic shape of operating ML in the
  field — see [ADR 0001](adr/0001-dataset-feed-and-cloud.md) and PLAN.md.
