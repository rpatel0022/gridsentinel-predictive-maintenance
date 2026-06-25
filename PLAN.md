# GridSentinel — Project Plan

## Context

GridSentinel is a production-grade machine-learning system for **predictive
maintenance and anomaly detection on a fleet of IoT-connected power units**. The goal
is not a notebook model but a **system that ships and operates**: data validation,
training, a served inference API, full MLOps (tracking, registry, CI/CD, monitoring,
drift-triggered retraining), and an edge story — all built on **100% real data**.

The design emphasis is deliberately on the **production layer**. Training a good model
is necessary but not sufficient; the differentiator is operating it safely over time.

---

## The system: "GridSentinel" — Predictive Maintenance & Anomaly Detection for IoT-Connected Power Systems

A production-grade system that ingests streaming telemetry from a **fleet of
IoT-connected power / battery units**, predicts failures and remaining useful life
(RUL), flags anomalies in real time, and runs the full MLOps lifecycle: tracking,
registry, CI/CD, monitoring, drift-triggered retraining, and a deployed inference
service.

**Why this problem:**
- **Predictive maintenance** is the canonical IoT-ML problem and exercises supervised,
  unsupervised, and deep-learning methods in one coherent system.
- **Fleet telemetry from remotely-connected devices** is a realistic data shape — the
  same problem structure as cellular IoT asset monitoring.
- The **MLOps surface area** is where most of the engineering effort goes, because that
  is where production ML actually lives and where most prototypes fall short.

### Capability → proof traceability

| ML capability | How the project delivers it |
|---|---|
| Supervised learning | Failure-within-N-cycles **classification** + RUL **regression** with RandomForest / XGBoost baselines |
| Unsupervised learning | **Anomaly detection** (Isolation Forest / autoencoder) + health-state **clustering** + PCA/UMAP on telemetry |
| Deep learning | **LSTM / Temporal-CNN** for sequence models; compared against the GBM baseline |
| Feature engineering + preprocessing + training workflows | Reproducible pipeline with shared train/serve aggregation + schema validation |
| Productionize (not prototypes) + maintain over time | **FastAPI** inference service, containerized, versioned, served |
| MLOps: versioning, CI/CD, monitoring, automated retraining, governance | **MLflow** (tracking + registry + stages), **GitHub Actions** CI/CD with metric gates, **Prometheus + Grafana**, drift → retrain trigger, model cards |
| Cloud + ML services | Inference service + artifact store target on **AWS** (ECS/Fargate + S3) |
| Drift / concept-drift detection & iteration | PSI/KS drift monitors that trigger automated retraining |
| MLOps tooling: Docker, CI/CD | Docker + docker-compose, all of the above |
| *Bonus:* Edge ML | Quantize / shrink the ensemble; measure size + latency reduction on a constrained target |
| *Bonus:* Observability | Prometheus + Grafana, core to Phase 4 |
| *Bonus:* Agentic AI + GenAI | **Optional Phase 6:** LLM agent that turns an anomaly alert + RAG over equipment manuals into a maintenance work-order |

The principle: **each capability maps to an openable artifact** (a commit, a results
doc, a dashboard) — the artifact makes the claim, not the prose.

### Data strategy — 100% real data, plus a genuinely live feed

Hard rule: **every byte of training signal is real, and the streaming/production layer
pulls real, continuously-updating public data.** Nothing is fabricated. The "simulator"
is not a generator of fake numbers — it is a **real-time ingestion service** that hits
live public APIs on a schedule, exactly like a production connector would.

**One honest design decision, stated up front:** the two data tiers do *different jobs*
and are deliberately not the same source pretending to be one. **Tier 1
(failure-labeled datasets) trains and evaluates the predictive-maintenance models** —
that is where ground-truth failure events live. **Tier 2 (live public feeds) drives the
production/MLOps layer** — ingestion, serving, and especially monitoring/drift/retraining
— because that is where *new data genuinely keeps arriving* and where non-stationarity
is real rather than injected. The live feeds carry no failure labels, so the production
model runs against them as a continuously-monitored stream with **absent/delayed ground
truth**, and true performance is backfilled when labels become available. That
separation is the realistic shape of operating ML in the field; naming it explicitly is
the senior move — the alternative (implying one magic feed both trains and serves a
labeled model live) is exactly the seam a sharp reviewer pulls on.

**Tier 1 — Real labeled-failure data (for training the predictive-maintenance models):**
- **MetroPT-3** *(primary — UCI / Zenodo)*: real Air Production Unit (compressor) sensor
  signals from Porto metro trains — pressure, temperature, motor current, valve states —
  with **real failure events from the operator's maintenance reports**. An actual
  industrial IoT predictive-maintenance dataset captured as a continuous data flow.
- **Backblaze Drive Stats** *(fleet-scale + ongoing)*: hundreds of thousands of real
  drives, real failures, severe natural class imbalance — and **a new batch is published
  every quarter**, so the dataset itself is "new incoming data." The strongest
  fleet-scale failure-data claim.
- **NASA Li-ion Battery Aging** + **NASA Bearing** run-to-failure *(PCoE repository)*:
  real run-to-failure experiments yielding clean RUL labels (optional).
- **Locked:** MetroPT-3 is the **primary** (drives Phases 1–4); Backblaze enters at
  Phase 2 as the **fleet-scale companion** (separate model track — strengthens the
  heterogeneous-IoT-fleet claim and supplies a genuine quarterly retraining cadence).
  NASA Battery kept as an optional RUL showcase. Confirm MetroPT-3 quality in Phase 0
  before building on it.

**Tier 2 — Genuinely live, continuously-updating feeds (the real "new incoming data"):**
- **EIA Open Data API v2** — near-real-time hourly US electricity demand / net generation
  / interchange per balancing authority, refreshed ~1 hour after each hour. Free with an
  API key. Power-domain.
- **GridStatus** (`gridstatus` lib + gridstatus.io API) — real-time ISO load / fuel-mix /
  prices (CAISO, ERCOT, PJM, …). Free tier is rate-limited, so cache locally.
- **Sensor.Community** — 12,000+ **physical** DIY sensors across 82 countries
  (PM2.5/PM10, temp, humidity, pressure) via a free public JSON API, updating ~every 10
  minutes. Authentic live device telemetry — real hardware, real drift, real outages.
- **Transport realism (optional):** republish the ingested live readings over **MQTT**
  (a public broker or self-hosted Mosquitto) so the pipeline consumes data the way real
  cellular IoT devices emit it.

**How this makes the production story real (not staged):**
- **Drift is naturally occurring**, not injected — grid demand shifts with
  weather/season/time-of-day; live sensors degrade, drop offline, and come back. The
  drift monitor catches *real* non-stationarity.
- **Delayed ground truth is modeled honestly** — in real predictive maintenance the
  "did it fail?" label arrives weeks later. The pipeline alerts on leading indicators now
  and **backfills true performance when labels land**.
- **Continuous arrival enables a real retraining cadence** — because new data genuinely
  keeps coming (live feeds hourly, Backblaze quarterly), automated retraining has
  something authentic to retrain on.

---

## Engineering principles — and the artifact that proves each

The aim is a system that survives technical interrogation. Each principle below is cheap
relative to the signal it sends, and each produces a concrete, openable artifact.

| Principle | The proof artifact |
|---|---|
| Domain / problem fit | Real power/IoT failure data + live grid feeds + a clear problem statement |
| Breadth of ML (sup / unsup / DL) | MLflow runs + a results table with baseline deltas |
| MLOps lifecycle | Grafana dashboard + a CI run + a recorded retrain/promote/rollback cycle |
| Shipped, not a prototype | A served endpoint + a load-test artifact + the CI/CD pipeline |
| Measurable ROI | A decision threshold tuned to a **$ cost function** vs a fixed-schedule baseline |
| SWE rigor (tests) | Green CI over the test suite + a published Google ML Test Score |
| Security / compliance | CI dependency + image scanning + a model-governance/audit trail |
| Senior judgment | ADRs for the real forks + a model card + an honest "what I'd do differently" |

The eight build choices that produce those artifacts:

1. **Optimize a business-cost metric, not an accuracy metric.** Define the asymmetric
   cost: a *missed* failure (false negative) = emergency truck-roll + downtime ≈ $$$; a
   *false alarm* (false positive) = wasted inspection ≈ $. Tune the decision threshold to
   **minimize expected cost**, and report *"cuts expected maintenance cost ~X% vs a
   fixed-schedule baseline at the same coverage."* This converts the project from "a
   model" to "ROI." **Always compare against a dumb baseline** so the lift is legible.

2. **A real test suite + an ML Test Score.** Unit tests (pipeline transforms),
   **data-validation tests** (pandera — schema, ranges, nulls), and **model behavioral
   tests** (invariance, directional-expectation, minimum-functionality à la CheckList).
   Self-grade against **Google's "ML Test Score"** rubric (data/model/infra/monitoring)
   and publish the score.

3. **Correctness rigor that survives interrogation.** **Temporal/grouped
   cross-validation** (no future leakage, no same-device rows in train+test),
   **probability calibration** (reliability curve), and class-imbalance handling beyond
   `class_weight`. A production-ML review *will* ask "how do you know you didn't leak?" —
   answer it in advance.

4. **Production SLOs + safe deployment.** Publish a **p99 latency target** and a
   load-test result, **pydantic request/response schema validation**, **model rollback**
   via the registry, and a **shadow / canary** path for new model versions.

5. **Monitoring that handles delayed ground truth.** Two-tier observability: *system*
   metrics (latency/throughput) **and** *model* metrics. Since labels lag, monitor
   **leading indicators** (per-feature drift, prediction-distribution shift) for
   alerting, and **backfill true performance** when labels arrive. Define what alert
   fires, the threshold, and a one-page **runbook**.

6. **Security/compliance.** Secrets via env/secret-manager (zero keys in git),
   **dependency + image scanning in CI** (`pip-audit` / Trivy), a basic **PII/data-handling
   note**, and a **model-governance/audit trail** (who promoted which version when).

7. **Edge ML with real numbers.** Go past "export to ONNX": **quantize / shrink**, then
   report **size and p99-latency reduction** under a memory-constrained target, and close
   with a crisp **cloud-vs-edge inference tradeoff**.

8. **Document like an engineer on a team.** **ADRs** for the real forks (model choice,
   serving stack, cloud target), a **model card**, and an honest **"what I'd do with more
   time / what I got wrong"** section.

**The headline:** *"A self-healing, IoT-scale predictive-maintenance service that cuts
expected maintenance cost ~X% vs schedule-based upkeep, runs in production with full
MLOps, detects its own drift, retrains automatically, and ships to the edge."* When that
sentence is **demonstrably true with a dashboard behind it**, the artifact — not the
prose — is what carries the claim.

---

## Phased build

Sequenced so the early phases build momentum and the middle phases — the MLOps layer —
are where the real depth lives.

- **Phase 0 — Foundation:** Problem framing, **the business-cost function (FN vs FP $)**
  and the **dumb baseline** to beat, acquire the **real** failure dataset (MetroPT-3 /
  Backblaze), stand up the **live-ingestion service skeleton** (real connectors to EIA /
  Sensor.Community / GridStatus), repo scaffold (`src/`, `pipelines/`, `serving/`,
  `monitoring/`, `infra/`, `tests/`, `docs/adr/`), architecture diagram, **CI green from
  day one**. *[Principles 1, 8]*
- **Phase 1 — Data + baselines:** Feature pipeline with **data-validation tests**, EDA,
  supervised baselines (RF / XGBoost), **temporal/grouped CV to prove no leakage**,
  **threshold tuned to the cost function** vs the baseline — all tracked in **MLflow**.
  *[Principles 1–3]*
- **Phase 2 — Advanced models:** LSTM/Temporal-CNN sequence model; unsupervised anomaly
  detection + health clustering; **probability calibration**; register the best model in
  the MLflow registry. *[Principle 3]*
- **Phase 3 — Productionize:** **FastAPI** serving with **pydantic schema validation**,
  Dockerize, docker-compose stack, **GitHub Actions** CI/CD (lint → unit + data +
  **model-behavioral tests** → train → evaluate → metric-gate → **dependency/image scan**
  → build), registry **stages + rollback**, secrets via env/secret-manager, **ML Test
  Score** published. *[Principles 2, 4, 6]*
- **Phase 4 — Observability + self-healing:** **Prometheus + Grafana** two-tier
  dashboards (system *and* model metrics), drift monitors on **leading indicators**,
  **delayed-label backfill** of true performance, drift → **automated retrain →
  canary/shadow → promote**, alert thresholds + a one-page **runbook**. *[Principles 4, 5]*
- **Phase 5 — Cloud + Edge + polish:** **AWS** (ECS/Fargate + S3) target with a **p99
  latency SLO + load-test result** and a short **cost note**; **quantize/shrink** the
  model and report **size/latency reduction** on a constrained edge target + cloud-vs-edge
  tradeoff; write README (ROI headline + traceability + ML Test Score), **ADRs**, model
  card, and a **"what I'd do differently"** section. *[Principles 4, 7, 8]*
- **Phase 6 — Optional GenAI bridge (stretch):** LLM agent + **RAG** over equipment
  manuals that converts an anomaly alert into a draft maintenance work-order.

---

## Deliverables (what "production-grade, not a prototype" means here)

1. **Served inference endpoint** + two-tier **Grafana** dashboard, with a published
   **p99 latency SLO** and load-test result.
2. **GitHub repo**: clean structure, architecture diagram, capability-traceability table,
   **green CI badge over a real test suite**, and a published **Google ML Test Score**.
3. **The ROI headline, quantified**: *"~X% lower expected maintenance cost vs
   schedule-based upkeep"* — front and center, with the baseline comparison reproducible.
4. **Engineering docs**: **ADRs** for the key forks, a **model card** (data, metrics,
   calibration, bias, drift behavior), a monitoring **runbook**, and a **"what I'd do
   differently"** section.
5. **A short video walkthrough** showing the full self-healing loop end-to-end.

---

## Verification — how each phase proves itself

"Tests" here = concrete demoable artifacts per phase:
- **Phase 1:** MLflow shows tracked runs; the cost-tuned model beats the **schedule-based
  baseline** on **expected $ cost**; data-validation tests pass; CV is grouped/temporal
  (leakage check documented).
- **Phase 2:** LSTM RUL error vs GBM baseline reported; anomaly detector precision/recall
  measured **against the real failure events** (not injected faults); calibration curve
  published (or calibration honestly rejected with evidence).
- **Phase 3:** `docker-compose up` brings the stack live; `curl` returns a
  schema-validated prediction; a green CI run shows the metric gate **and** the
  dependency/image scan; **ML Test Score** rendered.
- **Phase 4:** Grafana shows system + model metrics; **naturally-occurring drift in the
  live feed** trips the drift monitor and kicks off **retrain → canary → promote**, with
  **rollback** demonstrated; delayed labels backfill true performance.
- **Phase 5:** the served model runs under the stated **p99 SLO** (load-test artifact);
  the quantized model runs on the constrained edge target with **measured size/latency
  reduction**; the video walks the full loop end-to-end.

---

## Locked decisions

- **Primary dataset:** **MetroPT-3** (Phases 1–4), with **Backblaze** as the fleet-scale
  companion from Phase 2; NASA Battery optional. **No synthetic datasets — dropped per
  the real-data mandate.** Validate MetroPT-3 quality in Phase 0 before committing build
  effort.
- **Primary live feed:** **EIA Open Data API v2** (hourly US electricity demand —
  power-domain, free key). Drives the production/monitoring/retraining layer.
- **Cloud:** **AWS** (ECS/Fargate + S3; SageMaker as stretch) — broad match and the free
  tier covers the demo.

## Still open (decide while building)

- Whether to wire **Sensor.Community** as a second live feed (physical-device drift) —
  defer to Phase 4 unless Phase 0 has slack.
- Whether to add the **MQTT** transport layer in Phase 0 or defer to Phase 4.
- Whether to attempt the optional **GenAI Phase 6**.
