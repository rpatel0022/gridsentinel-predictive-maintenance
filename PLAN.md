# Flagship Project Plan — Targeting AMETEK Telular ML Engineer Role

## Context

Rushi starts an **AI Integration internship at AMETEK IntelliPower** (Orange, CA) on June 29, 2026 — work centered on AI-assisted datasheet standardization (LLMs, agents, document workflows). The end goal is to convert that foothold into a **Machine Learning Engineer role at AMETEK Telular** (Herndon, VA, $130–155k), an **IoT** business unit.

Critical insight from comparing the two postings: **they are different business units with different needs.** The internship is GenAI/document work; the target role is **production ML + MLOps + IoT time-series**, with a "3+ years / shipped to production, not just prototypes" bar. A Master's offsets up to 2 years *only if backed by real, production-grade projects*. So the portfolio must prove Rushi can **ship and operate** ML systems, not just train models.

**Decisions locked with the user:** Solid ML, new to MLOps/cloud → emphasize the production layer · One deep flagship · Core IoT-ML + MLOps emphasis · ~3 months, part-time alongside the internship.

**Outcome we're targeting:** a single, live, deployed, monitored ML system that a Telular hiring manager reads as "this person has already done the job," and that an IntelliPower mentor recognizes as their own domain — giving Rushi a concrete artifact to talk about internally for the referral.

---

## The Flagship: "GridSentinel" — Predictive Maintenance & Anomaly Detection for IoT-Connected Power Systems

*(Name is a placeholder — alternatives: VoltWatch, PowerPulse, FleetGuard.)*

A production-grade system that ingests streaming telemetry from a simulated **fleet of IoT-connected UPS / battery units**, predicts failures and remaining useful life (RUL), flags anomalies in real time, and runs with the full MLOps lifecycle: tracking, registry, CI/CD, monitoring, drift-triggered retraining, and a deployed inference service.

**Why this project specifically (the "curated to AMETEK" thesis):**
- **UPS / battery telemetry** = IntelliPower's exact product domain → instant credibility with the team Rushi is embedded in.
- **Fleet of remotely-connected devices streaming telemetry** = Telular's exact world (cellular IoT asset monitoring) → the data shape and problem match the ML role's "IoT datasets."
- **Predictive maintenance** is the canonical IoT-ML problem and exercises supervised + unsupervised + deep learning in one coherent system — all three are named in the posting.
- The MLOps surface area (the part Rushi hasn't done yet) is where 70% of the effort goes — closing the exact gap the role screens for.

### Requirement → Project traceability (this is the curation)

| Telular posting requirement | How the project delivers it |
|---|---|
| Supervised learning (logistic reg, random forest, gradient boosting) | Failure-within-N-cycles **classification** + RUL **regression** with RandomForest / XGBoost baselines |
| Unsupervised learning (clustering, dim. reduction) | **Anomaly detection** (Isolation Forest / autoencoder) + health-state **clustering** + PCA/UMAP on telemetry |
| Deep learning when appropriate | **LSTM / Temporal-CNN** for sequence RUL; compared against the GBM baseline |
| Feature engineering pipelines, preprocessing, training workflows | Reproducible pipeline + **Feature Store (Feast)** for online/offline features |
| Productionize models (not prototypes) + maintain over time | **FastAPI** inference service, containerized, deployed live, versioned |
| MLOps: versioning, CI/CD for ML, monitoring/alerting, automated retraining, governance | **MLflow** (tracking + model registry + stages), **GitHub Actions** CI/CD with metric gates, **Prometheus + Grafana**, **Evidently** drift → retrain trigger, model cards |
| Cloud (AWS/Azure/GCP) + ML services | Deploy inference service + artifact store on **AWS** (ECS/Fargate + S3; SageMaker as stretch) |
| Drift / concept drift detection & iteration | **Evidently** data + prediction drift monitors that trigger automated retraining |
| MLOps tooling: Docker, K8s, MLflow, Feature Stores, CI/CD | Docker + docker-compose (K8s manifest as stretch), all of the above |
| *Bonus:* Edge ML patterns | Export to **ONNX / TFLite**, demo inference on a simulated edge device |
| *Bonus:* Agentic AI + GenAI (vector DB, RAG, embeddings) | **Optional Phase 6:** LLM agent that turns an anomaly alert + RAG over UPS manuals into a maintenance work-order — also bridges the internship's GenAI work |
| *Bonus:* Observability (Prometheus, Grafana) | Core to Phase 4, not optional |

If a single line in the project README maps each bullet above to a commit/dashboard, the resume screen is basically pre-passed.

### Data strategy (no real AMETEK data needed — but real failure signal)

A skeptical hiring manager's first reflex is "simulated data = toy." Defuse it: the **core failure signal must be real**, and what's simulated must be *clearly labeled* and limited to the transport/streaming layer.
- **Primary recommendation — Backblaze drive-failure SMART logs:** millions of real devices, real failures, real telemetry, severe natural class imbalance. It *is* a real fleet — the most defensible "I worked with IoT-scale failure data" claim. (NASA Li-ion Battery is the more UPS-thematic alternate and gives clean RUL labels; NASA C-MAPSS is the RUL benchmark; pick in Phase 0.)
- **Fleet simulator (clearly scoped):** wraps the *real* records and replays them as N devices streaming over time, so there is a live stream to serve, monitor, and retrain on. Drift is induced by **shifting which real sub-population streams when** (e.g., a new drive model / operating regime appears) — not by injecting synthetic noise you then "detect." Be explicit in the README about what is real (the signal and labels) vs simulated (the arrival/transport), because honesty about this is itself a senior signal.
- **Delayed-label reality:** in real predictive maintenance, ground truth (did it actually fail?) arrives *weeks later*. Model the lag explicitly — it makes the monitoring/retraining story credible instead of magical (most portfolios ignore this).

---

## Hiring-Manager Review — grading the plan, and the upgrades to "Must-Hire"

> Read as the Telular ML Engineer hiring manager: I screen for *engineers who have operated ML in production and can show measurable ROI*, not Kaggle/notebook talent. Here is my honest read of the base plan and exactly what would make me say "we have to hire this person."

**Scorecard of the base plan (as written above):**

| What I screen for | Base-plan grade | Why |
|---|---|---|
| Domain/role fit | **A** | UPS + IoT fleet is uncannily on-target; clear they researched us |
| Breadth of ML (sup/unsup/DL) | **A−** | All three present and justified |
| MLOps lifecycle | **B+** | Tracking → registry → CI/CD → monitor → retrain is there; depth unproven |
| "Shipped to prod, not a prototype" | **B−** | Deployed + Dockerized, but no SLOs, tests, or rollback — reads as *could* be a demo |
| **Measurable customer ROI** | **C** | Posting says this twice. Base plan optimizes RMSE/F1, not dollars |
| Software-engineering rigor (tests) | **C** | No test strategy. For an *engineer* hire this is the #1 silent rejection |
| Security / compliance / privacy | **C** | Posting lists it explicitly; base plan is silent |
| Senior judgment (tradeoffs, honesty) | **B** | Good instincts; needs visible "here's what I'd do differently" |

**The eight upgrades that close the gap to must-hire** (each is cheap relative to its signal, and each maps to a line I actually screen on):

1. **Optimize a business-cost metric, not an accuracy metric.** Define the asymmetric cost: a *missed* failure (false negative) = emergency truck-roll + downtime ≈ $$$; a *false alarm* (false positive) = wasted inspection ≈ $. Tune the decision threshold to **minimize expected cost**, and report a headline like *"cuts expected maintenance cost ~X% vs a fixed-schedule baseline at the same coverage."* This single move converts the project from "a model" to "ROI," which is the exact language of the posting. **Always compare against a dumb baseline** (fixed-interval maintenance / "replace on threshold") so the lift is legible.

2. **A real test suite + an ML Test Score.** Most candidates have zero ML tests; this is the cleanest differentiator. Add: unit tests (pipeline transforms), **data-validation tests** (pandera/Great Expectations — schema, ranges, nulls), and **model behavioral tests** (invariance, directional-expectation, minimum-functionality à la CheckList). Self-grade against **Google's "ML Test Score"** rubric (data/model/infra/monitoring) and publish the score in the README. Green CI badge over a real test suite reads as "this person writes production code."

3. **Correctness rigor that survives interrogation.** State explicitly: **temporal/grouped cross-validation** (no future leakage, no same-device rows in train+test), **probability calibration** (reliability curve), and class-imbalance handling beyond `class_weight`. In the interview I *will* ask "how do you know you didn't leak?" — the plan should already answer it.

4. **Production SLOs + safe deployment.** Don't just deploy — operate it: publish a **p99 latency target** and a load-test result, **pydantic request/response schema validation**, **model rollback** via the registry, and a **shadow / canary** path for new model versions. This is the difference between "deployed" and "ran in production."

5. **Monitoring that handles delayed ground truth.** Two-tier observability: *system* metrics (latency/throughput) **and** *model* metrics. Since labels lag, monitor **leading indicators** (per-feature drift, prediction-distribution shift, confidence) for alerting, and **backfill true performance** when labels arrive. Define what alert fires, the threshold, and a one-page **runbook** for the on-call response. Awareness of the delayed-label problem is a senior tell.

6. **Security/compliance, because the posting names it.** Secrets via env/secret-manager (zero keys in git), **dependency + image scanning in CI** (e.g., `pip-audit`/Trivy), basic **PII/data-handling note**, and a **model-governance/audit trail** (who promoted which model version when, via MLflow stages). Cheap; its absence is conspicuous.

7. **Edge ML with real numbers (Telular ships edge devices).** Go past "export to ONNX": **quantize**, then report **size and p99-latency reduction** and run it under a memory-constrained container (or a Raspberry Pi if available). Close with a crisp **cloud-vs-edge inference tradeoff** paragraph. This speaks Telular's native language.

8. **Document like an engineer on a scrum team.** **ADRs** (architecture decision records) for the 3–4 real forks (model choice, serving stack, cloud target), a **model card**, and a **"what I'd do with more time / what I got wrong"** section. The honesty section is counter-intuitively the strongest senior signal — it says you've shipped enough to have scars.

**The undeniable headline** (put it at the top of the README and say it out loud in the referral conversation): *"A self-healing, IoT-scale predictive-maintenance service that cuts expected maintenance cost ~X% vs schedule-based upkeep, runs in production with full MLOps, detects its own drift, retrains automatically, and ships to the edge."* If that sentence is **demonstrably true on a live URL with a dashboard behind it**, the "3+ years in production" bar is functionally met by proof instead of tenure.

---

## Phased build (~12 weeks, part-time)

Sequenced so the early phases land in Rushi's comfort zone (build momentum) and the middle phases — the MLOps layer — are where the real growth and differentiation happen.

- **Phase 0 — Foundation (Wk 1):** Problem framing, **the business-cost function (FN vs FP $)** and the **dumb baseline** to beat, dataset acquisition, repo scaffold (`src/`, `pipelines/`, `serving/`, `monitoring/`, `infra/`, `tests/`, `docs/adr/`), architecture diagram, fleet-simulator skeleton, **CI green from day one** (lint + a trivial test). *[Upgrade 1, 8]*
- **Phase 1 — Data + baselines (Wk 2–3):** Feature pipeline with **data-validation tests** (pandera/Great Expectations), EDA, supervised baselines (RF / XGBoost), **temporal/grouped CV to prove no leakage**, **threshold tuned to the cost function** vs the baseline — all tracked in **MLflow**. *[comfort zone — ship fast; Upgrades 1–3]*
- **Phase 2 — Advanced models (Wk 4–5):** LSTM/Temporal-CNN sequence model; unsupervised anomaly detection + health clustering; **probability calibration**; stand up **Feast** feature store; register best model in MLflow registry. *[Upgrade 3]*
- **Phase 3 — Productionize (Wk 6–8):** **FastAPI** serving with **pydantic schema validation**, Dockerize, docker-compose stack, **GitHub Actions** CI/CD (lint → unit + data + **model-behavioral tests** → train → evaluate → metric-gate → **dependency/image scan** → build), registry **stages + rollback**, secrets via env/secret-manager, **ML Test Score** published. *[growth zone — the core; Upgrades 2, 4, 6]*
- **Phase 4 — Observability + self-healing (Wk 9–10):** **Prometheus + Grafana** two-tier dashboards (system *and* model metrics), **Evidently** drift monitors on **leading indicators**, **delayed-label backfill** of true performance, drift → **automated retrain → canary/shadow → promote**, alert thresholds + a one-page **runbook**. *[the headline differentiator; Upgrades 4, 5]*
- **Phase 5 — Cloud + Edge + polish (Wk 11–12):** Deploy live on **AWS** (ECS/Fargate + S3) with a **p99 latency SLO + load-test result** and a short **cost note**; **quantize** the model and report **size/latency reduction** on a constrained edge target + cloud-vs-edge tradeoff; write README (with the ROI headline + traceability + ML Test Score), **ADRs**, model card, **"what I'd do differently"**, and a 3–5 min Loom. *[Upgrades 4, 7, 8]*
- **Phase 6 — Optional GenAI bridge (stretch):** LLM agent + **RAG** over UPS manuals that converts an anomaly alert into a draft maintenance work-order. Reuses internship skills and uniquely targets both business units.

---

## Deliverables (what "production-grade, not a prototype" means here)

1. **Live deployed inference endpoint** (public URL) + two-tier **Grafana** dashboard, with a published **p99 latency SLO** and load-test result.
2. **GitHub repo**: clean structure, architecture diagram, requirement-traceability table, **green CI badge over a real test suite**, and a published **Google ML Test Score**.
3. **The ROI headline, quantified**: *"~X% lower expected maintenance cost vs schedule-based upkeep"* — front and center, with the baseline comparison reproducible.
4. **Engineering docs**: **ADRs** for the key forks, a **model card** (data, metrics, calibration, bias, drift behavior), a monitoring **runbook**, and a **"what I'd do differently"** section.
5. **3–5 min video walkthrough** showing the full self-healing loop live (highest-leverage artifact for a hiring manager) + a short write-up/blog — quotable in interviews and the referral conversation.

---

## Internal-referral playbook (parallel to the build)

- During the internship, do excellent visible work on the datasheet/GenAI project — that's the trust foundation.
- Mid-internship, mention GridSentinel to your IntelliPower mentor framed in *their* domain ("predictive maintenance for our UPS units") — they may know the Telular team.
- Near the end, present GridSentinel as evidence you can do the Telular role, and explicitly express interest in the ML Engineer path. Ask your mentor for a warm intro/referral to Telular.
- The optional GenAI phase lets you tell one continuous story: "I extended the document/agent work from my internship into a full IoT-ML system."

---

## Verification — how each phase proves itself

This is a portfolio system, so "tests" = concrete demoable artifacts per phase, each tied to a hiring-manager screen:
- **Phase 1:** MLflow shows tracked runs; the cost-tuned model beats the **schedule-based baseline** on **expected $ cost**; data-validation tests pass; CV is grouped/temporal (leakage check documented).
- **Phase 2:** LSTM RUL error vs GBM baseline reported; anomaly detector precision/recall measured; calibration curve published.
- **Phase 3:** `docker-compose up` brings the stack live; `curl` returns a schema-validated prediction; a green CI run shows the metric gate **and** the dependency/image scan blocking bad inputs; **ML Test Score** rendered.
- **Phase 4:** Grafana shows system + model metrics; shifting the streaming sub-population trips the drift monitor and kicks off **retrain → canary → promote**, with **rollback** demonstrated; delayed labels backfill true performance.
- **Phase 5:** Public AWS URL serves under the stated **p99 SLO** (load-test artifact); quantized model runs on the constrained edge target with **measured size/latency reduction**; the video walks the full loop end-to-end.

---

## Open / to-confirm before/while building

- Final dataset choice (NASA Battery vs C-MAPSS vs Backblaze) — pick in Phase 0 after a quick data-quality look.
- Cloud provider — AWS recommended (most job postings, incl. SageMaker, name it), but Azure is viable since AMETEK is a Microsoft-365 shop; confirm early since it shapes Phase 5.
- Whether to attempt the optional GenAI Phase 6 given the part-time timeline.

> Note: this is a strategy/portfolio plan, not a change to an existing codebase, so no code exploration was needed. On approval, the natural first step is **Phase 0**, and it's worth saving the AMETEK career context + this project to memory for continuity across sessions.
