# GridSentinel — Project Status & Interview Recall Guide

_Last updated: 2026-06-25. This is the **single source of truth** for the project: what
it is, every decision and why, every number and where it came from, the bugs found and
fixed, and the honest limits. Written so that months later you can reopen this file and
**explain the whole project end-to-end in an interview** without re-reading the code._

> **Read order for a fresh session or a quick refresher:** this file → `PLAN.md` (full
> strategy) → `README.md` (public face) → the `docs/` results files for any number you
> want to defend in detail.

---

## 0. The 60-second elevator pitch

**GridSentinel is a production-grade predictive-maintenance ML system for a fleet of
IoT-connected power units.** It ingests streaming sensor telemetry, predicts equipment
failures with early warning, flags anomalies in real time, serves predictions behind a
schema-validated API, and runs the **full MLOps lifecycle** — CI metric gate, drift
detection on a genuinely live data feed, and a self-healing retrain→gate→promote/rollback
loop. It is built on **100% real data** (no synthetic numbers anywhere) and is graded on
**dollars saved, not raw accuracy**, because a missed failure costs ~100× a false alarm.

**Why it exists:** a portfolio piece engineered to map 1:1 onto the AMETEK / Telular
**ML Engineer** job posting — every line item in that posting has an openable proof
artifact in this repo (see the traceability table in `README.md`).

**The single most important design idea:** the decision threshold is tuned to **minimize
expected dollar cost** under an asymmetric cost model, and every result is reported as
**lift over a dumb fixed-schedule baseline** — that is what a maintenance buyer actually
cares about.

---

## 1. Where everything lives (the map you'll forget first)

| You want… | Look at |
|---|---|
| The public pitch + traceability table | `README.md` |
| Full strategy & phased plan | `PLAN.md` |
| The ROI core (cost model, optimal threshold) | `src/gridsentinel/cost.py` |
| Temporal cross-validation (leakage guard) | `src/gridsentinel/cv.py` |
| Data validation contract (MetroPT-3) | `pipelines/metropt3_schema.py` |
| Supervised baselines (RF/XGBoost) | `pipelines/train_baseline.py` |
| Anomaly detector (the primary model) | `pipelines/anomaly.py` |
| Deep learning (LSTM) + MLP sequence baseline | `pipelines/lstm_model.py`, `pipelines/sequence_model.py` |
| Fleet-reliability model (Backblaze) | `pipelines/backblaze.py`, `pipelines/backblaze_schema.py` |
| The serving API (FastAPI) | `serving/app.py`, `serving/model.py` |
| Model registry (promote/rollback/audit) | `serving/registry.py` |
| Drift detection (PSI/KS) | `monitoring/drift.py`, `monitoring/eia_drift.py` |
| Self-healing retrain loop | `monitoring/self_heal.py`, `monitoring/drift_trigger.py` |
| CI metric gate | `pipelines/metric_gate.py` |
| Interactive results dashboard | `reports/app.py` (Streamlit), `reports/dashboard.py` (static) |
| Architecture decisions (the "why") | `docs/adr/0001`–`0004` |
| Every headline number, with caveats | `docs/*_results.md`, `docs/model_card.md` |

**Live artifacts:**
- **Repo (main):** https://github.com/rpatel0022/ametek-ml-engineer-project
- **Portfolio site (GitHub Pages):** https://rpatel0022.github.io/ametek-ml-engineer-project/
  — _one manual enable step pending; see §9 Deployment status._
- **Interactive dashboard locally:** `make dashboard` (Streamlit) or open the committed
  `docs/dashboard.png` hero / generated `docs/dashboard.html`.

---

## 2. The data foundation (and why each source)

All training and monitoring data is **real**. This was a hard rule, kept throughout.

| Role | Source | Scale / provenance | Why this one |
|---|---|---|---|
| **Primary sensor data** (predictive maintenance) | **MetroPT-3** Air Compressor — UCI #791 | **1,516,948 rows**, 15 signals (7 analog + 8 binary), 4 real failures | Real failure-labeled industrial sensor stream — the closest public analog to the IoT power-unit telemetry the role targets |
| **Fleet-scale failure data** | **Backblaze Drive Stats** lifetime table | **418,351 drives, 24,270 real failures, 161 models** | Gives *thousands* of failures where MetroPT has four — proves the methods scale to fleet reliability |
| **Live production/monitoring feed** | **EIA Open Data API v2** (hourly US electricity demand) | Genuinely live, key via `EIA_API_KEY` | A *real* drifting signal to drive drift detection + self-heal — a static dataset can't demonstrate drift honestly |
| **Cloud target** | AWS (ECS/Fargate + S3) | Documented, not live-deployed | Standard production target; task def + cost notes in `infra/aws/` |

**The "data seam" (a key talking point):** training data (failure-labeled, historical)
and the production-monitoring feed (live, unlabeled) are deliberately *different
sources*. This mirrors reality — you almost never get live labels — and it forces the
monitoring layer to lead on **unlabeled** indicators (anomaly-score distribution, input
drift) rather than accuracy, since labels lag. See `docs/architecture.md`.

**Provenance discipline:** real data is **fetched, never committed** (`.gitignore` blocks
`*.csv` except tiny committed dashboard assets and test fixtures). The EIA key and any
secret are **env-only, never in git**.

---

## 3. The core thesis: graded on dollars, not accuracy

A missed failure ⇒ emergency truck-roll + downtime. A false alarm ⇒ a wasted inspection.
So the cost model is asymmetric: `CostModel(cost_fn=1000, cost_fp=10, cost_tp=10)` — a
missed failure hurts ~100× a false positive. From that:

- `optimal_threshold(y_true, y_score, model)` picks the **cost-minimizing cutoff**, not
  0.5.
- Every result reports **lift vs the best fixed-schedule baseline** (`periodic_schedule_cost`).

This logic lives in `src/gridsentinel/cost.py`, is unit-tested, and is the spine every
phase reports against. **If asked "why not just maximize F1/AUC?" — this is the answer.**

---

## 4. Phase-by-phase: what, why, the number, the honest caveat

This is the interview narrative. For each phase: what was built, the design choice, the
**result with its real number**, and the honest limitation (always volunteer the caveat —
it reads as senior).

### Phase 0 — Data foundation & contract
- **Built:** repo scaffold; pandera validation schema for MetroPT-3
  (`pipelines/metropt3_schema.py`); a reproducible data-quality profiler
  (`pipelines/data_quality.py` → `docs/data_quality_metropt3.md`); the EIA connector
  (`pipelines/connectors.py`).
- **Why:** validate-before-train prevents silent garbage; physical-range bounds catch bad
  telemetry at the door (and later power the API's 422 rejection).
- **Result:** schema passes on **all 1,516,948 real rows** (0 nulls, monotonic
  timestamps, binaries strictly `{0,1}`). Analog ranges tightened from placeholder to
  **spike-derived physical bounds**. EIA connector **verified live** (24 hourly demand
  rows from API v2).
- **Caveat:** physical bounds are empirical (derived from observed spikes), documented as
  such with provenance in the data-quality doc.

### Phase 1 — Supervised baselines (the first real ROI number)
- **Built:** windowed feature pipeline (`pipelines/features.py`), real failure labels from
  the report table (`pipelines/labels.py`), **temporal CV with an embargo gap**
  (`src/gridsentinel/cv.py`), RF + XGBoost with a **train-tuned, test-frozen** cost
  threshold, MLflow tracking (`pipelines/train_baseline.py`).
- **Why temporal CV + embargo:** time series leaks through random splits; an embargo gap
  between train and test prevents windows that straddle the boundary from leaking.
- **Result:** cost-tuned **XGBoost** leak detector cuts expected maintenance cost **~60%
  vs the best fixed schedule** on the held-out failure (**ROC-AUC 0.92**; ~30% mean over
  the 2 scorable folds). Details: `docs/phase1_baseline_results.md`.
- **Caveat:** only 4 failures ⇒ only 2 folds are scorable; the 60% is one strong fold, the
  ~30% is the honest mean. State both.

### Phase 2 — Unsupervised anomaly detection (the primary model)
- **Built:** Isolation Forest fit on the **failure-free baseline period** (zero labels),
  evaluated against the 4 real failures (`pipelines/anomaly.py`).
- **Why primary:** with only 4 failures, an unsupervised detector that learns "normal" and
  flags deviation generalizes better than a supervised model starved of positive examples
  — and it yields **early warning** that supervised prediction couldn't (ADR-0002).
- **Result:** **ROC-AUC 0.95, recall 0.89** at a label-free threshold, with **19–48 h
  early warning** on 3 of 4 failures. Details: `docs/phase2_anomaly_results.md`.
- **Caveat (important, volunteer it):** probability **calibration was tried and
  rejected** — isotonic/sigmoid + validation-slice thresholding all *worsened* ROI
  (precision collapse). The bottleneck is **4-failure scarcity, not calibration**. Honesty
  over a prettier-looking but worse pipeline.

### Phase 3 — Productionize (serving + CI/CD gate)
- **Built:** FastAPI service (`serving/`): `POST /predict` (window of raw readings →
  anomaly score + alert), `/health`, `/metrics`. **Pydantic validates every reading
  against the Phase-0 physical contract → bad telemetry rejected 422.** Model bundle
  (`serving/model.py`) carries pipeline + threshold + feature order + provenance.
  **Docker** (non-root, healthcheck). **CI metric gate** (`pipelines/metric_gate.py`)
  rebuilds the model on real data and **fails the build if metrics regress** below
  committed floors. Security scanning: **pip-audit** (deps) + **Trivy** (image).
- **Why:** "not prototypes" — the model claim is only real if it's served, validated, and
  guarded against regression in CI. **Train/serve share one aggregation
  (`features.aggregate_window`) → no feature skew.**
- **Result:** gate passes on real data at **0.95 / 0.39 / 0.89** (ROC-AUC / PR-AUC /
  recall) vs floors **0.90 / 0.30 / 0.75**. ADR-0003.
- **Caveat:** the gate rebuilds on the same dataset; in production you'd gate against a
  rolling fresh slice.

### Phase 4 — Observability, drift, self-healing
- **Built:** Prometheus instrumentation + `/metrics` (two-tier: *model* signals —
  prediction/alert counts, anomaly-score distribution — and *system* signals — latency,
  validation errors). **Drift detection** (`monitoring/drift.py`): **PSI + two-sample KS**,
  dependency-light and deterministic. `monitoring/eia_drift.py` runs it on the **live EIA
  feed**. **Grafana + Prometheus stack** (`docker compose up`) with an auto-provisioned
  dashboard. **Model registry** (`serving/registry.py`): file-based stages (production/
  candidate), **promote + rollback + audit trail**. **Self-heal orchestrator**
  (`monitoring/self_heal.py`): drift → retrain candidate → metric-gate → **promote only if
  it clears the gate and doesn't regress**, else keep current. On-call **runbook**
  (`docs/runbook.md`).
- **Why:** labels lag, so monitoring leads on unlabeled drift; self-heal closes the loop
  from "drift detected" to "safely promoted or rolled back" with a governance audit trail.
- **Result:** real seasonal drift detected on the live feed — **PSI 0.46, KS p 0.009**
  (PJM winter demand vs now). Self-heal **demoed end-to-end on real data** (register →
  promote → audit). ADR-0004.
- **Caveat:** the live feed (electricity demand) is a *proxy* drift signal, not the
  compressor's own telemetry — chosen because it's genuinely live and genuinely drifts.

### Phase 5 — Deep learning, fleet scale, edge, governance, docs
- **Deep learning (LSTM):** real Keras/`tensorflow-cpu` LSTM under the same temporal CV
  (`pipelines/lstm_model.py`). **Result: ROC-AUC 0.63 — it underperforms everything.**
  Honest finding: **too few failures to fit a recurrent net; confirms the bottleneck is
  data, not model class.** An MLP-over-stacked-windows baseline
  (`pipelines/sequence_model.py`) does reach **ROC-AUC 0.93 / PR-AUC 0.51 / ROI 32%** —
  competitive with XGBoost but still below the anomaly detector.
- **Fleet-scale (Backblaze):** `pipelines/backblaze.py`. **The leakage trap and how it was
  handled is a top interview story:** predicting `failed` from drive age leaks, because a
  failed drive's age *is* its time-to-failure while a survivor's age is just
  time-observed (censoring). Fix: pick a **reference time T = 2022-01-01**, keep drives
  **alive at T**, predict failure **within the next 365 days** using only features known
  at T (capacity, age-so-far, target-encoded historical model reliability). Drives that
  leave non-failed within the horizon are dropped (unknown outcome). **Result:** cohort of
  **207,557 drives / 3,052 failures (1.47%)**; **RandomForest ROC-AUC 0.730**, beating a
  model-reliability-prior-only baseline (0.678); sanity check recovers the infamous
  `st3000dm001` ~25% annualized failure rate. Details: `docs/backblaze_results.md`.
- **Edge ML (bonus):** measured size/latency/accuracy across ensemble sizes
  (`serving/benchmark.py`). **300→50 trees = 5.9× smaller, ~4× faster p99 (24→6 ms), same
  ROC-AUC (~0.95)**, recall cost 0.89→0.75. Registry supports a **lower per-stage
  threshold** for the edge stage without rebuilding.
- **Delayed-label backfill** (`monitoring/backfill.py`): when real failure reports arrive,
  re-scores stored predictions against now-known labels and logs true precision/recall/
  ROC-AUC (JSONL). Honest handling of lagging labels.
- **Governance:** model card (`docs/model_card.md`); **Google ML Test Score self-assessment
  = 4.5** (min of the four sections), every point backed by a named artifact
  (`docs/ml_test_score.md`).
- **AWS:** Fargate `task-definition.json` (port 8000, `/health` check, `EIA_API_KEY` via
  SSM secret, CloudWatch logs) + deploy guide + cost note (**~$35–40/mo, ~$18 scaled
  down**). Documented target, not applied from the repo.

---

## 5. Headline numbers (single source of truth)

Copy these into interview prep. Every one is from real data and traceable to a file.

| Metric | Value | Source |
|---|---|---|
| MetroPT-3 rows validated | 1,516,948 | `docs/data_quality_metropt3.md` |
| Anomaly detector ROC-AUC / recall | **0.95 / 0.89** | `docs/phase2_anomaly_results.md` |
| Early warning | **19–48 h** on 3 of 4 failures | same |
| Supervised XGBoost ROC-AUC | 0.92 | `docs/phase1_baseline_results.md` |
| Cost cut vs fixed schedule | **~60%** (held-out failure), ~30% mean | same |
| MLP sequence ROC-AUC / PR-AUC / ROI | 0.93 / 0.51 / 32% | `docs/sequence_model_results.md` |
| LSTM ROC-AUC (honest underperform) | 0.63 | `docs/lstm_results.md` |
| Backblaze dataset | 418,351 drives / 24,270 failures / 161 models | `docs/backblaze_results.md` |
| Backblaze model cohort + ROC-AUC | 207,557 drives / 3,052 failures → **0.730** | same |
| Metric gate (actual vs floor) | 0.95/0.39/0.89 vs 0.90/0.30/0.75 | `pipelines/metric_gate.py` |
| Live-feed drift | PSI 0.46, KS p 0.009 | `monitoring/eia_drift.py` |
| Edge tradeoff | 5.9× smaller, 4× faster p99 (24→6 ms), same AUC | `docs/edge_benchmark.md` |
| Serving p99 latency | **31 ms** (< 50 ms SLO) | `docs/load_test_results.md` |
| Google ML Test Score | **4.5** | `docs/ml_test_score.md` |
| Test suite | **143 test functions / 27 modules** (CI is the green source of truth) | `tests/` |

---

## 6. Key decisions (ADRs) — the "why" behind the forks

- **ADR-0001** — Dataset (MetroPT-3 + Backblaze), live feed (EIA), cloud (AWS). Includes a
  revisit trigger (not hit: the 15-feature schema matched the real data exactly).
- **ADR-0002** — **Anomaly detection as the primary model** (over supervised), and
  **calibration rejected** with evidence.
- **ADR-0003** — Serving + file-based registry stack (why FastAPI + a file registry, not a
  heavyweight platform, at this scale).
- **ADR-0004** — In-house PSI/KS drift (dependency-light, deterministic, testable) over a
  drift framework.

---

## 7. War stories (bugs found & fixed — strong interview material)

These show engineering judgment, not just model-building. Each is a real fix in the repo.

1. **Thread oversubscription under load.** Load test surfaced p99 ~1245 ms. Root cause:
   `n_jobs=-1` in the Isolation Forest oversubscribed threads under concurrency. Fixed by
   pinning `iforest.n_jobs=1` in `serving/model.load_bundle`. Then discovered scoring is
   **GIL-bound (~40 rps/process)** → documented a **horizontal-scaling** capacity model
   instead of pretending one process scales. Final **p99 31 ms**.
2. **Censoring leakage in Backblaze** (see §4 Phase 5) — reframed the whole task to a
   reference-time + horizon cohort to make age measurable identically for failed and
   surviving drives.
3. **Calibration that looked good but wasn't** — measured it, it worsened ROI, rejected it
   and documented why (Phase 2).
4. **Schema edge case:** Backblaze uses `capacity = 0` as an "unknown capacity" marker, so
   the contract uses `ge(0)` not `gt(0)` (`pipelines/backblaze_schema.py`).
5. **Environment blockers turned into wins:** Backblaze's own host + Kaggle are
   network-blocked here, so the lifetime CSV was pulled from a **reachable GitHub mirror**
   (zero user download). PyTorch's CPU wheel index is proxy-blocked, so the LSTM uses
   **`tensorflow-cpu`** from default PyPI instead.
6. **MLflow 3.x** deprecated the file store → defaulted tracking to `sqlite:///mlflow.db`.

---

## 8. Honest limitations (say these before you're asked)

- **4 failures in MetroPT-3** is the real ceiling on the supervised/DL side — it's why the
  anomaly detector wins and why the LSTM underperforms. The Backblaze model exists
  specifically to show the methods hold at thousands of failures.
- **Backblaze is lifetime/reliability data, not SMART telemetry** — it powers a
  fleet-reliability (survival) model that *complements* MetroPT's sensor-based PdM, it
  doesn't replace it. SMART-at-scale needs a ~1 GB download that's environment-blocked.
- **The live drift feed (EIA demand) is a proxy** for equipment drift, chosen because it
  genuinely drifts and is genuinely live.
- **AWS is a documented target, not a live deployment.**
- **Optional Phase 6 (GenAI/RAG over UPS manuals → work-order)** is scoped but not built —
  no LLM key / SDK in-sandbox, so retrieval-only scaffolding was deliberately *not*
  shipped.

---

## 9. Deployment status (GitHub + Pages)

- **Code is on `main`.** The full system + portfolio site merged via **PR #2**
  (`claude/refine-plan-md-6swc1n` → `main`, merge commit `7c3cc65`). Direct pushes to
  `main` are policy-blocked, so the PR-and-merge path is the sanctioned route.
- **GitHub Pages — one manual step pending.** The deploy workflow
  (`.github/workflows/pages.yml`) ran on the merge but **failed at the
  `actions/configure-pages` step**: `Create Pages site failed — Resource not accessible by
  integration`. The workflow's `GITHUB_TOKEN` cannot *create* the Pages site
  programmatically.
  - **To finish (do once, in the GitHub UI):** repo **Settings → Pages → Build and
    deployment → Source: "GitHub Actions"**, then re-run the workflow (Actions → "Deploy
    portfolio to GitHub Pages" → Re-run).
  - **Plan caveat:** the repo is **private**. GitHub Pages on a private repo requires a
    paid plan (Pro/Team/Enterprise). On a free plan, either **make the repo public** or
    upgrade for the site to actually serve. The committed `site/` builds fine either way;
    this is purely a hosting-enablement constraint.
  - **Expected live URL once enabled:** https://rpatel0022.github.io/ametek-ml-engineer-project/

---

## 10. How to run / reproduce

```bash
make install                 # dev + pipelines + modeling + serving extras
# real MetroPT-3 is fetched (UCI #791), never committed; point DATA at the CSV
make data-quality            # validate + profile the real data
make anomaly                 # train + evaluate the anomaly detector (MLflow)
make gate                    # metric gate: fail if the model regresses
make artifact serve          # build the bundle + run the API (localhost:8000/docs)
make backblaze               # fleet-reliability model (pulls CSV from GitHub mirror)
make lstm                    # real LSTM (needs the dl extra)
make dashboard               # interactive Streamlit results board
make docker                  # API + Prometheus + Grafana stack
```

Operate it (the self-healing surface):

```bash
make selfheal                # one retrain → gate → promote/keep cycle
make retrain-if-drift        # retrain only if the live EIA feed has drifted
make status                  # live model, in-force threshold, audit trail
make loadtest edge           # p99 SLO load test · edge size/latency benchmark
```

`make help` lists every target. Runs are tracked in MLflow (`sqlite:///mlflow.db`).
Set `EIA_API_KEY` in the environment for the live-feed jobs (never commit it).

---

## 11. Likely interview questions → where the answer is

| Question | Answer / file |
|---|---|
| "Why not just maximize accuracy/AUC?" | Asymmetric cost model + dollar lift — §3, `src/gridsentinel/cost.py` |
| "How did you prevent leakage?" | Temporal CV + embargo (`cv.py`); Backblaze reference-time cohort (§4) |
| "Why anomaly detection over supervised?" | 4-failure scarcity + early warning — ADR-0002 |
| "How do you know it won't silently rot?" | CI metric gate + drift + self-heal + backfill — §4 Phase 3/4 |
| "What did you do when a model didn't work?" | LSTM 0.63, calibration rejected — reported honestly, §4 / §7 |
| "How does it scale?" | GIL-bound ~40 rps/process → horizontal scaling; edge benchmark — §7, `docs/` |
| "Is it really production-ready?" | FastAPI + pydantic 422 + Docker + registry + audit — §4 Phase 3 |
| "Show me the customer value." | ~60% cost cut vs schedule, ROI threshold — §3, Phase 1 |

---

## 12. Resume notes (for a future build session)

- **Branch:** `claude/refine-plan-md-6swc1n` (merged to `main` via PR #2).
- **Setup:** `pip install -e ".[dev,pipelines]"` then `pytest` (a fresh container needs
  deps installed first — collection fails with `No module named numpy` until then; CI on
  GitHub is the authoritative green signal).
- **Open follow-ons (all optional / environment-gated):** finish Pages enablement (§9);
  Backblaze SMART-telemetry at scale (~1 GB download, blocked here); optional Phase 6
  GenAI/RAG bridge (needs an LLM key).
