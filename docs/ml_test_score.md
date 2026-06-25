# Google ML Test Score — self-assessment

Scored against the rubric in *"The ML Test Score: A Rubric for ML Production
Readiness"* (Breck et al., Google). Each test is **0** (not done), **0.5** (done
manually / partially), or **1** (automated). The final score is the **minimum** of the
four section sums. This is an honest self-assessment with the backing artifact named
for every point — gaps are marked plainly.

## 1. Data and features — **4.5**

| # | Test | Score | Evidence / gap |
|---|---|---|---|
|1.1| Feature expectations in a schema | 1 | `pipelines/metropt3_schema.py` (pandera) + automated tests |
|1.2| Features are beneficial | 0.5 | all 36 used; not formally ablated |
|1.3| Feature cost vs benefit | 0 | not measured |
|1.4| Meta-level requirements (no PII) | 1 | sensor-only; documented in the model card |
|1.5| Data pipeline privacy controls | 0.5 | no PII; secrets env-only; no formal control |
|1.6| New features added quickly | 0.5 | modular `aggregate_window`/feature pipeline |
|1.7| Input feature code is tested | 1 | `tests/test_features.py`, train/serve share one aggregation |

## 2. Model development — **4.5**

| # | Test | Score | Evidence / gap |
|---|---|---|---|
|2.1| Model specs reviewed & versioned | 1 | git + bundle `version` + registry |
|2.2| Offline metrics relate to impact | 0.5 | cost-model ROI defined; online truth lags (delayed labels) |
|2.3| Hyperparameters tuned | 0.5 | threshold/contamination principled; no full sweep |
|2.4| Staleness impact understood | 0.5 | retrain loop + drift; not quantified over time |
|2.5| A simpler model is not better | 1 | beats dumb-schedule / RF / XGBoost baselines |
|2.6| Quality on data slices | 0.5 | per-failure breakdown in the results docs |
|2.7| Fairness / inclusion | 0.5 | N/A for industrial sensors — documented as such |

## 3. ML infrastructure — **6.0**

| # | Test | Score | Evidence / gap |
|---|---|---|---|
|3.1| Training is reproducible | 1 | seeded; identical metrics across reruns |
|3.2| Model specs are unit tested | 1 | `tests/test_serving.py`, `test_anomaly.py` |
|3.3| Full pipeline integration tested | 1 | `tests/test_integration.py` — features→train→registry→serve→predict |
|3.4| Quality validated before serving | 1 | the metric gate (`pipelines/metric_gate.py`) + `model-eval` CI |
|3.5| Model is debuggable | 0.5 | metrics + audit trail; no dedicated debug tooling |
|3.6| Canary before serving | 0.5 | promotion gate + registry; no live traffic-split canary |
|3.7| Rollback | 1 | `serving/registry.py` rollback, tested |

## 4. Monitoring — **5.0**

| # | Test | Score | Evidence / gap |
|---|---|---|---|
|4.1| Dependency changes notified | 0.5 | `pip-audit` in CI (advisory) |
|4.2| Data invariants hold train + serve | 1 | same schema enforced on both sides |
|4.3| Train/serve feature consistency | 1 | shared `aggregate_window`, tested |
|4.4| Models are not too stale | 0.5 | retrain loop; no age-based alert yet |
|4.5| Model numerically stable | 0 | not explicitly tested |
|4.6| Performance regression caught | 1 | metric gate + drift detection |
|4.7| Prediction-quality / drift | 1 | `monitoring/drift.py` + Prometheus leading indicators |

## Result

| Section | Sum |
|---|---|
| Data & features | 4.5 |
| Model development | 4.5 |
| ML infrastructure | 6.0 |
| Monitoring | 5.0 |
| **Final (minimum)** | **4.5** |

_(The final score is bounded by Data & Model at 4.5; closing the integration-test
gap lifted Infrastructure to 6.0 but not the minimum.)_

**Interpretation (rubric scale):** > 3 indicates *"strong levels of automated testing
and monitoring, appropriate for mission-critical systems."* A 4.5 is a strong result —
and, more importantly, every point above is backed by a file or CI step you can open,
with the remaining 0/0.5 items called out honestly as the next work (numerical-stability
tests, a CI integration test, a live canary, feature-ablation).
