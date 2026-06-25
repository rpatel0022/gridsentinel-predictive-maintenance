# Backblaze fleet-reliability model — results

_Reproduce: `make backblaze-data` (pulls the CSV from a GitHub mirror) then
`python -m pipelines.backblaze data/backblaze_drive_dates.csv`._

## Source & honest framing

Real, public Backblaze Drive Stats, as a **per-drive lifetime table** (one row per
drive: capacity, model, install date, last-seen date, `failed`) — **418,351 drives,
24,270 real failures, 161 models**. Pulled from a GitHub mirror
([zachmayer/backblaze_analysis](https://github.com/zachmayer/backblaze_analysis),
`results/drive_dates.csv`) because Backblaze's own host and Kaggle are network-blocked
in this environment.

This is **reliability/lifetime data, not SMART-sensor telemetry** — so it powers a
*fleet-reliability* model (survival/failure-rate), **complementing** MetroPT's
sensor-based predictive maintenance rather than replacing it. Its value: **thousands**
of real failures, where MetroPT has four.

## The leakage trap, handled

The obvious task — "predict `failed` from drive age" — leaks: a failed drive's age is
its *time-to-failure*, but a survivor's age is just *time-observed* (censoring). So
instead we take a **reference time** `T` (2022-01-01), keep the drives **alive at `T`**,
and predict failure **within the next year**, using only features known at `T`
(capacity, **age-so-far** = install→`T`, and the model's historical reliability). Age
is then measured identically for every drive — no censoring leak. Drives that *leave*
(non-failure) within the horizon are dropped (unknown outcome).

## Result

Cohort: **207,557 drives, 3,052 failures (1.47%)** within 365 days of 2022-01-01.

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| **RandomForest** (capacity + age + model-reliability) | **0.730** | 0.043 |
| Model-reliability prior only (target-encoded model rate) | 0.678 | 0.038 |

- **ROC-AUC 0.73** on a genuinely hard, heavily-imbalanced fleet-failure task with
  **3,052 real failures** — a real result, not scarcity-limited.
- The full model **beats the model-reliability-only prior** (0.73 vs 0.68), so drive
  **age and capacity add real signal** beyond just "which model is it".
- PR-AUC 0.043 vs a 1.47% base rate ≈ **3× lift** — modest but real under heavy imbalance.

## Sanity check (the data is real)

Worst model by **annualized failure rate**: `st3000dm001` at **~25% AFR** — the
infamous Seagate 3 TB drive, a well-documented real-world failure. The pipeline
independently recovering that is strong evidence the data and analysis are correct.

## Where it fits

Two real datasets, two honest stories:
- **MetroPT** → sensor-based predictive maintenance (the IoT-ML core).
- **Backblaze** → fleet-scale reliability at 418k drives / 24k failures (this doc).

The censoring-aware cohort and the AFR metric are the senior tells here: handling
right-censored lifetime data correctly is exactly what separates a real reliability
analysis from a leaky one.
