# Neural sequence baseline — results

_Reproduce: `python -m pipelines.sequence_model "MetroPT3(AirCompressor).csv"`._

## What this is (and isn't)

The plan calls for an **LSTM / Temporal-CNN**. A true recurrent/convolutional model
needs a deep-learning framework (PyTorch/TF), which **isn't installable in this
sandbox** — the CPU wheel index is proxy-blocked. So this is the honest stand-in: an
**MLP** (`sklearn.neural_network.MLPClassifier`) that consumes temporal context by
stacking the previous `k=6` feature-windows (1 hour of history) into one input,
evaluated under the **same strict temporal CV** as every other model.

## Result (vs the other models, same evaluation)

| Model | PR-AUC | ROC-AUC | ROI vs schedule |
|---|---|---|---|
| Unsupervised anomaly (Isolation Forest) | 0.39 | **0.95** | — (label-free) |
| Supervised XGBoost | 0.44 | 0.92 | 30.0% |
| **MLP sequence baseline (k=6)** | **0.51** | 0.93 | **31.9%** |

Honest read: the neural baseline is **competitive with the GBM** (and has the best
PR-AUC of the supervised models), so temporal context helps — but it **does not beat
the unsupervised anomaly detector** on ROC-AUC. Still the same two-scorable-fold
limit from four failures, so treat these as indicative.

## Deferred: the real LSTM / TCN

A genuine recurrent/convolutional sequence model is **deferred to an environment with
a DL framework** (or GPU). It would slot in exactly here — same features, same
temporal CV, same cost/ROI comparison — and is the natural thing to revisit once the
Backblaze fleet data lifts the failure count. Marking this honestly rather than
shipping an LSTM-shaped wrapper around a model I couldn't actually run.
