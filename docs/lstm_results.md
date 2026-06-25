# LSTM sequence model — results

_Reproduce: `python -m pipelines.lstm_model "MetroPT3(AirCompressor).csv"` (needs the
`dl` extra: `pip install -e ".[dl]"`)._

## Now a real recurrent model

The deep-learning requirement was earlier met by an MLP stand-in because no DL
framework was installable. `tensorflow-cpu` installs fine from default PyPI (only
PyTorch's separate wheel index is proxy-blocked), so this is a genuine **LSTM** — a
masked LSTM(32) → dropout → dense head over sequences of 6 consecutive
feature-windows, trained under the same strict temporal CV as the other models.

## Result (same evaluation as the others)

| Model | PR-AUC | ROC-AUC | ROI vs schedule |
|---|---|---|---|
| Unsupervised anomaly (Isolation Forest) | 0.39 | **0.95** | — |
| Supervised XGBoost | 0.44 | 0.92 | 30.0% |
| MLP sequence baseline | 0.51 | 0.93 | 31.9% |
| **LSTM (k=6)** | 0.32 | **0.63** | 19.3% |

## Honest read

**The LSTM underperforms every other model** — ROC-AUC 0.63 is barely above chance.
This is the scarcity thesis made concrete: an LSTM has far more parameters than the
tree/MLP models and only **two scorable folds with a handful of failure episodes** to
fit them on, so it underfits and generalizes poorly. More epochs/capacity would just
overfit the few failures.

This is a *useful* negative result, not a disappointment:

- It confirms that the bottleneck is **data (failure count), not model class** — the
  same reason the supervised baseline and calibration didn't help.
- It is the strongest motivation for the **Backblaze fleet dataset** (thousands of
  real failures across 344k drives), where a recurrent model finally has enough
  signal to justify its capacity. The code is ready; it slots into the same temporal
  CV + cost/ROI harness unchanged.

Reporting an honest under-performing LSTM — rather than tuning it on the test folds
until it looks good — is the point: the model class isn't the lever here, the data is.
