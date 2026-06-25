# Edge benchmark — size / latency / accuracy tradeoff

_Reproduce: `python -m serving.benchmark "MetroPT3(AirCompressor).csv"`._

Telular ships edge devices, so the real question isn't "can it run on the edge?" but
"what does a smaller, faster model *cost*?". Isolation Forests don't int8-quantize like
neural nets; the equivalent lever is the **ensemble size** (`n_estimators`). Measured on
the real MetroPT-3 data (single-window inference, the serving path):

| Trees | Artifact size | p50 latency | p99 latency | ROC-AUC | Recall |
|---|---|---|---|---|---|
| **300** (cloud default) | 3189 KB | 17.5 ms | 23.6 ms | 0.952 | 0.89 |
| 100 | 1034 KB | 6.3 ms | 8.8 ms | 0.937 | 0.74 |
| **50** (edge) | 539 KB | 3.4 ms | 6.0 ms | 0.948 | 0.75 |

## The tradeoff, in numbers

Going **300 → 50 trees**: the artifact shrinks **5.9×** (3189 → 539 KB) and p99 latency
drops **~4×** (23.6 → 6.0 ms), while **ranking quality barely moves** (ROC-AUC 0.952 →
0.948). The cost is **operating-point recall**: at the fixed 99th-percentile threshold it
falls from 0.89 to ~0.75 — the smaller ensemble's scores are noisier near the cutoff, so
some failure windows slip under it.

## Cloud vs. edge recommendation

- **Cloud (300 trees):** default. 3 MB and ~24 ms p99 are nothing server-side; take the
  full recall.
- **Edge (50 trees):** when the model runs on a constrained gateway, ship the 50-tree
  model — **6× smaller, 4× faster, essentially the same ROC-AUC**. Recover the recall by
  **lowering the alert threshold** for the edge model (it ranks just as well; only the
  fixed cutoff is pessimistic), accepting more false alarms — which the asymmetric cost
  model already says is the cheap kind of error.
- p99 inference is single-digit ms on the edge model, comfortably inside any reasonable
  latency SLO; the HTTP/serving overhead, not the model, dominates end-to-end latency.

## Honest caveats

- Recall is measured at a *fixed* threshold; a per-model tuned threshold narrows the gap.
  This is now supported — the registry carries an optional **per-stage threshold
  override** (`ModelRegistry.set_threshold`), so an edge stage can run a lower alert
  threshold than the cloud default without rebuilding the model.
- Only four real failures, so recall numbers are coarse; treat the size/latency reduction
  (which is data-independent) as the robust result and the accuracy delta as indicative.
