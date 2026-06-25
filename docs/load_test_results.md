# Load test — results & findings

_Reproduce: `MODEL_PATH=models/anomaly_detector.joblib python -m serving.load_test`.
Drives the ASGI app in-process via httpx (HTTP + validation + model end-to-end)._

## Result

| Scenario | Throughput | p50 | p99 | Errors |
|---|---|---|---|---|
| **Per-request (concurrency 1)** | ~44 rps | 22 ms | **31 ms** | 0 |
| Under load (concurrency 16) | ~32 rps | 492 ms | 628 ms | 0 |

**Per-request p99 = 31 ms, inside the 50 ms SLO.** ✅

## Two findings the load test earned

1. **`n_jobs=-1` oversubscribes the CPU under concurrency.** The Isolation Forest was
   trained with all-core parallelism; left on at serving time, every concurrent
   request spawned all-core work and the threads thrashed. **Fix:** `load_bundle`
   forces `n_jobs=1` — inference is single-threaded *per request*; concurrency belongs
   at the request level. (`serving/model.py`.)

2. **Single-process inference is GIL-bound, so throughput ≈ 1 / latency (~40 rps),
   regardless of concurrency.** Piling on concurrency doesn't parallelize CPU-bound
   scoring — it just queues, inflating tail latency (p99 30 ms → 628 ms at c=16). This
   is expected and the right lesson: **the SLO governs per-request latency; capacity
   comes from horizontal scaling**, not in-process concurrency.

## Capacity & SLO recommendation

- **SLO:** per-request p99 ≤ 50 ms — met at 31 ms.
- **Capacity:** ~40 rps per Fargate task. Size the ECS service to
  `ceil(peak_rps / 40)` tasks behind the ALB and autoscale on CPU / request count.
- For an order-of-magnitude more throughput per task, the model would need a
  GIL-releasing runtime (e.g. ONNX Runtime) — a future optimization, measured the
  same way.
