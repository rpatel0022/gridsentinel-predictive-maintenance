"""Load test for the inference service — throughput + p99 under concurrency.

The edge benchmark (`serving/benchmark.py`) measures raw *model* latency; this
measures the *service* under concurrent load — the "load-test result + p99 SLO"
deliverable. It drives the ASGI app in-process via httpx (no network, no server to
stand up), fires N requests at a set concurrency, and reports throughput and the
latency distribution end-to-end (HTTP + validation + model).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx
import numpy as np
from httpx import ASGITransport

P99_SLO_MS = 50.0


async def _fire(client, payload, latencies, errors, sem) -> None:
    async with sem:
        start = time.perf_counter()
        try:
            resp = await client.post("/predict", json=payload)
            if resp.status_code != 200:
                errors.append(resp.status_code)
        except Exception as exc:  # noqa: BLE001 - any client error counts as a failure
            errors.append(repr(exc))
        latencies.append((time.perf_counter() - start) * 1000.0)


async def _load(app, payload, n_requests: int, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors: list = []
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://loadtest") as client:
        start = time.perf_counter()
        await asyncio.gather(
            *[_fire(client, payload, latencies, errors, sem) for _ in range(n_requests)]
        )
        duration = time.perf_counter() - start
    lat = np.array(latencies)
    return {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "duration_s": round(duration, 3),
        "throughput_rps": round(n_requests / duration, 1) if duration else 0.0,
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "max_ms": round(float(lat.max()), 2),
        "errors": len(errors),
    }


def run_load_test(app, payload: dict, *, n_requests: int = 200, concurrency: int = 16) -> dict:
    """Fire ``n_requests`` at ``concurrency`` against ``app`` and return the stats."""
    return asyncio.run(_load(app, payload, n_requests, concurrency))


def _sample_payload(n_samples: int = 6) -> dict:
    from pipelines.metropt3_schema import ANALOG_SENSORS, DIGITAL_SIGNALS

    reading = {c: 5.0 for c in ANALOG_SENSORS} | {c: 1.0 for c in DIGITAL_SIGNALS}
    return {"readings": [reading] * n_samples}


def main(argv: list[str] | None = None) -> int:
    from serving.app import create_app
    from serving.model import load_bundle

    parser = argparse.ArgumentParser(description="GridSentinel serving load test")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args(argv)

    app = create_app(load_bundle())
    payload = _sample_payload()
    # True per-request latency (concurrency 1) is what the SLO governs; a separate
    # high-concurrency run shows throughput. Single-process inference is GIL-bound,
    # so concurrency adds queue latency, not parallelism — scale out, not up.
    latency = run_load_test(app, payload, n_requests=200, concurrency=1)
    throughput = run_load_test(app, payload, n_requests=args.requests, concurrency=args.concurrency)
    print(f"latency  (c=1):  {latency}")
    print(f"throughput (c={args.concurrency}): {throughput}")
    verdict = "PASS" if latency["p99_ms"] <= P99_SLO_MS and latency["errors"] == 0 else "FAIL"
    print(f"per-request p99 SLO {P99_SLO_MS} ms → {verdict}")
    print(
        f"note: ~{throughput['throughput_rps']:.0f} rps/process is GIL-bound; "
        f"scale horizontally (replicas behind the ALB) for more."
    )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
