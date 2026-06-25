"""Latency / size / accuracy benchmark — the cloud-vs-edge tradeoff with numbers.

Telular ships edge devices, so "export to ONNX" isn't enough; the question is what
a *smaller, faster* model costs in accuracy. Isolation Forests don't int8-quantize
like neural nets, but the analogous lever is the **ensemble size**: fewer trees → a
smaller artifact and lower inference latency, at some cost to score stability. This
benchmarks that lever on the real data so the tradeoff is measured, not asserted.

Measures, per ``n_estimators``: artifact size, single-window inference latency
(p50/p99 — the serving path), and ROC-AUC / recall against the real failures.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time

import numpy as np
import pandas as pd
from pipelines.anomaly import anomaly_score, build_detector, evaluate, normal_training_mask
from pipelines.features import WINDOW_START, build_windowed_features, feature_columns


def artifact_size_kb(detector) -> float:
    """Serialized model size in KB (what ships to the device)."""
    return len(pickle.dumps(detector)) / 1024.0


def inference_latency_ms(detector, x_row: np.ndarray, *, n_iters: int = 2000) -> dict:
    """Per-window inference latency over ``n_iters`` single-row scorings."""
    for _ in range(50):  # warm up
        anomaly_score(detector, x_row)
    times = np.empty(n_iters)
    for i in range(n_iters):
        start = time.perf_counter()
        anomaly_score(detector, x_row)
        times[i] = (time.perf_counter() - start) * 1000.0
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p99_ms": float(np.percentile(times, 99)),
        "mean_ms": float(times.mean()),
    }


def benchmark(feats: pd.DataFrame, n_estimators_list=(300, 100, 50)) -> list[dict]:
    """Size/latency/accuracy for each ensemble size."""
    cols = feature_columns(feats)
    X = feats[cols].to_numpy()
    mask = normal_training_mask(feats[WINDOW_START])
    x_row = X[mask][:1]

    rows: list[dict] = []
    for n in n_estimators_list:
        ev = evaluate(feats, n_estimators=n)
        detector = build_detector(n_estimators=n).fit(X[mask])
        rows.append(
            {
                "n_estimators": n,
                "size_kb": round(artifact_size_kb(detector), 1),
                **{k: round(v, 3) for k, v in inference_latency_ms(detector, x_row).items()},
                "roc_auc": round(ev["roc_auc"], 3),
                "recall": round(ev["recall"], 3),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GridSentinel edge/latency benchmark")
    parser.add_argument("csv", help="path to MetroPT3(AirCompressor).csv")
    parser.add_argument("--freq", default="10min")
    args = parser.parse_args(argv)

    feats = build_windowed_features(pd.read_csv(args.csv), freq=args.freq)
    rows = benchmark(feats)
    header = f"{'trees':>6} {'size_kb':>8} {'p50_ms':>8} {'p99_ms':>8} {'roc_auc':>8} {'recall':>7}"
    print(header)
    for r in rows:
        print(
            f"{r['n_estimators']:>6} {r['size_kb']:>8} {r['p50_ms']:>8} "
            f"{r['p99_ms']:>8} {r['roc_auc']:>8} {r['recall']:>7}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
