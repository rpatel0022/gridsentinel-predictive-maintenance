"""Train the anomaly detector on real MetroPT-3 data and save a serving bundle.

Run once (offline) to produce the artifact the API loads::

    python -m serving.build_artifact "MetroPT3(AirCompressor).csv"

The artifact is gitignored (like all data/model outputs) — it is rebuilt from the
real data, never committed. This mirrors how a CI/CD pipeline would build and
register the model before deployment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd
from pipelines.anomaly import anomaly_score, build_detector, evaluate, normal_training_mask
from pipelines.features import build_windowed_features, feature_columns

from serving.model import ModelBundle, save_bundle


def build(csv_path: str, *, freq: str = "10min", contamination: float = 0.02) -> ModelBundle:
    """Fit on the failure-free period, set the operating threshold, bundle it."""
    feats = build_windowed_features(pd.read_csv(csv_path), freq=freq)
    cols = feature_columns(feats)
    X = feats[cols].to_numpy()

    train_mask = normal_training_mask(feats["window_start"])
    detector = build_detector(contamination=contamination)
    detector.fit(X[train_mask])

    # Operating threshold: 99th percentile of anomaly score over healthy operation.
    train_scores = anomaly_score(detector, X[train_mask])
    threshold = float(np.quantile(train_scores, 0.99))

    metrics = evaluate(feats, contamination=contamination)
    version = dt.datetime.now(dt.timezone.utc).strftime("iforest-%Y%m%d-%H%M%S")
    return ModelBundle(
        detector=detector,
        threshold=threshold,
        features=cols,
        version=version,
        trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        metrics={k: metrics[k] for k in ("roc_auc", "pr_auc", "recall", "precision")},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the GridSentinel serving artifact")
    parser.add_argument("csv", help="path to MetroPT3(AirCompressor).csv")
    parser.add_argument("--out", default=None, help="artifact path (default: MODEL_PATH)")
    parser.add_argument("--freq", default="10min")
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args(argv)

    bundle = build(args.csv, freq=args.freq, contamination=args.contamination)
    path = save_bundle(bundle, args.out) if args.out else save_bundle(bundle)
    print(f"saved {bundle.version} -> {path}")
    print(f"  threshold={bundle.threshold:.4f}  metrics={bundle.metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
