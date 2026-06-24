"""Unsupervised anomaly detection for MetroPT-3 (Phase 2).

The Phase 1 supervised baseline hit a wall the data made unavoidable: with only
four real failures, a model that *needs* failure labels to train has almost nothing
to learn from, and long-horizon prediction is unreliable (see
``docs/phase1_baseline_results.md``). Anomaly detection sidesteps that entirely — it
learns what *normal* compressor operation looks like (abundant) and flags
deviations, so it needs **zero failure labels to train**.

The detector is an Isolation Forest over the windowed features, fit on a verified
**failure-free commissioning period** (everything before the first reported
failure — realistic: you establish a healthy baseline during known-good operation).
It is then scored on the later, failure-containing period. Because the fit uses no
labels, there is no label leakage; the real failures are used only to *evaluate*.

Two things it delivers that Phase 1 could not:
* **Real lead time** — the anomaly score rises *hours to days before* a failure,
  giving the early warning the supervised predict-ahead framing couldn't.
* **Robustness to label scarcity** — performance doesn't depend on how many
  failures we have, only on a clean baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.features import WINDOW_START, feature_columns
from pipelines.labels import FAILURE_EVENTS, FailureEvent, make_labels


def normal_training_mask(
    window_starts: pd.Series,
    events: tuple[FailureEvent, ...] = FAILURE_EVENTS,
) -> np.ndarray:
    """Boolean mask of windows in the failure-free period (before the first failure).

    This is the verified-healthy baseline the detector learns from. Using the
    report table only to *find* the clean period — not to label training rows —
    keeps the fit fully unsupervised.
    """
    ts = pd.to_datetime(pd.Series(window_starts).reset_index(drop=True))
    first_failure = min(ev.start for ev in events)
    return (ts < first_failure).to_numpy()


def build_detector(contamination: float = 0.02, random_state: int = 42) -> Pipeline:
    """Standardise then Isolation-Forest. ``contamination`` only sets the alert
    offset for ``predict``; the anomaly *ranking* (``score_samples``) is unaffected.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=300,
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def anomaly_score(detector: Pipeline, X: np.ndarray) -> np.ndarray:
    """Per-row anomaly score; higher = more anomalous (sign-flipped decision fn)."""
    return -detector.named_steps["iforest"].decision_function(
        detector.named_steps["scale"].transform(X)
    )


def lead_times(
    window_starts: pd.Series,
    scores: np.ndarray,
    threshold: float,
    *,
    lookback_hours: float = 48.0,
    events: tuple[FailureEvent, ...] = FAILURE_EVENTS,
) -> list[dict]:
    """How early the score first crosses ``threshold`` before each failure.

    Returns one dict per event with ``lead_hours`` (None if not pre-alerted) and
    ``flagged_during`` (whether any in-failure window crossed the threshold).
    """
    ts = pd.to_datetime(pd.Series(window_starts).reset_index(drop=True))
    out: list[dict] = []
    for ev in events:
        pre = (ts >= ev.start - pd.Timedelta(hours=lookback_hours)) & (ts < ev.start)
        pre_hits = np.where(pre.to_numpy() & (scores >= threshold))[0]
        during = (ts >= ev.start) & (ts <= ev.end)
        during_hit = bool(np.any(during.to_numpy() & (scores >= threshold)))
        lead = None
        if len(pre_hits):
            lead = float((ev.start - ts.iloc[pre_hits[0]]).total_seconds() / 3600.0)
        out.append(
            {
                "failure": str(ev.start.date()),
                "lead_hours": lead,
                "flagged_during": during_hit,
            }
        )
    return out


def evaluate(
    feats: pd.DataFrame,
    *,
    contamination: float = 0.02,
    warn_hours: float = 2.0,
    normal_quantile: float = 0.99,
) -> dict:
    """Fit on the failure-free period, score the rest, evaluate vs real failures.

    Returns a dict of ranking metrics (ROC-AUC, PR-AUC), the operating point at the
    ``normal_quantile`` threshold (recall/precision), and per-failure lead times.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    cols = feature_columns(feats)
    X = feats[cols].to_numpy()
    y = make_labels(feats[WINDOW_START], warn_hours=warn_hours, label_active=True).to_numpy()

    train_mask = normal_training_mask(feats[WINDOW_START])
    test_mask = ~train_mask

    detector = build_detector(contamination=contamination)
    detector.fit(X[train_mask])
    scores = anomaly_score(detector, X)

    yte, ste = y[test_mask], scores[test_mask]
    # Label-free operating threshold: the normal_quantile of scores on the clean
    # training period (e.g. the 99th percentile of healthy operation).
    threshold = float(np.quantile(scores[train_mask], normal_quantile))
    yhat = (ste >= threshold).astype(int)
    tp = int(((yhat == 1) & (yte == 1)).sum())
    fp = int(((yhat == 1) & (yte == 0)).sum())
    fn = int(((yhat == 0) & (yte == 1)).sum())

    return {
        "train_windows": int(train_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "test_positives": int(yte.sum()),
        "roc_auc": float(roc_auc_score(yte, ste)),
        "pr_auc": float(average_precision_score(yte, ste)),
        "threshold": threshold,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "lead_times": lead_times(feats[WINDOW_START], scores, threshold),
    }


def run(csv_path: str, *, freq: str = "10min", contamination: float = 0.02) -> dict:
    """Build features, evaluate the detector, log to MLflow, and print a summary."""
    import os

    import mlflow

    from pipelines.features import build_windowed_features

    feats = build_windowed_features(pd.read_csv(csv_path), freq=freq)
    result = evaluate(feats, contamination=contamination)

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("gridsentinel-phase2-anomaly")
    with mlflow.start_run(run_name=f"isolation-forest-{freq}"):
        mlflow.log_params(
            {"model": "isolation_forest", "freq": freq, "contamination": contamination}
        )
        mlflow.log_metrics({k: result[k] for k in ("roc_auc", "pr_auc", "recall", "precision")})

    print(
        f"train(normal)={result['train_windows']:,}  test={result['test_windows']:,}  "
        f"test_positives={result['test_positives']}"
    )
    print(
        f"ROC-AUC={result['roc_auc']:.3f}  PR-AUC={result['pr_auc']:.3f}  "
        f"recall={result['recall']:.2f}  precision={result['precision']:.2f} "
        f"(@99th-pct-of-normal threshold)"
    )
    print("early warning per real failure:")
    for lt in result["lead_times"]:
        if lt["lead_hours"] is not None:
            print(f"  {lt['failure']}: first alert {lt['lead_hours']:.1f}h before failure")
        else:
            print(f"  {lt['failure']}: no pre-alert; flagged during failure={lt['flagged_during']}")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="MetroPT-3 Phase 2 anomaly detection")
    parser.add_argument("csv", help="path to MetroPT3(AirCompressor).csv")
    parser.add_argument("--freq", default="10min", help="feature-window width")
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args(argv)
    run(args.csv, freq=args.freq, contamination=args.contamination)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
