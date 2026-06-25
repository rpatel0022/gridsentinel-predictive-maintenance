"""Neural sequence baseline — an MLP over stacked temporal windows.

The plan calls for an LSTM / Temporal-CNN. A true recurrent/convolutional model
needs a deep-learning framework (PyTorch/TF), which is **not installable in this
sandbox** (the CPU wheel index is proxy-blocked). So this is the honest stand-in: a
multilayer-perceptron (a neural network, `sklearn.neural_network.MLPClassifier`)
that consumes *temporal context* by stacking the previous ``k`` feature-windows into
one input vector. It is evaluated under the same strict temporal CV as the other
models so the comparison is fair.

Expectation, stated up front: with only four real failures this faces the exact
scarcity that capped the supervised baseline (Phase 1), so it is **not** expected to
beat the unsupervised anomaly detector. The point is a fair, honest neural data
point and a clear marker for where a real LSTM/TCN would slot in.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from gridsentinel.cost import CostModel, confusion_cost, optimal_threshold, periodic_schedule_cost
from gridsentinel.cv import temporal_splits
from pipelines.features import WINDOW_START, build_windowed_features, feature_columns
from pipelines.labels import make_labels

COST = CostModel(cost_fn=3000.0, cost_fp=200.0, cost_tp=200.0)


def stack_sequences(X: np.ndarray, k: int) -> np.ndarray:
    """Stack each row with the previous ``k-1`` rows → temporal-context vectors.

    Returns an array of shape ``(n-k+1, k*features)``; row i holds windows
    ``[i, i+k)`` flattened, so the model sees ``k`` consecutive windows of history.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(X) < k:
        return np.empty((0, X.shape[1] * k))
    return np.stack([X[i : i + k].ravel() for i in range(len(X) - k + 1)])


def run(csv_path: str, *, freq: str = "10min", warn_hours: float = 2.0, k: int = 6) -> dict:
    """Train the MLP sequence baseline under temporal CV; return mean metrics."""
    feats = build_windowed_features(pd.read_csv(csv_path), freq=freq)
    cols = feature_columns(feats)
    X = feats[cols].to_numpy()
    y = make_labels(feats[WINDOW_START], warn_hours=warn_hours, label_active=True).to_numpy()

    # Stacked sequences; the label for a stacked row is the label of its last window.
    Xs = stack_sequences(X, k)
    ys = y[k - 1 :]

    windows_per_day = pd.Timedelta(days=1) / pd.Timedelta(freq)
    rows = []
    for tr, te in temporal_splits(len(ys), n_splits=5, embargo=int(warn_hours * 6)):
        ytr, yte = ys[tr], ys[te]
        if ytr.sum() == 0 or yte.sum() == 0:
            continue
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=400,
            early_stopping=True,
            random_state=42,
        )
        clf.fit(Xs[tr], ytr)
        ptr = clf.predict_proba(Xs[tr])[:, 1]
        pte = clf.predict_proba(Xs[te])[:, 1]
        thr, _ = optimal_threshold(ytr.tolist(), ptr.tolist(), COST)
        yhat = (pte >= thr).astype(int)
        model_cost = confusion_cost(yte.tolist(), yhat.tolist(), COST)
        sched = min(
            periodic_schedule_cost(yte.tolist(), COST, max(1, round(d * windows_per_day)))
            for d in (1, 3, 7, 14)
        )
        rows.append(
            {
                "pr_auc": average_precision_score(yte, pte),
                "roc_auc": roc_auc_score(yte, pte),
                "roi_vs_schedule": (sched - model_cost) / sched if sched else 0.0,
            }
        )

    if not rows:
        print("no scorable folds")
        return {}
    mean = {k2: float(np.mean([r[k2] for r in rows])) for k2 in rows[0]}
    mean["scorable_folds"] = len(rows)
    print(
        f"MLP sequence baseline (k={k}): PR-AUC={mean['pr_auc']:.3f} "
        f"ROC-AUC={mean['roc_auc']:.3f} ROI={mean['roi_vs_schedule']:.1%} "
        f"({len(rows)} folds)"
    )
    return mean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLP sequence baseline")
    parser.add_argument("csv")
    parser.add_argument("--k", type=int, default=6, help="windows of temporal context")
    args = parser.parse_args(argv)
    run(args.csv, k=args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
