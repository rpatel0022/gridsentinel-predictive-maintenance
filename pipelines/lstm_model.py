"""LSTM sequence model for MetroPT-3 (the real deep-learning model).

Earlier the deep-learning requirement was met by an MLP stand-in because no DL
framework was installable. With ``tensorflow-cpu`` now available, this is the real
thing: a recurrent **LSTM** over sequences of consecutive feature-windows, trained
under the same strict temporal CV as every other model so the comparison is fair.

Honest expectation, stated up front: with only four real failures (2 scorable
folds), a recurrent net has very little to learn from and is *not* expected to beat
the unsupervised anomaly detector — same scarcity that capped the supervised
baseline. The value is a genuine recurrent-model data point, reported honestly.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from gridsentinel.cost import CostModel, confusion_cost, optimal_threshold, periodic_schedule_cost
from gridsentinel.cv import temporal_splits
from pipelines.features import WINDOW_START, build_windowed_features, feature_columns
from pipelines.labels import make_labels

COST = CostModel(cost_fn=3000.0, cost_fp=200.0, cost_tp=200.0)


def make_sequences(X: np.ndarray, k: int) -> np.ndarray:
    """3D sequences ``(n-k+1, k, n_features)`` of consecutive windows (for the LSTM)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(X) < k:
        return np.empty((0, k, X.shape[1]))
    return np.stack([X[i : i + k] for i in range(len(X) - k + 1)])


def _build_lstm(n_steps: int, n_features: int):
    import tensorflow as tf

    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((n_steps, n_features)),
            tf.keras.layers.Masking(),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def run(csv_path: str, *, freq: str = "10min", warn_hours: float = 2.0, k: int = 6) -> dict:
    """Train the LSTM under temporal CV on real data; return mean metrics."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    feats = build_windowed_features(pd.read_csv(csv_path), freq=freq)
    cols = feature_columns(feats)
    X = feats[cols].to_numpy()
    y = make_labels(feats[WINDOW_START], warn_hours=warn_hours, label_active=True).to_numpy()
    seqs = make_sequences(X, k)
    ys = y[k - 1 :]

    windows_per_day = pd.Timedelta(days=1) / pd.Timedelta(freq)
    rows = []
    for tr, te in temporal_splits(len(ys), n_splits=5, embargo=int(warn_hours * 6)):
        ytr, yte = ys[tr], ys[te]
        if ytr.sum() == 0 or yte.sum() == 0:
            continue
        # Scale features on train stats; reshape for the scaler then back to 3D.
        scaler = StandardScaler().fit(seqs[tr].reshape(-1, seqs.shape[2]))
        Xtr = scaler.transform(seqs[tr].reshape(-1, seqs.shape[2])).reshape(seqs[tr].shape)
        Xte = scaler.transform(seqs[te].reshape(-1, seqs.shape[2])).reshape(seqs[te].shape)

        model = _build_lstm(seqs.shape[1], seqs.shape[2])
        pos_w = float((ytr == 0).sum()) / float(max(1, (ytr == 1).sum()))
        model.fit(
            Xtr,
            ytr,
            epochs=8,
            batch_size=256,
            class_weight={0: 1.0, 1: pos_w},
            verbose=0,
        )
        pte = model.predict(Xte, verbose=0).ravel()
        ptr = model.predict(Xtr, verbose=0).ravel()
        thr, _ = optimal_threshold(ytr.tolist(), ptr.tolist(), COST)
        yhat = (pte >= thr).astype(int)
        model_cost = confusion_cost(yte.tolist(), yhat.tolist(), COST)
        sched = min(
            periodic_schedule_cost(yte.tolist(), COST, max(1, round(d * windows_per_day)))
            for d in (1, 3, 7, 14)
        )
        rows.append(
            {
                "pr_auc": float(average_precision_score(yte, pte)),
                "roc_auc": float(roc_auc_score(yte, pte)),
                "roi_vs_schedule": (sched - model_cost) / sched if sched else 0.0,
            }
        )

    if not rows:
        print("no scorable folds")
        return {}
    mean = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    mean["scorable_folds"] = len(rows)
    print(
        f"LSTM (k={k}): PR-AUC={mean['pr_auc']:.3f} ROC-AUC={mean['roc_auc']:.3f} "
        f"ROI={mean['roi_vs_schedule']:.1%} ({len(rows)} folds)"
    )
    return mean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSTM sequence model")
    parser.add_argument("csv")
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args(argv)
    run(args.csv, k=args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
