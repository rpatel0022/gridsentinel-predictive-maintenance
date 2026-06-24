"""Phase 1 baseline training: RF / XGBoost on MetroPT-3, scored on dollar cost.

The pipeline, end to end:

1. Aggregate the raw CSV into non-overlapping windows (``pipelines.features``).
2. Label each window for "failure within ``warn_hours``" from the real report
   table (``pipelines.labels``); drop in-failure windows.
3. Forward-chaining temporal CV with an embargo gap (``gridsentinel.cv``) — the
   leakage guard, since this is a single device.
4. Per fold, fit RandomForest and XGBoost with imbalance handling, **tune the
   decision threshold on the training fold** against the asymmetric cost model
   (``gridsentinel.cost``), then apply it to the held-out test fold — so the
   operating point is never chosen on the data it is judged on.
5. Compare expected cost against the *strongest* fixed-schedule baseline (the
   cheapest calendar interval) and report the ROI %: how much cheaper the model is.
6. Log params, per-fold and mean metrics to MLflow.

Run::

    python -m pipelines.train_baseline "MetroPT3(AirCompressor).csv"

Everything is evaluated per decision cycle (one window): every ``freq`` we decide
whether to dispatch maintenance, and the cost model prices that decision.
"""

from __future__ import annotations

import argparse
import os
import sys

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from gridsentinel.cost import (
    CostModel,
    confusion_cost,
    never_maintain_cost,
    optimal_threshold,
    periodic_schedule_cost,
)
from gridsentinel.cv import temporal_splits
from pipelines.features import WINDOW_START, build_windowed_features, feature_columns
from pipelines.labels import DROP, make_labels

# Asymmetric maintenance costs, in dollars. Illustrative but ordered correctly: a
# missed failure (emergency truck-roll + downtime) dwarfs a false alarm (a wasted
# inspection). Refine with IntelliPower's real numbers before quoting externally.
COST = CostModel(cost_fn=3000.0, cost_fp=200.0, cost_tp=200.0)

# Candidate fixed-maintenance intervals (in windows) for the dumb baseline. With
# 10-min windows: 1, 3, 7, 14 days. We compare against the *cheapest* of these, so
# the ROI is measured against the strongest schedule, not a strawman.
SCHEDULE_INTERVALS_DAYS = (1, 3, 7, 14)


def best_schedule_cost(y_true: list[int], windows_per_day: float) -> tuple[float, int]:
    """Cheapest fixed-calendar-schedule cost over the candidate intervals."""
    best_cost = float("inf")
    best_days = 0
    for days in SCHEDULE_INTERVALS_DAYS:
        interval = max(1, round(days * windows_per_day))
        cost = periodic_schedule_cost(y_true, COST, interval)
        if cost < best_cost:
            best_cost, best_days = cost, days
    return best_cost, best_days


def _fit(model_name: str, X, y, scale_pos_weight: float):
    if model_name == "random_forest":
        m = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    elif model_name == "xgboost":
        m = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown model {model_name}")
    m.fit(X, y)
    return m


def run(
    csv_path: str,
    *,
    freq: str = "10min",
    warn_hours: float = 2.0,
    label_active: bool = True,
    n_splits: int = 5,
) -> dict:
    """Train the baselines and return a summary of mean cross-validated metrics."""
    raw = pd.read_csv(csv_path)
    feats = build_windowed_features(raw, freq=freq)
    cols = feature_columns(feats)

    labels = make_labels(feats[WINDOW_START], warn_hours=warn_hours, label_active=label_active)
    keep = labels != DROP
    feats = feats[keep].reset_index(drop=True)
    y_all = labels[keep].reset_index(drop=True).to_numpy()
    X_all = feats[cols].to_numpy()

    windows_per_day = pd.Timedelta(days=1) / pd.Timedelta(freq)
    # One window's worth of rows = embargo just past the warning look-ahead, so a
    # pre-boundary window can't be labelled from a post-boundary failure.
    embargo = int(round(warn_hours * (pd.Timedelta(hours=1) / pd.Timedelta(freq))))

    n_pos = int(y_all.sum())
    mode = "detection" if label_active else "predict-ahead"
    print(
        f"windows={len(y_all):,}  positives={n_pos} ({n_pos / len(y_all):.2%})  "
        f"mode={mode}  freq={freq}  warn_hours={warn_hours}  embargo={embargo} windows"
    )

    # MLflow 3.x deprecated the bare file store; default to a local SQLite backend
    # (its recommended replacement) unless the caller points somewhere else.
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("gridsentinel-phase1-baseline")
    summary: dict[str, dict] = {}

    for model_name in ("random_forest", "xgboost"):
        fold_rows: list[dict] = []
        with mlflow.start_run(run_name=f"{model_name}-warn{int(warn_hours)}h-{freq}"):
            mlflow.log_params(
                {
                    "model": model_name,
                    "mode": mode,
                    "freq": freq,
                    "warn_hours": warn_hours,
                    "n_splits": n_splits,
                    "embargo_windows": embargo,
                    "cost_fn": COST.cost_fn,
                    "cost_fp": COST.cost_fp,
                    "cost_tp": COST.cost_tp,
                    "n_windows": int(len(y_all)),
                    "n_positives": n_pos,
                }
            )
            for fold, (tr, te) in enumerate(
                temporal_splits(len(y_all), n_splits=n_splits, embargo=embargo)
            ):
                ytr, yte = y_all[tr], y_all[te]
                if ytr.sum() == 0 or yte.sum() == 0:
                    # A fold with no failures on one side can't tune or score a
                    # cost threshold; skip and note it (failures are sparse/clustered).
                    continue
                pos_weight = float((ytr == 0).sum()) / float(max(1, (ytr == 1).sum()))
                model = _fit(model_name, X_all[tr], ytr, pos_weight)

                ptr = model.predict_proba(X_all[tr])[:, 1]
                pte = model.predict_proba(X_all[te])[:, 1]

                # Operating point chosen on TRAIN only, then frozen for TEST.
                thr, _ = optimal_threshold(ytr.tolist(), ptr.tolist(), COST)
                yhat = (pte >= thr).astype(int)

                model_cost = confusion_cost(yte.tolist(), yhat.tolist(), COST)
                sched_cost, sched_days = best_schedule_cost(yte.tolist(), windows_per_day)
                runtofail = never_maintain_cost(yte.tolist(), COST)
                roi = (sched_cost - model_cost) / sched_cost if sched_cost else 0.0

                tp = int(((yhat == 1) & (yte == 1)).sum())
                fp = int(((yhat == 1) & (yte == 0)).sum())
                fn = int(((yhat == 0) & (yte == 1)).sum())
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0

                row = {
                    "fold": fold,
                    "threshold": thr,
                    "pr_auc": average_precision_score(yte, pte),
                    "roc_auc": roc_auc_score(yte, pte),
                    "precision": precision,
                    "recall": recall,
                    "model_cost": model_cost,
                    "schedule_cost": sched_cost,
                    "schedule_days": sched_days,
                    "run_to_failure_cost": runtofail,
                    "roi_vs_schedule": roi,
                }
                fold_rows.append(row)
                for k in ("pr_auc", "roc_auc", "precision", "recall", "roi_vs_schedule"):
                    mlflow.log_metric(k, row[k], step=fold)
                print(
                    f"    [{model_name} fold {fold}] test_pos={int(yte.sum())} "
                    f"PR-AUC={row['pr_auc']:.3f} ROC-AUC={row['roc_auc']:.3f} "
                    f"recall={recall:.2f} precision={precision:.2f} "
                    f"ROI={roi:.1%} (vs {sched_days}d schedule)"
                )

            if not fold_rows:
                print(f"  {model_name}: no scorable folds (failures too clustered)")
                continue
            mean = {
                k: float(np.mean([r[k] for r in fold_rows]))
                for k in (
                    "pr_auc",
                    "roc_auc",
                    "precision",
                    "recall",
                    "model_cost",
                    "schedule_cost",
                    "roi_vs_schedule",
                )
            }
            mean["scorable_folds"] = len(fold_rows)
            mlflow.log_metrics({f"mean_{k}": v for k, v in mean.items() if k != "scorable_folds"})
            summary[model_name] = mean
            print(
                f"  {model_name}: PR-AUC={mean['pr_auc']:.3f} ROC-AUC={mean['roc_auc']:.3f} "
                f"recall={mean['recall']:.2f} precision={mean['precision']:.2f} "
                f"ROI vs schedule={mean['roi_vs_schedule']:.1%} "
                f"({len(fold_rows)} folds)"
            )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MetroPT-3 Phase 1 baseline training")
    parser.add_argument("csv", help="path to MetroPT3(AirCompressor).csv")
    parser.add_argument("--freq", default="10min", help="feature-window width")
    parser.add_argument("--warn-hours", type=float, default=2.0, help="warning lead-in (hours)")
    parser.add_argument(
        "--predict-ahead",
        action="store_true",
        help="predict-ahead framing (drop in-failure rows) instead of detection",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="temporal CV folds")
    args = parser.parse_args(argv)
    run(
        args.csv,
        freq=args.freq,
        warn_hours=args.warn_hours,
        label_active=not args.predict_ahead,
        n_splits=args.n_splits,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
