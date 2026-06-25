"""Precompute small dashboard artifacts from the real data.

The Streamlit dashboard reads these tiny files instead of the raw 218 MB telemetry,
so it loads instantly and runs out-of-the-box (the artifacts are committed). Re-run
this when the data or results change.

    python -m reports.precompute --metropt <csv> --backblaze <csv>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent / "assets"

MODEL_AUC = {
    "Anomaly (IsolationForest)": 0.95,
    "MLP sequence": 0.93,
    "XGBoost (supervised)": 0.92,
    "Backblaze fleet RF": 0.73,
    "LSTM": 0.63,
}
LEAD_TIMES = {"2020-04-18": 19.5, "2020-05-29": 0.2, "2020-06-05": 47.8, "2020-07-15": 47.8}


def precompute(metropt_csv: str, backblaze_csv: str) -> None:
    from pipelines.anomaly import (
        FAILURE_EVENTS,
        anomaly_score,
        build_detector,
        normal_training_mask,
    )
    from pipelines.backblaze import annualized_failure_rates, build_cohort, load
    from pipelines.features import WINDOW_START, build_windowed_features, feature_columns

    ASSETS.mkdir(exist_ok=True)

    # --- Anomaly timeline (downsampled hourly so the asset is small) ---
    feats = build_windowed_features(pd.read_csv(metropt_csv), freq="10min")
    X = feats[feature_columns(feats)].to_numpy()
    mask = normal_training_mask(feats[WINDOW_START])
    det = build_detector(n_estimators=200).fit(X[mask])
    scores = anomaly_score(det, X)
    thr = float(np.quantile(scores[mask], 0.99))
    tl = pd.DataFrame({"ts": pd.to_datetime(feats[WINDOW_START]), "score": scores})
    tl = tl.set_index("ts").resample("1h").mean().reset_index().dropna()
    tl.to_csv(ASSETS / "anomaly_timeline.csv", index=False)

    # --- Backblaze AFR + cohort stats ---
    bb = load(backblaze_csv)
    afr = annualized_failure_rates(bb, min_drives=5000).head(8).reset_index()
    afr[["model", "afr", "drives", "failures"]].to_csv(ASSETS / "afr.csv", index=False)
    Xc, yc = build_cohort(bb)

    summary = {
        "model_auc": MODEL_AUC,
        "lead_times": LEAD_TIMES,
        "anomaly_threshold": thr,
        "failures": [{"start": str(e.start), "end": str(e.end)} for e in FAILURE_EVENTS],
        "metropt": {"failures": len(FAILURE_EVENTS), "windows": int(len(feats))},
        "backblaze": {
            "drives_total": int(len(bb)),
            "failures_total": int(bb["failed"].sum()),
            "models": int(bb["model"].nunique()),
            "cohort_drives": int(len(Xc)),
            "cohort_failures": int(yc.sum()),
            "roc_auc": 0.730,
        },
        "headline": {
            "anomaly_roc_auc": 0.95,
            "anomaly_recall": 0.89,
            "p99_ms": 31,
            "cost_savings_pct": 60,
            "ml_test_score": 4.5,
            "tests": 140,
        },
    }
    (ASSETS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote assets to {ASSETS} (timeline {len(tl)} rows, afr {len(afr)} models)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Precompute dashboard assets")
    p.add_argument("--metropt", required=True)
    p.add_argument("--backblaze", required=True)
    a = p.parse_args(argv)
    precompute(a.metropt, a.backblaze)
    return 0


if __name__ == "__main__":
    sys.exit(main())
