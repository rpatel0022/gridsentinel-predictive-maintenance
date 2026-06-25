"""Backblaze fleet-reliability model — failure prediction at 418k-drive scale.

The lifetime table (one row per drive: install date, last-seen date, failed) is
turned into a **leakage-safe** supervised task: pick a reference time ``T``, take the
drives alive at ``T``, and predict which fail within ``horizon`` days *after* ``T``,
using only features known at ``T`` (capacity, drive age-so-far, and the model's
historical reliability). Age is measured identically for every drive (install → T),
so the censoring trap — where a failed drive's "age" is time-to-failure but a
survivor's is just time-observed — is avoided.

Also computes per-model **annualized failure rates** (AFR), Backblaze's own headline
reliability metric, which needs no model and no split.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

REFERENCE = "2022-01-01"
HORIZON_DAYS = 365
GLOBAL_SMOOTHING = 50.0  # pseudo-count toward the global rate for rare models


def load(csv_path: str) -> pd.DataFrame:
    """Load the lifetime CSV with dates parsed."""
    return pd.read_csv(csv_path, parse_dates=["min_date", "max_date"])


def annualized_failure_rates(df: pd.DataFrame, *, min_drives: int = 1000) -> pd.DataFrame:
    """Per-model AFR = failures / drive-years observed (Backblaze's headline metric)."""
    age_years = (df["max_date"] - df["min_date"]).dt.days.clip(lower=1) / 365.25
    g = df.assign(drive_years=age_years).groupby("model")
    out = g.agg(
        drives=("failed", "size"), failures=("failed", "sum"), drive_years=("drive_years", "sum")
    )
    out["afr"] = out["failures"] / out["drive_years"]
    return out[out["drives"] >= min_drives].sort_values("afr", ascending=False)


def build_cohort(df: pd.DataFrame, *, reference: str = REFERENCE, horizon_days: int = HORIZON_DAYS):
    """Leakage-safe cohort: drives alive at ``reference``, labelled by failure within horizon.

    Returns ``(features_df, y)`` where features are ``capacity_tb``, ``age_days``,
    ``model``; censored-within-horizon drives (left, not failed) are dropped.
    """
    T = pd.Timestamp(reference)
    horizon = pd.Timedelta(days=horizon_days)
    at_risk = df[(df["min_date"] <= T) & (df["max_date"] >= T)].copy()
    fails_in_h = (at_risk["failed"] == 1) & (at_risk["max_date"] <= T + horizon)
    censored = (at_risk["failed"] == 0) & (at_risk["max_date"] <= T + horizon)
    cohort = at_risk[~censored].copy()
    cohort["age_days"] = (T - cohort["min_date"]).dt.days
    y = (fails_in_h[~censored]).astype(int).to_numpy()
    return cohort[["capacity_tb", "age_days", "model"]].reset_index(drop=True), y


def target_encode(models_train, y_train, models_apply, *, smoothing: float = GLOBAL_SMOOTHING):
    """Smoothed per-model failure rate from TRAIN only; unseen models → global rate."""
    s = pd.DataFrame({"model": np.asarray(models_train), "y": np.asarray(y_train)})
    global_rate = s["y"].mean()
    stats = s.groupby("model")["y"].agg(["sum", "count"])
    enc = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
    return pd.Series(models_apply).map(enc).fillna(global_rate).to_numpy()


def run(csv_path: str, *, reference: str = REFERENCE, horizon_days: int = HORIZON_DAYS) -> dict:
    """Build the cohort, train an honest failure classifier, report metrics + AFR."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from pipelines.backblaze_schema import validate

    df = validate(load(csv_path))
    X, y = build_cohort(df, reference=reference, horizon_days=horizon_days)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Model-reliability prior (target encoding) fit on TRAIN only.
    te_tr = target_encode(Xtr["model"], ytr, Xtr["model"])
    te_te = target_encode(Xtr["model"], ytr, Xte["model"])

    feat_tr = np.column_stack([Xtr["capacity_tb"], Xtr["age_days"], te_tr])
    feat_te = np.column_stack([Xte["capacity_tb"], Xte["age_days"], te_te])

    clf = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=42
    )
    clf.fit(feat_tr, ytr)
    proba = clf.predict_proba(feat_te)[:, 1]

    result = {
        "reference": reference,
        "horizon_days": horizon_days,
        "n_drives": int(len(X)),
        "n_failures": int(y.sum()),
        "failure_rate": float(y.mean()),
        # Full model vs the model-reliability-only prior (does age/capacity add signal?).
        "roc_auc": float(roc_auc_score(yte, proba)),
        "pr_auc": float(average_precision_score(yte, proba)),
        "roc_auc_model_prior_only": float(roc_auc_score(yte, te_te)),
        "pr_auc_model_prior_only": float(average_precision_score(yte, te_te)),
    }
    afr = annualized_failure_rates(df)
    result["worst_models_afr"] = afr.head(3)["afr"].round(4).to_dict()
    result["best_models_afr"] = afr.tail(3)["afr"].round(4).to_dict()

    print(
        f"cohort: {result['n_drives']:,} drives, {result['n_failures']:,} failures "
        f"({result['failure_rate']:.2%}) within {horizon_days}d of {reference}"
    )
    print(
        f"RandomForest: ROC-AUC={result['roc_auc']:.3f} PR-AUC={result['pr_auc']:.3f}  "
        f"(model-prior-only: ROC-AUC={result['roc_auc_model_prior_only']:.3f})"
    )
    print(f"worst models by AFR: {result['worst_models_afr']}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backblaze fleet-reliability model")
    parser.add_argument("csv", help="path to the Backblaze drive-lifetime CSV")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    args = parser.parse_args(argv)
    run(args.csv, reference=args.reference, horizon_days=args.horizon_days)
    return 0


if __name__ == "__main__":
    sys.exit(main())
