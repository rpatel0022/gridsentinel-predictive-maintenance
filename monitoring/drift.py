"""Data-drift detection — the trigger signal for self-healing retraining (Phase 4).

In production the model serves the live EIA feed, whose distribution genuinely
shifts with weather, season, and time of day. Because true failure labels lag by
weeks, we cannot watch accuracy live; instead we watch the *inputs and predictions*
for drift away from a reference period. A sustained drift is the signal that the
model may be operating off-distribution and should be retrained.

Two complementary measures, both dependency-light and deterministic (so they're
trivially testable):

* **PSI (Population Stability Index)** — the standard industry gauge of how much a
  distribution has moved. Rule of thumb: < 0.1 stable, 0.1–0.2 moderate, > 0.2
  significant drift.
* **Two-sample KS test** — a distribution-free p-value for "are these two samples
  from the same distribution?".

A feature is flagged when *either* fires. We deliberately roll these here rather
than pull in a heavy reporting framework; a richer Evidently report can sit on top
later without changing this trigger logic.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

PSI_THRESHOLD = 0.2
KS_PVALUE = 0.05
_EPS = 1e-6


def psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10) -> float:
    """Population Stability Index between a reference and current sample.

    Bin edges come from the reference's quantiles, so bins are equally populated
    under no drift. Returns 0.0 for identical distributions, growing as they
    diverge.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("psi needs non-empty reference and current samples")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:  # reference is constant — no spread to compare against
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_frac = np.histogram(reference, bins=edges)[0] / reference.size
    cur_frac = np.histogram(current, bins=edges)[0] / current.size
    ref_frac = np.clip(ref_frac, _EPS, None)
    cur_frac = np.clip(cur_frac, _EPS, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov p-value (low p ⇒ different distributions)."""
    return float(ks_2samp(np.asarray(reference, float), np.asarray(current, float)).pvalue)


def feature_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """Drift verdict for one feature: PSI, KS p-value, and the combined flag."""
    p = psi(reference, current)
    ks_p = ks_pvalue(reference, current)
    return {
        "psi": p,
        "ks_pvalue": ks_p,
        "drifted": bool(p > PSI_THRESHOLD or ks_p < KS_PVALUE),
    }


def drift_report(reference, current, features: list[str]) -> dict:
    """Per-feature drift plus an overall verdict.

    Args:
        reference: Mapping/DataFrame of the reference-period samples.
        current: Mapping/DataFrame of the current-period samples.
        features: Columns to test.

    Returns:
        ``{"features": {name: {...}}, "n_drifted": int, "drift_detected": bool}``.
    """
    per_feature = {f: feature_drift(reference[f], current[f]) for f in features}
    drifted = [f for f, v in per_feature.items() if v["drifted"]]
    return {
        "features": per_feature,
        "drifted_features": drifted,
        "n_drifted": len(drifted),
        "drift_detected": len(drifted) > 0,
    }
