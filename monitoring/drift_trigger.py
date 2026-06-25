"""Drift-gated retraining — the actual "drift → retrain" wiring.

The self-heal job retrains unconditionally; in production you only want to spend a
retrain when the inputs have actually moved. This connects the two: check drift on
the live feed (reference vs current), and **only** run the self-heal cycle
(`monitoring.self_heal`) when drift is detected. Retraining still happens on the
labeled Tier-1 data — drift on the unlabeled Tier-2 stream is the *signal* that the
model may be stale, per the data seam (see `docs/architecture.md`).

The gating decision (:func:`decide_retrain`) is pure and unit-tested; the live path
pulls the EIA feed and calls the self-heal job.
"""

from __future__ import annotations

import argparse
import sys


def decide_retrain(report: dict) -> dict:
    """Gate: retrain iff the drift report flags drift. Pure."""
    drifted = report.get("drift_detected", False)
    feats = report.get("drifted_features", [])
    return {
        "retrain": bool(drifted),
        "reason": (f"drift on {feats}" if drifted else "no drift — model still in-distribution"),
    }


def run_on_samples(
    reference,
    current,
    features: list[str],
    *,
    csv_path: str,
    registry_root: str,
    at: str,
) -> dict:
    """Check drift on (reference, current); run the self-heal cycle only if drifted."""
    from monitoring.drift import drift_report  # scipy-backed; imported lazily so the

    # pure decide_retrain() stays importable in the lean (scipy-free) CI environment.
    report = drift_report(reference, current, features)
    decision = decide_retrain(report)
    heal = None
    if decision["retrain"]:
        from monitoring.self_heal import run as self_heal_run

        heal = self_heal_run(csv_path, registry_root, at)
    return {
        "drift_detected": report["drift_detected"],
        "drifted_features": report["drifted_features"],
        **decision,
        "self_heal": heal,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: EIA drift on the live feed → retrain on the labeled data if drifted."""
    import datetime as dt

    from pipelines.connectors import EIAConnector

    parser = argparse.ArgumentParser(description="Drift-gated retrain (EIA → self-heal)")
    parser.add_argument("csv", help="labeled training dataset for retraining")
    parser.add_argument("--registry", default="models/registry")
    parser.add_argument("--respondent", default="PJM")
    parser.add_argument("--ref-start", required=True, help="reference window start YYYY-MM-DDTHH")
    parser.add_argument("--ref-end", required=True, help="reference window end YYYY-MM-DDTHH")
    args = parser.parse_args(argv)

    conn = EIAConnector()
    ref = [
        float(r["value"])
        for r in conn.fetch_demand(
            respondent=args.respondent, start=args.ref_start, end=args.ref_end
        )
        if r.get("value") is not None
    ]
    cur = [
        float(r["value"])
        for r in conn.fetch_demand(respondent=args.respondent)
        if r.get("value") is not None
    ]
    at = dt.datetime.now(dt.timezone.utc).isoformat()
    result = run_on_samples(
        {"demand": ref},
        {"demand": cur},
        ["demand"],
        csv_path=args.csv,
        registry_root=args.registry,
        at=at,
    )
    print(f"drift_detected={result['drift_detected']} → {result['reason']}")
    if result["self_heal"]:
        h = result["self_heal"]
        print(f"  retrain: {'PROMOTED' if h['promote'] else 'KEPT'} — {h['reason']}")
    else:
        print("  retrain skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
