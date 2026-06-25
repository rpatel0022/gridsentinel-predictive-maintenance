"""Metric gate — block a model that regresses below committed thresholds.

A model that quietly gets worse is one of the most expensive failures in
production ML, and most portfolios have no guard against it. This is that guard:
CI rebuilds the anomaly detector on the real data, evaluates it, and runs these
gates. If any headline metric falls below its floor, the build fails and the model
does not ship.

The gate *logic* (:func:`check_gates`) is pure stdlib so it runs and is tested in
the lean CI; the actual rebuild (`--csv`) needs the data + modeling extra and runs
in the `model-eval` workflow.
"""

from __future__ import annotations

import argparse
import json
import sys

# Minimum acceptable offline metrics for the Phase 2 anomaly detector. Set a margin
# below the observed values (ROC-AUC 0.95, PR-AUC 0.39, recall 0.89) so normal
# run-to-run variation doesn't false-fail, but a genuine regression trips the gate.
GATES: dict[str, float] = {
    "roc_auc": 0.90,
    "pr_auc": 0.30,
    "recall": 0.75,
}


def check_gates(metrics: dict, gates: dict[str, float] = GATES) -> list[str]:
    """Return human-readable gate failures (empty list = all pass).

    A metric that is missing from ``metrics`` is itself a failure — a silently
    dropped metric must not pass the gate.
    """
    failures: list[str] = []
    for name, floor in gates.items():
        if name not in metrics:
            failures.append(f"{name}: missing from metrics")
        elif metrics[name] < floor:
            failures.append(f"{name}={metrics[name]:.3f} < required {floor}")
    return failures


def evaluate_csv(csv_path: str) -> dict:
    """Rebuild + evaluate the anomaly detector on real data (needs modeling extra)."""
    import pandas as pd

    from pipelines.anomaly import evaluate
    from pipelines.features import build_windowed_features

    feats = build_windowed_features(pd.read_csv(csv_path))
    return evaluate(feats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GridSentinel model metric gate")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="rebuild + evaluate the model on this dataset")
    src.add_argument("--metrics-json", help="gate a precomputed metrics JSON file")
    args = parser.parse_args(argv)

    if args.metrics_json:
        with open(args.metrics_json) as fh:
            metrics = json.load(fh)
    else:
        metrics = evaluate_csv(args.csv)

    failures = check_gates(metrics)
    print("metrics:", {k: round(metrics[k], 4) for k in GATES if k in metrics})
    print("gates  :", GATES)
    if failures:
        print("GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
