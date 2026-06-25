"""Self-healing: drift → retrain → gate → promote (or keep current).

This is the loop the whole MLOps story builds toward. When the drift monitor
(`monitoring.drift` on the live EIA feed) fires — or on a schedule — we retrain a
**candidate** on the latest real data, evaluate it, and let it into production
*only if* it (a) clears the metric gate and (b) doesn't regress against the current
production model. Everything moves through the registry, so a bad promotion is one
``rollback()`` away.

The promotion *decision* (:func:`promotion_decision`) is pure and unit-tested; the
orchestrator (:func:`run`) wires in retraining, the registry, and the real data.
"""

from __future__ import annotations

from pipelines.metric_gate import GATES, check_gates

PROMOTE_KEY = "roc_auc"
REGRESSION_TOLERANCE = 0.01


def promotion_decision(
    candidate: dict,
    production: dict | None,
    *,
    gates: dict[str, float] = GATES,
    key: str = PROMOTE_KEY,
    tolerance: float = REGRESSION_TOLERANCE,
) -> dict:
    """Decide whether a candidate should be promoted over the current production.

    Rules, in order:
      1. The candidate must clear the metric gate; a failing candidate is never
         promoted (a worse model must not reach production via the auto-loop).
      2. With no production model yet, a gate-passing candidate is promoted.
      3. Otherwise promote only if the candidate does not regress on ``key`` beyond
         ``tolerance`` — guards against silently shipping a slightly-worse model.

    Returns:
        ``{"promote": bool, "reason": str, ...}``.
    """
    gate_failures = check_gates(candidate, gates)
    if gate_failures:
        return {
            "promote": False,
            "reason": "candidate failed metric gate",
            "gate_failures": gate_failures,
        }
    if production is None:
        return {"promote": True, "reason": "no production model yet"}
    cand = candidate.get(key)
    prod = production.get(key)
    if cand is None or prod is None:
        return {"promote": False, "reason": f"missing '{key}' on candidate or production"}
    if cand >= prod - tolerance:
        return {
            "promote": True,
            "reason": f"candidate {key}={cand:.3f} ≥ production {prod:.3f} − {tolerance}",
        }
    return {
        "promote": False,
        "reason": f"candidate {key}={cand:.3f} regresses vs production {prod:.3f}",
    }


def run(csv_path: str, registry_root: str, at: str, *, key: str = PROMOTE_KEY) -> dict:
    """Retrain a candidate, gate it against production, and promote if it wins.

    Returns the decision plus the candidate version/metrics. Needs the real data +
    the modeling/serving extras (runs from the retrain workflow, not unit tests).
    """
    from serving.build_artifact import build
    from serving.registry import PRODUCTION, ModelRegistry

    reg = ModelRegistry(registry_root)
    prod_version = reg.stage_version(PRODUCTION)
    prod_metrics = None
    if prod_version is not None:
        prod_metrics = reg.state["versions"].get(prod_version, {}).get("metrics")

    candidate = build(csv_path)
    decision = promotion_decision(candidate.metrics, prod_metrics, key=key)
    reg.register(candidate, at, metrics=candidate.metrics)
    if decision["promote"]:
        reg.promote(candidate.version, at)

    return {
        **decision,
        "candidate_version": candidate.version,
        "candidate_metrics": candidate.metrics,
        "previous_production": prod_version,
        "production_now": reg.stage_version(PRODUCTION),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: run one drift-triggered retrain → gate → promote cycle.

    Exit code is 0 whether or not the candidate is promoted (keeping the current
    model is a valid outcome); it is non-zero only on an error.
    """
    import argparse
    import datetime as dt

    parser = argparse.ArgumentParser(description="GridSentinel self-heal retrain cycle")
    parser.add_argument("csv", help="path to the training dataset")
    parser.add_argument("--registry", default="models/registry", help="registry root dir")
    parser.add_argument("--at", default=None, help="ISO timestamp for the audit log")
    args = parser.parse_args(argv)

    at = args.at or dt.datetime.now(dt.timezone.utc).isoformat()
    result = run(args.csv, args.registry, at)
    verb = "PROMOTED" if result["promote"] else "KEPT current"
    print(f"{verb}: {result['reason']}")
    print(
        f"  candidate={result['candidate_version']} "
        f"production_now={result['production_now']} "
        f"(was {result['previous_production']})"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
