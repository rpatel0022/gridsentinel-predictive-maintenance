"""Operational status report — what's live, and how it got there.

One command answers an on-call's first questions: which model is in production, what
were its offline metrics, what alert threshold is in force, and what recently changed
(promotions, rollbacks, threshold tweaks). It reads the registry's ``registry.json``
— no model load required — so the report core is pure and unit-tested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from serving.registry import CANDIDATE, PRODUCTION


def _stage(state: dict, stage: str) -> dict:
    version = state.get("stages", {}).get(stage)
    versions = state.get("versions", {})
    thresholds = state.get("stage_thresholds", {})
    return {
        "version": version,
        "metrics": versions.get(version, {}).get("metrics", {}) if version else {},
        "threshold_override": thresholds.get(stage),
    }


def build_report(state: dict, *, recent: int = 5) -> dict:
    """Summarise registry state: stages, counts, and the recent audit tail."""
    audit = state.get("audit", [])
    actions = [e.get("action") for e in audit]
    return {
        "production": _stage(state, PRODUCTION),
        "candidate": _stage(state, CANDIDATE),
        "n_versions": len(state.get("versions", {})),
        "n_promotions": actions.count("promote"),
        "n_rollbacks": actions.count("rollback"),
        "recent_audit": audit[-recent:],
    }


def format_report(report: dict) -> str:
    """Render the report as plain text for an operator."""
    prod, cand = report["production"], report["candidate"]
    lines = [
        "GridSentinel — operational status",
        "=" * 34,
        f"production : {prod['version'] or '(none)'}"
        + (
            f"  threshold={prod['threshold_override']}"
            if prod["threshold_override"] is not None
            else ""
        ),
        f"  metrics  : {prod['metrics'] or '(none)'}",
        f"candidate  : {cand['version'] or '(none)'}",
        f"versions   : {report['n_versions']}   promotions: {report['n_promotions']}   "
        f"rollbacks: {report['n_rollbacks']}",
        "recent audit:",
    ]
    for entry in report["recent_audit"]:
        extra = {k: v for k, v in entry.items() if k not in ("action", "version", "at")}
        lines.append(
            f"  {entry.get('at', ''):<24} {entry.get('action', ''):<14} "
            f"{entry.get('version') or '':<28} {extra or ''}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GridSentinel operational status")
    parser.add_argument("--registry", default="models/registry", help="registry root dir")
    parser.add_argument("--recent", type=int, default=5)
    args = parser.parse_args(argv)

    path = os.path.join(args.registry, "registry.json")
    if not os.path.exists(path):
        print(f"no registry at {path}")
        return 1
    with open(path) as fh:
        state = json.load(fh)
    print(format_report(build_report(state, recent=args.recent)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
