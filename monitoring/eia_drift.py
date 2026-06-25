"""Drift on the live EIA feed — the real, naturally-occurring non-stationarity.

US electricity demand swings with weather, season, and time of day, so comparing a
reference period (e.g. winter) against a current one surfaces *genuine* drift — not
noise we injected. This is the Tier-2 signal that, in the self-healing loop, trips
the retraining trigger. Run::

    EIA_API_KEY=... python -m monitoring.eia_drift --respondent PJM \
        --ref-start 2026-01-08T00 --ref-end 2026-01-15T00
"""

from __future__ import annotations

import argparse
import sys

from pipelines.connectors import EIAConnector

from monitoring.drift import drift_report


def _values(rows: list[dict]) -> list[float]:
    return [float(r["value"]) for r in rows if r.get("value") is not None]


def run(respondent: str, ref_start: str, ref_end: str, *, length: int = 168) -> dict:
    """Fetch a reference vs current demand window and report drift on demand."""
    conn = EIAConnector()
    reference = _values(conn.fetch_demand(respondent=respondent, start=ref_start, end=ref_end))
    current = _values(conn.fetch_demand(respondent=respondent, length=length))
    if not reference or not current:
        raise RuntimeError("EIA returned no demand rows; check key/respondent/dates")
    report = drift_report({"demand": reference}, {"demand": current}, ["demand"])
    d = report["features"]["demand"]
    print(
        f"{respondent} demand drift  ref_n={len(reference)} cur_n={len(current)}  "
        f"PSI={d['psi']:.3f}  KS_p={d['ks_pvalue']:.2e}  "
        f"drift_detected={report['drift_detected']}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drift on the live EIA demand feed")
    parser.add_argument("--respondent", default="PJM")
    parser.add_argument("--ref-start", required=True, help="reference window start, YYYY-MM-DDTHH")
    parser.add_argument("--ref-end", required=True, help="reference window end, YYYY-MM-DDTHH")
    parser.add_argument("--length", type=int, default=168)
    args = parser.parse_args(argv)
    run(args.respondent, args.ref_start, args.ref_end, length=args.length)
    return 0


if __name__ == "__main__":
    sys.exit(main())
