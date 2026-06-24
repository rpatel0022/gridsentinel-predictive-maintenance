"""Live-feed connectors for the production/monitoring layer.

Phase 0 ships the EIA connector skeleton: the URL/param construction (a pure,
unit-tested function) and the IO method that fetches hourly US electricity demand.
The API key is read from the environment — **zero keys in git** (Upgrade 6,
security). The request build is separated from the IO so it can be tested without
network access, which matters because the data hosts are not always reachable from
every build environment.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass

EIA_BASE_URL = "https://api.eia.gov/v2"
# Hourly demand by balancing authority — the power-domain live feed (see ADR 0001).
EIA_DEMAND_ROUTE = "electricity/rto/region-data/data/"


class MissingAPIKeyError(RuntimeError):
    """Raised when no EIA API key is available in the environment or call."""


def build_demand_request(
    api_key: str,
    *,
    respondent: str = "PJM",
    start: str | None = None,
    end: str | None = None,
    length: int = 168,
) -> tuple[str, list[tuple[str, str]]]:
    """Build the (url, params) for an EIA hourly-demand query — pure, no IO.

    Args:
        api_key: EIA API key.
        respondent: Balancing-authority code (e.g. ``PJM``, ``CISO``, ``ERCO``).
        start: Inclusive start hour, ``YYYY-MM-DDTHH`` (optional).
        end: Inclusive end hour, ``YYYY-MM-DDTHH`` (optional).
        length: Max number of hourly rows to return.

    Returns:
        ``(url, params)`` where ``params`` is an ordered list of query pairs.

    Raises:
        MissingAPIKeyError: If ``api_key`` is empty.
        ValueError: If ``length`` is not positive.
    """
    if not api_key:
        raise MissingAPIKeyError("EIA API key is required")
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")

    url = f"{EIA_BASE_URL}/{EIA_DEMAND_ROUTE}"
    params: list[tuple[str, str]] = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[0]", "value"),
        ("facets[type][]", "D"),  # D = demand
        ("facets[respondent][]", respondent),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "desc"),
        ("length", str(length)),
    ]
    if start:
        params.append(("start", start))
    if end:
        params.append(("end", end))
    return url, params


@dataclass
class EIAConnector:
    """Fetches hourly electricity demand from the EIA Open Data API v2.

    Args:
        api_key: Explicit key; falls back to the ``EIA_API_KEY`` env var.
        timeout: Per-request timeout in seconds.
    """

    api_key: str | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("EIA_API_KEY")

    def fetch_demand(
        self,
        *,
        respondent: str = "PJM",
        start: str | None = None,
        end: str | None = None,
        length: int = 168,
    ) -> list[dict]:
        """Fetch hourly demand rows. Requires a key and outbound network access.

        Returns:
            The list of record dicts from ``response.data``.

        Raises:
            MissingAPIKeyError: If no API key is configured.
        """
        if not self.api_key:
            raise MissingAPIKeyError("no EIA API key: set EIA_API_KEY or pass api_key=...")
        url, params = build_demand_request(
            self.api_key, respondent=respondent, start=start, end=end, length=length
        )
        query = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{url}?{query}", timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("response", {}).get("data", [])
