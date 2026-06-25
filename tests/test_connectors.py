"""Tests for the live-feed connectors (network-free — only the pure parts)."""

import pytest
from pipelines.connectors import (
    EIAConnector,
    MissingAPIKeyError,
    build_demand_request,
)


def test_build_demand_request_shape():
    url, params = build_demand_request("KEY123", respondent="CISO", length=24)
    assert url.endswith("/electricity/rto/region-data/data/")
    d = dict(params)
    assert d["api_key"] == "KEY123"
    assert d["frequency"] == "hourly"
    assert d["facets[type][]"] == "D"  # demand
    assert d["facets[respondent][]"] == "CISO"
    assert d["length"] == "24"


def test_build_demand_request_optional_window():
    _, params = build_demand_request("K", start="2026-06-01T00", end="2026-06-02T00")
    d = dict(params)
    assert d["start"] == "2026-06-01T00"
    assert d["end"] == "2026-06-02T00"


def test_build_demand_request_requires_key():
    with pytest.raises(MissingAPIKeyError):
        build_demand_request("")


def test_build_demand_request_rejects_bad_length():
    with pytest.raises(ValueError, match="length must be positive"):
        build_demand_request("K", length=0)


def test_connector_without_key_raises(monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        EIAConnector().fetch_demand()


def test_connector_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("EIA_API_KEY", "from-env")
    assert EIAConnector().api_key == "from-env"
