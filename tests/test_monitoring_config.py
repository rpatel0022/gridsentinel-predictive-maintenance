"""Validate the observability config — dashboard, scrape, and compose wiring.

Stdlib + PyYAML only, so it runs in the lean CI and guards against the dashboard
drifting away from the metric names the service actually exports.
"""

import json
import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_dashboard_valid_and_references_real_metrics():
    dash = json.loads((ROOT / "monitoring/grafana/dashboards/gridsentinel.json").read_text())
    assert dash["title"]
    exprs = [t["expr"] for p in dash["panels"] for t in p["targets"]]
    assert exprs, "dashboard has no queries"

    metrics_src = (ROOT / "serving/metrics.py").read_text()
    used = {tok for e in exprs for tok in re.findall(r"gridsentinel_[a-z_]+", e)}
    assert used
    for metric in used:
        base = metric.replace("_bucket", "").replace("_count", "").replace("_sum", "")
        assert base in metrics_src, f"{base} referenced in dashboard but not defined in metrics.py"


def test_prometheus_scrapes_the_api():
    cfg = yaml.safe_load((ROOT / "monitoring/prometheus/prometheus.yml").read_text())
    scrape = cfg["scrape_configs"][0]
    assert scrape["metrics_path"] == "/metrics"
    targets = [t for grp in scrape["static_configs"] for t in grp["targets"]]
    assert any("api:8000" in t for t in targets)


def test_compose_has_observability_services():
    cfg = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert {"api", "prometheus", "grafana"} <= set(cfg["services"])
    grafana_vols = cfg["services"]["grafana"]["volumes"]
    assert any("provisioning" in v for v in grafana_vols)
