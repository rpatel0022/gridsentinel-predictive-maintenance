"""Guard the committed dashboard assets (stdlib + pandas, lean CI)."""

import json
import pathlib

import pytest

pd = pytest.importorskip("pandas")

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "reports" / "assets"


def test_summary_has_expected_shape():
    s = json.loads((ASSETS / "summary.json").read_text())
    assert s["headline"]["tests"] >= 1
    assert s["backblaze"]["failures_total"] > 10000  # real fleet, many failures
    assert 0 < s["headline"]["anomaly_roc_auc"] <= 1
    assert len(s["failures"]) == 4  # the four real MetroPT failures


def test_afr_asset_is_sorted_and_real():
    afr = pd.read_csv(ASSETS / "afr.csv")
    assert {"model", "afr", "drives", "failures"} <= set(afr.columns)
    assert (afr["afr"] >= 0).all()
    assert afr["afr"].is_monotonic_decreasing  # worst-first


def test_timeline_asset_nonempty():
    tl = pd.read_csv(ASSETS / "anomaly_timeline.csv")
    assert {"ts", "score"} <= set(tl.columns)
    assert len(tl) > 100
