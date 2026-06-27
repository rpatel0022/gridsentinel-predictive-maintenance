"""Regression guards for the interactive dashboard build (dashboard extra).

These would have caught the broken early-warning chart (date strings auto-parsed as
datetimes and collapsed) and the clipped capability table. Skipped where the dashboard
extra (matplotlib/plotly) isn't installed, so the lean CI stays green.
"""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("plotly")

from reports.dashboard import _load_from_assets, build_html  # noqa: E402

ASSETS = "reports/assets"


def test_load_from_assets_shapes():
    ts, scores, thr, events, model_auc, lead_times, afr_models, afrs = _load_from_assets(ASSETS)
    assert len(ts) == len(scores) > 100
    assert thr > 0
    assert len(events) == 4  # the four real MetroPT failures
    assert len(model_auc) == 5
    assert len(afr_models) == len(afrs) > 0


def test_build_html_returns_figure():
    fig = build_html(assets_dir=ASSETS, return_fig=True)
    assert fig.layout.height == 1180
    titled = [a for a in fig.layout.annotations if a.text]
    assert len(titled) >= 6  # six subplot titles


def test_early_warning_axis_is_categorical():
    """Regression: lead-time bars must use categorical month-day labels, not dates."""
    fig = build_html(assets_dir=ASSETS, return_fig=True)
    bars = [t for t in fig.data if t.type == "bar"]
    ew = [t for t in bars if all(isinstance(x, str) and any(c.isalpha() for c in x) for x in t.x)]
    assert ew, "no categorical early-warning bar found — dates may be auto-parsed again"
    assert all(len(x.split()) == 2 for x in ew[0].x)  # e.g. "Apr 18"


def test_capability_table_has_all_rows():
    fig = build_html(assets_dir=ASSETS, return_fig=True)
    tables = [t for t in fig.data if t.type == "table"]
    assert tables, "the 'What it does' table is missing"
    assert len(tables[0].cells.values[0]) == 5  # all five capability rows present
