"""Smoke test: the package imports and exposes its version."""

import gridsentinel


def test_package_has_version():
    assert isinstance(gridsentinel.__version__, str)
    assert gridsentinel.__version__.count(".") == 2


def test_public_api_is_importable():
    assert callable(gridsentinel.optimal_threshold)
    assert gridsentinel.CostModel is not None
