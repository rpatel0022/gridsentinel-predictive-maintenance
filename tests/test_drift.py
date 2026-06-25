"""Tests for drift detection (PSI + KS). Synthetic samples only."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from monitoring.drift import drift_report, feature_drift, ks_pvalue, psi  # noqa: E402


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    # Split one sample in two halves — same distribution → tiny PSI.
    assert psi(x[:2500], x[2500:]) < 0.1


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    near = rng.normal(0.2, 1, 5000)
    far = rng.normal(3.0, 1, 5000)
    assert psi(ref, far) > psi(ref, near)
    assert psi(ref, far) > 0.2


def test_psi_constant_reference_is_safe():
    assert psi(np.ones(100), np.ones(100)) == 0.0


def test_ks_detects_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 2000)
    same_p = ks_pvalue(ref, rng.normal(0, 1, 2000))
    shift_p = ks_pvalue(ref, rng.normal(2, 1, 2000))
    assert shift_p < 0.05  # a real shift is clearly significant
    assert shift_p < same_p  # and far more significant than no shift


def test_feature_drift_flag():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 3000)
    assert feature_drift(ref, rng.normal(0, 1, 3000))["drifted"] is False
    assert feature_drift(ref, rng.normal(4, 1, 3000))["drifted"] is True


def test_drift_report_overall():
    rng = np.random.default_rng(4)
    reference = {"a": rng.normal(0, 1, 2000), "b": rng.normal(5, 1, 2000)}
    current = {"a": rng.normal(0, 1, 2000), "b": rng.normal(9, 1, 2000)}  # b drifts
    rep = drift_report(reference, current, ["a", "b"])
    assert rep["drift_detected"] is True
    assert rep["drifted_features"] == ["b"]
    assert rep["n_drifted"] == 1
