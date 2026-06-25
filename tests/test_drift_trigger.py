"""Tests for the drift-gated retrain trigger."""

import numpy as np
import pytest
from monitoring.drift_trigger import decide_retrain


def test_decide_retrain_on_drift():
    d = decide_retrain({"drift_detected": True, "drifted_features": ["demand"]})
    assert d["retrain"] is True and "demand" in d["reason"]


def test_decide_no_retrain_without_drift():
    d = decide_retrain({"drift_detected": False, "drifted_features": []})
    assert d["retrain"] is False and "no drift" in d["reason"]


def test_decide_handles_missing_keys():
    assert decide_retrain({})["retrain"] is False


def test_run_on_samples_skips_self_heal_without_drift():
    pytest.importorskip("scipy")
    from monitoring.drift_trigger import run_on_samples

    rng = np.random.default_rng(0)
    ref = {"demand": rng.normal(0, 1, 2000)}
    cur = {"demand": rng.normal(0, 1, 2000)}  # same distribution → no drift
    # csv/registry are never touched because no retrain fires.
    out = run_on_samples(
        ref, cur, ["demand"], csv_path="unused.csv", registry_root="unused", at="t0"
    )
    assert out["drift_detected"] is False
    assert out["retrain"] is False
    assert out["self_heal"] is None
