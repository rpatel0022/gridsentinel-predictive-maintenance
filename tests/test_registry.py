"""Tests for the model registry — pure stage logic (lean CI) + a joblib round-trip."""

from dataclasses import dataclass

import pytest
from serving.registry import (
    CANDIDATE,
    PRODUCTION,
    empty_state,
    promote,
    register,
    rollback,
    set_threshold,
)


@dataclass
class _Bundle:
    """Module-level so joblib can pickle it in the round-trip test."""

    version: str
    metrics: dict
    threshold: float = 0.0


def test_register_sets_candidate():
    s = register(empty_state(), "v1", "t0", metrics={"roc_auc": 0.95})
    assert s["stages"][CANDIDATE] == "v1"
    assert s["stages"][PRODUCTION] is None
    assert s["audit"][-1]["action"] == "register"


def test_register_rejects_duplicate():
    s = register(empty_state(), "v1", "t0")
    with pytest.raises(ValueError):
        register(s, "v1", "t1")


def test_promote_sets_production_and_records_from():
    s = register(empty_state(), "v1", "t0")
    s = promote(s, "v1", "t1")
    assert s["stages"][PRODUCTION] == "v1"
    assert s["audit"][-1] == {"action": "promote", "version": "v1", "at": "t1", "from": None}


def test_promote_unknown_version_raises():
    with pytest.raises(ValueError):
        promote(empty_state(), "ghost", "t1")


def test_rollback_restores_previous():
    s = register(empty_state(), "v1", "t0")
    s = promote(s, "v1", "t1")
    s = register(s, "v2", "t2")
    s = promote(s, "v2", "t3")
    assert s["stages"][PRODUCTION] == "v2"
    s = rollback(s, "t4")
    assert s["stages"][PRODUCTION] == "v1"
    assert s["audit"][-1]["action"] == "rollback"


def test_rollback_without_history_raises():
    s = promote(register(empty_state(), "v1", "t0"), "v1", "t1")
    with pytest.raises(ValueError):  # only one production version ever
        rollback(s, "t2")


def test_set_threshold_records_override_and_audit():
    s = set_threshold(empty_state(), PRODUCTION, 0.123, "t0")
    assert s["stage_thresholds"][PRODUCTION] == 0.123
    assert s["audit"][-1]["action"] == "set_threshold"
    assert s["audit"][-1]["threshold"] == 0.123


def test_set_threshold_backward_compatible_state():
    # A registry written before stage_thresholds existed still accepts overrides.
    legacy = {"stages": {PRODUCTION: "v1"}, "versions": {}, "production_history": [], "audit": []}
    s = set_threshold(legacy, PRODUCTION, 0.5, "t0")
    assert s["stage_thresholds"][PRODUCTION] == 0.5


def test_pure_functions_do_not_mutate_input():
    s0 = register(empty_state(), "v1", "t0")
    before = s0["stages"][PRODUCTION]
    promote(s0, "v1", "t1")
    assert s0["stages"][PRODUCTION] == before  # unchanged — promote returns a copy


def test_registry_class_roundtrip(tmp_path):
    pytest.importorskip("joblib")
    from serving.registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    reg.register(_Bundle("v1", {"roc_auc": 0.95}), "t0")
    reg.promote("v1", "t1")
    reg.register(_Bundle("v2", {"roc_auc": 0.96}), "t2")
    reg.promote("v2", "t3")
    assert reg.load(PRODUCTION).version == "v2"
    reg.rollback("t4")
    assert reg.load(PRODUCTION).version == "v1"

    # Reopen from disk — state persisted.
    assert ModelRegistry(str(tmp_path)).stage_version(PRODUCTION) == "v1"


def test_load_applies_stage_threshold_override(tmp_path):
    pytest.importorskip("joblib")
    from serving.registry import ModelRegistry

    reg = ModelRegistry(str(tmp_path))
    reg.register(_Bundle("v1", {"roc_auc": 0.95}, threshold=0.9), "t0")
    reg.promote("v1", "t1")
    assert reg.load(PRODUCTION).threshold == 0.9  # bundle default
    reg.set_threshold(PRODUCTION, 0.4, "t2")  # operator lowers it
    assert reg.load(PRODUCTION).threshold == 0.4  # override applied
    # Persisted across reopen.
    assert ModelRegistry(str(tmp_path)).load(PRODUCTION).threshold == 0.4
