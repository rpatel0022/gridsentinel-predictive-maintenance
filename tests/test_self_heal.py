"""Tests for the self-healing promotion decision (pure — runs in lean CI)."""

from monitoring.self_heal import promotion_decision

PASSING = {"roc_auc": 0.95, "pr_auc": 0.39, "recall": 0.89}


def test_first_model_promotes_when_gate_passes():
    d = promotion_decision(PASSING, None)
    assert d["promote"] is True and "no production" in d["reason"]


def test_candidate_failing_gate_is_rejected():
    weak = {"roc_auc": 0.5, "pr_auc": 0.1, "recall": 0.2}
    d = promotion_decision(weak, None)
    assert d["promote"] is False and d["gate_failures"]


def test_better_candidate_promotes():
    better = {**PASSING, "roc_auc": 0.97}
    d = promotion_decision(better, PASSING)
    assert d["promote"] is True


def test_regressing_candidate_is_kept_out():
    worse = {**PASSING, "roc_auc": 0.91}  # passes gate but below production
    d = promotion_decision(worse, PASSING)
    assert d["promote"] is False and "regress" in d["reason"]


def test_small_regression_within_tolerance_promotes():
    slightly = {**PASSING, "roc_auc": 0.945}  # 0.005 < 0.01 tolerance
    d = promotion_decision(slightly, PASSING)
    assert d["promote"] is True


def test_missing_key_blocks_promotion():
    d = promotion_decision({"pr_auc": 0.4, "recall": 0.8, "roc_auc": 0.95}, {"pr_auc": 0.4})
    assert d["promote"] is False
