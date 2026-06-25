"""Tests for the operational status report (pure — runs in lean CI)."""

from serving.registry import PRODUCTION, empty_state, promote, register, rollback, set_threshold
from serving.status import build_report, format_report


def _populated():
    s = register(empty_state(), "v1", "t0", metrics={"roc_auc": 0.95})
    s = promote(s, "v1", "t1")
    s = register(s, "v2", "t2", metrics={"roc_auc": 0.96})
    s = promote(s, "v2", "t3")
    s = rollback(s, "t4")  # back to v1
    s = set_threshold(s, PRODUCTION, 0.42, "t5")
    return s


def test_report_reflects_current_production():
    r = build_report(_populated())
    assert r["production"]["version"] == "v1"
    assert r["production"]["metrics"] == {"roc_auc": 0.95}
    assert r["production"]["threshold_override"] == 0.42
    assert r["candidate"]["version"] == "v2"


def test_report_counts_transitions():
    r = build_report(_populated())
    assert r["n_versions"] == 2
    assert r["n_promotions"] == 2
    assert r["n_rollbacks"] == 1


def test_recent_audit_is_tail():
    r = build_report(_populated(), recent=2)
    assert len(r["recent_audit"]) == 2
    assert r["recent_audit"][-1]["action"] == "set_threshold"


def test_empty_registry_report():
    r = build_report(empty_state())
    assert r["production"]["version"] is None
    assert r["n_versions"] == 0


def test_format_is_readable_text():
    text = format_report(build_report(_populated()))
    assert "production : v1" in text
    assert "threshold=0.42" in text
    assert "recent audit:" in text
