"""Tests for the metric gate (pure stdlib — runs in the lean CI)."""

from pipelines.metric_gate import GATES, check_gates, main


def test_passes_when_metrics_meet_floor():
    metrics = {"roc_auc": 0.95, "pr_auc": 0.39, "recall": 0.89}
    assert check_gates(metrics) == []


def test_passes_at_exact_floor():
    assert check_gates(dict(GATES)) == []


def test_fails_on_regression():
    metrics = {"roc_auc": 0.80, "pr_auc": 0.39, "recall": 0.89}
    failures = check_gates(metrics)
    assert len(failures) == 1 and "roc_auc" in failures[0]


def test_missing_metric_is_a_failure():
    metrics = {"roc_auc": 0.95, "pr_auc": 0.39}  # recall dropped
    failures = check_gates(metrics)
    assert any("recall" in f and "missing" in f for f in failures)


def test_custom_gates():
    assert check_gates({"recall": 0.5}, {"recall": 0.9}) != []
    assert check_gates({"recall": 0.95}, {"recall": 0.9}) == []


def test_main_gates_a_metrics_file(tmp_path):
    import json

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"roc_auc": 0.95, "pr_auc": 0.39, "recall": 0.89}))
    assert main(["--metrics-json", str(good)]) == 0

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"roc_auc": 0.5, "pr_auc": 0.1, "recall": 0.2}))
    assert main(["--metrics-json", str(bad)]) == 1
