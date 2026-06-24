"""Tests for the asymmetric maintenance-cost model."""

import pytest

from gridsentinel.cost import (
    CostModel,
    always_maintain_cost,
    confusion_cost,
    expected_cost,
    never_maintain_cost,
    optimal_threshold,
    periodic_schedule_cost,
)

# A missed failure costs 100x a wasted inspection — the realistic asymmetry.
MODEL = CostModel(cost_fn=1000.0, cost_fp=10.0, cost_tp=10.0)


def test_costmodel_rejects_negative_costs():
    with pytest.raises(ValueError, match="non-negative"):
        CostModel(cost_fn=1000.0, cost_fp=-1.0)


def test_costmodel_requires_asymmetry():
    with pytest.raises(ValueError, match="missed failure"):
        CostModel(cost_fn=10.0, cost_fp=10.0)


def test_confusion_cost_known_example():
    # y_true / y_pred chosen to give exactly 1 TP, 1 FN, 1 FP, 1 TN.
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]
    # tp*10 + fn*1000 + fp*10 + tn*0
    assert confusion_cost(y_true, y_pred, MODEL) == 10.0 + 1000.0 + 10.0


def test_true_negatives_are_free():
    assert confusion_cost([0, 0, 0], [0, 0, 0], MODEL) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        confusion_cost([1, 0], [1], MODEL)


def test_expected_cost_thresholding():
    y_true = [1, 0]
    y_score = [0.9, 0.4]
    # threshold 0.5 -> predict [1, 0] -> one TP, one TN
    assert expected_cost(y_true, y_score, 0.5, MODEL) == MODEL.cost_tp
    # threshold above both -> predict nothing -> the failure is missed
    assert expected_cost(y_true, y_score, 1.0, MODEL) == MODEL.cost_fn


def test_optimal_threshold_beats_naive_baselines():
    # Perfectly separable: failures score high, healthy units score low.
    y_true = [1, 1, 0, 0, 0]
    y_score = [0.95, 0.80, 0.20, 0.10, 0.05]
    threshold, cost = optimal_threshold(y_true, y_score, MODEL)
    # The model catches both failures at planned cost, no false alarms.
    assert cost == 2 * MODEL.cost_tp
    assert 0.20 < threshold <= 0.80
    assert cost < always_maintain_cost(y_true, MODEL)
    assert cost < never_maintain_cost(y_true, MODEL)


def test_optimal_threshold_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        optimal_threshold([], [], MODEL)


def test_naive_baselines():
    y_true = [1, 0, 0]  # one failure, two healthy
    # Always maintain: 1 TP + 2 FP
    assert always_maintain_cost(y_true, MODEL) == MODEL.cost_tp + 2 * MODEL.cost_fp
    # Never maintain: the one failure is missed
    assert never_maintain_cost(y_true, MODEL) == MODEL.cost_fn


def test_periodic_schedule_catches_aligned_failures():
    # Failures at index 0 and 2; maintain every 2nd cycle -> both caught.
    y_true = [1, 0, 1, 0]
    assert periodic_schedule_cost(y_true, MODEL, interval=2) == 2 * MODEL.cost_tp


def test_periodic_schedule_misses_unaligned_failure():
    # Failure at index 1; maintenance lands on indices 0 and 2 -> missed (FN) + 2 FP.
    y_true = [0, 1, 0]
    assert periodic_schedule_cost(y_true, MODEL, interval=2) == MODEL.cost_fn + 2 * MODEL.cost_fp


def test_periodic_schedule_rejects_bad_interval():
    with pytest.raises(ValueError, match="positive integer"):
        periodic_schedule_cost([1, 0], MODEL, interval=0)
