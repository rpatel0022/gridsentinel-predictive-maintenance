"""Asymmetric maintenance-cost model.

The project is graded on *expected dollar cost*, not accuracy. The asymmetry is
the whole point: a **missed failure** (false negative) triggers an emergency
truck-roll plus downtime, while a **false alarm** (false positive) only wastes an
inspection. These helpers turn predictions into an expected cost, find the
decision threshold that minimises it, and compare against dumb baselines
(inspect-everything, inspect-nothing, fixed-schedule upkeep) so the model's lift
is legible.

Pure standard library on purpose — Phase 0 stays dependency-light so CI is fast
and the cost logic is trivially auditable. Later phases feed it model
probabilities; the contract here does not change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Labels = Sequence[int]
Scores = Sequence[float]


@dataclass(frozen=True)
class CostModel:
    """Per-event costs for one predictive-maintenance decision, in dollars.

    A true negative (correctly quiet) is free. A true positive is an alert we
    acted on in time — it costs a *planned* inspection/repair, far less than the
    emergency a missed failure causes.

    Args:
        cost_fn: Cost of a missed failure (emergency repair + downtime).
        cost_fp: Cost of a false alarm (wasted inspection).
        cost_tp: Cost of acting on a correct alert (planned maintenance).

    Raises:
        ValueError: If any cost is negative, or if ``cost_fn`` does not exceed
            ``cost_fp`` — without that asymmetry the whole exercise is moot.
    """

    cost_fn: float
    cost_fp: float
    cost_tp: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("cost_fn", self.cost_fn),
            ("cost_fp", self.cost_fp),
            ("cost_tp", self.cost_tp),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.cost_fn <= self.cost_fp:
            raise ValueError(
                "cost_fn must exceed cost_fp: a missed failure has to hurt more "
                f"than a false alarm (got cost_fn={self.cost_fn}, cost_fp={self.cost_fp})"
            )


def _check_lengths(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")


def confusion_cost(y_true: Labels, y_pred: Labels, model: CostModel) -> float:
    """Total expected cost of a set of hard 0/1 predictions.

    Args:
        y_true: Ground-truth labels (1 = failure occurred).
        y_pred: Predicted labels (1 = maintenance triggered).
        model: The :class:`CostModel` defining per-event costs.

    Returns:
        The summed dollar cost across all predictions (true negatives are free).
    """
    _check_lengths(y_true, y_pred)
    fn = fp = tp = 0
    for t, p in zip(y_true, y_pred, strict=False):  # lengths checked above
        if t and p:
            tp += 1
        elif t and not p:
            fn += 1
        elif not t and p:
            fp += 1
    return fn * model.cost_fn + fp * model.cost_fp + tp * model.cost_tp


def expected_cost(y_true: Labels, y_score: Scores, threshold: float, model: CostModel) -> float:
    """Expected cost when alerting wherever ``score >= threshold``."""
    _check_lengths(y_true, y_score)
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return confusion_cost(y_true, y_pred, model)


def optimal_threshold(y_true: Labels, y_score: Scores, model: CostModel) -> tuple[float, float]:
    """Find the decision threshold that minimises expected cost.

    Sweeps every distinct score as a candidate cutoff (plus a cutoff above the
    maximum, i.e. "never alert"). This is the single move that turns a model into
    ROI: instead of the default 0.5, we pick the threshold the *cost function*
    prefers.

    Args:
        y_true: Ground-truth labels.
        y_score: Model scores / probabilities, higher = more failure-like.
        model: The :class:`CostModel`.

    Returns:
        ``(threshold, cost)`` for the cost-minimising cutoff. On ties, the
        highest threshold (fewest alarms) wins.

    Raises:
        ValueError: If inputs are empty or mismatched in length.
    """
    _check_lengths(y_true, y_score)
    if not y_true:
        raise ValueError("cannot optimise a threshold on empty inputs")

    # "Never alert" sits just above the largest score; every real score is also a
    # candidate cutoff (alert where score >= cutoff).
    candidates = sorted(set(y_score)) + [max(y_score) + 1.0]
    best_threshold = candidates[-1]
    best_cost = expected_cost(y_true, y_score, best_threshold, model)
    for threshold in reversed(candidates[:-1]):
        cost = expected_cost(y_true, y_score, threshold, model)
        if cost < best_cost:
            best_cost, best_threshold = cost, threshold
    return best_threshold, best_cost


def always_maintain_cost(y_true: Labels, model: CostModel) -> float:
    """Baseline: inspect every unit every cycle (perfect recall, max inspection)."""
    return confusion_cost(y_true, [1] * len(y_true), model)


def never_maintain_cost(y_true: Labels, model: CostModel) -> float:
    """Baseline: run to failure — never maintain (every failure is a miss)."""
    return confusion_cost(y_true, [0] * len(y_true), model)


def periodic_schedule_cost(y_true: Labels, model: CostModel, interval: int) -> float:
    """Baseline: fixed-schedule upkeep — maintain every ``interval``-th cycle.

    A stand-in for "replace on a calendar" maintenance, which is what GridSentinel
    must beat on expected cost. ``y_true`` is assumed time-ordered. A failure is
    "caught" only if a scheduled maintenance lands on the same cycle; otherwise it
    is a miss. The richer same-coverage comparison lands in Phase 1 — this gives
    Phase 0 a concrete, tested baseline to anchor on.

    Args:
        y_true: Time-ordered ground-truth labels.
        model: The :class:`CostModel`.
        interval: Maintain on every cycle whose index is a multiple of this.

    Raises:
        ValueError: If ``interval`` is not a positive integer.
    """
    if interval < 1:
        raise ValueError(f"interval must be a positive integer, got {interval}")
    y_pred = [1 if (i % interval == 0) else 0 for i in range(len(y_true))]
    return confusion_cost(y_true, y_pred, model)
