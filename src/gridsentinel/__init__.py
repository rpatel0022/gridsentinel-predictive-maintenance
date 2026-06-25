"""GridSentinel — predictive maintenance & anomaly detection for IoT power systems.

Phase 0 ships a single real, tested module: the asymmetric maintenance-cost
model (:mod:`gridsentinel.cost`). Everything in this project is graded on
*expected dollar cost*, not accuracy, so the cost model is the foundation every
later phase builds on.
"""

from gridsentinel.cost import (
    CostModel,
    always_maintain_cost,
    confusion_cost,
    expected_cost,
    never_maintain_cost,
    optimal_threshold,
    periodic_schedule_cost,
)

__version__ = "0.0.1"

__all__ = [
    "CostModel",
    "always_maintain_cost",
    "confusion_cost",
    "expected_cost",
    "never_maintain_cost",
    "optimal_threshold",
    "periodic_schedule_cost",
    "__version__",
]
