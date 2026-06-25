"""Model-behavioral tests (CheckList-style) for the anomaly detector.

Beyond "the code runs", these assert the model *behaves* sensibly — the
minimum-functionality, directional-expectation, and invariance tests the plan calls
for. They run on a tiny detector fitted to synthetic "normal" operation, so they
test behavior, not the specific real model. Skipped without the modeling stack.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from pipelines.anomaly import anomaly_score, build_detector  # noqa: E402
from pipelines.features import aggregate_window, feature_names  # noqa: E402
from pipelines.metropt3_schema import ANALOG_SENSORS, DIGITAL_SIGNALS  # noqa: E402
from serving.model import ModelBundle, score_features  # noqa: E402

NORMAL = 5.0


def _window(level: float = NORMAL, n: int = 6, seed: int = 0) -> list[dict]:
    """A window of n readings with all analog channels around ``level``."""
    rng = np.random.default_rng(int(level * 100) + seed)
    out = []
    for _ in range(n):
        r = {c: level + float(rng.normal(0, 0.03)) for c in ANALOG_SENSORS}
        r |= {c: 1.0 for c in DIGITAL_SIGNALS}
        out.append(r)
    return out


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    feats = feature_names()
    rows = [aggregate_window(pd.DataFrame(_window(seed=i))) for i in range(150)]
    X = pd.DataFrame(rows)[feats].to_numpy()
    det = build_detector(n_estimators=50).fit(X)
    threshold = float(np.quantile(anomaly_score(det, X), 0.99))
    return ModelBundle(det, threshold, feats, "behavior-v1")


def _score(bundle, readings) -> float:
    return score_features(bundle, aggregate_window(pd.DataFrame(readings)))[0]


def test_minimum_functionality_anomalous_scores_above_normal(bundle):
    # A window far from trained-normal operation must look more anomalous than an
    # in-distribution one.
    assert _score(bundle, _window(level=10.0)) > _score(bundle, _window(level=NORMAL))


def test_directional_expectation_score_rises_with_deviation(bundle):
    # Any clear deviation from normal operation scores as more anomalous. (We do NOT
    # assert monotonicity *between* far-out points — Isolation-Forest path length
    # isn't monotone in the tail, so that would be a false expectation.)
    s_normal = _score(bundle, _window(level=NORMAL))
    assert _score(bundle, _window(level=8.0)) > s_normal
    assert _score(bundle, _window(level=10.0)) > s_normal


def test_invariance_to_reading_order(bundle):
    # Window aggregation (mean/std/min/max/duty) is order-invariant, so reversing
    # the readings must give an identical score.
    w = _window(level=7.0)
    assert _score(bundle, w) == pytest.approx(_score(bundle, list(reversed(w))))


def test_stability_to_tiny_noise(bundle):
    # A sub-noise perturbation must barely move the score (no chaotic flips).
    rng = np.random.default_rng(0)
    base = _window(level=NORMAL)
    jittered = [
        {k: v + float(rng.normal(0, 1e-4)) if k in ANALOG_SENSORS else v for k, v in r.items()}
        for r in base
    ]
    assert abs(_score(bundle, base) - _score(bundle, jittered)) < 1e-3
