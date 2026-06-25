"""Tests for the sequence-stacking helper (numpy-only, lean CI)."""

import numpy as np
import pytest
from pipelines.sequence_model import stack_sequences


def test_stack_shape():
    X = np.arange(20).reshape(10, 2)  # 10 windows, 2 features
    out = stack_sequences(X, k=3)
    assert out.shape == (8, 6)  # n-k+1 rows, k*features cols


def test_stack_contents_are_consecutive_windows():
    X = np.arange(12).reshape(6, 2)
    out = stack_sequences(X, k=2)
    # First stacked row = windows 0 and 1 flattened.
    assert list(out[0]) == [0, 1, 2, 3]


def test_k_one_is_identity():
    X = np.arange(8).reshape(4, 2)
    assert np.array_equal(stack_sequences(X, k=1), X)


def test_too_short_returns_empty():
    assert stack_sequences(np.zeros((2, 3)), k=5).shape == (0, 15)


def test_invalid_k():
    with pytest.raises(ValueError):
        stack_sequences(np.zeros((4, 2)), k=0)
