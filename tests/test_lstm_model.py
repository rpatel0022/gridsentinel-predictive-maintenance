"""Tests for the LSTM sequence builder (numpy core) + a guarded TF smoke test."""

import numpy as np
import pytest
from pipelines.lstm_model import make_sequences


def test_sequence_shape_is_3d():
    X = np.arange(30).reshape(10, 3)  # 10 windows, 3 features
    out = make_sequences(X, k=4)
    assert out.shape == (7, 4, 3)  # (n-k+1, timesteps, features)


def test_sequence_contents_are_consecutive_windows():
    X = np.arange(12).reshape(6, 2)
    out = make_sequences(X, k=2)
    assert np.array_equal(out[0], X[0:2])
    assert np.array_equal(out[1], X[1:3])


def test_too_short_returns_empty_3d():
    assert make_sequences(np.zeros((2, 3)), k=5).shape == (0, 5, 3)


def test_invalid_k():
    with pytest.raises(ValueError):
        make_sequences(np.zeros((4, 2)), k=0)


def test_lstm_builds_and_predicts():
    pytest.importorskip("tensorflow")
    from pipelines.lstm_model import _build_lstm

    model = _build_lstm(n_steps=4, n_features=3)
    out = model.predict(np.zeros((2, 4, 3)), verbose=0)
    assert out.shape == (2, 1)
