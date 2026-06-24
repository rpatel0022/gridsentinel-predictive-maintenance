"""Tests for the temporal CV splitter — the leakage guard."""

import pytest

from gridsentinel.cv import temporal_splits


def test_train_always_precedes_test():
    for train_idx, test_idx in temporal_splits(100, n_splits=5):
        assert max(train_idx) < min(test_idx), "train must be strictly before test"


def test_folds_are_forward_chaining():
    splits = list(temporal_splits(120, n_splits=4))
    assert len(splits) == 4
    # Training sets grow monotonically as folds advance.
    train_sizes = [len(tr) for tr, _ in splits]
    assert train_sizes == sorted(train_sizes)


def test_embargo_creates_gap():
    embargo = 7
    for train_idx, test_idx in temporal_splits(200, n_splits=5, embargo=embargo):
        assert min(test_idx) - max(train_idx) - 1 == embargo


def test_no_embargo_is_contiguous():
    for train_idx, test_idx in temporal_splits(200, n_splits=5, embargo=0):
        assert min(test_idx) == max(train_idx) + 1


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        list(temporal_splits(100, n_splits=0))
    with pytest.raises(ValueError):
        list(temporal_splits(100, n_splits=5, embargo=-1))
    with pytest.raises(ValueError):
        list(temporal_splits(2, n_splits=5))
