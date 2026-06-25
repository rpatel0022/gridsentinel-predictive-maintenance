"""Temporal cross-validation with an embargo gap — the leakage guard.

A production-ML screen *will* ask "how do you know you didn't leak?". MetroPT-3 is a
single device, so there are no groups to hold out; the leakage risk is purely
temporal — testing on the past after training on the future, or letting rows that
straddle the train/test boundary share information. This module answers both:

* **Forward-chaining splits** — every test fold is strictly *after* its training
  data, never before. No future ever trains a model that is judged on the past.
* **Embargo** — a gap of ``embargo`` windows is dropped between the end of train and
  the start of test. Even with non-overlapping feature windows, the failure
  *labels* look ahead (``warn_hours``), so a window just before the boundary can be
  labelled from a failure that lands just after it; the embargo removes that bleed.

Deliberately dependency-light (stdlib + a tiny index helper) so the leakage logic
is auditable without reading sklearn internals.
"""

from __future__ import annotations

from collections.abc import Iterator


def temporal_splits(
    n: int,
    *,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield forward-chaining ``(train_idx, test_idx)`` splits over ``n`` rows.

    Rows are assumed already sorted in time. The tail of the series is divided into
    ``n_splits`` contiguous test folds; each fold trains on everything before it
    (minus the embargo gap) and tests on the fold itself — so train always precedes
    test in time.

    Args:
        n: Number of (time-ordered) rows.
        n_splits: Number of forward-chaining folds.
        embargo: Rows to drop between the end of train and the start of test.

    Yields:
        ``(train_idx, test_idx)`` lists of integer positions. Folds whose training
        set is emptied by the embargo are skipped.

    Raises:
        ValueError: If ``n_splits < 1``, ``embargo < 0``, or ``n`` is too small to
            form the requested folds.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    if n < n_splits + 1:
        raise ValueError(f"need at least n_splits+1={n_splits + 1} rows, got {n}")

    fold = n // (n_splits + 1)
    if fold == 0:
        raise ValueError(f"{n} rows is too few for {n_splits} splits")

    for i in range(1, n_splits + 1):
        test_start = fold * i
        test_end = n if i == n_splits else fold * (i + 1)
        train_end = max(0, test_start - embargo)
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        if not train_idx or not test_idx:
            continue
        yield train_idx, test_idx
