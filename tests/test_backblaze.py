"""Tests for the Backblaze fleet-reliability pipeline (pandas core, lean CI)."""

import pytest

pd = pytest.importorskip("pandas")

from pipelines.backblaze import (  # noqa: E402
    annualized_failure_rates,
    build_cohort,
    target_encode,
)


def _frame():
    # Reference T = 2022-01-01, horizon 365d.
    return pd.DataFrame(
        {
            "serial_number": ["A", "B", "C", "D", "E"],
            "capacity_tb": [4.0, 8.0, 4.0, 12.0, 4.0],
            "model": ["m1", "m2", "m1", "m2", "m1"],
            "min_date": pd.to_datetime(
                ["2021-06-01", "2021-06-01", "2021-06-01", "2023-01-01", "2020-01-01"]
            ),
            "max_date": pd.to_datetime(
                ["2022-06-01", "2023-06-01", "2022-06-01", "2023-06-01", "2021-06-01"]
            ),
            "failed": [1, 0, 0, 1, 1],
        }
    )


def test_cohort_is_leakage_safe():
    X, y = build_cohort(_frame(), reference="2022-01-01", horizon_days=365)
    # A = alive at T, fails within horizon -> positive.
    # B = alive at T, survives -> negative.
    # C = alive at T but left (not failed) within horizon -> dropped.
    # D = installed after T -> excluded. E = gone before T -> excluded.
    assert len(X) == 2
    assert int(y.sum()) == 1
    assert set(X["model"]) == {"m1", "m2"}


def test_cohort_age_measured_from_install():
    X, _ = build_cohort(_frame(), reference="2022-01-01", horizon_days=365)
    # age = T - min_date = 2022-01-01 - 2021-06-01 = 214 days, same for both survivors.
    assert (X["age_days"] == 214).all()


def test_target_encode_unseen_model_gets_global_rate():
    enc = target_encode(["m1", "m1", "m2"], [1, 0, 0], ["m_unseen"])
    assert enc[0] == pytest.approx(1 / 3)  # global train rate


def test_target_encode_orders_by_reliability():
    models = ["bad"] * 10 + ["good"] * 10
    y = [1] * 10 + [0] * 10
    enc = target_encode(models, y, ["bad", "good"], smoothing=1.0)
    assert enc[0] > enc[1]  # the all-failure model encodes higher


def test_afr_matches_failures_over_drive_years():
    df = _frame()
    afr = annualized_failure_rates(df, min_drives=1)
    # m1 has 3 drives, 2 failures; AFR = 2 / (sum of drive-years) should be positive.
    assert afr.loc["m1", "failures"] == 2
    assert afr.loc["m1", "afr"] > 0


def test_schema_rejects_negative_lifetime():
    pytest.importorskip("pandera")
    from pipelines.backblaze_schema import validate

    bad = _frame()
    bad.loc[0, "max_date"] = pd.Timestamp("2019-01-01")  # before min_date
    with pytest.raises(Exception):  # noqa: B017 - pandera SchemaError(s)
        validate(bad)
