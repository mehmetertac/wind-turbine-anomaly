"""Tests for SCADA cleaning and healthy masks."""

from datetime import datetime, timezone

import pandas as pd

from wind_turbine_anomaly.config import FEATURE_COLUMNS, GearboxFailure
from wind_turbine_anomaly.data.clean import (
    clean_turbine_df,
    get_failure_for_turbine,
    healthy_training_mask,
)


def _sample_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    data = {col: range(n) for col in FEATURE_COLUMNS}
    return pd.DataFrame(data, index=idx)


def test_clean_turbine_df_drops_nan():
    df = _sample_df(5)
    df.iloc[2, 0] = float("nan")
    cleaned = clean_turbine_df(df)
    assert len(cleaned) == 4


def test_clean_turbine_df_missing_column_raises():
    df = _sample_df(3).drop(columns=["Gear_Oil_Temp_Avg"])
    try:
        clean_turbine_df(df)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_healthy_training_mask_excludes_buffer():
    idx = pd.date_range("2016-01-01", "2016-12-31", freq="10min", tz="UTC")
    failure = GearboxFailure(
        turbine_id="T01",
        timestamp=datetime(2016, 7, 18, 2, 10, tzinfo=timezone.utc),
        remarks="test",
    )
    mask = healthy_training_mask(idx, failure, buffer_days=90)
    cutoff = pd.Timestamp(failure.timestamp) - pd.Timedelta(days=90)
    assert mask.sum() == (idx < cutoff).sum()
    assert mask.loc[idx >= cutoff].sum() == 0


def test_healthy_training_mask_no_failure():
    idx = pd.date_range("2016-01-01", periods=100, freq="10min", tz="UTC")
    mask = healthy_training_mask(idx, None)
    assert mask.sum() == 90


def test_get_failure_for_turbine():
    failures = [
        GearboxFailure("T01", datetime(2016, 7, 18, tzinfo=timezone.utc), "a"),
        GearboxFailure("T06", datetime(2017, 10, 17, tzinfo=timezone.utc), "b"),
    ]
    assert get_failure_for_turbine("T06", failures).turbine_id == "T06"
    assert get_failure_for_turbine("T07", failures) is None
