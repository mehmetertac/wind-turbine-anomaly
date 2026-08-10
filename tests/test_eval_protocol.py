"""Tests for evaluation protocol."""

from datetime import datetime, timezone

import pandas as pd

from wind_turbine_anomaly.config import GearboxFailure
from wind_turbine_anomaly.eval.protocol import (
    compute_lead_time_days,
    count_false_alarms,
    detect_alarm_episodes,
    evaluate_turbine,
    false_alarms_per_turbine_year,
    is_false_alarm,
    results_to_dict,
)
from wind_turbine_anomaly.eval.protocol import AlarmEpisode


def test_detect_alarm_episodes_persistence():
    idx = pd.date_range("2016-01-01", periods=20, freq="10min", tz="UTC")
    scores = pd.Series([0.0] * 10 + [2.0] * 6 + [0.0] * 4, index=idx)
    episodes = detect_alarm_episodes(scores, threshold=1.0, persistence_samples=3)
    assert len(episodes) == 1
    assert episodes[0].start == idx[10]


def test_compute_lead_time_days():
    failure = pd.Timestamp("2016-07-18", tz="UTC")
    episodes = [
        AlarmEpisode(
            start=pd.Timestamp("2016-06-01", tz="UTC"),
            end=pd.Timestamp("2016-06-01", tz="UTC"),
        )
    ]
    first, days = compute_lead_time_days(episodes, failure)
    assert first is not None
    assert days is not None
    assert days > 40


def test_is_false_alarm_with_failure():
    failure = pd.Timestamp("2016-07-18", tz="UTC")
    alarm = pd.Timestamp("2016-07-10", tz="UTC")
    assert not is_false_alarm(alarm, failure, horizon_days=30)
    early_alarm = pd.Timestamp("2016-01-01", tz="UTC")
    assert is_false_alarm(early_alarm, failure, horizon_days=30)


def test_evaluate_turbine():
    idx = pd.date_range("2016-01-01", periods=100, freq="10min", tz="UTC")
    scores = pd.Series([0.1] * 80 + [5.0] * 20, index=idx)
    failure = GearboxFailure(
        "T01",
        datetime(2016, 1, 1, 16, 0, tzinfo=timezone.utc),
        "test",
    )
    result = evaluate_turbine(
        "T01",
        scores,
        threshold=1.0,
        failure=failure,
        horizon_days=1,
        persistence_samples=3,
        cooldown_hours=0,
    )
    assert result.lead_time_days is not None
    assert result.has_gearbox_failure


def test_false_alarms_per_turbine_year():
    idx = pd.date_range("2016-01-01", periods=100, freq="10min", tz="UTC")
    results = [
        evaluate_turbine("T07", pd.Series([0.1] * 100, index=idx), 1.0, None),
    ]
    rate = false_alarms_per_turbine_year(results)
    assert rate == 0.0

    scores = pd.Series([5.0] * 10 + [0.1] * 90, index=idx)
    results = [
        evaluate_turbine(
            "T07",
            scores,
            1.0,
            None,
            persistence_samples=3,
            cooldown_hours=0,
        )
    ]
    rate = false_alarms_per_turbine_year(results)
    assert rate > 0


def test_results_to_dict():
    idx = pd.date_range("2016-01-01", periods=10, freq="10min", tz="UTC")
    results = [
        evaluate_turbine("T07", pd.Series([0.1] * 10, index=idx), 1.0, None)
    ]
    d = results_to_dict(results, horizon_days=30)
    assert "turbines" in d
    assert d["horizon_days"] == 30
