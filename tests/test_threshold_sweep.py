"""Tests for threshold sweep and headline claim."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from wind_turbine_anomaly.config import GearboxFailure, PHYSICS_HYBRID_DETECTOR
from wind_turbine_anomaly.eval.protocol import TurbineEvalResult, evaluate_turbine
from wind_turbine_anomaly.eval.threshold_sweep import (
    build_headline_claim,
    median_lead_time_failure_turbines,
    threshold_at_percentile,
)


def _make_failure_turbine_scores(
    alarm_day: int,
    baseline: float = 0.1,
    alarm_level: float = 5.0,
) -> tuple[pd.Series, pd.Series]:
    """Build train + post-train scores with a known pre-failure alarm."""
    idx = pd.date_range("2016-01-01", periods=200, freq="10min", tz="UTC")
    train = pd.Series([baseline] * 80, index=idx[:80])
    post = pd.Series([baseline] * len(idx[80:]), index=idx[80:])
    alarm_start = idx[80 + alarm_day * 6]
    post.loc[alarm_start : alarm_start + pd.Timedelta(hours=1)] = alarm_level
    return train, post


def test_threshold_at_percentile():
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert threshold_at_percentile(scores, 100.0) == pytest.approx(5.0)


def test_median_lead_time_failure_turbines_one_nan():
    results = [
        TurbineEvalResult(
            turbine_id="T01",
            has_gearbox_failure=True,
            failure_time=pd.Timestamp("2016-07-18", tz="UTC"),
            first_alarm_time=pd.Timestamp("2016-06-01", tz="UTC"),
            lead_time_days=47.0,
            successful_warning=True,
            false_alarm_episodes=0,
            scored_days=100.0,
            precision_at_horizon=1.0,
        ),
        TurbineEvalResult(
            turbine_id="T06",
            has_gearbox_failure=True,
            failure_time=pd.Timestamp("2017-10-17", tz="UTC"),
            first_alarm_time=None,
            lead_time_days=None,
            successful_warning=False,
            false_alarm_episodes=0,
            scored_days=100.0,
            precision_at_horizon=None,
        ),
    ]
    median, n = median_lead_time_failure_turbines(results)
    assert median == 47.0
    assert n == 1


def test_sweep_lower_threshold_more_false_alarms():
    """Lower percentile threshold should not decrease false-alarm count."""
    failure = GearboxFailure(
        "T01",
        datetime(2016, 1, 10, tzinfo=timezone.utc),
        "test",
    )
    train, post = _make_failure_turbine_scores(alarm_day=5)

    results_by_pct: list[int] = []
    for pct in [90, 95, 99]:
        threshold = threshold_at_percentile(train, pct)
        result = evaluate_turbine(
            "T01",
            post,
            threshold=threshold,
            failure=failure,
            horizon_days=30,
            persistence_samples=3,
            cooldown_hours=0,
        )
        results_by_pct.append(result.false_alarm_episodes)

    assert results_by_pct[0] >= results_by_pct[-1]


def test_build_headline_claim():
    sweep_df = pd.DataFrame(
        [
            {
                "detector": PHYSICS_HYBRID_DETECTOR,
                "threshold_percentile": 99.0,
                "false_alarms_per_turbine_year": 12.4,
                "median_lead_time_days": 33.0,
                "n_failure_turbines_with_alarm": 2,
                "T01_lead_time_days": 18.3,
                "T06_lead_time_days": 47.6,
            },
            {
                "detector": "isolation_forest",
                "threshold_percentile": 99.0,
                "false_alarms_per_turbine_year": 106.0,
                "median_lead_time_days": 18.3,
                "n_failure_turbines_with_alarm": 1,
                "T01_lead_time_days": 18.3,
                "T06_lead_time_days": None,
            },
        ]
    )
    claim = build_headline_claim(
        sweep_df,
        best_pure_ml_detector="isolation_forest",
        data_note="test data",
    )
    assert claim["horizon_days"] == 30
    assert claim["operating_threshold_percentile"] == 99.0
    assert claim["physics_hybrid"]["median_lead_time_days"] == 33.0
    assert claim["best_pure_ml"]["detector"] == "isolation_forest"
    assert "33.0 days" in claim["headline"]
    assert "12.4 false alarms" in claim["headline"]

    # JSON-serializable
    json.dumps(claim)
