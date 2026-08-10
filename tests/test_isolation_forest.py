"""Tests for Isolation Forest baseline."""

import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import FEATURE_COLUMNS
from wind_turbine_anomaly.models.isolation_forest import (
    fit_isolation_forest,
    threshold_from_training,
)


def test_fit_and_score():
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    normal = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    df = pd.DataFrame(normal, index=idx, columns=FEATURE_COLUMNS)

    pipeline = fit_isolation_forest(df.iloc[:400], FEATURE_COLUMNS, contamination=0.05)
    scores = pipeline.score(df)

    assert len(scores) == n
    assert scores.name == "anomaly_score"
    # Injected spike should score higher than median
    df_spike = df.copy()
    df_spike.iloc[-1] = df_spike.iloc[-1] + 20
    spike_score = pipeline.score(df_spike.iloc[[-1]]).iloc[0]
    median_score = pipeline.score(df.iloc[400:]).median()
    assert spike_score > median_score


def test_threshold_from_training():
    scores = pd.Series([0.1, 0.2, 0.3, 0.4, 1.0])
    t = threshold_from_training(scores, percentile=80)
    assert t >= 0.4
    assert t <= 1.0
