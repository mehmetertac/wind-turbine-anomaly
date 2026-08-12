"""Tests for dense autoencoder baseline."""

import numpy as np
import pandas as pd
import pytest

from wind_turbine_anomaly.config import FEATURE_COLUMNS


def test_fit_and_score():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"PyTorch unavailable: {exc}")

    from wind_turbine_anomaly.models.dense_autoencoder import fit_dense_autoencoder

    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    normal = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    df = pd.DataFrame(normal, index=idx, columns=FEATURE_COLUMNS)

    pipeline = fit_dense_autoencoder(
        df.iloc[:400],
        FEATURE_COLUMNS,
        epochs=3,
        patience=2,
        batch_size=64,
    )
    scores = pipeline.score(df)

    assert len(scores) == n
    assert scores.name == "anomaly_score"

    df_spike = df.copy()
    df_spike.iloc[-1] = df_spike.iloc[-1] + 20
    spike_score = pipeline.score(df_spike.iloc[[-1]]).iloc[0]
    median_score = pipeline.score(df.iloc[400:]).median()
    assert spike_score > median_score
