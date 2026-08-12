"""Tests for LSTM autoencoder baseline."""

import numpy as np
import pandas as pd
import pytest

from wind_turbine_anomaly.config import FEATURE_COLUMNS
from wind_turbine_anomaly.models.sequences import build_sliding_windows


def test_build_sliding_windows_shape():
    X = np.random.default_rng(0).normal(size=(50, 7))
    windows, end_indices = build_sliding_windows(X, window_size=10)
    assert windows.shape == (41, 10, 7)
    assert len(end_indices) == 41
    assert end_indices[-1] == 49


def test_fit_and_score():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"PyTorch unavailable: {exc}")

    from wind_turbine_anomaly.models.lstm_autoencoder import fit_lstm_autoencoder

    rng = np.random.default_rng(42)
    n = 300
    window_size = 24
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    normal = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    df = pd.DataFrame(normal, index=idx, columns=FEATURE_COLUMNS)

    pipeline = fit_lstm_autoencoder(
        df.iloc[:200],
        FEATURE_COLUMNS,
        window_size=window_size,
        epochs=3,
        patience=2,
        batch_size=32,
    )
    scores = pipeline.score(df)

    assert scores.name == "anomaly_score"
    assert len(scores) == n - window_size + 1

    df_degraded = df.copy()
    df_degraded.iloc[-window_size:] += 5
    tail_scores = pipeline.score(df_degraded.iloc[-window_size:])
    healthy_scores = pipeline.score(df.iloc[200 : 200 + window_size])
    assert tail_scores.iloc[-1] > healthy_scores.median()
