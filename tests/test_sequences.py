"""Tests for sliding-window sequence helpers."""

import numpy as np

from wind_turbine_anomaly.models.sequences import build_sliding_windows


def test_build_sliding_windows_empty():
    X = np.random.default_rng(0).normal(size=(5, 7))
    windows, end_indices = build_sliding_windows(X, window_size=10)
    assert len(windows) == 0
    assert len(end_indices) == 0
