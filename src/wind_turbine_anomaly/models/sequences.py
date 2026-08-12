"""Sliding-window helpers for sequence models."""

from __future__ import annotations

import numpy as np


def build_sliding_windows(
    X_scaled: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows from a 2D array (n_samples, n_features).

    Returns (windows, end_indices) where windows has shape
    (n_windows, window_size, n_features) and end_indices maps each window
    to its end row index in X_scaled.
    """
    n_samples, _n_features = X_scaled.shape
    if n_samples < window_size:
        return np.empty((0, window_size, X_scaled.shape[1])), np.empty(0, dtype=int)

    view = np.lib.stride_tricks.sliding_window_view(
        X_scaled, window_size, axis=0
    )
    windows = np.moveaxis(view, -1, 1)
    end_indices = np.arange(window_size - 1, n_samples)
    return windows, end_indices
