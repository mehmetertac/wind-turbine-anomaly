"""Shared anomaly score persistence and threshold helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_scores(scores: pd.Series, path: Path | str) -> None:
    """Persist anomaly score time series to parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_frame().to_parquet(path)


def threshold_from_training(
    train_scores: pd.Series,
    percentile: float = 99.0,
) -> float:
    """Compute alarm threshold from healthy training score distribution."""
    return float(np.percentile(train_scores.values, percentile))
