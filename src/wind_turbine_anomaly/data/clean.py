"""SCADA cleaning and healthy-period masking."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from wind_turbine_anomaly.utils import to_utc

from wind_turbine_anomaly.config import (
    DEFAULT_BUFFER_DAYS,
    FEATURE_COLUMNS,
    GearboxFailure,
    MIN_POWER_KW,
    POWER_COLUMN,
)


def clean_turbine_df(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    min_power_kw: float = MIN_POWER_KW,
) -> pd.DataFrame:
    """Clean per-turbine SCADA: dedupe, select features, drop NaNs, optional power filter."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    out = df.copy()
    out = out[~out.index.duplicated(keep="first")].sort_index()
    missing = [c for c in feature_columns if c not in out.columns]
    if missing:
        raise ValueError(f"Missing SCADA columns: {missing}")

    out = out[feature_columns]
    if min_power_kw > 0 and POWER_COLUMN in out.columns:
        out = out[out[POWER_COLUMN] > min_power_kw]
    out = out.dropna(how="any")
    return out


def healthy_training_mask(
    index: pd.DatetimeIndex,
    failure: GearboxFailure | None,
    buffer_days: int = DEFAULT_BUFFER_DAYS,
) -> pd.Series:
    """
    Boolean mask for healthy training rows.

    True for timestamps strictly before (failure_time - buffer_days).
    If no failure is provided, use all but the last 10% of rows.
    """
    if failure is None:
        cutoff_idx = int(len(index) * 0.9)
        if cutoff_idx <= 0:
            return pd.Series(False, index=index)
        cutoff = index[cutoff_idx - 1]
        return pd.Series(index <= cutoff, index=index)

    cutoff = to_utc(failure.timestamp) - timedelta(days=buffer_days)
    return pd.Series(index < cutoff, index=index)


def get_failure_for_turbine(
    turbine_id: str,
    failures: list[GearboxFailure],
) -> GearboxFailure | None:
    """Return the first gearbox failure for a turbine, if any."""
    matches = [f for f in failures if f.turbine_id == turbine_id]
    if not matches:
        return None
    return min(matches, key=lambda f: f.timestamp)
