"""Feature engineering on physics thermal residuals for hybrid detection."""

from __future__ import annotations

import pandas as pd

from wind_turbine_anomaly.config import (
    RESIDUAL_EWMA_SPAN,
    RESIDUAL_ROLLING_WINDOWS,
    THERMAL_TARGET_COLUMNS,
)
from wind_turbine_anomaly.models.gearbox_thermal import GearboxThermalModel

OIL_TARGET = THERMAL_TARGET_COLUMNS[0]
BEAR_TARGET = THERMAL_TARGET_COLUMNS[1]


def degradation_signal(residuals: pd.Series) -> pd.Series:
    """Map physics residual to anomaly direction (higher = hotter than expected)."""
    return -residuals


def compute_dual_residuals(
    df: pd.DataFrame,
    oil_model: GearboxThermalModel,
    bear_model: GearboxThermalModel,
) -> pd.DataFrame:
    """Return aligned oil/bear residual series on operating rows."""
    return pd.DataFrame(
        {
            "oil_residual": oil_model.residual(df),
            "bear_residual": bear_model.residual(df),
        },
        index=df.index,
    )


def ewma_feature(series: pd.Series, span: int = RESIDUAL_EWMA_SPAN) -> pd.Series:
    """EWMA control-chart style smoothed degradation signal."""
    return degradation_signal(series).ewm(span=span, adjust=False).mean()


def rolling_features(
    series: pd.Series,
    prefix: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling mean/std of degradation signal at multiple horizons."""
    windows = windows or RESIDUAL_ROLLING_WINDOWS
    signal = degradation_signal(series)
    parts: dict[str, pd.Series] = {}
    for window in windows:
        parts[f"{prefix}_roll_mean_{window}"] = signal.rolling(window).mean()
        parts[f"{prefix}_roll_std_{window}"] = signal.rolling(window).std()
    return pd.DataFrame(parts, index=series.index)


def build_residual_feature_frame(
    df: pd.DataFrame,
    oil_model: GearboxThermalModel,
    bear_model: GearboxThermalModel,
    ewma_span: int = RESIDUAL_EWMA_SPAN,
    rolling_windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Per-timestamp feature vector from oil/bear residuals.

    Features: current degradation signal, EWMA, rolling mean/std for each target.
    Rows with incomplete rolling windows are dropped.
    """
    rolling_windows = rolling_windows or RESIDUAL_ROLLING_WINDOWS
    residuals = compute_dual_residuals(df, oil_model, bear_model)

    oil_deg = degradation_signal(residuals["oil_residual"])
    bear_deg = degradation_signal(residuals["bear_residual"])

    features = pd.DataFrame(
        {
            "oil_deg": oil_deg,
            "bear_deg": bear_deg,
            "oil_ewma": ewma_feature(residuals["oil_residual"], span=ewma_span),
            "bear_ewma": ewma_feature(residuals["bear_residual"], span=ewma_span),
        },
        index=df.index,
    )
    features = pd.concat(
        [
            features,
            rolling_features(residuals["oil_residual"], "oil", rolling_windows),
            rolling_features(residuals["bear_residual"], "bear", rolling_windows),
        ],
        axis=1,
    )
    return features.dropna(how="any")


__all__ = [
    "BEAR_TARGET",
    "OIL_TARGET",
    "build_residual_feature_frame",
    "compute_dual_residuals",
    "degradation_signal",
    "ewma_feature",
    "rolling_features",
]
