"""Physics-residual hybrid detector: thermal model + IF on residual features."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wind_turbine_anomaly.config import (
    DEFAULT_CONTAMINATION,
    RESIDUAL_EWMA_SPAN,
    RESIDUAL_ROLLING_WINDOWS,
    THERMAL_DRIVER_COLUMNS,
    THERMAL_TARGET_COLUMNS,
)
from wind_turbine_anomaly.models.gearbox_thermal import (
    GearboxThermalModel,
    fit_gearbox_thermal_with_selection,
)
from wind_turbine_anomaly.models.isolation_forest import (
    IsolationForestPipeline,
    fit_isolation_forest,
)
from wind_turbine_anomaly.models.residual_features import build_residual_feature_frame


@dataclass
class PhysicsHybridPipeline:
    """Thermal normal-behavior models + IF on residual-window features."""

    oil_model: GearboxThermalModel
    bear_model: GearboxThermalModel
    if_pipeline: IsolationForestPipeline
    feature_columns: list[str]
    ewma_span: int = RESIDUAL_EWMA_SPAN
    rolling_windows: list[int] | None = None

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Return anomaly scores (higher = more anomalous) indexed by timestamp."""
        features = build_residual_feature_frame(
            X,
            self.oil_model,
            self.bear_model,
            ewma_span=self.ewma_span,
            rolling_windows=self.rolling_windows,
        )
        if features.empty:
            return pd.Series(dtype=float, name="anomaly_score")
        raw = self.if_pipeline.score(features)
        return pd.Series(raw.values, index=features.index, name="anomaly_score")


def fit_physics_hybrid(
    X_train: pd.DataFrame,
    feature_columns: list[str] | None = None,
    contamination: float = DEFAULT_CONTAMINATION,
    ewma_span: int = RESIDUAL_EWMA_SPAN,
    rolling_windows: list[int] | None = None,
) -> PhysicsHybridPipeline:
    """
    Fit per-turbine physics hybrid: thermal models + IF on residual features.

    ``feature_columns`` is accepted for baseline-runner API compatibility but
    ignored — thermal targets/drivers are fixed in config.
    """
    del feature_columns  # API compatibility with run_detector_baseline
    rolling_windows = rolling_windows or RESIDUAL_ROLLING_WINDOWS
    oil_target, bear_target = THERMAL_TARGET_COLUMNS

    oil_model, _, _, _ = fit_gearbox_thermal_with_selection(
        X_train, oil_target, THERMAL_DRIVER_COLUMNS
    )
    bear_model, _, _, _ = fit_gearbox_thermal_with_selection(
        X_train, bear_target, THERMAL_DRIVER_COLUMNS
    )

    train_features = build_residual_feature_frame(
        X_train,
        oil_model,
        bear_model,
        ewma_span=ewma_span,
        rolling_windows=rolling_windows,
    )
    if len(train_features) < 50:
        raise ValueError(
            f"Insufficient residual feature rows for IF training ({len(train_features)})"
        )

    feature_cols = train_features.columns.tolist()
    if_pipeline = fit_isolation_forest(
        train_features,
        feature_cols,
        contamination=contamination,
    )

    return PhysicsHybridPipeline(
        oil_model=oil_model,
        bear_model=bear_model,
        if_pipeline=if_pipeline,
        feature_columns=feature_cols,
        ewma_span=ewma_span,
        rolling_windows=rolling_windows,
    )


__all__ = [
    "PhysicsHybridPipeline",
    "fit_physics_hybrid",
]
