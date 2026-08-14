"""Physics-informed normal-behavior model for gearbox oil/bearing temperature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wind_turbine_anomaly.config import (
    POWER_COLUMN,
    THERMAL_DRIVER_COLUMNS,
    THERMAL_GBM_RMSE_IMPROVEMENT,
    THERMAL_TRAIN_FRACTION,
)

ModelKind = Literal["linear", "gbm"]

THERMAL_VALIDATION_THRESHOLDS = {
    "max_abs_mean_residual_c": 0.5,
    "max_driver_correlation": 0.1,
    "max_time_trend_per_day_c": 0.01,
}


@dataclass
class GearboxThermalModel:
    """Per-turbine thermal model: predict expected gearbox temperature."""

    model: RegressorMixin
    model_kind: ModelKind
    driver_columns: list[str]
    target_column: str
    feature_names: list[str] | None = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predicted temperature (°C)."""
        X_arr = self._build_features(X)
        pred = self.model.predict(X_arr)
        return pd.Series(pred, index=X.index, name=f"{self.target_column}_pred")

    def residual(self, X: pd.DataFrame) -> pd.Series:
        """Return predicted minus actual temperature (°C)."""
        actual = X[self.target_column]
        predicted = self.predict(X)
        return pd.Series(
            predicted.values - actual.values,
            index=X.index,
            name=f"{self.target_column}_residual",
        )

    def residual_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return actual, predicted, and residual columns."""
        actual = X[self.target_column]
        predicted = self.predict(X)
        residual = predicted - actual
        return pd.DataFrame(
            {
                "actual": actual,
                "predicted": predicted,
                "residual": residual,
            },
            index=X.index,
        )

    def _build_features(self, X: pd.DataFrame) -> np.ndarray:
        drivers = X[self.driver_columns]
        if self.model_kind == "linear":
            power = drivers[POWER_COLUMN].values.reshape(-1, 1)
            rpm = drivers["Rtr_RPM_Avg"].values.reshape(-1, 1)
            nac = drivers["Nac_Temp_Avg"].values.reshape(-1, 1)
            return np.hstack([power, rpm, nac, power**2])
        return drivers.values


def healthy_train_validate_split(
    healthy_df: pd.DataFrame,
    train_fraction: float = THERMAL_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered split of healthy rows: first fraction train, remainder validate."""
    if healthy_df.empty:
        return healthy_df, healthy_df.iloc[0:0]
    n_train = max(1, int(len(healthy_df) * train_fraction))
    if n_train >= len(healthy_df):
        n_train = max(1, len(healthy_df) - 1)
    train_df = healthy_df.iloc[:n_train]
    val_df = healthy_df.iloc[n_train:]
    return train_df, val_df


def _fit_linear(train_df: pd.DataFrame, target_column: str, driver_columns: list[str]) -> GearboxThermalModel:
    power = train_df[POWER_COLUMN].values.reshape(-1, 1)
    rpm = train_df["Rtr_RPM_Avg"].values.reshape(-1, 1)
    nac = train_df["Nac_Temp_Avg"].values.reshape(-1, 1)
    X_train = np.hstack([power, rpm, nac, power**2])
    y_train = train_df[target_column].values

    pipeline: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X_train, y_train)
    feature_names = [POWER_COLUMN, "Rtr_RPM_Avg", "Nac_Temp_Avg", f"{POWER_COLUMN}_sq"]
    return GearboxThermalModel(
        model=pipeline,
        model_kind="linear",
        driver_columns=driver_columns,
        target_column=target_column,
        feature_names=feature_names,
    )


def _fit_gbm(train_df: pd.DataFrame, target_column: str, driver_columns: list[str]) -> GearboxThermalModel:
    X_train = train_df[driver_columns].values
    y_train = train_df[target_column].values
    model = GradientBoostingRegressor(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return GearboxThermalModel(
        model=model,
        model_kind="gbm",
        driver_columns=driver_columns,
        target_column=target_column,
        feature_names=driver_columns,
    )


def _validation_rmse(model: GearboxThermalModel, val_df: pd.DataFrame) -> float:
    if val_df.empty:
        return float("inf")
    y_true = val_df[model.target_column].values
    y_pred = model.predict(val_df).values
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_gearbox_thermal(
    train_df: pd.DataFrame,
    target_column: str,
    driver_columns: list[str] | None = None,
) -> GearboxThermalModel:
    """
    Fit normal-behavior thermal model, selecting linear vs GBM on validation RMSE.

    Expects train_df to be the time-ordered training portion of healthy data.
    """
    driver_columns = driver_columns or THERMAL_DRIVER_COLUMNS
    linear_model = _fit_linear(train_df, target_column, driver_columns)
    return linear_model


def fit_gearbox_thermal_with_selection(
    healthy_df: pd.DataFrame,
    target_column: str,
    driver_columns: list[str] | None = None,
    train_fraction: float = THERMAL_TRAIN_FRACTION,
    gbm_improvement: float = THERMAL_GBM_RMSE_IMPROVEMENT,
) -> tuple[GearboxThermalModel, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Fit on healthy data with time-ordered train/val split and model selection.

    Returns (final_model, train_df, val_df, selection_info).
    The final model is refit on all healthy rows using the selected model kind.
    """
    driver_columns = driver_columns or THERMAL_DRIVER_COLUMNS
    train_df, val_df = healthy_train_validate_split(healthy_df, train_fraction)

    linear_model = _fit_linear(train_df, target_column, driver_columns)
    linear_rmse = _validation_rmse(linear_model, val_df)

    gbm_model = _fit_gbm(train_df, target_column, driver_columns)
    gbm_rmse = _validation_rmse(gbm_model, val_df)

    chosen_kind: ModelKind = "linear"
    if val_df.empty:
        chosen_kind = "linear"
    elif linear_rmse > 0 and gbm_rmse < linear_rmse * (1.0 - gbm_improvement):
        chosen_kind = "gbm"

    if chosen_kind == "gbm":
        final_model = _fit_gbm(healthy_df, target_column, driver_columns)
    else:
        final_model = _fit_linear(healthy_df, target_column, driver_columns)

    selection_info = {
        "chosen_model": chosen_kind,
        "linear_val_rmse": linear_rmse,
        "gbm_val_rmse": gbm_rmse,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
    }
    return final_model, train_df, val_df, selection_info


def validate_thermal_model(
    model: GearboxThermalModel,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Check held-out healthy residuals: small, unbiased, structureless.

    Returns metrics dict with pass/fail flags per criterion.
    """
    if val_df.empty:
        return {
            "n_samples": 0,
            "passed": False,
            "reason": "empty validation set",
        }

    residuals = model.residual(val_df)
    actual = val_df[model.target_column]
    predicted = model.predict(val_df)

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    mean_res = float(residuals.mean())

    driver_corrs: dict[str, float] = {}
    for col in model.driver_columns:
        if val_df[col].std() > 0 and residuals.std() > 0:
            driver_corrs[col] = float(np.corrcoef(val_df[col], residuals)[0, 1])
        else:
            driver_corrs[col] = 0.0

    # Residual trend vs time (°C per day)
    t_days = (val_df.index - val_df.index[0]).total_seconds().values / 86400.0
    if len(t_days) > 1 and np.std(t_days) > 0:
        time_trend = float(np.polyfit(t_days, residuals.values, 1)[0])
    else:
        time_trend = 0.0

    unbiased = abs(mean_res) < THERMAL_VALIDATION_THRESHOLDS["max_abs_mean_residual_c"]
    structureless_drivers = all(
        abs(c) < THERMAL_VALIDATION_THRESHOLDS["max_driver_correlation"]
        for c in driver_corrs.values()
    )
    structureless_time = (
        abs(time_trend) < THERMAL_VALIDATION_THRESHOLDS["max_time_trend_per_day_c"]
    )

    model_details: dict[str, Any] = {"model_kind": model.model_kind}
    if model.model_kind == "linear" and model.feature_names:
        pipeline = model.model
        ridge = pipeline.named_steps["ridge"]
        coefs = ridge.coef_.tolist()
        model_details["coefficients"] = dict(zip(model.feature_names, coefs))
        model_details["intercept"] = float(ridge.intercept_)
    elif model.model_kind == "gbm" and model.feature_names:
        model_details["feature_importances"] = dict(
            zip(model.feature_names, model.model.feature_importances_.tolist())
        )

    passed = unbiased and structureless_drivers and structureless_time

    return {
        "n_samples": len(val_df),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mean_residual": mean_res,
        "driver_correlations": driver_corrs,
        "time_trend_per_day": time_trend,
        "unbiased": unbiased,
        "structureless_drivers": structureless_drivers,
        "structureless_time": structureless_time,
        "passed": passed,
        "model_details": model_details,
    }


__all__ = [
    "GearboxThermalModel",
    "fit_gearbox_thermal",
    "fit_gearbox_thermal_with_selection",
    "healthy_train_validate_split",
    "validate_thermal_model",
]
