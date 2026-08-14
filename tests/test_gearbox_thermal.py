"""Tests for gearbox thermal normal-behavior model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import POWER_COLUMN, THERMAL_DRIVER_COLUMNS
from wind_turbine_anomaly.models.gearbox_thermal import (
    fit_gearbox_thermal,
    fit_gearbox_thermal_with_selection,
    healthy_train_validate_split,
    validate_thermal_model,
)


def _synthetic_thermal_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build SCADA-like rows with known thermal physics."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    power = rng.uniform(200, 1800, n)
    rpm = 8.0 + 0.004 * power + rng.normal(0, 0.2, n)
    nac = 15.0 + rng.normal(0, 2.0, n)
    load = power / 2000.0
    oil = 44.0 + 8.0 * load + 0.15 * nac + rng.normal(0, 0.3, n)
    bear = 49.0 + 10.0 * load + 0.12 * nac + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "Gear_Oil_Temp_Avg": oil,
            "Gear_Bear_Temp_Avg": bear,
            POWER_COLUMN: power,
            "Rtr_RPM_Avg": rpm,
            "Nac_Temp_Avg": nac,
        },
        index=idx,
    )


def test_fit_and_residual_near_zero_on_train():
    df = _synthetic_thermal_df()
    model = fit_gearbox_thermal(df.iloc[:400], "Gear_Oil_Temp_Avg")
    residuals = model.residual(df.iloc[:400])
    assert len(residuals) == 400
    assert abs(residuals.mean()) < 1.0
    assert residuals.std() < 2.0


def test_injected_offset_shifts_residual_mean():
    df = _synthetic_thermal_df()
    train_df, val_df = healthy_train_validate_split(df)
    model = fit_gearbox_thermal(train_df, "Gear_Oil_Temp_Avg")

    val_shifted = val_df.copy()
    val_shifted["Gear_Oil_Temp_Avg"] = val_shifted["Gear_Oil_Temp_Avg"] + 5.0
    residuals = model.residual(val_shifted)
    assert residuals.mean() < -4.0


def test_time_ordered_split_no_future_in_train():
    df = _synthetic_thermal_df(n=100)
    train_df, val_df = healthy_train_validate_split(df, train_fraction=0.8)
    assert train_df.index.max() <= val_df.index.min()
    assert len(train_df) == 80
    assert len(val_df) == 20


def test_validate_flags_structured_residuals():
    df = _synthetic_thermal_df()
    train_df, val_df = healthy_train_validate_split(df)
    model = fit_gearbox_thermal(train_df, "Gear_Oil_Temp_Avg")

    good = validate_thermal_model(model, val_df)
    assert good["n_samples"] == len(val_df)
    assert good["rmse"] < 2.0

    # Inject systematic residual slope vs power
    val_bad = val_df.copy()
    val_bad["Gear_Oil_Temp_Avg"] = (
        val_bad["Gear_Oil_Temp_Avg"] - 0.01 * val_bad[POWER_COLUMN]
    )
    bad = validate_thermal_model(model, val_bad)
    assert bad["driver_correlations"][POWER_COLUMN] != 0.0


def test_model_selection_returns_info():
    df = _synthetic_thermal_df(n=600)
    model, train_df, val_df, info = fit_gearbox_thermal_with_selection(
        df, "Gear_Bear_Temp_Avg", driver_columns=THERMAL_DRIVER_COLUMNS
    )
    assert info["chosen_model"] in ("linear", "gbm")
    assert info["train_rows"] > 0
    assert info["val_rows"] > 0
    preds = model.predict(df)
    assert len(preds) == len(df)


def test_residual_frame_columns():
    df = _synthetic_thermal_df(n=50)
    model = fit_gearbox_thermal(df.iloc[:40], "Gear_Oil_Temp_Avg")
    frame = model.residual_frame(df.iloc[40:])
    assert list(frame.columns) == ["actual", "predicted", "residual"]
    assert (frame["residual"] == frame["predicted"] - frame["actual"]).all()
