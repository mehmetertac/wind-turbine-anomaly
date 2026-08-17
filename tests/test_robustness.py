"""Tests for robustness validation."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import GearboxFailure, POWER_COLUMN
from wind_turbine_anomaly.eval.robustness import (
    audit_leakage_for_turbine,
    compute_healthy_residual_drift,
    compute_seasonal_residual_metrics,
    meteorological_season,
    seasonal_bucket,
)
from wind_turbine_anomaly.models.gearbox_thermal import fit_gearbox_thermal


def _synthetic_thermal_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-01", periods=n, freq="10min", tz="UTC")
    power = rng.uniform(200, 1800, n)
    rpm = 8.0 + 0.004 * power + rng.normal(0, 0.2, n)
    nac = 15.0 + 8.0 * np.sin(2 * np.pi * idx.dayofyear / 365.25) + rng.normal(0, 0.4, n)
    load = power / 2000.0
    oil = 44.0 + 8.0 * load + 0.15 * nac + rng.normal(0, 0.3, n)
    bear = 49.0 + 10.0 * load + 0.12 * nac + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "Gear_Oil_Temp_Avg": oil,
            "Gear_Bear_Temp_Avg": bear,
            POWER_COLUMN: power,
            "Rtr_RPM_Avg": rpm,
            "Amb_WindSpeed_Avg": rng.uniform(3, 12, n),
            "Amb_WindDir_Relative_Avg": rng.uniform(0, 360, n),
            "Nac_Temp_Avg": nac,
        },
        index=idx,
    )


def test_meteorological_season():
    assert meteorological_season(1) == "winter"
    assert meteorological_season(7) == "summer"
    assert meteorological_season(4) == "shoulder"


def test_seasonal_bucket():
    idx = pd.date_range("2016-01-15", periods=3, freq="D", tz="UTC")
    buckets = seasonal_bucket(idx)
    assert buckets.iloc[0] == "winter"


def test_compute_seasonal_residual_metrics_passes_on_synthetic():
    df = _synthetic_thermal_df(n=8000)
    model = fit_gearbox_thermal(df.iloc[:6400], "Gear_Oil_Temp_Avg")
    metrics = compute_seasonal_residual_metrics(model, df.iloc[6400:])
    assert "seasons" in metrics
    assert metrics["rmse_ratio"] >= 1.0
    assert "passed" in metrics


def test_compute_healthy_residual_drift_near_zero():
    idx = pd.date_range("2016-01-01", periods=2000, freq="10min", tz="UTC")
    frame = pd.DataFrame({"residual": np.random.normal(0, 0.1, len(idx))}, index=idx)
    drift = compute_healthy_residual_drift(frame, window_days=7)
    assert drift < 0.5


def test_audit_leakage_train_before_failure_buffer():
    df = _synthetic_thermal_df(n=500)
    failure = GearboxFailure(
        turbine_id="T01",
        timestamp=pd.Timestamp("2016-07-18T02:10:00+00:00"),
        remarks="test",
    )
    audit = audit_leakage_for_turbine("T01", df, failure, buffer_days=90)
    check = audit["checks"]["train_before_failure_buffer"]
    assert check["passed"] is True


def test_audit_leakage_fails_when_train_after_cutoff():
    df = _synthetic_thermal_df(n=500)
    failure = GearboxFailure(
        turbine_id="T01",
        timestamp=df.index[100],
        remarks="test",
    )
    audit = audit_leakage_for_turbine("T01", df, failure, buffer_days=90)
    check = audit["checks"]["train_before_failure_buffer"]
    assert check["passed"] is False


def test_seasonal_terms_in_linear_model():
    df = _synthetic_thermal_df(n=400)
    model = fit_gearbox_thermal(df.iloc[:300], "Gear_Oil_Temp_Avg", seasonal_terms=True)
    assert model.seasonal_terms is True
    assert "month_sin" in (model.feature_names or [])
    preds = model.predict(df.iloc[300:])
    assert len(preds) == 100
