"""Tests for physics-residual hybrid detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import POWER_COLUMN, THERMAL_TARGET_COLUMNS
from wind_turbine_anomaly.models.gearbox_thermal import fit_gearbox_thermal
from wind_turbine_anomaly.models.physics_hybrid import PhysicsHybridPipeline, fit_physics_hybrid
from wind_turbine_anomaly.models.residual_features import (
    build_residual_feature_frame,
    compute_dual_residuals,
    degradation_signal,
)


def _synthetic_thermal_df(n: int = 800, seed: int = 42) -> pd.DataFrame:
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
            THERMAL_TARGET_COLUMNS[0]: oil,
            THERMAL_TARGET_COLUMNS[1]: bear,
            POWER_COLUMN: power,
            "Rtr_RPM_Avg": rpm,
            "Nac_Temp_Avg": nac,
        },
        index=idx,
    )


def test_degradation_signal_direction():
    residuals = pd.Series([1.0, -3.0, 0.0])
    signal = degradation_signal(residuals)
    assert signal.iloc[0] == -1.0
    assert signal.iloc[1] == 3.0
    assert signal.iloc[2] == 0.0


def test_build_residual_feature_frame_drops_incomplete_windows():
    df = _synthetic_thermal_df(n=300)
    oil_model = fit_gearbox_thermal(df.iloc[:200], THERMAL_TARGET_COLUMNS[0])
    bear_model = fit_gearbox_thermal(df.iloc[:200], THERMAL_TARGET_COLUMNS[1])
    features = build_residual_feature_frame(df, oil_model, bear_model)
    assert not features.empty
    assert "oil_deg" in features.columns
    assert "bear_ewma" in features.columns
    assert "oil_roll_mean_144" in features.columns
    assert len(features) < len(df)


def test_injected_offset_increases_degradation_features():
    df = _synthetic_thermal_df(n=400)
    train = df.iloc[:250]
    test = df.iloc[250:].copy()
    oil_model = fit_gearbox_thermal(train, THERMAL_TARGET_COLUMNS[0])
    bear_model = fit_gearbox_thermal(train, THERMAL_TARGET_COLUMNS[1])

    healthy_features = build_residual_feature_frame(test, oil_model, bear_model)
    shifted = test.copy()
    shifted[THERMAL_TARGET_COLUMNS[0]] += 5.0
    shifted[THERMAL_TARGET_COLUMNS[1]] += 5.0
    degraded_features = build_residual_feature_frame(shifted, oil_model, bear_model)

    assert degraded_features["oil_deg"].mean() > healthy_features["oil_deg"].mean()
    assert degraded_features["bear_deg"].mean() > healthy_features["bear_deg"].mean()


def test_compute_dual_residuals_aligned():
    df = _synthetic_thermal_df(n=100)
    oil_model = fit_gearbox_thermal(df.iloc[:60], THERMAL_TARGET_COLUMNS[0])
    bear_model = fit_gearbox_thermal(df.iloc[:60], THERMAL_TARGET_COLUMNS[1])
    residuals = compute_dual_residuals(df, oil_model, bear_model)
    assert list(residuals.columns) == ["oil_residual", "bear_residual"]
    assert len(residuals) == len(df)


def test_physics_hybrid_scores_higher_on_degraded_data():
    df = _synthetic_thermal_df(n=800)
    train = df.iloc[:400]
    test_healthy = df.iloc[400:600]
    test_shifted = test_healthy.copy()
    test_shifted[THERMAL_TARGET_COLUMNS[0]] += 6.0
    test_shifted[THERMAL_TARGET_COLUMNS[1]] += 6.0

    pipeline = fit_physics_hybrid(train)
    assert isinstance(pipeline, PhysicsHybridPipeline)

    healthy_scores = pipeline.score(test_healthy)
    degraded_scores = pipeline.score(test_shifted)
    assert len(healthy_scores) > 0
    assert len(degraded_scores) > 0
    assert degraded_scores.mean() > healthy_scores.mean()


def test_physics_hybrid_end_to_end_smoke():
    df = _synthetic_thermal_df(n=500)
    pipeline = fit_physics_hybrid(df.iloc[:300])
    scores = pipeline.score(df.iloc[300:])
    assert (scores > 0).any()
