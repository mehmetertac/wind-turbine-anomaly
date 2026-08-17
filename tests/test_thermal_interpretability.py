"""Tests for thermal interpretability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wind_turbine_anomaly.config import POWER_COLUMN
from wind_turbine_anomaly.eval.thermal_interpretability import (
    explain_thermal_model,
    plot_shap_summary,
)
from wind_turbine_anomaly.models.gearbox_thermal import fit_gearbox_thermal


def _synthetic_thermal_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
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


def test_explain_thermal_model_linear():
    df = _synthetic_thermal_df()
    model = fit_gearbox_thermal(df.iloc[:480], "Gear_Oil_Temp_Avg")
    explanation = explain_thermal_model(model, df.iloc[:480])
    assert explanation["model_kind"] == "linear"
    assert len(explanation["drivers"]) >= 4
    assert explanation["drivers"][0]["mean_abs_shap"] >= 0


def test_plot_shap_summary(tmp_path):
    df = _synthetic_thermal_df()
    model = fit_gearbox_thermal(df.iloc[:400], "Gear_Oil_Temp_Avg")
    explanation = explain_thermal_model(model, df.iloc[:400])
    out = plot_shap_summary(explanation, "test", tmp_path / "shap.png")
    assert out.exists()


def test_power_is_top_driver_on_synthetic():
    df = _synthetic_thermal_df(n=800)
    model = fit_gearbox_thermal(df.iloc[:640], "Gear_Oil_Temp_Avg")
    explanation = explain_thermal_model(model, df.iloc[:640])
    top = explanation["drivers"][0]["feature"]
    assert top in (POWER_COLUMN, f"{POWER_COLUMN}_sq", "Nac_Temp_Avg")
