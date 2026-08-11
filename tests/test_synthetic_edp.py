"""Tests for synthetic EDP dataset generation."""

from pathlib import Path

import pandas as pd

from wind_turbine_anomaly.config import FEATURE_COLUMNS
from wind_turbine_anomaly.data.load_edp import load_edp_dataset
from wind_turbine_anomaly.data.synthetic_edp import (
    SYNTHETIC_TURBINES,
    generate_synthetic_edp_dataset,
    is_synthetic_dataset,
)


def test_generate_synthetic_edp_dataset(tmp_path: Path):
    paths = generate_synthetic_edp_dataset(tmp_path, random_state=0)
    assert len(paths) == 4
    assert is_synthetic_dataset(tmp_path)

    sig_2016 = pd.read_csv(paths["wind-farm-1-signals-2016.csv"])
    assert len(sig_2016) > 100_000
    assert set(sig_2016["Turbine_ID"].unique()) == set(SYNTHETIC_TURBINES)
    for col in FEATURE_COLUMNS:
        assert col in sig_2016.columns


def test_synthetic_loadable_by_pipeline(tmp_path: Path):
    generate_synthetic_edp_dataset(tmp_path, random_state=1)
    turbines, failures = load_edp_dataset(tmp_path)
    assert len(turbines) == 4
    gearbox_ids = {f.turbine_id for f in failures}
    assert "T01" in gearbox_ids
    assert "T06" in gearbox_ids
    assert all(len(df) > 50_000 for df in turbines.values())


def test_synthetic_gearbox_degradation_before_failure(tmp_path: Path):
    generate_synthetic_edp_dataset(tmp_path, random_state=2)
    turbines, _ = load_edp_dataset(tmp_path)
    t01 = turbines["T01"]
    t07 = turbines["T07"]
    window = slice("2016-06-15", "2016-07-17")
    # T01 gets an injected pre-failure ramp; T07 does not.
    assert (
        t01.loc[window, "Gear_Oil_Temp_Avg"].mean()
        > t07.loc[window, "Gear_Oil_Temp_Avg"].mean() + 0.5
    )
