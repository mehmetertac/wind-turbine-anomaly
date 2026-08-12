"""Tests for consolidated metrics table."""

import json

import pandas as pd
import pytest

from wind_turbine_anomaly.eval.metrics_table import (
    build_long_metrics_table,
    build_wide_metrics_table,
    write_metrics_csv,
)


@pytest.fixture
def sample_results_dir(tmp_path):
    for detector in ("isolation_forest", "dense_autoencoder"):
        det_dir = tmp_path / detector
        det_dir.mkdir()
        metrics = {
            "horizon_days": 30,
            "false_alarms_per_turbine_year": 10.5,
            "thresholds": {"T01": 1.0, "T06": 2.0},
            "turbines": [
                {
                    "turbine_id": "T01",
                    "has_gearbox_failure": True,
                    "lead_time_days": 18.3,
                    "successful_warning": False,
                    "false_alarm_episodes": 5,
                    "scored_days": 365.0,
                    "precision_at_horizon": 0.1,
                },
                {
                    "turbine_id": "T06",
                    "has_gearbox_failure": True,
                    "lead_time_days": 47.6,
                    "successful_warning": True,
                    "false_alarm_episodes": 2,
                    "scored_days": 400.0,
                    "precision_at_horizon": 0.5,
                },
            ],
        }
        (det_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return tmp_path


def test_build_wide_metrics_table(sample_results_dir):
    df = build_wide_metrics_table(
        ["isolation_forest", "dense_autoencoder"],
        results_dir=sample_results_dir,
    )
    assert len(df) == 2
    assert "T01_lead_time_days" in df.columns
    assert df.loc[df.detector == "isolation_forest", "T06_lead_time_days"].iloc[0] == 47.6


def test_write_metrics_csv(sample_results_dir):
    wide_path, long_path = write_metrics_csv(
        ["isolation_forest", "dense_autoencoder"],
        results_dir=sample_results_dir,
    )
    wide = pd.read_csv(wide_path)
    long = pd.read_csv(long_path)
    assert len(wide) == 2
    assert len(long) == 4
